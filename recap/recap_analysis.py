"""Offline RECAP analysis helpers for paper figures and diagnostics."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

from recap.aggregate import coverage_error, coverage_hallucination_fpr, ensure_probe_score
from recap.io import write_jsonl
from recap.metrics import mean, roc_auc
from recap.recap_ablation import (
    DEFAULT_RISKS,
    RISK_KEYS,
    add_recap_ablation_fields,
    group_probe_scores,
    risk_table,
)
from recap.relations import relation_family
from recap.scoring import prepare_sample_scores

LOW_COST_VARIANTS = (
    "confidence",
    "recap_claim_delta",
    "recap_anti_delta",
    "recap_pair_delta",
    "recap_img_pair",
    "recap_evidence",
    "rice_recap_selector",
)


def build_recap_analysis_samples(
    records: list[dict[str, Any]],
    *,
    synthesize_claim_reuse: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    """Prepare sample-level scores with RECAP ablation fields attached."""
    probe_scores = [ensure_probe_score(record) for record in records]
    grouped = group_probe_scores(probe_scores)
    if synthesize_claim_reuse:
        grouped = synthesize_reused_claim_candidates(grouped)
        probe_scores = [record for sample_probes in grouped.values() for record in sample_probes.values()]
    samples = prepare_sample_scores(probe_scores)
    add_recap_ablation_fields(samples, grouped)
    return samples, grouped


def recap_coverage_curves(
    records: list[dict[str, Any]],
    *,
    risks: list[str] | None = None,
    min_coverage: int = 10,
    max_coverage: int = 100,
    step: int = 5,
) -> dict[str, Any]:
    """Return risk-coverage curves for selected risk scores."""
    samples, _ = build_recap_analysis_samples(records)
    risk_names = _valid_risks(risks)
    coverages = _coverage_grid(min_coverage=min_coverage, max_coverage=max_coverage, step=step)
    labels = [float(sample["error"]) for sample in samples]
    hallucination_labels = [float(sample["hallucination"]) for sample in samples]

    curves: dict[str, list[dict[str, float]]] = {}
    for risk_name in risk_names:
        risk_key = RISK_KEYS[risk_name]
        scores = [float(sample.get(risk_key, 0.0)) for sample in samples]
        rows = []
        for coverage in coverages:
            accepted_count = max(1, int(math.ceil(len(samples) * coverage)))
            error_rate = coverage_error(samples, coverage, risk_key=risk_key)
            rows.append(
                {
                    "coverage": coverage,
                    "coverage_pct": coverage * 100.0,
                    "accepted_count": float(accepted_count),
                    "selective_accuracy": 1.0 - error_rate,
                    "error_rate": error_rate,
                    "hallucination_fpr": coverage_hallucination_fpr(samples, coverage, risk_key=risk_key),
                }
            )
        curves[risk_name] = rows

    return {
        "overview": {
            "num_samples": float(len(samples)),
            "direct_accuracy": mean([1.0 - float(sample["error"]) for sample in samples]),
            "direct_hallucination_fpr": _hallucination_fpr(samples),
        },
        "risks": risk_names,
        "summary": {
            risk_name: {
                "error_auroc": roc_auc(labels, [float(sample.get(RISK_KEYS[risk_name], 0.0)) for sample in samples]),
                "hallucination_auroc": roc_auc(
                    hallucination_labels,
                    [float(sample.get(RISK_KEYS[risk_name], 0.0)) for sample in samples],
                ),
            }
            for risk_name in risk_names
        },
        "curves": curves,
    }


def coverage_curves_to_csv_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for risk_name, curve in report.get("curves", {}).items():
        for point in curve:
            row = {"risk": risk_name}
            row.update(point)
            rows.append(row)
    return rows


def recap_cost_utility(
    records: list[dict[str, Any]],
    *,
    risks: list[str] | None = None,
    coverage: int = 80,
) -> dict[str, Any]:
    """Return a compute-utility table for low-cost RECAP variants."""
    samples, grouped = build_recap_analysis_samples(records)
    risk_names = _valid_risks(risks or list(LOW_COST_VARIANTS))
    coverage_fraction = coverage / 100.0
    rows = risk_table(samples, risk_names, coverage_fraction)
    cost_rows: dict[str, dict[str, Any]] = {}
    suffix = int(round(coverage_fraction * 100))

    for risk_name in risk_names:
        required_counts = [len(required_probe_names(grouped.get(str(sample["sample_id"]), {}), risk_name)) for sample in samples]
        calls = [count * 2.0 for count in required_counts]
        metric_row = rows[risk_name]
        cost_rows[risk_name] = {
            "avg_probe_count": mean([float(count) for count in required_counts]),
            "avg_inference_calls": mean(calls),
            "selective_accuracy": metric_row[f"selective_accuracy_{suffix}cov"],
            "hallucination_fpr": metric_row[f"hallucination_fpr_{suffix}cov"],
            "error_auroc": metric_row["error_auroc"],
            "hallucination_auroc": metric_row["hallucination_auroc"],
        }

    current_probe_counts = [float(sample.get("probe_count", len(grouped.get(str(sample["sample_id"]), {})))) for sample in samples]
    return {
        "overview": {
            "num_samples": float(len(samples)),
            "coverage": coverage,
            "current_avg_probe_count": mean(current_probe_counts),
            "current_avg_inference_calls": mean([count * 2.0 for count in current_probe_counts]),
        },
        "risks": cost_rows,
    }


def recap_bootstrap_ci(
    records: list[dict[str, Any]],
    *,
    risks: list[str] | None = None,
    coverage: int = 80,
    n_bootstrap: int = 1000,
    seed: int = 13,
    ci: float = 0.95,
) -> dict[str, Any]:
    """Estimate sample-level bootstrap confidence intervals for RECAP metrics.

    This resamples already-scored examples, so it does not require model
    inference. Intervals reflect dataset sampling uncertainty, not stochastic
    decoding uncertainty.
    """
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    if not (0.0 < ci < 1.0):
        raise ValueError("ci must be in (0, 1)")

    samples, _ = build_recap_analysis_samples(records)
    risk_names = _valid_risks(risks)
    coverage_fraction = coverage / 100.0
    actual = risk_table(samples, risk_names, coverage_fraction)
    metric_keys = _risk_metric_keys(coverage_fraction)
    draws: dict[str, dict[str, list[float]]] = {
        risk_name: {metric_key: [] for metric_key in metric_keys} for risk_name in risk_names
    }
    rng = random.Random(seed)

    for _ in range(n_bootstrap):
        resampled = [samples[rng.randrange(len(samples))] for _ in range(len(samples))] if samples else []
        boot_rows = risk_table(resampled, risk_names, coverage_fraction)
        for risk_name in risk_names:
            for metric_key in metric_keys:
                draws[risk_name][metric_key].append(float(boot_rows[risk_name][metric_key]))

    alpha = (1.0 - ci) / 2.0
    risks_out: dict[str, dict[str, Any]] = {}
    for risk_name in risk_names:
        risks_out[risk_name] = {}
        for metric_key in metric_keys:
            values = sorted(draws[risk_name][metric_key])
            risks_out[risk_name][metric_key] = {
                "estimate": float(actual[risk_name][metric_key]),
                "mean": mean(values),
                "ci_low": percentile(values, alpha),
                "ci_high": percentile(values, 1.0 - alpha),
            }

    return {
        "overview": {
            "num_samples": float(len(samples)),
            "coverage": coverage,
            "n_bootstrap": float(n_bootstrap),
            "seed": float(seed),
            "ci": ci,
        },
        "risks": risks_out,
    }


def bootstrap_ci_to_csv_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    overview = report.get("overview", {})
    for risk_name, metrics in report.get("risks", {}).items():
        for metric_name, values in metrics.items():
            rows.append(
                {
                    "risk": risk_name,
                    "metric": metric_name,
                    "estimate": values.get("estimate", 0.0),
                    "mean": values.get("mean", 0.0),
                    "ci_low": values.get("ci_low", 0.0),
                    "ci_high": values.get("ci_high", 0.0),
                    "n_bootstrap": overview.get("n_bootstrap", 0.0),
                    "seed": overview.get("seed", 0.0),
                    "ci": overview.get("ci", 0.0),
                }
            )
    return rows


def recap_low_cost_equivalence(
    records: list[dict[str, Any]],
    *,
    variants: list[str] | None = None,
    coverage: int = 80,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Check that low-cost probe subsets reproduce full-cache metrics.

    The check filters a full RECAP score cache to each variant's required
    probes, synthesizes claim candidate records from ``orig`` and
    ``text_only`` when those reusable probes replace duplicate claim probes,
    and recomputes metrics from the reduced cache.
    """
    variant_names = _valid_risks(variants or list(LOW_COST_VARIANTS))
    coverage_fraction = coverage / 100.0
    full_samples, grouped = build_recap_analysis_samples(records)
    full_rows = risk_table(full_samples, variant_names, coverage_fraction)
    full_by_id = {str(sample["sample_id"]): sample for sample in full_samples}
    metric_keys = _risk_metric_keys(coverage_fraction)

    rows: dict[str, dict[str, Any]] = {}
    for variant in variant_names:
        filtered_records = filter_recap_probes(records, variant=variant)
        low_samples, low_grouped = build_recap_analysis_samples(filtered_records, synthesize_claim_reuse=True)
        low_rows = risk_table(low_samples, [variant], coverage_fraction)
        low_by_id = {str(sample["sample_id"]): sample for sample in low_samples}
        metric_diffs = {
            metric_key: abs(float(full_rows[variant][metric_key]) - float(low_rows[variant][metric_key]))
            for metric_key in metric_keys
        }

        risk_key = RISK_KEYS[variant]
        sample_diffs = [
            abs(float(full_sample.get(risk_key, 0.0)) - float(low_by_id.get(sample_id, {}).get(risk_key, 0.0)))
            for sample_id, full_sample in full_by_id.items()
            if sample_id in low_by_id
        ]
        required_counts = [len(required_probe_names(grouped.get(str(sample["sample_id"]), {}), variant)) for sample in full_samples]
        rows[variant] = {
            "avg_probe_count": mean([float(count) for count in required_counts]),
            "avg_inference_calls": mean([float(count) * 2.0 for count in required_counts]),
            "full_sample_count": float(len(full_samples)),
            "low_cost_sample_count": float(len(low_samples)),
            "sample_count_match": len(full_samples) == len(low_samples),
            "full_metrics": full_rows[variant],
            "low_cost_metrics": low_rows[variant],
            "metric_abs_diffs": metric_diffs,
            "max_metric_abs_diff": max(metric_diffs.values()) if metric_diffs else 0.0,
            "mean_sample_risk_abs_diff": mean(sample_diffs),
            "max_sample_risk_abs_diff": max(sample_diffs) if sample_diffs else 0.0,
            "equivalent": (
                len(full_samples) == len(low_samples)
                and (max(metric_diffs.values()) if metric_diffs else 0.0) <= tolerance
                and (max(sample_diffs) if sample_diffs else 0.0) <= tolerance
            ),
        }

    return {
        "overview": {
            "num_samples": float(len(full_samples)),
            "coverage": coverage,
            "tolerance": tolerance,
            "note": "Low-cost metrics are recomputed from filtered probes; claim candidates are synthesized from orig/text_only when reused.",
        },
        "variants": rows,
    }


def low_cost_equivalence_to_csv_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant, row in report.get("variants", {}).items():
        out = {
            "variant": variant,
            "avg_probe_count": row.get("avg_probe_count", 0.0),
            "avg_inference_calls": row.get("avg_inference_calls", 0.0),
            "full_sample_count": row.get("full_sample_count", 0.0),
            "low_cost_sample_count": row.get("low_cost_sample_count", 0.0),
            "sample_count_match": row.get("sample_count_match", False),
            "max_metric_abs_diff": row.get("max_metric_abs_diff", 0.0),
            "mean_sample_risk_abs_diff": row.get("mean_sample_risk_abs_diff", 0.0),
            "max_sample_risk_abs_diff": row.get("max_sample_risk_abs_diff", 0.0),
            "equivalent": row.get("equivalent", False),
        }
        for metric_name, diff in row.get("metric_abs_diffs", {}).items():
            out[f"diff_{metric_name}"] = diff
        rows.append(out)
    return rows


def extract_recap_case_studies(
    records: list[dict[str, Any]],
    *,
    baseline: str = "confidence",
    method: str = "recap_evidence",
    coverage: int = 80,
    examples: int = 5,
) -> dict[str, Any]:
    """Select representative examples for qualitative analysis."""
    samples, grouped = build_recap_analysis_samples(records)
    baseline_key = RISK_KEYS[baseline]
    method_key = RISK_KEYS[method]
    coverage_fraction = coverage / 100.0
    baseline_accept = accepted_ids(samples, baseline_key, coverage_fraction)
    method_accept = accepted_ids(samples, method_key, coverage_fraction)

    enriched = [enrich_case_sample(sample, grouped.get(str(sample["sample_id"]), {}), baseline_key, method_key) for sample in samples]

    categories = {
        "confidence_accepts_wrong_method_rejects": _top_cases(
            enriched,
            lambda row: row["direct_error"] and row["sample_id"] in baseline_accept and row["sample_id"] not in method_accept,
            sort_key=lambda row: row["method_risk"] - row["baseline_risk"],
            examples=examples,
        ),
        "confidence_rejects_correct_method_accepts": _top_cases(
            enriched,
            lambda row: (not row["direct_error"]) and row["sample_id"] not in baseline_accept and row["sample_id"] in method_accept,
            sort_key=lambda row: row["baseline_risk"] - row["method_risk"],
            examples=examples,
        ),
        "hallucination_rejected_by_method": _top_cases(
            enriched,
            lambda row: row["direct_hallucination"] and row["sample_id"] not in method_accept,
            sort_key=lambda row: row["method_risk"],
            examples=examples,
        ),
        "high_confidence_hallucinations": _top_cases(
            enriched,
            lambda row: row["direct_hallucination"],
            sort_key=lambda row: -row["baseline_risk"],
            examples=examples,
        ),
    }
    return {
        "overview": {
            "num_samples": float(len(samples)),
            "coverage": coverage,
            "baseline": baseline,
            "method": method,
            "baseline_accepted": float(len(baseline_accept)),
            "method_accepted": float(len(method_accept)),
        },
        "categories": categories,
    }


def flatten_case_studies(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for category, examples in report.get("categories", {}).items():
        for rank, example in enumerate(examples, start=1):
            row = {"category": category, "rank": rank}
            row.update(example)
            rows.append(row)
    return rows


def filter_recap_probes(probes: list[dict[str, Any]], *, variant: str) -> list[dict[str, Any]]:
    """Filter an already-built probe JSONL down to the probes required by a variant."""
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    order: list[str] = []
    for probe in probes:
        sample_id = str(probe["sample_id"])
        if sample_id not in grouped:
            grouped[sample_id] = {}
            order.append(sample_id)
        grouped[sample_id][str(probe["probe"])] = probe

    filtered: list[dict[str, Any]] = []
    for sample_id in order:
        sample_probes = grouped[sample_id]
        keep = required_probe_names(sample_probes, variant)
        kept = [dict(probe) for name, probe in sample_probes.items() if name in keep]
        for probe in kept:
            probe["probe_count"] = len(kept)
        filtered.extend(kept)
    return filtered


def synthesize_reused_claim_candidates(
    grouped: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Add synthetic claim candidate records backed by orig/text_only probes.

    Low-cost variants can reuse the direct image query as the claim image
    probe, and the direct text-only query as the claim text probe. This helper
    makes that reuse explicit for offline recomputation without mutating the
    input grouping.
    """
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for sample_id, probes in grouped.items():
        copied = {name: dict(probe) for name, probe in probes.items()}
        orig = copied.get("orig")
        if orig and not _has_recap_role_kind(copied, role="claim", kind="img"):
            copied[_synthetic_claim_probe_name(orig, kind="img")] = _synthetic_claim_probe(orig, kind="img")
        text_only = copied.get("text_only")
        if text_only and not _has_recap_role_kind(copied, role="claim", kind="text"):
            copied[_synthetic_claim_probe_name(text_only, kind="text")] = _synthetic_claim_probe(text_only, kind="text")
        out[sample_id] = copied
    return out


def required_probe_names(probes: dict[str, dict[str, Any]], variant: str) -> set[str]:
    """Return probe names needed to evaluate a low-cost variant.

    The counts use safe claim reuse: ``orig`` acts as the image claim probe and
    ``text_only`` acts as the text-only claim probe only when their margins
    match the corresponding claim candidate. Canonicalized relation probes are
    kept explicitly when reuse would change the score.
    """
    variant = variant.strip()
    keep = {"orig"} if "orig" in probes else set()
    if variant in {"confidence"}:
        return keep
    if variant in {"recap_claim_delta", "selector_claim_delta"}:
        _add_claim_or_reuse(keep, probes, kind="img")
        _add_claim_or_reuse(keep, probes, kind="text")
        return keep
    if variant in {"recap_anti_delta", "selector_anti_delta"}:
        _add_recap_role(keep, probes, role="anti", kinds={"img", "text"})
        return keep
    if variant in {"recap_img_pair", "selector_img_pair"}:
        _add_claim_or_reuse(keep, probes, kind="img")
        _add_recap_role(keep, probes, role="anti", kinds={"img"})
        return keep
    if variant in {"recap_text_pair"}:
        _add_claim_or_reuse(keep, probes, kind="text")
        _add_recap_role(keep, probes, role="anti", kinds={"text"})
        return keep
    if variant in {"recap_pair_delta", "selector_pair_delta"}:
        _add_claim_or_reuse(keep, probes, kind="img")
        _add_claim_or_reuse(keep, probes, kind="text")
        _add_recap_role(keep, probes, role="anti", kinds={"img", "text"})
        return keep
    if variant in {"recap_evidence", "rice_recap_wo_conf", "rice_recap_selector"}:
        _add_claim_or_reuse(keep, probes, kind="img")
        _add_claim_or_reuse(keep, probes, kind="text")
        _add_recap_role(keep, probes, role="anti", kinds={"img", "text"})
        _add_recap_role(keep, probes, role="support", kinds={"img", "text"})
        return keep
    return set(probes)


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        import csv

        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_case_jsonl(path: str | Path, report: dict[str, Any]) -> None:
    write_jsonl(path, flatten_case_studies(report))


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if q <= 0.0:
        return float(sorted_values[0])
    if q >= 1.0:
        return float(sorted_values[-1])
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    weight = pos - lo
    return float(sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight)


def accepted_ids(samples: list[dict[str, Any]], risk_key: str, coverage: float) -> set[str]:
    accepted_count = max(1, int(math.ceil(len(samples) * coverage)))
    accepted = sorted(samples, key=lambda sample: float(sample.get(risk_key, 0.0)))[:accepted_count]
    return {str(sample["sample_id"]) for sample in accepted}


def enrich_case_sample(
    sample: dict[str, Any],
    probes: dict[str, dict[str, Any]],
    baseline_key: str,
    method_key: str,
) -> dict[str, Any]:
    orig = probes.get("orig", {})
    out = {
        "sample_id": str(sample["sample_id"]),
        "dataset": sample.get("dataset", ""),
        "image": orig.get("image", ""),
        "image_id": orig.get("image_id", ""),
        "question": orig.get("question", ""),
        "source_caption": orig.get("source_caption", ""),
        "base_relation": sample.get("base_relation", ""),
        "relation_family": sample.get("relation_family", ""),
        "target": sample.get("direct_target", ""),
        "prediction": sample.get("direct_pred", ""),
        "direct_error": bool(sample.get("error", 0.0)),
        "direct_hallucination": bool(sample.get("hallucination", 0.0)),
        "baseline_risk": float(sample.get(baseline_key, 0.0)),
        "method_risk": float(sample.get(method_key, 0.0)),
        "confidence_risk": float(sample.get("confidence_risk", 0.0)),
        "recap_evidence_risk": float(sample.get("recap_evidence_risk", 0.0)),
        "recap_claim_delta": float(sample.get("recap_claim_delta", 0.0)),
        "recap_best_anti_relation": sample.get("recap_best_anti_relation", ""),
        "recap_best_anti_delta": float(sample.get("recap_best_anti_delta", 0.0)),
        "recap_pair_margin": float(sample.get("recap_pair_margin", 0.0)),
        "recap_support_penalty": float(sample.get("recap_support_penalty", 0.0)),
        "support_orig": float(sample.get("support_orig", 0.0)),
    }
    return out


def _top_cases(
    rows: list[dict[str, Any]],
    predicate: Any,
    *,
    sort_key: Any,
    examples: int,
) -> list[dict[str, Any]]:
    selected = [row for row in rows if predicate(row)]
    selected.sort(key=sort_key, reverse=True)
    return selected[:examples]


def _coverage_grid(*, min_coverage: int, max_coverage: int, step: int) -> list[float]:
    if step <= 0:
        raise ValueError("--step must be positive")
    if min_coverage <= 0 or max_coverage > 100 or min_coverage > max_coverage:
        raise ValueError("coverage bounds must satisfy 0 < min <= max <= 100")
    return [value / 100.0 for value in range(min_coverage, max_coverage + 1, step)]


def _valid_risks(risks: list[str] | None) -> list[str]:
    risk_names = [name for name in (risks or list(DEFAULT_RISKS)) if name in RISK_KEYS]
    if not risk_names:
        raise ValueError("No valid risk names selected")
    return risk_names


def _risk_metric_keys(coverage: float) -> tuple[str, str, str, str]:
    suffix = int(round(coverage * 100))
    return (
        f"selective_accuracy_{suffix}cov",
        f"hallucination_fpr_{suffix}cov",
        "error_auroc",
        "hallucination_auroc",
    )


def _hallucination_fpr(samples: list[dict[str, Any]]) -> float:
    negatives = [sample for sample in samples if sample.get("target_is_negative")]
    if not negatives:
        return 0.0
    return mean([float(sample.get("hallucination", 0.0)) for sample in negatives])


def _add_if_present(keep: set[str], probes: dict[str, dict[str, Any]], name: str) -> None:
    if name in probes:
        keep.add(name)


def _add_recap_role(
    keep: set[str],
    probes: dict[str, dict[str, Any]],
    *,
    role: str,
    kinds: set[str],
) -> None:
    for name, probe in probes.items():
        if not str(name).startswith("recap_"):
            continue
        if str(probe.get("recap_candidate_role", "")) != role:
            continue
        if str(probe.get("recap_candidate_kind", "")) not in kinds:
            continue
        keep.add(name)


def _add_claim_or_reuse(keep: set[str], probes: dict[str, dict[str, Any]], *, kind: str) -> None:
    reuse_probe = "orig" if kind == "img" else "text_only"
    claim_names = _recap_role_kind_names(probes, role="claim", kind=kind)
    if not claim_names:
        _add_if_present(keep, probes, reuse_probe)
        return
    if _claim_kind_reusable(probes, kind=kind):
        _add_if_present(keep, probes, reuse_probe)
        return
    keep.update(claim_names)


def _recap_role_kind_names(probes: dict[str, dict[str, Any]], *, role: str, kind: str) -> set[str]:
    return {
        str(name)
        for name, probe in probes.items()
        if str(name).startswith("recap_")
        and str(probe.get("recap_candidate_role", "")) == role
        and str(probe.get("recap_candidate_kind", "")) == kind
    }


def _claim_kind_reusable(probes: dict[str, dict[str, Any]], *, kind: str, eps: float = 1e-8) -> bool:
    reuse_probe = probes.get("orig" if kind == "img" else "text_only")
    if reuse_probe is None:
        return False
    claim_names = _recap_role_kind_names(probes, role="claim", kind=kind)
    if not claim_names:
        return True
    reuse_margin = float(reuse_probe.get("margin", 0.0))
    return all(abs(float(probes[name].get("margin", 0.0)) - reuse_margin) <= eps for name in claim_names)


def _has_recap_role_kind(probes: dict[str, dict[str, Any]], *, role: str, kind: str) -> bool:
    return bool(_recap_role_kind_names(probes, role=role, kind=kind))


def _synthetic_claim_probe_name(probe: dict[str, Any], *, kind: str) -> str:
    relation = str(probe.get("base_relation", probe.get("relation", "claim")) or "claim")
    safe_relation = relation.replace(" ", "_")
    return f"recap_{kind}__claim__{safe_relation}__reused"


def _synthetic_claim_probe(probe: dict[str, Any], *, kind: str) -> dict[str, Any]:
    relation = str(probe.get("base_relation", probe.get("relation", "")) or "")
    out = dict(probe)
    out["probe"] = _synthetic_claim_probe_name(probe, kind=kind)
    out["recap_candidate_relation"] = relation
    out["recap_candidate_kind"] = kind
    out["recap_candidate_role"] = "claim"
    out["recap_candidate_family"] = relation_family(relation)
    out["recap_canonical_relation"] = relation
    out["recap_support_beta"] = 0.0
    return out
