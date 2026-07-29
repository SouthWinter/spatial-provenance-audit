"""Offline RECAP component ablations from already-scored probes."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from recap.aggregate import (
    coverage_error,
    coverage_hallucination_fpr,
    detector_failure_rate,
    ensure_probe_score,
    hallucination_fpr,
)
from recap.metrics import average_precision, mean, rank_norm, roc_auc
from recap.scoring import prepare_sample_scores, recap_candidate_records, support_for_answer

DEFAULT_RISKS = (
    "confidence",
    "recap_claim_delta",
    "recap_anti_delta",
    "recap_pair_delta",
    "recap_evidence",
    "selector_claim_delta",
    "selector_pair_delta",
    "rice_recap_selector",
    "recap_img_pair",
    "recap_text_pair",
    "recap_support_penalty",
)

RISK_KEYS = {
    "confidence": "confidence_risk",
    "recap_claim_delta": "recap_claim_delta_ablation_risk",
    "recap_anti_delta": "recap_anti_delta_ablation_risk",
    "recap_pair_delta": "recap_pair_delta_ablation_risk",
    "recap_evidence": "recap_evidence_risk",
    "rice_recap_wo_conf": "rice_recap_wo_conf",
    "rice_recap_selector": "rice_recap_selector",
    "selector_claim_delta": "selector_claim_delta",
    "selector_anti_delta": "selector_anti_delta",
    "selector_pair_delta": "selector_pair_delta",
    "selector_img_pair": "selector_img_pair",
    "recap_img_pair": "recap_img_pair_ablation_risk",
    "recap_text_pair": "recap_text_pair_ablation_risk",
    "recap_claim_img": "recap_claim_img_ablation_risk",
    "recap_claim_text_prior": "recap_claim_text_prior_ablation_risk",
    "recap_support_penalty": "recap_support_penalty",
}

CORE_METRICS = (
    "selective_accuracy_80cov",
    "hallucination_fpr_80cov",
    "error_auroc",
    "hallucination_auroc",
)


def ablate_recap_probe_scores(
    records: list[dict[str, Any]],
    *,
    coverage: int = 80,
    risks: list[str] | None = None,
    include_by_family: bool = True,
    include_by_relation: bool = False,
    include_auprc: bool = False,
) -> dict[str, Any]:
    """Build an offline ablation report from probe-level scores."""
    probe_scores = [ensure_probe_score(record) for record in records]
    grouped = group_probe_scores(probe_scores)
    samples = prepare_sample_scores(probe_scores)
    add_recap_ablation_fields(samples, grouped)

    risk_names = [name for name in (risks or list(DEFAULT_RISKS)) if name in RISK_KEYS]
    coverage_fraction = coverage / 100.0

    report: dict[str, Any] = {
        "overview": overview(samples),
        "coverage": coverage,
        "risks": risk_table(samples, risk_names, coverage_fraction, include_auprc=include_auprc),
        "cost": cost_report(samples, grouped),
        "best": best_rows(samples, risk_names, coverage_fraction),
    }
    if include_by_family:
        report["by_relation_family"] = grouped_risk_table(
            samples,
            key="relation_family",
            risk_names=risk_names,
            coverage=coverage_fraction,
            include_auprc=include_auprc,
        )
    if include_by_relation:
        report["by_relation"] = grouped_risk_table(
            samples,
            key="base_relation",
            risk_names=risk_names,
            coverage=coverage_fraction,
            include_auprc=include_auprc,
        )
    return report


def group_probe_scores(records: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        grouped[str(record["sample_id"])][str(record["probe"])] = record
    return grouped


def add_recap_ablation_fields(
    samples: list[dict[str, Any]],
    grouped: dict[str, dict[str, dict[str, Any]]],
) -> None:
    for sample in samples:
        probes = grouped.get(str(sample["sample_id"]), {})
        predicted_answer = str(sample.get("direct_pred", "yes"))
        fields = recap_ablation_fields(probes, predicted_answer=predicted_answer)
        sample.update(fields)

    add_selector_fields(
        samples,
        {
            "selector_claim_delta": "recap_claim_delta_ablation_risk",
            "selector_anti_delta": "recap_anti_delta_ablation_risk",
            "selector_pair_delta": "recap_pair_delta_ablation_risk",
            "selector_img_pair": "recap_img_pair_ablation_risk",
        },
    )


def recap_ablation_fields(probes: dict[str, dict[str, Any]], *, predicted_answer: str) -> dict[str, float]:
    candidates = recap_candidate_records(probes)
    claim_img = 0.0
    claim_text = 0.0
    claim_delta = 0.0
    anti_img_values: list[float] = []
    anti_text_values: list[float] = []
    anti_delta_values: list[float] = []
    support_penalty = 0.0
    support_count = 0

    for (_, role), record in candidates.items():
        img_margin = record.get("img")
        text_margin = float(record.get("text", 0.0) or 0.0)
        if img_margin is None:
            continue
        img_margin = float(img_margin)
        delta = img_margin - text_margin
        if role == "claim":
            claim_img = img_margin
            claim_text = text_margin
            claim_delta = delta
        elif role == "anti":
            anti_img_values.append(img_margin)
            anti_text_values.append(text_margin)
            anti_delta_values.append(delta)
        elif role == "support":
            support_count += 1
            support_beta = float(record.get("support_beta", 0.0) or 0.0)
            support_penalty += support_beta * max(0.0, -delta)

    best_anti_img = max(anti_img_values) if anti_img_values else 0.0
    best_anti_text = max(anti_text_values) if anti_text_values else 0.0
    best_anti_delta = max(anti_delta_values) if anti_delta_values else 0.0
    has_contrast = bool(anti_delta_values or support_count)

    pair_delta = claim_delta - best_anti_delta
    evidence_margin = pair_delta - support_penalty if has_contrast else 0.0
    img_pair = claim_img - best_anti_img
    text_pair = claim_text - best_anti_text

    return {
        "recap_claim_delta_ablation_risk": -support_for_answer(claim_delta, predicted_answer) if has_contrast else 0.0,
        "recap_anti_delta_ablation_risk": support_for_answer(best_anti_delta, predicted_answer) if has_contrast else 0.0,
        "recap_pair_delta_ablation_risk": -support_for_answer(pair_delta, predicted_answer) if has_contrast else 0.0,
        "recap_img_pair_ablation_risk": -support_for_answer(img_pair, predicted_answer) if has_contrast else 0.0,
        "recap_text_pair_ablation_risk": -support_for_answer(text_pair, predicted_answer) if has_contrast else 0.0,
        "recap_claim_img_ablation_risk": -support_for_answer(claim_img, predicted_answer) if has_contrast else 0.0,
        "recap_claim_text_prior_ablation_risk": support_for_answer(claim_text, predicted_answer) if has_contrast else 0.0,
        "recap_pair_delta_ablation_margin": pair_delta,
        "recap_evidence_ablation_margin": evidence_margin,
        "recap_support_count": float(support_count),
    }


def add_selector_fields(samples: list[dict[str, Any]], component_keys: dict[str, str]) -> None:
    conf_ranks = active_rank_norm([float(sample.get("confidence_risk", 0.0)) for sample in samples])
    for selector_name, component_key in component_keys.items():
        component_ranks = active_rank_norm([float(sample.get(component_key, 0.0)) for sample in samples])
        for i, sample in enumerate(samples):
            sample[selector_name] = max(conf_ranks[i], component_ranks[i]) if samples else 0.0


def active_rank_norm(values: list[float]) -> list[float]:
    if not values:
        return []
    if max(values) - min(values) <= 1e-8:
        return [0.0 for _ in values]
    return rank_norm(values)


def overview(samples: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "num_samples": float(len(samples)),
        "direct_accuracy": mean([1.0 - float(sample["error"]) for sample in samples]),
        "direct_hallucination_fpr": hallucination_fpr(samples),
        "avg_inference_calls": mean([float(sample.get("probe_count", 0.0)) * 2.0 for sample in samples]),
        "detector_failure_rate": detector_failure_rate(samples),
    }


def risk_table(
    samples: list[dict[str, Any]],
    risk_names: list[str],
    coverage: float,
    *,
    include_auprc: bool = False,
) -> dict[str, dict[str, float]]:
    labels = [float(sample["error"]) for sample in samples]
    hallucination_labels = [float(sample["hallucination"]) for sample in samples]
    out: dict[str, dict[str, float]] = {}
    suffix = int(round(coverage * 100))
    for risk_name in risk_names:
        risk_key = RISK_KEYS[risk_name]
        scores = [float(sample.get(risk_key, 0.0)) for sample in samples]
        row = {
            f"selective_accuracy_{suffix}cov": 1.0 - coverage_error(samples, coverage, risk_key=risk_key),
            f"hallucination_fpr_{suffix}cov": coverage_hallucination_fpr(samples, coverage, risk_key=risk_key),
            "error_auroc": roc_auc(labels, scores),
            "hallucination_auroc": roc_auc(hallucination_labels, scores),
        }
        if include_auprc:
            row["error_auprc"] = average_precision(labels, scores)
            row["hallucination_auprc"] = average_precision(hallucination_labels, scores)
        out[risk_name] = row
    return out


def grouped_risk_table(
    samples: list[dict[str, Any]],
    *,
    key: str,
    risk_names: list[str],
    coverage: float,
    include_auprc: bool = False,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        groups[str(sample.get(key, "") or "<empty>")].append(sample)

    out: dict[str, dict[str, Any]] = {}
    for name, group in sorted(groups.items()):
        out[name] = {
            "overview": {
                "num_samples": float(len(group)),
                "direct_accuracy": mean([1.0 - float(sample["error"]) for sample in group]),
                "direct_hallucination_fpr": hallucination_fpr(group),
            },
            "risks": risk_table(group, risk_names, coverage, include_auprc=include_auprc),
        }
    return out


def best_rows(samples: list[dict[str, Any]], risk_names: list[str], coverage: float) -> dict[str, str]:
    rows = risk_table(samples, risk_names, coverage)
    suffix = int(round(coverage * 100))
    if not rows:
        return {}
    acc_key = f"selective_accuracy_{suffix}cov"
    fpr_key = f"hallucination_fpr_{suffix}cov"
    return {
        "best_selective_accuracy": max(rows.items(), key=lambda item: item[1][acc_key])[0],
        "best_hallucination_fpr": min(rows.items(), key=lambda item: item[1][fpr_key])[0],
        "best_error_auroc": max(rows.items(), key=lambda item: item[1]["error_auroc"])[0],
        "best_hallucination_auroc": max(rows.items(), key=lambda item: item[1]["hallucination_auroc"])[0],
    }


def cost_report(
    samples: list[dict[str, Any]],
    grouped: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    current_probe_counts = [float(sample.get("probe_count", 0.0)) for sample in samples]
    reuse_probe_counts = [claim_reuse_probe_count(grouped.get(str(sample["sample_id"]), {})) for sample in samples]
    return {
        "current": {
            "avg_probe_count": mean(current_probe_counts),
            "avg_inference_calls": mean([count * 2.0 for count in current_probe_counts]),
        },
        "claim_reuse_estimate": {
            "avg_probe_count": mean(reuse_probe_counts),
            "avg_inference_calls": mean([count * 2.0 for count in reuse_probe_counts]),
            "avg_inference_calls_saved": mean(
                [(current - reuse) * 2.0 for current, reuse in zip(current_probe_counts, reuse_probe_counts)]
            ),
            "note": "Estimate assumes orig/text_only can replace recap claim image/text probes.",
        },
    }


def claim_reuse_probe_count(probes: dict[str, dict[str, Any]]) -> float:
    candidates = recap_candidate_records(probes)
    anti_count = 0
    support_count = 0
    for (_, role), record in candidates.items():
        has_img = "img" in record
        has_text = "text" in record
        pair_count = float(has_img) + float(has_text)
        if role == "anti":
            anti_count += int(pair_count)
        elif role == "support":
            support_count += int(pair_count)
    if anti_count or support_count:
        return 2.0 + float(anti_count + support_count)
    return float(len(probes) or 0)
