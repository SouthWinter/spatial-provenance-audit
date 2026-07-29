#!/usr/bin/env python
"""Search split-safe multi-signal adaptive risk policies for TextOCR-Hard.

This is an offline simulator over cached fixed-budget Qwen runs. It extends the
single-feature policy search with small feature ensembles and explicitly
separates deployable features from oracle audit features. The goal is to test a
problem.md risk: can a unified calibrated policy replace manually selecting a
fixed keep ratio?
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RUNS = {
    "0.20": "runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_topk_0p20",
    "0.25": "runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_topk_0p25",
    "0.30": "runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_topk_0p30",
    "0.35": "runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_topk_0p35",
    "0.40": "runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_topk_0p40",
    "0.50": "runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_topk_0p50",
    "1.00": "runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_topk_1p00",
}

FEATURE_SPECS = (
    # name, risk direction, source group
    ("abs_margin", "low", "deployable_answer"),
    ("entropy_risk", "high", "deployable_answer"),
    ("u_conf_rank", "high", "deployable_answer"),
    ("rice_profile_mean", "high", "deployable_answer"),
    ("rice_profile_max", "high", "deployable_answer"),
    ("rice_recap_selector", "high", "deployable_answer"),
    ("target_text_token_count", "high", "deployable_selector"),
    ("full_visual_tokens", "high", "deployable_selector"),
    ("selected_row_span", "high", "deployable_selector"),
    ("selected_col_span", "high", "deployable_selector"),
    ("selected_area_span", "high", "deployable_selector"),
    ("selected_row_coverage", "high", "deployable_selector"),
    ("selected_col_coverage", "high", "deployable_selector"),
    ("selected_spatial_entropy", "high", "deployable_selector"),
    ("selected_compactness", "low", "deployable_selector"),
    ("ecr", "low", "oracle_audit"),
    ("evidence_center_recall", "low", "oracle_audit"),
    ("evidence_patch_recall", "low", "oracle_audit"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="runs/textocr_adaptive_policy/qwen_multisignal_risk_v1")
    parser.add_argument("--dev-buckets", type=int, default=5)
    parser.add_argument("--target-hfpr", type=float, default=0.224)
    parser.add_argument("--lambda-hfpr", type=float, default=2.0)
    parser.add_argument("--target-ecr", type=float, default=0.0)
    parser.add_argument("--lambda-ecr", type=float, default=0.0)
    parser.add_argument("--lambda-final-keep", type=float, default=0.05)
    parser.add_argument("--lambda-cascade-keep", type=float, default=0.02)
    parser.add_argument("--top-k", type=int, default=80)
    args = parser.parse_args()

    scores, traces, ids = load_runs(DEFAULT_RUNS)
    split_by_id = {sid: stable_split(sid, args.dev_buckets) for sid in ids}
    fixed_rows = build_fixed_rows(scores, traces, ids, split_by_id)

    candidates = []
    for mode in ("deployable", "oracle_audit"):
        allowed = feature_specs_for_mode(mode)
        candidates.extend(search_mode(mode, allowed, scores, traces, ids, split_by_id, args))

    candidates.sort(key=candidate_sort_key, reverse=True)
    best_by_mode = []
    for mode in ("deployable", "oracle_audit"):
        mode_rows = [row for row in candidates if row["mode"] == mode]
        if mode_rows:
            best_by_mode.append(mode_rows[0])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "fixed_budget_summary.csv", fixed_rows)
    write_csv(out_dir / "policy_candidates.csv", candidates)
    write_csv(out_dir / "top_policy_candidates.csv", candidates[: args.top_k])
    write_csv(out_dir / "best_by_mode.csv", best_by_mode)
    write_json(out_dir / "policy_config.json", {"best_by_mode": best_by_mode})
    write_report(out_dir / "policy_report.md", fixed_rows, candidates[: args.top_k], best_by_mode, len(ids))

    print(f"Wrote multi-signal adaptive policy search to {out_dir}")
    for row in best_by_mode:
        print(
            f"{row['mode']}: {row['policy']} test_acc={row['test_accuracy']:.3f} "
            f"test_hFPR={row['test_hFPR']:.3f} test_keep={row['test_mean_keep_ratio']:.3f} "
            f"test_cascade={row['test_cascade_keep_ratio']:.3f} test_ECR={row['test_mean_ecr']:.3f}"
        )


def search_mode(
    mode: str,
    feature_specs: list[tuple[str, str, str]],
    scores: dict[str, dict[str, dict[str, Any]]],
    traces: dict[str, dict[str, dict[str, Any]]],
    ids: list[str],
    split_by_id: dict[str, str],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    bases = ["0.20", "0.25", "0.30"]
    mids = ["0.25", "0.30", "0.35", "0.40"]
    highs = ["0.30", "0.35", "0.40", "0.50", "1.00"]
    quantiles = (0.50, 0.60, 0.75, 0.90, 0.95)
    weight_patterns = ((1.0,), (1.0, 1.0))
    rows = []
    for base in bases:
        feature_ranks = {
            spec[0]: percentile_risk(scores, traces, ids, base, spec[0], spec[1]) for spec in feature_specs
        }
        feature_sets = curated_feature_sets(feature_specs)
        for specs in feature_sets:
            patterns = [p for p in weight_patterns if len(p) == len(specs)]
            for weights in patterns:
                risk = ensemble_risk(ids, feature_ranks, specs, weights)
                thresholds = quantile_thresholds(list(risk.values()), quantiles)
                for mid in mids:
                    if float(mid) <= float(base):
                        continue
                    for high in highs:
                        if float(high) <= float(mid):
                            continue
                        for low_t in thresholds:
                            for high_t in thresholds:
                                if low_t > high_t:
                                    continue
                                dev = evaluate_policy(
                                    ids, split_by_id, scores, traces, risk, base, mid, high, low_t, high_t, "dev"
                                )
                                test = evaluate_policy(
                                    ids, split_by_id, scores, traces, risk, base, mid, high, low_t, high_t, "test"
                                )
                                objective = (
                                    dev["accuracy"]
                                    - args.lambda_hfpr * max(0.0, dev["hFPR"] - args.target_hfpr)
                                    - args.lambda_ecr * max(0.0, args.target_ecr - dev["mean_ecr"])
                                    - args.lambda_final_keep * dev["mean_keep_ratio"]
                                    - args.lambda_cascade_keep * dev["cascade_keep_ratio"]
                                )
                                policy = policy_name(base, mid, high, specs, weights, low_t, high_t)
                                rows.append(
                                    {
                                        "mode": mode,
                                        "policy": policy,
                                        "feature_names": "+".join(spec[0] for spec in specs),
                                        "feature_sources": "+".join(sorted(set(spec[2] for spec in specs))),
                                        "weights": "+".join(f"{w:g}" for w in weights),
                                        "base_budget": base,
                                        "mid_budget": mid,
                                        "high_budget": high,
                                        "low_threshold": low_t,
                                        "high_threshold": high_t,
                                        "objective": objective,
                                        **prefixed("dev", dev),
                                        **prefixed("test", test),
                                    }
                                )
    rows.sort(key=candidate_sort_key, reverse=True)
    for row in rows[:200]:
        base = str(row["base_budget"])
        mid = str(row["mid_budget"])
        high = str(row["high_budget"])
        specs = tuple(spec for spec in feature_specs if spec[0] in str(row["feature_names"]).split("+"))
        weights = tuple(float(value) for value in str(row["weights"]).split("+"))
        feature_ranks = {
            spec[0]: percentile_risk(scores, traces, ids, base, spec[0], spec[1]) for spec in specs
        }
        risk = ensemble_risk(ids, feature_ranks, specs, weights)
        all_metrics = evaluate_policy(
            ids,
            split_by_id,
            scores,
            traces,
            risk,
            base,
            mid,
            high,
            float(row["low_threshold"]),
            float(row["high_threshold"]),
            "all",
        )
        row.update(prefixed("all", all_metrics))
    return rows


def curated_feature_sets(
    feature_specs: list[tuple[str, str, str]],
) -> list[tuple[tuple[str, str, str], ...]]:
    by_name = {spec[0]: spec for spec in feature_specs}
    sets: list[tuple[tuple[str, str, str], ...]] = [(spec,) for spec in feature_specs]
    pair_names = (
        ("abs_margin", "entropy_risk"),
        ("abs_margin", "target_text_token_count"),
        ("entropy_risk", "target_text_token_count"),
        ("u_conf_rank", "target_text_token_count"),
        ("rice_profile_mean", "target_text_token_count"),
        ("abs_margin", "selected_row_span"),
        ("abs_margin", "selected_col_span"),
        ("abs_margin", "selected_area_span"),
        ("abs_margin", "selected_spatial_entropy"),
        ("entropy_risk", "selected_row_span"),
        ("entropy_risk", "selected_spatial_entropy"),
        ("target_text_token_count", "selected_row_span"),
        ("target_text_token_count", "selected_spatial_entropy"),
        ("selected_row_span", "selected_col_span"),
        ("selected_area_span", "selected_spatial_entropy"),
        ("selected_row_coverage", "selected_col_coverage"),
        ("ecr", "abs_margin"),
        ("ecr", "entropy_risk"),
        ("ecr", "target_text_token_count"),
        ("evidence_center_recall", "abs_margin"),
        ("evidence_patch_recall", "abs_margin"),
    )
    for left, right in pair_names:
        if left in by_name and right in by_name:
            sets.append((by_name[left], by_name[right]))
    return sets


def feature_specs_for_mode(mode: str) -> list[tuple[str, str, str]]:
    if mode == "deployable":
        return [spec for spec in FEATURE_SPECS if spec[2].startswith("deployable")]
    if mode == "oracle_audit":
        return list(FEATURE_SPECS)
    raise ValueError(mode)


def percentile_risk(
    scores: dict[str, dict[str, dict[str, Any]]],
    traces: dict[str, dict[str, dict[str, Any]]],
    ids: list[str],
    budget: str,
    feature: str,
    direction: str,
) -> dict[str, float]:
    values = []
    for sid in ids:
        value = feature_value(scores[budget][sid], traces[budget][sid], feature)
        risk = -value if direction == "low" else value
        values.append((risk, sid))
    values.sort()
    denom = max(1, len(values) - 1)
    return {sid: idx / denom for idx, (_, sid) in enumerate(values)}


def ensemble_risk(
    ids: list[str],
    feature_ranks: dict[str, dict[str, float]],
    specs: tuple[tuple[str, str, str], ...],
    weights: tuple[float, ...],
) -> dict[str, float]:
    denom = sum(weights)
    return {
        sid: sum(weight * feature_ranks[spec[0]][sid] for weight, spec in zip(weights, specs)) / denom
        for sid in ids
    }


def evaluate_policy(
    ids: list[str],
    split_by_id: dict[str, str],
    scores: dict[str, dict[str, dict[str, Any]]],
    traces: dict[str, dict[str, dict[str, Any]]],
    risk: dict[str, float],
    base: str,
    mid: str,
    high: str,
    low_threshold: float,
    high_threshold: float,
    split: str,
) -> dict[str, float]:
    selected = [sid for sid in ids if split == "all" or split_by_id[sid] == split]
    if not selected:
        return empty_metrics()
    correct = hallucinations = negatives = mid_count = high_count = 0
    final_keep = cascade_keep = ecr_total = center_total = patch_total = 0.0
    for sid in selected:
        budget = choose_budget(risk[sid], base, mid, high, low_threshold, high_threshold)
        row = scores[budget][sid]
        trace = traces[budget][sid]
        correct += int(bool(row.get("direct_correct", False)))
        is_negative = bool(row.get("target_is_negative", False))
        negatives += int(is_negative)
        hallucinations += int(is_negative and bool(row.get("hallucination", False)))
        final_keep += float(budget)
        cascade_keep += float(base) + (float(budget) if float(budget) > float(base) else 0.0)
        ecr_total += finite_float(trace.get("ecr", 0.0))
        center_total += finite_float(trace.get("evidence_center_recall", 0.0))
        patch_total += finite_float(trace.get("evidence_patch_recall", 0.0))
        mid_count += int(budget == mid)
        high_count += int(budget == high)
    n = len(selected)
    return {
        "n": float(n),
        "accuracy": correct / n,
        "hFPR": hallucinations / negatives if negatives else 0.0,
        "mean_keep_ratio": final_keep / n,
        "cascade_keep_ratio": cascade_keep / n,
        "mean_ecr": ecr_total / n,
        "mean_center_recall": center_total / n,
        "mean_patch_recall": patch_total / n,
        "mid_rate": mid_count / n,
        "high_rate": high_count / n,
    }


def build_fixed_rows(
    scores: dict[str, dict[str, dict[str, Any]]],
    traces: dict[str, dict[str, dict[str, Any]]],
    ids: list[str],
    split_by_id: dict[str, str],
) -> list[dict[str, Any]]:
    rows = []
    dummy_risk = {sid: 0.0 for sid in ids}
    for budget in sorted(scores, key=float):
        row: dict[str, Any] = {"policy": f"Fixed {budget}", "budget": budget}
        for split in ("dev", "test", "all"):
            row.update(
                prefixed(
                    split,
                    evaluate_policy(ids, split_by_id, scores, traces, dummy_risk, budget, budget, budget, 1.0, 2.0, split),
                )
            )
        rows.append(row)
    return rows


def choose_budget(risk: float, base: str, mid: str, high: str, low_threshold: float, high_threshold: float) -> str:
    if risk >= high_threshold:
        return high
    if risk >= low_threshold:
        return mid
    return base


def feature_value(score: dict[str, Any], trace: dict[str, Any], feature: str) -> float:
    if feature == "abs_margin":
        margin = score.get("text_yes_margin", score.get("support_orig", 0.0))
        return abs(finite_float(margin))
    if feature.startswith("selected_"):
        return selected_spatial_feature(trace, feature)
    if feature in trace:
        return finite_float(trace.get(feature, 0.0))
    return finite_float(score.get(feature, 0.0))


def selected_spatial_feature(trace: dict[str, Any], feature: str) -> float:
    kept = trace.get("kept_indices", [])
    if not isinstance(kept, list) or not kept:
        return 0.0
    grid_h = max(1, int(finite_float(trace.get("grid_h", 1))))
    grid_w = max(1, int(finite_float(trace.get("grid_w", 1))))
    rows = [max(0, min(grid_h - 1, int(idx) // grid_w)) for idx in kept]
    cols = [max(0, min(grid_w - 1, int(idx) % grid_w)) for idx in kept]
    row_span = (max(rows) - min(rows) + 1) / grid_h
    col_span = (max(cols) - min(cols) + 1) / grid_w
    row_cov = len(set(rows)) / grid_h
    col_cov = len(set(cols)) / grid_w
    area_span = row_span * col_span
    compactness = len(kept) / max(1.0, (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1))
    values = {
        "selected_row_span": row_span,
        "selected_col_span": col_span,
        "selected_area_span": area_span,
        "selected_row_coverage": row_cov,
        "selected_col_coverage": col_cov,
        "selected_spatial_entropy": spatial_entropy(rows, cols, grid_h=grid_h, grid_w=grid_w),
        "selected_compactness": compactness,
    }
    return values.get(feature, 0.0)


def spatial_entropy(rows: list[int], cols: list[int], *, grid_h: int, grid_w: int) -> float:
    # Coarse bins keep the risk feature comparable across dynamic image sizes.
    bins_h = min(8, max(1, grid_h))
    bins_w = min(8, max(1, grid_w))
    counts: dict[tuple[int, int], int] = {}
    for row, col in zip(rows, cols):
        brow = min(bins_h - 1, int(row * bins_h / max(1, grid_h)))
        bcol = min(bins_w - 1, int(col * bins_w / max(1, grid_w)))
        counts[(brow, bcol)] = counts.get((brow, bcol), 0) + 1
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log(p)
    return entropy / math.log(max(2, bins_h * bins_w))


def quantile_thresholds(values: list[float], quantiles: tuple[float, ...]) -> list[float]:
    values = sorted(v for v in values if math.isfinite(v))
    if not values:
        return [0.0]
    out = []
    for q in quantiles:
        idx = min(len(values) - 1, max(0, round(q * (len(values) - 1))))
        out.append(values[idx])
    return sorted(set(out))


def policy_name(
    base: str,
    mid: str,
    high: str,
    specs: tuple[tuple[str, str, str], ...],
    weights: tuple[float, ...],
    low_t: float,
    high_t: float,
) -> str:
    parts = [f"{spec[0]}:{weight:g}" for spec, weight in zip(specs, weights)]
    return f"{base}->{mid}->{high} by risk({'+'.join(parts)}) >= {low_t:.3f}/{high_t:.3f}"


def candidate_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        finite_float(row.get("objective", 0.0)),
        finite_float(row.get("dev_accuracy", 0.0)),
        -finite_float(row.get("dev_hFPR", 0.0)),
        -finite_float(row.get("dev_mean_keep_ratio", 0.0)),
        -finite_float(row.get("dev_cascade_keep_ratio", 0.0)),
    )


def load_runs(run_map: dict[str, str]) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, dict[str, Any]]], list[str]]:
    scores = {}
    traces = {}
    for budget, rel_path in run_map.items():
        path = ROOT / rel_path
        scores[budget] = load_jsonl_by_id(path / "sample_scores.jsonl")
        traces[budget] = load_jsonl_by_id(path / "prune_traces.jsonl")
    ids = sorted(set.intersection(*(set(rows) for rows in scores.values())))
    if not ids:
        raise ValueError("No common sample ids across score runs.")
    return scores, traces, ids


def load_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row.get("sample_id", row.get("id", "")))] = row
    return rows


def stable_split(sample_id_value: str, dev_buckets: int) -> str:
    image_id = sample_id_value.split(":", 1)[0]
    bucket = int(hashlib.sha1(image_id.encode("utf-8")).hexdigest()[:8], 16) % 10
    return "dev" if bucket < dev_buckets else "test"


def finite_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(out) or math.isinf(out):
        return 0.0
    return out


def prefixed(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def empty_metrics() -> dict[str, float]:
    return {
        "n": 0.0,
        "accuracy": 0.0,
        "hFPR": 0.0,
        "mean_keep_ratio": 0.0,
        "cascade_keep_ratio": 0.0,
        "mean_ecr": 0.0,
        "mean_center_recall": 0.0,
        "mean_patch_recall": 0.0,
        "mid_rate": 0.0,
        "high_rate": 0.0,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, fixed_rows: list[dict[str, Any]], candidates: list[dict[str, Any]], best_by_mode: list[dict[str, Any]], n: int) -> None:
    lines = [
        "# TextOCR-Hard Multi-Signal Adaptive Risk Policy",
        "",
        f"Samples: {n}. Split is image-level, so both probes from one image stay in the same split.",
        "",
        "The search is an offline simulation over cached fixed-budget outputs. Deployable policies use only answer/selector-side signals available during inference. Oracle-audit policies additionally use annotated ECR-style evidence coverage and are reported only as an upper-bound diagnostic.",
        "",
        "## Best By Mode",
        "",
        "| Mode | Policy | test Acc. | test hFPR | test Keep | test Cascade | test ECR | Interpretation |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    fixed_test_best = max(fixed_rows, key=lambda row: row["test_accuracy"])
    for row in best_by_mode:
        interpretation = interpret_policy(row, fixed_test_best)
        lines.append(
            f"| {row['mode']} | {row['policy']} | {row['test_accuracy']:.3f} | {row['test_hFPR']:.3f} | "
            f"{row['test_mean_keep_ratio']:.3f} | {row['test_cascade_keep_ratio']:.3f} | "
            f"{row['test_mean_ecr']:.3f} | {interpretation} |"
        )
    lines.extend(
        [
            "",
            "## Fixed Budgets",
            "",
            "| Policy | dev Acc. | dev hFPR | test Acc. | test hFPR | test Keep | test ECR | all Acc. | all hFPR | all Keep | all ECR |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in fixed_rows:
        lines.append(
            f"| {row['policy']} | {row['dev_accuracy']:.3f} | {row['dev_hFPR']:.3f} | "
            f"{row['test_accuracy']:.3f} | {row['test_hFPR']:.3f} | {row['test_mean_keep_ratio']:.3f} | "
            f"{row['test_mean_ecr']:.3f} | {row['all_accuracy']:.3f} | {row['all_hFPR']:.3f} | "
            f"{row['all_mean_keep_ratio']:.3f} | {row['all_mean_ecr']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Top Candidate Policies",
            "",
            "| Mode | Policy | Sources | dev Acc. | dev hFPR | dev Keep | test Acc. | test hFPR | test Keep | test Cascade | test ECR |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in candidates:
        lines.append(
            f"| {row['mode']} | {row['policy']} | {row['feature_sources']} | {row['dev_accuracy']:.3f} | "
            f"{row['dev_hFPR']:.3f} | {row['dev_mean_keep_ratio']:.3f} | {row['test_accuracy']:.3f} | "
            f"{row['test_hFPR']:.3f} | {row['test_mean_keep_ratio']:.3f} | "
            f"{row['test_cascade_keep_ratio']:.3f} | {row['test_mean_ecr']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def interpret_policy(row: dict[str, Any], fixed_test_best: dict[str, Any]) -> str:
    if row["test_accuracy"] > fixed_test_best["test_accuracy"] and row["test_mean_keep_ratio"] < fixed_test_best["test_mean_keep_ratio"]:
        return "beats the best fixed test-accuracy point at lower final keep, but cascade cost must be considered"
    if row["test_accuracy"] >= fixed_test_best["test_accuracy"]:
        return "matches or exceeds the best fixed test accuracy but does not clearly reduce total fallback cost"
    return "does not beat the fixed-budget frontier; treat as negative or diagnostic"


if __name__ == "__main__":
    main()
