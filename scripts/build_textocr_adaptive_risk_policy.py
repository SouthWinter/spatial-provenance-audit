#!/usr/bin/env python
"""Search split-safe adaptive risk policies for TextOCR-Hard.

This script is an offline simulator over cached fixed-budget runs. It asks
whether a single calibrated risk signal can replace manual keep-ratio choice:
run a cheap prefix first, then assign a larger budget only for risky samples.
It reports both the final visual-token ratio and the cascade ratio, because a
real fallback system pays for the cheap pass plus any rerun.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_RUNS = {
    "0.20": "runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_topk_0p20",
    "0.25": "runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_topk_0p25",
    "0.30": "runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_topk_0p30",
    "0.35": "runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_topk_0p35",
    "0.40": "runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_topk_0p40",
    "0.50": "runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_topk_0p50",
    "1.00": "runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_topk_1p00",
}

FEATURES = (
    "abs_margin:risk_low:answer",
    "entropy_risk:risk_high:answer",
    "u_conf_rank:risk_high:answer",
    "rice_profile_mean:risk_high:answer",
    "rice_profile_max:risk_high:answer",
    "rice_recap_selector:risk_high:answer",
    "ecr:risk_low:evidence",
    "evidence_center_recall:risk_low:evidence",
    "evidence_patch_recall:risk_low:evidence",
    "target_text_token_count:risk_high:selector",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="runs/textocr_adaptive_policy/qwen_target_risk_v1")
    parser.add_argument("--run", action="append", default=[], help="budget:path with sample_scores.jsonl and prune_traces.jsonl")
    parser.add_argument("--base-budgets", default="0.20,0.25,0.30")
    parser.add_argument("--mid-budgets", default="0.25,0.30,0.35,0.40")
    parser.add_argument("--high-budgets", default="0.30,0.35,0.40,0.50,1.00")
    parser.add_argument("--dev-buckets", type=int, default=5)
    parser.add_argument("--target-hfpr", type=float, default=0.224)
    parser.add_argument("--lambda-hfpr", type=float, default=2.0)
    parser.add_argument("--lambda-final-keep", type=float, default=0.05)
    parser.add_argument("--lambda-cascade-keep", type=float, default=0.02)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    runs = dict(DEFAULT_RUNS)
    for item in args.run:
        budget, path = item.split(":", 1)
        runs[normalize_budget(budget)] = path

    scores = {budget: load_jsonl_by_id(Path(path) / "sample_scores.jsonl") for budget, path in runs.items()}
    traces = {budget: load_jsonl_by_id(Path(path) / "prune_traces.jsonl") for budget, path in runs.items()}
    ids = sorted(set.intersection(*(set(scores[b]) for b in sorted(scores))))
    if not ids:
        raise ValueError("No common sample ids across score runs.")

    base_budgets = [normalize_budget(x) for x in split_csv(args.base_budgets)]
    mid_budgets = [normalize_budget(x) for x in split_csv(args.mid_budgets)]
    high_budgets = [normalize_budget(x) for x in split_csv(args.high_budgets)]

    fixed_rows = []
    for budget in sorted(scores, key=float):
        row: dict[str, Any] = {"policy": f"Fixed {budget}", "base_budget": budget, "mid_budget": "", "high_budget": ""}
        for split in ("dev", "test", "all"):
            row.update(prefixed(split, evaluate_fixed(ids, scores, traces, budget, split, args.dev_buckets)))
        fixed_rows.append(row)

    candidates = []
    for base in base_budgets:
        feature_specs = [parse_feature(spec) for spec in FEATURES]
        for feature, direction, source in feature_specs:
            values = sorted(feature_value(scores[base], traces[base], sid, feature) for sid in ids)
            thresholds = quantile_thresholds(values)
            for mid in mid_budgets:
                if float(mid) <= float(base):
                    continue
                for high in high_budgets:
                    if float(high) <= float(mid):
                        continue
                    for low_t in thresholds:
                        for high_t in thresholds:
                            if low_t > high_t:
                                continue
                            policy = {
                                "base_budget": base,
                                "mid_budget": mid,
                                "high_budget": high,
                                "feature": feature,
                                "risk_direction": direction,
                                "feature_source": source,
                                "low_threshold": low_t,
                                "high_threshold": high_t,
                            }
                            dev = evaluate_policy(ids, scores, traces, policy, "dev", args.dev_buckets)
                            test = evaluate_policy(ids, scores, traces, policy, "test", args.dev_buckets)
                            all_metrics = evaluate_policy(ids, scores, traces, policy, "all", args.dev_buckets)
                            objective = (
                                dev["accuracy"]
                                - args.lambda_hfpr * max(0.0, dev["hFPR"] - args.target_hfpr)
                                - args.lambda_final_keep * dev["mean_keep_ratio"]
                                - args.lambda_cascade_keep * dev["cascade_keep_ratio"]
                            )
                            candidates.append(
                                {
                                    "policy": policy_name(policy),
                                    "objective": objective,
                                    **policy,
                                    **prefixed("dev", dev),
                                    **prefixed("test", test),
                                    **prefixed("all", all_metrics),
                                }
                            )

    candidates.sort(
        key=lambda row: (
            row["objective"],
            row["dev_accuracy"],
            -row["dev_hFPR"],
            -row["dev_mean_keep_ratio"],
            -row["dev_cascade_keep_ratio"],
        ),
        reverse=True,
    )
    best = candidates[0]
    predictions = build_predictions(ids, scores, traces, best, args.dev_buckets)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "fixed_budget_summary.csv", fixed_rows)
    write_csv(out_dir / "policy_candidates.csv", candidates)
    write_csv(out_dir / "top_policy_candidates.csv", candidates[: args.top_k])
    write_json(out_dir / "policy_config.json", best)
    write_jsonl(out_dir / "policy_predictions.jsonl", predictions)
    write_report(out_dir / "policy_report.md", fixed_rows, candidates[: args.top_k], best, len(ids))

    print(f"Wrote adaptive policy search to {out_dir}")
    print(
        f"Best {best['policy']} test_acc={best['test_accuracy']:.3f} "
        f"test_hFPR={best['test_hFPR']:.3f} test_keep={best['test_mean_keep_ratio']:.3f} "
        f"test_cascade={best['test_cascade_keep_ratio']:.3f} test_ECR={best['test_mean_ecr']:.3f}"
    )


def evaluate_fixed(
    ids: list[str],
    scores: dict[str, dict[str, dict[str, Any]]],
    traces: dict[str, dict[str, dict[str, Any]]],
    budget: str,
    split: str,
    dev_buckets: int,
) -> dict[str, float]:
    policy = {
        "base_budget": budget,
        "mid_budget": budget,
        "high_budget": budget,
        "feature": "abs_margin",
        "risk_direction": "risk_low",
        "feature_source": "answer",
        "low_threshold": -math.inf,
        "high_threshold": -math.inf,
    }
    return evaluate_policy(ids, scores, traces, policy, split, dev_buckets, fixed_budget=budget)


def evaluate_policy(
    ids: list[str],
    scores: dict[str, dict[str, dict[str, Any]]],
    traces: dict[str, dict[str, dict[str, Any]]],
    policy: dict[str, Any],
    split: str,
    dev_buckets: int,
    *,
    fixed_budget: str | None = None,
) -> dict[str, float]:
    selected = [sid for sid in ids if split == "all" or stable_split(sid, dev_buckets) == split]
    if not selected:
        return empty_metrics()
    correct = hallucinations = negatives = 0
    final_keep = cascade_keep = ecr_total = center_total = patch_total = 0.0
    mid_count = high_count = 0
    for sid in selected:
        budget = fixed_budget or choose_budget(scores, traces, sid, policy)
        row = scores[budget][sid]
        trace = traces.get(budget, {}).get(sid, {})
        correct += int(bool(row.get("direct_correct", False)))
        is_negative = bool(row.get("target_is_negative", False))
        negatives += int(is_negative)
        hallucinations += int(is_negative and bool(row.get("hallucination", False)))
        final_keep += float(budget)
        base = float(policy["base_budget"])
        cascade_keep += float(budget) if fixed_budget else base + (float(budget) if float(budget) > base else 0.0)
        ecr_total += finite_float(trace.get("ecr", 0.0))
        center_total += finite_float(trace.get("evidence_center_recall", 0.0))
        patch_total += finite_float(trace.get("evidence_patch_recall", 0.0))
        mid_count += int((not fixed_budget) and budget == policy["mid_budget"])
        high_count += int((not fixed_budget) and budget == policy["high_budget"])
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


def choose_budget(
    scores: dict[str, dict[str, dict[str, Any]]],
    traces: dict[str, dict[str, dict[str, Any]]],
    sid: str,
    policy: dict[str, Any],
) -> str:
    base = str(policy["base_budget"])
    value = feature_value(scores[base], traces.get(base, {}), sid, str(policy["feature"]))
    if is_high_risk(value, str(policy["risk_direction"]), float(policy["high_threshold"])):
        return str(policy["high_budget"])
    if is_high_risk(value, str(policy["risk_direction"]), float(policy["low_threshold"])):
        return str(policy["mid_budget"])
    return base


def is_high_risk(value: float, direction: str, threshold: float) -> bool:
    if direction == "risk_low":
        return value <= threshold
    return value >= threshold


def build_predictions(
    ids: list[str],
    scores: dict[str, dict[str, dict[str, Any]]],
    traces: dict[str, dict[str, dict[str, Any]]],
    policy: dict[str, Any],
    dev_buckets: int,
) -> list[dict[str, Any]]:
    rows = []
    for sid in ids:
        budget = choose_budget(scores, traces, sid, policy)
        score = scores[budget][sid]
        trace = traces.get(budget, {}).get(sid, {})
        base = str(policy["base_budget"])
        rows.append(
            {
                "sample_id": sid,
                "split": stable_split(sid, dev_buckets),
                "selected_budget": float(budget),
                "cascade_keep_ratio": float(base) + (float(budget) if float(budget) > float(base) else 0.0),
                "direct_correct": bool(score.get("direct_correct", False)),
                "target_is_negative": bool(score.get("target_is_negative", False)),
                "hallucination": bool(score.get("hallucination", False)),
                "ecr": finite_float(trace.get("ecr", 0.0)),
                "policy_feature": policy["feature"],
                "policy_score": feature_value(scores[base], traces.get(base, {}), sid, str(policy["feature"])),
            }
        )
    return rows


def feature_value(
    score_rows: dict[str, dict[str, Any]],
    trace_rows: dict[str, dict[str, Any]],
    sid: str,
    feature: str,
) -> float:
    score = score_rows.get(sid, {})
    trace = trace_rows.get(sid, {})
    margin = finite_float(score.get("text_yes_margin", score.get("support_orig", 0.0)))
    if feature == "abs_margin":
        return abs(margin)
    if feature in trace:
        return finite_float(trace.get(feature, 0.0))
    return finite_float(score.get(feature, 0.0))


def quantile_thresholds(values: list[float]) -> list[float]:
    values = sorted(v for v in values if math.isfinite(v))
    if not values:
        return [0.0]
    qs = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    out = []
    for q in qs:
        idx = min(len(values) - 1, max(0, round(q * (len(values) - 1))))
        out.append(values[idx])
    return sorted(set(out))


def parse_feature(spec: str) -> tuple[str, str, str]:
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"Bad feature spec: {spec}")
    return parts[0], parts[1], parts[2]


def load_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[sample_id(row)] = row
    return rows


def sample_id(row: dict[str, Any]) -> str:
    return str(row.get("sample_id", row.get("id", "")))


def stable_split(sample_id_value: str, dev_buckets: int) -> str:
    image_id = sample_id_value.split(":", 1)[0]
    bucket = int(hashlib.sha1(image_id.encode("utf-8")).hexdigest()[:8], 16) % 10
    return "dev" if bucket < dev_buckets else "test"


def split_csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def normalize_budget(value: str) -> str:
    return f"{float(value):.2f}"


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


def policy_name(policy: dict[str, Any]) -> str:
    direction = "<=" if policy["risk_direction"] == "risk_low" else ">="
    return (
        f"{policy['base_budget']}->{policy['mid_budget']}->{policy['high_budget']} "
        f"by {policy['feature']} {direction} "
        f"{float(policy['low_threshold']):.6g}/{float(policy['high_threshold']):.6g}"
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    fixed_rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    best: dict[str, Any],
    n: int,
) -> None:
    lines = [
        "# TextOCR-Hard Adaptive Risk Policy",
        "",
        f"Samples: {n}. Split is image-level: both probes from the same image stay in the same split.",
        "",
        "The search is an offline simulation over cached fixed-budget outputs. It calibrates a single risk feature on the dev split and reports the selected policy on the held-out split. `Keep` is the final retained visual-token ratio; `Cascade` is the cost proxy for a real fallback system that first runs the base budget and reruns risky samples.",
        "",
        "## Best Dev-Selected Policy",
        "",
        f"`{best['policy']}`",
        "",
        "| Split | Acc. | hFPR | Keep | Cascade | ECR | Mid rate | High rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("dev", "test", "all"):
        lines.append(
            f"| {split} | {best[f'{split}_accuracy']:.3f} | {best[f'{split}_hFPR']:.3f} | "
            f"{best[f'{split}_mean_keep_ratio']:.3f} | {best[f'{split}_cascade_keep_ratio']:.3f} | "
            f"{best[f'{split}_mean_ecr']:.3f} | {best[f'{split}_mid_rate']:.3f} | {best[f'{split}_high_rate']:.3f} |"
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
            "| Policy | Source | dev Acc. | dev hFPR | dev Keep | test Acc. | test hFPR | test Keep | test Cascade | test ECR |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in candidates:
        lines.append(
            f"| {row['policy']} | {row['feature_source']} | {row['dev_accuracy']:.3f} | "
            f"{row['dev_hFPR']:.3f} | {row['dev_mean_keep_ratio']:.3f} | "
            f"{row['test_accuracy']:.3f} | {row['test_hFPR']:.3f} | {row['test_mean_keep_ratio']:.3f} | "
            f"{row['test_cascade_keep_ratio']:.3f} | {row['test_mean_ecr']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
