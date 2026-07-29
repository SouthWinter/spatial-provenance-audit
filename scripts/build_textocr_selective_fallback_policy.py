#!/usr/bin/env python
"""Build split-safe selective fallback policies for TextOCR-Hard.

The policy starts from a cheap visual-token budget and falls back to a larger
budget when the cheap run is uncertain. This is an offline simulation over
cached fixed-budget outputs: it does not claim a new model run, but it gives a
clean dev/test estimate of whether selective fallback can replace manual ratio
selection.
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
    "abs_margin",
    "entropy_risk",
    "negative_margin",
    "u_conf_rank",
    "rice_profile_mean",
    "rice_profile_max",
    "rice_recap_selector",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="runs/textocr_selective_fallback/qwen_target_uncertainty_v1")
    parser.add_argument("--run", action="append", default=[], help="budget:path with sample_scores.jsonl")
    parser.add_argument("--base-budgets", default="0.20,0.25,0.30")
    parser.add_argument("--fallback-budgets", default="0.30,0.35,0.50,1.00")
    parser.add_argument("--dev-buckets", type=int, default=5)
    parser.add_argument("--target-hfpr", type=float, default=0.224)
    parser.add_argument("--lambda-hfpr", type=float, default=2.0)
    parser.add_argument("--lambda-keep", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=30)
    args = parser.parse_args()

    runs = dict(DEFAULT_RUNS)
    for item in args.run:
        budget, path = item.split(":", 1)
        runs[normalize_budget(budget)] = path

    base_budgets = [normalize_budget(x) for x in split_csv(args.base_budgets)]
    fallback_budgets = [normalize_budget(x) for x in split_csv(args.fallback_budgets)]
    scores = {budget: load_scores(Path(path) / "sample_scores.jsonl") for budget, path in runs.items()}
    ids = sorted(set.intersection(*(set(scores[b]) for b in sorted(scores))))
    if not ids:
        raise ValueError("No common sample ids across budget runs.")

    fixed_rows = []
    for budget in sorted(scores, key=float):
        row: dict[str, Any] = {"policy": f"Fixed {budget}", "budget": budget}
        for split in ("dev", "test", "all"):
            row.update(prefixed(split, evaluate(ids, scores, budget, None, None, split, args.dev_buckets)))
        fixed_rows.append(row)

    candidates: list[dict[str, Any]] = []
    for base in base_budgets:
        for fallback in fallback_budgets:
            if float(fallback) <= float(base):
                continue
            rows = scores[base]
            feature_values = {feature: sorted(feature_value(row, feature) for row in rows.values()) for feature in FEATURES}
            for feature in FEATURES:
                thresholds = quantile_thresholds(feature_values[feature])
                for direction in ("low", "high"):
                    for threshold in thresholds:
                        dev = evaluate(ids, scores, base, fallback, (feature, direction, threshold), "dev", args.dev_buckets)
                        test = evaluate(ids, scores, base, fallback, (feature, direction, threshold), "test", args.dev_buckets)
                        all_metrics = evaluate(ids, scores, base, fallback, (feature, direction, threshold), "all", args.dev_buckets)
                        objective = (
                            dev["accuracy"]
                            - args.lambda_hfpr * max(0.0, dev["hFPR"] - args.target_hfpr)
                            - args.lambda_keep * dev["mean_keep_ratio"]
                        )
                        candidates.append(
                            {
                                "policy": f"{base}->{fallback} if {feature} {direction} {threshold:.6g}",
                                "base_budget": base,
                                "fallback_budget": fallback,
                                "feature": feature,
                                "direction": direction,
                                "threshold": threshold,
                                "objective": objective,
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
            -row["dev_fallback_rate"],
        ),
        reverse=True,
    )
    best = candidates[0]
    best_policy = (
        str(best["base_budget"]),
        str(best["fallback_budget"]),
        (str(best["feature"]), str(best["direction"]), float(best["threshold"])),
    )
    prediction_rows = build_predictions(ids, scores, best_policy, args.dev_buckets)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "fixed_budget_summary.csv", fixed_rows)
    write_csv(out_dir / "policy_candidates.csv", candidates)
    write_jsonl(out_dir / "policy_predictions.jsonl", prediction_rows)
    write_json(out_dir / "policy_config.json", best)
    write_report(out_dir / "policy_report.md", fixed_rows, candidates[: args.top_k], best, len(ids))
    print(f"Wrote selective fallback policy to {out_dir}")
    print(
        f"Best {best['policy']} test_acc={best['test_accuracy']:.3f} "
        f"test_hFPR={best['test_hFPR']:.3f} test_keep={best['test_mean_keep_ratio']:.3f} "
        f"test_fallback={best['test_fallback_rate']:.3f}"
    )


def evaluate(
    ids: list[str],
    scores: dict[str, dict[str, dict[str, Any]]],
    base_budget: str,
    fallback_budget: str | None,
    rule: tuple[str, str, float] | None,
    split: str,
    dev_buckets: int,
) -> dict[str, float]:
    selected = [sid for sid in ids if split == "all" or stable_split(sid, dev_buckets) == split]
    if not selected:
        return empty_metrics()
    correct = 0
    negatives = 0
    hallucinations = 0
    fallback_count = 0
    keep_total = 0.0
    cascade_keep_total = 0.0
    for sid in selected:
        use_fallback = False
        if fallback_budget is not None and rule is not None:
            use_fallback = should_fallback(scores[base_budget][sid], rule)
        budget = fallback_budget if use_fallback and fallback_budget is not None else base_budget
        row = scores[budget][sid]
        correct += int(bool(row.get("direct_correct", False)))
        is_negative = bool(row.get("target_is_negative", False))
        negatives += int(is_negative)
        hallucinations += int(is_negative and bool(row.get("hallucination", False)))
        fallback_count += int(use_fallback)
        keep_total += float(budget)
        cascade_keep_total += float(base_budget) + (float(budget) if use_fallback else 0.0)
    n = len(selected)
    return {
        "n": float(n),
        "accuracy": correct / n,
        "hFPR": hallucinations / negatives if negatives else 0.0,
        "mean_keep_ratio": keep_total / n,
        "cascade_keep_ratio": cascade_keep_total / n,
        "fallback_rate": fallback_count / n,
    }


def build_predictions(
    ids: list[str],
    scores: dict[str, dict[str, dict[str, Any]]],
    policy: tuple[str, str, tuple[str, str, float]],
    dev_buckets: int,
) -> list[dict[str, Any]]:
    base, fallback, rule = policy
    rows = []
    for sid in ids:
        fallback_used = should_fallback(scores[base][sid], rule)
        budget = fallback if fallback_used else base
        score = scores[budget][sid]
        rows.append(
            {
                "sample_id": sid,
                "split": stable_split(sid, dev_buckets),
                "base_budget": float(base),
                "selected_budget": float(budget),
                "fallback_used": fallback_used,
                "direct_correct": bool(score.get("direct_correct", False)),
                "target_is_negative": bool(score.get("target_is_negative", False)),
                "hallucination": bool(score.get("hallucination", False)),
                "policy_feature": rule[0],
                "policy_direction": rule[1],
                "policy_threshold": rule[2],
                "policy_score": feature_value(scores[base][sid], rule[0]),
            }
        )
    return rows


def should_fallback(row: dict[str, Any], rule: tuple[str, str, float]) -> bool:
    feature, direction, threshold = rule
    value = feature_value(row, feature)
    return value <= threshold if direction == "low" else value >= threshold


def feature_value(row: dict[str, Any], feature: str) -> float:
    margin = float(row.get("text_yes_margin", row.get("support_orig", 0.0)) or 0.0)
    if feature == "abs_margin":
        return abs(margin)
    if feature == "negative_margin":
        return -margin
    value = row.get(feature, 0.0)
    try:
        value_f = float(value)
    except Exception:
        return 0.0
    if math.isnan(value_f) or math.isinf(value_f):
        return 0.0
    return value_f


def quantile_thresholds(values: list[float]) -> list[float]:
    values = sorted(v for v in values if not math.isnan(v) and not math.isinf(v))
    if not values:
        return [0.0]
    qs = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    thresholds = []
    for q in qs:
        idx = min(len(values) - 1, max(0, round(q * (len(values) - 1))))
        thresholds.append(values[idx])
    return sorted(set(thresholds))


def load_scores(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
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


def prefixed(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def empty_metrics() -> dict[str, float]:
    return {
        "n": 0.0,
        "accuracy": 0.0,
        "hFPR": 0.0,
        "mean_keep_ratio": 0.0,
        "cascade_keep_ratio": 0.0,
        "fallback_rate": 0.0,
    }


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
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
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
        "# TextOCR-Hard Selective Fallback Policy",
        "",
        f"Samples: {n}. Split is image-level: both probes from an image stay in the same split.",
        "",
        "The policy first runs a cheap target-conditioned prefix and reruns with a larger prefix only when the cheap answer is uncertain. It is selected on the dev split and reported on the held-out test split.",
        "",
        "## Readout",
        "",
        f"The dev-selected policy is `{best['policy']}`. On the held-out split it obtains {best['test_accuracy']:.3f} accuracy, {best['test_hFPR']:.3f} hFPR, and {best['test_mean_keep_ratio']:.3f} final keep ratio. This is not a stable improvement over the best fixed-budget rows on the same split, so the current simple uncertainty-triggered fallback should be treated as a negative or preliminary result rather than a main-paper positive claim.",
        "",
        "## Fixed Budgets",
        "",
        "| Policy | dev Acc. | dev hFPR | test Acc. | test hFPR | all Acc. | all hFPR | Keep |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in fixed_rows:
        lines.append(
            f"| {row['policy']} | {row['dev_accuracy']:.3f} | {row['dev_hFPR']:.3f} | "
            f"{row['test_accuracy']:.3f} | {row['test_hFPR']:.3f} | "
            f"{row['all_accuracy']:.3f} | {row['all_hFPR']:.3f} | {row['all_mean_keep_ratio']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Best Dev-Selected Policy",
            "",
            "```json",
            json.dumps(best, indent=2, ensure_ascii=True),
            "```",
            "",
            "## Top Candidate Policies",
            "",
            "| Policy | dev Acc. | dev hFPR | dev Keep | test Acc. | test hFPR | test Keep | test Fallback |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in candidates:
        lines.append(
            f"| {row['policy']} | {row['dev_accuracy']:.3f} | {row['dev_hFPR']:.3f} | "
            f"{row['dev_mean_keep_ratio']:.3f} | {row['test_accuracy']:.3f} | {row['test_hFPR']:.3f} | "
            f"{row['test_mean_keep_ratio']:.3f} | {row['test_fallback_rate']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
