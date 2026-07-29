#!/usr/bin/env python
"""Compute paired image-cluster intervals on the locked confirmation set.

The two probes associated with one TextOCR image are not independent. This
script first averages paired method differences within each image, then
bootstraps the 500 image-level means.
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs" / "textocr_confirmation"
OUT_DIR = ROOT / "runs" / "problem_optimization_audit"
N_BOOTSTRAP = 10_000
SEED = 20260724


@dataclass(frozen=True)
class Comparison:
    name: str
    left: str
    right: str


RUNS = {
    "qwen_full": "qwen3_8b_full",
    "qwen_target30": "qwen3_8b_target_0p30",
    "qwen_random30": "qwen3_8b_random_0p30",
    "qwen_grid30": "qwen3_8b_grid_0p30",
    "qwen_visionzip30": "qwen3_8b_visionzip_0p30",
    "llava_full": "llava15_7b_full",
    "llava_protected40": "llava15_7b_protected_0p40",
    "llava_random40": "llava15_7b_random_0p40",
    "llava_target40": "llava15_7b_target_0p40",
    "llava_scope40": "llava15_7b_scope_0p40",
    "llava_coin40": "llava15_7b_coin_0p40",
    "llava_visionzip40": "llava15_7b_visionzip_0p40",
}

COMPARISONS = [
    Comparison("qwen_target30_vs_full", "qwen_target30", "qwen_full"),
    Comparison("qwen_target30_vs_random30", "qwen_target30", "qwen_random30"),
    Comparison("qwen_target30_vs_grid30", "qwen_target30", "qwen_grid30"),
    Comparison("llava_protected40_vs_full", "llava_protected40", "llava_full"),
    Comparison("llava_protected40_vs_random40", "llava_protected40", "llava_random40"),
    Comparison("llava_target40_vs_scope40", "llava_target40", "llava_scope40"),
    Comparison("llava_coin40_vs_scope40", "llava_coin40", "llava_scope40"),
    Comparison("llava_protected40_vs_visionzip40", "llava_protected40", "llava_visionzip40"),
    Comparison("qwen_target30_vs_visionzip30", "qwen_target30", "qwen_visionzip30"),
]


def read_scores(run_name: str) -> dict[str, dict]:
    path = RUN_ROOT / run_name / "probe_scores.jsonl"
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return {row["sample_id"]: row for row in rows}


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def image_differences(
    left: dict[str, dict],
    right: dict[str, dict],
    value: Callable[[dict], float],
    required_polarity: str | None = None,
) -> tuple[list[float], int]:
    by_image: dict[str, list[float]] = defaultdict(list)
    common = sorted(set(left) & set(right))
    for sample_id in common:
        left_row = left[sample_id]
        right_row = right[sample_id]
        if required_polarity and left_row.get("binary_polarity") != required_polarity:
            continue
        image_id = str(left_row.get("image_id") or sample_id.split(":")[0])
        by_image[image_id].append(value(left_row) - value(right_row))
    return [mean(values) for values in by_image.values()], len(common)


def bootstrap_ci(values: list[float], seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    draws = []
    for _ in range(N_BOOTSTRAP):
        draws.append(mean([values[rng.randrange(len(values))] for _ in values]))
    draws.sort()
    return draws[int(0.025 * N_BOOTSTRAP)], draws[int(0.975 * N_BOOTSTRAP)]


def correct(row: dict) -> float:
    return float(bool(row["correct"]))


def predicts_yes(row: dict) -> float:
    return float(row["pred_answer"] == "yes")


def anchor_coverage(row: dict) -> float:
    return float(row.get("prune_anchor_ecr", row.get("prune_ecr", 1.0)))


def coverage(row: dict) -> float:
    if str(row.get("prune_selector", "")).strip().lower() in {
        "visionzip",
        "official_visionzip",
        "qwen3_visionzip",
    }:
        return 1.0
    return float(row.get("prune_ecr", 1.0))


def summarize(comparison: Comparison, runs: dict[str, dict[str, dict]], index: int) -> dict:
    left = runs[comparison.left]
    right = runs[comparison.right]
    acc, n_probes = image_differences(left, right, correct)
    hfpr, _ = image_differences(
        left, right, predicts_yes, required_polarity="negative"
    )
    ecr, _ = image_differences(left, right, coverage)
    pos_ecr, _ = image_differences(
        left, right, coverage, required_polarity="positive"
    )
    neg_src, _ = image_differences(
        left, right, coverage, required_polarity="negative"
    )
    pos_anchor_ecr, _ = image_differences(
        left, right, anchor_coverage, required_polarity="positive"
    )
    acc_ci = bootstrap_ci(acc, SEED + 11 * index)
    hfpr_ci = bootstrap_ci(hfpr, SEED + 11 * index + 1)
    ecr_ci = bootstrap_ci(ecr, SEED + 11 * index + 2)
    pos_ecr_ci = bootstrap_ci(pos_ecr, SEED + 11 * index + 3)
    neg_src_ci = bootstrap_ci(neg_src, SEED + 11 * index + 4)
    pos_anchor_ecr_ci = bootstrap_ci(pos_anchor_ecr, SEED + 11 * index + 5)
    return {
        "comparison": comparison.name,
        "n_probes": n_probes,
        "n_images": len(acc),
        "acc_diff": f"{mean(acc):.6f}",
        "acc_ci_low": f"{acc_ci[0]:.6f}",
        "acc_ci_high": f"{acc_ci[1]:.6f}",
        "hFPR_diff": f"{mean(hfpr):.6f}",
        "hFPR_ci_low": f"{hfpr_ci[0]:.6f}",
        "hFPR_ci_high": f"{hfpr_ci[1]:.6f}",
        "ECR_diff": f"{mean(ecr):.6f}",
        "ECR_ci_low": f"{ecr_ci[0]:.6f}",
        "ECR_ci_high": f"{ecr_ci[1]:.6f}",
        "PosECR_diff": f"{mean(pos_ecr):.6f}",
        "PosECR_ci_low": f"{pos_ecr_ci[0]:.6f}",
        "PosECR_ci_high": f"{pos_ecr_ci[1]:.6f}",
        "NegSRC_diff": f"{mean(neg_src):.6f}",
        "NegSRC_ci_low": f"{neg_src_ci[0]:.6f}",
        "NegSRC_ci_high": f"{neg_src_ci[1]:.6f}",
        "PosAnchorECR_diff": f"{mean(pos_anchor_ecr):.6f}",
        "PosAnchorECR_ci_low": f"{pos_anchor_ecr_ci[0]:.6f}",
        "PosAnchorECR_ci_high": f"{pos_anchor_ecr_ci[1]:.6f}",
        "bootstrap_unit": "image_id",
        "bootstrap_draws": N_BOOTSTRAP,
        "seed": SEED + 11 * index,
    }


def summarize_run(name: str, rows: dict[str, dict]) -> dict:
    values = list(rows.values())
    negatives = [row for row in values if row.get("binary_polarity") == "negative"]
    positives = [row for row in values if row.get("binary_polarity") == "positive"]
    return {
        "run": name,
        "n_probes": len(values),
        "n_images": len({str(row.get("image_id")) for row in values}),
        "accuracy": f"{mean([correct(row) for row in values]):.6f}",
        "hFPR": f"{mean([predicts_yes(row) for row in negatives]):.6f}",
        "keep_ratio": f"{mean([float(row.get('prune_keep_ratio', 1.0)) for row in values]):.6f}",
        "PosECR": f"{mean([coverage(row) for row in positives]):.6f}",
        "NegSRC": f"{mean([coverage(row) for row in negatives]):.6f}",
        "PosAnchorECR": f"{mean([anchor_coverage(row) for row in positives]):.6f}",
        "NegAnchorECR": f"{mean([anchor_coverage(row) for row in negatives]):.6f}",
    }


def main() -> None:
    runs = {key: read_scores(path) for key, path in RUNS.items()}
    rows = [summarize(comparison, runs, index) for index, comparison in enumerate(COMPARISONS)]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUT_DIR / "confirmation_image_cluster_stats.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_rows = [summarize_run(name, run) for name, run in runs.items()]
    summary_output = OUT_DIR / "confirmation_method_summary.csv"
    with summary_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    for row in rows:
        print(
            f"{row['comparison']}: accuracy {row['acc_diff']} "
            f"[{row['acc_ci_low']}, {row['acc_ci_high']}], "
            f"hFPR {row['hFPR_diff']} [{row['hFPR_ci_low']}, {row['hFPR_ci_high']}], "
            f"PosECR {row['PosECR_diff']} "
            f"[{row['PosECR_ci_low']}, {row['PosECR_ci_high']}]"
        )
    print(f"Wrote {output}")
    print(f"Wrote {summary_output}")


if __name__ == "__main__":
    main()
