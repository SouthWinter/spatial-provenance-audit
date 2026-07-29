#!/usr/bin/env python3
"""Integrate image and random-mask uncertainty for the Qwen development control."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from statistics import fmean
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs/problem_optimization_audit"
TARGET = (
    ROOT
    / "runs/prune_textocr_hard_full1000/"
    "qwen3_8b_textocr_hard_full1000_target_embed_topk_0p30_targetfix_802816/"
    "probe_scores.jsonl"
)
RANDOM_RUNS = sorted(
    ROOT.glob(
        "runs/prune_textocr_hard_full1000/"
        "qwen3_8b_textocr_hard_full1000_random_0p30*/probe_scores.jsonl"
    )
)
N_BOOTSTRAP = 20_000
SEED = 20260724


def read(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {str(row["sample_id"]): row for row in rows}


def image_id(row: dict[str, Any]) -> str:
    return str(row.get("image_id") or str(row["sample_id"]).split(":")[0])


def correct(row: dict[str, Any]) -> float:
    return float(bool(row["correct"]))


def positive_ecr(row: dict[str, Any]) -> float:
    return float(row["prune_ecr"])


def seed_image_differences(
    target: dict[str, dict[str, Any]],
    controls: list[dict[str, dict[str, Any]]],
    value: Callable[[dict[str, Any]], float],
    *,
    polarity: str | None = None,
) -> tuple[list[str], list[list[float]]]:
    images = sorted({image_id(row) for row in target.values()})
    matrices = []
    for control in controls:
        by_image: dict[str, list[float]] = {key: [] for key in images}
        for sample_id, target_row in target.items():
            if polarity and target_row.get("binary_polarity") != polarity:
                continue
            control_row = control[sample_id]
            by_image[image_id(target_row)].append(value(target_row) - value(control_row))
        matrices.append([fmean(by_image[key]) for key in images if by_image[key]])
    if any(len(row) != len(matrices[0]) for row in matrices):
        raise ValueError("Seed runs do not share the same image clusters")
    return images, matrices


def two_stage_ci(matrix: list[list[float]], seed: int) -> tuple[float, float, float]:
    rng = random.Random(seed)
    n_seeds = len(matrix)
    n_images = len(matrix[0])
    observed = fmean(value for row in matrix for value in row)
    draws = []
    for _ in range(N_BOOTSTRAP):
        sampled_seeds = [rng.randrange(n_seeds) for _ in range(n_seeds)]
        sampled_images = [rng.randrange(n_images) for _ in range(n_images)]
        draws.append(
            fmean(matrix[seed_idx][image_idx] for seed_idx in sampled_seeds for image_idx in sampled_images)
        )
    draws.sort()
    return observed, draws[int(0.025 * N_BOOTSTRAP)], draws[int(0.975 * N_BOOTSTRAP)]


def main() -> None:
    if len(RANDOM_RUNS) != 6:
        raise ValueError(f"Expected six Qwen random runs, found {len(RANDOM_RUNS)}")
    target = read(TARGET)
    controls = [read(path) for path in RANDOM_RUNS]
    specs = (
        ("accuracy", correct, None),
        ("PosECR", positive_ecr, "positive"),
    )
    rows = []
    for index, (metric, value, polarity) in enumerate(specs):
        images, matrix = seed_image_differences(target, controls, value, polarity=polarity)
        estimate, low, high = two_stage_ci(matrix, SEED + index)
        rows.append(
            {
                "comparison": "Qwen Target (30%) - Random (30%), development",
                "metric": metric,
                "n_images": len(images),
                "n_random_seeds": len(matrix),
                "difference": estimate,
                "ci_low": low,
                "ci_high": high,
                "bootstrap": "two-stage seed-and-image",
                "draws": N_BOOTSTRAP,
            }
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "random_seed_integrated_stats.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(path)
    for row in rows:
        print(
            f"{row['metric']}: {row['difference']:+.3f} "
            f"[{row['ci_low']:+.3f},{row['ci_high']:+.3f}]"
        )


if __name__ == "__main__":
    main()
