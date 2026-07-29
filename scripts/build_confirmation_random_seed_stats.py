#!/usr/bin/env python3
"""Integrate random-mask and image uncertainty on the locked confirmation set."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs/textocr_confirmation"
OUT_DIR = ROOT / "runs/problem_optimization_audit/random_seed_confirmation_stats"
TARGET = RUN_ROOT / "qwen3_8b_target_0p30/probe_scores.jsonl"
RANDOM_RUNS = [
    RUN_ROOT / "qwen3_8b_random_0p30/probe_scores.jsonl",
    *(RUN_ROOT / f"qwen3_8b_random_0p30_seed{seed}/probe_scores.jsonl" for seed in (101, 202, 303, 404, 505)),
]
N_BOOTSTRAP = 20_000
BOOTSTRAP_SEED = 20260725


def read(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != 1000:
        raise ValueError(f"Expected 1000 probes in {path}, found {len(rows)}")
    return {str(row["sample_id"]): row for row in rows}


def image_id(row: dict[str, Any]) -> str:
    return str(row.get("image_id") or str(row["sample_id"]).split(":")[0])


def correct(row: dict[str, Any]) -> float:
    return float(bool(row["correct"]))


def ecr(row: dict[str, Any]) -> float:
    return float(row["prune_ecr"])


def h_fpr(rows: dict[str, dict[str, Any]]) -> float:
    values = [
        float(str(row["pred_answer"]).strip().lower() == "yes")
        for row in rows.values()
        if row.get("binary_polarity") == "negative"
    ]
    return fmean(values)


def accuracy(rows: dict[str, dict[str, Any]]) -> float:
    return fmean(correct(row) for row in rows.values())


def positive_ecr(rows: dict[str, dict[str, Any]]) -> float:
    return fmean(ecr(row) for row in rows.values() if row.get("binary_polarity") == "positive")


def seed_image_differences(
    target: dict[str, dict[str, Any]],
    controls: list[dict[str, dict[str, Any]]],
    value: Callable[[dict[str, Any]], float],
    *,
    polarity: str | None = None,
) -> tuple[list[str], np.ndarray]:
    images = sorted({image_id(row) for row in target.values()})
    matrices = []
    for control in controls:
        if set(control) != set(target):
            raise ValueError("Target and random runs do not share identical sample IDs")
        by_image: dict[str, list[float]] = {key: [] for key in images}
        for sample_id, target_row in target.items():
            if polarity and target_row.get("binary_polarity") != polarity:
                continue
            by_image[image_id(target_row)].append(value(target_row) - value(control[sample_id]))
        matrices.append([fmean(by_image[key]) for key in images if by_image[key]])
    matrix = np.asarray(matrices, dtype=float)
    if matrix.shape != (len(controls), len(images)):
        raise ValueError(f"Unexpected seed-image matrix shape: {matrix.shape}")
    return images, matrix


def two_stage_ci(matrix: np.ndarray, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n_seeds, n_images = matrix.shape
    observed = float(matrix.mean())
    draws = np.empty(N_BOOTSTRAP, dtype=float)
    for start in range(0, N_BOOTSTRAP, 500):
        count = min(500, N_BOOTSTRAP - start)
        seed_idx = rng.integers(0, n_seeds, size=(count, n_seeds))
        image_idx = rng.integers(0, n_images, size=(count, n_images))
        sampled = matrix[seed_idx[:, :, None], image_idx[:, None, :]]
        draws[start : start + count] = sampled.mean(axis=(1, 2))
    low, high = np.quantile(draws, [0.025, 0.975])
    return observed, float(low), float(high)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    missing = [path for path in [TARGET, *RANDOM_RUNS] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing confirmation runs:\n" + "\n".join(map(str, missing)))

    target = read(TARGET)
    controls = [read(path) for path in RANDOM_RUNS]
    seed_rows = []
    for path, control in zip(RANDOM_RUNS, controls):
        seed_rows.append(
            {
                "run": path.parent.name,
                "accuracy": accuracy(control),
                "hFPR": h_fpr(control),
                "PosECR": positive_ecr(control),
            }
        )

    summary_rows = []
    for metric in ("accuracy", "hFPR", "PosECR"):
        values = [float(row[metric]) for row in seed_rows]
        summary_rows.append(
            {
                "metric": metric,
                "n_random_seeds": len(values),
                "mean": fmean(values),
                "population_sd": pstdev(values),
                "minimum": min(values),
                "maximum": max(values),
            }
        )

    comparison_rows = []
    for index, (metric, value, polarity) in enumerate(
        (("accuracy", correct, None), ("PosECR", ecr, "positive"))
    ):
        images, matrix = seed_image_differences(target, controls, value, polarity=polarity)
        estimate, low, high = two_stage_ci(matrix, BOOTSTRAP_SEED + index)
        comparison_rows.append(
            {
                "comparison": "Qwen Target (30%) - Random (30%), locked confirmation",
                "metric": metric,
                "n_images": len(images),
                "n_random_seeds": len(controls),
                "difference": estimate,
                "ci_low": low,
                "ci_high": high,
                "bootstrap": "two-stage seed-and-image",
                "draws": N_BOOTSTRAP,
            }
        )

    write_csv(OUT_DIR / "per_seed_metrics.csv", seed_rows)
    write_csv(OUT_DIR / "random_seed_summary.csv", summary_rows)
    write_csv(OUT_DIR / "target_minus_random_two_stage_bootstrap.csv", comparison_rows)

    report = [
        "# Locked-Confirmation Random-Seed Audit",
        "",
        "Six fixed random masks are evaluated on the same 500-image locked confirmation split. "
        "The two-stage percentile bootstrap first resamples random-mask seeds and then image clusters.",
        "",
        "| Random control | Accuracy | hFPR | PosECR |",
        "| --- | ---: | ---: | ---: |",
    ]
    report.extend(
        f"| {row['run']} | {row['accuracy']:.3f} | {row['hFPR']:.3f} | {row['PosECR']:.3f} |"
        for row in seed_rows
    )
    report.extend(
        [
            "",
            "| Target minus Random | Difference (95% CI) | Seeds | Images |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    report.extend(
        f"| {row['metric']} | {row['difference']:+.3f} "
        f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}] | "
        f"{row['n_random_seeds']} | {row['n_images']} |"
        for row in comparison_rows
    )
    report.extend(
        [
            "",
            "The target selector is deterministic; seed resampling quantifies uncertainty from the "
            "matched-budget random-mask control, while image resampling preserves positive/negative pairing.",
            "",
        ]
    )
    (OUT_DIR / "random_seed_confirmation_report.md").write_text("\n".join(report), encoding="utf-8")
    print(OUT_DIR / "random_seed_confirmation_report.md")
    for row in comparison_rows:
        print(
            f"{row['metric']}: {row['difference']:+.3f} "
            f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]"
        )


if __name__ == "__main__":
    main()
