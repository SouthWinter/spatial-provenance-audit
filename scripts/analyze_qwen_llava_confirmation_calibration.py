#!/usr/bin/env python
"""Decompose selector ranking and calibration on locked Qwen/LLaVA confirmation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recap.metrics import roc_auc


@dataclass(frozen=True)
class RunSpec:
    model: str
    method: str
    path: str


FULL_DEVELOPMENT = {
    "Qwen3-VL-8B": (
        "runs/prune_textocr_hard_full1000/"
        "qwen3_8b_textocr_hard_full1000_topk_1p00/probe_scores.jsonl"
    ),
    "LLaVA-1.5-7B": (
        "runs/llava_textocr_hard/"
        "llava15_7b_textocr_hard_full1000_direct/probe_scores.jsonl"
    ),
}

CONFIRMATION_RUNS = [
    RunSpec("Qwen3-VL-8B", "Full", "runs/textocr_confirmation/qwen3_8b_full/probe_scores.jsonl"),
    RunSpec(
        "Qwen3-VL-8B",
        "Target (30%)",
        "runs/textocr_confirmation/qwen3_8b_target_0p30/probe_scores.jsonl",
    ),
    RunSpec(
        "Qwen3-VL-8B",
        "Random (30%)",
        "runs/textocr_confirmation/qwen3_8b_random_0p30/probe_scores.jsonl",
    ),
    RunSpec(
        "Qwen3-VL-8B",
        "Grid (30%)",
        "runs/textocr_confirmation/qwen3_8b_grid_0p30/probe_scores.jsonl",
    ),
    RunSpec(
        "Qwen3-VL-8B",
        "VisionZip (30%)",
        "runs/textocr_confirmation/qwen3_8b_visionzip_0p30/probe_scores.jsonl",
    ),
    RunSpec("LLaVA-1.5-7B", "Full", "runs/textocr_confirmation/llava15_7b_full/probe_scores.jsonl"),
    RunSpec(
        "LLaVA-1.5-7B",
        "Protected (40%)",
        "runs/textocr_confirmation/llava15_7b_protected_0p40/probe_scores.jsonl",
    ),
    RunSpec(
        "LLaVA-1.5-7B",
        "Random (40%)",
        "runs/textocr_confirmation/llava15_7b_random_0p40/probe_scores.jsonl",
    ),
    RunSpec(
        "LLaVA-1.5-7B",
        "Target (40%)",
        "runs/textocr_confirmation/llava15_7b_target_0p40/probe_scores.jsonl",
    ),
    RunSpec(
        "LLaVA-1.5-7B",
        "SCOPE (40%)",
        "runs/textocr_confirmation/llava15_7b_scope_0p40/probe_scores.jsonl",
    ),
    RunSpec(
        "LLaVA-1.5-7B",
        "AnchorPrune (40%)",
        "runs/anchorprune_textocr/confirmation_llava15_anchorprune_0p40/probe_scores.jsonl",
    ),
    RunSpec(
        "LLaVA-1.5-7B",
        "CoIn (40%)",
        "runs/textocr_confirmation/llava15_7b_coin_0p40/probe_scores.jsonl",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/problem_optimization_audit/qwen_llava_calibration_decomposition"),
    )
    return parser.parse_args()


def load_scores(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            rows.append(
                {
                    "sample_id": str(raw["sample_id"]),
                    "target_yes": str(raw["target_answer"]).lower() == "yes",
                    "margin": float(raw["no_loss"]) - float(raw["yes_loss"]),
                }
            )
    if not rows:
        raise ValueError(f"No score rows in {path}")
    return rows


def evaluate(rows: list[dict[str, Any]], threshold: float) -> dict[str, float]:
    predictions = [row["margin"] >= threshold for row in rows]
    labels = [bool(row["target_yes"]) for row in rows]
    negatives = [index for index, label in enumerate(labels) if not label]
    return {
        "accuracy": sum(pred == label for pred, label in zip(predictions, labels)) / len(rows),
        "hfpr": sum(predictions[index] for index in negatives) / len(negatives),
    }


def best_threshold(rows: list[dict[str, Any]]) -> float:
    margins = sorted({float(row["margin"]) for row in rows})
    candidates = [margins[0] - 1.0]
    candidates.extend((left + right) / 2.0 for left, right in zip(margins, margins[1:]))
    candidates.append(margins[-1] + 1.0)
    best_key: tuple[float, float, float] | None = None
    best_value = 0.0
    for threshold in candidates:
        metrics = evaluate(rows, threshold)
        yes_rate = sum(row["margin"] >= threshold for row in rows) / len(rows)
        key = (metrics["accuracy"], -metrics["hfpr"], -abs(yes_rate - 0.5))
        if best_key is None or key > best_key:
            best_key = key
            best_value = threshold
    return float(best_value)


def summarize() -> tuple[list[dict[str, Any]], dict[str, float]]:
    thresholds = {
        model: best_threshold(load_scores(ROOT / path))
        for model, path in FULL_DEVELOPMENT.items()
    }
    summaries: list[dict[str, Any]] = []
    full_aurocs: dict[str, float] = {}
    reference_ids: dict[str, list[str]] = {}

    for spec in CONFIRMATION_RUNS:
        rows = load_scores(ROOT / spec.path)
        sample_ids = [row["sample_id"] for row in rows]
        if spec.model not in reference_ids:
            reference_ids[spec.model] = sample_ids
        elif sample_ids != reference_ids[spec.model]:
            raise ValueError(f"Confirmation sample order mismatch for {spec.model} {spec.method}")

        labels = [1.0 if row["target_yes"] else 0.0 for row in rows]
        margins = [float(row["margin"]) for row in rows]
        auc = roc_auc(labels, margins)
        if spec.method == "Full":
            full_aurocs[spec.model] = auc
        default = evaluate(rows, 0.0)
        shared = evaluate(rows, thresholds[spec.model])
        summaries.append(
            {
                "model": spec.model,
                "method": spec.method,
                "n": len(rows),
                "auroc": auc,
                "default_accuracy": default["accuracy"],
                "default_hfpr": default["hfpr"],
                "full_dev_threshold": thresholds[spec.model],
                "shared_accuracy": shared["accuracy"],
                "shared_hfpr": shared["hfpr"],
                "source": spec.path,
            }
        )

    for row in summaries:
        delta = row["auroc"] - full_aurocs[row["model"]]
        row["delta_auroc_vs_full"] = 0.0 if abs(delta) < 0.0005 else delta
    return summaries, thresholds


def write_outputs(
    output_dir: Path,
    summaries: list[dict[str, Any]],
    thresholds: dict[str, float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "calibration_decomposition.json").write_text(
        json.dumps(
            {"full_development_thresholds": thresholds, "rows": summaries},
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    fields = [
        "model",
        "method",
        "n",
        "auroc",
        "delta_auroc_vs_full",
        "default_accuracy",
        "default_hfpr",
        "full_dev_threshold",
        "shared_accuracy",
        "shared_hfpr",
        "source",
    ]
    with (output_dir / "calibration_decomposition.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)

    lines = [
        "# Qwen/LLaVA Confirmation Calibration Decomposition",
        "",
        "A single threshold is selected from each backbone's independent Full-prefix "
        "development scores and then applied unchanged to every locked-confirmation "
        "selector. AUROC is threshold-free. The primary paper still uses the frozen "
        "zero threshold for these backbones; this analysis is diagnostic.",
        "",
        "| Model | Method | n | AUROC | Delta vs. Full | t=0 Acc. | t=0 hFPR | Full-dev t | Shared Acc. | Shared hFPR |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {model} | {method} | {n} | {auroc:.3f} | {delta:+.3f} | "
            "{default_accuracy:.3f} | {default_hfpr:.3f} | {threshold:.3f} | "
            "{shared_accuracy:.3f} | {shared_hfpr:.3f} |".format(
                model=row["model"],
                method=row["method"],
                n=row["n"],
                auroc=row["auroc"],
                delta=row["delta_auroc_vs_full"],
                default_accuracy=row["default_accuracy"],
                default_hfpr=row["default_hfpr"],
                threshold=row["full_dev_threshold"],
                shared_accuracy=row["shared_accuracy"],
                shared_hfpr=row["shared_hfpr"],
            )
        )
    lines.extend(
        [
            "",
            "Qwen Target matches Full's threshold-free ranking (0.855 versus 0.856), "
            "whereas Random, Grid, and VisionZip are lower. LLaVA Target has higher "
            "AUROC than Full (0.713 versus 0.666), but its shared-threshold hFPR remains "
            "0.442. The LLaVA result therefore contains a substantial calibration shift "
            "without reducing the comparison to calibration alone.",
            "",
        ]
    )
    (output_dir / "calibration_decomposition.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    summaries, thresholds = summarize()
    write_outputs(args.output_dir, summaries, thresholds)
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
