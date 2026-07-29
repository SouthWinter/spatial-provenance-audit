#!/usr/bin/env python3
"""Aggregate repeated exclusive Full, Target, and AnchorPrune timing."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("full", "target", "anchorprune")
METRICS = (
    "mean_vision_ms",
    "mean_prune_overhead_ms",
    "mean_language_ms",
    "mean_forward_ms",
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(detail: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for method in METHODS:
        group = [row for row in detail if row["method"] == method]
        summary: dict[str, Any] = {
            "method": method,
            "repetitions": len(group),
            "timed_probes_per_rep": group[0]["timed_probes"],
            "keep_ratio": statistics.mean(row["keep_ratio"] for row in group),
        }
        for metric in METRICS:
            values = [row[metric] for row in group]
            summary[f"{metric}_mean"] = statistics.mean(values)
            summary[f"{metric}_std"] = statistics.stdev(values)
        rows.append(summary)
    full_ms = rows[0]["mean_forward_ms_mean"]
    for row in rows:
        row["speedup_vs_full"] = full_ms / row["mean_forward_ms_mean"]
    return rows


def plus_minus(mean: float, std: float) -> str:
    return f"{mean:.1f} +/- {std:.1f}"


def markdown(summary: list[dict[str, Any]], warmup: int) -> str:
    lines = [
        "# Repeated Exclusive AnchorPrune Timing",
        "",
        "Three fresh-process repetitions use the same first 100 image-disjoint TextOCR-Hard confirmation probes, one A800 GPU, float16, eager attention, and rotated method order. "
        f"The first {warmup} probes of each process are discarded as warm-up. Selector cost is included online.",
        "",
        "| Method | Keep | Vision ms | Select/materialize ms | LLM ms | Total ms | Speedup |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['method']} | {row['keep_ratio']:.3f} | "
            f"{plus_minus(row['mean_vision_ms_mean'], row['mean_vision_ms_std'])} | "
            f"{plus_minus(row['mean_prune_overhead_ms_mean'], row['mean_prune_overhead_ms_std'])} | "
            f"{plus_minus(row['mean_language_ms_mean'], row['mean_language_ms_std'])} | "
            f"{plus_minus(row['mean_forward_ms_mean'], row['mean_forward_ms_std'])} | "
            f"{row['speedup_vs_full']:.2f}x |"
        )
    lines.extend(
        [
            "",
            "Values are mean +/- sample standard deviation across three repetitions; each repetition averages 95 post-warm-up probes. These are single-sample likelihood/prefill timings, not end-to-end long-generation speedups.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=ROOT / "runs/efficiency/anchorprune_repeated_exclusive")
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    runs_root = args.runs_root.resolve()

    detail = []
    for method in METHODS:
        for repetition in range(1, 4):
            path = runs_root / f"{method}_rep{repetition}" / "prune_traces.jsonl"
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            kept = rows[args.warmup :]
            if not kept:
                raise ValueError(f"No timed rows remain after warm-up in {path}")
            detail.append(
                {
                    "method": method,
                    "repetition": repetition,
                    "warmup_discarded": args.warmup,
                    "timed_probes": len(kept),
                    "keep_ratio": statistics.mean(float(row["effective_keep_ratio"]) for row in kept),
                    **{
                        metric: statistics.mean(float(row[metric]) for row in kept)
                        for metric in METRICS
                    },
                }
            )
    summary = summarize(detail)
    write_csv(runs_root / "anchorprune_repeated_timing_runs.csv", detail)
    write_csv(runs_root / "anchorprune_repeated_timing_summary.csv", summary)
    (runs_root / "anchorprune_repeated_timing_report.md").write_text(
        markdown(summary, args.warmup), encoding="utf-8"
    )
    print(f"Wrote repeated AnchorPrune timing report to {runs_root}")


if __name__ == "__main__":
    main()
