#!/usr/bin/env python3
"""Aggregate repeated exclusive SCOPE timing after per-process warm-up."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("full", "protected", "scope")
METRICS = (
    "mean_vision_ms",
    "mean_prune_overhead_ms",
    "mean_language_ms",
    "mean_forward_ms",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=ROOT / "runs" / "efficiency" / "scope_repeated_exclusive",
    )
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    runs_root = args.runs_root.resolve()

    detail = []
    for method in METHODS:
        for rep in range(1, 4):
            path = runs_root / f"{method}_rep{rep}" / "prune_traces.jsonl"
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            kept = rows[args.warmup :]
            if not kept:
                raise ValueError(f"No timed rows remain after warm-up in {path}")
            detail.append(
                {
                    "method": method,
                    "rep": rep,
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
    write_csv(runs_root / "scope_repeated_timing_runs.csv", detail)
    write_csv(runs_root / "scope_repeated_timing_summary.csv", summary)
    (runs_root / "scope_repeated_timing_report.md").write_text(
        markdown(detail, summary, args.warmup), encoding="utf-8"
    )
    print(f"Wrote repeated timing report to {runs_root}")


def summarize(detail: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for method in METHODS:
        group = [row for row in detail if row["method"] == method]
        row: dict[str, Any] = {
            "method": method,
            "repetitions": len(group),
            "timed_probes_per_rep": group[0]["timed_probes"],
            "keep_ratio": statistics.mean(row["keep_ratio"] for row in group),
        }
        for metric in METRICS:
            values = [item[metric] for item in group]
            row[f"{metric}_mean"] = statistics.mean(values)
            row[f"{metric}_std"] = statistics.stdev(values)
        rows.append(row)
    full_ms = rows[0]["mean_forward_ms_mean"]
    for row in rows:
        row["speedup_vs_full"] = full_ms / row["mean_forward_ms_mean"]
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown(detail: list[dict[str, Any]], summary: list[dict[str, Any]], warmup: int) -> str:
    lines = [
        "# Repeated Exclusive LLaVA Timing",
        "",
        "Three fresh-process repetitions use the same first 100 image-disjoint TextOCR-Hard confirmation probes, "
        "one A800 GPU, float16, eager attention, and rotated method order. "
        f"The first {warmup} probes of each process are discarded as warm-up.",
        "",
        "## Summary",
        "",
        "| method | keep | vision ms | selector ms | LLM ms | total ms | speedup |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['method']} | {row['keep_ratio']:.3f} | "
            f"{pm(row['mean_vision_ms_mean'], row['mean_vision_ms_std'])} | "
            f"{pm(row['mean_prune_overhead_ms_mean'], row['mean_prune_overhead_ms_std'])} | "
            f"{pm(row['mean_language_ms_mean'], row['mean_language_ms_std'])} | "
            f"{pm(row['mean_forward_ms_mean'], row['mean_forward_ms_std'])} | "
            f"{row['speedup_vs_full']:.2f}x |"
        )
    lines.extend(
        [
            "",
            "Values are mean +/- sample standard deviation across three repetitions; each repetition averages 95 post-warm-up probes.",
            "",
            "## Per-Run Means",
            "",
            "| method | rep | total ms | selector ms | LLM ms |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in detail:
        lines.append(
            f"| {row['method']} | {row['rep']} | {row['mean_forward_ms']:.2f} | "
            f"{row['mean_prune_overhead_ms']:.2f} | {row['mean_language_ms']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- Protected shortens the LLM prefix enough to overcome its target/evidence selection overhead.",
            "- SCOPE also shortens LLM time, but its greedy coverage selector is measured inside the online path and offsets that saving.",
            "- These are single-sample prefill/likelihood timings, not end-to-end long-generation speedups.",
            "",
        ]
    )
    return "\n".join(lines)


def pm(mean: float, std: float) -> str:
    return f"{mean:.1f} +/- {std:.1f}"


if __name__ == "__main__":
    main()
