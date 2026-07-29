#!/usr/bin/env python
"""Summarize completed pruning runs into CSV and Markdown tables."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="runs/prune_main")
    parser.add_argument("--csv-out", default="runs/prune_main/prune_main_summary.csv")
    parser.add_argument("--md-out", default="runs/prune_main/prune_main_summary.md")
    parser.add_argument("--include-smoke", action="store_true", help="Include smoke/debug runs in the summary.")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    rows = []
    for metrics_path in sorted(runs_dir.glob("*/metrics.json")):
        run_dir = metrics_path.parent
        if run_dir.name.startswith("smoke_") and not args.include_smoke:
            continue
        trace_path = run_dir / "prune_traces.jsonl"
        if not trace_path.exists():
            continue
        metrics = read_json(metrics_path)
        traces = read_jsonl(trace_path)
        if not traces:
            continue
        row = {
            "group": group_for_run(run_dir.name),
            "run": run_dir.name,
            "acc": float(metrics.get("direct_accuracy", 0.0)),
            "hFPR": float(metrics.get("direct_hallucination_fpr", 0.0)),
            "keep": mean_ratio(traces),
            "target_keep": mean_value(traces, "sample_budget_ratio", fallback_key="target_keep_ratio"),
            "ECR": mean_value(traces, "ecr"),
            "CenterR": mean_value(traces, "evidence_center_recall"),
            "PatchR": mean_value(traces, "evidence_patch_recall"),
            "lang_ms": mean_value(traces, "mean_language_ms", fallback_key="language_ms"),
            "selector": metrics.get("prune_selector", traces[0].get("selector", "")),
            "score_source": metrics.get("prune_score_source", traces[0].get("score_source", "")),
            "budget_mode": metrics.get("prune_budget_mode", traces[0].get("budget_mode", "fixed")),
        }
        rows.append(row)

    rows.sort(key=lambda r: (str(r["group"]), str(r["run"])))
    write_csv(Path(args.csv_out), rows)
    write_markdown(Path(args.md_out), rows)
    print(f"Wrote {len(rows)} completed runs to {args.csv_out} and {args.md_out}")


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def group_for_run(run: str) -> str:
    if "_textocr_" in run:
        return "textocr_8b" if run.startswith("qwen3_8b") else "textocr_2b"
    if "_gsr_" in run:
        return "gsr_8b" if run.startswith("qwen3_8b") else "gsr_2b"
    return "other"


def mean_ratio(rows: list[dict[str, Any]]) -> float:
    values = []
    for row in rows:
        kept = row.get("kept_visual_tokens")
        total = row.get("full_visual_tokens")
        if isinstance(kept, (int, float)) and isinstance(total, (int, float)) and total:
            values.append(float(kept) / float(total))
    return sum(values) / len(values) if values else 0.0


def mean_value(rows: list[dict[str, Any]], key: str, *, fallback_key: str | None = None) -> float:
    values = []
    for row in rows:
        value = row.get(key)
        if value is None and fallback_key is not None:
            value = row.get(fallback_key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return sum(values) / len(values) if values else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group",
        "run",
        "acc",
        "hFPR",
        "keep",
        "target_keep",
        "ECR",
        "CenterR",
        "PatchR",
        "lang_ms",
        "selector",
        "score_source",
        "budget_mode",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["group"])].append(row)

    lines = ["# Pruning main summary", "", f"Total completed runs: {len(rows)}", ""]
    for group in sorted(grouped):
        group_rows = grouped[group]
        lines.extend(
            [
                f"## {group} ({len(group_rows)})",
                "",
                "| run | acc | hFPR | keep | ECR | CenterR | PatchR | lang_ms |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in group_rows:
            lines.append(
                f"| {row['run']} | {row['acc']:.4f} | {row['hFPR']:.4f} | "
                f"{row['keep']:.4f} | {row['ECR']:.4f} | {row['CenterR']:.4f} | "
                f"{row['PatchR']:.4f} | {row['lang_ms']:.2f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
