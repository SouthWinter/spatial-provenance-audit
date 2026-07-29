#!/usr/bin/env python
"""Build a compact cross-model batch-prefill efficiency report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ENTRIES = [
    {
        "model": "Qwen3-VL-8B",
        "group": "target_embed_topk",
        "point": "target0p20",
        "summary": "runs/efficiency/qwen3_8b_textocr_hard_batch_prefill_802816_target0p20_b32_limit992_overhead/batch_prefill_summary.json",
        "note": "primary high-efficiency point",
    },
    {
        "model": "Qwen3-VL-8B",
        "group": "target_embed_topk",
        "point": "target0p30",
        "summary": "runs/efficiency/qwen3_8b_textocr_hard_batch_prefill_802816_target0p30_b32_limit992_overhead/batch_prefill_summary.json",
        "note": "accuracy-oriented point",
    },
    {
        "model": "LLaVA-1.5-7B",
        "group": "grid",
        "point": "grid0p40",
        "summary": "runs/efficiency/llava15_7b_textocr_hard_grid0p40_b100/batch_prefill_summary.json",
        "note": "low-hallucination spatial baseline",
    },
    {
        "model": "LLaVA-1.5-7B",
        "group": "embed_topk",
        "point": "embed0p40",
        "summary": "runs/efficiency/llava15_7b_textocr_hard_embed_topk0p40_b100/batch_prefill_summary.json",
        "note": "accuracy-oriented LLaVA point",
    },
    {
        "model": "LLaVA-1.5-7B",
        "group": "embed_protected_topk",
        "point": "protected_embed0p40",
        "quality": "runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_embed_protected_topk_0p40/metrics.json",
        "summary": "runs/efficiency/llava15_7b_textocr_hard_embed_protected_topk0p40_b100/batch_prefill_summary.json",
        "note": "bbox/OCR evidence-protected LLaVA point",
    },
    {
        "model": "InternVL3.5-8B",
        "group": "grid",
        "point": "grid0p50",
        "summary": "runs/efficiency/internvl35_8b_textocr_hard_grid0p50_b50/batch_prefill_summary.json",
        "note": "calibrated accuracy baseline",
    },
    {
        "model": "InternVL3.5-8B",
        "group": "target_embed_topk",
        "point": "target0p50",
        "summary": "runs/efficiency/internvl35_8b_textocr_hard_target_embed_topk0p50_b50/batch_prefill_summary.json",
        "note": "risk-aware low-hFPR point",
    },
    {
        "model": "InternVL3.5-8B",
        "group": "target_embed_soft_evidence_topk",
        "point": "soft_evidence0p50_hfpr",
        "quality": "runs/internvl_textocr_hard/calibrated_test_target_soft_evidence0p50_b0p05_hfprconstr_devthr/metrics.json",
        "summary": "runs/efficiency/internvl35_8b_textocr_hard_target_soft_evidence_topk0p50_b50/batch_prefill_summary.json",
        "note": "risk-constrained balanced point",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-csv", default="runs/efficiency/cross_model_actual_overhead_limit100.csv")
    parser.add_argument("--output-md", default="runs/efficiency/cross_model_batch_prefill_report.md")
    parser.add_argument("--output-csv", default="runs/efficiency/cross_model_batch_prefill_report.csv")
    args = parser.parse_args()

    quality = load_quality(Path(args.quality_csv))
    rows = []
    for entry in ENTRIES:
        summary_path = Path(entry["summary"])
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing batch summary: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        q = quality.get((entry["model"], entry["group"], entry["point"]), {})
        if not q and entry.get("quality"):
            q = load_metrics_quality(Path(entry["quality"]))
        rows.append(build_row(entry, summary, q))

    output_md = Path(args.output_md)
    output_csv = Path(args.output_csv)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(markdown(rows), encoding="utf-8")
    write_csv(output_csv, rows)
    print(f"Wrote {output_md}")
    print(f"Wrote {output_csv}")


def load_quality(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    out: dict[tuple[str, str, str], dict[str, str]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[(row["model"], row["group"], row["point"])] = row
    return out


def load_metrics_quality(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing quality metrics: {path}")
    metrics = json.loads(path.read_text(encoding="utf-8"))
    return {
        "quality_n": str(metrics.get("num_samples", "")),
        "quality_acc": str(metrics.get("direct_accuracy", "")),
        "quality_hFPR": str(metrics.get("direct_hallucination_fpr", "")),
    }


def build_row(entry: dict[str, str], summary: dict[str, Any], quality: dict[str, str]) -> dict[str, Any]:
    args = summary["args"]
    return {
        "model": entry["model"],
        "selector": entry["group"],
        "point": entry["point"],
        "note": entry["note"],
        "quality_n": as_float(quality.get("quality_n")),
        "quality_acc": as_float(quality.get("quality_acc")),
        "quality_hFPR": as_float(quality.get("quality_hFPR")),
        "batch_samples": int(summary["num_samples"]),
        "batch_size": int(args["batch_size"]),
        "keep_ratio": as_float(summary.get("mean_visual_keep_ratio")),
        "token_keep_ratio": as_float(summary.get("mean_token_keep_ratio")),
        "full_samples_per_s": as_float(summary.get("mean_full_samples_per_s")),
        "pruned_samples_per_s": as_float(summary.get("mean_pruned_samples_per_s")),
        "batch_prefill_speedup": as_float(summary.get("mean_speedup")),
        "time_reduction_pct": as_float(summary.get("mean_time_reduction_pct")),
        "full_peak_allocated_gb": as_float(summary.get("mean_full_peak_allocated_gb")),
        "pruned_peak_allocated_gb": as_float(summary.get("mean_pruned_peak_allocated_gb")),
        "peak_allocated_reduction_pct": as_float(summary.get("mean_peak_allocated_reduction_pct")),
        "incremental_peak_allocated_reduction_pct": as_float(
            summary.get("mean_incremental_peak_allocated_reduction_pct")
        ),
        "prune_over_saved_prefill_pct": as_float(summary.get("mean_prune_over_saved_prefill_pct")),
        "selector_over_saved_prefill_pct": as_float(summary.get("mean_selector_over_saved_prefill_pct")),
        "ecr": as_float(summary.get("mean_mean_ecr")),
        "summary_path": str(Path(entry["summary"])),
    }


def markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Cross-model batch prefill efficiency",
        "",
        "Quality columns are from the existing single-sample/quality report; batch columns are from prefill-only batch benchmarks.",
        "",
        "| model | point | Acc | hFPR | B | samples | keep | samples/s full -> pruned | speedup | peak GB full -> pruned | inc. peak red. | overhead/saved | note |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {point} | {acc} | {hfpr} | {batch_size} | {batch_samples} | {keep} | "
            "{full_sps} -> {pruned_sps} | {speedup} | {full_peak} -> {pruned_peak} | {inc_red} | {overhead} | {note} |".format(
                model=row["model"],
                point=row["point"],
                acc=fmt(row["quality_acc"], 3),
                hfpr=fmt(row["quality_hFPR"], 3),
                batch_size=row["batch_size"],
                batch_samples=row["batch_samples"],
                keep=fmt(row["keep_ratio"], 3),
                full_sps=fmt(row["full_samples_per_s"], 2),
                pruned_sps=fmt(row["pruned_samples_per_s"], 2),
                speedup=fmt(row["batch_prefill_speedup"], 2) + "x",
                full_peak=fmt(row["full_peak_allocated_gb"], 1),
                pruned_peak=fmt(row["pruned_peak_allocated_gb"], 1),
                inc_red=fmt(row["incremental_peak_allocated_reduction_pct"], 1) + "%",
                overhead=fmt(row["prune_over_saved_prefill_pct"], 1) + "%",
                note=row["note"],
            )
        )
    lines.extend(
        [
            "",
            "## Output Paths",
            "",
        ]
    )
    for row in rows:
        lines.append(f"- {row['model']} {row['point']}: `{row['summary_path']}`")
    lines.append("")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def fmt(value: float, digits: int) -> str:
    return f"{float(value):.{digits}f}"


if __name__ == "__main__":
    main()
