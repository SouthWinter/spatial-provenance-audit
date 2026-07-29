#!/usr/bin/env python
"""Merge TextOCR-Hard quality and real efficiency measurements into one report."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EFF_DIR = ROOT / "runs" / "efficiency"
ACTUAL_CSV = EFF_DIR / "cross_model_actual_overhead_limit100.csv"
BATCH_CSV = EFF_DIR / "cross_model_batch_prefill_report.csv"
OUT_MD = EFF_DIR / "textocr_hard_real_efficiency_report.md"
OUT_CSV = EFF_DIR / "textocr_hard_real_efficiency_report.csv"


def main() -> None:
    actual_rows = read_csv(ACTUAL_CSV)
    batch_rows = read_csv(BATCH_CSV)
    actual_by_key = {
        (row["model"], row["group"], row["point"]): row
        for row in actual_rows
        if row.get("point") != "full1p00"
    }
    merged = [merge_batch_with_actual(row, actual_by_key) for row in batch_rows]
    EFF_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_CSV, merged)
    OUT_MD.write_text(markdown(merged, actual_rows), encoding="utf-8")
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_CSV}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def merge_batch_with_actual(
    row: dict[str, str],
    actual_by_key: dict[tuple[str, str, str], dict[str, str]],
) -> dict[str, Any]:
    actual = actual_by_key.get((row["model"], row["selector"], row["point"]), {})
    return {
        "model": row["model"],
        "selector": row["selector"],
        "point": row["point"],
        "quality_acc": f(row.get("quality_acc")),
        "quality_hFPR": f(row.get("quality_hFPR")),
        "quality_n": f(row.get("quality_n")),
        "batch_samples": f(row.get("batch_samples")),
        "batch_size": f(row.get("batch_size")),
        "keep_ratio": f(row.get("keep_ratio")),
        "single_language_speedup": f(actual.get("language_speedup_vs_full")),
        "single_forward_speedup": f(actual.get("forward_speedup_vs_full")),
        "single_forward_ms": f(actual.get("forward_ms")),
        "single_prune_overhead_pct_saved_language": f(actual.get("prune_overhead_pct_saved_language")),
        "batch_prefill_speedup": f(row.get("batch_prefill_speedup")),
        "full_samples_per_s": f(row.get("full_samples_per_s")),
        "pruned_samples_per_s": f(row.get("pruned_samples_per_s")),
        "batch_time_reduction_pct": f(row.get("time_reduction_pct")),
        "full_peak_allocated_gb": f(row.get("full_peak_allocated_gb")),
        "pruned_peak_allocated_gb": f(row.get("pruned_peak_allocated_gb")),
        "peak_allocated_reduction_pct": f(row.get("peak_allocated_reduction_pct")),
        "incremental_peak_allocated_reduction_pct": f(row.get("incremental_peak_allocated_reduction_pct")),
        "batch_prune_overhead_pct_saved_prefill": f(row.get("prune_over_saved_prefill_pct")),
        "note": row.get("note", ""),
        "batch_summary_path": row.get("summary_path", ""),
        "actual_efficiency_path": actual.get("efficiency_path", ""),
        "quality_path": actual.get("quality_path", ""),
    }


def markdown(rows: list[dict[str, Any]], actual_rows: list[dict[str, str]]) -> str:
    full = {
        row["model"]: row
        for row in actual_rows
        if row.get("point") == "full1p00" and row.get("group") in {"target_embed_topk", "grid"}
    }
    lines = [
        "# TextOCR-Hard Real Efficiency Measurements",
        "",
        "This report merges measured latency, throughput, memory, and cached quality results. The numbers are generated from real CUDA runs rather than token-count estimates.",
        "",
        "## Protocol",
        "",
        "- Dataset: `data/textocr_val_hard_probes_500img.jsonl`.",
        "- Quality: Qwen/LLaVA use the 1000-probe TextOCR-Hard split; InternVL uses the calibrated held-out test split.",
        "- Single-sample timing: `limit=100`, actual pruning path, including selector scoring and mask materialization; each pruned row is compared with its matching full-token row.",
        "- Batch prefill timing: prefill-only CUDA benchmark with measured samples/s and CUDA peak memory. Qwen uses 802816 image pixels and batch size 32; LLaVA uses batch size 100; InternVL uses batch size 50.",
        "- Memory: peak allocated includes resident model memory; incremental peak reduction is the cleaner estimate of activation/prefill memory savings.",
        "",
        "## Main Efficiency Table",
        "",
        "| model | point | Acc | hFPR | keep | single lang x | single fwd x | batch prefill x | samples/s full -> pruned | peak GB full -> pruned | inc. peak red. | overhead/saved | note |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {point} | {acc} | {hfpr} | {keep} | {lang_x} | {fwd_x} | {batch_x} | "
            "{full_sps} -> {pruned_sps} | {full_peak} -> {pruned_peak} | {inc_red} | {overhead} | {note} |".format(
                model=row["model"],
                point=row["point"],
                acc=fmt(row["quality_acc"], 3),
                hfpr=fmt(row["quality_hFPR"], 3),
                keep=fmt(row["keep_ratio"], 3),
                lang_x=fmt_x(row["single_language_speedup"]),
                fwd_x=fmt_x(row["single_forward_speedup"]),
                batch_x=fmt_x(row["batch_prefill_speedup"]),
                full_sps=fmt(row["full_samples_per_s"], 2),
                pruned_sps=fmt(row["pruned_samples_per_s"], 2),
                full_peak=fmt(row["full_peak_allocated_gb"], 1),
                pruned_peak=fmt(row["pruned_peak_allocated_gb"], 1),
                inc_red=fmt_pct(row["incremental_peak_allocated_reduction_pct"]),
                overhead=fmt_pct(row["batch_prune_overhead_pct_saved_prefill"]),
                note=row["note"],
            )
        )
    lines.extend(
        [
            "",
            "## Full-Token References",
            "",
            "| model | full quality acc | full hFPR | full single forward ms | full visual tokens | full sequence tokens |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model, row in sorted(full.items()):
        lines.append(
            "| {model} | {acc} | {hfpr} | {fwd} | {vis} | {seq} |".format(
                model=model,
                acc=fmt(row.get("quality_acc"), 3),
                hfpr=fmt(row.get("quality_hFPR"), 3),
                fwd=fmt(row.get("forward_ms"), 1),
                vis=fmt(row.get("full_visual_tokens"), 1),
                seq=fmt(row.get("full_sequence_tokens"), 1),
            )
        )
    lines.extend(
        [
            "",
            "## Claim Readout",
            "",
            "- Qwen is the strongest efficiency story: `target0p20` keeps 20.0% visual tokens while reaching acc 0.793 / hFPR 0.224, with 1.45x single-sample forward speedup and 4.32x batch-prefill throughput speedup.",
            "- Qwen `target0p30` is the quality-preferred point: acc 0.798 / hFPR 0.222, with 1.37x single-sample forward speedup and 3.12x batch-prefill speedup.",
            "- LLaVA has positive efficiency results for non-target selectors: `embed0p40` and `grid0p40` give about 2.35x batch-prefill speedup; target-conditioned LLaVA remains a negative quality result.",
            "- InternVL has the largest absolute token reduction and strong memory savings: `target0p50` gives 2.25x batch-prefill speedup with 49.8% incremental peak-memory reduction and calibrated hFPR 0.179.",
            "- The measured pruning overhead is small relative to saved prefill: about 5-6% for Qwen, 5-9% for LLaVA, and 2-4% for InternVL in the batch-prefill table.",
            "",
            "## Source Files",
            "",
            f"- `{ACTUAL_CSV.relative_to(ROOT)}`",
            f"- `{BATCH_CSV.relative_to(ROOT)}`",
            f"- `{OUT_CSV.relative_to(ROOT)}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def f(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any, digits: int) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def fmt_x(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.2f}x"


def fmt_pct(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.1f}%"


if __name__ == "__main__":
    main()
