#!/usr/bin/env python
"""Build a detector-aware TextOCR-Hard efficiency decomposition table."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EFF_DIR = ROOT / "runs" / "efficiency"
ACTUAL_CSV = EFF_DIR / "cross_model_actual_overhead_limit100.csv"
REAL_EFF_CSV = EFF_DIR / "textocr_hard_real_efficiency_report.csv"
EASYOCR_JSON = ROOT / "data" / "textocr_easyocr" / "easyocr_input_summary.json"
OUT_CSV = EFF_DIR / "textocr_efficiency_decomposition.csv"
OUT_MD = EFF_DIR / "textocr_efficiency_decomposition.md"


POINTS = [
    {
        "model": "Qwen3-VL-8B",
        "group": "target_embed_topk",
        "point": "target0p20",
        "label": "Qwen Target 20%",
        "detector": "none",
        "note": "box-free efficiency point",
    },
    {
        "model": "Qwen3-VL-8B",
        "group": "target_embed_topk",
        "point": "target0p30",
        "label": "Qwen Target 30%",
        "detector": "none",
        "note": "box-free quality point",
    },
    {
        "model": "LLaVA-1.5-7B",
        "group": "embed_protected_topk",
        "point": "protected_embed0p40",
        "label": "LLaVA Protected 40%",
        "detector": "optional_easyocr",
        "note": "box-aware; detector cost shown separately",
    },
    {
        "model": "InternVL3.5-8B",
        "group": "target_embed_soft_evidence_topk",
        "point": "soft_evidence0p50_hfpr",
        "label": "InternVL Soft evidence 50%",
        "detector": "optional_easyocr",
        "note": "box-aware; detector cost shown separately",
    },
]


def main() -> None:
    actual_rows = read_csv(ACTUAL_CSV)
    real_rows = read_csv(REAL_EFF_CSV)
    easyocr = json.loads(EASYOCR_JSON.read_text(encoding="utf-8"))
    mean_detector_ms = float(easyocr["mean_detector_ms_per_image"])

    actual_by_key = {(row["model"], row["group"], row["point"]): row for row in actual_rows}
    real_by_key = {(row["model"], row["selector"], row["point"]): row for row in real_rows}

    rows = []
    for spec in POINTS:
        model = spec["model"]
        group = spec["group"]
        point = spec["point"]
        full = actual_by_key[(model, group, "full1p00")]
        pruned = actual_by_key[(model, group, point)]
        real = real_by_key.get((model, group, point), {})
        detector_ms = mean_detector_ms if spec["detector"] == "optional_easyocr" else 0.0
        full_forward_ms = fnum(full["forward_ms"])
        pruned_forward_ms = fnum(pruned["forward_ms"])
        detector_inclusive_ms = pruned_forward_ms + detector_ms
        rows.append(
            {
                "label": spec["label"],
                "model": model,
                "point": point,
                "quality_acc": fmt(real.get("quality_acc", pruned.get("quality_acc")), 3),
                "quality_hFPR": fmt(real.get("quality_hFPR", pruned.get("quality_hFPR")), 3),
                "keep_ratio": fmt(pruned["keep_ratio"], 3),
                "full_vision_ms": fmt(full["vision_ms"], 1),
                "full_llm_prefill_ms": fmt(full["language_ms"], 1),
                "full_ttft_ms": fmt(full_forward_ms, 1),
                "pruned_vision_ms": fmt(pruned["vision_ms"], 1),
                "pruned_llm_prefill_ms": fmt(pruned["language_ms"], 1),
                "selector_overhead_ms": fmt(pruned["prune_overhead_ms"], 1),
                "pruned_ttft_ms": fmt(pruned_forward_ms, 1),
                "ttft_speedup_no_detector": fmt(speedup(full_forward_ms, pruned_forward_ms), 2),
                "detector_ms": fmt(detector_ms, 1),
                "detector_inclusive_ttft_ms": fmt(detector_inclusive_ms, 1),
                "ttft_speedup_with_detector": fmt(speedup(full_forward_ms, detector_inclusive_ms), 2),
                "batch_prefill_speedup": fmt(real.get("batch_prefill_speedup"), 2),
                "incremental_peak_allocated_reduction_pct": fmt(
                    real.get("incremental_peak_allocated_reduction_pct"), 1
                ),
                "note": spec["note"],
            }
        )

    write_csv(OUT_CSV, rows)
    OUT_MD.write_text(markdown(rows, easyocr), encoding="utf-8")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown(rows: list[dict[str, Any]], easyocr: dict[str, Any]) -> str:
    lines = [
        "# TextOCR-Hard Efficiency Decomposition",
        "",
        "This table separates the parts of the measured single-sample prefill path. "
        "The no-detector TTFT column uses the actual pruning path with selector scoring and mask materialization. "
        "The detector-inclusive column adds EasyOCR latency only for box-aware settings, representing a deployment "
        "case where OCR boxes are not already available.",
        "",
        f"EasyOCR mean detector latency: {float(easyocr['mean_detector_ms_per_image']):.1f} ms/image "
        f"(median {float(easyocr['median_detector_ms_per_image']):.1f} ms/image).",
        "",
        "| point | Acc. | hFPR | keep | full TTFT | pruned TTFT | no-det. speedup | detector | det.-incl. TTFT | det.-incl. speedup | batch prefill | inc. peak red. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {quality_acc} | {quality_hFPR} | {keep_ratio} | {full_ttft_ms} | "
            "{pruned_ttft_ms} | {ttft_speedup_no_detector}x | {detector_ms} | "
            "{detector_inclusive_ttft_ms} | {ttft_speedup_with_detector}x | "
            "{batch_prefill_speedup}x | {incremental_peak_allocated_reduction_pct}% |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- Box-free Qwen pruning gives real single-sample TTFT gains because the vision encoder is unchanged but LLM prefill is much shorter.",
            "- Box-aware rows should be interpreted in two deployment regimes: if OCR/layout boxes are already available, use the no-detector TTFT; if EasyOCR must be run online per image, the detector-inclusive latency can erase single-sample speedups.",
            "- The batch-prefill column remains useful for throughput because it measures the shortened visual-prefix LLM path directly and is not a token-count proxy.",
            "",
            "## Source Files",
            "",
            f"- `{ACTUAL_CSV.relative_to(ROOT)}`",
            f"- `{REAL_EFF_CSV.relative_to(ROOT)}`",
            f"- `{EASYOCR_JSON.relative_to(ROOT)}`",
            "",
        ]
    )
    return "\n".join(lines)


def fnum(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def fmt(value: Any, digits: int) -> str:
    if value is None or value == "":
        return "0." + "0" * digits
    return f"{float(value):.{digits}f}"


def speedup(full_ms: float, pruned_ms: float) -> float:
    return full_ms / pruned_ms if pruned_ms > 0.0 else 0.0


if __name__ == "__main__":
    main()
