#!/usr/bin/env python
"""Build end-to-end length-sensitivity efficiency audit from cached timings."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "problem_optimization_audit" / "end_to_end_efficiency_length"
TEXTOCR_DECOMP_CSV = ROOT / "runs" / "efficiency" / "textocr_efficiency_decomposition.csv"
QWEN_DECODE32_JSON = (
    ROOT
    / "runs"
    / "prune_main"
    / "efficiency_qwen3_8b_highres_ecr_policy_decode32_limit50_v1"
    / "efficiency_summary.json"
)
QWEN_PREFILL_JSON = (
    ROOT
    / "runs"
    / "prune_main"
    / "efficiency_qwen3_8b_highres_ecr_policy_limit100_v1"
    / "efficiency_summary.json"
)
TOKENS = [0, 1, 4, 16, 32, 64, 128]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    decomp = read_csv(TEXTOCR_DECOMP_CSV)
    qwen_decode32 = read_json(QWEN_DECODE32_JSON)
    qwen_prefill = read_json(QWEN_PREFILL_JSON)
    decode_ms = measured_decode_ms_per_token(qwen_decode32)

    measured_rows = build_measured_decode_rows(qwen_decode32, qwen_prefill)
    sweep_rows = build_textocr_length_sweep(decomp, decode_ms)
    key_rows = build_key_rows(sweep_rows)

    write_csv(OUT_DIR / "measured_qwen_decode32_summary.csv", measured_rows)
    write_csv(OUT_DIR / "textocr_length_sensitivity_summary.csv", sweep_rows)
    write_csv(OUT_DIR / "textocr_length_sensitivity_key_points.csv", key_rows)
    (OUT_DIR / "end_to_end_efficiency_length_report.md").write_text(
        markdown(measured_rows, key_rows, decode_ms), encoding="utf-8"
    )
    print(f"Wrote {OUT_DIR / 'end_to_end_efficiency_length_report.md'}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def measured_decode_ms_per_token(summary: dict[str, Any]) -> float:
    full = float(summary.get("mean_full_decode_ms_per_token", 0.0) or 0.0)
    pruned = float(summary.get("mean_pruned_decode_ms_per_token", 0.0) or 0.0)
    return (full + pruned) / 2.0


def build_measured_decode_rows(
    decode_summary: dict[str, Any],
    prefill_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "run": "qwen_highres_policy_prefill_only",
            "dataset": "GSR-Bench high-resolution policy subset",
            "num_samples": prefill_summary["num_samples"],
            "decode_steps": prefill_summary["args"]["decode_steps"],
            "mean_visual_keep_ratio": fmt(prefill_summary["mean_visual_keep_ratio"]),
            "mean_ECR": fmt(prefill_summary["mean_ecr"]),
            "mean_full_TTFT_ms": fmt(prefill_summary["mean_end_to_end_full_est_ms"]),
            "mean_pruned_TTFT_ms": fmt(prefill_summary["mean_end_to_end_pruned_est_ms"]),
            "mean_decode_full_ms_total": fmt(prefill_summary["mean_full_decode_ms_total"]),
            "mean_decode_pruned_ms_total": fmt(prefill_summary["mean_pruned_decode_ms_total"]),
            "mean_TTFT_speedup": fmt(prefill_summary["mean_end_to_end_est_speedup"]),
            "mean_generation_speedup": fmt(prefill_summary["mean_generation_est_speedup"]),
            "source": str(QWEN_PREFILL_JSON.relative_to(ROOT)),
        },
        {
            "run": "qwen_highres_policy_decode32",
            "dataset": "GSR-Bench high-resolution policy subset",
            "num_samples": decode_summary["num_samples"],
            "decode_steps": decode_summary["args"]["decode_steps"],
            "mean_visual_keep_ratio": fmt(decode_summary["mean_visual_keep_ratio"]),
            "mean_ECR": fmt(decode_summary["mean_ecr"]),
            "mean_full_TTFT_ms": fmt(decode_summary["mean_end_to_end_full_est_ms"]),
            "mean_pruned_TTFT_ms": fmt(decode_summary["mean_end_to_end_pruned_est_ms"]),
            "mean_decode_full_ms_total": fmt(decode_summary["mean_full_decode_ms_total"]),
            "mean_decode_pruned_ms_total": fmt(decode_summary["mean_pruned_decode_ms_total"]),
            "mean_TTFT_speedup": fmt(decode_summary["mean_end_to_end_est_speedup"]),
            "mean_generation_speedup": fmt(decode_summary["mean_generation_est_speedup"]),
            "source": str(QWEN_DECODE32_JSON.relative_to(ROOT)),
        },
    ]


def build_textocr_length_sweep(rows: list[dict[str, str]], decode_ms: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        detector_ms = to_float(row.get("detector_ms", 0.0))
        modes = [("no_detector", to_float(row["pruned_ttft_ms"]))]
        if detector_ms > 0:
            modes.append(("online_detector", to_float(row["detector_inclusive_ttft_ms"])))
        for mode, pruned_base in modes:
            for tokens in TOKENS:
                full_base = to_float(row["full_ttft_ms"])
                full_total = full_base + tokens * decode_ms
                pruned_total = pruned_base + tokens * decode_ms
                speedup = full_total / pruned_total if pruned_total > 0 else 0.0
                out.append(
                    {
                        "label": row["label"],
                        "model": row["model"],
                        "point": row["point"],
                        "detector_mode": mode,
                        "generated_tokens": tokens,
                        "decode_ms_per_token_assumed": fmt(decode_ms),
                        "full_TTFT_ms": row["full_ttft_ms"],
                        "pruned_base_TTFT_ms": fmt(pruned_base),
                        "total_full_ms": fmt(full_total),
                        "total_pruned_ms": fmt(pruned_total),
                        "speedup": fmt(speedup),
                        "latency_reduction_pct": fmt(100.0 * (1.0 - pruned_total / full_total))
                        if full_total > 0
                        else "0.000",
                        "assumption": "equal per-token decode latency from measured Qwen decode32; conservative for pruning gains",
                    }
                )
    return out


def build_key_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep_tokens = {0, 16, 32, 128}
    out = []
    for row in rows:
        if int(row["generated_tokens"]) in keep_tokens:
            out.append(row)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown(measured_rows: list[dict[str, Any]], key_rows: list[dict[str, Any]], decode_ms: float) -> str:
    lines = [
        "# End-to-End Efficiency Length Audit",
        "",
        "This audit separates shortened-prefix TTFT/prefill gains from full generation latency.",
        "",
        f"The length sweep uses {fmt(decode_ms)} ms/token, measured as the mean of full and pruned Qwen decode32 per-token latency. The assumption intentionally gives no decode-stage advantage to pruning, so it is a conservative estimate of how generation length dilutes TTFT gains.",
        "",
        "## Measured Qwen Decode32 Boundary",
        "",
        md_table(measured_rows),
        "",
        "## TextOCR Main-Point Length Sensitivity",
        "",
        md_table(key_rows),
        "",
        "## Interpretation",
        "",
        "- Batch-prefill speedups and memory reductions remain valid system measurements.",
        "- Single-sample TTFT speedups are smaller because the vision encoder still runs.",
        "- Full generation speedups shrink as generated tokens increase because decode latency is mostly unchanged after prefill.",
        "- Online OCR detection must be counted separately for box-aware policies.",
    ]
    return "\n".join(lines) + "\n"


def md_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    cols = list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(col, "")).replace("|", "\\|") for col in cols) + " |")
    return "\n".join(out)


def to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def fmt(value: Any) -> str:
    try:
        numeric = float(value)
        if abs(numeric) < 0.0005:
            numeric = 0.0
        return f"{numeric:.3f}"
    except Exception:
        return str(value)


if __name__ == "__main__":
    main()
