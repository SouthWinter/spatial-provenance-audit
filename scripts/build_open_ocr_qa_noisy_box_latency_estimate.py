#!/usr/bin/env python
"""Estimate detector-aware latency for open-QA noisy-box fallback policies.

The policy scores are cached from open OCR/document QA runs. This script does
not rerun the MLLM. It anchors latency to measured Qwen TextOCR-Hard TTFT
endpoints and adds measured EasyOCR latency only for policies that require
online evidence boxes.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "problem_optimization_audit" / "open_ocr_qa_noisy_box_latency"

FALLBACK_KEY_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_noisy_box_fallback"
    / "noisy_box_fallback_key_summary.csv"
)
EFF_DECOMP_CSV = ROOT / "runs" / "efficiency" / "textocr_efficiency_decomposition.csv"
EASYOCR_SUMMARY_JSON = ROOT / "data" / "textocr_easyocr" / "easyocr_input_summary.json"


def main() -> None:
    fallback_rows = read_csv(FALLBACK_KEY_CSV)
    eff_rows = read_csv(EFF_DECOMP_CSV)
    easyocr = read_json(EASYOCR_SUMMARY_JSON)

    qwen30 = next(row for row in eff_rows if row["label"] == "Qwen Target 30%")
    full_ttft_ms = float(qwen30["full_ttft_ms"])
    anchor_keep = float(qwen30["keep_ratio"])
    anchor_ttft_ms = float(qwen30["pruned_ttft_ms"])
    detector_mean_ms = float(easyocr["mean_detector_ms_per_image"])
    detector_median_ms = float(easyocr["median_detector_ms_per_image"])

    latency_rows = []
    for row in fallback_rows:
        keep = float(row["mean_keep"])
        estimated_no_detector_ms = estimate_ttft(
            keep_ratio=keep,
            anchor_keep=anchor_keep,
            anchor_ttft_ms=anchor_ttft_ms,
            full_ttft_ms=full_ttft_ms,
        )
        requires_detector = policy_requires_detector(row)
        mean_detector = detector_mean_ms if requires_detector else 0.0
        median_detector = detector_median_ms if requires_detector else 0.0
        mean_inclusive = estimated_no_detector_ms + mean_detector
        median_inclusive = estimated_no_detector_ms + median_detector
        latency_rows.append(
            {
                "scope": row["scope"],
                "split": row["split"],
                "policy": row["policy"],
                "family": row["family"],
                "score": fmt(row["score"]),
                "mean_keep": fmt(row["mean_keep"]),
                "requires_detector": "yes" if requires_detector else "no",
                "full_ttft_ms": fmt(full_ttft_ms),
                "estimated_no_detector_ttft_ms": fmt(estimated_no_detector_ms),
                "estimated_detector_mean_ttft_ms": fmt(mean_inclusive),
                "estimated_detector_median_ttft_ms": fmt(median_inclusive),
                "speedup_no_detector": fmt(full_ttft_ms / estimated_no_detector_ms),
                "speedup_with_mean_detector": fmt(full_ttft_ms / mean_inclusive),
                "speedup_with_median_detector": fmt(full_ttft_ms / median_inclusive),
                "detector_mean_ms": fmt(mean_detector),
                "detector_median_ms": fmt(median_detector),
                "estimate_note": estimate_note(row, requires_detector),
            }
        )

    key_rows = [
        row
        for row in latency_rows
        if row["scope"] in {"drop_40pct", "mixed_light"}
        and (
            row["policy"] in {"fixed_0.30", "fixed_0.70", "fixed_1.00", "oracle_best_budget"}
            or row["family"]
            in {
                "missing_box_to_0p70",
                "noisy_box_ECR_0p30_to_0p70",
                "noisy_box_all_regions_ECR_ge_0p50_0p30_to_0p70",
            }
        )
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "noisy_box_latency_estimate.csv", latency_rows)
    write_csv(OUT_DIR / "noisy_box_latency_key_summary.csv", key_rows)
    (OUT_DIR / "noisy_box_latency_report.md").write_text(
        build_report(latency_rows, key_rows, detector_mean_ms, detector_median_ms),
        encoding="utf-8",
    )
    print(f"Wrote {OUT_DIR / 'noisy_box_latency_report.md'}")


def estimate_ttft(
    keep_ratio: float,
    anchor_keep: float,
    anchor_ttft_ms: float,
    full_ttft_ms: float,
) -> float:
    if abs(keep_ratio - anchor_keep) < 0.005:
        return anchor_ttft_ms
    if abs(keep_ratio - 1.0) < 0.005:
        return full_ttft_ms
    slope = (full_ttft_ms - anchor_ttft_ms) / (1.0 - anchor_keep)
    estimate = anchor_ttft_ms + (keep_ratio - anchor_keep) * slope
    return max(min(estimate, full_ttft_ms), anchor_ttft_ms)


def policy_requires_detector(row: dict[str, str]) -> bool:
    family = row["family"]
    if family in {"fixed", "oracle"}:
        return False
    return "missing_box" in family or "noisy_box" in family


def estimate_note(row: dict[str, str], requires_detector: bool) -> str:
    if row["family"] == "oracle":
        return "diagnostic upper bound from cached scores; not a deployable latency path"
    if requires_detector:
        return (
            "estimated from measured Qwen TTFT endpoints plus measured EasyOCR latency; "
            "not an end-to-end detector-in-the-loop MLLM run"
        )
    return "estimated from measured Qwen TTFT endpoints; no external detector required"


def build_report(
    rows: list[dict[str, str]],
    key_rows: list[dict[str, str]],
    detector_mean_ms: float,
    detector_median_ms: float,
) -> str:
    drop_all = find_row(
        key_rows,
        scope="drop_40pct",
        family="noisy_box_all_regions_ECR_ge_0p50_0p30_to_0p70",
    )
    drop_fixed70 = find_row(key_rows, scope="drop_40pct", policy="fixed_0.70")
    mixed_all = find_row(
        key_rows,
        scope="mixed_light",
        family="noisy_box_all_regions_ECR_ge_0p50_0p30_to_0p70",
    )
    lines = [
        "# Open-QA Noisy-Box Latency Estimate",
        "",
        "This report estimates latency for cached open-QA noisy-box fallback policies.",
        "It is anchored by measured Qwen TextOCR-Hard TTFT endpoints and measured EasyOCR latency.",
        "It is not an end-to-end detector-in-the-loop MLLM measurement.",
        "",
        "## Inputs",
        "",
        f"- Measured Qwen full TTFT: {drop_fixed70['full_ttft_ms']} ms.",
        f"- Measured EasyOCR latency: mean {detector_mean_ms:.1f} ms/image, median {detector_median_ms:.1f} ms/image.",
        "- Fixed-budget policies do not require an external detector.",
        "- Missing-box and noisy-ECR fallback policies require online evidence boxes in this deployment estimate.",
        "",
        "## Key Reading",
        "",
        (
            f"Under 40% simulated box dropout, noisy all-region ECR fallback matches the fixed-70 quality "
            f"score ({drop_all['score']} vs {drop_fixed70['score']}) with slightly lower mean keep "
            f"({drop_all['mean_keep']} vs {drop_fixed70['mean_keep']}). Without detector cost, the estimated "
            f"TTFT is {drop_all['estimated_no_detector_ttft_ms']} ms ({drop_all['speedup_no_detector']}x). "
            f"With mean EasyOCR cost added online, detector-inclusive TTFT becomes "
            f"{drop_all['estimated_detector_mean_ttft_ms']} ms ({drop_all['speedup_with_mean_detector']}x), "
            f"so it is slower than the full-prefix Qwen baseline."
        ),
        "",
        (
            f"Under mixed light noise, noisy all-region ECR fallback scores {mixed_all['score']} at keep "
            f"{mixed_all['mean_keep']}; estimated no-detector TTFT is "
            f"{mixed_all['estimated_no_detector_ttft_ms']} ms, but mean detector-inclusive TTFT is "
            f"{mixed_all['estimated_detector_mean_ttft_ms']} ms."
        ),
        "",
        "## Deployment Implication",
        "",
        (
            "The noisy-box fallback audit supports quality robustness under missing/noisy boxes, but it should not "
            "be claimed as a single-sample end-to-end speedup when a slow external OCR detector is run online. "
            "It is a plausible efficiency path only when evidence boxes are precomputed, reused, supplied by the "
            "application, or produced by an already-running layout/OCR module."
        ),
        "",
        "## Key Rows",
        "",
        markdown_table(
            key_rows,
            [
                "scope",
                "policy",
                "family",
                "score",
                "mean_keep",
                "requires_detector",
                "estimated_no_detector_ttft_ms",
                "estimated_detector_mean_ttft_ms",
                "speedup_no_detector",
                "speedup_with_mean_detector",
            ],
        ),
        "",
    ]
    return "\n".join(lines)


def find_row(rows: list[dict[str, str]], **criteria: str) -> dict[str, str]:
    return next(row for row in rows if all(row.get(k) == v for k, v in criteria.items()))


def markdown_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(str(row.get(col, "")).replace("|", "/") for col in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    main()
