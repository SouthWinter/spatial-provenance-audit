#!/usr/bin/env python3
"""Audit the TextOCR-Hard detector-in-loop pruning evidence.

The paper's oracle-box criticism is only weakened if detector boxes are visible
to the selector at pruning time. This script checks that property directly for
the EasyOCR probe file, joins the corresponding LLaVA/InternVL pruning runs,
and reports the quality/latency boundary with detector cost separated from
selector overhead.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "problem_optimization_audit" / "textocr_detector_in_loop_audit"

EASYOCR_PROBES = ROOT / "data" / "textocr_easyocr" / "textocr_hard_easyocr_all.jsonl"
EASYOCR_SUMMARY = ROOT / "data" / "textocr_easyocr" / "easyocr_input_summary.json"
BOX_ROBUSTNESS = ROOT / "runs" / "box_robustness" / "box_robustness_summary.csv"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    easyocr_rows = read_jsonl(EASYOCR_PROBES)
    easyocr_summary = read_json(EASYOCR_SUMMARY)
    box_rows = read_csv(BOX_ROBUSTNESS)

    input_rows = build_input_integrity_rows(easyocr_rows, easyocr_summary)
    model_rows = build_model_rows(box_rows, easyocr_summary)
    key_rows = build_key_rows(input_rows, model_rows, easyocr_summary)

    write_csv(OUT_DIR / "detector_in_loop_input_integrity.csv", input_rows)
    write_csv(OUT_DIR / "detector_in_loop_model_rows.csv", model_rows)
    write_csv(OUT_DIR / "detector_in_loop_key_summary.csv", key_rows)
    (OUT_DIR / "detector_in_loop_report.md").write_text(
        build_report(input_rows, model_rows, key_rows, easyocr_summary),
        encoding="utf-8",
    )
    print(f"Wrote detector-in-loop audit to {OUT_DIR}")


def build_input_integrity_rows(rows: list[dict[str, Any]], summary: dict[str, Any]) -> list[dict[str, Any]]:
    num_rows = len(rows)
    selector_visible = [normalize_boxes(row.get("evidence_regions")) for row in rows]
    oracle = [normalize_boxes(row.get("oracle_evidence_regions")) for row in rows]
    selector_counts = [len(boxes) for boxes in selector_visible]
    oracle_counts = [len(boxes) for boxes in oracle]
    bbox_sources = {str(row.get("bbox_source", "")) for row in rows}
    detector_names = {str(row.get("detector_name", "")) for row in rows}
    same_as_oracle = sum(1 for sel, gold in zip(selector_visible, oracle) if boxes_equal(sel, gold))
    rows_with_detector_elapsed = sum(1 for row in rows if float(row.get("detector_elapsed_sec", 0.0) or 0.0) > 0.0)

    return [
        {"check": "probe_file_exists", "value": int(EASYOCR_PROBES.exists()), "note": str(EASYOCR_PROBES)},
        {"check": "num_probes", "value": num_rows, "note": "EasyOCR probe rows"},
        {"check": "bbox_source_values", "value": ";".join(sorted(bbox_sources)), "note": "Expected easyocr_detected_all_text"},
        {"check": "detector_name_values", "value": ";".join(sorted(detector_names)), "note": "Expected EasyOCR"},
        {
            "check": "selector_visible_boxes_mean",
            "value": fmt(mean(selector_counts)),
            "note": "Mean len(evidence_regions); these boxes are used by selectors.",
        },
        {
            "check": "oracle_boxes_mean",
            "value": fmt(mean(oracle_counts)),
            "note": "Mean len(oracle_evidence_regions); these boxes are audit-only.",
        },
        {
            "check": "selector_visible_rows_with_boxes_rate",
            "value": fmt(mean(1.0 if count > 0 else 0.0 for count in selector_counts)),
            "note": "Rows with EasyOCR boxes visible to pruning selector.",
        },
        {
            "check": "oracle_rows_with_boxes_rate",
            "value": fmt(mean(1.0 if count > 0 else 0.0 for count in oracle_counts)),
            "note": "Rows with TextOCR-Hard oracle boxes preserved for audit.",
        },
        {
            "check": "selector_equals_oracle_rate",
            "value": fmt(same_as_oracle / num_rows if num_rows else 0.0),
            "note": "Should be low: selector uses detector boxes, not copied oracle boxes.",
        },
        {
            "check": "rows_with_detector_elapsed",
            "value": rows_with_detector_elapsed,
            "note": "Detector timing is recorded once per image then copied to probes from that image.",
        },
        {
            "check": "detector_mean_ms_per_image",
            "value": fmt(float(summary.get("mean_detector_ms_per_image", 0.0) or 0.0)),
            "note": "EasyOCR detector time from input builder.",
        },
        {
            "check": "detector_oracle_missing_rate",
            "value": fmt(float(summary.get("oracle_missing_rate", 0.0) or 0.0)),
            "note": "Mean best IoU to oracle is zero for this fraction of probes.",
        },
        {
            "check": "detector_mean_best_iou_to_oracle",
            "value": fmt(float(summary.get("mean_best_iou_to_oracle", 0.0) or 0.0)),
            "note": "Detector quality against TextOCR-Hard evidence boxes.",
        },
    ]


def build_model_rows(rows: list[dict[str, str]], summary: dict[str, Any]) -> list[dict[str, Any]]:
    detector_ms_image = float(summary.get("mean_detector_ms_per_image", 0.0) or 0.0)
    num_images = float(summary.get("num_images", 0.0) or 0.0)
    num_probes = float(summary.get("num_probes", 0.0) or 0.0)
    detector_ms_probe_amortized = detector_ms_image * num_images / num_probes if num_probes else 0.0

    selected = [
        row
        for row in rows
        if row.get("variant") in {"GT boxes", "EasyOCR detected boxes", "missing boxes"}
        and row.get("model") in {"LLaVA-1.5-7B", "InternVL3.5-8B"}
    ]
    out: list[dict[str, Any]] = []
    for row in selected:
        variant = row["variant"]
        selector_overhead = f(row.get("mean_prune_overhead_ms"))
        detector_included = variant == "EasyOCR detected boxes"
        detector_ms_probe = detector_ms_probe_amortized if detector_included else 0.0
        detector_ms_image_for_row = detector_ms_image if detector_included else 0.0
        out.append(
            {
                "model": row["model"],
                "box_source": variant,
                "selector": row["selector"],
                "num_samples": int(float(row.get("num_samples", 0) or 0)),
                "accuracy": fmt(f(row.get("accuracy"))),
                "hFPR": fmt(f(row.get("hFPR"))),
                "keep_ratio": fmt(f(row.get("keep_ratio"))),
                "true_ECR": fmt(f(row.get("true_ECR"))),
                "true_CenterR": fmt(f(row.get("true_CenterR"))),
                "selector_box_ECR": fmt(f(row.get("selector_box_ECR"))),
                "selector_overhead_ms_per_probe": fmt(selector_overhead),
                "detector_ms_per_image": fmt(detector_ms_image_for_row),
                "detector_ms_per_probe_amortized": fmt(detector_ms_probe),
                "detector_plus_selector_ms_per_probe": fmt(detector_ms_probe + selector_overhead),
                "detector_included": int(detector_included),
            }
        )
    return out


def build_key_rows(
    input_rows: list[dict[str, Any]],
    model_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    by_key = {(row["model"], row["box_source"]): row for row in model_rows}
    rows: list[dict[str, Any]] = [
        {
            "scope": "input",
            "metric": "detector_source",
            "value": "EasyOCR boxes in evidence_regions; TextOCR boxes in oracle_evidence_regions",
            "interpretation": "Detector boxes are selector-visible; oracle boxes are audit-only.",
        },
        {
            "scope": "input",
            "metric": "oracle_missing_rate",
            "value": fmt(float(summary.get("oracle_missing_rate", 0.0) or 0.0)),
            "interpretation": "This is a difficult detector source, not a clean oracle replacement.",
        },
        {
            "scope": "input",
            "metric": "mean_detector_ms_per_image",
            "value": fmt(float(summary.get("mean_detector_ms_per_image", 0.0) or 0.0)),
            "interpretation": "Include this only for detector-inclusive deployment claims.",
        },
    ]
    for model in ("LLaVA-1.5-7B", "InternVL3.5-8B"):
        gt = by_key.get((model, "GT boxes"))
        easy = by_key.get((model, "EasyOCR detected boxes"))
        missing = by_key.get((model, "missing boxes"))
        if not easy:
            continue
        rows.append(
            {
                "scope": model,
                "metric": "easyocr_detector_in_loop",
                "value": f"acc={easy['accuracy']}; hFPR={easy['hFPR']}; true_ECR={easy['true_ECR']}",
                "interpretation": "Selection-time EasyOCR boxes, audited against held-out oracle boxes.",
            }
        )
        if gt:
            rows.append(
                {
                    "scope": model,
                    "metric": "easyocr_minus_gt",
                    "value": (
                        f"acc_delta={fmt(float(easy['accuracy']) - float(gt['accuracy']))}; "
                        f"hFPR_delta={fmt(float(easy['hFPR']) - float(gt['hFPR']))}; "
                        f"true_ECR_delta={fmt(float(easy['true_ECR']) - float(gt['true_ECR']))}"
                    ),
                    "interpretation": "Shows how much is lost when oracle boxes are replaced by EasyOCR.",
                }
            )
        if missing:
            rows.append(
                {
                    "scope": model,
                    "metric": "easyocr_minus_missing",
                    "value": (
                        f"acc_delta={fmt(float(easy['accuracy']) - float(missing['accuracy']))}; "
                        f"hFPR_delta={fmt(float(easy['hFPR']) - float(missing['hFPR']))}; "
                        f"true_ECR_delta={fmt(float(easy['true_ECR']) - float(missing['true_ECR']))}"
                    ),
                    "interpretation": "Shows whether detector boxes add value over no selector-visible boxes.",
                }
            )
    return rows


def build_report(
    input_rows: list[dict[str, Any]],
    model_rows: list[dict[str, Any]],
    key_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    lines = [
        "# TextOCR-Hard Detector-in-Loop Audit",
        "",
        "This audit checks whether real OCR detector boxes are used during pruning selection, not only during post-hoc ECR measurement.",
        "",
        "## Verdict",
        "",
        verdict(input_rows, model_rows, summary),
        "",
        "## Key Summary",
        "",
        table_md(key_rows, ["scope", "metric", "value", "interpretation"]),
        "",
        "## Input Integrity",
        "",
        table_md(input_rows, ["check", "value", "note"]),
        "",
        "## Model Rows",
        "",
        table_md(
            model_rows,
            [
                "model",
                "box_source",
                "selector",
                "num_samples",
                "accuracy",
                "hFPR",
                "keep_ratio",
                "true_ECR",
                "true_CenterR",
                "selector_box_ECR",
                "selector_overhead_ms_per_probe",
                "detector_ms_per_probe_amortized",
                "detector_plus_selector_ms_per_probe",
            ],
        ),
        "",
        "## Claim Boundary",
        "",
        "Safe claim: TextOCR-Hard includes a real detector-in-loop stress test where EasyOCR boxes are visible to the selector and TextOCR-Hard boxes are retained only for audit. InternVL soft evidence remains close to its GT-box operating point under this difficult detector source, while LLaVA hard protection degrades. Unsafe claim: detector-assisted pruning provides single-sample end-to-end speedup when EasyOCR must be run online; detector cost must be reported separately or amortized.",
    ]
    return "\n".join(lines) + "\n"


def verdict(input_rows: list[dict[str, Any]], model_rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    checks = {row["check"]: str(row["value"]) for row in input_rows}
    source_ok = checks.get("bbox_source_values") == "easyocr_detected_all_text"
    detector_ok = checks.get("detector_name_values") == "EasyOCR"
    selector_not_oracle = float(checks.get("selector_equals_oracle_rate", 1.0)) < 0.05
    has_model_rows = any(row["box_source"] == "EasyOCR detected boxes" for row in model_rows)
    if source_ok and detector_ok and selector_not_oracle and has_model_rows:
        return (
            "Detector-in-loop evidence is present for TextOCR-Hard: EasyOCR boxes are selector-visible, "
            "oracle boxes are audit-only, and LLaVA/InternVL EasyOCR pruning runs exist. This addresses "
            "the narrow oracle-box criticism on TextOCR-Hard, while detector latency and open-QA detector "
            "pipelines remain separate deployment boundaries."
        )
    return "Detector-in-loop evidence is incomplete or inconsistent; inspect the input integrity table before using the results."


def normalize_boxes(value: Any) -> list[tuple[float, float, float, float]]:
    if not isinstance(value, list):
        return []
    boxes: list[tuple[float, float, float, float]] = []
    for item in value:
        raw = item.get("bbox") if isinstance(item, dict) else item
        if not isinstance(raw, (list, tuple)) or len(raw) < 4:
            continue
        try:
            boxes.append(tuple(round(float(raw[idx]), 6) for idx in range(4)))
        except Exception:
            continue
    return boxes


def boxes_equal(a: list[tuple[float, float, float, float]], b: list[tuple[float, float, float, float]]) -> bool:
    return sorted(a) == sorted(b)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def table_md(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "(empty)"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")).replace("|", "\\|") for col in columns) + " |")
    return "\n".join(lines)


def mean(values: Any) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def f(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def fmt(value: float) -> str:
    return f"{float(value):.3f}"


if __name__ == "__main__":
    main()
