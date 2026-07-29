#!/usr/bin/env python3
"""Validate exported bbox annotations for the open OCR QA stress pack."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
TOOL_SEED = ROOT / "runs" / "problem_optimization_audit" / "open_ocr_qa_bbox_annotation_tool" / "annotation_seed.jsonl"
PREFILL_JSONL = ROOT / "runs" / "problem_optimization_audit" / "open_ocr_qa_evidence_prefill" / "evidence_prefill_pack.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default=str(TOOL_SEED), help="JSON list or JSONL annotations exported by the tool")
    parser.add_argument("--output-dir", default="runs/problem_optimization_audit/open_ocr_qa_bbox_annotation_validation")
    args = parser.parse_args()

    annotations = load_annotations(Path(args.annotations))
    prefill = {row["sample_id"]: row for row in read_jsonl(PREFILL_JSONL)}
    rows = []
    for ann in annotations:
        sid = str(ann.get("sample_id", ""))
        meta = prefill.get(sid, {})
        image_path = Path(str(meta.get("image_path", "")))
        width = height = 0
        if image_path.exists():
            with Image.open(image_path) as img:
                width, height = img.size
        boxes = ann.get("boxes", [])
        if not isinstance(boxes, list):
            boxes = []
        invalid = [box for box in boxes if not valid_box(box, width, height)]
        unlabeled = [box for box in boxes if not str(box.get("label", "")).strip()]
        status = str(ann.get("status", ""))
        annotated_without_boxes = status == "annotated" and len(boxes) == 0
        rows.append(
            {
                "sample_id": sid,
                "task": ann.get("task", meta.get("task", "")),
                "status": status,
                "box_count": len(boxes),
                "invalid_box_count": len(invalid),
                "unlabeled_box_count": len(unlabeled),
                "annotated_without_boxes": int(annotated_without_boxes),
                "ready_for_ecr": int(len(boxes) > 0 and not invalid and not unlabeled),
                "image_width": width,
                "image_height": height,
                "prefill_units": meta.get("prefill_evidence_units", ""),
                "prefill_complexity": meta.get("prefill_complexity", ""),
                "suggested_region_count": meta.get("prefill_suggested_region_count", ""),
                "notes": ann.get("notes", ""),
            }
        )
    summary = build_summary(rows)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "bbox_annotation_validation_rows.csv", rows)
    write_csv(out_dir / "bbox_annotation_validation_summary.csv", summary)
    (out_dir / "bbox_annotation_validation_report.md").write_text(build_report(summary, rows), encoding="utf-8")
    print(f"Validated {len(rows)} annotation rows; annotated={sum(int(r['box_count']) > 0 for r in rows)}")


def valid_box(box: Any, width: int, height: int) -> bool:
    if not isinstance(box, dict) or width <= 0 or height <= 0:
        return False
    try:
        x = float(box.get("x", 0))
        y = float(box.get("y", 0))
        w = float(box.get("w", 0))
        h = float(box.get("h", 0))
    except (TypeError, ValueError):
        return False
    return w > 0 and h > 0 and x >= 0 and y >= 0 and x + w <= width + 1 and y + h <= height + 1


def build_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = [
        {"scope": "all", "metric": "rows", "value": len(rows)},
        {"scope": "all", "metric": "annotated_rows", "value": sum(int(row["box_count"]) > 0 for row in rows)},
        {"scope": "all", "metric": "empty_rows", "value": sum(int(row["box_count"]) == 0 for row in rows)},
        {"scope": "all", "metric": "invalid_box_rows", "value": sum(int(row["invalid_box_count"]) > 0 for row in rows)},
        {"scope": "all", "metric": "unlabeled_box_rows", "value": sum(int(row["unlabeled_box_count"]) > 0 for row in rows)},
        {"scope": "all", "metric": "annotated_without_box_rows", "value": sum(int(row["annotated_without_boxes"]) > 0 for row in rows)},
        {"scope": "all", "metric": "ready_for_ecr_rows", "value": sum(int(row["ready_for_ecr"]) > 0 for row in rows)},
    ]
    for task in sorted({row["task"] for row in rows}):
        task_rows = [row for row in rows if row["task"] == task]
        out.append({"scope": task, "metric": "rows", "value": len(task_rows)})
        out.append({"scope": task, "metric": "annotated_rows", "value": sum(int(row["box_count"]) > 0 for row in task_rows)})
        out.append({"scope": task, "metric": "ready_for_ecr_rows", "value": sum(int(row["ready_for_ecr"]) > 0 for row in task_rows)})
    return out


def build_report(summary: list[dict[str, Any]], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# BBox Annotation Validation",
        "",
        "| Scope | Metric | Value |",
        "| --- | --- | ---: |",
    ]
    for row in summary:
        lines.append(f"| {row['scope']} | {row['metric']} | {row['value']} |")
    lines.extend(["", "## Rows Not Ready For ECR", "", "| Sample | Task | Boxes | Invalid | Unlabeled | Status | Prefill units |", "| --- | --- | ---: | ---: | ---: | --- | --- |"])
    for row in rows:
        if int(row["ready_for_ecr"]) == 0:
            lines.append(
                f"| {row['sample_id']} | {row['task']} | {row['box_count']} | "
                f"{row['invalid_box_count']} | {row['unlabeled_box_count']} | "
                f"{row['status']} | {row['prefill_units']} |"
            )
    return "\n".join(lines) + "\n"


def load_annotations(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
