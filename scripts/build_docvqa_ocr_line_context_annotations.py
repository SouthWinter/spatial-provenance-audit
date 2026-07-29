#!/usr/bin/env python3
"""Expand DocVQA OCR answer-token boxes with nearby same-line OCR context."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

import pyarrow.parquet as pq

from build_open_ocr_qa_external_bbox_annotations import infer_answer_span_len


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATIONS = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_docvqa_hxlinh_bbox_expanded"
    / "external_bbox_annotations.jsonl"
)
DEFAULT_PARQUETS = sorted((ROOT / "data" / "external_docvqa_hxlinh" / "data").glob("val-*.parquet"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS))
    parser.add_argument("--parquet", action="append", default=[str(path) for path in DEFAULT_PARQUETS])
    parser.add_argument("--neighbor-words", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        default="runs/problem_optimization_audit/open_ocr_qa_docvqa_hxlinh_line_context_bbox",
    )
    args = parser.parse_args()

    annotations = read_jsonl(Path(args.annotations))
    wanted_ids = {f"val_{ann['question_id']}" for ann in annotations if ann.get("task") == "DocVQA-lite"}
    source_rows = load_source_rows([Path(path) for path in args.parquet], wanted_ids)

    out_annotations = []
    detail_rows = []
    for ann in annotations:
        if ann.get("task") != "DocVQA-lite":
            out_annotations.append(dict(ann))
            continue
        qid = str(ann.get("question_id", ""))
        source = source_rows.get(f"val_{qid}")
        expanded, detail = expand_annotation(ann, source, neighbor_words=args.neighbor_words)
        out_annotations.append(expanded)
        detail_rows.append(detail)

    summary = build_summary(detail_rows, args.neighbor_words)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "external_bbox_annotations.jsonl", out_annotations)
    write_csv(out_dir / "line_context_rows.csv", detail_rows)
    write_csv(out_dir / "line_context_summary.csv", summary)
    (out_dir / "line_context_report.md").write_text(build_report(summary, detail_rows), encoding="utf-8")
    print(f"Wrote DocVQA OCR line-context annotations to {out_dir}")


def load_source_rows(paths: list[Path], wanted_ids: set[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    columns = ["id", "words", "bounding_boxes", "answer"]
    for path in paths:
        table = pq.read_table(path, columns=columns)
        for row in table.to_pylist():
            if row.get("id") in wanted_ids:
                out[row["id"]] = row
        if len(out) >= len(wanted_ids):
            break
    return out


def expand_annotation(
    ann: dict[str, Any],
    source: dict[str, Any] | None,
    *,
    neighbor_words: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base_boxes = [box for box in ann.get("boxes", []) if isinstance(box, dict)]
    detail = {
        "sample_id": ann.get("sample_id", ""),
        "question_id": ann.get("question_id", ""),
        "status": ann.get("status", ""),
        "source_found": int(source is not None),
        "base_box_count": len(base_boxes),
        "line_context_box_count": len(base_boxes),
        "added_context_boxes": 0,
        "answer_span_len": 0,
        "scale_status": "not_attempted",
        "notes": "",
    }
    if not source or not base_boxes:
        expanded = dict(ann)
        detail["notes"] = "missing_source_or_base_boxes"
        return expanded, detail

    words = source.get("words")
    raw_boxes = source.get("bounding_boxes")
    answer = source.get("answer")
    if not isinstance(words, list) or not isinstance(raw_boxes, list) or not isinstance(answer, dict):
        expanded = dict(ann)
        detail["notes"] = "missing_ocr_fields"
        return expanded, detail
    try:
        start = int(answer.get("start"))
    except (TypeError, ValueError):
        expanded = dict(ann)
        detail["notes"] = "missing_answer_start"
        return expanded, detail
    span_len = infer_answer_span_len(
        words,
        start,
        answer_text=str(answer.get("text") or ""),
        matched_text=str(answer.get("matched_text") or ""),
    )
    answer_indices = list(range(start, min(len(words), len(raw_boxes), start + span_len)))
    detail["answer_span_len"] = len(answer_indices)
    scales = infer_scales([raw_boxes[i] for i in answer_indices], base_boxes)
    if scales is None:
        expanded = dict(ann)
        detail["scale_status"] = "failed"
        detail["notes"] = "could_not_infer_scale"
        return expanded, detail
    sx, sy = scales
    detail["scale_status"] = "ok"

    context_indices = same_line_context_indices(raw_boxes, answer_indices, neighbor_words=neighbor_words)
    context_boxes = []
    for idx in context_indices:
        box = scale_raw_box(raw_boxes[idx], sx, sy)
        if not box:
            continue
        box["label"] = str(words[idx])
        box["context_role"] = "answer_token" if idx in answer_indices else "line_context"
        context_boxes.append(box)
    merged = dedupe_boxes(context_boxes or base_boxes)
    expanded = dict(ann)
    expanded["boxes"] = merged
    expanded["status"] = "line_context_bbox_proposed" if merged else ann.get("status", "unmatched")
    expanded["notes"] = f"{ann.get('notes', '')}; line_context_neighbor_words={neighbor_words}".strip("; ")
    detail["line_context_box_count"] = len(merged)
    detail["added_context_boxes"] = max(0, len(merged) - len(base_boxes))
    detail["notes"] = "ok"
    return expanded, detail


def infer_scales(raw_answer_boxes: list[Any], scaled_boxes: list[dict[str, Any]]) -> tuple[float, float] | None:
    sx_values = []
    sy_values = []
    for raw, scaled in zip(raw_answer_boxes, scaled_boxes):
        parsed = parse_raw_xyxy(raw)
        if not parsed:
            continue
        x1, y1, x2, y2 = parsed
        raw_w = abs(x2 - x1)
        raw_h = abs(y2 - y1)
        try:
            scaled_w = float(scaled.get("w", 0))
            scaled_h = float(scaled.get("h", 0))
        except (TypeError, ValueError):
            continue
        if raw_w > 0 and scaled_w > 0:
            sx_values.append(scaled_w / raw_w)
        if raw_h > 0 and scaled_h > 0:
            sy_values.append(scaled_h / raw_h)
    if not sx_values or not sy_values:
        return None
    return median(sx_values), median(sy_values)


def same_line_context_indices(raw_boxes: list[Any], answer_indices: list[int], *, neighbor_words: int) -> list[int]:
    parsed = [parse_raw_xyxy(box) for box in raw_boxes]
    if not any(0 <= idx < len(parsed) and parsed[idx] for idx in answer_indices):
        return answer_indices
    context: set[int] = set(answer_indices)
    for answer_idx in answer_indices:
        if answer_idx < 0 or answer_idx >= len(parsed) or not parsed[answer_idx]:
            continue
        answer_box = parsed[answer_idx]
        center_y = (answer_box[1] + answer_box[3]) / 2.0
        height = abs(answer_box[3] - answer_box[1])
        tol = max(8.0, height * 0.75)
        same_line = []
        for idx, box in enumerate(parsed):
            if not box:
                continue
            y = (box[1] + box[3]) / 2.0
            if abs(y - center_y) <= tol:
                same_line.append(idx)
        same_line.sort(key=lambda idx: parsed[idx][0] if parsed[idx] else math.inf)
        if answer_idx not in same_line:
            continue
        pos = same_line.index(answer_idx)
        left = max(0, pos - neighbor_words)
        right = min(len(same_line), pos + neighbor_words + 1)
        context.update(same_line[left:right])
    return sorted(context, key=lambda idx: ((parsed[idx][1] + parsed[idx][3]) / 2.0, parsed[idx][0]) if parsed[idx] else (math.inf, math.inf))


def parse_raw_xyxy(box: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(box[i]) for i in range(4)]
    except (TypeError, ValueError):
        return None
    return x1, y1, x2, y2


def scale_raw_box(raw: Any, sx: float, sy: float) -> dict[str, Any] | None:
    parsed = parse_raw_xyxy(raw)
    if not parsed:
        return None
    x1, y1, x2, y2 = parsed
    x = min(x1, x2) * sx
    y = min(y1, y2) * sy
    w = abs(x2 - x1) * sx
    h = abs(y2 - y1) * sy
    if w <= 0 or h <= 0:
        return None
    return {"x": round(x, 2), "y": round(y, 2), "w": round(w, 2), "h": round(h, 2)}


def dedupe_boxes(boxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for box in boxes:
        key = tuple(int(round(float(box[field]) * 10)) for field in ("x", "y", "w", "h"))
        if key in seen:
            continue
        seen.add(key)
        out.append(box)
    return out


def build_summary(rows: list[dict[str, Any]], neighbor_words: int) -> list[dict[str, Any]]:
    annotated = [row for row in rows if int(row["line_context_box_count"]) > 0]
    return [
        {"scope": "all", "metric": "rows", "value": len(rows)},
        {"scope": "all", "metric": "source_found_rows", "value": sum(int(row["source_found"]) > 0 for row in rows)},
        {"scope": "all", "metric": "annotated_rows", "value": len(annotated)},
        {"scope": "all", "metric": "base_boxes", "value": sum(int(row["base_box_count"]) for row in rows)},
        {"scope": "all", "metric": "line_context_boxes", "value": sum(int(row["line_context_box_count"]) for row in rows)},
        {"scope": "all", "metric": "added_context_boxes", "value": sum(int(row["added_context_boxes"]) for row in rows)},
        {"scope": "all", "metric": "multi_box_rows", "value": sum(int(row["line_context_box_count"]) > 1 for row in rows)},
        {"scope": "all", "metric": "neighbor_words", "value": neighbor_words},
        {"scope": "all", "metric": "scale_failed_rows", "value": sum(row["scale_status"] == "failed" for row in rows)},
    ]


def build_report(summary: list[dict[str, Any]], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# DocVQA OCR Line-Context BBox Annotations",
        "",
        "This artifact expands OCR answer-token boxes with nearby OCR words on the same line. It is a deterministic context heuristic, not manual layout annotation.",
        "",
        "| Scope | Metric | Value |",
        "| --- | --- | ---: |",
    ]
    for row in summary:
        lines.append(f"| {row['scope']} | {row['metric']} | {row['value']} |")
    lines.extend(["", "## Preview", "", "| Sample | Base boxes | Context boxes | Added | Span len | Notes |", "| --- | ---: | ---: | ---: | ---: | --- |"])
    for row in rows[:20]:
        lines.append(
            f"| {row['sample_id']} | {row['base_box_count']} | {row['line_context_box_count']} | "
            f"{row['added_context_boxes']} | {row['answer_span_len']} | {row['notes']} |"
        )
    return "\n".join(lines) + "\n"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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
