#!/usr/bin/env python3
"""Evaluate bbox evidence coverage for the open OCR QA stress pack."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recap.prune.metrics import (
    Box,
    box_area,
    evidence_center_recall,
    evidence_coverage,
    evidence_patch_recall,
    make_token_grid,
)


DEFAULT_ANNOTATIONS = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_bbox_annotation_tool"
    / "annotation_seed.jsonl"
)
PREFILL_JSONL = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_evidence_prefill"
    / "evidence_prefill_pack.jsonl"
)
TRACE_RUNS = (
    (
        "TextVQA-lite",
        "0.30",
        ROOT / "runs" / "open_ocr_qa" / "qwen3_8b_textvqa_lite_target_grid0p30_full500" / "prune_traces.jsonl",
    ),
    (
        "TextVQA-lite",
        "0.50",
        ROOT / "runs" / "open_ocr_qa" / "qwen3_8b_textvqa_lite_target_grid0p50_full500" / "prune_traces.jsonl",
    ),
    (
        "TextVQA-lite",
        "0.70",
        ROOT / "runs" / "open_ocr_qa" / "qwen3_8b_textvqa_lite_target_grid0p70_full500" / "prune_traces.jsonl",
    ),
    (
        "DocVQA-lite",
        "0.30",
        ROOT / "runs" / "open_ocr_qa" / "qwen3_8b_docvqa_lite_target_grid0p30_full500" / "prune_traces.jsonl",
    ),
    (
        "DocVQA-lite",
        "0.50",
        ROOT / "runs" / "open_ocr_qa" / "qwen3_8b_docvqa_lite_target_grid0p50_full500" / "prune_traces.jsonl",
    ),
    (
        "DocVQA-lite",
        "0.70",
        ROOT / "runs" / "open_ocr_qa" / "qwen3_8b_docvqa_lite_target_grid0p70_full500" / "prune_traces.jsonl",
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS), help="JSON list or JSONL exported by the HTML tool")
    parser.add_argument("--output-dir", default="runs/problem_optimization_audit/open_ocr_qa_bbox_ecr")
    args = parser.parse_args()

    annotations = load_annotations(Path(args.annotations))
    prefill = {str(row["sample_id"]): row for row in read_jsonl(PREFILL_JSONL)}
    traces = load_traces()

    rows: list[dict[str, Any]] = []
    for ann in annotations:
        sample_id = str(ann.get("sample_id", ""))
        meta = prefill.get(sample_id, {})
        task = str(ann.get("task") or meta.get("task") or task_from_sample_id(sample_id))
        image_path = Path(str(meta.get("image_path", "")))
        width, height = image_size(image_path)
        evidence_boxes = normalize_annotation_boxes(ann.get("boxes", []), width, height)
        base = {
            "sample_id": sample_id,
            "task": task,
            "question_id": ann.get("question_id", meta.get("question_id", "")),
            "status": ann.get("status", ""),
            "box_count": len(evidence_boxes),
            "image_width": width,
            "image_height": height,
            "prefill_units": meta.get("prefill_evidence_units", ann.get("evidence_units", "")),
        }
        for ratio in ("0.30", "0.50", "0.70"):
            trace = traces.get((task, ratio), {}).get(sample_id)
            row = dict(base)
            row.update({"budget_keep_ratio": ratio, "trace_found": int(trace is not None)})
            if not evidence_boxes or trace is None or width <= 0 or height <= 0:
                row.update(empty_metric_fields(reason=missing_reason(evidence_boxes, trace, width, height)))
                rows.append(row)
                continue
            token_count = int(trace.get("full_visual_tokens", 0) or 0)
            kept_indices = [int(i) for i in trace.get("kept_indices", [])]
            grid_rows, grid_cols = infer_grid_shape(token_count, width, height)
            token_boxes = make_token_grid(grid_rows, grid_cols)
            per_box_ecr = [
                evidence_coverage(kept_indices, token_boxes, [box])
                for box in evidence_boxes
            ]
            row.update(
                {
                    "metric_status": "scored",
                    "grid_source": "aspect_ratio_inferred",
                    "grid_rows": grid_rows,
                    "grid_cols": grid_cols,
                    "token_count": token_count,
                    "kept_token_count": len(kept_indices),
                    "effective_keep_ratio": fmt_float(len(kept_indices) / token_count if token_count else 0.0),
                    "ECR": fmt_float(evidence_coverage(kept_indices, token_boxes, evidence_boxes)),
                    "CenterR": fmt_float(evidence_center_recall(kept_indices, token_boxes, evidence_boxes)),
                    "PatchR": fmt_float(evidence_patch_recall(kept_indices, token_boxes, evidence_boxes)),
                    "mean_region_ECR": fmt_float(mean(per_box_ecr)),
                    "worst_region_ECR": fmt_float(min(per_box_ecr) if per_box_ecr else 0.0),
                    "all_regions_ECR_ge_0p50": fmt_float(1.0 if per_box_ecr and all(v >= 0.5 for v in per_box_ecr) else 0.0),
                    "all_regions_center_hit": fmt_float(
                        1.0 if evidence_center_recall(kept_indices, token_boxes, evidence_boxes) >= 1.0 else 0.0
                    ),
                    "reason": "",
                }
            )
            rows.append(row)

    summary = build_summary(rows)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "bbox_ecr_rows.csv", rows)
    write_csv(out_dir / "bbox_ecr_summary.csv", summary)
    (out_dir / "bbox_ecr_report.md").write_text(build_report(summary, rows, Path(args.annotations)), encoding="utf-8")
    print(f"Wrote bbox ECR report for {len(annotations)} annotations to {out_dir}")


def load_traces() -> dict[tuple[str, str], dict[str, dict[str, Any]]]:
    out: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for task, ratio, path in TRACE_RUNS:
        traces: dict[str, dict[str, Any]] = {}
        if path.exists():
            for row in read_jsonl(path):
                traces[str(row.get("sample_id", ""))] = row
        out[(task, ratio)] = traces
    return out


def normalize_annotation_boxes(raw_boxes: Any, width: int, height: int) -> list[Box]:
    if not isinstance(raw_boxes, list) or width <= 0 or height <= 0:
        return []
    boxes: list[Box] = []
    for raw in raw_boxes:
        if not isinstance(raw, dict):
            continue
        parsed = parse_xywh_box(raw, width, height)
        if parsed is not None:
            boxes.append(parsed)
    return dedupe_boxes(boxes)


def parse_xywh_box(raw: dict[str, Any], width: int, height: int) -> Box | None:
    try:
        if {"x", "y", "w", "h"}.issubset(raw):
            x1 = float(raw["x"]) / width
            y1 = float(raw["y"]) / height
            x2 = (float(raw["x"]) + float(raw["w"])) / width
            y2 = (float(raw["y"]) + float(raw["h"])) / height
        elif {"x1", "y1", "x2", "y2"}.issubset(raw):
            x1 = float(raw["x1"]) / width
            y1 = float(raw["y1"]) / height
            x2 = float(raw["x2"]) / width
            y2 = float(raw["y2"]) / height
        else:
            return None
    except (TypeError, ValueError):
        return None
    x1, x2 = sorted((clamp01(x1), clamp01(x2)))
    y1, y2 = sorted((clamp01(y1), clamp01(y2)))
    box = (x1, y1, x2, y2)
    return box if box_area(box) > 0.0 else None


def infer_grid_shape(token_count: int, width: int, height: int) -> tuple[int, int]:
    if token_count <= 0:
        return 1, 1
    aspect = width / max(1.0, float(height))
    candidates = []
    for rows in range(1, int(math.sqrt(token_count)) + 1):
        if token_count % rows == 0:
            cols = token_count // rows
            candidates.append((rows, cols))
            if rows != cols:
                candidates.append((cols, rows))
    if not candidates:
        rows = int(round(math.sqrt(token_count / max(aspect, 1e-6))))
        rows = max(1, rows)
        cols = max(1, math.ceil(token_count / rows))
        return rows, cols
    return min(
        candidates,
        key=lambda rc: (abs((rc[1] / rc[0]) - aspect), abs(rc[0] * rc[1] - token_count), rc[0]),
    )


def build_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    summary.extend(summary_for_scope("all", rows))
    for task in sorted({str(row["task"]) for row in rows}):
        summary.extend(summary_for_scope(task, [row for row in rows if row["task"] == task]))
    for task in sorted({str(row["task"]) for row in rows}):
        for ratio in ("0.30", "0.50", "0.70"):
            group = [row for row in rows if row["task"] == task and row["budget_keep_ratio"] == ratio]
            summary.extend(summary_for_scope(f"{task}@{ratio}", group))
    return summary


def summary_for_scope(scope: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored = [row for row in rows if row.get("metric_status") == "scored"]
    annotated_sample_ids = {row["sample_id"] for row in rows if int(row.get("box_count") or 0) > 0}
    out = [
        {"scope": scope, "metric": "rows", "value": len(rows)},
        {"scope": scope, "metric": "unique_samples", "value": len({row["sample_id"] for row in rows})},
        {"scope": scope, "metric": "annotated_samples", "value": len(annotated_sample_ids)},
        {"scope": scope, "metric": "scored_rows", "value": len(scored)},
    ]
    for metric in ("ECR", "CenterR", "PatchR", "mean_region_ECR", "worst_region_ECR", "all_regions_ECR_ge_0p50"):
        vals = [float(row[metric]) for row in scored if row.get(metric) not in ("", None)]
        out.append({"scope": scope, "metric": f"mean_{metric}", "value": fmt_float(mean(vals)) if vals else ""})
    return out


def build_report(summary: list[dict[str, Any]], rows: list[dict[str, Any]], annotations_path: Path) -> str:
    scored = [row for row in rows if row.get("metric_status") == "scored"]
    lines = [
        "# Open OCR QA BBox Evidence Coverage",
        "",
        f"Annotations: `{annotations_path}`",
        "",
        "This report projects exported evidence boxes onto cached pruning masks for the TextVQA-lite and DocVQA-lite stress pack. If the annotation file has no boxes, the report only validates that the metric path is ready.",
        "",
        "Grid note: the cached Qwen open-QA traces do not store native `grid_h/grid_w`, so scored rows infer a row-major grid from visual-token count and exported image aspect ratio.",
        "",
        "## Summary",
        "",
        "| Scope | Metric | Value |",
        "| --- | --- | ---: |",
    ]
    for row in summary:
        lines.append(f"| {row['scope']} | {row['metric']} | {row['value']} |")
    lines.extend(["", "## Scored Rows", ""])
    if not scored:
        lines.append("No rows contain validated evidence boxes yet; this is expected for the seed annotation file.")
    else:
        lines.extend(
            [
                "| Sample | Task | Keep | Boxes | ECR | Worst Region ECR | CenterR | Grid |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in scored[:200]:
            lines.append(
                f"| {row['sample_id']} | {row['task']} | {row['budget_keep_ratio']} | "
                f"{row['box_count']} | {row['ECR']} | {row['worst_region_ECR']} | "
                f"{row['CenterR']} | {row['grid_rows']}x{row['grid_cols']} |"
            )
    return "\n".join(lines) + "\n"


def empty_metric_fields(reason: str) -> dict[str, Any]:
    return {
        "metric_status": "not_scored",
        "grid_source": "",
        "grid_rows": "",
        "grid_cols": "",
        "token_count": "",
        "kept_token_count": "",
        "effective_keep_ratio": "",
        "ECR": "",
        "CenterR": "",
        "PatchR": "",
        "mean_region_ECR": "",
        "worst_region_ECR": "",
        "all_regions_ECR_ge_0p50": "",
        "all_regions_center_hit": "",
        "reason": reason,
    }


def missing_reason(evidence_boxes: list[Box], trace: dict[str, Any] | None, width: int, height: int) -> str:
    reasons = []
    if not evidence_boxes:
        reasons.append("no_valid_boxes")
    if trace is None:
        reasons.append("missing_trace")
    if width <= 0 or height <= 0:
        reasons.append("missing_image_size")
    return ";".join(reasons)


def task_from_sample_id(sample_id: str) -> str:
    if sample_id.startswith("docvqa"):
        return "DocVQA-lite"
    if sample_id.startswith("textvqa"):
        return "TextVQA-lite"
    return ""


def image_size(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    with Image.open(path) as img:
        return img.size


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


def dedupe_boxes(boxes: list[Box]) -> list[Box]:
    seen: set[tuple[int, int, int, int]] = set()
    out: list[Box] = []
    for box in boxes:
        key = tuple(int(round(v * 1_000_000)) for v in box)
        if key in seen:
            continue
        seen.add(key)
        out.append(box)
    return out


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def fmt_float(value: float) -> str:
    return f"{float(value):.3f}"


if __name__ == "__main__":
    main()
