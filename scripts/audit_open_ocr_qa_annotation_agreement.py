#!/usr/bin/env python3
"""Audit agreement between two manual evidence-box annotation exports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-a", required=True, help="First JSON/JSONL annotation export")
    parser.add_argument("--annotations-b", required=True, help="Second JSON/JSONL annotation export")
    parser.add_argument(
        "--output-dir",
        default="runs/problem_optimization_audit/open_ocr_qa_annotation_agreement",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument(
        "--sample-scope",
        choices=("intersection", "union"),
        default="intersection",
        help=(
            "Compare only samples present in both exports (the default for a "
            "calibration subset), or the union when both exports should cover "
            "the same complete sample set."
        ),
    )
    args = parser.parse_args()

    ann_a = load_by_sample(Path(args.annotations_a))
    ann_b = load_by_sample(Path(args.annotations_b))
    rows = []
    sample_ids = set(ann_a) & set(ann_b)
    if args.sample_scope == "union":
        sample_ids = set(ann_a) | set(ann_b)
    for sample_id in sorted(sample_ids):
        a = ann_a.get(sample_id)
        b = ann_b.get(sample_id)
        rows.append(compare_sample(sample_id, a, b, args.iou_threshold))

    summary = build_summary(rows)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "annotation_agreement_rows.csv", rows)
    write_csv(out_dir / "annotation_agreement_summary.csv", summary)
    (out_dir / "annotation_agreement_report.md").write_text(
        build_report(summary, rows, args.iou_threshold), encoding="utf-8"
    )
    print(
        f"Audited {len(rows)} samples; mean_best_iou={summary_value(summary, 'all', 'mean_best_iou')}"
    )


def compare_sample(sample_id: str, a: dict[str, Any] | None, b: dict[str, Any] | None, iou_threshold: float) -> dict[str, Any]:
    boxes_a = normalized_boxes(a)
    boxes_b = normalized_boxes(b)
    matches = greedy_match(boxes_a, boxes_b)
    matched_ious = [iou for _, _, iou in matches]
    label_types_a = label_types(boxes_a)
    label_types_b = label_types(boxes_b)
    exact_type_match = int(label_types_a == label_types_b)
    enough_iou_matches = sum(iou >= iou_threshold for iou in matched_ious)
    set_overlap = region_set_overlap(boxes_a, boxes_b)
    return {
        "sample_id": sample_id,
        "task": first_nonempty(a, b, "task"),
        "status_a": "" if a is None else str(a.get("status", "")),
        "status_b": "" if b is None else str(b.get("status", "")),
        "present_a": int(a is not None),
        "present_b": int(b is not None),
        "box_count_a": len(boxes_a),
        "box_count_b": len(boxes_b),
        "box_count_delta": len(boxes_a) - len(boxes_b),
        "label_types_a": ";".join(label_types_a),
        "label_types_b": ";".join(label_types_b),
        "label_type_set_match": exact_type_match,
        "matched_box_pairs": len(matches),
        "iou_ge_threshold_pairs": enough_iou_matches,
        "mean_matched_iou": round(sum(matched_ious) / len(matched_ious), 4) if matched_ious else 0.0,
        "min_matched_iou": round(min(matched_ious), 4) if matched_ious else 0.0,
        "union_region_iou": round(set_overlap["iou"], 4),
        "primary_covered_by_secondary": round(set_overlap["a_covered_by_b"], 4),
        "secondary_covered_by_primary": round(set_overlap["b_covered_by_a"], 4),
        "all_boxes_matched_iou": int(
            len(boxes_a) == len(boxes_b)
            and len(matches) == len(boxes_a)
            and len(boxes_a) > 0
            and enough_iou_matches == len(boxes_a)
        ),
        "needs_adjudication": int(
            a is None
            or b is None
            or len(boxes_a) != len(boxes_b)
            or not exact_type_match
            or (len(boxes_a) > 0 and enough_iou_matches < max(len(boxes_a), len(boxes_b)))
            or (len(boxes_a) == 0 and len(boxes_b) == 0)
        ),
    }


def build_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = [
        {"scope": "all", "metric": "samples", "value": len(rows)},
        {"scope": "all", "metric": "present_in_both", "value": sum(row["present_a"] and row["present_b"] for row in rows)},
        {"scope": "all", "metric": "box_count_match", "value": sum(row["box_count_a"] == row["box_count_b"] for row in rows)},
        {"scope": "all", "metric": "label_type_set_match", "value": sum(int(row["label_type_set_match"]) for row in rows)},
        {"scope": "all", "metric": "all_boxes_matched_iou", "value": sum(int(row["all_boxes_matched_iou"]) for row in rows)},
        {"scope": "all", "metric": "needs_adjudication", "value": sum(int(row["needs_adjudication"]) for row in rows)},
        {"scope": "all", "metric": "mean_best_iou", "value": round(mean([float(row["mean_matched_iou"]) for row in rows]), 4)},
        {"scope": "all", "metric": "mean_union_region_iou", "value": round(mean([float(row["union_region_iou"]) for row in rows]), 4)},
        {"scope": "all", "metric": "mean_primary_covered_by_secondary", "value": round(mean([float(row["primary_covered_by_secondary"]) for row in rows]), 4)},
        {"scope": "all", "metric": "mean_secondary_covered_by_primary", "value": round(mean([float(row["secondary_covered_by_primary"]) for row in rows]), 4)},
    ]
    for task in sorted({row["task"] for row in rows if row["task"]}):
        task_rows = [row for row in rows if row["task"] == task]
        out.extend(
            [
                {"scope": task, "metric": "samples", "value": len(task_rows)},
                {"scope": task, "metric": "box_count_match", "value": sum(row["box_count_a"] == row["box_count_b"] for row in task_rows)},
                {"scope": task, "metric": "label_type_set_match", "value": sum(int(row["label_type_set_match"]) for row in task_rows)},
                {"scope": task, "metric": "all_boxes_matched_iou", "value": sum(int(row["all_boxes_matched_iou"]) for row in task_rows)},
                {"scope": task, "metric": "needs_adjudication", "value": sum(int(row["needs_adjudication"]) for row in task_rows)},
                {"scope": task, "metric": "mean_best_iou", "value": round(mean([float(row["mean_matched_iou"]) for row in task_rows]), 4)},
                {"scope": task, "metric": "mean_union_region_iou", "value": round(mean([float(row["union_region_iou"]) for row in task_rows]), 4)},
                {"scope": task, "metric": "mean_primary_covered_by_secondary", "value": round(mean([float(row["primary_covered_by_secondary"]) for row in task_rows]), 4)},
                {"scope": task, "metric": "mean_secondary_covered_by_primary", "value": round(mean([float(row["secondary_covered_by_primary"]) for row in task_rows]), 4)},
            ]
        )
    return out


def build_report(summary: list[dict[str, Any]], rows: list[dict[str, Any]], iou_threshold: float) -> str:
    lines = [
        "# Annotation Agreement Audit",
        "",
        f"IoU threshold for a matched evidence box: {iou_threshold:.2f}.",
        "",
        "| Scope | Metric | Value |",
        "| --- | --- | ---: |",
    ]
    for row in summary:
        lines.append(f"| {row['scope']} | {row['metric']} | {row['value']} |")
    lines.extend(
        [
            "",
            "## Rows Needing Adjudication",
            "",
            "| Sample | Task | Boxes A/B | Types A | Types B | Matched IoU | Union IoU | Reason |",
            "| --- | --- | ---: | --- | --- | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        if int(row["needs_adjudication"]):
            reason = adjudication_reason(row)
            lines.append(
                f"| {row['sample_id']} | {row['task']} | {row['box_count_a']}/{row['box_count_b']} | "
                f"{row['label_types_a']} | {row['label_types_b']} | {row['mean_matched_iou']} | "
                f"{row['union_region_iou']} | {reason} |"
            )
    return "\n".join(lines) + "\n"


def adjudication_reason(row: dict[str, Any]) -> str:
    reasons = []
    if not row["present_a"] or not row["present_b"]:
        reasons.append("missing export")
    if row["box_count_a"] == 0 and row["box_count_b"] == 0:
        reasons.append("no boxes")
    if row["box_count_a"] != row["box_count_b"]:
        reasons.append("box count")
    if not int(row["label_type_set_match"]):
        reasons.append("label types")
    if row["box_count_a"] and not int(row["all_boxes_matched_iou"]):
        reasons.append("box IoU")
    return ", ".join(reasons) or "manual check"


def normalized_boxes(row: dict[str, Any] | None) -> list[dict[str, Any]]:
    if row is None:
        return []
    boxes = row.get("boxes", [])
    if not isinstance(boxes, list):
        return []
    out = []
    for box in boxes:
        if not isinstance(box, dict):
            continue
        try:
            x = float(box.get("x", 0))
            y = float(box.get("y", 0))
            w = float(box.get("w", 0))
            h = float(box.get("h", 0))
        except (TypeError, ValueError):
            continue
        if w <= 0 or h <= 0:
            continue
        out.append({"x": x, "y": y, "w": w, "h": h, "label": str(box.get("label", "")).strip()})
    return out


def greedy_match(boxes_a: list[dict[str, Any]], boxes_b: list[dict[str, Any]]) -> list[tuple[int, int, float]]:
    candidates = []
    for i, box_a in enumerate(boxes_a):
        for j, box_b in enumerate(boxes_b):
            candidates.append((box_iou(box_a, box_b), i, j))
    candidates.sort(reverse=True)
    used_a: set[int] = set()
    used_b: set[int] = set()
    matches = []
    for iou, i, j in candidates:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matches.append((i, j, round(iou, 4)))
    return matches


def box_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax1, ay1, ax2, ay2 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]
    bx1, by1, bx2, by2 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return 0.0 if union <= 0 else inter / union


def region_set_overlap(boxes_a: list[dict[str, Any]], boxes_b: list[dict[str, Any]]) -> dict[str, float]:
    rectangles_a = [box_to_rect(box) for box in boxes_a]
    rectangles_b = [box_to_rect(box) for box in boxes_b]
    area_a = rectangle_union_area(rectangles_a)
    area_b = rectangle_union_area(rectangles_b)
    union = rectangle_union_area(rectangles_a + rectangles_b)
    intersection = max(0.0, area_a + area_b - union)
    return {
        "iou": 0.0 if union <= 0 else intersection / union,
        "a_covered_by_b": 0.0 if area_a <= 0 else intersection / area_a,
        "b_covered_by_a": 0.0 if area_b <= 0 else intersection / area_b,
    }


def box_to_rect(box: dict[str, Any]) -> tuple[float, float, float, float]:
    return (box["x"], box["y"], box["x"] + box["w"], box["y"] + box["h"])


def rectangle_union_area(rectangles: list[tuple[float, float, float, float]]) -> float:
    rectangles = [rect for rect in rectangles if rect[2] > rect[0] and rect[3] > rect[1]]
    if not rectangles:
        return 0.0
    xs = sorted({coordinate for rect in rectangles for coordinate in (rect[0], rect[2])})
    area = 0.0
    for x1, x2 in zip(xs, xs[1:]):
        if x2 <= x1:
            continue
        intervals = sorted(
            (rect[1], rect[3]) for rect in rectangles if rect[0] < x2 and rect[2] > x1
        )
        if not intervals:
            continue
        covered_y = 0.0
        current_start, current_end = intervals[0]
        for start, end in intervals[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                covered_y += current_end - current_start
                current_start, current_end = start, end
        covered_y += current_end - current_start
        area += (x2 - x1) * covered_y
    return area


def label_types(boxes: list[dict[str, Any]]) -> list[str]:
    types = []
    for box in boxes:
        label = str(box.get("label", "")).strip()
        label_type = label.split(":", 1)[0] if ":" in label else label
        if label_type:
            types.append(label_type)
    return sorted(set(types))


def first_nonempty(a: dict[str, Any] | None, b: dict[str, Any] | None, key: str) -> str:
    for row in (a, b):
        if row is not None and row.get(key):
            return str(row[key])
    return ""


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summary_value(summary: list[dict[str, Any]], scope: str, metric: str) -> Any:
    for row in summary:
        if row["scope"] == scope and row["metric"] == metric:
            return row["value"]
    return ""


def load_by_sample(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_annotations(path)
    out = {}
    for row in rows:
        sid = str(row.get("sample_id", ""))
        if sid:
            out[sid] = row
    return out


def load_annotations(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


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
