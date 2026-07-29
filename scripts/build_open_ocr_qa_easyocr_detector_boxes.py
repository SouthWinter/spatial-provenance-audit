#!/usr/bin/env python3
"""Build EasyOCR detector boxes for the open OCR/DocVQA stress pack.

This prepares the missing open-QA detector-in-loop artifact: detector boxes are
converted both to the annotation-tool pixel-box schema and to selector-ready
normalized `evidence_regions` probes. It does not run the MLLM; it makes the
detector source auditable and ready for a later pruning run.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACK = ROOT / "runs" / "problem_optimization_audit" / "open_ocr_qa_annotation_pack" / "annotation_pack.csv"
DEFAULT_OUT = ROOT / "runs" / "problem_optimization_audit" / "open_ocr_qa_easyocr_detector_boxes"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-pack", default=str(DEFAULT_PACK))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--cache-name", default="easyocr_detections.json")
    parser.add_argument("--languages", default="en")
    parser.add_argument("--gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-confidence", type=float, default=0.20)
    parser.add_argument("--limit", type=int, default=None, help="Limit rows for smoke/debug runs.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    pack_rows = read_csv(Path(args.annotation_pack))
    if args.limit is not None:
        pack_rows = pack_rows[: args.limit]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / args.cache_name
    detections = {} if args.force else read_json(cache_path, default={})

    reader = None
    image_paths = unique_image_paths(pack_rows)
    for image_path in image_paths:
        if image_path in detections and not args.force:
            continue
        path = Path(image_path)
        if not path.exists():
            detections[image_path] = empty_detection(image_path, missing=True)
            write_json(cache_path, detections)
            continue
        if reader is None:
            import cv2  # noqa: F401
            import easyocr

            reader = easyocr.Reader(
                [lang.strip() for lang in args.languages.split(",") if lang.strip()],
                gpu=bool(args.gpu),
                verbose=False,
            )
        detections[image_path] = detect_image(
            reader=reader,
            image_path=path,
            min_confidence=args.min_confidence,
        )
        write_json(cache_path, detections)

    annotation_rows = []
    selector_rows = []
    detail_rows = []
    for row in pack_rows:
        detection = detections.get(row["image_path"], empty_detection(row["image_path"]))
        ann, selector_probe, detail = convert_row(row, detection)
        annotation_rows.append(ann)
        selector_rows.append(selector_probe)
        detail_rows.append(detail)

    summary = build_summary(detail_rows, detections, args)
    write_jsonl(out_dir / "easyocr_bbox_annotations.jsonl", annotation_rows)
    write_jsonl(out_dir / "easyocr_selector_probes.jsonl", selector_rows)
    write_csv(out_dir / "easyocr_detector_rows.csv", detail_rows)
    write_csv(out_dir / "easyocr_detector_summary.csv", summary)
    (out_dir / "easyocr_detector_report.md").write_text(
        build_report(summary, detail_rows),
        encoding="utf-8",
    )
    print(
        f"Wrote EasyOCR detector boxes for {len(pack_rows)} open-QA stress rows to {out_dir}; "
        f"rows_with_boxes={sum(int(row['box_count']) > 0 for row in detail_rows)}"
    )


def detect_image(*, reader: Any, image_path: Path, min_confidence: float) -> dict[str, Any]:
    with Image.open(image_path) as image:
        width, height = image.size
    start = time.perf_counter()
    raw = reader.readtext(str(image_path), detail=1, paragraph=False)
    elapsed = time.perf_counter() - start

    boxes = []
    tokens = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        points, text, confidence = item[0], str(item[1]).strip(), float(item[2])
        if confidence < min_confidence:
            continue
        pixel_box = polygon_to_pixel_xywh(points, width=width, height=height)
        if pixel_box is None:
            continue
        norm_box = xywh_to_norm_xyxy(pixel_box, width=width, height=height)
        label = text or "detected_text"
        boxes.append({"x": pixel_box[0], "y": pixel_box[1], "w": pixel_box[2], "h": pixel_box[3], "label": label})
        tokens.append({"text": text, "confidence": confidence, "bbox": norm_box})
    boxes, tokens = dedupe_detection(boxes, tokens)
    return {
        "image": str(image_path),
        "missing": False,
        "width": width,
        "height": height,
        "boxes": boxes,
        "tokens": tokens,
        "elapsed_sec": elapsed,
    }


def convert_row(row: dict[str, str], detection: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    pixel_boxes = list(detection.get("boxes", []) or [])
    tokens = list(detection.get("tokens", []) or [])
    norm_boxes = [token["bbox"] for token in tokens if isinstance(token, dict) and isinstance(token.get("bbox"), list)]
    evidence_units = row.get("evidence_text_candidates", "") or row.get("gold_answers", "")
    status = "easyocr_bbox_proposed" if pixel_boxes else "detector_empty"
    annotation = {
        "sample_id": row["sample_id"],
        "task": row["task"],
        "question_id": row["question_id"],
        "evidence_units": evidence_units,
        "boxes": pixel_boxes,
        "notes": "EasyOCR detector boxes over all visible text; not manually verified evidence boxes.",
        "status": status,
        "detector_name": "EasyOCR",
        "bbox_source": "easyocr_detected_all_text",
    }
    selector_probe = {
        "sample_id": row["sample_id"],
        "id": row["sample_id"],
        "dataset": row["task"],
        "question_id": row["question_id"],
        "image": row["image_path"],
        "image_path": row["image_path"],
        "question": row["question"],
        "target_text": row["question"],
        "source_text": row["question"],
        "task_family": "open_ocr_qa",
        "gold_answers": split_gold_answers(row.get("gold_answers", "")),
        "full_answer": row.get("full_answer", ""),
        "full_score": row.get("full_score", ""),
        "evidence_regions": norm_boxes,
        "ocr_regions": norm_boxes,
        "ocr_tokens": tokens,
        "bbox_source": "easyocr_detected_all_text",
        "detector_name": "EasyOCR",
        "detector_elapsed_sec": float(detection.get("elapsed_sec", 0.0) or 0.0),
        "detector_observed_count": len(norm_boxes),
        "selector_target_source": "question_plus_easyocr_boxes",
    }
    detail = {
        "sample_id": row["sample_id"],
        "task": row["task"],
        "question_id": row["question_id"],
        "image_path": row["image_path"],
        "image_width": int(detection.get("width", 0) or 0),
        "image_height": int(detection.get("height", 0) or 0),
        "box_count": len(pixel_boxes),
        "token_count": len(tokens),
        "mean_confidence": fmt(mean(float(token.get("confidence", 0.0) or 0.0) for token in tokens)),
        "detector_elapsed_ms": fmt(1000.0 * float(detection.get("elapsed_sec", 0.0) or 0.0)),
        "status": status,
        "selection_reasons": row.get("selection_reasons", ""),
        "stress_tags": row.get("stress_tags", ""),
    }
    return annotation, selector_probe, detail


def polygon_to_pixel_xywh(points: Any, *, width: int, height: int) -> list[float] | None:
    if not isinstance(points, (list, tuple)):
        return None
    xs = []
    ys = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        xs.append(float(point[0]))
        ys.append(float(point[1]))
    if not xs or not ys or width <= 0 or height <= 0:
        return None
    x1 = max(0.0, min(float(width), min(xs)))
    x2 = max(0.0, min(float(width), max(xs)))
    y1 = max(0.0, min(float(height), min(ys)))
    y2 = max(0.0, min(float(height), max(ys)))
    if x2 <= x1 or y2 <= y1:
        return None
    return [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]


def xywh_to_norm_xyxy(box: list[float], *, width: int, height: int) -> list[float]:
    x, y, w, h = box
    return [
        clamp01(x / width),
        clamp01(y / height),
        clamp01((x + w) / width),
        clamp01((y + h) / height),
    ]


def dedupe_detection(boxes: list[dict[str, Any]], tokens: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seen = set()
    out_boxes = []
    out_tokens = []
    for box, token in zip(boxes, tokens):
        key = (
            round(float(box["x"]), 2),
            round(float(box["y"]), 2),
            round(float(box["w"]), 2),
            round(float(box["h"]), 2),
            str(box.get("label", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        out_boxes.append(box)
        out_tokens.append(token)
    return out_boxes, out_tokens


def build_summary(detail_rows: list[dict[str, Any]], detections: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    image_rows = [item for item in detections.values() if not item.get("missing")]
    elapsed = [float(item.get("elapsed_sec", 0.0) or 0.0) for item in image_rows]
    per_image_boxes = [len(item.get("boxes", []) or []) for item in image_rows]
    rows_with_boxes = [row for row in detail_rows if int(row["box_count"]) > 0]
    out = [
        {"scope": "all", "metric": "rows", "value": len(detail_rows)},
        {"scope": "all", "metric": "unique_images", "value": len(unique_image_paths(detail_rows))},
        {"scope": "all", "metric": "rows_with_boxes", "value": len(rows_with_boxes)},
        {"scope": "all", "metric": "row_box_coverage_rate", "value": fmt(len(rows_with_boxes) / len(detail_rows) if detail_rows else 0.0)},
        {"scope": "all", "metric": "boxes", "value": sum(int(row["box_count"]) for row in detail_rows)},
        {"scope": "all", "metric": "mean_boxes_per_row", "value": fmt(mean(int(row["box_count"]) for row in detail_rows))},
        {"scope": "all", "metric": "mean_boxes_per_image", "value": fmt(mean(per_image_boxes))},
        {"scope": "all", "metric": "mean_detector_ms_per_image", "value": fmt(1000.0 * mean(elapsed))},
        {"scope": "all", "metric": "median_detector_ms_per_image", "value": fmt(1000.0 * median(elapsed))},
        {"scope": "all", "metric": "min_confidence", "value": args.min_confidence},
        {"scope": "all", "metric": "languages", "value": args.languages},
    ]
    for task in sorted({row["task"] for row in detail_rows}):
        task_rows = [row for row in detail_rows if row["task"] == task]
        task_box_rows = [row for row in task_rows if int(row["box_count"]) > 0]
        out.extend(
            [
                {"scope": task, "metric": "rows", "value": len(task_rows)},
                {"scope": task, "metric": "rows_with_boxes", "value": len(task_box_rows)},
                {
                    "scope": task,
                    "metric": "row_box_coverage_rate",
                    "value": fmt(len(task_box_rows) / len(task_rows) if task_rows else 0.0),
                },
                {"scope": task, "metric": "mean_boxes_per_row", "value": fmt(mean(int(row["box_count"]) for row in task_rows))},
            ]
        )
    return out


def build_report(summary: list[dict[str, Any]], detail_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Open OCR/DocVQA EasyOCR Detector Boxes",
        "",
        "This artifact prepares real OCR detector boxes for the fixed 96-row open OCR/document QA stress pack. The pixel boxes are compatible with the bbox annotation/ECR tools, and the normalized boxes are written as selector-visible `evidence_regions` in `easyocr_selector_probes.jsonl`.",
        "",
        "## Summary",
        "",
        table_md(summary, ["scope", "metric", "value"]),
        "",
        "## Boundary",
        "",
        "This is detector-source preparation, not an MLLM pruning result. It closes the data-path gap needed for an open-QA detector-in-loop run, but quality/latency claims require a subsequent generation run that consumes `easyocr_selector_probes.jsonl`.",
        "",
        "## Empty Detector Rows",
        "",
        "| Sample | Task | Question ID | Image | |",
        "| --- | --- | --- | --- | ---: |",
    ]
    empty = [row for row in detail_rows if int(row["box_count"]) == 0]
    for row in empty[:30]:
        lines.append(f"| {row['sample_id']} | {row['task']} | {row['question_id']} | {row['image_path']} | 0 |")
    if not empty:
        lines.append("| none | | | | |")
    return "\n".join(lines) + "\n"


def split_gold_answers(value: str) -> list[str]:
    return [part.strip() for part in str(value).split("|") if part.strip()]


def unique_image_paths(rows: list[dict[str, Any]]) -> list[str]:
    seen = set()
    out = []
    for row in rows:
        path = str(row.get("image_path", ""))
        if path and path not in seen:
            seen.add(path)
            out.append(path)
    return out


def empty_detection(image_path: str, *, missing: bool = False) -> dict[str, Any]:
    width = height = 0
    if image_path and Path(image_path).exists():
        with Image.open(image_path) as image:
            width, height = image.size
    return {"image": image_path, "missing": missing, "width": width, "height": height, "boxes": [], "tokens": [], "elapsed_sec": 0.0}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path, *, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def table_md(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")).replace("|", "\\|") for col in columns) + " |")
    return "\n".join(lines)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def mean(values: Any) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def median(values: Any) -> float:
    vals = list(values)
    return float(statistics.median(vals)) if vals else 0.0


def fmt(value: float) -> str:
    return f"{float(value):.3f}"


if __name__ == "__main__":
    main()
