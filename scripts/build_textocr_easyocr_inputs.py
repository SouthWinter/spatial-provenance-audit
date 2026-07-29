#!/usr/bin/env python
"""Build TextOCR-Hard probe files with real EasyOCR detector boxes.

The emitted probes use EasyOCR-detected text boxes as ``evidence_regions`` for
selection, while preserving the original TextOCR-Hard supporting boxes as
``oracle_evidence_regions`` for post-hoc evidence coverage auditing.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

Box = list[float]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/textocr_val_hard_probes_500img.jsonl")
    parser.add_argument("--output-dir", default="data/textocr_easyocr")
    parser.add_argument("--output-name", default="textocr_hard_easyocr_all.jsonl")
    parser.add_argument("--cache-name", default="easyocr_detections.json")
    parser.add_argument("--languages", default="en")
    parser.add_argument("--gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-confidence", type=float, default=0.20)
    parser.add_argument("--limit-images", type=int, default=None)
    parser.add_argument("--limit-probes", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if args.limit_probes is not None:
        rows = rows[: args.limit_probes]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / args.cache_name
    detections = {} if args.force else read_json(cache_path, default={})

    image_paths = unique_images(rows)
    if args.limit_images is not None:
        allowed = set(image_paths[: args.limit_images])
        rows = [row for row in rows if str(row.get("image", "")) in allowed]
        image_paths = image_paths[: args.limit_images]

    reader = None
    missing_images: list[str] = []
    for image_path in image_paths:
        if image_path in detections and not args.force:
            continue
        path = Path(image_path)
        if not path.exists():
            missing_images.append(image_path)
            detections[image_path] = {
                "image": image_path,
                "missing": True,
                "boxes": [],
                "tokens": [],
                "elapsed_sec": 0.0,
            }
            continue
        if reader is None:
            # Import cv2 before easyocr to avoid a libtiff/libjpeg load-order
            # conflict observed in the local conda environment.
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

    out_rows = [
        attach_detections(row, detections.get(str(row.get("image", "")), empty_detection(str(row.get("image", "")))))
        for row in rows
    ]
    output_path = output_dir / args.output_name
    write_jsonl(output_path, out_rows)

    image_summary = summarize_images(image_paths, detections)
    probe_summary = summarize_probes(out_rows)
    summary = {
        "input": args.input,
        "output": str(output_path),
        "cache": str(cache_path),
        "detector": "EasyOCR",
        "languages": args.languages,
        "gpu": bool(args.gpu),
        "min_confidence": args.min_confidence,
        "num_images": len(image_paths),
        "num_probes": len(out_rows),
        "missing_images": len(missing_images),
        **image_summary,
        **probe_summary,
    }
    write_json(output_dir / "easyocr_input_summary.json", summary)
    write_markdown(output_dir / "easyocr_input_summary.md", summary)
    print(f"Wrote {len(out_rows)} probes with EasyOCR boxes to {output_path}")


def detect_image(*, reader: Any, image_path: Path, min_confidence: float) -> dict[str, Any]:
    width, height = image_size(image_path)
    start = time.perf_counter()
    raw = reader.readtext(str(image_path), detail=1, paragraph=False)
    elapsed = time.perf_counter() - start
    boxes: list[Box] = []
    tokens: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        points, text, confidence = item[0], str(item[1]), float(item[2])
        if confidence < min_confidence:
            continue
        box = polygon_to_box(points, width=width, height=height)
        if box is None:
            continue
        boxes.append(box)
        tokens.append({"text": text, "confidence": confidence, "bbox": box})
    return {
        "image": str(image_path),
        "missing": False,
        "width": width,
        "height": height,
        "boxes": dedupe_boxes(boxes),
        "tokens": tokens,
        "elapsed_sec": elapsed,
    }


def attach_detections(row: dict[str, Any], detection: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    oracle = normalize_regions(row.get("oracle_evidence_regions") or row.get("evidence_regions") or row.get("ocr_regions"))
    observed = normalize_regions(detection.get("boxes"))
    out["oracle_evidence_regions"] = oracle
    out["oracle_ocr_regions"] = oracle
    out["evidence_regions"] = observed
    out["ocr_regions"] = observed
    out["ocr_tokens"] = detection.get("tokens", [])
    out["evidence_region_count"] = len(observed)
    out["has_bbox"] = bool(observed)
    out["base_has_bbox"] = bool(observed)
    out["bbox_source"] = "easyocr_detected_all_text"
    out["box_robustness_variant"] = "easyocr_all"
    out["detector_name"] = "EasyOCR"
    out["detector_elapsed_sec"] = float(detection.get("elapsed_sec", 0.0) or 0.0)
    out["detector_observed_count"] = len(observed)
    out["detector_mean_best_iou"] = mean_best_iou(oracle, observed)
    out["detector_missing_oracle"] = mean_best_iou(oracle, observed) <= 0.0
    return out


def image_size(path: Path) -> tuple[int, int]:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    height, width = image.shape[:2]
    return int(width), int(height)


def polygon_to_box(points: Any, *, width: int, height: int) -> Box | None:
    if not isinstance(points, (list, tuple)):
        return None
    xs: list[float] = []
    ys: list[float] = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        xs.append(float(point[0]))
        ys.append(float(point[1]))
    if not xs or not ys or width <= 0 or height <= 0:
        return None
    return clip_box([min(xs) / width, min(ys) / height, max(xs) / width, max(ys) / height])


def normalize_regions(value: Any) -> list[Box]:
    if not isinstance(value, (list, tuple)):
        return []
    regions: list[Box] = []
    for item in value:
        raw = item.get("bbox") if isinstance(item, dict) else item
        if not isinstance(raw, (list, tuple)) or len(raw) < 4:
            continue
        try:
            box = [float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])]
        except Exception:
            continue
        box = clip_box(box)
        if (box[2] - box[0]) * (box[3] - box[1]) > 0:
            regions.append(box)
    return dedupe_boxes(regions)


def clip_box(box: Box) -> Box:
    x1, y1, x2, y2 = box
    x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
    if x2 <= x1:
        x2 = min(1.0, x1 + 1e-6)
    if y2 <= y1:
        y2 = min(1.0, y1 + 1e-6)
    return [x1, y1, x2, y2]


def dedupe_boxes(boxes: list[Box]) -> list[Box]:
    seen: set[tuple[int, int, int, int]] = set()
    out: list[Box] = []
    for box in boxes:
        key = tuple(int(round(coord * 1_000_000)) for coord in box)
        if key in seen:
            continue
        seen.add(key)
        out.append(box)
    return out


def mean_best_iou(oracle: list[Box], observed: list[Box]) -> float:
    if not oracle:
        return 0.0
    return sum(max((iou(a, b) for b in observed), default=0.0) for a in oracle) / len(oracle)


def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def summarize_images(image_paths: list[str], detections: dict[str, Any]) -> dict[str, float]:
    selected = [detections.get(path, empty_detection(path)) for path in image_paths]
    elapsed = [float(item.get("elapsed_sec", 0.0) or 0.0) for item in selected]
    counts = [float(len(item.get("boxes", []) or [])) for item in selected]
    total_elapsed = sum(elapsed)
    return {
        "total_detector_sec": total_elapsed,
        "mean_detector_ms_per_image": 1000.0 * mean(elapsed),
        "median_detector_ms_per_image": 1000.0 * median(elapsed),
        "mean_detected_boxes_per_image": mean(counts),
        "median_detected_boxes_per_image": median(counts),
        "images_without_detected_boxes_rate": mean(1.0 if count == 0 else 0.0 for count in counts),
    }


def summarize_probes(rows: list[dict[str, Any]]) -> dict[str, float]:
    ious = [float(row.get("detector_mean_best_iou", 0.0) or 0.0) for row in rows]
    counts = [float(row.get("detector_observed_count", 0.0) or 0.0) for row in rows]
    return {
        "mean_best_iou_to_oracle": mean(ious),
        "median_best_iou_to_oracle": median(ious),
        "oracle_missing_rate": mean(1.0 if iou_value <= 0.0 else 0.0 for iou_value in ious),
        "probe_mean_detected_boxes": mean(counts),
    }


def unique_images(rows: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    images: list[str] = []
    for row in rows:
        image = str(row.get("image", ""))
        if not image or image in seen:
            continue
        seen.add(image)
        images.append(image)
    return images


def empty_detection(image: str) -> dict[str, Any]:
    return {"image": image, "missing": True, "boxes": [], "tokens": [], "elapsed_sec": 0.0}


def mean(values: Any) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def median(values: Any) -> float:
    values = list(values)
    return float(statistics.median(values)) if values else 0.0


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: str | Path, *, default: Any) -> Any:
    source = Path(path)
    if not source.exists() or source.stat().st_size == 0:
        return default
    return json.loads(source.read_text(encoding="utf-8"))


def write_json(path: str | Path, obj: Any) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_markdown(path: str | Path, summary: dict[str, Any]) -> None:
    lines = [
        "# TextOCR-Hard EasyOCR Detector Inputs",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    keys = [
        "num_images",
        "num_probes",
        "missing_images",
        "mean_detector_ms_per_image",
        "median_detector_ms_per_image",
        "mean_detected_boxes_per_image",
        "median_detected_boxes_per_image",
        "images_without_detected_boxes_rate",
        "mean_best_iou_to_oracle",
        "median_best_iou_to_oracle",
        "oracle_missing_rate",
        "probe_mean_detected_boxes",
    ]
    for key in keys:
        value = summary.get(key, "")
        if isinstance(value, float):
            value = f"{value:.4f}"
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            f"Output: `{summary['output']}`",
            f"Cache: `{summary['cache']}`",
            "",
            "EasyOCR detections are used only as selector-visible boxes. Original TextOCR-Hard boxes are preserved as oracle boxes for auditing.",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
