"""Prepare OCR/document samples with explicit text-region evidence boxes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from recap.prune.metrics import normalize_box


def prepare_ocr_region_samples(
    *,
    input_path: str | Path,
    image_root: str | Path | None = None,
    dataset_name: str = "OCR-Regions",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Convert common OCR JSON/JSONL schemas to canonical evidence samples.

    Supported region fields include evidence_regions, ocr_regions, ocr_tokens,
    ocr_bboxes, answer_bboxes, and grounding_bboxes. Region items may be plain
    boxes, dicts with bbox/box, or polygon/points entries.
    """
    rows = _read_rows(input_path)
    if limit is not None:
        rows = rows[:limit]
    root = Path(image_root).expanduser() if image_root else None
    samples: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        question = str(row.get("question", row.get("query", row.get("prompt", "")))).strip()
        answer = row.get("answer", row.get("answers", row.get("label", "")))
        image = _resolve_image(row, root)
        regions = _extract_regions(row)
        sample = {
            "id": str(row.get("id", row.get("sample_id", f"ocr-region-{idx}"))),
            "dataset": str(row.get("dataset", dataset_name) or dataset_name),
            "source_dataset": dataset_name,
            "image": image,
            "image_id": row.get("image_id", ""),
            "question": question,
            "answer": _answer_text(answer),
            "task_family": "ocr_text",
            "bbox_source": "ocr_regions" if regions else "none",
            "evidence_regions": regions,
            "ocr_regions": regions,
            "evidence_region_count": len(regions),
        }
        for key in ("width", "height", "image_width", "image_height", "image_size"):
            if key in row:
                sample[key] = row[key]
        samples.append(sample)
    return samples


def _read_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser()
    if source.suffix.lower() == ".jsonl":
        rows = []
        with source.open("r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    if isinstance(payload, dict):
        for key in ("data", "examples", "samples", "annotations"):
            if isinstance(payload.get(key), list):
                return [dict(row) for row in payload[key]]
    raise ValueError(f"Unsupported OCR region payload: {source}")


def _resolve_image(row: dict[str, Any], root: Path | None) -> str:
    value = str(row.get("image", row.get("image_path", row.get("file_name", row.get("filename", "")))) or "")
    if root and value and not Path(value).is_absolute():
        return str((root / value).expanduser())
    return value


def _extract_regions(row: dict[str, Any]) -> list[list[float]]:
    raw_regions = (
        row.get("evidence_regions")
        or row.get("evidence_bboxes")
        or row.get("answer_bboxes")
        or row.get("grounding_bboxes")
        or row.get("ocr_regions")
        or row.get("ocr_bboxes")
        or row.get("ocr_boxes")
        or row.get("ocr_tokens")
        or []
    )
    if isinstance(raw_regions, dict):
        raw_regions = list(raw_regions.values())
    if not isinstance(raw_regions, (list, tuple)):
        return []
    image_size = _image_size(row)
    regions: list[list[float]] = []
    for item in raw_regions:
        box = normalize_box(item, image_size=image_size)
        if box is not None:
            regions.append([box[0], box[1], box[2], box[3]])
    return _dedupe(regions)


def _image_size(row: dict[str, Any]) -> tuple[float, float] | None:
    width = row.get("width", row.get("image_width"))
    height = row.get("height", row.get("image_height"))
    size = row.get("image_size")
    if (width is None or height is None) and isinstance(size, (list, tuple)) and len(size) >= 2:
        width, height = size[0], size[1]
    try:
        if width is None or height is None:
            return None
        width_f, height_f = float(width), float(height)
    except Exception:
        return None
    if width_f <= 0 or height_f <= 0:
        return None
    return width_f, height_f


def _answer_text(answer: Any) -> str:
    if isinstance(answer, (list, tuple)):
        return str(answer[0]) if answer else ""
    return str(answer)


def _dedupe(regions: list[list[float]]) -> list[list[float]]:
    seen: set[tuple[int, int, int, int]] = set()
    out: list[list[float]] = []
    for region in regions:
        key = tuple(int(round(coord * 1_000_000)) for coord in region)
        if key in seen:
            continue
        seen.add(key)
        out.append(region)
    return out
