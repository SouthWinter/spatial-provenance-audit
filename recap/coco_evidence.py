"""Attach COCO instance boxes as evidence regions for canonical samples."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from recap.prune.metrics import normalize_box

COCO_ALIASES = {
    "airplane": {"airplane", "plane", "aeroplane"},
    "bicycle": {"bicycle", "bike"},
    "cell phone": {"cell phone", "phone", "mobile phone"},
    "couch": {"couch", "sofa"},
    "dining table": {"dining table", "table"},
    "motorcycle": {"motorcycle", "motorbike"},
    "potted plant": {"potted plant", "plant"},
    "refrigerator": {"refrigerator", "fridge"},
    "remote": {"remote", "remote control"},
    "sink": {"sink", "basin"},
    "sports ball": {"sports ball", "ball"},
    "tv": {"tv", "television", "monitor"},
}


def attach_coco_evidence(
    samples: list[dict[str, Any]],
    *,
    instances_file: str | Path,
    min_area: float = 1.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach subject/object boxes by matching sample entity names to COCO categories."""
    index = load_coco_index(instances_file, min_area=min_area)
    enriched: list[dict[str, Any]] = []
    matched_subject = 0
    matched_object = 0
    matched_any = 0
    matched_both = 0

    for sample in samples:
        out = dict(sample)
        image_id = _image_id(sample)
        subject_name = _entity_category_name(sample.get("subject", ""), index["aliases"])
        object_name = _entity_category_name(sample.get("object", ""), index["aliases"])
        subject_box = _best_box(index, image_id=image_id, category_name=subject_name)
        object_box = _best_box(index, image_id=image_id, category_name=object_name)

        evidence_regions: list[list[float]] = []
        if subject_box is not None:
            out["subject_bbox"] = subject_box
            evidence_regions.append(subject_box)
            matched_subject += 1
        if object_box is not None:
            out["object_bbox"] = object_box
            evidence_regions.append(object_box)
            matched_object += 1
        if evidence_regions:
            out["evidence_regions"] = evidence_regions
            out["bbox_source"] = "coco_instances"
            out["evidence_region_count"] = len(evidence_regions)
            out["coco_subject_category"] = subject_name or ""
            out["coco_object_category"] = object_name or ""
            matched_any += 1
            if subject_box is not None and object_box is not None:
                matched_both += 1
        enriched.append(out)

    report = {
        "num_samples": len(samples),
        "matched_any": matched_any,
        "matched_both": matched_both,
        "matched_subject": matched_subject,
        "matched_object": matched_object,
        "instances_file": str(instances_file),
    }
    return enriched, report


def load_coco_index(instances_file: str | Path, *, min_area: float = 1.0) -> dict[str, Any]:
    path = Path(instances_file).expanduser()
    payload = json.loads(path.read_text(encoding="utf-8"))
    categories = {int(cat["id"]): str(cat["name"]) for cat in payload.get("categories", [])}
    aliases = _build_aliases(categories.values())
    image_sizes = {
        int(image["id"]): (float(image["width"]), float(image["height"]))
        for image in payload.get("images", [])
    }
    boxes_by_image_category: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for ann in payload.get("annotations", []):
        if ann.get("iscrowd"):
            continue
        area = float(ann.get("area", 0.0) or 0.0)
        if area < min_area:
            continue
        image_id = int(ann["image_id"])
        category_name = categories.get(int(ann.get("category_id", -1)))
        image_size = image_sizes.get(image_id)
        if not category_name or image_size is None:
            continue
        coco_bbox = ann.get("bbox")
        if not isinstance(coco_bbox, (list, tuple)) or len(coco_bbox) < 4:
            continue
        x, y, w, h = [float(value) for value in coco_bbox[:4]]
        box = normalize_box([x, y, x + w, y + h], image_size=image_size)
        if box is None:
            continue
        boxes_by_image_category[(image_id, category_name)].append(
            {
                "box": [box[0], box[1], box[2], box[3]],
                "area": area,
            }
        )
    return {
        "categories": categories,
        "aliases": aliases,
        "boxes": boxes_by_image_category,
    }


def _build_aliases(category_names: Any) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for name in category_names:
        canonical = str(name)
        for alias in {canonical, canonical.replace("_", " "), canonical.replace(" ", "")}:
            aliases[_normalize_entity(alias)] = canonical
        for alias in COCO_ALIASES.get(canonical, set()):
            aliases[_normalize_entity(alias)] = canonical
    return aliases


def _entity_category_name(value: Any, aliases: dict[str, str]) -> str | None:
    text = _normalize_entity(value)
    if text in aliases:
        return aliases[text]
    # Prefer longest aliases so "dining table" wins over "table".
    for alias, category in sorted(aliases.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(rf"(^| ){re.escape(alias)}($| )", text):
            return category
    return None


def _best_box(index: dict[str, Any], *, image_id: int | None, category_name: str | None) -> list[float] | None:
    if image_id is None or category_name is None:
        return None
    candidates = index["boxes"].get((image_id, category_name), [])
    if not candidates:
        return None
    best = max(candidates, key=lambda item: float(item["area"]))
    return list(best["box"])


def _image_id(sample: dict[str, Any]) -> int | None:
    value = sample.get("image_id", sample.get("coco_image_id"))
    if value in (None, ""):
        image = str(sample.get("image", ""))
        match = re.search(r"(\d{6,12})\.(?:jpg|jpeg|png)$", image, flags=re.IGNORECASE)
        if match:
            value = match.group(1)
    try:
        return int(value)
    except Exception:
        return None


def _normalize_entity(value: Any) -> str:
    text = str(value or "").lower()
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\b(a|an|the|photo|of|picture|image)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()
