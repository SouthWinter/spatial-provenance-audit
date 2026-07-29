"""Bounding-box parsing utilities without image dependencies."""

from __future__ import annotations

import ast
import re
from typing import Any


def get_bbox(sample: dict[str, Any], role: str) -> list[float] | None:
    keys = [f"{role}_bbox", f"bbox_{role}", f"{role}_box", f"{role}Box"]
    if role == "subject":
        keys.extend(["subj_bbox", "subj_box"])
    else:
        keys.extend(["obj_bbox", "obj_box"])
    for key in keys:
        if key in sample and sample[key] not in (None, ""):
            return parse_bbox(sample[key])
    boxes = sample.get("bboxes") or sample.get("boxes")
    if isinstance(boxes, dict):
        aliases = (role, "subj" if role == "subject" else "obj")
        for alias in aliases:
            if alias in boxes:
                return parse_bbox(boxes[alias])
    return None


def parse_bbox(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except Exception:
            value = [float(part) for part in re.split(r"[,\s]+", value.strip()) if part]
    if isinstance(value, dict):
        if all(key in value for key in ("x1", "y1", "x2", "y2")):
            return [float(value["x1"]), float(value["y1"]), float(value["x2"]), float(value["y2"])]
        if all(key in value for key in ("x", "y", "w", "h")):
            x, y, w, h = float(value["x"]), float(value["y"]), float(value["w"]), float(value["h"])
            return [x, y, x + w, y + h]
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        return [float(value[0]), float(value[1]), float(value[2]), float(value[3])]
    return None


def to_pixels(bbox: list[float] | None, width: int, height: int) -> tuple[int, int, int, int] | None:
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = int(max(0, min(width - 1, round(x1))))
    x2 = int(max(x1 + 1, min(width, round(x2))))
    y1 = int(max(0, min(height - 1, round(y1))))
    y2 = int(max(y1 + 1, min(height, round(y2))))
    return x1, y1, x2, y2

