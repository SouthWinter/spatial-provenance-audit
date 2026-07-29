"""Evidence and geometry metrics for visual token pruning."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterable

from recap.bbox import get_bbox, parse_bbox

Box = tuple[float, float, float, float]


def make_token_grid(rows: int, cols: int | None = None) -> list[Box]:
    """Build row-major normalized patch boxes for a visual token grid."""
    if rows <= 0:
        raise ValueError(f"rows must be positive, got {rows}.")
    cols = rows if cols is None else cols
    if cols <= 0:
        raise ValueError(f"cols must be positive, got {cols}.")
    boxes: list[Box] = []
    for row in range(rows):
        for col in range(cols):
            boxes.append((col / cols, row / rows, (col + 1) / cols, (row + 1) / rows))
    return boxes


def box_area(box: Box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(a: Box, b: Box) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return box_area((x1, y1, x2, y2))


def union_area(boxes: Iterable[Box]) -> float:
    """Exact rectangle union area using an edge-sweep over x stripes."""
    box_tuple = tuple(
        sorted(
            (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
            for box in boxes
            if box_area(box) > 0.0
        )
    )
    if not box_tuple:
        return 0.0
    return _union_area_cached(box_tuple)


@lru_cache(maxsize=8192)
def _union_area_cached(boxes: tuple[Box, ...]) -> float:
    xs = sorted({coord for box in boxes for coord in (box[0], box[2])})
    area = 0.0
    for x1, x2 in zip(xs, xs[1:]):
        if x2 <= x1:
            continue
        y_intervals = [
            (box[1], box[3])
            for box in boxes
            if box[0] < x2 and box[2] > x1
        ]
        if not y_intervals:
            continue
        y_intervals.sort()
        merged_y = 0.0
        cur_y1, cur_y2 = y_intervals[0]
        for y1, y2 in y_intervals[1:]:
            if y1 <= cur_y2:
                cur_y2 = max(cur_y2, y2)
            else:
                merged_y += max(0.0, cur_y2 - cur_y1)
                cur_y1, cur_y2 = y1, y2
        merged_y += max(0.0, cur_y2 - cur_y1)
        area += (x2 - x1) * merged_y
    return area


def normalize_box(raw_box: Any, *, image_size: tuple[float, float] | None = None) -> Box | None:
    """Parse and normalize a bbox to [0, 1] coordinates when possible."""
    raw_box = _unwrap_box(raw_box)
    parsed = parse_bbox(raw_box)
    if parsed is None:
        return None
    x1, y1, x2, y2 = parsed
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) > 1.5:
        if image_size is None:
            return None
        width, height = image_size
        if width <= 0 or height <= 0:
            return None
        x1, x2 = x1 / width, x2 / width
        y1, y2 = y1 / height, y2 / height
    x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
    box = (x1, y1, x2, y2)
    return box if box_area(box) > 0.0 else None


def image_size_from_sample(sample: dict[str, Any]) -> tuple[float, float] | None:
    """Infer image size from common width/height fields."""
    width = sample.get("width", sample.get("image_width"))
    height = sample.get("height", sample.get("image_height"))
    if width is None or height is None:
        size = sample.get("image_size")
        if isinstance(size, (list, tuple)) and len(size) >= 2:
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


def evidence_regions_from_sample(sample: dict[str, Any]) -> list[Box]:
    """Collect normalized evidence regions from common sample schemas."""
    image_size = image_size_from_sample(sample)
    regions: list[Box] = []

    explicit = (
        sample.get("evidence_regions")
        or sample.get("evidence_bboxes")
        or sample.get("evidence_boxes")
        or sample.get("answer_bboxes")
        or sample.get("grounding_bboxes")
        or sample.get("ocr_regions")
        or sample.get("ocr_bboxes")
        or sample.get("ocr_boxes")
        or sample.get("ocr_tokens")
    )
    if isinstance(explicit, dict):
        explicit = list(explicit.values())
    if isinstance(explicit, (list, tuple)):
        for item in explicit:
            box = normalize_box(item, image_size=image_size)
            if box is not None:
                regions.append(box)

    for role in ("subject", "object"):
        box = normalize_box(get_bbox(sample, role), image_size=image_size)
        if box is not None:
            regions.append(box)

    return _dedupe_boxes(regions)


def _unwrap_box(raw_box: Any) -> Any:
    if isinstance(raw_box, dict):
        for key in ("bbox", "box", "region", "rect", "bounds"):
            if key in raw_box:
                return raw_box[key]
        for key in ("points", "polygon", "vertices"):
            if key in raw_box:
                box = _box_from_points(raw_box[key])
                if box is not None:
                    return box
    return raw_box


def _box_from_points(points: Any) -> list[float] | None:
    if not isinstance(points, (list, tuple)) or not points:
        return None
    xs: list[float] = []
    ys: list[float] = []
    for point in points:
        if isinstance(point, dict):
            x = point.get("x", point.get("X"))
            y = point.get("y", point.get("Y"))
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            x, y = point[0], point[1]
        else:
            continue
        try:
            xs.append(float(x))
            ys.append(float(y))
        except Exception:
            continue
    if not xs or not ys:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


def evidence_coverage(
    kept_indices: Iterable[int],
    token_boxes: list[Box],
    evidence_regions: list[Box],
) -> float:
    """Fraction of evidence-region area covered by retained token patches."""
    if not evidence_regions:
        return 0.0
    kept_boxes = [token_boxes[i] for i in kept_indices if 0 <= int(i) < len(token_boxes)]
    if not kept_boxes:
        return 0.0
    evidence_area = union_area(evidence_regions)
    if evidence_area <= 0.0:
        return 0.0
    covered_parts: list[Box] = []
    for evidence in evidence_regions:
        for token in kept_boxes:
            x1 = max(evidence[0], token[0])
            y1 = max(evidence[1], token[1])
            x2 = min(evidence[2], token[2])
            y2 = min(evidence[3], token[3])
            part = (x1, y1, x2, y2)
            if box_area(part) > 0.0:
                covered_parts.append(part)
    if len(covered_parts) > 5000:
        covered_area = sum(box_area(part) for part in covered_parts)
    else:
        covered_area = union_area(covered_parts)
    return min(1.0, covered_area / evidence_area)


def evidence_center_recall(
    kept_indices: Iterable[int],
    token_boxes: list[Box],
    evidence_regions: list[Box],
) -> float:
    """Fraction of evidence boxes whose center falls inside a retained patch."""
    if not evidence_regions:
        return 0.0
    kept_boxes = [token_boxes[int(i)] for i in kept_indices if 0 <= int(i) < len(token_boxes)]
    if not kept_boxes:
        return 0.0
    hits = 0
    for evidence in evidence_regions:
        cx = (evidence[0] + evidence[2]) / 2.0
        cy = (evidence[1] + evidence[3]) / 2.0
        if any(_point_in_box(cx, cy, kept_box) for kept_box in kept_boxes):
            hits += 1
    return hits / len(evidence_regions)


def evidence_patch_recall(
    kept_indices: Iterable[int],
    token_boxes: list[Box],
    evidence_regions: list[Box],
    *,
    min_overlap: float = 0.0,
) -> float:
    """Fraction of token patches that touch evidence and are retained."""
    if not evidence_regions:
        return 0.0
    evidence_patch_indices = {
        idx
        for idx, token in enumerate(token_boxes)
        if any(intersection_area(token, evidence) > min_overlap for evidence in evidence_regions)
    }
    if not evidence_patch_indices:
        return 0.0
    kept = {int(i) for i in kept_indices if 0 <= int(i) < len(token_boxes)}
    return len(evidence_patch_indices & kept) / len(evidence_patch_indices)


def exhaustive_merge_source_indices(
    num_tokens: int,
    anchor_indices: Iterable[int],
    *,
    contextual_tokens: int,
) -> list[int]:
    """Return source-cell lineage for a selector that merges every non-anchor token."""
    anchors = sorted({int(i) for i in anchor_indices if 0 <= int(i) < num_tokens})
    if contextual_tokens > 0:
        return list(range(num_tokens))
    return anchors


def _point_in_box(x: float, y: float, box: Box) -> bool:
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def saa_lite(answer_correct: bool, evidence_coverage_value: float, *, threshold: float = 0.5) -> float:
    """Strict attributed accuracy proxy for a retained-token mask."""
    return 1.0 if bool(answer_correct) and evidence_coverage_value >= threshold else 0.0


def _dedupe_boxes(boxes: list[Box]) -> list[Box]:
    seen: set[tuple[int, int, int, int]] = set()
    out: list[Box] = []
    for box in boxes:
        key = tuple(int(round(coord * 1_000_000)) for coord in box)
        if key in seen:
            continue
        seen.add(key)
        out.append(box)
    return out
