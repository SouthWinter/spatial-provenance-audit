"""Image loading and view transforms for RECAP probes."""

from __future__ import annotations

import ast
import base64
import io
import os
import re
from typing import Any

from PIL import Image, ImageDraw, ImageOps


def encode_image_data_url(image: Image.Image) -> str:
    """Encode a PIL image as a JPEG data URL for multimodal chat templates."""

    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def probe_to_visual(probe: dict[str, Any], *, strict: bool = False) -> list[Image.Image]:
    if probe.get("rice_view") == "text_only":
        return []

    image = load_image(probe)
    if image is None:
        if strict:
            raise FileNotFoundError(_missing_image_message(probe))
        return []

    view = probe.get("rice_view", "original")
    if view in {"flip_mapped", "flip_unmapped"}:
        image = ImageOps.mirror(image)
    elif view in {"vertical_flip_mapped", "vertical_flip_unmapped"}:
        image = ImageOps.flip(image)
    elif view == "subject_masked":
        image = mask_bbox(image, get_bbox(probe, "subject"))
    elif view == "object_masked":
        image = mask_bbox(image, get_bbox(probe, "object"))
    elif view == "evidence_masked":
        image = mask_regions(image, probe.get("evidence_regions") or probe.get("ocr_regions"))
    elif view in {"random_evidence_sized_masked", "mask_regions"}:
        image = mask_regions(image, probe.get("mask_regions") or probe.get("random_regions"))
    elif view == "crop":
        image = crop_union(image, get_bbox(probe, "subject"), get_bbox(probe, "object"), padding=float(probe.get("crop_padding", 0.15)))
    elif view == "boxed":
        image = draw_relation_boxes(image, get_bbox(probe, "subject"), get_bbox(probe, "object"))
    return [image.convert("RGB")]


def probe_requires_visual(probe: dict[str, Any]) -> bool:
    return probe.get("rice_view") != "text_only"


def validate_probe_images(probes: list[dict[str, Any]], *, limit_examples: int = 10) -> dict[str, Any]:
    checked = 0
    missing_count = 0
    missing: list[dict[str, Any]] = []
    text_only = 0
    for probe in probes:
        if not probe_requires_visual(probe):
            text_only += 1
            continue
        checked += 1
        if load_image(probe) is None:
            missing_count += 1
            if len(missing) < limit_examples:
                missing.append(
                    {
                        "sample_id": probe.get("sample_id", probe.get("id", "")),
                        "probe": probe.get("probe", ""),
                        "rice_view": probe.get("rice_view", ""),
                        "image": probe.get("image", probe.get("image_path", "")),
                        "image_root": probe.get("image_root", ""),
                    }
                )
    return {
        "num_probes": len(probes),
        "num_text_only_probes": text_only,
        "num_visual_probes": checked,
        "missing_visual_count": missing_count,
        "missing_visual_rate": missing_count / checked if checked else 0.0,
        "missing_examples": [item for item in missing if item],
    }


def load_image(sample: dict[str, Any]) -> Image.Image | None:
    image = sample.get("image")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, dict):
        if image.get("path"):
            return open_image_path(str(image["path"]), sample)
        if image.get("bytes") is not None:
            return Image.open(io.BytesIO(image["bytes"])).convert("RGB")
    if isinstance(image, str) and image:
        resolved = resolve_path(image, sample)
        if os.path.exists(resolved):
            return Image.open(resolved).convert("RGB")
        try:
            return Image.open(io.BytesIO(base64.b64decode(image))).convert("RGB")
        except Exception:
            pass
    if sample.get("image_path"):
        return open_image_path(str(sample["image_path"]), sample)
    if sample.get("image_base64"):
        return Image.open(io.BytesIO(base64.b64decode(str(sample["image_base64"])))).convert("RGB")
    return None


def _missing_image_message(sample: dict[str, Any]) -> str:
    image = sample.get("image", sample.get("image_path", ""))
    resolved = resolve_path(str(image), sample) if isinstance(image, str) and image else ""
    return (
        "RECAP visual probe could not load its image. "
        f"sample_id={sample.get('sample_id', sample.get('id', ''))}, "
        f"probe={sample.get('probe', '')}, rice_view={sample.get('rice_view', '')}, "
        f"image={image!r}, resolved={resolved!r}, image_root={sample.get('image_root', '')!r}. "
        "Fix --coco-root/--image-dir/--image-root, or pass --allow-missing-images only for debugging."
    )


def open_image_path(path: str, sample: dict[str, Any]) -> Image.Image:
    return Image.open(resolve_path(path, sample)).convert("RGB")


def resolve_path(path: str, sample: dict[str, Any]) -> str:
    path = os.path.expandvars(os.path.expanduser(path))
    if os.path.isabs(path):
        return path
    image_root = sample.get("image_root") or os.getenv("RECAP_IMAGE_ROOT")
    if image_root:
        return os.path.join(os.path.expandvars(os.path.expanduser(str(image_root))), path)
    return path


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


def mask_bbox(image: Image.Image, bbox: list[float] | None) -> Image.Image:
    out = image.copy().convert("RGB")
    box = to_pixels(bbox, out.width, out.height)
    if box is not None:
        ImageDraw.Draw(out).rectangle(box, fill=(127, 127, 127))
    return out


def mask_regions(image: Image.Image, regions: Any) -> Image.Image:
    out = image.copy().convert("RGB")
    if not isinstance(regions, (list, tuple)):
        return out
    draw = ImageDraw.Draw(out)
    for region in regions:
        box = to_pixels(parse_bbox(region), out.width, out.height)
        if box is not None:
            draw.rectangle(box, fill=(127, 127, 127))
    return out


def crop_union(image: Image.Image, subject_bbox: list[float] | None, object_bbox: list[float] | None, padding: float = 0.15) -> Image.Image:
    subject = to_pixels(subject_bbox, image.width, image.height)
    obj = to_pixels(object_bbox, image.width, image.height)
    if subject is None or obj is None:
        return image.copy().convert("RGB")
    x1 = min(subject[0], obj[0])
    y1 = min(subject[1], obj[1])
    x2 = max(subject[2], obj[2])
    y2 = max(subject[3], obj[3])
    pad_x = int(round((x2 - x1) * padding))
    pad_y = int(round((y2 - y1) * padding))
    return image.crop((max(0, x1 - pad_x), max(0, y1 - pad_y), min(image.width, x2 + pad_x), min(image.height, y2 + pad_y))).convert("RGB")


def draw_relation_boxes(image: Image.Image, subject_bbox: list[float] | None, object_bbox: list[float] | None) -> Image.Image:
    out = image.copy().convert("RGB")
    line_width = max(3, min(out.width, out.height) // 120)
    draw = ImageDraw.Draw(out)
    _draw_labeled_box(draw, to_pixels(object_bbox, out.width, out.height), outline=(0, 96, 255), label="B", width=line_width)
    _draw_labeled_box(draw, to_pixels(subject_bbox, out.width, out.height), outline=(255, 32, 32), label="A", width=line_width)
    return out


def _draw_labeled_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int] | None,
    *,
    outline: tuple[int, int, int],
    label: str,
    width: int,
) -> None:
    if box is None:
        return
    draw.rectangle(box, outline=outline, width=width)
    x1, y1, _, _ = box
    label_pad = max(2, width // 2)
    label_size = max(14, width * 5)
    label_box = (x1, y1, min(x1 + label_size, box[2]), min(y1 + label_size, box[3]))
    draw.rectangle(label_box, fill=outline)
    draw.text((label_box[0] + label_pad, label_box[1] + label_pad), label, fill=(255, 255, 255))
