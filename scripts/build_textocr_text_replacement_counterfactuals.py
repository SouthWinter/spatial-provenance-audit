#!/usr/bin/env python3
"""Build TextOCR-Hard text-replacement counterfactual probes.

The generated images replace the positive hard target word with its paired
near-miss negative string. Each edited image yields two yes/no probes:

1. the original source word should no longer be present;
2. the inserted near-miss string should now be present.

With ``--paired-controls``, the script renders sham, replacement, and erase
variants from the same locally inpainted image. This controls for rendering
artifacts when measuring whether a model follows the edited text semantics.
The resulting subset must still pass OCR and human quality control before it is
used in paper claims.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageStat


DEFAULT_FONT = "/usr/share/fonts/dejavu/DejaVuSans.ttf"
FONT_CANDIDATES = (
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/dejavu/DejaVuSansMono-Bold.ttf",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/textocr_val_hard_probes_500img.jsonl")
    parser.add_argument("--output", required=True, help="Output counterfactual probe JSONL.")
    parser.add_argument("--image-dir", required=True, help="Directory for edited images.")
    parser.add_argument("--manifest", default="", help="Optional JSONL edit manifest.")
    parser.add_argument("--summary", default="", help="Optional JSON summary.")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--min-box-width", type=float, default=20.0)
    parser.add_argument("--min-box-height", type=float, default=12.0)
    parser.add_argument("--pad-frac", type=float, default=0.20)
    parser.add_argument("--font", default=DEFAULT_FONT)
    parser.add_argument(
        "--paired-controls",
        action="store_true",
        help="Render sham, replacement, and erase variants from one erased image.",
    )
    parser.add_argument(
        "--require-same-length",
        action="store_true",
        help="Keep only source/replacement pairs with equal character counts.",
    )
    args = parser.parse_args()

    rows = read_jsonl(Path(args.input))
    pairs = paired_hard_rows(rows)
    image_dir = Path(args.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    probes: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}

    for pos, neg in pairs:
        if args.limit and len(manifest) >= args.limit:
            break
        reason = skip_reason(
            pos,
            neg,
            min_width=args.min_box_width,
            min_height=args.min_box_height,
            require_same_length=args.require_same_length,
        )
        if reason:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        source_text = str(pos["target_text"])
        replacement = str(neg["target_text"])
        if args.paired_controls:
            edit = render_control_pack(
                pos,
                source_text=source_text,
                replacement=replacement,
                image_dir=image_dir,
                preferred_font=args.font,
                pad_frac=args.pad_frac,
            )
        else:
            edit = render_replacement(
                pos,
                replacement=replacement,
                image_dir=image_dir,
                font_path=args.font,
                pad_frac=args.pad_frac,
            )
        if edit.get("error"):
            skipped[str(edit["error"])] = skipped.get(str(edit["error"]), 0) + 1
            continue
        common = {
            "dataset": "TextOCR-HardTextReplace",
            "source_dataset": pos.get("source_dataset", "TextOCR"),
            "image_id": pos["image_id"],
            "source_image": pos["image"],
            "source_sample_id": pos["sample_id"],
            "negative_sample_id": neg["sample_id"],
            "task_family": "ocr_text_counterfactual",
            "relation": "ocr_exact_text_visible_counterfactual",
            "base_relation": "ocr_exact_text_visible_counterfactual",
            "bbox_source": pos.get("bbox_source", "textocr_word_annotations"),
            "has_bbox": True,
            "base_has_bbox": True,
            "evidence_regions": pos["evidence_regions"],
            "ocr_regions": pos["evidence_regions"],
            "evidence_region_count": 1,
            "source_text": source_text,
            "replacement_text": replacement,
            "edit_type": "text_replace_nearmiss",
        }
        probes.append(
            {
                **common,
                "image": edit["edited_image"],
                "id": f"{pos['sample_id']}:replace-source-now-absent",
                "sample_id": f"{pos['sample_id']}:replace-source-now-absent",
                "probe": "counterfactual_source_absent",
                "probe_count": 6 if args.paired_controls else 2,
                "edit_variant": "replacement",
                "question": exact_text_question(source_text),
                "target_text": source_text,
                "target_answer": "no",
                "answer": "no",
                "binary_polarity": "negative",
            }
        )
        probes.append(
            {
                **common,
                "image": edit["edited_image"],
                "id": f"{pos['sample_id']}:replace-nearmiss-now-present",
                "sample_id": f"{pos['sample_id']}:replace-nearmiss-now-present",
                "probe": "counterfactual_replacement_present",
                "probe_count": 6 if args.paired_controls else 2,
                "edit_variant": "replacement",
                "question": exact_text_question(replacement),
                "target_text": replacement,
                "target_answer": "yes",
                "answer": "yes",
                "binary_polarity": "positive",
            }
        )
        if args.paired_controls:
            probes.extend(control_probes(common, pos, neg, edit, source_text, replacement))
        manifest.append(
            {
                **edit,
                "source_sample_id": pos["sample_id"],
                "negative_sample_id": neg["sample_id"],
                "source_text": source_text,
                "replacement_text": replacement,
            }
        )

    write_jsonl(Path(args.output), probes)
    if args.manifest:
        write_jsonl(Path(args.manifest), manifest)
    summary = {
        "input": args.input,
        "output": args.output,
        "image_dir": args.image_dir,
        "num_pairs": len(manifest),
        "num_probes": len(probes),
        "paired_controls": bool(args.paired_controls),
        "require_same_length": bool(args.require_same_length),
        "skipped": skipped,
        "avg_box_width_px": mean([row["box_width_px"] for row in manifest]) if manifest else 0.0,
        "avg_box_height_px": mean([row["box_height_px"] for row in manifest]) if manifest else 0.0,
        "avg_font_size": mean([row["font_size"] for row in manifest]) if manifest else 0.0,
    }
    if args.summary:
        Path(args.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def paired_hard_rows(rows: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    by_image: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_image.setdefault(str(row.get("image_id", "")), []).append(row)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for image_id in sorted(by_image):
        positives = [r for r in by_image[image_id] if r.get("binary_polarity") == "positive"]
        negatives = [r for r in by_image[image_id] if r.get("binary_polarity") == "negative"]
        if not positives or not negatives:
            continue
        pairs.append((positives[0], negatives[0]))
    return pairs


def skip_reason(
    pos: dict[str, Any],
    neg: dict[str, Any],
    *,
    min_width: float,
    min_height: float,
    require_same_length: bool = False,
) -> str:
    if not pos.get("evidence_regions") or len(pos["evidence_regions"]) != 1:
        return "missing_single_source_box"
    if not Path(str(pos.get("image", ""))).exists():
        return "missing_image"
    source = str(pos.get("target_text", ""))
    replacement = str(neg.get("target_text", ""))
    if not source or not replacement:
        return "missing_text"
    if require_same_length and len(source) != len(replacement):
        return "unequal_character_count"
    if not re.search(r"[A-Za-z0-9]", source) or not re.search(r"[A-Za-z0-9]", replacement):
        return "non_alnum_text"
    with Image.open(pos["image"]) as image:
        width, height = image.size
    x0, y0, x1, y1 = pos["evidence_regions"][0]
    box_width = (float(x1) - float(x0)) * width
    box_height = (float(y1) - float(y0)) * height
    if box_width < min_width:
        return "box_too_narrow"
    if box_height < min_height:
        return "box_too_short"
    return ""


def control_probes(
    common: dict[str, Any],
    pos: dict[str, Any],
    neg: dict[str, Any],
    edit: dict[str, Any],
    source_text: str,
    replacement: str,
) -> list[dict[str, Any]]:
    variants = (
        ("sham", edit["sham_image"], source_text, "yes", "positive"),
        ("sham", edit["sham_image"], replacement, "no", "negative"),
        ("erase", edit["erase_image"], source_text, "no", "negative"),
        ("erase", edit["erase_image"], replacement, "no", "negative"),
    )
    rows: list[dict[str, Any]] = []
    for variant, image_path, target, answer, polarity in variants:
        target_role = "source" if target == source_text else "replacement"
        rows.append(
            {
                **common,
                "image": image_path,
                "id": f"{pos['sample_id']}:{variant}-{target_role}",
                "sample_id": f"{pos['sample_id']}:{variant}-{target_role}",
                "probe": f"counterfactual_{variant}_{target_role}",
                "probe_count": 6,
                "question": exact_text_question(target),
                "target_text": target,
                "target_answer": answer,
                "answer": answer,
                "binary_polarity": polarity,
                "edit_variant": variant,
                "negative_sample_id": neg["sample_id"],
            }
        )
    return rows


def render_control_pack(
    sample: dict[str, Any],
    *,
    source_text: str,
    replacement: str,
    image_dir: Path,
    preferred_font: str,
    pad_frac: float,
) -> dict[str, Any]:
    try:
        source_image = Image.open(sample["image"]).convert("RGB")
    except Exception as exc:
        return {"error": f"image_open_failed:{exc}"}
    width, height = source_image.size
    x0, y0, x1, y1 = [float(v) for v in sample["evidence_regions"][0]]
    left = max(0, int(math.floor(x0 * width)))
    top = max(0, int(math.floor(y0 * height)))
    right = min(width, int(math.ceil(x1 * width)))
    bottom = min(height, int(math.ceil(y1 * height)))
    if right <= left or bottom <= top:
        return {"error": "empty_box"}

    box_w = right - left
    box_h = bottom - top
    pad_x = max(1, int(round(box_w * pad_frac)))
    pad_y = max(1, int(round(box_h * pad_frac)))
    original_crop = source_image.crop((left, top, right, bottom))
    text_color = estimate_text_color(original_crop)
    erased, erase_mode = erase_text_region(source_image, left, top, right, bottom, pad_x, pad_y)
    font_path, font_size = choose_font(
        source_text,
        replacement,
        preferred_font=preferred_font,
        max_width=max(4, box_w - 2),
        max_height=max(4, box_h - 2),
    )
    sham = draw_antialiased_text(erased, source_text, (left, top, right, bottom), font_path, font_size, text_color)
    replacement_image = draw_antialiased_text(
        erased,
        replacement,
        (left, top, right, bottom),
        font_path,
        font_size,
        text_color,
    )

    stem = f"{sample['image_id']}_{sanitize(source_text)}_to_{sanitize(replacement)}"
    erase_path = image_dir / f"{stem}_erase.png"
    sham_path = image_dir / f"{stem}_sham.png"
    replacement_path = image_dir / f"{stem}_replace.png"
    erased.save(erase_path, compress_level=3)
    sham.save(sham_path, compress_level=3)
    replacement_image.save(replacement_path, compress_level=3)
    return {
        "edited_image": str(replacement_path),
        "replacement_image": str(replacement_path),
        "sham_image": str(sham_path),
        "erase_image": str(erase_path),
        "source_image": sample["image"],
        "image_id": sample["image_id"],
        "box_px": [left, top, right, bottom],
        "box_width_px": box_w,
        "box_height_px": box_h,
        "font_path": font_path,
        "font_size": font_size,
        "text_rgb": text_color,
        "erase_mode": erase_mode,
        "control_pack": True,
    }


def render_replacement(
    sample: dict[str, Any],
    *,
    replacement: str,
    image_dir: Path,
    font_path: str,
    pad_frac: float,
) -> dict[str, Any]:
    try:
        image = Image.open(sample["image"]).convert("RGB")
    except Exception as exc:
        return {"error": f"image_open_failed:{exc}"}
    width, height = image.size
    x0, y0, x1, y1 = [float(v) for v in sample["evidence_regions"][0]]
    left = max(0, int(math.floor(x0 * width)))
    top = max(0, int(math.floor(y0 * height)))
    right = min(width, int(math.ceil(x1 * width)))
    bottom = min(height, int(math.ceil(y1 * height)))
    if right <= left or bottom <= top:
        return {"error": "empty_box"}

    box_w = right - left
    box_h = bottom - top
    pad_x = max(1, int(round(box_w * pad_frac)))
    pad_y = max(1, int(round(box_h * pad_frac)))
    fill = estimate_background(image, left, top, right, bottom, pad_x, pad_y)
    text_color = estimate_text_color(image.crop((left, top, right, bottom)))

    draw = ImageDraw.Draw(image)
    draw.rectangle((left, top, right, bottom), fill=fill)
    font_size = fit_font_size(replacement, font_path, max(4, box_w - 2 * pad_x), max(4, box_h - 2 * pad_y))
    font = ImageFont.truetype(font_path, font_size)
    bbox = draw.textbbox((0, 0), replacement, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    text_x = left + max(0, (box_w - text_w) // 2) - bbox[0]
    text_y = top + max(0, (box_h - text_h) // 2) - bbox[1]
    draw.text((text_x, text_y), replacement, fill=text_color, font=font)

    out_name = f"{sample['image_id']}_{sanitize(replacement)}_replace.jpg"
    out_path = image_dir / out_name
    image.save(out_path, quality=95)
    return {
        "edited_image": str(out_path),
        "source_image": sample["image"],
        "image_id": sample["image_id"],
        "box_px": [left, top, right, bottom],
        "box_width_px": box_w,
        "box_height_px": box_h,
        "font_size": font_size,
        "fill_rgb": fill,
        "text_rgb": text_color,
    }


def erase_text_region(
    image: Image.Image,
    left: int,
    top: int,
    right: int,
    bottom: int,
    pad_x: int,
    pad_y: int,
) -> tuple[Image.Image, str]:
    """Erase one tight word box while preserving nearby texture when possible."""
    try:
        import cv2
        import numpy as np

        rgb = np.asarray(image.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
        mask[top:bottom, left:right] = 255
        radius = max(2, min(9, int(round(min(right - left, bottom - top) * 0.30))))
        repaired = cv2.inpaint(bgr, mask, radius, cv2.INPAINT_TELEA)
        return Image.fromarray(cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB)), "opencv_telea"
    except Exception:
        fallback = image.copy()
        fill = estimate_background(fallback, left, top, right, bottom, pad_x, pad_y)
        ImageDraw.Draw(fallback).rectangle((left, top, right - 1, bottom - 1), fill=fill)
        return fallback, "median_ring_fill"


def choose_font(
    source_text: str,
    replacement: str,
    *,
    preferred_font: str,
    max_width: int,
    max_height: int,
) -> tuple[str, int]:
    candidates = [preferred_font, *FONT_CANDIDATES]
    unique_candidates: list[str] = []
    for candidate in candidates:
        if candidate not in unique_candidates and Path(candidate).is_file():
            unique_candidates.append(candidate)
    if not unique_candidates:
        raise FileNotFoundError("No usable TrueType font was found.")
    best_font = unique_candidates[0]
    best_size = 4
    for candidate in unique_candidates:
        size = min(
            fit_font_size(source_text, candidate, max_width, max_height),
            fit_font_size(replacement, candidate, max_width, max_height),
        )
        if size > best_size:
            best_font = candidate
            best_size = size
    return best_font, best_size


def draw_antialiased_text(
    image: Image.Image,
    text: str,
    box: tuple[int, int, int, int],
    font_path: str,
    font_size: int,
    color: tuple[int, int, int],
    *,
    scale: int = 4,
) -> Image.Image:
    left, top, right, bottom = box
    box_w = max(1, right - left)
    box_h = max(1, bottom - top)
    overlay = Image.new("RGBA", (box_w * scale, box_h * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.truetype(font_path, max(4, font_size * scale))
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = max(0, (overlay.width - text_w) // 2) - bbox[0]
    y = max(0, (overlay.height - text_h) // 2) - bbox[1]
    draw.text((x, y), text, fill=(*color, 255), font=font)
    resampling = getattr(Image, "Resampling", Image)
    overlay = overlay.resize((box_w, box_h), resample=resampling.LANCZOS)
    result = image.convert("RGBA")
    result.alpha_composite(overlay, dest=(left, top))
    return result.convert("RGB")


def estimate_background(image: Image.Image, left: int, top: int, right: int, bottom: int, pad_x: int, pad_y: int) -> tuple[int, int, int]:
    width, height = image.size
    outer = (
        max(0, left - pad_x),
        max(0, top - pad_y),
        min(width, right + pad_x),
        min(height, bottom + pad_y),
    )
    crop = image.crop(outer)
    stat = ImageStat.Stat(crop)
    return tuple(int(v) for v in stat.median[:3])


def estimate_text_color(crop: Image.Image) -> tuple[int, int, int]:
    gray = crop.convert("L")
    stat = ImageStat.Stat(gray)
    median = float(stat.median[0])
    return (20, 20, 20) if median >= 128.0 else (235, 235, 235)


def fit_font_size(text: str, font_path: str, max_width: int, max_height: int) -> int:
    max_size = max(4, min(max_height + 6, 64))
    dummy = Image.new("RGB", (max_width + 20, max_height + 20))
    draw = ImageDraw.Draw(dummy)
    for size in range(max_size, 3, -1):
        font = ImageFont.truetype(font_path, size)
        bbox = draw.textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= max_width and bbox[3] - bbox[1] <= max_height:
            return size
    return 4


def exact_text_question(text: str) -> str:
    return f'Does the image contain the exact text "{text}"? Answer yes or no.'


def sanitize(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return cleaned[:40] or "text"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
