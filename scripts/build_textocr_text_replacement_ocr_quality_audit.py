#!/usr/bin/env python3
"""Audit OCR readability of TextOCR text-replacement counterfactual images."""

from __future__ import annotations

import argparse
import csv
import json
import re
import tempfile
import time
from pathlib import Path
from statistics import mean
from typing import Any

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Counterfactual edit manifest JSONL.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default="text_replacement_ocr_quality")
    parser.add_argument("--languages", default="en")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--crop-pad-frac", type=float, default=0.75)
    parser.add_argument("--cache", default="")
    args = parser.parse_args()

    manifest_rows = read_jsonl(Path(args.manifest))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache) if args.cache else output_dir / f"{args.name}_easyocr_cache.json"
    cache = read_json(cache_path) if cache_path.exists() else {}

    reader = None
    rows: list[dict[str, Any]] = []
    for item in manifest_rows:
        if reader is None:
            import cv2  # noqa: F401
            import easyocr

            reader = easyocr.Reader(
                [lang.strip() for lang in args.languages.split(",") if lang.strip()],
                gpu=bool(args.gpu),
                verbose=False,
            )
        rows.append(
            audit_one(
                reader=reader,
                item=item,
                min_confidence=args.min_confidence,
                crop_pad_frac=args.crop_pad_frac,
                cache=cache,
            )
        )
        write_json(cache_path, cache)

    summary = summarize(rows, manifest=str(args.manifest), name=args.name)
    write_csv(output_dir / f"{args.name}_rows.csv", rows)
    write_csv(output_dir / f"{args.name}_summary.csv", [summary])
    write_json(output_dir / f"{args.name}_summary.json", summary)
    (output_dir / f"{args.name}_report.md").write_text(render_markdown(summary, rows), encoding="utf-8")
    print(f"Wrote OCR-quality audit to {output_dir}")


def audit_one(
    *,
    reader: Any,
    item: dict[str, Any],
    min_confidence: float,
    crop_pad_frac: float,
    cache: dict[str, Any],
) -> dict[str, Any]:
    edited_image = str(item["edited_image"])
    source_image = str(item.get("source_image", ""))
    source_text = str(item.get("source_text", ""))
    replacement_text = str(item.get("replacement_text", ""))
    box_px = [int(v) for v in item["box_px"]]

    edited_full = detect_cached(reader, Path(edited_image), min_confidence, cache, "edited_full")
    edited_crop = detect_crop_cached(
        reader,
        Path(edited_image),
        box_px,
        min_confidence,
        crop_pad_frac,
        cache,
        "edited_crop",
    )
    source_crop = detect_crop_cached(
        reader,
        Path(source_image),
        box_px,
        min_confidence,
        crop_pad_frac,
        cache,
        "source_crop",
    ) if source_image else {"tokens": [], "elapsed_sec": 0.0}

    edited_crop_text = " ".join(tok["text"] for tok in edited_crop["tokens"])
    edited_full_text = " ".join(tok["text"] for tok in edited_full["tokens"])
    source_crop_text = " ".join(tok["text"] for tok in source_crop["tokens"])

    crop_replacement = contains_text(edited_crop_text, replacement_text)
    crop_source = contains_text(edited_crop_text, source_text)
    full_replacement = contains_text(edited_full_text, replacement_text)
    full_source = contains_text(edited_full_text, source_text)
    source_original_detected = contains_text(source_crop_text, source_text)

    sham_crop = detect_optional_crop(
        reader,
        item.get("sham_image"),
        box_px,
        min_confidence,
        crop_pad_frac,
        cache,
        "sham_crop",
    )
    erase_crop = detect_optional_crop(
        reader,
        item.get("erase_image"),
        box_px,
        min_confidence,
        crop_pad_frac,
        cache,
        "erase_crop",
    )
    sham_crop_text = " ".join(tok["text"] for tok in sham_crop["tokens"])
    erase_crop_text = " ".join(tok["text"] for tok in erase_crop["tokens"])
    has_control_pack = bool(item.get("sham_image") and item.get("erase_image"))
    sham_source = contains_text(sham_crop_text, source_text)
    sham_replacement = contains_text(sham_crop_text, replacement_text)
    erase_source = contains_text(erase_crop_text, source_text)
    erase_replacement = contains_text(erase_crop_text, replacement_text)
    sham_success = sham_source and not sham_replacement
    erase_success = not erase_source and not erase_replacement
    replacement_success = crop_replacement and not crop_source

    return {
        "image_id": item.get("image_id", ""),
        "source_sample_id": item.get("source_sample_id", ""),
        "negative_sample_id": item.get("negative_sample_id", ""),
        "source_text": source_text,
        "replacement_text": replacement_text,
        "box_width_px": item.get("box_width_px", ""),
        "box_height_px": item.get("box_height_px", ""),
        "font_size": item.get("font_size", ""),
        "edited_crop_text": edited_crop_text,
        "edited_full_text": edited_full_text,
        "source_crop_text": source_crop_text,
        "sham_crop_text": sham_crop_text,
        "erase_crop_text": erase_crop_text,
        "has_control_pack": has_control_pack,
        "source_original_detected_in_crop": source_original_detected,
        "edited_crop_replacement_detected": crop_replacement,
        "edited_crop_source_detected": crop_source,
        "edited_crop_source_absent": not crop_source,
        "edited_crop_ocr_success": replacement_success,
        "edited_full_replacement_detected": full_replacement,
        "edited_full_source_detected": full_source,
        "edited_full_source_absent": not full_source,
        "edited_full_ocr_success": full_replacement and not full_source,
        "edited_crop_num_tokens": len(edited_crop["tokens"]),
        "edited_full_num_tokens": len(edited_full["tokens"]),
        "source_crop_num_tokens": len(source_crop["tokens"]),
        "sham_crop_source_detected": sham_source,
        "sham_crop_replacement_detected": sham_replacement,
        "sham_crop_ocr_success": sham_success,
        "erase_crop_source_detected": erase_source,
        "erase_crop_replacement_detected": erase_replacement,
        "erase_crop_ocr_success": erase_success,
        "control_pack_ocr_success": (
            source_original_detected and sham_success and replacement_success and erase_success
            if has_control_pack
            else False
        ),
        "edited_crop_elapsed_sec": f"{float(edited_crop.get('elapsed_sec', 0.0)):.6f}",
        "edited_full_elapsed_sec": f"{float(edited_full.get('elapsed_sec', 0.0)):.6f}",
        "source_crop_elapsed_sec": f"{float(source_crop.get('elapsed_sec', 0.0)):.6f}",
        "edited_image": edited_image,
        "source_image": source_image,
        "sham_image": item.get("sham_image", ""),
        "erase_image": item.get("erase_image", ""),
    }


def detect_optional_crop(
    reader: Any,
    image_path: Any,
    box_px: list[int],
    min_confidence: float,
    crop_pad_frac: float,
    cache: dict[str, Any],
    namespace: str,
) -> dict[str, Any]:
    if not image_path:
        return {"tokens": [], "elapsed_sec": 0.0}
    path = Path(str(image_path))
    if not path.is_file():
        return {"tokens": [], "elapsed_sec": 0.0}
    return detect_crop_cached(
        reader,
        path,
        box_px,
        min_confidence,
        crop_pad_frac,
        cache,
        namespace,
    )


def detect_cached(
    reader: Any,
    image_path: Path,
    min_confidence: float,
    cache: dict[str, Any],
    namespace: str,
) -> dict[str, Any]:
    key = f"{namespace}:{image_path}:{min_confidence}"
    if key not in cache:
        cache[key] = detect_image(reader, image_path, min_confidence=min_confidence)
    return cache[key]


def detect_crop_cached(
    reader: Any,
    image_path: Path,
    box_px: list[int],
    min_confidence: float,
    crop_pad_frac: float,
    cache: dict[str, Any],
    namespace: str,
) -> dict[str, Any]:
    key = f"{namespace}:{image_path}:{box_px}:{min_confidence}:{crop_pad_frac}"
    if key in cache:
        return cache[key]
    crop = crop_image(image_path, box_px, crop_pad_frac)
    with tempfile.NamedTemporaryFile(suffix=".jpg") as handle:
        crop.save(handle.name, quality=95)
        cache[key] = detect_image(reader, Path(handle.name), min_confidence=min_confidence)
    return cache[key]


def detect_image(reader: Any, image_path: Path, *, min_confidence: float) -> dict[str, Any]:
    start = time.perf_counter()
    raw = reader.readtext(str(image_path), detail=1, paragraph=False)
    elapsed = time.perf_counter() - start
    tokens: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        text = str(item[1])
        confidence = float(item[2])
        if confidence < min_confidence:
            continue
        tokens.append({"text": text, "confidence": confidence})
    return {"image": str(image_path), "tokens": tokens, "elapsed_sec": elapsed}


def crop_image(image_path: Path, box_px: list[int], crop_pad_frac: float) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    left, top, right, bottom = box_px
    box_w = max(1, right - left)
    box_h = max(1, bottom - top)
    pad_x = int(round(box_w * crop_pad_frac))
    pad_y = int(round(box_h * crop_pad_frac))
    crop_box = (
        max(0, left - pad_x),
        max(0, top - pad_y),
        min(width, right + pad_x),
        min(height, bottom + pad_y),
    )
    return image.crop(crop_box)


def contains_text(observed: str, target: str) -> bool:
    target_norm = normalize(target)
    observed_norm = normalize(observed)
    if not target_norm:
        return False
    return target_norm in observed_norm


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def summarize(rows: list[dict[str, Any]], *, manifest: str, name: str) -> dict[str, Any]:
    return {
        "name": name,
        "manifest": manifest,
        "n_pairs": len(rows),
        "source_original_detected_crop_rate": rate(rows, "source_original_detected_in_crop"),
        "edited_crop_replacement_detected_rate": rate(rows, "edited_crop_replacement_detected"),
        "edited_crop_source_absent_rate": rate(rows, "edited_crop_source_absent"),
        "edited_crop_ocr_success_rate": rate(rows, "edited_crop_ocr_success"),
        "edited_full_replacement_detected_rate": rate(rows, "edited_full_replacement_detected"),
        "edited_full_source_absent_rate": rate(rows, "edited_full_source_absent"),
        "edited_full_ocr_success_rate": rate(rows, "edited_full_ocr_success"),
        "sham_crop_ocr_success_rate": rate(rows, "sham_crop_ocr_success"),
        "erase_crop_ocr_success_rate": rate(rows, "erase_crop_ocr_success"),
        "control_pack_ocr_success_rate": rate(rows, "control_pack_ocr_success"),
        "mean_box_width_px": mean_numeric(rows, "box_width_px"),
        "mean_box_height_px": mean_numeric(rows, "box_height_px"),
        "mean_font_size": mean_numeric(rows, "font_size"),
        "mean_edited_crop_elapsed_sec": mean_numeric(rows, "edited_crop_elapsed_sec"),
        "mean_edited_full_elapsed_sec": mean_numeric(rows, "edited_full_elapsed_sec"),
    }


def rate(rows: list[dict[str, Any]], key: str) -> str:
    if not rows:
        return "0.000000"
    return f"{sum(bool(row.get(key)) for row in rows) / len(rows):.6f}"


def mean_numeric(rows: list[dict[str, Any]], key: str) -> str:
    values: list[float] = []
    for row in rows:
        try:
            values.append(float(row.get(key, "")))
        except Exception:
            continue
    return f"{mean(values):.6f}" if values else "0.000000"


def render_markdown(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    columns = list(summary.keys())
    lines = [
        "# TextOCR Text-Replacement OCR Quality Audit",
        "",
        "This audit runs EasyOCR on the original, replacement, sham, and erase crops. A paired control pack passes only when the original and sham preserve the source, replacement introduces only the near-miss, and erase contains neither string.",
        "",
        "## Summary",
        "",
        table_md([summary], columns),
        "",
        "## Boundary",
        "",
        "Safe claim: OCR-quality auditing separates renderer quality from model causal behavior. Unsafe claim: EasyOCR readability is human verification or photorealism.",
        "",
        "## Example Rows",
        "",
        table_md(
            rows[:20],
            [
                "image_id",
                "source_text",
                "replacement_text",
                "source_crop_text",
                "edited_crop_text",
                "sham_crop_text",
                "erase_crop_text",
                "edited_crop_replacement_detected",
                "edited_crop_source_absent",
                "edited_crop_ocr_success",
                "sham_crop_ocr_success",
                "erase_crop_ocr_success",
                "control_pack_ocr_success",
            ],
        ),
    ]
    return "\n".join(lines) + "\n"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def table_md(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "(empty)"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")).replace("|", "\\|") for col in columns) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
