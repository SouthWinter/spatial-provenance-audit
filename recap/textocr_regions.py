"""Prepare TextOCR word annotations as OCR-region evidence samples."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

from recap.prune.metrics import normalize_box

TEXTOCR_URLS = {
    "train": "https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_train.json",
    "val": "https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_val.json",
}


def download_textocr_annotations(*, split: str, output_dir: str | Path) -> Path:
    """Download official TextOCR annotation JSON if it is not already present."""
    split = _normalize_split(split)
    output_root = Path(output_dir).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / f"TextOCR_0.1_{split}.json"
    if path.exists() and path.stat().st_size > 0:
        return path
    urllib.request.urlretrieve(TEXTOCR_URLS[split], path)
    return path


def prepare_textocr_region_samples(
    *,
    annotation_file: str | Path,
    image_root: str | Path | None = None,
    limit: int | None = None,
    min_area: float = 1.0,
    max_regions: int | None = None,
) -> list[dict[str, Any]]:
    """Convert TextOCR word annotations to canonical OCR evidence samples."""
    payload = json.loads(Path(annotation_file).expanduser().read_text(encoding="utf-8"))
    imgs = payload.get("imgs", {})
    anns = payload.get("anns", {})
    img_to_anns = payload.get("imgToAnns", {})
    root = Path(image_root).expanduser() if image_root else None

    samples: list[dict[str, Any]] = []
    for image_id, image in imgs.items():
        ann_ids = img_to_anns.get(image_id, [])
        regions: list[list[float]] = []
        tokens: list[str] = []
        image_size = (float(image.get("width", 0.0)), float(image.get("height", 0.0)))
        if image_size[0] <= 0 or image_size[1] <= 0:
            image_size = None
        for ann_id in ann_ids:
            ann = anns.get(ann_id)
            if not ann:
                continue
            if float(ann.get("area", 0.0) or 0.0) < min_area:
                continue
            bbox = ann.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                continue
            x, y, w, h = [float(value) for value in bbox[:4]]
            box = normalize_box([x, y, x + w, y + h], image_size=image_size)
            if box is None:
                continue
            regions.append([box[0], box[1], box[2], box[3]])
            token = str(ann.get("utf8_string", "") or "").strip()
            if token:
                tokens.append(token)
        if max_regions is not None and len(regions) > max_regions:
            regions = regions[:max_regions]
            tokens = tokens[:max_regions]
        if not regions:
            continue
        file_name = str(image.get("file_name", f"{image_id}.jpg"))
        image_path = str((root / file_name).expanduser()) if root else file_name
        samples.append(
            {
                "id": f"textocr-{image_id}",
                "dataset": "TextOCR",
                "source_dataset": "TextOCR",
                "split": image.get("set", ""),
                "image": image_path,
                "image_id": image_id,
                "width": image.get("width"),
                "height": image.get("height"),
                "question": "What text is visible in the image?",
                "answer": " ".join(tokens[:20]),
                "task_family": "ocr_text",
                "bbox_source": "textocr_word_annotations",
                "ocr_tokens": [
                    {"text": token, "bbox": region}
                    for token, region in zip(tokens, regions)
                ],
                "ocr_regions": regions,
                "evidence_regions": regions,
                "evidence_region_count": len(regions),
            }
        )
        if limit is not None and len(samples) >= limit:
            break
    return samples


def _normalize_split(split: str) -> str:
    value = str(split or "val").strip().lower()
    if value in {"validation", "valid", "val"}:
        return "val"
    if value == "train":
        return "train"
    raise ValueError(f"Unsupported TextOCR split {split!r}; use train or val.")
