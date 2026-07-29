#!/usr/bin/env python3
"""Build human-validated replacement probes with exact rendered edit regions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACK = ROOT / "runs/problem_optimization_audit/text_replacement_control_pack_v3"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/textocr_counterfactual/text_replacement_edit_region_valid102.jsonl",
    )
    args = parser.parse_args()

    pack = args.pack.resolve()
    valid_ids = read_valid_image_ids(
        pack / "human_qc_launch/text_replacement_human_qc_completed.csv"
    )
    manifests = {
        str(row["image_id"]): row for row in read_jsonl(pack / "manifest.jsonl")
    }
    source_probes = read_jsonl(pack / "cross_backbone_eval/probes.jsonl")

    output: list[dict[str, Any]] = []
    for row in source_probes:
        image_id = str(row.get("image_id", ""))
        if (
            image_id not in valid_ids
            or row.get("probe") != "counterfactual_replacement_present"
        ):
            continue
        manifest = manifests[image_id]
        edit_box = rendered_changed_glyph_box(manifest)
        width, height = image_size(Path(manifest["replacement_image"]))
        word_regions = row.get("evidence_regions") or row.get("ocr_regions")
        if not word_regions:
            raise ValueError(f"Missing word region for {row['sample_id']}")
        normalized_edit = [
            edit_box[0] / width,
            edit_box[1] / height,
            edit_box[2] / width,
            edit_box[3] / height,
        ]
        output.append(
            {
                **row,
                "evidence_regions": [normalized_edit],
                "ocr_regions": [normalized_edit],
                "word_evidence_regions": word_regions,
                "edit_region_px": list(edit_box),
                "edit_region_source": "renderer_derived_changed_glyph_bbox",
                "human_qc_decision": "valid_semantic_edit",
            }
        )

    if len(output) != len(valid_ids):
        raise RuntimeError(
            f"Expected one replacement-positive probe for {len(valid_ids)} valid edits, "
            f"found {len(output)}."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, output)
    print(f"Wrote {len(output)} probes to {args.output}")


def read_valid_image_ids(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        str(row["image_id"])
        for row in rows
        if row.get("qc_decision") == "valid_semantic_edit"
    }


def rendered_changed_glyph_box(
    manifest: dict[str, Any], *, scale: int = 4
) -> tuple[int, int, int, int]:
    replacement_path = ROOT / str(manifest["replacement_image"])
    left, top, right, bottom = [int(value) for value in manifest["box_px"]]
    source = str(manifest["source_text"])
    replacement = str(manifest["replacement_text"])
    differing = [
        index
        for index, (source_char, replacement_char) in enumerate(
            zip(source, replacement)
        )
        if source_char != replacement_char
    ]
    if len(source) != len(replacement) or len(differing) != 1:
        raise ValueError(f"Expected one substitution for {manifest['image_id']}")

    overlay = Image.new(
        "L", (max(1, right - left) * scale, max(1, bottom - top) * scale), 0
    )
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.truetype(
        str(manifest["font_path"]), max(4, int(manifest["font_size"]) * scale)
    )
    full_bbox = draw.textbbox((0, 0), replacement, font=font)
    text_width = full_bbox[2] - full_bbox[0]
    text_height = full_bbox[3] - full_bbox[1]
    text_x = max(0, (overlay.width - text_width) // 2) - full_bbox[0]
    text_y = max(0, (overlay.height - text_height) // 2) - full_bbox[1]

    index = differing[0]
    char = replacement[index]
    prefix = replacement[:index]
    if prefix:
        char_start = (
            draw.textlength(prefix + char, font=font)
            - draw.textlength(char, font=font)
        )
    else:
        char_start = 0.0
    glyph_bbox = draw.textbbox(
        (text_x + char_start, text_y), char, font=font
    )
    width, height = image_size(replacement_path)
    return (
        max(0, left + math.floor(glyph_bbox[0] / scale) - 1),
        max(0, top + math.floor(glyph_bbox[1] / scale) - 1),
        min(width, left + math.ceil(glyph_bbox[2] / scale) + 1),
        min(height, top + math.ceil(glyph_bbox[3] / scale) + 1),
    )


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
