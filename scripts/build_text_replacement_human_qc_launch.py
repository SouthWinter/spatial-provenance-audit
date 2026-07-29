#!/usr/bin/env python3
"""Build a human-QC launch package for TextOCR text-replacement edits."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "problem_optimization_audit" / "text_replacement_human_qc_launch"
PAIRS_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "text_replacement_stratified"
    / "qwen_hq49_joined_rows.csv"
)
MANIFEST_JSONL = ROOT / "data" / "textocr_counterfactual" / "textocr_hard_replace_hq50_manifest.jsonl"

QC_FIELDS = [
    "human_source_visible_original",
    "human_source_readable_sham",
    "human_replacement_readable_edited",
    "human_source_absent_edited",
    "human_source_absent_erase",
    "human_replacement_absent_erase",
    "human_local_edit_plausible",
    "human_no_unrelated_text_changed",
    "qc_decision",
    "invalid_reason",
    "annotator_id",
    "annotator_notes",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-csv", default=str(PAIRS_CSV))
    parser.add_argument("--manifest", default=str(MANIFEST_JSONL))
    parser.add_argument(
        "--ocr-rows",
        default="",
        help="Optional control-pack OCR rows CSV; enables pre-model four-variant QC launch mode.",
    )
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument(
        "--require-auto-replacement-pass",
        action="store_true",
        help="Include only rows where replacement, sham, and erase crop OCR checks pass.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = read_jsonl(Path(args.manifest))
    manifest = {row["image_id"]: row for row in manifest_rows}
    if args.ocr_rows:
        quality = {row["image_id"]: row for row in read_csv(Path(args.ocr_rows))}
        rows = [
            build_control_row(item, quality.get(item["image_id"], {}))
            for item in manifest_rows
        ]
        if args.require_auto_replacement_pass:
            rows = [row for row in rows if row["auto_control_render_pass"] == "True"]
    else:
        pairs = read_csv(Path(args.pairs_csv))
        rows = [build_row(row, manifest[row["image_id"]]) for row in pairs if row["image_id"] in manifest]
    rows = sorted(rows, key=lambda row: (row["auto_priority"], row["image_id"]))
    template = [{**row, **{field: "" for field in QC_FIELDS}} for row in rows]
    summary = build_summary(rows)

    write_csv(output_dir / "text_replacement_human_qc_manifest.csv", rows)
    write_csv(output_dir / "text_replacement_human_qc_template.csv", template)
    write_jsonl(output_dir / "text_replacement_human_qc_seed.jsonl", template)
    write_csv(output_dir / "text_replacement_human_qc_launch_summary.csv", summary)
    (output_dir / "annotator_handbook.md").write_text(build_handbook(), encoding="utf-8")
    (output_dir / "text_replacement_human_qc_tool.html").write_text(build_html(template), encoding="utf-8")
    (output_dir / "commands.sh").write_text(build_commands(output_dir), encoding="utf-8")
    print(f"Wrote text-replacement human QC launch package to {output_dir}")


def build_control_row(manifest: dict[str, Any], quality: dict[str, str]) -> dict[str, Any]:
    original_image = str(manifest.get("source_image", ""))
    edited_image = str(manifest.get("replacement_image", manifest.get("edited_image", "")))
    sham_image = str(manifest.get("sham_image", ""))
    erase_image = str(manifest.get("erase_image", ""))
    render_pass = all(
        quality.get(field) == "True"
        for field in ("edited_crop_ocr_success", "sham_crop_ocr_success", "erase_crop_ocr_success")
    )
    return {
        "image_id": manifest.get("image_id", ""),
        "source_sample_id": manifest.get("source_sample_id", ""),
        "negative_sample_id": manifest.get("negative_sample_id", ""),
        "original_image": original_image,
        "sham_image": sham_image,
        "edited_image": edited_image,
        "erase_image": erase_image,
        "source_text": manifest.get("source_text", ""),
        "replacement_text": manifest.get("replacement_text", ""),
        "box_px_json": json.dumps(manifest.get("box_px", []), ensure_ascii=True),
        "font_size": manifest.get("font_size", ""),
        "box_width_px": manifest.get("box_width_px", ""),
        "box_height_px": manifest.get("box_height_px", ""),
        "auto_priority": "control_pack_auto_pass" if render_pass else "control_pack_boundary",
        "auto_control_render_pass": str(render_pass),
        "ocr_source_crop_text": quality.get("source_crop_text", ""),
        "ocr_sham_crop_text": quality.get("sham_crop_text", ""),
        "ocr_edited_crop_text": quality.get("edited_crop_text", ""),
        "ocr_erase_crop_text": quality.get("erase_crop_text", ""),
        "original_image_exists": str(resolve_path(original_image).exists()),
        "sham_image_exists": str(resolve_path(sham_image).exists()),
        "edited_image_exists": str(resolve_path(edited_image).exists()),
        "erase_image_exists": str(resolve_path(erase_image).exists()),
    }


def build_row(pair: dict[str, str], manifest: dict[str, Any]) -> dict[str, Any]:
    edited_image = str(pair.get("edited_image", ""))
    original_image = str(pair.get("original_image", ""))
    priority = auto_priority(pair)
    return {
        "image_id": pair["image_id"],
        "source_sample_id": pair.get("source_sample_id", ""),
        "negative_sample_id": pair.get("negative_sample_id", ""),
        "original_image": original_image,
        "edited_image": edited_image,
        "source_text": pair.get("source_text", ""),
        "replacement_text": pair.get("replacement_text", ""),
        "box_px_json": json.dumps(manifest.get("box_px", []), ensure_ascii=True),
        "font_size": pair.get("font_size", ""),
        "box_width_px": pair.get("box_width_px", ""),
        "box_height_px": pair.get("box_height_px", ""),
        "auto_priority": priority,
        "original_pair_correct": pair.get("original_pair_correct", ""),
        "edited_pair_correct": pair.get("edited_pair_correct", ""),
        "full_semantic_switch": pair.get("full_semantic_switch", ""),
        "replacement_presence_switch": pair.get("replacement_presence_switch", ""),
        "source_absence_switch": pair.get("source_absence_switch", ""),
        "ocr_edited_crop_replacement_detected": pair.get("ocr_edited_crop_replacement_detected", ""),
        "ocr_edited_crop_source_absent": pair.get("ocr_edited_crop_source_absent", ""),
        "ocr_edited_crop_text": pair.get("ocr_edited_crop_text", ""),
        "source_yes_support_drop": pair.get("source_yes_support_drop", ""),
        "replacement_yes_support_gain": pair.get("replacement_yes_support_gain", ""),
        "original_image_exists": str(Path(original_image).exists()),
        "edited_image_exists": str((ROOT / edited_image).exists() if not Path(edited_image).is_absolute() else Path(edited_image).exists()),
    }


def auto_priority(row: dict[str, str]) -> str:
    if row.get("full_semantic_switch") == "True":
        return "switch_success"
    if row.get("ocr_edited_crop_replacement_detected") == "True":
        return "ocr_readable_no_full_switch"
    if row.get("original_pair_correct") == "True" and row.get("edited_pair_correct") != "True":
        return "model_lost_after_edit"
    return "renderer_or_model_boundary"


def build_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    priorities = sorted({row["auto_priority"] for row in rows})
    out = [
        {"metric": "qc_manifest_rows", "value": len(rows)},
        {"metric": "original_image_paths_exist", "value": sum(row["original_image_exists"] == "True" for row in rows)},
        {"metric": "edited_image_paths_exist", "value": sum(row["edited_image_exists"] == "True" for row in rows)},
        {"metric": "sham_image_paths_exist", "value": sum(row.get("sham_image_exists") == "True" for row in rows)},
        {"metric": "erase_image_paths_exist", "value": sum(row.get("erase_image_exists") == "True" for row in rows)},
        {"metric": "automatic_control_render_pass_rows", "value": sum(row.get("auto_control_render_pass") == "True" for row in rows)},
        {"metric": "human_qc_completed_rows", "value": 0},
        {"metric": "paper_claim_status", "value": "launch_ready_not_human_verified"},
    ]
    for priority in priorities:
        out.append({"metric": f"priority_{priority}_rows", "value": sum(row["auto_priority"] == priority for row in rows)})
    return out


def build_handbook() -> str:
    return """# Text-Replacement Human QC Handbook

Goal: decide whether each generated text-replacement image is a valid semantic counterfactual.

For each row, inspect the original and edited image around the highlighted box.

Required fields:

- `human_source_visible_original`: yes/no/unclear. Is the source text visible in the original image?
- `human_source_readable_sham`: yes/no/unclear. Does the sham edit preserve the source reading?
- `human_replacement_readable_edited`: yes/no/unclear. Is the replacement text readable in the edited image?
- `human_source_absent_edited`: yes/no/unclear. Is the original source text absent from the edited local region?
- `human_source_absent_erase`: yes/no/unclear. Is the source absent from the erase control?
- `human_replacement_absent_erase`: yes/no/unclear. Is the replacement absent from the erase control?
- `human_local_edit_plausible`: yes/no/unclear. Does the edit look visually plausible enough for a semantic counterfactual?
- `human_no_unrelated_text_changed`: yes/no/unclear. Are unrelated regions effectively unchanged?

Set `qc_decision` to:

- `valid_semantic_edit` only when all required checks are yes.
- `invalid_unreadable_replacement` when the replacement cannot be read.
- `invalid_source_not_visible` when the original source text is not visible.
- `invalid_source_still_visible` when the source text remains visible after editing.
- `invalid_bad_local_edit` when the local rendering is too implausible.
- `invalid_unrelated_change` when unrelated content changes materially.
- `unclear` when the image is too ambiguous.

This package is not paper evidence until all rows are completed and the progress audit reports ready.
"""


def build_html(rows: list[dict[str, Any]]) -> str:
    cards = []
    for row in rows:
        original = abs_url(row["original_image"])
        edited = abs_url(row["edited_image"])
        sham = abs_url(row.get("sham_image", "")) if row.get("sham_image") else ""
        erase = abs_url(row.get("erase_image", "")) if row.get("erase_image") else ""
        box = html.escape(row["box_px_json"])
        figures = [
            f'<figure><figcaption>Original</figcaption><div class="imgbox" data-box="{box}"><img src="{original}"></div></figure>',
        ]
        if sham:
            figures.append(f'<figure><figcaption>Sham</figcaption><div class="imgbox" data-box="{box}"><img src="{sham}"></div></figure>')
        figures.append(f'<figure><figcaption>Replacement</figcaption><div class="imgbox" data-box="{box}"><img src="{edited}"></div></figure>')
        if erase:
            figures.append(f'<figure><figcaption>Erase</figcaption><div class="imgbox" data-box="{box}"><img src="{erase}"></div></figure>')
        cards.append(
            "<section class=\"card\">"
            f"<h2>{html.escape(row['image_id'])}: {html.escape(row['source_text'])} -> {html.escape(row['replacement_text'])}</h2>"
            f"<p>priority={html.escape(row['auto_priority'])}; OCR crop={html.escape(str(row.get('ocr_edited_crop_text', '')))}; full_switch={html.escape(str(row.get('full_semantic_switch', 'not-run')))}</p>"
            "<div class=\"pair\">" + "".join(figures) + "</div>"
            "</section>"
        )
    return """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Text Replacement Human QC</title>
<style>
body { font-family: Arial, sans-serif; margin: 20px; background: #f7f7f7; }
.card { background: white; border: 1px solid #ccc; margin: 16px 0; padding: 12px; }
.pair { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
.imgbox { position: relative; display: inline-block; max-width: 100%; }
img { max-width: 100%; max-height: 360px; display: block; }
.box { position: absolute; border: 3px solid #ff3b30; box-sizing: border-box; pointer-events: none; }
figcaption { font-weight: bold; margin-bottom: 6px; }
</style>
</head>
<body>
<h1>Text Replacement Human QC</h1>
<p>Use the CSV template for decisions. Red boxes show the edited/source word region.</p>
""" + "\n".join(cards) + """
<script>
function addBoxes() {
  for (const wrap of document.querySelectorAll('.imgbox')) {
    const img = wrap.querySelector('img');
    const box = JSON.parse(wrap.dataset.box || '[]');
    if (!img || box.length !== 4 || img.dataset.boxReady) continue;
    const place = () => {
      const scaleX = img.clientWidth / img.naturalWidth;
      const scaleY = img.clientHeight / img.naturalHeight;
      const b = document.createElement('div');
      b.className = 'box';
      b.style.left = (box[0] * scaleX) + 'px';
      b.style.top = (box[1] * scaleY) + 'px';
      b.style.width = ((box[2] - box[0]) * scaleX) + 'px';
      b.style.height = ((box[3] - box[1]) * scaleY) + 'px';
      wrap.appendChild(b);
      img.dataset.boxReady = '1';
    };
    if (img.complete) place(); else img.addEventListener('load', place);
  }
}
addBoxes();
</script>
</body>
</html>
"""


def abs_url(path: str) -> str:
    p = resolve_path(path)
    return "file://" + str(p)


def resolve_path(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def build_commands(output_dir: Path) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

QC_EXPORT=${{QC_EXPORT:-{output_dir}/text_replacement_human_qc_template.csv}}

python scripts/audit_text_replacement_human_qc_progress.py \\
  --qc-export "$QC_EXPORT" \\
  --output-dir runs/problem_optimization_audit/text_replacement_human_qc_progress
"""


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
