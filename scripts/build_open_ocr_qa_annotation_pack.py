#!/usr/bin/env python3
"""Export images for the open OCR/document QA stress annotation manifest.

The previous manifest fixes which TextVQA-lite/DocVQA-lite cases should be
inspected. This script resolves those sample IDs back to dataset images, writes
local JPEGs, and emits CSV/JSONL/Markdown files with absolute image paths and
blank evidence-annotation fields.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

MANIFEST_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_stress_manifest"
    / "open_ocr_qa_stress_manifest.csv"
)

DATASET_SPECS = {
    "TextVQA-lite": {
        "task": "textvqa_val_lite",
        "dataset_path": "lmms-lab/LMMs-Eval-Lite",
        "dataset_name": "textvqa_val",
        "split": "lite",
    },
    "DocVQA-lite": {
        "task": "docvqa_val_lite",
        "dataset_path": "lmms-lab/LMMs-Eval-Lite",
        "dataset_name": "docvqa_val",
        "split": "lite",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(MANIFEST_CSV))
    parser.add_argument("--output-dir", default="runs/problem_optimization_audit/open_ocr_qa_annotation_pack")
    parser.add_argument("--image-format", default="JPEG", choices=["JPEG", "PNG"])
    parser.add_argument("--max-side", type=int, default=1600)
    args = parser.parse_args()

    manifest_rows = read_csv(Path(args.manifest))
    wanted = {(row["task"], row["question_id"]) for row in manifest_rows}
    image_lookup = load_images(wanted)

    out_dir = Path(args.output_dir)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    packed_rows = []
    missing_rows = []
    for row in manifest_rows:
        key = (row["task"], row["question_id"])
        image = image_lookup.get(key)
        if image is None:
            missing_rows.append({"task": row["task"], "question_id": row["question_id"], "sample_id": row["sample_id"]})
            continue
        rel_image_path = Path("images") / image_filename(row, args.image_format)
        abs_image_path = (out_dir / rel_image_path).resolve()
        save_image(image, abs_image_path, image_format=args.image_format, max_side=args.max_side)
        packed_rows.append(
            {
                **row,
                "image_path": str(abs_image_path),
                "relative_image_path": str(rel_image_path),
                "annotation_status": "image_exported",
                "manual_evidence_region_count": row.get("manual_evidence_region_count", ""),
                "manual_evidence_texts": row.get("manual_evidence_texts", ""),
                "manual_bbox_or_region_notes": row.get("manual_bbox_or_region_notes", ""),
            }
        )

    write_csv(out_dir / "annotation_pack.csv", packed_rows)
    write_jsonl(out_dir / "annotation_pack.jsonl", packed_rows)
    write_csv(
        out_dir / "missing_images.csv",
        missing_rows,
        fieldnames=["task", "question_id", "sample_id"],
    )
    summary = build_summary(packed_rows, missing_rows)
    write_csv(out_dir / "annotation_pack_summary.csv", summary)
    (out_dir / "annotation_pack.md").write_text(build_markdown(packed_rows, missing_rows), encoding="utf-8")
    print(f"Wrote {len(packed_rows)} annotation rows with images to {out_dir}")
    if missing_rows:
        print(f"Missing images: {len(missing_rows)}")


def load_images(wanted: set[tuple[str, str]]) -> dict[tuple[str, str], Any]:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from datasets import load_dataset

    lookup = {}
    for task, spec in DATASET_SPECS.items():
        task_ids = {question_id for task_name, question_id in wanted if task_name == task}
        if not task_ids:
            continue
        kwargs: dict[str, Any] = {}
        if spec["dataset_name"]:
            kwargs["name"] = spec["dataset_name"]
        dataset = load_dataset(spec["dataset_path"], **kwargs, split=spec["split"])
        for idx, doc in enumerate(dataset):
            question_id = str(doc.get("question_id", doc.get("questionId", doc.get("id", idx))))
            if question_id not in task_ids:
                continue
            image = doc.get("image")
            if image is not None:
                lookup[(task, question_id)] = image.convert("RGB") if hasattr(image, "convert") else image
            if len([key for key in lookup if key[0] == task]) >= len(task_ids):
                break
    return lookup


def save_image(image: Any, path: Path, *, image_format: str, max_side: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(image, "copy"):
        image = image.copy()
    if hasattr(image, "convert"):
        image = image.convert("RGB")
    width, height = image.size
    scale = min(1.0, max_side / max(width, height)) if max_side > 0 else 1.0
    if scale < 1.0:
        image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))))
    save_kwargs: dict[str, Any] = {}
    if image_format == "JPEG":
        save_kwargs.update({"quality": 92, "optimize": True})
    image.save(path, format=image_format, **save_kwargs)


def image_filename(row: dict[str, str], image_format: str) -> str:
    suffix = ".jpg" if image_format == "JPEG" else ".png"
    return f"{slug(row['task'])}_{slug(row['question_id'])}{suffix}"


def build_summary(packed_rows: list[dict[str, str]], missing_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {"scope": "all", "group": "exported", "count": len(packed_rows)},
        {"scope": "all", "group": "missing", "count": len(missing_rows)},
    ]
    for task in sorted({row["task"] for row in [*packed_rows, *missing_rows]}):
        task_rows = [row for row in packed_rows if row["task"] == task]
        rows.append({"scope": task, "group": "exported", "count": len(task_rows)})
        for reason in ("persistent_failure", "recovered_by_70", "stress_control_no_large_drop"):
            rows.append(
                {
                    "scope": task,
                    "group": reason,
                    "count": sum(reason in row["selection_reasons"].split(";") for row in task_rows),
                }
            )
    return rows


def build_markdown(packed_rows: list[dict[str, str]], missing_rows: list[dict[str, str]]) -> str:
    lines = [
        "# Open OCR/Document QA Annotation Pack",
        "",
        "This pack exports images for the fixed stress manifest. It is ready for manual evidence-region annotation, but no evidence boxes have been filled yet.",
        "",
        f"Exported rows: {len(packed_rows)}. Missing images: {len(missing_rows)}.",
        "",
        "## Preview",
        "",
        "| Image | Task | Reasons | Question | Gold | Full | 30% / 50% / 70% |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in packed_rows[:32]:
        lines.append(
            f"| ![]({row['relative_image_path']}) | {row['task']} | {row['selection_reasons']} | "
            f"{escape(row['question'])} | {escape(row['gold_answers'])} | {escape(row['full_answer'])} | "
            f"{escape(row['pruned_0p30_answer'])} ({row['delta_0p30']}) / "
            f"{escape(row['pruned_0p50_answer'])} ({row['delta_0p50']}) / "
            f"{escape(row['pruned_0p70_answer'])} ({row['delta_0p70']}) |"
        )
    lines.extend(
        [
            "",
            "## Annotation Fields",
            "",
            "- `manual_evidence_region_count`: number of regions needed to answer the question.",
            "- `manual_evidence_texts`: visible text/value snippets that should be preserved.",
            "- `manual_bbox_or_region_notes`: free-form box coordinates or layout notes.",
            "",
            "The CSV keeps absolute `image_path` values for external annotation tools and relative paths for portable review.",
        ]
    )
    return "\n".join(lines) + "\n"


def slug(text: str) -> str:
    out = re.sub(r"[^a-zA-Z0-9]+", "_", str(text)).strip("_").lower()
    return out or "item"


def escape(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        if fieldnames:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        else:
            path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
