#!/usr/bin/env python3
"""Adapt external OCR/QA bbox annotations to the open OCR QA stress pack."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ANNOTATION_PACK_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_annotation_pack"
    / "annotation_pack.csv"
)
PREFILL_JSONL = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_evidence_prefill"
    / "evidence_prefill_pack.jsonl"
)
TEXTVQA_GT_BBOX_README = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "datasets--jrzhang--TextVQA_GT_bbox"
    / "snapshots"
    / "4a0fae0da919af44db46d93ab1a1e128daad2de6"
    / "README.md"
)

DATASET_SPECS = {
    "TextVQA-lite": {
        "dataset_path": "lmms-lab/LMMs-Eval-Lite",
        "dataset_name": "textvqa_val",
        "split": "lite",
    },
    "DocVQA-lite": {
        "dataset_path": "lmms-lab/LMMs-Eval-Lite",
        "dataset_name": "docvqa_val",
        "split": "lite",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-pack", default=str(ANNOTATION_PACK_CSV))
    parser.add_argument("--prefill", default=str(PREFILL_JSONL))
    parser.add_argument(
        "--external-bbox",
        action="append",
        default=[],
        help="External JSON/JSONL/CSV bbox file. Can be repeated. If omitted, only a no-match dry run is written.",
    )
    parser.add_argument(
        "--external-task",
        default="",
        choices=["", "TextVQA-lite", "DocVQA-lite"],
        help="Optional task filter applied to all external bbox rows, useful when question IDs overlap across datasets.",
    )
    parser.add_argument("--output-dir", default="runs/problem_optimization_audit/open_ocr_qa_external_bbox_annotations")
    parser.add_argument("--default-box-format", default="xyxy", choices=["xyxy", "xywh"])
    parser.add_argument("--require-text-match", action="store_true")
    parser.add_argument("--min-text-similarity", type=float, default=0.75)
    parser.add_argument("--skip-dataset-metadata", action="store_true")
    args = parser.parse_args()

    pack_rows = read_csv(Path(args.annotation_pack))
    prefill_rows = {row["sample_id"]: row for row in read_jsonl(Path(args.prefill))}
    metadata = {} if args.skip_dataset_metadata else load_lite_metadata(pack_rows)
    external_rows = []
    for path in args.external_bbox:
        external_rows.extend(load_external_rows(Path(path), source_path=str(path), external_task=args.external_task))
    index = build_external_index(external_rows)

    annotations: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for row in pack_rows:
        sample_id = row["sample_id"]
        meta = metadata.get(sample_id, {})
        prefill = prefill_rows.get(sample_id, {})
        image_width, image_height = image_size(Path(row["image_path"]))
        matches = find_matches(row, meta, index)
        evidence_units = str(prefill.get("prefill_evidence_units", row.get("evidence_text_candidates", "")))
        boxes = []
        source_notes = []
        for match in matches:
            parsed = boxes_from_external_row(
                match,
                export_size=(image_width, image_height),
                metadata=meta,
                default_box_format=args.default_box_format,
                evidence_units=evidence_units,
                require_text_match=args.require_text_match,
                min_text_similarity=args.min_text_similarity,
            )
            boxes.extend(parsed)
            if match.get("_source_path"):
                source_notes.append(str(match["_source_path"]))
        boxes = dedupe_output_boxes(boxes)
        status = "external_bbox_proposed" if boxes else "unmatched"
        annotations.append(
            {
                "sample_id": sample_id,
                "task": row["task"],
                "question_id": row["question_id"],
                "evidence_units": evidence_units,
                "boxes": boxes,
                "notes": "; ".join(sorted(set(source_notes))),
                "status": status,
            }
        )
        detail_rows.append(
            {
                "sample_id": sample_id,
                "task": row["task"],
                "question_id": row["question_id"],
                "image_id": meta.get("image_id", ""),
                "doc_id": meta.get("doc_id", ""),
                "external_match_rows": len(matches),
                "box_count": len(boxes),
                "status": status,
                "evidence_units": evidence_units,
                "source_paths": "; ".join(sorted(set(source_notes))),
            }
        )

    summary = build_summary(detail_rows, external_rows, args.external_bbox)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "external_bbox_annotations.jsonl", annotations)
    write_csv(out_dir / "external_bbox_annotation_rows.csv", detail_rows)
    write_csv(out_dir / "external_bbox_annotation_summary.csv", summary)
    (out_dir / "external_bbox_annotation_report.md").write_text(
        build_report(summary, detail_rows, args.external_bbox),
        encoding="utf-8",
    )
    print(
        f"Wrote {len(annotations)} adapted annotation rows to {out_dir}; "
        f"annotated={sum(1 for row in detail_rows if row['box_count'] > 0)}"
    )


def load_lite_metadata(pack_rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from datasets import load_dataset

    wanted: dict[str, set[str]] = {}
    for row in pack_rows:
        wanted.setdefault(row["task"], set()).add(str(row["question_id"]))
    out: dict[str, dict[str, Any]] = {}
    for task, ids in wanted.items():
        spec = DATASET_SPECS.get(task)
        if not spec:
            continue
        kwargs: dict[str, Any] = {}
        if spec["dataset_name"]:
            kwargs["name"] = spec["dataset_name"]
        dataset = load_dataset(spec["dataset_path"], **kwargs, split=spec["split"])
        for idx, doc in enumerate(dataset):
            qid = str(doc.get("question_id", doc.get("questionId", doc.get("id", idx))))
            if qid not in ids:
                continue
            image = doc.get("image")
            width = doc.get("image_width")
            height = doc.get("image_height")
            if (width is None or height is None) and image is not None and hasattr(image, "size"):
                width, height = image.size
            sample_id = f"{task_key(task)}:{qid}"
            out[sample_id] = {
                "question_id": qid,
                "image_id": str(doc.get("image_id", "")),
                "doc_id": str(doc.get("docId", doc.get("doc_id", ""))),
                "image_width": int(width or 0),
                "image_height": int(height or 0),
            }
            if len([key for key in out if key.startswith(task_key(task) + ":")]) >= len(ids):
                break
    return out


def load_external_rows(path: Path, *, source_path: str, external_task: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]]
    if not path.exists():
        return []
    if path.suffix.lower() == ".parquet":
        rows = read_parquet_rows(path)
    elif path.suffix.lower() == ".csv":
        rows = read_csv(path)
    else:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            rows = []
        elif path.suffix.lower() == ".jsonl" or not text.startswith(("{", "[")):
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            obj = json.loads(text)
            if isinstance(obj, list):
                rows = obj
            elif isinstance(obj, dict):
                for key in ("data", "annotations", "rows", "samples", "instances"):
                    if isinstance(obj.get(key), list):
                        rows = obj[key]
                        break
                else:
                    rows = list(obj.values()) if all(isinstance(v, dict) for v in obj.values()) else [obj]
            else:
                rows = []
    out = []
    for row in rows:
        if isinstance(row, dict):
            copied = dict(row)
            copied["_source_path"] = source_path
            if external_task:
                copied["_external_task"] = external_task
            out.append(copied)
    return out


def build_external_index(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        for key in external_keys(row):
            index.setdefault(key, []).append(row)
    return index


def external_keys(row: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for field in ("sample_id", "id", "question_id", "questionId", "questionID", "qid", "dataset_id"):
        value = row.get(field)
        if value not in (None, ""):
            keys.add(f"qid:{norm_id(value)}")
            if field in {"sample_id", "id"}:
                keys.add(f"sample:{norm_id(value)}")
            stripped = strip_split_prefix(value)
            if stripped != norm_id(value):
                keys.add(f"qid:{stripped}")
    for field in ("image_id", "imageId", "imageID", "image"):
        value = row.get(field)
        if value not in (None, "") and not looks_like_path(str(value)):
            keys.add(f"image:{norm_id(value)}")
    for field in ("docId", "doc_id", "document_id"):
        value = row.get(field)
        if value not in (None, ""):
            keys.add(f"doc:{norm_id(value)}")
    return keys


def find_matches(row: dict[str, str], meta: dict[str, Any], index: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    keys = {
        f"sample:{norm_id(row['sample_id'])}",
        f"qid:{norm_id(row['question_id'])}",
    }
    if meta.get("image_id"):
        keys.add(f"image:{norm_id(meta['image_id'])}")
    if meta.get("doc_id"):
        keys.add(f"doc:{norm_id(meta['doc_id'])}")
    matches: list[dict[str, Any]] = []
    seen: set[int] = set()
    for key in keys:
        for match in index.get(key, []):
            if match.get("_external_task") and match["_external_task"] != row["task"]:
                continue
            if id(match) in seen:
                continue
            seen.add(id(match))
            matches.append(match)
    return matches


def boxes_from_external_row(
    row: dict[str, Any],
    *,
    export_size: tuple[int, int],
    metadata: dict[str, Any],
    default_box_format: str,
    evidence_units: str,
    require_text_match: bool,
    min_text_similarity: float,
) -> list[dict[str, Any]]:
    raw_items = collect_box_items(row)
    out = []
    for item in raw_items:
        text = item_text(item)
        if require_text_match and text_similarity(text, evidence_units) < min_text_similarity:
            continue
        parsed = parse_box_item(item, default_box_format=default_box_format)
        if parsed is None:
            continue
        box = scale_box_to_export(
            parsed,
            export_size=export_size,
            source_size=source_size(row, metadata),
        )
        if box is None:
            continue
        if text:
            box["label"] = text
        out.append(box)
    return out


def collect_box_items(row: dict[str, Any]) -> list[Any]:
    answer_span_items = collect_answer_span_items(row)
    if answer_span_items:
        return answer_span_items
    for key in (
        "boxes",
        "bboxes",
        "bbox",
        "gt_bbox",
        "gt_bboxes",
        "grounding_bboxes",
        "answer_bboxes",
        "ocr_boxes",
        "ocr_tokens",
        "annotations",
        "regions",
    ):
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            if len(value) == 4 and all(is_number(v) for v in value):
                return [value]
            return value
        if isinstance(value, dict):
            return [value]
    if any(key in row for key in ("x", "y", "w", "h", "x1", "y1", "x2", "y2")):
        return [row]
    return []


def collect_answer_span_items(row: dict[str, Any]) -> list[dict[str, Any]]:
    words = row.get("words")
    boxes = row.get("bounding_boxes")
    answer = row.get("answer")
    if not isinstance(words, list) or not isinstance(boxes, list) or not isinstance(answer, dict):
        return []
    try:
        start = int(answer.get("start"))
    except (TypeError, ValueError):
        return []
    if start < 0 or start >= len(words) or start >= len(boxes):
        return []
    answer_text = str(answer.get("text") or answer.get("matched_text") or "").strip()
    matched_text = str(answer.get("matched_text") or "").strip()
    span_len = infer_answer_span_len(words, start, answer_text=answer_text, matched_text=matched_text)
    out = []
    for idx in range(start, min(len(words), len(boxes), start + span_len)):
        out.append({"bbox": boxes[idx], "text": str(words[idx])})
    return out


def infer_answer_span_len(words: list[Any], start: int, *, answer_text: str, matched_text: str) -> int:
    answer_norm = normalize_text(answer_text)
    if not answer_norm:
        answer_norm = normalize_text(matched_text)
    if not answer_norm:
        return 1
    best_len = 1
    best_score = -1.0
    max_len = min(12, len(words) - start)
    for span_len in range(1, max_len + 1):
        span_text = " ".join(str(word) for word in words[start : start + span_len])
        span_norm = normalize_text(span_text)
        if not span_norm:
            continue
        if span_norm == answer_norm or answer_norm in span_norm:
            return span_len
        score = token_jaccard(span_norm, answer_norm)
        # Do not stop on a prefix/sub-string match: answer.start often points
        # to the first OCR token of a multi-token answer.
        if score > best_score or (score > 0 and score == best_score and span_len > best_len):
            best_len = span_len
            best_score = score
    return max(1, best_len)


def parse_box_item(item: Any, *, default_box_format: str) -> tuple[float, float, float, float, str] | None:
    if isinstance(item, dict):
        if all(key in item for key in ("x", "y", "w", "h")):
            try:
                return float(item["x"]), float(item["y"]), float(item["w"]), float(item["h"]), "xywh"
            except (TypeError, ValueError):
                return None
        if all(key in item for key in ("x1", "y1", "x2", "y2")):
            try:
                return float(item["x1"]), float(item["y1"]), float(item["x2"]), float(item["y2"]), "xyxy"
            except (TypeError, ValueError):
                return None
        for key in ("bbox", "box", "bounds", "rect"):
            if key in item:
                return parse_box_item(item[key], default_box_format=default_box_format)
        for key in ("points", "polygon", "vertices"):
            if key in item:
                return parse_points(item[key])
    if isinstance(item, (list, tuple)) and len(item) >= 4:
        try:
            vals = [float(item[i]) for i in range(4)]
        except (TypeError, ValueError):
            return None
        return vals[0], vals[1], vals[2], vals[3], default_box_format
    return None


def parse_points(points: Any) -> tuple[float, float, float, float, str] | None:
    if not isinstance(points, list):
        return None
    xs, ys = [], []
    for point in points:
        if isinstance(point, dict):
            x, y = point.get("x"), point.get("y")
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            x, y = point[0], point[1]
        else:
            continue
        try:
            xs.append(float(x))
            ys.append(float(y))
        except (TypeError, ValueError):
            continue
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys), "xyxy"


def scale_box_to_export(
    parsed: tuple[float, float, float, float, str],
    *,
    export_size: tuple[int, int],
    source_size: tuple[int, int],
) -> dict[str, Any] | None:
    a, b, c, d, fmt = parsed
    export_w, export_h = export_size
    if export_w <= 0 or export_h <= 0:
        return None
    coords = [a, b, c, d]
    normalized = max(abs(v) for v in coords) <= 1.5
    if fmt == "xyxy":
        x1, y1, x2, y2 = a, b, c, d
        if normalized:
            x1, x2 = x1 * export_w, x2 * export_w
            y1, y2 = y1 * export_h, y2 * export_h
        else:
            source_w, source_h = source_size or export_size
            x1, x2 = x1 * export_w / source_w, x2 * export_w / source_w
            y1, y2 = y1 * export_h / source_h, y2 * export_h / source_h
        x, y, w, h = min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)
    else:
        x, y, w, h = a, b, c, d
        if normalized:
            x, w = x * export_w, w * export_w
            y, h = y * export_h, h * export_h
        else:
            source_w, source_h = source_size or export_size
            x, w = x * export_w / source_w, w * export_w / source_w
            y, h = y * export_h / source_h, h * export_h / source_h
    x = max(0.0, min(float(export_w), x))
    y = max(0.0, min(float(export_h), y))
    w = max(0.0, min(float(export_w) - x, w))
    h = max(0.0, min(float(export_h) - y, h))
    if w <= 0.0 or h <= 0.0:
        return None
    return {"x": round(x, 2), "y": round(y, 2), "w": round(w, 2), "h": round(h, 2)}


def source_size(row: dict[str, Any], metadata: dict[str, Any]) -> tuple[int, int]:
    width = first_present(row, ("image_width", "width", "original_width", "source_width", "img_width"))
    height = first_present(row, ("image_height", "height", "original_height", "source_height", "img_height"))
    try:
        width_i, height_i = int(float(width)), int(float(height))
    except (TypeError, ValueError):
        image_size = image_size_from_external_row(row)
        width_i = image_size[0] or int(metadata.get("image_width", 0) or 0)
        height_i = image_size[1] or int(metadata.get("image_height", 0) or 0)
    return width_i, height_i


def image_size_from_external_row(row: dict[str, Any]) -> tuple[int, int]:
    image = row.get("image")
    data = None
    if isinstance(image, dict):
        data = image.get("bytes")
    elif isinstance(image, (bytes, bytearray)):
        data = image
    if not data:
        return 0, 0
    try:
        with Image.open(BytesIO(data)) as img:
            return img.size
    except Exception:
        return 0, 0


def item_text(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    for key in ("text", "ocr_text", "word", "label", "token", "answer", "value"):
        value = item.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def text_similarity(text: str, evidence_units: str) -> float:
    text_norm = normalize_text(text)
    units = [normalize_text(part) for part in re.split(r"[|;,]", evidence_units) if normalize_text(part)]
    if not text_norm or not units:
        return 0.0
    if any(text_norm in unit or unit in text_norm for unit in units):
        return 1.0
    text_tokens = set(text_norm.split())
    best = 0.0
    for unit in units:
        unit_tokens = set(unit.split())
        if not text_tokens or not unit_tokens:
            continue
        best = max(best, len(text_tokens & unit_tokens) / len(text_tokens | unit_tokens))
    return best


def build_summary(
    rows: list[dict[str, Any]],
    external_rows: list[dict[str, Any]],
    external_paths: list[str],
) -> list[dict[str, Any]]:
    out = [
        {"scope": "all", "metric": "annotation_rows", "value": len(rows)},
        {"scope": "all", "metric": "external_rows_loaded", "value": len(external_rows)},
        {"scope": "all", "metric": "external_paths", "value": len(external_paths)},
        {"scope": "all", "metric": "rows_with_external_match", "value": sum(int(row["external_match_rows"]) > 0 for row in rows)},
        {"scope": "all", "metric": "rows_with_boxes", "value": sum(int(row["box_count"]) > 0 for row in rows)},
        {"scope": "all", "metric": "boxes", "value": sum(int(row["box_count"]) for row in rows)},
        {"scope": "local_hint", "metric": "textvqa_gt_bbox_readme_cached", "value": int(TEXTVQA_GT_BBOX_README.exists())},
    ]
    for task in sorted({row["task"] for row in rows}):
        task_rows = [row for row in rows if row["task"] == task]
        out.extend(
            [
                {"scope": task, "metric": "annotation_rows", "value": len(task_rows)},
                {"scope": task, "metric": "rows_with_external_match", "value": sum(int(row["external_match_rows"]) > 0 for row in task_rows)},
                {"scope": task, "metric": "rows_with_boxes", "value": sum(int(row["box_count"]) > 0 for row in task_rows)},
                {"scope": task, "metric": "boxes", "value": sum(int(row["box_count"]) for row in task_rows)},
            ]
        )
    return out


def build_report(summary: list[dict[str, Any]], rows: list[dict[str, Any]], external_paths: list[str]) -> str:
    lines = [
        "# External BBox Annotation Adapter",
        "",
        "This adapter converts external TextVQA/DocVQA/OCR detector boxes into the same JSONL schema used by the local bbox annotation tool. It does not infer boxes when no external source is provided.",
        "",
        "## Inputs",
        "",
    ]
    if external_paths:
        for path in external_paths:
            lines.append(f"- `{path}`")
    else:
        lines.append("- No external bbox file was provided; this is a dry run.")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| Scope | Metric | Value |",
            "| --- | --- | ---: |",
        ]
    )
    for row in summary:
        lines.append(f"| {row['scope']} | {row['metric']} | {row['value']} |")
    lines.extend(
        [
            "",
            "## Local Source Hint",
            "",
            f"- TextVQA GT bbox README cached: `{TEXTVQA_GT_BBOX_README}` ({'yes' if TEXTVQA_GT_BBOX_README.exists() else 'no'}).",
            "- The cached LMMs-Eval-Lite TextVQA split has OCR tokens but no bbox fields; DocVQA-lite has no OCR-token field.",
            "",
            "## Unmatched Preview",
            "",
            "| Sample | Task | Question | Image/Doc | Evidence units |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        if int(row["box_count"]) == 0:
            image_or_doc = row.get("image_id") or row.get("doc_id") or ""
            lines.append(
                f"| {row['sample_id']} | {row['task']} | {row['question_id']} | "
                f"{image_or_doc} | {escape_md(row['evidence_units'])} |"
            )
            if len(lines) > 80:
                break
    return "\n".join(lines) + "\n"


def task_key(task: str) -> str:
    return "textvqa_val_lite" if task == "TextVQA-lite" else "docvqa_val_lite"


def image_size(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    with Image.open(path) as image:
        return image.size


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)
    columns = [name for name in parquet_file.schema_arrow.names if name != "image"]
    table = pq.read_table(path, columns=columns or None)
    return [row for row in table.to_pylist() if isinstance(row, dict)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dedupe_output_boxes(boxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[int, int, int, int]] = set()
    out = []
    for box in boxes:
        key = tuple(int(round(float(box[field]) * 10)) for field in ("x", "y", "w", "h"))
        if key in seen:
            continue
        seen.add(key)
        out.append(box)
    return out


def first_present(row: dict[str, Any], fields: tuple[str, ...]) -> Any:
    for field in fields:
        if row.get(field) not in (None, ""):
            return row[field]
    return None


def norm_id(value: Any) -> str:
    return str(value).strip().lower()


def strip_split_prefix(value: Any) -> str:
    text = norm_id(value)
    match = re.match(r"^(?:train|val|validation|test)[_-](.+)$", text)
    return match.group(1) if match else text


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text).lower())).strip()


def token_jaccard(left: str, right: str) -> float:
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def looks_like_path(text: str) -> bool:
    return "/" in text or "\\" in text or text.lower().endswith((".jpg", ".jpeg", ".png"))


def is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def escape_md(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
