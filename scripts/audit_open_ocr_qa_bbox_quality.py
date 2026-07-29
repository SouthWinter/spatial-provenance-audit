#!/usr/bin/env python3
"""Audit external bbox annotation quality for the open OCR QA stress pack."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, median
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PREFILL_JSONL = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_evidence_prefill"
    / "evidence_prefill_pack.jsonl"
)
DEFAULT_SOURCES = (
    (
        "TextVQA-GT",
        ROOT
        / "runs"
        / "problem_optimization_audit"
        / "open_ocr_qa_textvqa_gt_bbox"
        / "external_bbox_annotations.jsonl",
    ),
    (
        "DocVQA-OCR-answer-span-expanded",
        ROOT
        / "runs"
        / "problem_optimization_audit"
        / "open_ocr_qa_docvqa_hxlinh_bbox_expanded"
        / "external_bbox_annotations.jsonl",
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Annotation source as label=path. Can be repeated. Defaults to TextVQA GT and DocVQA OCR answer-span sources.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/problem_optimization_audit/open_ocr_qa_bbox_quality_audit_expanded",
    )
    args = parser.parse_args()

    sources = parse_sources(args.source)
    prefill = {row["sample_id"]: row for row in read_jsonl(PREFILL_JSONL)}
    image_sizes = build_image_sizes(prefill)

    rows: list[dict[str, Any]] = []
    for source_label, path in sources:
        for ann in load_jsonl(path):
            rows.append(build_row(source_label, path, ann, prefill, image_sizes))

    summary = build_summary(rows)
    examples = build_examples(rows)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "bbox_quality_rows.csv", rows)
    write_csv(out_dir / "bbox_quality_summary.csv", summary)
    write_csv(out_dir / "bbox_quality_examples.csv", examples)
    (out_dir / "bbox_quality_report.md").write_text(build_report(summary, examples), encoding="utf-8")
    print(f"Wrote bbox quality audit for {len(rows)} rows to {out_dir}")


def parse_sources(items: list[str]) -> list[tuple[str, Path]]:
    if not items:
        return [(label, path) for label, path in DEFAULT_SOURCES]
    out = []
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--source must be label=path, got: {item}")
        label, raw_path = item.split("=", 1)
        out.append((label, Path(raw_path)))
    return out


def build_image_sizes(prefill: dict[str, dict[str, Any]]) -> dict[str, tuple[int, int]]:
    sizes = {}
    for sid, row in prefill.items():
        path = Path(str(row.get("image_path", "")))
        if path.exists():
            with Image.open(path) as img:
                sizes[sid] = img.size
    return sizes


def build_row(
    source_label: str,
    source_path: Path,
    ann: dict[str, Any],
    prefill: dict[str, dict[str, Any]],
    image_sizes: dict[str, tuple[int, int]],
) -> dict[str, Any]:
    sid = str(ann.get("sample_id", ""))
    meta = prefill.get(sid, {})
    boxes = ann.get("boxes") if isinstance(ann.get("boxes"), list) else []
    width, height = image_sizes.get(sid, (0, 0))
    applicable_task = infer_applicable_task(source_label)
    task = str(ann.get("task", meta.get("task", "")))
    labels = [str(box.get("label", "")).strip() for box in boxes if isinstance(box, dict) and str(box.get("label", "")).strip()]
    evidence_text = str(ann.get("evidence_units") or meta.get("prefill_evidence_units") or "")
    text_quality = score_text_coverage(evidence_text, labels)
    area_fracs = [box_area_frac(box, width, height) for box in boxes if isinstance(box, dict)]
    area_fracs = [x for x in area_fracs if x >= 0]
    stress_tags = str(meta.get("stress_tags", ""))
    suggested_regions = safe_int(meta.get("prefill_suggested_region_count", 0))
    return {
        "source": source_label,
        "source_path": str(source_path.relative_to(ROOT) if source_path.is_absolute() and source_path.is_relative_to(ROOT) else source_path),
        "sample_id": sid,
        "task": task,
        "source_applicable_task": applicable_task or "",
        "is_source_applicable_task": int(applicable_task is None or task == applicable_task),
        "question_id": ann.get("question_id", meta.get("question_id", "")),
        "status": ann.get("status", ""),
        "box_count": len(boxes),
        "has_box": int(len(boxes) > 0),
        "has_box_label": int(bool(labels)),
        "label_text": " | ".join(labels),
        "evidence_text": evidence_text,
        "best_evidence_option": text_quality["best_option"],
        "evidence_token_count": text_quality["evidence_token_count"],
        "label_token_count": text_quality["label_token_count"],
        "label_token_recall": fmt_float(text_quality["recall"]),
        "label_token_precision": fmt_float(text_quality["precision"]),
        "label_token_f1": fmt_float(text_quality["f1"]),
        "label_quality_bucket": text_quality["bucket"],
        "mean_box_area_frac": fmt_float(mean(area_fracs) if area_fracs else math.nan),
        "max_box_area_frac": fmt_float(max(area_fracs) if area_fracs else math.nan),
        "image_width": width,
        "image_height": height,
        "stress_tags": stress_tags,
        "prefill_complexity": meta.get("prefill_complexity", ""),
        "prefill_suggested_region_count": suggested_regions,
        "multi_token_answer": int(text_quality["evidence_token_count"] >= 2),
        "multi_region_hint": int(suggested_regions > 1 or "multi" in str(meta.get("prefill_complexity", "")).lower()),
    }


def infer_applicable_task(source_label: str) -> str | None:
    lowered = source_label.lower()
    if "textvqa" in lowered:
        return "TextVQA-lite"
    if "docvqa" in lowered:
        return "DocVQA-lite"
    return None


def score_text_coverage(evidence_text: str, labels: list[str]) -> dict[str, Any]:
    options = [part.strip() for part in evidence_text.split("|") if part.strip()]
    if not options and evidence_text.strip():
        options = [evidence_text.strip()]
    label_text = " ".join(labels)
    label_tokens = norm_tokens(label_text)
    if not options:
        return {
            "best_option": "",
            "evidence_token_count": 0,
            "label_token_count": len(label_tokens),
            "recall": math.nan,
            "precision": math.nan,
            "f1": math.nan,
            "bucket": "no_evidence_text",
        }
    best = None
    for option in options:
        ev_tokens = norm_tokens(option)
        recall, precision, f1 = token_scores(ev_tokens, label_tokens)
        candidate = (f1, recall, precision, option, len(ev_tokens))
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    f1, recall, precision, option, ev_count = best
    if not label_tokens:
        bucket = "no_box_label"
    elif recall >= 0.999 and precision >= 0.999:
        bucket = "exact_label_match"
    elif recall >= 0.999:
        bucket = "full_answer_covered"
    elif recall >= 0.5:
        bucket = "partial_ge_0p50"
    elif recall > 0:
        bucket = "partial_lt_0p50"
    else:
        bucket = "no_token_overlap"
    return {
        "best_option": option,
        "evidence_token_count": ev_count,
        "label_token_count": len(label_tokens),
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "bucket": bucket,
    }


def token_scores(evidence_tokens: list[str], label_tokens: list[str]) -> tuple[float, float, float]:
    if not evidence_tokens or not label_tokens:
        return (0.0 if evidence_tokens else math.nan, 0.0 if label_tokens else math.nan, 0.0)
    label_remaining = list(label_tokens)
    overlap = 0
    for tok in evidence_tokens:
        if tok in label_remaining:
            overlap += 1
            label_remaining.remove(tok)
    recall = overlap / len(evidence_tokens)
    precision = overlap / len(label_tokens)
    f1 = 0.0 if recall + precision == 0 else 2 * recall * precision / (recall + precision)
    return recall, precision, f1


def norm_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def box_area_frac(box: dict[str, Any], width: int, height: int) -> float:
    if width <= 0 or height <= 0:
        return -1.0
    try:
        w = max(0.0, float(box.get("w", 0)))
        h = max(0.0, float(box.get("h", 0)))
    except (TypeError, ValueError):
        return -1.0
    return (w * h) / float(width * height)


def build_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    applicable_rows = [row for row in rows if safe_int(row.get("is_source_applicable_task", 1)) > 0]
    scopes = [("all_rows", rows), ("all_applicable_source_tasks", applicable_rows)]
    for source in sorted({row["source"] for row in rows}):
        source_rows = [row for row in rows if row["source"] == source]
        applicable_source_rows = [
            row for row in source_rows if safe_int(row.get("is_source_applicable_task", 1)) > 0
        ]
        scopes.append((source, source_rows))
        scopes.append((f"{source}:applicable_task", applicable_source_rows))
        for task in sorted({row["task"] for row in source_rows}):
            scopes.append((f"{source}:{task}", [row for row in source_rows if row["task"] == task]))
    for scope, subset in scopes:
        if not subset:
            continue
        annotated = [row for row in subset if int(row["has_box"]) > 0]
        labelled = [row for row in annotated if int(row["has_box_label"]) > 0]
        multi_token = [row for row in annotated if int(row["multi_token_answer"]) > 0]
        area_values = [safe_float(row["mean_box_area_frac"]) for row in annotated]
        area_values = [x for x in area_values if not math.isnan(x)]
        out.extend(
            [
                {"scope": scope, "metric": "rows", "value": len(subset)},
                {"scope": scope, "metric": "annotated_rows", "value": len(annotated)},
                {"scope": scope, "metric": "empty_rows", "value": len(subset) - len(annotated)},
                {"scope": scope, "metric": "boxes", "value": sum(safe_int(row["box_count"]) for row in subset)},
                {"scope": scope, "metric": "multi_box_rows", "value": sum(safe_int(row["box_count"]) > 1 for row in annotated)},
                {"scope": scope, "metric": "rows_with_box_labels", "value": len(labelled)},
                {"scope": scope, "metric": "multi_token_annotated_rows", "value": len(multi_token)},
                {
                    "scope": scope,
                    "metric": "multi_token_full_label_recall_rows",
                    "value": sum(safe_float(row["label_token_recall"]) >= 0.999 for row in multi_token),
                },
                {
                    "scope": scope,
                    "metric": "multi_token_partial_label_rows",
                    "value": sum(0.0 < safe_float(row["label_token_recall"]) < 0.999 for row in multi_token),
                },
                {
                    "scope": scope,
                    "metric": "mean_box_area_frac",
                    "value": fmt_float(mean(area_values) if area_values else math.nan),
                },
                {
                    "scope": scope,
                    "metric": "median_box_area_frac",
                    "value": fmt_float(median(area_values) if area_values else math.nan),
                },
            ]
        )
        for bucket in sorted({row["label_quality_bucket"] for row in subset}):
            out.append(
                {
                    "scope": scope,
                    "metric": f"label_bucket:{bucket}",
                    "value": sum(row["label_quality_bucket"] == bucket for row in subset),
                }
            )
    return out


def build_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if row["label_quality_bucket"] in {"partial_lt_0p50", "partial_ge_0p50", "no_token_overlap"}
        and int(row["has_box"]) > 0
    ]
    candidates.sort(
        key=lambda row: (
            safe_float(row["label_token_recall"]) if not math.isnan(safe_float(row["label_token_recall"])) else 2.0,
            row["source"],
            row["sample_id"],
        )
    )
    keep = []
    for row in candidates[:20]:
        keep.append(
            {
                "source": row["source"],
                "sample_id": row["sample_id"],
                "task": row["task"],
                "evidence_text": row["evidence_text"],
                "label_text": row["label_text"],
                "label_token_recall": row["label_token_recall"],
                "label_quality_bucket": row["label_quality_bucket"],
                "stress_tags": row["stress_tags"],
            }
        )
    return keep


def build_report(summary: list[dict[str, Any]], examples: list[dict[str, Any]]) -> str:
    lines = [
        "# Open OCR QA BBox Quality Audit",
        "",
        "This audit describes the quality and scope of external bbox annotations used for open OCR/DocQA stress evidence checks. It does not measure model accuracy.",
        "",
        "## Summary",
        "",
        "| Scope | Metric | Value |",
        "| --- | --- | ---: |",
    ]
    for row in summary:
        lines.append(f"| {row['scope']} | {row['metric']} | {row['value']} |")
    lines.extend(
        [
            "",
            "## Weak Text-Span Examples",
            "",
            "| Source | Sample | Task | Evidence text | Box label text | Token recall | Bucket |",
            "| --- | --- | --- | --- | --- | ---: | --- |",
        ]
    )
    for row in examples:
        lines.append(
            f"| {row['source']} | {row['sample_id']} | {row['task']} | {escape_md(row['evidence_text'])} | "
            f"{escape_md(row['label_text'])} | {row['label_token_recall']} | {row['label_quality_bucket']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- TextVQA GT boxes provide geometry but no per-box recognized text labels in the imported source, so text-span completeness is not scored for those rows.",
            "- DocVQA boxes are OCR answer-token spans. They are useful for answer-token availability audits, but partial-span rows should not be described as complete multi-region document evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return load_jsonl(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def safe_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def fmt_float(value: float) -> str:
    if math.isnan(value):
        return ""
    return f"{value:.3f}"


def escape_md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
