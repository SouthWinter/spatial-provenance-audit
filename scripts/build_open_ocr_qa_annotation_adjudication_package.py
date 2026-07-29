#!/usr/bin/env python3
"""Build consensus draft and adjudication queue from two annotation exports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from audit_open_ocr_qa_annotation_agreement import (
    compare_sample,
    load_by_sample,
    normalized_boxes,
    write_csv,
)


ROOT = Path(__file__).resolve().parents[1]
PREFILL_JSONL = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_evidence_prefill"
    / "evidence_prefill_pack.jsonl"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-a", required=True, help="First JSON/JSONL annotation export")
    parser.add_argument("--annotations-b", required=True, help="Second JSON/JSONL annotation export")
    parser.add_argument(
        "--output-dir",
        default="runs/problem_optimization_audit/open_ocr_qa_annotation_adjudication",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument(
        "--sample-scope",
        choices=("intersection", "union"),
        default="intersection",
        help=(
            "Adjudicate only samples present in both exports (the default for "
            "a calibration subset), or the union for two complete exports."
        ),
    )
    args = parser.parse_args()

    ann_a = load_by_sample(Path(args.annotations_a))
    ann_b = load_by_sample(Path(args.annotations_b))
    prefill = {row["sample_id"]: row for row in read_jsonl(PREFILL_JSONL)}
    consensus = []
    queue_rows = []
    queue_jsonl = []

    sample_ids = set(ann_a) & set(ann_b)
    if args.sample_scope == "union":
        sample_ids = set(ann_a) | set(ann_b)
    for sample_id in sorted(sample_ids):
        a = ann_a.get(sample_id)
        b = ann_b.get(sample_id)
        agreement = compare_sample(sample_id, a, b, args.iou_threshold)
        meta = prefill.get(sample_id, {})
        if int(agreement["needs_adjudication"]) == 0:
            consensus.append(consensus_row(a, b, meta, agreement))
        else:
            queue_rows.append(queue_csv_row(a, b, meta, agreement))
            queue_jsonl.append(queue_json_row(a, b, meta, agreement))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "consensus_draft.json", consensus)
    write_jsonl(out_dir / "adjudication_queue.jsonl", queue_jsonl)
    write_csv(out_dir / "adjudication_queue.csv", queue_rows)
    write_csv(out_dir / "adjudication_summary.csv", summary_rows(consensus, queue_rows))
    (out_dir / "adjudication_report.md").write_text(
        build_report(consensus, queue_rows, args.iou_threshold), encoding="utf-8"
    )
    print(f"Consensus draft rows={len(consensus)}; adjudication rows={len(queue_rows)}")


def consensus_row(a: dict[str, Any] | None, b: dict[str, Any] | None, meta: dict[str, Any], agreement: dict[str, Any]) -> dict[str, Any]:
    source = a if a is not None else b
    assert source is not None
    notes = " | ".join(
        part
        for part in [
            str(a.get("notes", "")).strip() if a else "",
            str(b.get("notes", "")).strip() if b else "",
        ]
        if part
    )
    return {
        "sample_id": source["sample_id"],
        "task": source.get("task", meta.get("task", "")),
        "question_id": source.get("question_id", meta.get("question_id", "")),
        "status": "consensus_draft",
        "boxes": source.get("boxes", []),
        "notes": notes,
        "agreement_mean_iou": agreement["mean_matched_iou"],
        "agreement_source": "auto_accept_same_count_type_iou",
    }


def queue_csv_row(a: dict[str, Any] | None, b: dict[str, Any] | None, meta: dict[str, Any], agreement: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": agreement["sample_id"],
        "task": agreement["task"] or meta.get("task", ""),
        "question": meta.get("question", ""),
        "gold_answers": meta.get("gold_answers", ""),
        "evidence_units": meta.get("prefill_evidence_units", ""),
        "image_path": meta.get("image_path", ""),
        "box_count_a": agreement["box_count_a"],
        "box_count_b": agreement["box_count_b"],
        "label_types_a": agreement["label_types_a"],
        "label_types_b": agreement["label_types_b"],
        "mean_matched_iou": agreement["mean_matched_iou"],
        "reason": adjudication_reason(agreement),
        "annotator_a_boxes": json.dumps(normalized_boxes(a), ensure_ascii=False),
        "annotator_b_boxes": json.dumps(normalized_boxes(b), ensure_ascii=False),
    }


def queue_json_row(a: dict[str, Any] | None, b: dict[str, Any] | None, meta: dict[str, Any], agreement: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": agreement["sample_id"],
        "task": agreement["task"] or meta.get("task", ""),
        "question_id": meta.get("question_id", ""),
        "question": meta.get("question", ""),
        "gold_answers": meta.get("gold_answers", ""),
        "evidence_units": meta.get("prefill_evidence_units", ""),
        "image_path": meta.get("image_path", ""),
        "agreement": agreement,
        "annotator_a": a,
        "annotator_b": b,
        "adjudicated_boxes": [],
        "adjudication_notes": "",
        "status": "needs_adjudication",
    }


def adjudication_reason(row: dict[str, Any]) -> str:
    reasons = []
    if not row["present_a"] or not row["present_b"]:
        reasons.append("missing export")
    if row["box_count_a"] == 0 and row["box_count_b"] == 0:
        reasons.append("no boxes")
    if row["box_count_a"] != row["box_count_b"]:
        reasons.append("box count")
    if not int(row["label_type_set_match"]):
        reasons.append("label types")
    if row["box_count_a"] and not int(row["all_boxes_matched_iou"]):
        reasons.append("box IoU")
    return ", ".join(reasons) or "manual check"


def summary_rows(consensus: list[dict[str, Any]], queue: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {"scope": "all", "metric": "consensus_draft_rows", "value": len(consensus)},
        {"scope": "all", "metric": "adjudication_rows", "value": len(queue)},
        {"scope": "all", "metric": "total_rows", "value": len(consensus) + len(queue)},
    ]
    for task in sorted({row.get("task", "") for row in consensus + queue if row.get("task", "")}):
        rows.append({"scope": task, "metric": "consensus_draft_rows", "value": sum(row.get("task", "") == task for row in consensus)})
        rows.append({"scope": task, "metric": "adjudication_rows", "value": sum(row.get("task", "") == task for row in queue)})
    return rows


def build_report(consensus: list[dict[str, Any]], queue: list[dict[str, Any]], iou_threshold: float) -> str:
    lines = [
        "# Annotation Adjudication Package",
        "",
        f"Rows with matching box count, label-type set, and all matched IoU >= {iou_threshold:.2f} are copied into `consensus_draft.json`. Other rows are written to `adjudication_queue.jsonl` and `adjudication_queue.csv`.",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| consensus draft rows | {len(consensus)} |",
        f"| adjudication rows | {len(queue)} |",
        f"| total rows | {len(consensus) + len(queue)} |",
        "",
        "## Adjudication Queue",
        "",
        "| Sample | Task | Reason | Boxes A/B | Mean IoU |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for row in queue:
        lines.append(
            f"| {row['sample_id']} | {row['task']} | {row['reason']} | "
            f"{row['box_count_a']}/{row['box_count_b']} | {row['mean_matched_iou']} |"
        )
    return "\n".join(lines) + "\n"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
