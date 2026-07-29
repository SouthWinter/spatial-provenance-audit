#!/usr/bin/env python3
"""Summarize open-QA evidence-region sources and their claim boundaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "runs" / "problem_optimization_audit" / "open_ocr_qa_evidence_source_boundary"
ANNOTATION_PACK = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_annotation_pack"
    / "annotation_pack.csv"
)
MANUAL_READINESS = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "manual_evidence_readiness_gate"
    / "manual_evidence_readiness_gates.csv"
)
TEXTVQA_ROWS = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_textvqa_gt_bbox"
    / "external_bbox_annotation_rows.csv"
)
TEXTVQA_ECR = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_textvqa_gt_bbox_ecr"
    / "bbox_ecr_rows.csv"
)
DOCVQA_ANSWER_ROWS = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_docvqa_hxlinh_bbox_expanded"
    / "external_bbox_annotation_rows.csv"
)
DOCVQA_ANSWER_ECR = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_docvqa_hxlinh_bbox_expanded_ecr"
    / "bbox_ecr_rows.csv"
)
DOCVQA_LINE_ROWS = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_docvqa_hxlinh_line_context_bbox"
    / "line_context_rows.csv"
)
DOCVQA_LINE_ECR = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_docvqa_hxlinh_line_context_bbox_ecr"
    / "bbox_ecr_rows.csv"
)
FINAL_MANUAL = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_final_annotations"
    / "final_annotations.jsonl"
)
FINAL_MANUAL_ECR = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_final_annotations_ecr"
    / "bbox_ecr_rows.csv"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    rows = build_rows()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "evidence_source_boundary_summary.csv", rows)
    (out_dir / "evidence_source_boundary_report.md").write_text(build_report(rows), encoding="utf-8")
    print(f"Wrote evidence-source boundary audit to {out_dir}")


def build_rows() -> list[dict[str, Any]]:
    pack = read_csv(ANNOTATION_PACK)
    manual = read_csv(MANUAL_READINESS) if MANUAL_READINESS.exists() else []
    rows = [
        manual_row(pack, manual),
        external_row(
            source="TextVQA external GT boxes",
            task="TextVQA-lite",
            source_type="external manual-style GT",
            evidence_scope="mostly single answer-area boxes",
            applicable=[row for row in pack if row["task"] == "TextVQA-lite"],
            ann_rows=read_csv(TEXTVQA_ROWS),
            ecr_rows=read_csv(TEXTVQA_ECR),
            caveat="covers TextVQA stress rows only; mostly single-area boxes, not verified multi-evidence layout annotations",
        ),
        external_row(
            source="DocVQA OCR answer-token boxes",
            task="DocVQA-lite",
            source_type="public OCR-derived answer-token boxes",
            evidence_scope="answer tokens from OCR spans",
            applicable=[row for row in pack if row["task"] == "DocVQA-lite"],
            ann_rows=read_csv(DOCVQA_ANSWER_ROWS),
            ecr_rows=read_csv(DOCVQA_ANSWER_ECR),
            caveat="OCR-derived and answer-token focused; may miss field labels, row/column headers, and broader layout context",
        ),
        line_context_row(pack),
    ]
    return rows


def manual_row(pack: list[dict[str, str]], gates: list[dict[str, str]]) -> dict[str, Any]:
    gate_status = {row["gate"]: row["status"] for row in gates}
    final_rows = read_jsonl(FINAL_MANUAL) if FINAL_MANUAL.exists() else []
    boxed = [row for row in final_rows if distinct_box_count(row) > 0]
    box_counts = [distinct_box_count(row) for row in boxed]
    ecr70 = scored_budget(read_csv(FINAL_MANUAL_ECR), "0.70") if FINAL_MANUAL_ECR.exists() else []
    ready = (
        gate_status.get("final_manual_annotations_exist") == "pass"
        and gate_status.get("final_validation_ready_for_ecr") == "pass"
    )
    return {
        "source": "Human manual multi-region annotations",
        "task_scope": "TextVQA-lite + DocVQA-lite",
        "source_type": "human-confirmed assisted primary with independent calibration",
        "evidence_scope": "intended complete multi-region answer evidence",
        "applicable_samples": len(pack),
        "samples_with_source_match": len(final_rows),
        "samples_with_boxes": len(boxed),
        "coverage_rate": fmt(rate(len(boxed), len(pack))),
        "total_boxes": sum(box_counts),
        "mean_box_count_per_boxed_sample": fmt(mean(box_counts)) if box_counts else "",
        "multi_box_sample_rate": fmt(rate(sum(count > 1 for count in box_counts), len(box_counts))),
        "mean_added_context_boxes": "",
        "mean_ecr_0p70": avg(ecr70, "ECR"),
        "mean_worst_region_ecr_0p70": avg(ecr70, "worst_region_ECR"),
        "all_regions_ge_0p50_rate_0p70": avg(ecr70, "all_regions_ECR_ge_0p50"),
        "readiness": summarize_manual_readiness(gate_status),
        "claim_use": "scoped human-adjudicated multi-region evidence audit" if ready else "not a positive evidence result yet",
        "caveat": (
            "complete 96-sample stress audit; boxes are retrospective evidence labels and are not used by the box-free selector"
            if ready
            else "final human annotations or validation are incomplete"
        ),
    }


def external_row(
    *,
    source: str,
    task: str,
    source_type: str,
    evidence_scope: str,
    applicable: list[dict[str, str]],
    ann_rows: list[dict[str, str]],
    ecr_rows: list[dict[str, str]],
    caveat: str,
) -> dict[str, Any]:
    boxed = [row for row in ann_rows if safe_int(row.get("box_count")) > 0]
    box_counts = [safe_int(row.get("box_count")) for row in boxed]
    ecr70 = scored_budget(ecr_rows, "0.70")
    return {
        "source": source,
        "task_scope": task,
        "source_type": source_type,
        "evidence_scope": evidence_scope,
        "applicable_samples": len(applicable),
        "samples_with_source_match": sum(safe_int(row.get("external_match_rows")) > 0 for row in ann_rows),
        "samples_with_boxes": len(boxed),
        "coverage_rate": fmt(rate(len(boxed), len(applicable))),
        "total_boxes": sum(box_counts),
        "mean_box_count_per_boxed_sample": fmt(mean(box_counts)) if box_counts else "",
        "multi_box_sample_rate": fmt(rate(sum(count > 1 for count in box_counts), len(box_counts))),
        "mean_added_context_boxes": "",
        "mean_ecr_0p70": avg(ecr70, "ECR"),
        "mean_worst_region_ecr_0p70": avg(ecr70, "worst_region_ECR"),
        "all_regions_ge_0p50_rate_0p70": avg(ecr70, "all_regions_ECR_ge_0p50"),
        "readiness": "usable as scoped external evidence",
        "claim_use": "scoped evidence-availability audit",
        "caveat": caveat,
    }


def line_context_row(pack: list[dict[str, str]]) -> dict[str, Any]:
    applicable = [row for row in pack if row["task"] == "DocVQA-lite"]
    rows = read_csv(DOCVQA_LINE_ROWS)
    ecr70 = scored_budget(read_csv(DOCVQA_LINE_ECR), "0.70")
    boxed = [row for row in rows if safe_int(row.get("line_context_box_count")) > 0]
    line_counts = [safe_int(row.get("line_context_box_count")) for row in boxed]
    added_counts = [safe_int(row.get("added_context_boxes")) for row in boxed]
    return {
        "source": "DocVQA OCR line-context boxes",
        "task_scope": "DocVQA-lite",
        "source_type": "deterministic OCR context expansion",
        "evidence_scope": "answer tokens plus same-line OCR context",
        "applicable_samples": len(applicable),
        "samples_with_source_match": sum(safe_int(row.get("source_found")) > 0 for row in rows),
        "samples_with_boxes": len(boxed),
        "coverage_rate": fmt(rate(len(boxed), len(applicable))),
        "total_boxes": sum(line_counts),
        "mean_box_count_per_boxed_sample": fmt(mean(line_counts)) if line_counts else "",
        "multi_box_sample_rate": fmt(rate(sum(count > 1 for count in line_counts), len(line_counts))),
        "mean_added_context_boxes": fmt(mean(added_counts)) if added_counts else "",
        "mean_ecr_0p70": avg(ecr70, "ECR"),
        "mean_worst_region_ecr_0p70": avg(ecr70, "worst_region_ECR"),
        "all_regions_ge_0p50_rate_0p70": avg(ecr70, "all_regions_ECR_ge_0p50"),
        "readiness": "usable as deterministic boundary audit",
        "claim_use": "stricter document-context evidence-availability audit",
        "caveat": "same-line OCR context is not human-verified complete reasoning evidence and may over/under-include layout context",
    }


def scored_budget(rows: list[dict[str, str]], budget: str) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("metric_status") == "scored" and f"{safe_float(row.get('budget_keep_ratio')):.2f}" == budget
    ]


def summarize_manual_readiness(status: dict[str, str]) -> str:
    if not status:
        return "missing readiness gate"
    if status.get("final_manual_annotations_exist") == "pass" and status.get("final_validation_ready_for_ecr") == "pass":
        return "final manual annotations ready"
    if status.get("paper_evidence_excludes_manual_smoke") == "pass":
        return "guarded, not complete"
    return "not ready"


def avg(rows: list[dict[str, str]], key: str) -> str:
    vals = [safe_float(row.get(key)) for row in rows if row.get(key, "") != ""]
    return fmt(mean(vals)) if vals else ""


def rate(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def fmt(value: float) -> str:
    return f"{float(value):.3f}"


def safe_int(value: Any) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return 0


def safe_float(value: Any) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return 0.0


def build_report(rows: list[dict[str, Any]]) -> str:
    cols = [
        "source",
        "task_scope",
        "source_type",
        "samples_with_boxes",
        "coverage_rate",
        "total_boxes",
        "multi_box_sample_rate",
        "mean_ecr_0p70",
        "mean_worst_region_ecr_0p70",
        "all_regions_ge_0p50_rate_0p70",
        "readiness",
        "claim_use",
        "caveat",
    ]
    lines = [
        "# Open OCR QA Evidence Source Boundary Audit",
        "",
        "This audit distinguishes evidence-region sources used in the open TextVQA/DocVQA stress analyses. It is designed to prevent source types with different evidential strength from being collapsed into a single manual multi-evidence claim.",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in cols) + " |")
    lines.extend(
        [
            "",
            "## Safe Use",
            "",
            "Use this table to distinguish the completed 96-sample human-adjudicated stress audit from external GT, OCR-derived, and deterministic context-expansion evidence. The manual row supports scoped evidence-availability claims, not full-benchmark or causal-use claims.",
        ]
    )
    return "\n".join(lines) + "\n"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def distinct_box_count(row: dict[str, Any]) -> int:
    boxes = row.get("boxes", [])
    if not isinstance(boxes, list):
        return 0
    geometries = set()
    for box in boxes:
        if not isinstance(box, dict):
            continue
        geometries.add(tuple(box.get(key) for key in ("x", "y", "w", "h")))
    return len(geometries)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
