#!/usr/bin/env python3
"""Audit manual-evidence artifacts against the problem.md evidence standard.

The goal is not to create new evidence boxes. It prevents synthetic smoke
annotations from leaking into paper-facing evidence and records whether the
manual multi-region evidence gap is actually closed.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "runs" / "problem_optimization_audit" / "manual_evidence_readiness_gate"
PAPER_EVIDENCE_DIR = ROOT / "runs" / "paper_evidence"
ANNOTATION_PACK_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_annotation_pack"
    / "annotation_pack.csv"
)
FINAL_ANNOTATIONS_JSONL = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_final_annotations"
    / "final_annotations.jsonl"
)
FINAL_VALIDATION_SUMMARY = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_final_annotations_validation"
    / "bbox_annotation_validation_summary.csv"
)
LAUNCH_SUMMARY = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_manual_annotation_launch"
    / "launch_summary.csv"
)
PRIMARY_TOOL_HTML = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_manual_annotation_launch"
    / "primary_full_tool"
    / "index.html"
)
SECONDARY_TOOL_HTML = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_manual_annotation_launch"
    / "secondary_calibration_tool"
    / "index.html"
)
ANNOTATOR_HANDBOOK = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_manual_annotation_launch"
    / "annotator_handbook.md"
)
FINAL_DELIVERY_CHECKLIST = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_manual_annotation_launch"
    / "final_delivery_checklist.csv"
)
PRIMARY_PREFILL = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_manual_annotation_launch"
    / "primary_full_prefill.jsonl"
)
SECONDARY_PREFILL = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_manual_annotation_launch"
    / "secondary_calibration_prefill.jsonl"
)
PRIMARY_EXPORT = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_manual_annotation_launch"
    / "primary_full_tool"
    / "primary_full_tool_annotations.json"
)
SECONDARY_EXPORT = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_manual_annotation_launch"
    / "secondary_calibration_tool"
    / "secondary_calibration_tool_annotations.json"
)
PROGRESS_DECISION = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_manual_annotation_progress"
    / "manual_annotation_progress_decision.csv"
)
CURRENT_PROGRESS_DECISION = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_manual_annotation_launch"
    / "progress_audit"
    / "manual_annotation_progress_decision.csv"
)
SMOKE_FINAL_JSONL = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_final_annotation_smoke"
    / "final_annotations.jsonl"
)
SMOKE_ADJUDICATED_JSONL = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_final_annotation_smoke"
    / "adjudicated_queue.jsonl"
)

FORBIDDEN_PAPER_PATTERNS = [
    "open_ocr_qa_manual_evidence_audit_smoke",
    "open_ocr_qa_final_annotation_smoke",
    "open_ocr_qa_annotation_tool_smoke",
    "open_ocr_qa_annotation_agreement_smoke",
    "open_ocr_qa_annotation_adjudication_smoke",
    "answer_value:foo",
    "field_label:foo",
    "field_label:bar",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    gate_rows = build_gate_rows()
    detail_rows = build_detail_rows()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "manual_evidence_readiness_gates.csv", gate_rows)
    write_csv(out_dir / "manual_evidence_readiness_details.csv", detail_rows)
    (out_dir / "manual_evidence_readiness_report.md").write_text(
        build_report(gate_rows, detail_rows), encoding="utf-8"
    )
    print(f"Wrote manual-evidence readiness audit to {out_dir}")


def build_gate_rows() -> list[dict[str, Any]]:
    paper_hits = scan_paper_evidence_forbidden_patterns()
    pack_rows = read_csv(ANNOTATION_PACK_CSV) if ANNOTATION_PACK_CSV.exists() else []
    pack_manual_filled = [
        row
        for row in pack_rows
        if str(row.get("manual_evidence_region_count", "")).strip()
        or str(row.get("manual_evidence_texts", "")).strip()
        or str(row.get("manual_bbox_or_region_notes", "")).strip()
    ]
    final_rows = read_jsonl(FINAL_ANNOTATIONS_JSONL) if FINAL_ANNOTATIONS_JSONL.exists() else []
    final_unresolved = [row for row in final_rows if str(row.get("status", "")) != "annotated"]
    final_empty = [row for row in final_rows if not row.get("boxes")]
    final_smoke_like = [row for row in final_rows if is_smoke_like_annotation(row)]
    validation = validation_metrics()
    launch = launch_metrics()
    checklist = checklist_metrics()
    progress = progress_metrics()
    smoke_path = SMOKE_FINAL_JSONL if SMOKE_FINAL_JSONL.exists() else SMOKE_ADJUDICATED_JSONL
    smoke_rows = read_jsonl(smoke_path) if smoke_path.exists() else []
    smoke_like_rows = [row for row in smoke_rows if is_smoke_like_annotation(row)]

    return [
        gate(
            "paper_evidence_excludes_manual_smoke",
            not paper_hits,
            "paper-facing evidence must not reference synthetic manual smoke artifacts",
            f"{len(paper_hits)} forbidden hit(s)",
        ),
        gate(
            "annotation_pack_not_claimed_as_completed",
            bool(pack_rows) and not pack_manual_filled,
            "raw 96-row launch pack remains separate from completed final annotations",
            f"rows={len(pack_rows)}, rows_with_manual_fields={len(pack_manual_filled)}",
        ),
        gate(
            "manual_annotation_launch_package_ready",
            launch.get("primary_rows", "0") == "96"
            and launch.get("secondary_calibration_rows", "0") == "12"
            and launch.get("image_paths_exist", "0") == "96"
            and PRIMARY_TOOL_HTML.exists()
            and SECONDARY_TOOL_HTML.exists()
            and ANNOTATOR_HANDBOOK.exists()
            and PRIMARY_EXPORT.exists()
            and SECONDARY_EXPORT.exists()
            and bool(progress),
            "paper-facing manual annotation launch package should have 96 primary rows, 12 calibration rows, all images, generated tools, annotator handbook, both exports, and progress dashboard support",
            format_launch(launch, checklist, progress),
        ),
        gate(
            "final_manual_annotations_exist",
            bool(final_rows),
            "a positive manual-evidence claim requires final_annotations.jsonl",
            f"rows={len(final_rows)}",
        ),
        gate(
            "final_manual_annotations_no_unresolved_rows",
            bool(final_rows) and not final_unresolved,
            "final annotations must not contain unresolved rows",
            f"unresolved={len(final_unresolved)}",
        ),
        gate(
            "final_manual_annotations_no_empty_rows",
            bool(final_rows) and not final_empty,
            "final annotations must contain evidence boxes for each annotated row",
            f"empty={len(final_empty)}",
        ),
        gate(
            "final_manual_annotations_no_smoke_placeholders",
            bool(final_rows) and not final_smoke_like,
            "final annotations must not use synthetic placeholder labels or boxes",
            f"smoke_like={len(final_smoke_like)}",
        ),
        gate(
            "final_validation_ready_for_ecr",
            bool(validation)
            and validation.get("ready_for_ecr_rows", "0") == validation.get("rows", "")
            and validation.get("invalid_box_rows", "1") == "0"
            and validation.get("unlabeled_box_rows", "1") == "0",
            "validator must report all final rows ready for ECR",
            format_validation(validation),
        ),
        gate(
            "smoke_artifacts_detected_as_synthetic",
            not smoke_rows or len(smoke_like_rows) == len(smoke_rows),
            "when a non-production smoke fixture exists, every row should be identifiable as synthetic",
            (
                f"smoke_rows={len(smoke_rows)}, smoke_like={len(smoke_like_rows)}"
                if smoke_rows
                else "non-production smoke fixture absent"
            ),
        ),
    ]


def build_detail_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for hit in scan_paper_evidence_forbidden_patterns():
        rows.append({"category": "paper_forbidden_hit", **hit})
    if FINAL_ANNOTATIONS_JSONL.exists():
        for row in read_jsonl(FINAL_ANNOTATIONS_JSONL):
            if is_smoke_like_annotation(row) or str(row.get("status", "")) != "annotated" or not row.get("boxes"):
                rows.append(
                    {
                        "category": "final_annotation_problem_row",
                        "path": str(FINAL_ANNOTATIONS_JSONL.relative_to(ROOT)),
                        "pattern": "",
                        "sample_id": row.get("sample_id", ""),
                        "task": row.get("task", ""),
                        "status": row.get("status", ""),
                        "box_count": len(row.get("boxes", []) or []),
                    }
                )
    smoke_path = SMOKE_FINAL_JSONL if SMOKE_FINAL_JSONL.exists() else SMOKE_ADJUDICATED_JSONL
    if smoke_path.exists():
        for row in read_jsonl(smoke_path):
            rows.append(
                {
                    "category": "known_smoke_row",
                    "path": str(smoke_path.relative_to(ROOT)),
                    "pattern": "smoke_like" if is_smoke_like_annotation(row) else "not_detected",
                    "sample_id": row.get("sample_id", ""),
                    "task": row.get("task", ""),
                    "status": row.get("status", ""),
                    "box_count": len(row.get("boxes", []) or []),
                }
            )
    return rows


def gate(name: str, passed: bool, requirement: str, evidence: str) -> dict[str, Any]:
    return {
        "gate": name,
        "status": "pass" if passed else "fail",
        "requirement": requirement,
        "evidence": evidence,
    }


def scan_paper_evidence_forbidden_patterns() -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    if not PAPER_EVIDENCE_DIR.exists():
        return hits
    for path in sorted(PAPER_EVIDENCE_DIR.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".csv", ".json", ".jsonl", ".tex"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in FORBIDDEN_PAPER_PATTERNS:
            if pattern in text:
                hits.append(
                    {
                        "path": str(path.relative_to(ROOT)),
                        "pattern": pattern,
                        "sample_id": "",
                        "task": "",
                        "status": "",
                        "box_count": "",
                    }
                )
    return hits


def validation_metrics() -> dict[str, str]:
    if not FINAL_VALIDATION_SUMMARY.exists():
        return {}
    out = {}
    for row in read_csv(FINAL_VALIDATION_SUMMARY):
        if row.get("scope") == "all":
            out[row.get("metric", "")] = row.get("value", "")
    return out


def launch_metrics() -> dict[str, str]:
    if LAUNCH_SUMMARY.exists():
        out = {}
        for row in read_csv(LAUNCH_SUMMARY):
            if row.get("scope") == "all":
                out[row.get("metric", "")] = row.get("value", "")
        return out
    primary = read_jsonl(PRIMARY_PREFILL) if PRIMARY_PREFILL.exists() else []
    secondary = read_jsonl(SECONDARY_PREFILL) if SECONDARY_PREFILL.exists() else []
    return {
        "primary_rows": str(len(primary)),
        "secondary_calibration_rows": str(len(secondary)),
        "image_paths_exist": str(
            sum(Path(str(row.get("image_path", ""))).exists() for row in primary)
        ),
    }


def checklist_metrics() -> dict[str, str]:
    if not FINAL_DELIVERY_CHECKLIST.exists():
        return {}
    rows = read_csv(FINAL_DELIVERY_CHECKLIST)
    paper_required = [row for row in rows if str(row.get("required_for_paper", "")).strip().lower() == "yes"]
    return {
        "rows": str(len(rows)),
        "paper_required_rows": str(len(paper_required)),
    }


def progress_metrics() -> dict[str, str]:
    path = CURRENT_PROGRESS_DECISION if CURRENT_PROGRESS_DECISION.exists() else PROGRESS_DECISION
    if not path.exists():
        return {}
    rows = read_csv(path)
    if not rows:
        return {}
    return rows[0]


def format_validation(metrics: dict[str, str]) -> str:
    if not metrics:
        return "missing validation summary"
    keys = ["rows", "ready_for_ecr_rows", "invalid_box_rows", "unlabeled_box_rows", "empty_rows"]
    return ", ".join(f"{key}={metrics.get(key, '')}" for key in keys)


def format_launch(metrics: dict[str, str], checklist: dict[str, str], progress: dict[str, str]) -> str:
    if not metrics:
        return "missing launch summary"
    keys = ["primary_rows", "secondary_calibration_rows", "image_paths_exist"]
    values = ", ".join(f"{key}={metrics.get(key, '')}" for key in keys)
    progress_values = "missing progress dashboard"
    if progress:
        status = progress.get("manual_annotation_progress_status", "") or progress.get("progress_status", "")
        progress_values = (
            f"progress_status={status}, "
            f"primary_ready_rows={progress.get('primary_ready_rows', '')}/{progress.get('total_primary_rows', '')}, "
            f"calibration_ready_rows={progress.get('calibration_ready_rows', '')}/{progress.get('total_calibration_rows', '')}"
        )
    return (
        f"{values}, primary_tool={PRIMARY_TOOL_HTML.exists()}, "
        f"secondary_tool={SECONDARY_TOOL_HTML.exists()}, "
        f"handbook={ANNOTATOR_HANDBOOK.exists()}, "
        f"checklist_rows={checklist.get('rows', '0')}, "
        f"paper_required_checklist_rows={checklist.get('paper_required_rows', '0')}, "
        f"{progress_values}"
    )


def is_smoke_like_annotation(row: dict[str, Any]) -> bool:
    labels = []
    boxes = row.get("boxes", [])
    if not boxes and isinstance(row.get("adjudicated_boxes"), list):
        boxes = row.get("adjudicated_boxes", [])
    if not isinstance(boxes, list):
        return False
    for box in boxes:
        if isinstance(box, dict):
            labels.append(str(box.get("label", "")))
    words = set(re.findall(r"[a-z0-9_]+", " ".join(labels).lower()))
    note_words = set(re.findall(r"[a-z0-9_]+", str(row.get("notes", "")).lower()))
    return bool((words | note_words) & {"foo", "bar", "smoke"})


def build_report(gates: list[dict[str, Any]], details: list[dict[str, Any]]) -> str:
    passed = sum(row["status"] == "pass" for row in gates)
    lines = [
        "# Manual Evidence Readiness Gate",
        "",
        "This audit checks whether the manual multi-region evidence artifacts are ready to support problem.md claims. It also ensures synthetic smoke annotations are not used as paper-facing evidence.",
        "",
        f"Passed gates: {passed}/{len(gates)}.",
        "",
        "| Gate | Status | Requirement | Evidence |",
        "| --- | --- | --- | --- |",
    ]
    for row in gates:
        lines.append(f"| {row['gate']} | {row['status']} | {row['requirement']} | {row['evidence']} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        "All final-manual-annotation gates must pass before the 96-row human audit is paper-facing. The raw launch pack remains deliberately unmodified; completed annotations are stored and validated in the separate final package.",
            "",
            "## Detail Rows",
            "",
            "| Category | Path | Pattern | Sample | Task | Status | Boxes |",
            "| --- | --- | --- | --- | --- | --- | ---: |",
        ]
    )
    for row in details[:100]:
        lines.append(
            f"| {row.get('category', '')} | {row.get('path', '')} | {row.get('pattern', '')} | "
            f"{row.get('sample_id', '')} | {row.get('task', '')} | {row.get('status', '')} | "
            f"{row.get('box_count', '')} |"
        )
    if len(details) > 100:
        lines.append(f"| omitted | {len(details) - 100} additional rows |  |  |  |  |  |")
    return "\n".join(lines) + "\n"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


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
