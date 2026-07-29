#!/usr/bin/env python3
"""Build a launch package for final manual multi-evidence annotation.

The existing annotation protocol describes the task. This script packages the
actual files and commands needed to run it without accidentally using smoke or
detector-derived boxes as final human evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PREFILL_JSONL = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_evidence_prefill"
    / "evidence_prefill_pack.jsonl"
)
BATCH_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_manual_annotation_protocol"
    / "manual_annotation_batches.csv"
)
CALIBRATION_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_manual_annotation_protocol"
    / "manual_annotation_calibration_subset.csv"
)
TOOL_DIR = ROOT / "runs" / "problem_optimization_audit" / "open_ocr_qa_bbox_annotation_tool"
DEFAULT_OUT = ROOT / "runs" / "problem_optimization_audit" / "open_ocr_qa_manual_annotation_launch"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    prefill = {row["sample_id"]: row for row in read_jsonl(PREFILL_JSONL)}
    batches = read_csv(BATCH_CSV)
    calibration_ids = {row["sample_id"] for row in read_csv(CALIBRATION_CSV)}
    rows = [merge_row(row, prefill[row["sample_id"]]) for row in batches if row["sample_id"] in prefill]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    primary_rows = rows
    calibration_rows = [row for row in rows if row["sample_id"] in calibration_ids]
    write_jsonl(out_dir / "primary_full_prefill.jsonl", primary_rows)
    write_jsonl(out_dir / "secondary_calibration_prefill.jsonl", calibration_rows)
    write_jsonl(out_dir / "primary_full_seed.jsonl", [seed_row(row, "primary") for row in primary_rows])
    write_jsonl(out_dir / "secondary_calibration_seed.jsonl", [seed_row(row, "secondary_calibration") for row in calibration_rows])

    for batch in sorted({row["annotation_batch"] for row in rows}):
        batch_rows = [row for row in rows if row["annotation_batch"] == batch]
        write_jsonl(out_dir / f"{batch}_primary_prefill.jsonl", batch_rows)
        write_jsonl(out_dir / f"{batch}_primary_seed.jsonl", [seed_row(row, "primary") for row in batch_rows])

    write_csv(out_dir / "launch_summary.csv", summary_rows(rows, calibration_rows))
    write_csv(out_dir / "calibration_manifest.csv", calibration_manifest(calibration_rows))
    write_csv(out_dir / "final_delivery_checklist.csv", final_delivery_checklist())
    (out_dir / "annotator_handbook.md").write_text(build_annotator_handbook(rows, calibration_rows), encoding="utf-8")
    (out_dir / "manual_annotation_launch.md").write_text(
        build_launch_markdown(out_dir, rows, calibration_rows), encoding="utf-8"
    )
    (out_dir / "commands.sh").write_text(build_commands(out_dir), encoding="utf-8")
    print(f"Wrote manual annotation launch package to {out_dir}")


def merge_row(batch_row: dict[str, str], prefill_row: dict[str, Any]) -> dict[str, Any]:
    row = dict(prefill_row)
    for key, value in batch_row.items():
        if value != "":
            row[key] = value
    return row


def seed_row(row: dict[str, Any], annotator_role: str) -> dict[str, Any]:
    return {
        "sample_id": row["sample_id"],
        "task": row["task"],
        "question_id": row["question_id"],
        "annotator_role": annotator_role,
        "annotation_batch": row.get("annotation_batch", ""),
        "double_annotation": row.get("double_annotation", ""),
        "evidence_units": row.get("prefill_evidence_units", ""),
        "required_region_policy": row.get("required_region_policy", ""),
        "boxes": [],
        "notes": "",
        "status": "unannotated",
    }


def summary_rows(rows: list[dict[str, Any]], calibration_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [
        {"scope": "all", "metric": "primary_rows", "value": len(rows)},
        {"scope": "all", "metric": "secondary_calibration_rows", "value": len(calibration_rows)},
        {"scope": "all", "metric": "image_paths_exist", "value": sum(Path(str(row.get("image_path", ""))).exists() for row in rows)},
    ]
    for task in sorted({row["task"] for row in rows}):
        task_rows = [row for row in rows if row["task"] == task]
        cal_rows = [row for row in calibration_rows if row["task"] == task]
        out.append({"scope": task, "metric": "primary_rows", "value": len(task_rows)})
        out.append({"scope": task, "metric": "secondary_calibration_rows", "value": len(cal_rows)})
    for batch in sorted({row.get("annotation_batch", "") for row in rows}):
        batch_rows = [row for row in rows if row.get("annotation_batch", "") == batch]
        out.append({"scope": batch, "metric": "primary_rows", "value": len(batch_rows)})
    return out


def calibration_manifest(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        out.append(
            {
                "sample_id": row["sample_id"],
                "task": row["task"],
                "question_id": row["question_id"],
                "annotation_batch": row.get("annotation_batch", ""),
                "question": row.get("question", ""),
                "gold_answers": row.get("gold_answers", ""),
                "prefill_evidence_units": row.get("prefill_evidence_units", ""),
                "prefill_complexity": row.get("prefill_complexity", ""),
                "selection_reasons": row.get("selection_reasons", ""),
                "stress_tags": row.get("stress_tags", ""),
                "image_path": row.get("image_path", ""),
            }
        )
    return out


def final_delivery_checklist() -> list[dict[str, str]]:
    return [
        {
            "stage": "before_annotation",
            "check": "primary_and_secondary_tools_open",
            "pass_condition": "primary_full_tool/index.html and secondary_calibration_tool/index.html load images and row metadata",
            "required_for_paper": "yes",
        },
        {
            "stage": "primary_annotation",
            "check": "all_primary_rows_annotated",
            "pass_condition": "96 rows have status=annotated",
            "required_for_paper": "yes",
        },
        {
            "stage": "primary_annotation",
            "check": "no_empty_evidence",
            "pass_condition": "each primary row has at least one valid evidence box",
            "required_for_paper": "yes",
        },
        {
            "stage": "primary_annotation",
            "check": "multi_region_context_marked",
            "pass_condition": "field labels, row/column headers, comparison anchors, and discontiguous spans are boxed when needed",
            "required_for_paper": "yes",
        },
        {
            "stage": "secondary_calibration",
            "check": "secondary_rows_complete",
            "pass_condition": "12 calibration rows have independent secondary annotations",
            "required_for_paper": "yes",
        },
        {
            "stage": "agreement",
            "check": "calibration_agreement_reviewed",
            "pass_condition": "missing or low-overlap calibration rows are adjudicated before promotion",
            "required_for_paper": "yes",
        },
        {
            "stage": "progress_audit",
            "check": "progress_dashboard_ready",
            "pass_condition": "manual annotation progress audit reports all primary rows and calibration rows ready before promotion",
            "required_for_paper": "yes",
        },
        {
            "stage": "validation",
            "check": "validation_ready_for_ecr",
            "pass_condition": "validator reports all final rows ready for ECR",
            "required_for_paper": "yes",
        },
        {
            "stage": "promotion",
            "check": "final_annotations_promoted",
            "pass_condition": "final_annotations.jsonl exists and contains no smoke placeholders",
            "required_for_paper": "yes",
        },
        {
            "stage": "paper_evidence",
            "check": "final_package_gate_passes",
            "pass_condition": "build_open_ocr_qa_manual_final_package.py --build-derived --require-ready validates, scores, audits, and copies manual tables into runs/paper_evidence",
            "required_for_paper": "yes",
        },
    ]


def build_annotator_handbook(rows: list[dict[str, Any]], calibration_rows: list[dict[str, Any]]) -> str:
    task_counts: dict[str, int] = {}
    complexity_counts: dict[str, int] = {}
    for row in rows:
        task_counts[str(row.get("task", ""))] = task_counts.get(str(row.get("task", "")), 0) + 1
        complexity = str(row.get("prefill_complexity", "unknown"))
        complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
    lines = [
        "# Annotator Handbook for Manual Multi-Region Evidence",
        "",
        "This handbook defines the final human evidence annotation used to close the multi-region evidence gap in `problem.md`. It is for human evidence boxes only. Do not copy detector boxes, external boxes, smoke annotations, or model-generated boxes into the final export.",
        "",
        "## Scope",
        "",
        f"- Primary set: {len(rows)} rows.",
        f"- Secondary calibration set: {len(calibration_rows)} rows.",
        *[f"- {task}: {count} primary rows." for task, count in sorted(task_counts.items())],
        *[f"- Complexity {name}: {count} rows." for name, count in sorted(complexity_counts.items())],
        "",
        "## What Counts as Evidence",
        "",
        "Mark the minimal visible image region(s) that a careful human would need to answer the question. Evidence is not limited to the final answer string. Include context when it is required to identify, disambiguate, compare, or read the answer.",
        "",
        "Use one box per contiguous region. Draw separate boxes for discontiguous evidence.",
        "",
        "Required evidence types:",
        "",
        "- `answer_value`: the visible answer text, number, symbol, or phrase.",
        "- `field_label`: the label that connects an answer value to the question, such as Purchase Order Number or Total.",
        "- `row_header`: the row identifier needed to select the right row.",
        "- `column_header`: the column identifier needed to select the right column or year.",
        "- `comparison_anchor`: a competing item or reference value needed for comparison questions.",
        "- `context`: nearby text needed to disambiguate the answer when no tighter label applies.",
        "",
        "## Decision Rules",
        "",
        "- If the question asks for a field value, box both the value and the field label unless the value is uniquely identifiable without the label.",
        "- If the question refers to a row, serial number, person, product, date, year, or category, box the row/column/header anchor as a separate region.",
        "- If multiple visible strings could answer the question, box the disambiguating context, not only the selected answer.",
        "- If the answer spans multiple words on the same line, one tight phrase box is acceptable. If the words are separated or wrap across lines, use multiple boxes.",
        "- If the answer requires comparing two or more regions, box every compared region plus the final answer region.",
        "- If the answer is inferred from a chart/table label and value, box the label, axis/table header, and value region needed for the inference.",
        "- If the evidence is unreadable or impossible to localize, set `status=needs_review`, leave a note, and draw the smallest uncertain region if possible.",
        "- Do not box the whole document unless the evidence is genuinely diffuse and cannot be localized; explain this in notes.",
        "",
        "## Box Labels",
        "",
        "Use `type:text` labels. Examples:",
        "",
        "- `answer_value:3973`",
        "- `field_label:Purchase Order Number`",
        "- `row_header:26`",
        "- `column_header:1969`",
        "- `comparison_anchor:sugar`",
        "- `context:Follow-up suggestions`",
        "",
        "Every box must have a label. Unlabeled boxes should fail validation or adjudication.",
        "",
        "## Status Values",
        "",
        "- `annotated`: all required evidence regions are boxed and labeled.",
        "- `needs_review`: the row is ambiguous, unreadable, or requires adjudication.",
        "- `not_visible`: the requested evidence cannot be found in the exported image.",
        "- `in_progress`: temporary status only; not accepted for final promotion.",
        "",
        "Only `annotated` rows can be promoted to final paper evidence without adjudication.",
        "",
        "## Calibration and Adjudication",
        "",
        "The 12 calibration rows are independently annotated by a secondary annotator. They are not a separate result table by themselves; they are a quality-control gate for the 96-row primary set.",
        "",
        "Adjudicate before promotion when:",
        "",
        "- primary or secondary annotation is missing;",
        "- one annotator marks no evidence;",
        "- labels disagree on required context type;",
        "- a multi-region question is annotated with only the final answer value;",
        "- the secondary annotation reveals a missing row/column/header/comparison anchor.",
        "",
        "## Final Export Requirements",
        "",
        "- 96 primary rows exported.",
        "- 12 secondary calibration rows exported.",
        "- Progress dashboard reports 96/96 primary rows and 12/12 calibration rows ready.",
        "- All primary rows have `status=annotated` after adjudication.",
        "- Every final primary row has at least one valid labeled box.",
        "- No smoke placeholders, detector-derived labels, or empty boxes.",
        "- Final validation, ECR projection, manual evidence audit, and final-package gate all pass before any paper-facing manual table is copied.",
        "",
    ]
    return "\n".join(lines)


def build_launch_markdown(out_dir: Path, rows: list[dict[str, Any]], calibration_rows: list[dict[str, Any]]) -> str:
    rel = lambda path: str(path.relative_to(ROOT))
    lines = [
        "# Manual Multi-Evidence Annotation Launch",
        "",
        "This package is the paper-facing path for closing the manual multi-region evidence gap in `problem.md`. It intentionally starts from empty human annotation seeds; detector boxes, external boxes, and smoke annotations are not used as final evidence.",
        "",
        "## What To Annotate",
        "",
        f"- Primary annotator: {len(rows)} rows in `primary_full_prefill.jsonl`.",
        f"- Secondary annotator: {len(calibration_rows)} calibration rows in `secondary_calibration_prefill.jsonl`.",
        "- Draw the minimal visible regions needed to answer the question, including row headers, column headers, field labels, comparison anchors, and nearby context when they disambiguate the answer.",
        "- Use `status=annotated` only when all required regions are boxed. Do not leave unlabeled boxes.",
        "",
        "## Files",
        "",
        f"- Primary prefill: `{rel(out_dir / 'primary_full_prefill.jsonl')}`",
        f"- Secondary calibration prefill: `{rel(out_dir / 'secondary_calibration_prefill.jsonl')}`",
        f"- Primary empty seed: `{rel(out_dir / 'primary_full_seed.jsonl')}`",
        f"- Secondary empty seed: `{rel(out_dir / 'secondary_calibration_seed.jsonl')}`",
        f"- Calibration manifest: `{rel(out_dir / 'calibration_manifest.csv')}`",
        f"- Annotator handbook: `{rel(out_dir / 'annotator_handbook.md')}`",
        f"- Final delivery checklist: `{rel(out_dir / 'final_delivery_checklist.csv')}`",
        f"- Static annotation tool source: `{rel(TOOL_DIR / 'index.html')}`",
        "",
        "## Recommended Workflow",
        "",
        "1. Build a primary full annotation tool from `primary_full_prefill.jsonl` and export the completed JSON as `primary_full_export.json`.",
        "2. Build a secondary calibration tool from `secondary_calibration_prefill.jsonl` and export the completed JSON as `secondary_calibration_export.json`.",
        "3. Validate the primary export, audit agreement on the 12 calibration rows, and only then promote the primary export to final manual annotations.",
        "4. Run the progress dashboard after any partial or final export; it lists incomplete batches and rows blocking promotion.",
        "5. Run the strict final package command with `--build-derived --require-ready`. It derives validation, ECR, manual-evidence quality association, readiness gates, and paper-evidence tables from the promoted final annotations.",
        "",
        "## Completion Gates",
        "",
        "- Primary export has 96 rows, all `status=annotated`, at least one box per row, no invalid boxes, and no unlabeled boxes.",
        "- Calibration agreement has 12 rows and no missing secondary rows; disagreements are reviewed before promotion.",
        "- Progress dashboard has zero blocker rows.",
        "- Final annotations validate as ready for ECR.",
        "- Final ECR tables report all-region recall and worst-region ECR for each keep ratio.",
        "- Manual-evidence quality tables join evidence availability with native open-answer generation quality.",
        "- `audit_problem_md_manual_evidence_readiness.py` passes the final manual annotation gates.",
        "",
        "## Commands",
        "",
        "The full command list is in `commands.sh`. Paths containing `TO_FILL` are intentionally left for human export files.",
        "",
        "```bash",
        f"bash {rel(out_dir / 'commands.sh')}",
        "```",
        "",
    ]
    return "\n".join(lines)


def build_commands(out_dir: Path) -> str:
    rel = lambda path: str(path.relative_to(ROOT))
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            "# Build task-specific static annotation tools.",
            f"python scripts/build_open_ocr_qa_bbox_annotation_tool.py --input {rel(out_dir / 'primary_full_prefill.jsonl')} --output-dir runs/problem_optimization_audit/open_ocr_qa_manual_annotation_launch/primary_full_tool",
            f"python scripts/build_open_ocr_qa_bbox_annotation_tool.py --input {rel(out_dir / 'secondary_calibration_prefill.jsonl')} --output-dir runs/problem_optimization_audit/open_ocr_qa_manual_annotation_launch/secondary_calibration_tool",
            "",
            "# After human annotation, set these paths to the exported JSON files.",
            "PRIMARY_EXPORT=TO_FILL/primary_full_export.json",
            "SECONDARY_EXPORT=TO_FILL/secondary_calibration_export.json",
            "",
            "# Validate the primary full export.",
            "python scripts/validate_open_ocr_qa_bbox_annotations.py \\",
            "  --annotations \"$PRIMARY_EXPORT\" \\",
            "  --output-dir runs/problem_optimization_audit/open_ocr_qa_manual_annotation_launch/primary_validation",
            "",
            "# Track completion and blocker rows before promotion.",
            "python scripts/audit_open_ocr_qa_manual_annotation_progress.py \\",
            "  --primary-export \"$PRIMARY_EXPORT\" \\",
            "  --secondary-export \"$SECONDARY_EXPORT\" \\",
            "  --output-dir runs/problem_optimization_audit/open_ocr_qa_manual_annotation_launch/progress_audit",
            "",
            "# Audit 12-row calibration agreement.",
            "python scripts/audit_open_ocr_qa_annotation_agreement.py \\",
            "  --annotations-a \"$PRIMARY_EXPORT\" \\",
            "  --annotations-b \"$SECONDARY_EXPORT\" \\",
            "  --output-dir runs/problem_optimization_audit/open_ocr_qa_manual_annotation_launch/calibration_agreement",
            "",
            "# Promote the validated primary export to final manual annotations.",
            "python scripts/promote_open_ocr_qa_primary_annotations.py \\",
            "  --annotations \"$PRIMARY_EXPORT\" \\",
            "  --agreement-summary runs/problem_optimization_audit/open_ocr_qa_manual_annotation_launch/calibration_agreement/annotation_agreement_summary.csv \\",
            "  --output-dir runs/problem_optimization_audit/open_ocr_qa_final_annotations",
            "",
            "# Final validation, ECR evaluation, manual-evidence quality audit,",
            "# readiness gates, and paper-evidence copy. This command is strict:",
            "# it fails if final human annotations are missing, incomplete, invalid,",
            "# unlabeled, smoke-like, or not ready for ECR.",
            "python scripts/build_open_ocr_qa_manual_final_package.py --build-derived --require-ready",
            "",
        ]
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
