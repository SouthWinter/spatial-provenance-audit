#!/usr/bin/env python3
"""Build a reproducible manual multi-evidence annotation plan."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(PREFILL_JSONL))
    parser.add_argument(
        "--output-dir",
        default="runs/problem_optimization_audit/open_ocr_qa_manual_annotation_protocol",
    )
    parser.add_argument("--batch-size", type=int, default=24)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.input))
    planned = assign_batches(rows, args.batch_size)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(out_dir / "manual_annotation_batches.csv", planned)
    write_csv(out_dir / "manual_annotation_summary.csv", summary_rows(planned))
    write_csv(out_dir / "manual_annotation_calibration_subset.csv", calibration_subset(planned))
    for batch in sorted({row["annotation_batch"] for row in planned}):
        write_jsonl(
            out_dir / f"{batch}_seed.jsonl",
            [seed_row(row) for row in planned if row["annotation_batch"] == batch],
        )
    (out_dir / "manual_annotation_protocol.md").write_text(build_protocol(planned), encoding="utf-8")
    print(f"Wrote manual annotation protocol for {len(planned)} rows to {out_dir}")


def assign_batches(rows: list[dict[str, Any]], batch_size: int) -> list[dict[str, Any]]:
    enriched = []
    for row in rows:
        reasons = split_tags(row.get("selection_reasons", ""))
        stress_tags = split_tags(row.get("stress_tags", ""))
        priority = priority_score(reasons, stress_tags, row)
        enriched.append(
            {
                **row,
                "priority_score": priority,
                "annotation_batch": "",
                "double_annotation": "no",
                "annotation_priority": priority_label(priority),
                "required_region_policy": region_policy(row),
                "box_label_policy": "type:text, e.g. answer_value:3973 or row_header:Follow-up suggestions",
            }
        )

    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in enriched:
        buckets.setdefault((row["task"], row.get("prefill_complexity", "")), []).append(row)
    for bucket_rows in buckets.values():
        bucket_rows.sort(key=lambda r: (-int(r["priority_score"]), r["sample_id"]))

    ordered: list[dict[str, Any]] = []
    while any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key]:
                ordered.append(buckets[key].pop(0))

    for i, row in enumerate(ordered):
        row["annotation_batch"] = f"batch_{i // batch_size + 1:02d}"

    calibration_ids = {row["sample_id"] for row in calibration_subset(ordered)}
    for row in ordered:
        if row["sample_id"] in calibration_ids:
            row["double_annotation"] = "yes"
    return ordered


def priority_score(reasons: list[str], stress_tags: list[str], row: dict[str, Any]) -> int:
    score = 0
    if "persistent_failure" in reasons:
        score += 4
    if "recovered_by_70" in reasons:
        score += 3
    if "low_budget_failure" in reasons:
        score += 2
    if "mid_budget_failure" in reasons:
        score += 1
    if "layout_field_question" in stress_tags:
        score += 3
    if "multi_constraint_question" in stress_tags:
        score += 2
    if "multi_token_answer_ge3" in stress_tags:
        score += 1
    if int(row.get("prefill_has_numeric", 0)):
        score += 1
    if row.get("prefill_complexity") in {"hard_persistent", "likely_multi_region"}:
        score += 2
    return score


def priority_label(score: int) -> str:
    if score >= 8:
        return "highest"
    if score >= 5:
        return "high"
    if score >= 3:
        return "medium"
    return "standard"


def region_policy(row: dict[str, Any]) -> str:
    stress = split_tags(row.get("stress_tags", ""))
    if "layout_field_question" in stress or "multi_constraint_question" in stress:
        return "mark answer value plus all necessary field labels, row/column headers, and comparison anchors"
    if row.get("prefill_complexity") == "likely_multi_region":
        return "mark every discontiguous answer span or supporting context region separately"
    return "mark the minimal visible answer evidence; add label/context boxes only if needed to disambiguate"


def calibration_subset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for task in ["DocVQA-lite", "TextVQA-lite"]:
        task_rows = [row for row in rows if row["task"] == task]
        hard = [row for row in task_rows if row["annotation_priority"] in {"highest", "high"}]
        rest = [row for row in task_rows if row not in hard]
        selected.extend((hard + rest)[:6])
    return selected


def seed_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": row["sample_id"],
        "task": row["task"],
        "question_id": row["question_id"],
        "annotation_batch": row["annotation_batch"],
        "double_annotation": row["double_annotation"],
        "evidence_units": row["prefill_evidence_units"],
        "required_region_policy": row["required_region_policy"],
        "boxes": [],
        "notes": "",
        "status": "unannotated",
    }


def summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = [
        {"scope": "all", "metric": "rows", "value": len(rows)},
        {"scope": "all", "metric": "double_annotation_rows", "value": sum(row["double_annotation"] == "yes" for row in rows)},
    ]
    for batch in sorted({row["annotation_batch"] for row in rows}):
        batch_rows = [row for row in rows if row["annotation_batch"] == batch]
        out.append({"scope": batch, "metric": "rows", "value": len(batch_rows)})
        for task in sorted({row["task"] for row in batch_rows}):
            out.append({"scope": batch, "metric": f"task_{task}", "value": sum(row["task"] == task for row in batch_rows)})
        out.append({"scope": batch, "metric": "highest_or_high_priority", "value": sum(row["annotation_priority"] in {"highest", "high"} for row in batch_rows)})
    for task in sorted({row["task"] for row in rows}):
        task_rows = [row for row in rows if row["task"] == task]
        out.append({"scope": task, "metric": "rows", "value": len(task_rows)})
        for complexity in sorted({row.get("prefill_complexity", "") for row in task_rows}):
            out.append({"scope": task, "metric": f"complexity_{complexity}", "value": sum(row.get("prefill_complexity", "") == complexity for row in task_rows)})
    return out


def build_protocol(rows: list[dict[str, Any]]) -> str:
    summary = summary_rows(rows)
    lines = [
        "# Manual Multi-Evidence Annotation Protocol",
        "",
        "This protocol turns the fixed 96-row TextVQA-lite/DocVQA-lite stress pack into a reproducible manual evidence-region annotation task. It is a preparation artifact only; the paper should not claim completed manual multi-evidence results until exported annotations pass validation and ECR evaluation.",
        "",
        "## Goal",
        "",
        "For each question, draw the minimal visible image regions that a reader would need to answer correctly. The target is not merely the answer string. If a table header, row label, field name, comparison anchor, or nearby context is necessary to know which value is correct, annotate that context as a separate evidence region.",
        "",
        "## Region Rules",
        "",
        "- Use one box per contiguous evidence region.",
        "- Label every box as `type:text`, for example `answer_value:3973`, `field_label:Purchase Order Number`, `row_header:26`, `column_header:1969`, `comparison_anchor:sugar`, or `context:Follow-up suggestions`.",
        "- Mark all discontiguous answer spans, all needed row/column headers, and any field label that disambiguates the answer.",
        "- Do not draw a whole page or large paragraph unless the evidence is genuinely not localizable; if unavoidable, explain why in `notes`.",
        "- Use `status=annotated` only when all necessary regions are boxed. Use `status=uncertain` and notes when the answer is not visibly supported or the image is unreadable.",
        "",
        "## Quality Gates",
        "",
        "- Validation must report 96 rows, 0 invalid-box rows, and no unlabeled boxes.",
        "- All `annotated` rows must have at least one box.",
        "- The 12-row calibration subset should be independently double annotated before the remaining rows are finalized.",
        "- Disagreements should be resolved by checking whether missing boxes change all-region ECR or answer interpretation.",
        "",
        "## Double-Annotation Agreement",
        "",
        "After two annotators export the 12 calibration rows, run:",
        "",
        "```bash",
        "python scripts/audit_open_ocr_qa_annotation_agreement.py \\",
        "  --annotations-a path/to/annotator_a.json \\",
        "  --annotations-b path/to/annotator_b.json \\",
        "  --output-dir runs/problem_optimization_audit/open_ocr_qa_annotation_agreement",
        "```",
        "",
        "Rows with different box counts, different evidence-label type sets, missing boxes, or low box IoU should be adjudicated before full annotation. Agreement is a quality gate for the annotation process, not a scientific result by itself.",
        "",
        "To split agreed rows from rows needing manual adjudication, run:",
        "",
        "```bash",
        "python scripts/build_open_ocr_qa_annotation_adjudication_package.py \\",
        "  --annotations-a path/to/annotator_a.json \\",
        "  --annotations-b path/to/annotator_b.json \\",
        "  --output-dir runs/problem_optimization_audit/open_ocr_qa_annotation_adjudication",
        "```",
        "",
        "This creates `consensus_draft.json` for agreed rows and `adjudication_queue.jsonl` for rows that need a final human decision.",
        "",
        "After adjudication fills `adjudicated_boxes`, merge the final annotation file and run the downstream checks:",
        "",
        "```bash",
        "python scripts/finalize_open_ocr_qa_annotations.py \\",
        "  --consensus runs/problem_optimization_audit/open_ocr_qa_annotation_adjudication/consensus_draft.json \\",
        "  --adjudication runs/problem_optimization_audit/open_ocr_qa_annotation_adjudication/adjudication_queue.jsonl \\",
        "  --output-dir runs/problem_optimization_audit/open_ocr_qa_final_annotations",
        "python scripts/validate_open_ocr_qa_bbox_annotations.py \\",
        "  --annotations runs/problem_optimization_audit/open_ocr_qa_final_annotations/final_annotations.json \\",
        "  --output-dir runs/problem_optimization_audit/open_ocr_qa_final_annotations_validation",
        "python scripts/evaluate_open_ocr_qa_bbox_ecr.py \\",
        "  --annotations runs/problem_optimization_audit/open_ocr_qa_final_annotations/final_annotations.json \\",
        "  --output-dir runs/problem_optimization_audit/open_ocr_qa_final_annotations_ecr",
        "python scripts/build_open_ocr_qa_manual_evidence_audit.py \\",
        "  --ecr-rows runs/problem_optimization_audit/open_ocr_qa_final_annotations_ecr/bbox_ecr_rows.csv \\",
        "  --output-dir runs/problem_optimization_audit/open_ocr_qa_manual_evidence_audit",
        "```",
        "",
        "The final file is ready for paper evidence only if validation reports no invalid, unlabeled, empty, or unresolved annotation rows. The final evidence audit should be described as evidence availability and quality association, not as causal proof of model evidence use.",
        "",
        "## Files",
        "",
        "- `manual_annotation_batches.csv`: full ordered worklist with batch, priority, stress tags, questions, model outputs, and image paths.",
        "- `manual_annotation_calibration_subset.csv`: 12 rows for double annotation.",
        "- `batch_XX_seed.jsonl`: per-batch JSONL seeds compatible with the annotation schema.",
        "- HTML tool: `runs/problem_optimization_audit/open_ocr_qa_bbox_annotation_tool/index.html`.",
        "",
        "## Batch Summary",
        "",
        "| Scope | Metric | Value |",
        "| --- | --- | ---: |",
    ]
    for row in summary:
        lines.append(f"| {row['scope']} | {row['metric']} | {row['value']} |")
    return "\n".join(lines) + "\n"


def split_tags(text: Any) -> list[str]:
    return [part for part in str(text or "").split(";") if part]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
