#!/usr/bin/env python3
"""Merge consensus draft and adjudicated rows into final annotation JSON."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consensus", required=True, help="consensus_draft.json from adjudication package")
    parser.add_argument("--adjudication", required=True, help="adjudication_queue.jsonl/json after human adjudication")
    parser.add_argument(
        "--output-dir",
        default="runs/problem_optimization_audit/open_ocr_qa_final_annotations",
    )
    parser.add_argument(
        "--allow-unresolved",
        action="store_true",
        help="Include unresolved adjudication rows as status=unresolved instead of failing",
    )
    args = parser.parse_args()

    consensus_rows = load_annotations(Path(args.consensus))
    adjudication_rows = load_annotations(Path(args.adjudication))
    final_rows = [normalize_consensus(row) for row in consensus_rows]
    unresolved = []
    for row in adjudication_rows:
        boxes = row.get("adjudicated_boxes", [])
        if isinstance(boxes, list) and boxes:
            final_rows.append(normalize_adjudicated(row))
        else:
            unresolved.append(row)
            if args.allow_unresolved:
                final_rows.append(normalize_unresolved(row))

    if unresolved and not args.allow_unresolved:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_csv(out_dir / "unresolved_adjudication_rows.csv", unresolved_summary(unresolved))
        raise SystemExit(
            f"{len(unresolved)} adjudication rows have no adjudicated_boxes; "
            f"wrote unresolved_adjudication_rows.csv. Use --allow-unresolved only for dry runs."
        )

    final_rows.sort(key=lambda row: (str(row.get("task", "")), str(row.get("sample_id", ""))))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "final_annotations.json", final_rows)
    write_jsonl(out_dir / "final_annotations.jsonl", final_rows)
    write_csv(out_dir / "final_annotation_summary.csv", summary_rows(final_rows, unresolved))
    (out_dir / "final_annotation_report.md").write_text(
        build_report(final_rows, unresolved, args.allow_unresolved), encoding="utf-8"
    )
    print(f"Final annotations={len(final_rows)}; unresolved={len(unresolved)}")


def normalize_consensus(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": row.get("sample_id", ""),
        "task": row.get("task", ""),
        "question_id": row.get("question_id", ""),
        "status": "annotated",
        "boxes": row.get("boxes", []),
        "notes": row.get("notes", ""),
        "source": "consensus_draft",
    }


def normalize_adjudicated(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": row.get("sample_id", ""),
        "task": row.get("task", ""),
        "question_id": row.get("question_id", ""),
        "status": "annotated",
        "boxes": row.get("adjudicated_boxes", []),
        "notes": row.get("adjudication_notes", ""),
        "source": "human_adjudication",
    }


def normalize_unresolved(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": row.get("sample_id", ""),
        "task": row.get("task", ""),
        "question_id": row.get("question_id", ""),
        "status": "unresolved",
        "boxes": [],
        "notes": "unresolved adjudication row",
        "source": "unresolved_adjudication",
    }


def summary_rows(final_rows: list[dict[str, Any]], unresolved: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {"scope": "all", "metric": "final_rows", "value": len(final_rows)},
        {"scope": "all", "metric": "annotated_rows", "value": sum(bool(row.get("boxes")) for row in final_rows)},
        {"scope": "all", "metric": "unresolved_adjudication_rows", "value": len(unresolved)},
        {"scope": "all", "metric": "consensus_source_rows", "value": sum(row.get("source") == "consensus_draft" for row in final_rows)},
        {"scope": "all", "metric": "adjudicated_source_rows", "value": sum(row.get("source") == "human_adjudication" for row in final_rows)},
    ]
    for task in sorted({row.get("task", "") for row in final_rows if row.get("task", "")}):
        task_rows = [row for row in final_rows if row.get("task", "") == task]
        rows.append({"scope": task, "metric": "final_rows", "value": len(task_rows)})
        rows.append({"scope": task, "metric": "annotated_rows", "value": sum(bool(row.get("boxes")) for row in task_rows)})
    return rows


def unresolved_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        out.append(
            {
                "sample_id": row.get("sample_id", ""),
                "task": row.get("task", ""),
                "question": row.get("question", ""),
                "reason": row.get("agreement", {}).get("needs_adjudication", ""),
                "evidence_units": row.get("evidence_units", ""),
            }
        )
    return out


def build_report(final_rows: list[dict[str, Any]], unresolved: list[dict[str, Any]], allow_unresolved: bool) -> str:
    summary = summary_rows(final_rows, unresolved)
    lines = [
        "# Final Open OCR QA Annotations",
        "",
        "This file merges auto-accepted consensus rows with human-adjudicated rows into the schema consumed by the bbox ECR evaluator.",
        "",
        f"Unresolved rows allowed: {allow_unresolved}.",
        "",
        "| Scope | Metric | Value |",
        "| --- | --- | ---: |",
    ]
    for row in summary:
        lines.append(f"| {row['scope']} | {row['metric']} | {row['value']} |")
    if unresolved:
        lines.extend(["", "## Unresolved Rows", "", "| Sample | Task | Evidence units |", "| --- | --- | --- |"])
        for row in unresolved:
            lines.append(f"| {row.get('sample_id', '')} | {row.get('task', '')} | {row.get('evidence_units', '')} |")
    return "\n".join(lines) + "\n"


def load_annotations(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
