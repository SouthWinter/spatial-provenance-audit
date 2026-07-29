#!/usr/bin/env python3
"""Track progress for final manual multi-region evidence annotation.

This is a process dashboard, not a paper result. It reads the launch package
and optional human annotation exports, then reports batch/task completion,
calibration coverage, and rows that block promotion to final manual evidence.
When no export exists it falls back to the empty launch seeds and reports the
current 0/96 state explicitly.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LAUNCH_DIR = ROOT / "runs" / "problem_optimization_audit" / "open_ocr_qa_manual_annotation_launch"
DEFAULT_OUTPUT = ROOT / "runs" / "problem_optimization_audit" / "open_ocr_qa_manual_annotation_progress"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-dir", default=str(LAUNCH_DIR))
    parser.add_argument("--primary-export", default="", help="Primary annotation JSON/JSONL export; defaults to launch seed")
    parser.add_argument("--secondary-export", default="", help="Secondary calibration JSON/JSONL export; defaults to launch seed")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    launch_dir = Path(args.launch_dir)
    primary_path = Path(args.primary_export) if args.primary_export else launch_dir / "primary_full_seed.jsonl"
    secondary_path = Path(args.secondary_export) if args.secondary_export else launch_dir / "secondary_calibration_seed.jsonl"
    primary_prefill = load_by_sample(launch_dir / "primary_full_prefill.jsonl")
    secondary_prefill = load_by_sample(launch_dir / "secondary_calibration_prefill.jsonl")
    primary = merge_annotations(primary_prefill, load_by_sample(primary_path))
    secondary = merge_annotations(secondary_prefill, load_by_sample(secondary_path))

    row_status = build_row_status(primary, secondary)
    summary = build_summary(row_status)
    blockers = [row for row in row_status if row["blocks_promotion"] == 1]
    decision = build_decision(summary, blockers, primary_path, secondary_path)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "manual_annotation_progress_rows.csv", row_status)
    write_csv(out_dir / "manual_annotation_progress_summary.csv", summary)
    write_csv(out_dir / "manual_annotation_progress_blockers.csv", blockers)
    write_csv(out_dir / "manual_annotation_progress_decision.csv", [decision])
    (out_dir / "manual_annotation_progress_report.md").write_text(
        build_markdown(summary, blockers, decision), encoding="utf-8"
    )
    print(f"Wrote manual annotation progress audit to {out_dir}")
    print(f"manual_annotation_progress_status={decision['progress_status']}")


def merge_annotations(prefill: dict[str, dict[str, Any]], annotations: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for sample_id, meta in prefill.items():
        row = dict(meta)
        ann = annotations.get(sample_id, {})
        row["annotation_status"] = str(ann.get("status", "unannotated"))
        row["annotation_boxes"] = ann.get("boxes", []) if isinstance(ann.get("boxes", []), list) else []
        row["annotation_notes"] = str(ann.get("notes", ""))
        row["annotation_present"] = int(sample_id in annotations)
        merged[sample_id] = row
    return merged


def build_row_status(primary: dict[str, dict[str, Any]], secondary: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    secondary_ids = set(secondary)
    rows: list[dict[str, Any]] = []
    for sample_id, row in sorted(primary.items()):
        boxes = row["annotation_boxes"]
        box_count = len(boxes)
        unlabeled = sum(1 for box in boxes if not str(box.get("label", "")).strip())
        invalid = sum(1 for box in boxes if not valid_box_shape(box))
        status = row["annotation_status"]
        needs_review = status in {"needs_review", "uncertain", "not_visible", "in_progress", "unannotated", ""}
        primary_ready = status == "annotated" and box_count > 0 and unlabeled == 0 and invalid == 0
        sec = secondary.get(sample_id)
        secondary_required = sample_id in secondary_ids
        secondary_ready = True
        secondary_status = ""
        secondary_box_count = ""
        if secondary_required:
            secondary_status = str(sec.get("annotation_status", "")) if sec else ""
            sec_boxes = sec.get("annotation_boxes", []) if sec else []
            secondary_box_count = len(sec_boxes)
            secondary_ready = (
                secondary_status == "annotated"
                and len(sec_boxes) > 0
                and all(valid_box_shape(box) and str(box.get("label", "")).strip() for box in sec_boxes)
            )
        reason = blocker_reason(primary_ready, needs_review, secondary_required, secondary_ready, box_count, unlabeled, invalid)
        rows.append(
            {
                "sample_id": sample_id,
                "task": row.get("task", ""),
                "annotation_batch": row.get("annotation_batch", ""),
                "is_calibration": int(secondary_required),
                "primary_status": status,
                "primary_box_count": box_count,
                "primary_unlabeled_box_count": unlabeled,
                "primary_invalid_box_count": invalid,
                "secondary_status": secondary_status,
                "secondary_box_count": secondary_box_count,
                "primary_ready": int(primary_ready),
                "secondary_ready": int(secondary_ready),
                "blocks_promotion": int(bool(reason)),
                "blocker_reason": reason,
                "prefill_complexity": row.get("prefill_complexity", ""),
                "prefill_evidence_units": row.get("prefill_evidence_units", ""),
                "image_path": row.get("image_path", ""),
            }
        )
    return rows


def blocker_reason(
    primary_ready: bool,
    needs_review: bool,
    secondary_required: bool,
    secondary_ready: bool,
    box_count: int,
    unlabeled: int,
    invalid: int,
) -> str:
    reasons = []
    if not primary_ready:
        if box_count == 0:
            reasons.append("primary_empty")
        if unlabeled:
            reasons.append("primary_unlabeled")
        if invalid:
            reasons.append("primary_invalid")
        if needs_review:
            reasons.append("primary_status")
    if secondary_required and not secondary_ready:
        reasons.append("secondary_calibration_not_ready")
    return ";".join(reasons)


def valid_box_shape(box: Any) -> bool:
    if not isinstance(box, dict):
        return False
    try:
        return float(box.get("w", 0)) > 0 and float(box.get("h", 0)) > 0
    except (TypeError, ValueError):
        return False


def build_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for scope_name, scope_rows in scoped_rows(rows):
        total = len(scope_rows)
        ready = sum(row["primary_ready"] for row in scope_rows)
        blockers = sum(row["blocks_promotion"] for row in scope_rows)
        calibration = [row for row in scope_rows if row["is_calibration"] == 1]
        out.append(
            {
                "scope": scope_name,
                "rows": total,
                "primary_ready_rows": ready,
                "primary_ready_rate": fmt_rate(ready, total),
                "calibration_rows": len(calibration),
                "calibration_ready_rows": sum(row["secondary_ready"] for row in calibration),
                "blocker_rows": blockers,
                "promotion_ready": int(total > 0 and blockers == 0),
            }
        )
    return out


def scoped_rows(rows: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    scopes: list[tuple[str, list[dict[str, Any]]]] = [("all", rows)]
    for task in sorted({row["task"] for row in rows}):
        scopes.append((f"task:{task}", [row for row in rows if row["task"] == task]))
    for batch in sorted({row["annotation_batch"] for row in rows}):
        scopes.append((f"batch:{batch}", [row for row in rows if row["annotation_batch"] == batch]))
    scopes.append(("calibration", [row for row in rows if row["is_calibration"] == 1]))
    return scopes


def build_decision(
    summary: list[dict[str, Any]],
    blockers: list[dict[str, Any]],
    primary_path: Path,
    secondary_path: Path,
) -> dict[str, Any]:
    all_row = next(row for row in summary if row["scope"] == "all")
    progress_status = "ready_for_final_promotion" if all_row["promotion_ready"] == 1 else "not_ready_for_final_promotion"
    return {
        "progress_status": progress_status,
        "primary_export": str(primary_path),
        "secondary_export": str(secondary_path),
        "primary_ready_rows": all_row["primary_ready_rows"],
        "total_primary_rows": all_row["rows"],
        "blocker_rows": len(blockers),
        "calibration_ready_rows": next(row for row in summary if row["scope"] == "calibration")["calibration_ready_rows"],
        "total_calibration_rows": next(row for row in summary if row["scope"] == "calibration")["rows"],
        "next_action": (
            "Promote and run build_open_ocr_qa_manual_final_package.py --build-derived --require-ready."
            if progress_status == "ready_for_final_promotion"
            else "Finish or adjudicate blocker rows before promotion."
        ),
    }


def build_markdown(
    summary: list[dict[str, Any]],
    blockers: list[dict[str, Any]],
    decision: dict[str, Any],
) -> str:
    lines = [
        "# Manual Annotation Progress Audit",
        "",
        "This dashboard tracks process readiness for final human multi-region evidence annotation. It is not a paper result and does not close the manual-evidence gap by itself.",
        "",
        "## Decision",
        "",
        f"- Status: `{decision['progress_status']}`",
        f"- Primary ready rows: {decision['primary_ready_rows']} / {decision['total_primary_rows']}",
        f"- Calibration ready rows: {decision['calibration_ready_rows']} / {decision['total_calibration_rows']}",
        f"- Blocker rows: {decision['blocker_rows']}",
        f"- Next action: {decision['next_action']}",
        "",
        "## Summary",
        "",
        table_md(summary, ["scope", "rows", "primary_ready_rows", "primary_ready_rate", "calibration_rows", "calibration_ready_rows", "blocker_rows", "promotion_ready"]),
        "",
        "## First Blockers",
        "",
        table_md(
            blockers[:50],
            [
                "sample_id",
                "task",
                "annotation_batch",
                "is_calibration",
                "primary_status",
                "primary_box_count",
                "secondary_status",
                "secondary_box_count",
                "blocker_reason",
                "prefill_evidence_units",
            ],
        ),
        "",
    ]
    return "\n".join(lines)


def load_by_sample(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_annotations(path) if path.exists() else []
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        sid = str(row.get("sample_id", ""))
        if sid:
            out[sid] = row
    return out


def load_annotations(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table_md(rows: list[dict[str, Any]], columns: list[str]) -> str:
    out = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(clean_cell(row.get(col, "")) for col in columns) + " |")
    return "\n".join(out)


def clean_cell(value: Any) -> str:
    return str(value).replace("\n", " ").replace("|", "/")


def fmt_rate(numerator: int, denominator: int) -> str:
    return "0.000" if denominator == 0 else f"{numerator / denominator:.3f}"


if __name__ == "__main__":
    main()
