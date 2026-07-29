#!/usr/bin/env python3
"""Audit completion of text-replacement human QC."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "text_replacement_control_pack_v3"
    / "human_qc_launch"
    / "text_replacement_human_qc_template.csv"
)
DEFAULT_OUT = ROOT / "runs" / "problem_optimization_audit" / "text_replacement_human_qc_progress"

REQUIRED_FIELDS = [
    "human_source_visible_original",
    "human_replacement_readable_edited",
    "human_source_absent_edited",
    "human_local_edit_plausible",
    "human_no_unrelated_text_changed",
    "qc_decision",
]
CONTROL_REQUIRED_FIELDS = [
    "human_source_readable_sham",
    "human_source_absent_erase",
    "human_replacement_absent_erase",
]
YES_NO_UNCLEAR = {"yes", "no", "unclear"}
VALID_DECISIONS = {
    "valid_semantic_edit",
    "invalid_unreadable_replacement",
    "invalid_source_not_visible",
    "invalid_source_still_visible",
    "invalid_bad_local_edit",
    "invalid_unrelated_change",
    "invalid_sham_control",
    "invalid_erase_control",
    "unclear",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qc-export", default=str(DEFAULT_EXPORT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--min-valid-edits", type=int, default=0)
    args = parser.parse_args()

    rows = read_csv(Path(args.qc_export))
    progress_rows = [audit_row(row) for row in rows]
    blockers = [row for row in progress_rows if row["ready_for_paper"] == "0"]
    summary = build_summary(progress_rows)
    decision = build_decision(progress_rows, blockers, Path(args.qc_export), args.min_valid_edits)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "text_replacement_human_qc_progress_rows.csv", progress_rows)
    write_csv(out_dir / "text_replacement_human_qc_progress_summary.csv", summary)
    write_csv(out_dir / "text_replacement_human_qc_progress_blockers.csv", blockers)
    write_csv(out_dir / "text_replacement_human_qc_progress_decision.csv", [decision])
    (out_dir / "text_replacement_human_qc_progress_report.md").write_text(
        build_report(summary, decision, blockers),
        encoding="utf-8",
    )
    print(f"Wrote text-replacement human QC progress audit to {out_dir}")
    print(f"text_replacement_human_qc_status={decision['text_replacement_human_qc_status']}")


def audit_row(row: dict[str, str]) -> dict[str, str]:
    reasons = []
    control_pack = bool(row.get("sham_image", "").strip() or row.get("erase_image", "").strip())
    required_fields = REQUIRED_FIELDS + (CONTROL_REQUIRED_FIELDS if control_pack else [])
    for field in required_fields:
        if not row.get(field, "").strip():
            reasons.append(f"missing_{field}")
    yes_no_fields = REQUIRED_FIELDS[:5] + (CONTROL_REQUIRED_FIELDS if control_pack else [])
    for field in yes_no_fields:
        value = row.get(field, "").strip().lower()
        if value and value not in YES_NO_UNCLEAR:
            reasons.append(f"invalid_{field}")
    decision = row.get("qc_decision", "").strip().lower()
    if decision and decision not in VALID_DECISIONS:
        reasons.append("invalid_qc_decision")
    if decision == "valid_semantic_edit" and any(
        row.get(field, "").strip().lower() != "yes" for field in yes_no_fields
    ):
        reasons.append("valid_decision_conflicts_with_checks")
    return {
        "image_id": row.get("image_id", ""),
        "source_text": row.get("source_text", ""),
        "replacement_text": row.get("replacement_text", ""),
        "auto_priority": row.get("auto_priority", ""),
        "qc_decision": decision,
        "ready_for_paper": "0" if reasons else "1",
        "blocker_reasons": ";".join(reasons),
    }


def build_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    total = len(rows)
    ready = sum(row["ready_for_paper"] == "1" for row in rows)
    valid = sum(row["qc_decision"] == "valid_semantic_edit" for row in rows)
    invalid = sum(row["qc_decision"].startswith("invalid") for row in rows)
    unclear = sum(row["qc_decision"] == "unclear" for row in rows)
    out = [
        {"scope": "all", "metric": "rows", "value": total},
        {"scope": "all", "metric": "ready_rows", "value": ready},
        {"scope": "all", "metric": "ready_rate", "value": fmt(ready / max(1, total))},
        {"scope": "all", "metric": "valid_semantic_edit_rows", "value": valid},
        {"scope": "all", "metric": "invalid_rows", "value": invalid},
        {"scope": "all", "metric": "unclear_rows", "value": unclear},
    ]
    for priority in sorted({row["auto_priority"] for row in rows}):
        group = [row for row in rows if row["auto_priority"] == priority]
        out.append({"scope": f"priority:{priority}", "metric": "rows", "value": len(group)})
        out.append(
            {
                "scope": f"priority:{priority}",
                "metric": "ready_rows",
                "value": sum(row["ready_for_paper"] == "1" for row in group),
            }
        )
    return out


def build_decision(
    rows: list[dict[str, str]],
    blockers: list[dict[str, str]],
    export_path: Path,
    min_valid_edits: int,
) -> dict[str, str]:
    total = len(rows)
    valid = sum(row["qc_decision"] == "valid_semantic_edit" for row in rows)
    ready = total > 0 and not blockers and valid >= min_valid_edits
    return {
        "text_replacement_human_qc_status": "ready_for_verified_semantic_counterfactual_claim" if ready else "not_ready_for_verified_semantic_counterfactual_claim",
        "qc_export": str(export_path),
        "rows": str(total),
        "ready_rows": str(total - len(blockers)),
        "blocker_rows": str(len(blockers)),
        "valid_semantic_edit_rows": str(valid),
        "valid_semantic_edit_rate": fmt(valid / max(1, total)),
        "minimum_valid_edits": str(min_valid_edits),
        "safe_claim": (
            "The text-replacement edits have completed human QC."
            if ready
            else "Text-replacement human QC is launch-ready but not complete; keep semantic counterfactual claims as an automatic/OCR-assisted boundary audit."
        ),
    }


def build_report(
    summary: list[dict[str, Any]],
    decision: dict[str, str],
    blockers: list[dict[str, str]],
) -> str:
    lines = [
        "# Text-Replacement Human QC Progress",
        "",
        f"- status: `{decision['text_replacement_human_qc_status']}`",
        f"- ready rows: {decision['ready_rows']} / {decision['rows']}",
        f"- blocker rows: {decision['blocker_rows']}",
        f"- safe claim: {decision['safe_claim']}",
        "",
        "## Summary",
        "",
        table_md(summary),
        "",
        "## Blockers",
        "",
        table_md(blockers[:100]) if blockers else "_No blocker rows._",
    ]
    return "\n".join(lines) + "\n"


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


def table_md(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    cols = list(rows[0])
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(col, "")).replace("|", "/") for col in cols) + " |")
    return "\n".join(out)


def fmt(value: float) -> str:
    return f"{value:.3f}"


if __name__ == "__main__":
    main()
