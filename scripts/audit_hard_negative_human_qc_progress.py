#!/usr/bin/env python3
"""Audit completion of the hard-negative human QC package."""

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
    / "hard_negative_human_qc_launch"
    / "hard_negative_human_qc_template.csv"
)
DEFAULT_OUT = ROOT / "runs" / "problem_optimization_audit" / "hard_negative_human_qc_progress"

REQUIRED_FIELDS = [
    "human_source_text_visible",
    "human_target_text_visible_same_image",
    "target_absent_after_case_punct_normalization",
    "source_bbox_matches_source_text",
    "qc_decision",
]
YES_NO_UNCLEAR = {"yes", "no", "unclear"}
VALID_DECISIONS = {
    "valid_negative",
    "invalid_target_present",
    "invalid_source_not_visible",
    "invalid_bad_box",
    "unclear",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qc-export", default=str(DEFAULT_EXPORT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--expected-rows",
        type=int,
        default=0,
        help="Require this exact row count; 0 disables the count check.",
    )
    args = parser.parse_args()

    rows = read_csv(Path(args.qc_export))
    progress_rows = [audit_row(row) for row in rows]
    blockers = [row for row in progress_rows if row["ready_for_paper"] == "0"]
    summary = build_summary(progress_rows)
    decision = build_decision(
        progress_rows,
        blockers,
        Path(args.qc_export),
        args.expected_rows,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "hard_negative_human_qc_progress_rows.csv", progress_rows)
    write_csv(out_dir / "hard_negative_human_qc_progress_summary.csv", summary)
    write_csv(out_dir / "hard_negative_human_qc_progress_blockers.csv", blockers)
    write_csv(out_dir / "hard_negative_human_qc_progress_decision.csv", [decision])
    (out_dir / "hard_negative_human_qc_progress_report.md").write_text(
        build_report(summary, decision, blockers),
        encoding="utf-8",
    )
    print(f"Wrote hard-negative human QC progress audit to {out_dir}")
    print(f"hard_negative_human_qc_status={decision['hard_negative_human_qc_status']}")


def audit_row(row: dict[str, str]) -> dict[str, str]:
    reasons = []
    for field in REQUIRED_FIELDS:
        if not row.get(field, "").strip():
            reasons.append(f"missing_{field}")
    for field in REQUIRED_FIELDS[:4]:
        value = row.get(field, "").strip().lower()
        if value and value not in YES_NO_UNCLEAR:
            reasons.append(f"invalid_{field}")
    decision = row.get("qc_decision", "").strip().lower()
    if decision and decision not in VALID_DECISIONS:
        reasons.append("invalid_qc_decision")
    return {
        "sample_id": row.get("sample_id", ""),
        "image_id": row.get("image_id", ""),
        "selection_priority": row.get("selection_priority", ""),
        "qc_decision": decision,
        "ready_for_paper": "0" if reasons else "1",
        "blocker_reasons": ";".join(reasons),
    }


def build_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    total = len(rows)
    ready = sum(row["ready_for_paper"] == "1" for row in rows)
    invalid = sum(row["qc_decision"].startswith("invalid") for row in rows)
    unclear = sum(row["qc_decision"] == "unclear" for row in rows)
    valid = sum(row["qc_decision"] == "valid_negative" for row in rows)
    priorities = sorted({row["selection_priority"] for row in rows})
    out = [
        {"scope": "all", "metric": "rows", "value": total},
        {"scope": "all", "metric": "ready_rows", "value": ready},
        {"scope": "all", "metric": "ready_rate", "value": fmt(ready / max(1, total))},
        {"scope": "all", "metric": "valid_negative_rows", "value": valid},
        {"scope": "all", "metric": "invalid_rows", "value": invalid},
        {"scope": "all", "metric": "unclear_rows", "value": unclear},
    ]
    for priority in priorities:
        group = [row for row in rows if row["selection_priority"] == priority]
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
    expected_rows: int = 0,
) -> dict[str, str]:
    total = len(rows)
    row_count_ready = not expected_rows or total == expected_rows
    ready = total > 0 and not blockers and row_count_ready
    valid = sum(row["qc_decision"] == "valid_negative" for row in rows)
    invalid = sum(row["qc_decision"].startswith("invalid") for row in rows)
    unclear = sum(row["qc_decision"] == "unclear" for row in rows)
    return {
        "hard_negative_human_qc_status": "ready_for_human_validated_claim" if ready else "not_ready_for_human_validated_claim",
        "qc_export": str(export_path),
        "rows": str(total),
        "ready_rows": str(total - len(blockers)),
        "blocker_rows": str(len(blockers)),
        "expected_rows": str(expected_rows),
        "row_count_ready": str(int(row_count_ready)),
        "valid_negative_rows": str(valid),
        "invalid_rows": str(invalid),
        "unclear_rows": str(unclear),
        "safe_claim": (
            "The sampled hard negatives have completed human QC."
            if ready
            else "Hard-negative human QC is launch-ready but not complete; keep claims limited to automatic lexical validation."
        ),
    }


def build_report(
    summary: list[dict[str, Any]],
    decision: dict[str, str],
    blockers: list[dict[str, str]],
) -> str:
    lines = [
        "# Hard-Negative Human QC Progress",
        "",
        f"- status: `{decision['hard_negative_human_qc_status']}`",
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
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
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
