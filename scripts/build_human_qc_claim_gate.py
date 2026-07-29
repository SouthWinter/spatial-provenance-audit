#!/usr/bin/env python3
"""Aggregate human-QC readiness gates for paper-facing claims.

The individual progress audits track text-replacement edit validity and
hard-negative label validity separately. This script combines them into one
paper-facing claim gate so unfinished QC cannot be accidentally promoted to
human-verified or human-validated evidence.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "problem_optimization_audit" / "human_qc_claim_gate"
PAPER_EVIDENCE = ROOT / "runs" / "paper_evidence"

TEXT_QC_DECISION = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "text_replacement_human_qc_progress"
    / "text_replacement_human_qc_progress_decision.csv"
)
HARD_QC_DECISION = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "hard_negative_human_qc_progress"
    / "hard_negative_human_qc_progress_decision.csv"
)

STALE_PATTERNS = (
    "table_text_replacement_human_verified*",
    "table_text_replacement_human_validated*",
    "table_hard_negative_human_verified*",
    "table_hard_negative_human_validated*",
)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    text_rows = read_csv(TEXT_QC_DECISION) if TEXT_QC_DECISION.exists() else []
    hard_rows = read_csv(HARD_QC_DECISION) if HARD_QC_DECISION.exists() else []
    text = text_rows[0] if text_rows else {}
    hard = hard_rows[0] if hard_rows else {}

    stale_tables = sorted(
        {
            path
            for pattern in STALE_PATTERNS
            for path in PAPER_EVIDENCE.glob(pattern)
            if path.is_file() and path.stat().st_size > 0
        }
    )

    gate_rows = [
        gate_row(
            "text_replacement_qc_complete",
            text.get("text_replacement_human_qc_status") == "ready_for_verified_semantic_counterfactual_claim",
            (
                f"status={text.get('text_replacement_human_qc_status', 'missing')}; "
                f"ready={text.get('ready_rows', '0')}/{text.get('rows', '0')}; "
                f"valid_rate={text.get('valid_semantic_edit_rate', '0.000')}"
            ),
        ),
        gate_row(
            "hard_negative_qc_complete",
            hard.get("hard_negative_human_qc_status") == "ready_for_human_validated_claim",
            (
                f"status={hard.get('hard_negative_human_qc_status', 'missing')}; "
                f"ready={hard.get('ready_rows', '0')}/{hard.get('rows', '0')}; "
                f"valid={hard.get('valid_negative_rows', '0')}; "
                f"invalid={hard.get('invalid_rows', '0')}"
            ),
        ),
        gate_row(
            "stale_human_qc_paper_tables_absent",
            not stale_tables,
            (
                "no stale human-QC paper-evidence tables"
                if not stale_tables
                else "; ".join(str(path.relative_to(ROOT)) for path in stale_tables)
            ),
        ),
    ]
    all_ready = all(row["status"] == "pass" for row in gate_rows)
    decision = {
        "human_qc_claim_status": "ready_for_human_qc_claims" if all_ready else "not_ready_for_human_qc_claims",
        "text_replacement_ready": "1" if gate_rows[0]["status"] == "pass" else "0",
        "hard_negative_ready": "1" if gate_rows[1]["status"] == "pass" else "0",
        "stale_tables_absent": "1" if gate_rows[2]["status"] == "pass" else "0",
        "safe_claim": (
            "Text-replacement and hard-negative human QC are complete."
            if all_ready
            else (
                "Human-QC packages are launch-ready/progress-tracked but not complete; "
                "do not claim human-verified semantic edits or human-validated hard negatives."
            )
        ),
    }

    write_csv(OUT_DIR / "human_qc_claim_gate_rows.csv", gate_rows)
    write_csv(OUT_DIR / "human_qc_claim_gate_decision.csv", [decision])
    (OUT_DIR / "human_qc_claim_gate_report.md").write_text(
        build_report(gate_rows, decision), encoding="utf-8"
    )
    print(f"human_qc_claim_status={decision['human_qc_claim_status']}")


def gate_row(gate: str, passed: bool, evidence: str) -> dict[str, str]:
    return {
        "gate": gate,
        "status": "pass" if passed else "fail",
        "evidence": evidence,
    }


def build_report(rows: list[dict[str, str]], decision: dict[str, str]) -> str:
    lines = [
        "# Human-QC Claim Gate",
        "",
        f"Claim status: **{decision['human_qc_claim_status']}**.",
        "",
        "| Gate | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    for row in rows:
        lines.append(f"| {row['gate']} | {row['status']} | {row['evidence']} |")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            decision["safe_claim"],
        ]
    )
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


if __name__ == "__main__":
    main()
