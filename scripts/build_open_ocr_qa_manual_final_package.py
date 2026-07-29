#!/usr/bin/env python3
"""Package completed manual multi-region evidence for paper use.

This script is deliberately conservative. It never fabricates manual evidence
tables. If the final human annotations are missing or any readiness gate fails,
it writes a blocked status report and does not copy result tables into
`runs/paper_evidence`.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "runs" / "problem_optimization_audit" / "open_ocr_qa_manual_final_package"
PAPER_EVIDENCE = ROOT / "runs" / "paper_evidence"

READINESS_GATES = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "manual_evidence_readiness_gate"
    / "manual_evidence_readiness_gates.csv"
)
FINAL_SUMMARY = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_final_annotations"
    / "final_annotation_summary.csv"
)
FINAL_ECR_SUMMARY = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_final_annotations_ecr"
    / "bbox_ecr_summary.csv"
)
MANUAL_AUDIT_DIR = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_manual_evidence_audit"
)
MANUAL_TABLES = {
    "manual_evidence_key_summary.csv": "table_manual_multi_region_evidence_key_summary.csv",
    "manual_evidence_bucket_summary.csv": "table_manual_multi_region_evidence_bucket_summary.csv",
    "manual_evidence_correlation_summary.csv": "table_manual_multi_region_evidence_correlation_summary.csv",
    "manual_evidence_counterexamples.csv": "table_manual_multi_region_evidence_counterexamples.csv",
}
EXTRA_TABLES = {
    FINAL_SUMMARY: "table_manual_multi_region_annotation_summary.csv",
    FINAL_ECR_SUMMARY: "table_manual_multi_region_bbox_ecr_summary.csv",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Exit non-zero if the manual evidence package is not ready.",
    )
    parser.add_argument(
        "--build-derived",
        action="store_true",
        help=(
            "Before checking readiness, derive validation, ECR, manual quality, "
            "and readiness-gate tables from final_annotations.jsonl. This does "
            "not create human annotations; it only processes a completed final export."
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    derived_rows: list[dict[str, Any]] = []
    if args.build_derived:
        derived_rows = build_derived_tables()
    status_rows, copy_plan = build_status()
    ready = all(row["status"] == "pass" for row in status_rows)
    copied_rows: list[dict[str, Any]] = []

    if ready:
        PAPER_EVIDENCE.mkdir(parents=True, exist_ok=True)
        for source, dest_name in copy_plan:
            dest = PAPER_EVIDENCE / dest_name
            shutil.copy2(source, dest)
            copied_rows.append(
                {
                    "source": str(source.relative_to(ROOT)),
                    "destination": str(dest.relative_to(ROOT)),
                    "copied": 1,
                }
            )

    write_csv(out_dir / "manual_final_package_status.csv", status_rows)
    write_csv(out_dir / "manual_final_package_derived_steps.csv", derived_rows)
    write_csv(out_dir / "manual_final_package_copied_tables.csv", copied_rows)
    (out_dir / "manual_final_package_report.md").write_text(
        build_report(status_rows, copied_rows, copy_plan, ready, derived_rows), encoding="utf-8"
    )

    if args.require_ready and not ready:
        raise SystemExit("Manual final package is not ready; see manual_final_package_report.md")
    print(f"Manual final package status: {'ready' if ready else 'blocked'}")


def build_derived_tables() -> list[dict[str, Any]]:
    """Run deterministic downstream processing for completed manual annotations."""
    final_annotations = (
        ROOT
        / "runs"
        / "problem_optimization_audit"
        / "open_ocr_qa_final_annotations"
        / "final_annotations.jsonl"
    )
    if not final_annotations.exists():
        return [
            {
                "step": "missing_final_annotations",
                "status": "fail",
                "returncode": 1,
                "command": str(final_annotations),
                "stdout_tail": "",
                "stderr_tail": "final_annotations.jsonl is absent; run promotion after completing human annotation",
            }
        ]
    steps = [
        (
            "validate_final_annotations",
            [
                sys.executable,
                str(ROOT / "scripts" / "validate_open_ocr_qa_bbox_annotations.py"),
                "--annotations",
                str(final_annotations),
                "--output-dir",
                str(
                    ROOT
                    / "runs"
                    / "problem_optimization_audit"
                    / "open_ocr_qa_final_annotations_validation"
                ),
            ],
        ),
        (
            "evaluate_final_annotation_ecr",
            [
                sys.executable,
                str(ROOT / "scripts" / "evaluate_open_ocr_qa_bbox_ecr.py"),
                "--annotations",
                str(
                    ROOT
                    / "runs"
                    / "problem_optimization_audit"
                    / "open_ocr_qa_final_annotations"
                    / "final_annotations.jsonl"
                ),
                "--output-dir",
                str(
                    ROOT
                    / "runs"
                    / "problem_optimization_audit"
                    / "open_ocr_qa_final_annotations_ecr"
                ),
            ],
        ),
        (
            "build_manual_evidence_audit",
            [
                sys.executable,
                str(ROOT / "scripts" / "build_open_ocr_qa_manual_evidence_audit.py"),
            ],
        ),
        (
            "refresh_manual_readiness_gate",
            [
                sys.executable,
                str(ROOT / "scripts" / "audit_problem_md_manual_evidence_readiness.py"),
            ],
        ),
    ]
    rows: list[dict[str, Any]] = []
    for name, command in steps:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        rows.append(
            {
                "step": name,
                "status": "pass" if result.returncode == 0 else "fail",
                "returncode": result.returncode,
                "command": " ".join(command),
                "stdout_tail": tail(result.stdout),
                "stderr_tail": tail(result.stderr),
            }
        )
        if result.returncode != 0:
            break
    return rows


def build_status() -> tuple[list[dict[str, Any]], list[tuple[Path, str]]]:
    gates = read_csv(READINESS_GATES) if READINESS_GATES.exists() else []
    failed_gates = [row for row in gates if row.get("status") != "pass"]
    copy_plan: list[tuple[Path, str]] = []
    missing_sources: list[str] = []

    for src_name, dest_name in MANUAL_TABLES.items():
        source = MANUAL_AUDIT_DIR / src_name
        if source.exists() and source.stat().st_size > 0:
            copy_plan.append((source, dest_name))
        else:
            missing_sources.append(str(source.relative_to(ROOT)))
    for source, dest_name in EXTRA_TABLES.items():
        if source.exists() and source.stat().st_size > 0:
            copy_plan.append((source, dest_name))
        else:
            missing_sources.append(str(source.relative_to(ROOT)))

    expected_destinations = {
        *(PAPER_EVIDENCE / dest_name for dest_name in MANUAL_TABLES.values()),
        *(PAPER_EVIDENCE / dest_name for dest_name in EXTRA_TABLES.values()),
    }
    stale_destinations = [
        str(path.relative_to(ROOT))
        for path in sorted(expected_destinations)
        if path.exists() and path.stat().st_size > 0
    ]
    ready_without_stale_check = bool(gates) and not failed_gates and not missing_sources

    rows = [
        status_row(
            "readiness_gates_exist",
            bool(gates),
            f"{len(gates)} gate row(s)",
        ),
        status_row(
            "readiness_gates_all_pass",
            bool(gates) and not failed_gates,
            f"failed_gates={len(failed_gates)}",
        ),
        status_row(
            "manual_audit_tables_exist",
            not missing_sources,
            "; ".join(missing_sources) if missing_sources else "all expected sources present",
        ),
        status_row(
            "stale_paper_evidence_tables_absent",
            ready_without_stale_check or not stale_destinations,
            (
                "ready package may overwrite existing paper-evidence tables"
                if ready_without_stale_check
                else (
                    "; ".join(stale_destinations)
                    if stale_destinations
                    else "no stale manual multi-region paper-evidence tables"
                )
            ),
        ),
        status_row(
            "paper_evidence_copy_allowed",
            ready_without_stale_check,
            (
                "copies are allowed only after final human annotations, validation, ECR, and audit are complete; "
                "if blocked, no stale manual multi-region paper-evidence tables should remain"
            ),
        ),
    ]
    return rows, copy_plan


def status_row(name: str, passed: bool, evidence: str) -> dict[str, Any]:
    return {
        "gate": name,
        "status": "pass" if passed else "fail",
        "evidence": evidence,
    }


def build_report(
    status_rows: list[dict[str, Any]],
    copied_rows: list[dict[str, Any]],
    copy_plan: list[tuple[Path, str]],
    ready: bool,
    derived_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Manual Multi-Region Final Package",
        "",
        "This report controls whether completed human multi-region evidence can be copied into `runs/paper_evidence`.",
        "",
        f"Package status: **{'ready' if ready else 'blocked'}**.",
        "",
        "| Gate | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    for row in status_rows:
        lines.append(f"| {row['gate']} | {row['status']} | {row['evidence']} |")
    if derived_rows:
        lines.extend(
            [
                "",
                "## Derived Processing Steps",
                "",
                "| Step | Status | Return code | Stdout tail | Stderr tail |",
                "| --- | --- | ---: | --- | --- |",
            ]
        )
        for row in derived_rows:
            lines.append(
                f"| {row['step']} | {row['status']} | {row['returncode']} | "
                f"{escape_cell(row['stdout_tail'])} | {escape_cell(row['stderr_tail'])} |"
            )
    lines.extend(["", "## Copy Plan", "", "| Source | Paper evidence table | Copied |", "| --- | --- | ---: |"])
    copied_dest = {row["destination"] for row in copied_rows}
    for source, dest_name in copy_plan:
        dest = str((PAPER_EVIDENCE / dest_name).relative_to(ROOT))
        lines.append(
            f"| {source.relative_to(ROOT)} | {dest} | {1 if dest in copied_dest else 0} |"
        )
    if not copy_plan:
        lines.append("| none | none | 0 |")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "A blocked status means the manuscript must not claim completed manual multi-region evidence. The package becomes ready only after final human annotations are promoted, validated, projected onto pruning masks, joined with open-QA generation, and all readiness gates pass.",
        ]
    )
    return "\n".join(lines) + "\n"


def tail(text: str, max_chars: int = 500) -> str:
    normalized = " ".join(text.strip().split())
    if len(normalized) <= max_chars:
        return normalized
    return "..." + normalized[-max_chars:]


def escape_cell(value: Any) -> str:
    return str(value).replace("|", "\\|")


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
