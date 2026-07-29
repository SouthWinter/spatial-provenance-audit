#!/usr/bin/env python3
"""Audit final open-QA and controller claims across active paper artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = ROOT / "runs" / "open_ocr_qa_full" / "report" / "full_open_ocr_qa_runs.csv"
MATCHED_CSV = (
    ROOT
    / "runs"
    / "open_ocr_qa_full"
    / "report"
    / "full_open_ocr_qa_matched_comparisons.csv"
)
STRICT_JSON = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "strict_remaining_plan_go_no_go"
    / "strict_remaining_plan_report.json"
)
MAIN = ROOT / "paper_aaai2027" / "main.tex"
SUPPLEMENT = ROOT / "paper_aaai2027" / "SupplementaryMaterial.tex"
ARXIV = ROOT / "arxiv_submission" / "main.tex"
REQUIREMENT_AUDIT = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "problem_md_requirement_audit"
    / "problem_md_requirement_audit.csv"
)
BLOCKER_DASHBOARD = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "problem_md_remaining_blockers"
    / "problem_md_remaining_blockers.csv"
)
OUT_DIR = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "full_open_qa_postintegration_integrity"
)


def main() -> None:
    checks: list[dict[str, str]] = []
    rows = read_csv(SOURCE_CSV)
    matched_rows = read_csv(MATCHED_CSV)
    strict = read_json(STRICT_JSON)
    main_tex = MAIN.read_text(encoding="utf-8")
    supplement_tex = SUPPLEMENT.read_text(encoding="utf-8")
    arxiv_tex = ARXIV.read_text(encoding="utf-8")

    add(checks, "source_has_sixteen_runs", len(rows) == 16, f"rows={len(rows)}")
    rows70 = [row for row in rows if row["budget"] == "70%"]
    target_rows70 = [
        row for row in rows70 if row["method"] in {"Target", "Target-Grid"}
    ]
    baseline_rows70 = [row for row in rows70 if row["method"] in {"Random", "Grid"}]
    add(checks, "source_has_twelve_70pct_runs", len(rows70) == 12, f"rows={len(rows70)}")
    add(
        checks,
        "source_has_four_target_70pct_runs",
        len(target_rows70) == 4,
        f"rows={len(target_rows70)}",
    )
    add(
        checks,
        "source_has_eight_matched_baselines",
        len(baseline_rows70) == 8,
        f"rows={len(baseline_rows70)}",
    )
    add(
        checks,
        "source_has_eight_matched_comparisons",
        len(matched_rows) == 8,
        f"rows={len(matched_rows)}",
    )

    expected_keys = {
        ("Qwen3-VL-8B", "TextVQA"),
        ("Qwen3-VL-8B", "DocVQA"),
        ("LLaVA-1.5-7B", "TextVQA"),
        ("LLaVA-1.5-7B", "DocVQA"),
    }
    actual_keys = {(row["model"], row["task"]) for row in target_rows70}
    add(
        checks,
        "source_has_two_models_two_tasks",
        actual_keys == expected_keys,
        f"keys={sorted(actual_keys)}",
    )

    for row in target_rows70:
        key = f"{row['model']}_{row['task']}_70pct"
        fragments = table_fragments(row)
        add(
            checks,
            f"supplement_{key}",
            all(fragment in supplement_tex for fragment in fragments),
            "; ".join(fragments),
        )
        add(
            checks,
            f"arxiv_{key}",
            all(fragment in arxiv_tex for fragment in fragments),
            "; ".join(fragments),
        )

    run_by_key = {
        (row["model"], row["task"], row["method"]): row for row in rows70
    }
    comparison_by_key = {
        (row["model"], row["task"], row["comparison"]): row
        for row in matched_rows
    }
    for model, task in sorted(expected_keys):
        target_method = "Target-Grid" if model == "Qwen3-VL-8B" else "Target"
        target = run_by_key[(model, task, target_method)]
        random = run_by_key[(model, task, "Random")]
        grid = run_by_key[(model, task, "Grid")]
        tr = comparison_by_key[(model, task, f"{target_method} minus Random")]
        tg = comparison_by_key[(model, task, f"{target_method} minus Grid")]
        fragments = matched_table_fragments(model, task, target, random, grid, tr, tg)
        key = f"{model}_{task}_matched_controls"
        add(
            checks,
            f"supplement_{key}",
            all(fragment in supplement_tex for fragment in fragments),
            "; ".join(fragments),
        )
        add(
            checks,
            f"arxiv_{key}",
            all(fragment in arxiv_tex for fragment in fragments),
            "; ".join(fragments),
        )

    main_fragments = [
        "At 70\\% retention on full TextVQA/DocVQA validation",
        "Qwen Target+Grid scores 0.795/0.846 versus 0.828/0.940 for Full",
        "LLaVA Target scores 0.359/0.162 versus 0.485/0.216",
        "all paired intervals exclude zero",
        "trails Random in all four model--task pairs",
        "exceeds Grid only on Qwen DocVQA",
        "worst-region ECR from 0.216 to 0.613",
        "all-regions-covered fraction from 0.105 to 0.729",
    ]
    add(
        checks,
        "main_open_qa_boundary_paragraph",
        all(fragment in main_tex for fragment in main_fragments),
        "; ".join(main_fragments),
    )
    add(
        checks,
        "arxiv_open_qa_boundary_paragraph",
        all(fragment in arxiv_tex for fragment in main_fragments),
        "; ".join(main_fragments),
    )

    contract = strict.get("controller_contract", {})
    add(
        checks,
        "strict_controller_stop_status",
        contract.get("status") == "no_go_stop_controller_expansion",
        f"status={contract.get('status')}",
    )
    add(
        checks,
        "strict_controller_zero_passing_candidates",
        contract.get("passing_candidates") == [],
        f"passing_candidates={contract.get('passing_candidates')}",
    )
    add(
        checks,
        "strict_controller_contract_values",
        contract.get("quality_tolerance") == 0.01
        and contract.get("max_mean_keep") == 0.6
        and contract.get("fixed70_cost") == 0.7,
        (
            f"quality_tolerance={contract.get('quality_tolerance')}; "
            f"max_mean_keep={contract.get('max_mean_keep')}; "
            f"fixed70_cost={contract.get('fixed70_cost')}"
        ),
    )
    controller_fragments = [
        "both task scores within 0.01 of fixed 70\\%",
        "mean keep at most 0.60",
        "No candidate satisfies all three constraints",
    ]
    add(
        checks,
        "supplement_controller_contract",
        all(fragment in supplement_tex for fragment in controller_fragments),
        "; ".join(controller_fragments),
    )
    add(
        checks,
        "arxiv_controller_contract",
        all(fragment in arxiv_tex for fragment in controller_fragments),
        "; ".join(controller_fragments),
    )

    requirement_rows = read_csv(REQUIREMENT_AUDIT)
    r2 = next((row for row in requirement_rows if row.get("id") == "R2"), {})
    r4 = next((row for row in requirement_rows if row.get("id") == "R4"), {})
    add(
        checks,
        "current_requirement_r2",
        r2.get("status") == "addressed_as_full_validation_boundary",
        f"status={r2.get('status')}",
    )
    add(
        checks,
        "current_requirement_r4",
        r4.get("status") == "not_solved_boundary_quantified"
        and "controller expansion is stopped" in r4.get("key_reading", ""),
        f"status={r4.get('status')}",
    )

    blocker_rows = read_csv(BLOCKER_DASHBOARD)
    open_gate = next(
        (row for row in blocker_rows if row.get("gate") == "native open-answer OCR/document QA"),
        {},
    )
    controller_gate = next(
        (row for row in blocker_rows if row.get("gate") == "unified adaptive controller"),
        {},
    )
    add(
        checks,
        "current_dashboard_open_qa",
        open_gate.get("paper_status") == "full-validation cross-model boundary evidence",
        f"paper_status={open_gate.get('paper_status')}",
    )
    add(
        checks,
        "current_dashboard_controller",
        "no_go_stop_controller_expansion" in controller_gate.get("key_numbers", ""),
        controller_gate.get("key_numbers", ""),
    )

    errors = [check for check in checks if check["status"] == "fail"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "full_open_qa_postintegration_checks.csv", checks)
    (OUT_DIR / "full_open_qa_postintegration_integrity.md").write_text(
        build_markdown(checks), encoding="utf-8"
    )
    print(f"Wrote post-integration audit to {OUT_DIR}")
    print(f"checks={len(checks)} errors={len(errors)}")
    if errors:
        raise SystemExit(1)


def table_fragments(row: dict[str, str]) -> list[str]:
    return [
        row["model"],
        f"& {int(row['n'])} &",
        f"& {float(row['mean_keep']):.3f} &",
        f"& {float(row['full_score']):.4f} & {float(row['pruned_score']):.4f} &",
        f"{float(row['paired_delta']):+.4f}",
        f"{row['wins']}/{row['losses']}/{row['ties']}",
    ]


def matched_table_fragments(
    model: str,
    task: str,
    target: dict[str, str],
    random: dict[str, str],
    grid: dict[str, str],
    target_random: dict[str, str],
    target_grid: dict[str, str],
) -> list[str]:
    return [
        (
            f"{model} & {task} & {float(target['pruned_score']):.4f} & "
            f"{float(random['pruned_score']):.4f} & {float(grid['pruned_score']):.4f}"
        ),
        (
            f"{float(target_random['paired_delta']):+.4f} & "
            f"$[{float(target_random['ci_low']):.4f},{float(target_random['ci_high']):.4f}]$"
        ),
        (
            f"{float(target_grid['paired_delta']):+.4f} / "
            f"$[{float(target_grid['ci_low']):.4f},{float(target_grid['ci_high']):.4f}]$"
        ),
    ]


def add(checks: list[dict[str, str]], name: str, passed: bool, evidence: str) -> None:
    checks.append({"check": name, "status": "pass" if passed else "fail", "evidence": evidence})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check", "status", "evidence"])
        writer.writeheader()
        writer.writerows(rows)


def build_markdown(checks: list[dict[str, str]]) -> str:
    passed = sum(check["status"] == "pass" for check in checks)
    lines = [
        "# Full Open-QA Post-Integration Integrity Audit",
        "",
        "This audit treats the full-validation result CSV and strict controller report as source artifacts, then checks the active AAAI, supplementary, arXiv, and current problem.md status files. Historical review logs are intentionally excluded.",
        "",
        f"- Result: `{'pass' if passed == len(checks) else 'fail'}`",
        f"- Passed checks: {passed}/{len(checks)}",
        "",
        "| Check | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    for check in checks:
        evidence = check["evidence"].replace("|", "/").replace("\n", " ")
        lines.append(f"| {check['check']} | {check['status']} | {evidence} |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
