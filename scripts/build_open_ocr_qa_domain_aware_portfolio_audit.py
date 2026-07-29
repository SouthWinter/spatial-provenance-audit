#!/usr/bin/env python3
"""Audit a conservative domain-aware adaptive portfolio for open OCR QA.

problem.md asks for a unified adaptive controller. Existing audits show that
current pre-generation signals do not pass the near-fixed70/lower-cost gate on
both TextVQA-lite and DocVQA-lite. This script tests a weaker but deployable
portfolio rule:

  use a task's best pre-generation candidate only if it already passes the
  near-fixed70/lower-cost gate; otherwise fall back to fixed70.

The result is useful only as a boundary/portfolio diagnostic. It must not be
promoted as a solved unified adaptive controller because any aggregate saving
can come entirely from the easier task.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "problem_optimization_audit" / "open_ocr_qa_domain_aware_portfolio"
DEPLOYMENT_ROWS = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_deployment_contract"
    / "deployment_contract_rows.csv"
)

NEAR_QUALITY_TOL = 0.01


def main() -> None:
    rows = read_csv(DEPLOYMENT_ROWS)
    task_rows = build_task_rows(rows)
    decision = build_decision(task_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "domain_aware_portfolio_rows.csv", task_rows)
    write_csv(OUT_DIR / "domain_aware_portfolio_decision.csv", [decision])
    (OUT_DIR / "domain_aware_portfolio_report.md").write_text(
        build_report(task_rows, decision),
        encoding="utf-8",
    )
    print(f"Wrote domain-aware portfolio audit to {OUT_DIR}")
    print(f"portfolio_status={decision['portfolio_status']}")


def build_task_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    tasks = sorted({row.get("task", "") for row in rows if row.get("task")})
    for task in tasks:
        task_rows = [row for row in rows if row.get("task") == task]
        fixed70 = find_policy(task_rows, "fixed_0.70")
        pregen = [
            row
            for row in task_rows
            if row.get("deployable_before_llm_prefill") == "1"
            and row.get("uses_oracle_scores") == "0"
            and row.get("family") not in {"fixed", "oracle"}
        ]
        passing = [row for row in pregen if row.get("passes_near_fixed70_lower_cost") == "1"]
        selected = max(passing, key=lambda row: fnum(row.get("test_score")), default=fixed70)
        selected_reason = (
            "best_passing_pregen_candidate"
            if selected is not fixed70
            else "fallback_to_fixed70_no_pregen_candidate_passed"
        )
        out.append(
            {
                "task": task,
                "selected_policy": selected.get("policy", ""),
                "selected_family": selected.get("family", ""),
                "selected_reason": selected_reason,
                "test_score": fmt(selected.get("test_score")),
                "deployment_cost_proxy": fmt(selected.get("deployment_cost_proxy")),
                "fixed70_score": fmt(fixed70.get("test_score")),
                "fixed70_cost": fmt(fixed70.get("deployment_cost_proxy")),
                "delta_score_vs_fixed70": fmt(fnum(selected.get("test_score")) - fnum(fixed70.get("test_score"))),
                "delta_cost_vs_fixed70": fmt(
                    fnum(selected.get("deployment_cost_proxy")) - fnum(fixed70.get("deployment_cost_proxy"))
                ),
                "near_fixed70": int(fnum(selected.get("test_score")) >= fnum(fixed70.get("test_score")) - NEAR_QUALITY_TOL),
                "lower_cost_than_fixed70": int(
                    fnum(selected.get("deployment_cost_proxy")) < fnum(fixed70.get("deployment_cost_proxy"))
                ),
                "pregen_candidates": len(pregen),
                "passing_pregen_candidates": len(passing),
            }
        )
    return out


def build_decision(rows: list[dict[str, Any]]) -> dict[str, str]:
    n = len(rows)
    selected_score = mean(fnum(row["test_score"]) for row in rows)
    fixed_score = mean(fnum(row["fixed70_score"]) for row in rows)
    selected_cost = mean(fnum(row["deployment_cost_proxy"]) for row in rows)
    fixed_cost = mean(fnum(row["fixed70_cost"]) for row in rows)
    all_near = all(str(row["near_fixed70"]) == "1" for row in rows)
    all_lower = all(str(row["lower_cost_than_fixed70"]) == "1" for row in rows)
    aggregate_lower = selected_cost < fixed_cost
    aggregate_near = selected_score >= fixed_score - NEAR_QUALITY_TOL
    fallback_tasks = [row["task"] for row in rows if row["selected_reason"].startswith("fallback")]

    if all_near and all_lower:
        status = "go_for_per_task_domain_aware_controller_claim"
    elif all_near and aggregate_lower:
        status = "partial_aggregate_portfolio_only"
    else:
        status = "no_go_for_domain_aware_portfolio_claim"

    return {
        "portfolio_status": status,
        "tasks": str(n),
        "aggregate_score": fmt(selected_score),
        "fixed70_aggregate_score": fmt(fixed_score),
        "delta_score_vs_fixed70": fmt(selected_score - fixed_score),
        "aggregate_cost": fmt(selected_cost),
        "fixed70_aggregate_cost": fmt(fixed_cost),
        "delta_cost_vs_fixed70": fmt(selected_cost - fixed_cost),
        "all_tasks_near_fixed70": str(int(all_near)),
        "all_tasks_lower_cost": str(int(all_lower)),
        "aggregate_near_fixed70": str(int(aggregate_near)),
        "aggregate_lower_cost": str(int(aggregate_lower)),
        "fallback_tasks": ",".join(fallback_tasks) or "none",
        "safe_claim": safe_claim(status, fallback_tasks),
    }


def safe_claim(status: str, fallback_tasks: list[str]) -> str:
    if status == "go_for_per_task_domain_aware_controller_claim":
        return "A domain-aware pre-generation portfolio passes near-fixed70/lower-cost on every audited task."
    if status == "partial_aggregate_portfolio_only":
        return (
            "A task-aware portfolio can slightly reduce aggregate cost by using the TextVQA passing pre-generation policy "
            f"and fixed70 fallback for {','.join(fallback_tasks)}, but it does not solve per-task adaptive control."
        )
    return "The domain-aware portfolio does not preserve near-fixed70 aggregate quality at lower aggregate cost."


def build_report(rows: list[dict[str, Any]], decision: dict[str, str]) -> str:
    lines = [
        "# Open OCR QA Domain-Aware Portfolio Audit",
        "",
        "This diagnostic tests a conservative portfolio rule: use a task's best passing pre-generation policy when one exists; otherwise use fixed70. It is not a universal adaptive controller.",
        "",
        "## Decision",
        "",
        f"- portfolio_status: `{decision['portfolio_status']}`",
        f"- aggregate score: {decision['aggregate_score']} vs fixed70 {decision['fixed70_aggregate_score']}",
        f"- aggregate cost: {decision['aggregate_cost']} vs fixed70 {decision['fixed70_aggregate_cost']}",
        f"- fallback tasks: {decision['fallback_tasks']}",
        f"- safe claim: {decision['safe_claim']}",
        "",
        "## Task Rows",
        "",
        table_md(
            rows,
            [
                "task",
                "selected_policy",
                "selected_reason",
                "test_score",
                "deployment_cost_proxy",
                "fixed70_score",
                "fixed70_cost",
                "delta_score_vs_fixed70",
                "delta_cost_vs_fixed70",
                "near_fixed70",
                "lower_cost_than_fixed70",
            ],
        ),
        "",
        "## Interpretation",
        "",
        "If the status is partial, the result should be used only to show a limited deployment portfolio opportunity. It does not close the problem.md critique that a deployable adaptive controller should beat fixed70 on both TextVQA-lite and DocVQA-lite.",
    ]
    return "\n".join(lines) + "\n"


def find_policy(rows: list[dict[str, str]], policy: str) -> dict[str, str]:
    for row in rows:
        if row.get("policy") == policy:
            return row
    raise ValueError(f"missing policy {policy}")


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


def table_md(rows: list[dict[str, Any]], columns: list[str]) -> str:
    out = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(col, "")).replace("|", "/") for col in columns) + " |")
    return "\n".join(out)


def fnum(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def mean(values: Any) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def fmt(value: Any) -> str:
    return f"{fnum(value):.3f}"


if __name__ == "__main__":
    main()
