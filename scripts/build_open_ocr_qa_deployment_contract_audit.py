#!/usr/bin/env python3
"""Deployment-contract audit for open OCR QA adaptive pruning policies.

The adaptive results include fixed budgets, oracle frontiers, pre-generation
risk probes, learned policies that use a low-budget answer, and serial answer
stability cascades. This script puts them under one deployment contract so the
paper can distinguish deployable policies from diagnostics.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "problem_optimization_audit" / "open_ocr_qa_deployment_contract"
PAPER_DIR = ROOT / "runs" / "paper_evidence"

LEARNED_POLICY = PAPER_DIR / "table_open_ocr_qa_learned_risk_policy.csv"
PREGEN_POLICY = PAPER_DIR / "table_open_ocr_qa_pregen_risk_signal_policy.csv"
ANSWER_STABILITY = PAPER_DIR / "table_open_ocr_qa_answer_stability_cascade.csv"
ANSWER_STABILITY_SELECTION = PAPER_DIR / "table_open_ocr_qa_answer_stability_selection.csv"
RISK_FRONTIER = PAPER_DIR / "table_open_ocr_qa_risk_coverage_frontier.csv"

PREGEN_GROUPS = {"question_only", "mask30_only", "mask_stability_pregen", "question_mask_pregen"}
LOWANSWER_GROUPS = {"question_mask_lowanswer"}
TASKS = ("DocVQA-lite", "TextVQA-lite")
TOLERANCE = 0.01


def main() -> None:
    fixed_rows = read_csv(LEARNED_POLICY)
    pregen_rows = read_csv(PREGEN_POLICY)
    cascade_rows = read_csv(ANSWER_STABILITY)
    cascade_selection_rows = read_csv(ANSWER_STABILITY_SELECTION)
    oracle_rows = read_csv(RISK_FRONTIER)

    rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for task in TASKS:
        fixed70 = find_one(fixed_rows, task=task, split="test", family="fixed", policy="fixed_0.70")
        fixed30 = find_one(fixed_rows, task=task, split="test", family="fixed", policy="fixed_0.30")
        full = find_one(fixed_rows, task=task, split="test", family="fixed", policy="fixed_1.00")
        fixed70_score = f(fixed70["score"])
        fixed70_cost = f(fixed70["mean_keep"])

        candidates = []
        candidates.append(contract_row(task, "fixed_0.30", "fixed", "fixed", fixed30, fixed70_score, fixed70_cost))
        candidates.append(contract_row(task, "fixed_0.70", "fixed", "fixed", fixed70, fixed70_score, fixed70_cost))
        candidates.append(contract_row(task, "full_prefix", "fixed", "fixed", full, fixed70_score, fixed70_cost))

        for feature_group in sorted(PREGEN_GROUPS):
            row = find_one(pregen_rows, task=task, feature_group=feature_group)
            candidates.append(pregen_contract_row(task, row, fixed70_score, fixed70_cost, "pre_generation"))
        for feature_group in sorted(LOWANSWER_GROUPS):
            row = find_one(pregen_rows, task=task, feature_group=feature_group)
            candidates.append(pregen_contract_row(task, row, fixed70_score, fixed70_cost, "post_low_budget_answer"))

        for family in ("learned_to70", "learned_threeway", "learned_fullfallback"):
            row = find_one(fixed_rows, task=task, split="test", family=family)
            candidates.append(learned_contract_row(task, row, fixed70_score, fixed70_cost))

        for family in ("two_pass_stability", "two_pass_stability_full", "three_pass_stability", "three_pass_majority"):
            selected = find_one(cascade_selection_rows, task=task, family=family)
            row = find_one(cascade_rows, task=task, split="test", family=family, policy=selected["selected_policy"])
            candidates.append(cascade_contract_row(task, row, fixed70_score, fixed70_cost))

        oracle = find_one(oracle_rows, task=task, split="test", tolerance="0.000")
        candidates.append(oracle_contract_row(task, oracle, fixed70_score, fixed70_cost))

        rows.extend(candidates)
        summary.extend(task_summary(task, candidates, fixed70_score, fixed70_cost))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "deployment_contract_rows.csv", rows)
    write_csv(OUT_DIR / "deployment_contract_summary.csv", summary)
    write_csv(PAPER_DIR / "table_open_ocr_qa_deployment_contract.csv", rows)
    write_csv(PAPER_DIR / "table_open_ocr_qa_deployment_contract_summary.csv", summary)
    (OUT_DIR / "deployment_contract_report.md").write_text(build_report(rows, summary), encoding="utf-8")
    print(f"Wrote deployment contract audit to {OUT_DIR}")


def contract_row(
    task: str,
    policy: str,
    family: str,
    contract: str,
    row: dict[str, Any],
    fixed70_score: float,
    fixed70_cost: float,
) -> dict[str, Any]:
    score = f(row.get("score"))
    selected_keep = f(row.get("mean_keep"))
    cost = selected_keep
    return base_row(
        task=task,
        policy=policy,
        family=family,
        deployment_contract=contract,
        test_score=score,
        fixed70_score=fixed70_score,
        selected_keep=selected_keep,
        deployment_cost_proxy=cost,
        fixed70_cost=fixed70_cost,
        uses_oracle_scores=0,
        requires_extra_generation=0,
        uses_generated_answer_signal=0,
        deployable_before_llm_prefill=1 if policy.startswith("fixed") or policy == "full_prefix" else 0,
        note=str(row.get("note", "")),
    )


def pregen_contract_row(
    task: str,
    row: dict[str, Any],
    fixed70_score: float,
    fixed70_cost: float,
    contract: str,
) -> dict[str, Any]:
    score = f(row["test_score"])
    selected_keep = f(row["test_mean_keep"])
    fallback70 = f(row["test_fallback70_rate"])
    uses_answer = int(contract == "post_low_budget_answer")
    cost = selected_keep + (0.30 * fallback70 if uses_answer else 0.0)
    return base_row(
        task=task,
        policy=f"pregen_{row['feature_group']}",
        family="pregen_risk_signal",
        deployment_contract=contract,
        test_score=score,
        fixed70_score=fixed70_score,
        selected_keep=selected_keep,
        deployment_cost_proxy=cost,
        fixed70_cost=fixed70_cost,
        uses_oracle_scores=0,
        requires_extra_generation=uses_answer,
        uses_generated_answer_signal=uses_answer,
        deployable_before_llm_prefill=0 if uses_answer else 1,
        note=str(row.get("note", "")),
    )


def learned_contract_row(
    task: str,
    row: dict[str, Any],
    fixed70_score: float,
    fixed70_cost: float,
) -> dict[str, Any]:
    score = f(row["score"])
    selected_keep = f(row["mean_keep"])
    fallback50 = f(row.get("fallback_rate_ge_0p50", 0.0))
    cost = selected_keep + 0.30 * fallback50
    return base_row(
        task=task,
        policy=str(row["policy"]),
        family=str(row["family"]),
        deployment_contract="post_low_budget_answer",
        test_score=score,
        fixed70_score=fixed70_score,
        selected_keep=selected_keep,
        deployment_cost_proxy=cost,
        fixed70_cost=fixed70_cost,
        uses_oracle_scores=0,
        requires_extra_generation=1,
        uses_generated_answer_signal=1,
        deployable_before_llm_prefill=0,
        note=str(row.get("note", "")),
    )


def cascade_contract_row(
    task: str,
    row: dict[str, Any],
    fixed70_score: float,
    fixed70_cost: float,
) -> dict[str, Any]:
    score = f(row["score"])
    selected_keep = f(row["selected_mean_keep"])
    cost = f(row["cascade_mean_cost"])
    return base_row(
        task=task,
        policy=str(row["policy"]),
        family=str(row["family"]),
        deployment_contract="serial_answer_stability",
        test_score=score,
        fixed70_score=fixed70_score,
        selected_keep=selected_keep,
        deployment_cost_proxy=cost,
        fixed70_cost=fixed70_cost,
        uses_oracle_scores=0,
        requires_extra_generation=1,
        uses_generated_answer_signal=1,
        deployable_before_llm_prefill=0,
        note=str(row.get("note", "")),
    )


def oracle_contract_row(
    task: str,
    row: dict[str, Any],
    fixed70_score: float,
    fixed70_cost: float,
) -> dict[str, Any]:
    score = f(row["oracle_score"])
    selected_keep = f(row["mean_keep"])
    return base_row(
        task=task,
        policy="oracle_0tol",
        family="oracle",
        deployment_contract="oracle_upper_bound",
        test_score=score,
        fixed70_score=fixed70_score,
        selected_keep=selected_keep,
        deployment_cost_proxy=selected_keep,
        fixed70_cost=fixed70_cost,
        uses_oracle_scores=1,
        requires_extra_generation=0,
        uses_generated_answer_signal=0,
        deployable_before_llm_prefill=0,
        note="uses full-prefix/gold score to choose budget; diagnostic upper bound only",
    )


def base_row(
    *,
    task: str,
    policy: str,
    family: str,
    deployment_contract: str,
    test_score: float,
    fixed70_score: float,
    selected_keep: float,
    deployment_cost_proxy: float,
    fixed70_cost: float,
    uses_oracle_scores: int,
    requires_extra_generation: int,
    uses_generated_answer_signal: int,
    deployable_before_llm_prefill: int,
    note: str,
) -> dict[str, Any]:
    delta_score = test_score - fixed70_score
    delta_cost = deployment_cost_proxy - fixed70_cost
    deployable = int(not uses_oracle_scores)
    pass_contract = int(deployable and delta_score >= -TOLERANCE and deployment_cost_proxy < fixed70_cost)
    return {
        "task": task,
        "policy": policy,
        "family": family,
        "deployment_contract": deployment_contract,
        "test_score": fmt(test_score),
        "fixed70_score": fmt(fixed70_score),
        "delta_score_vs_fixed70": fmt(delta_score),
        "selected_keep": fmt(selected_keep),
        "deployment_cost_proxy": fmt(deployment_cost_proxy),
        "fixed70_cost": fmt(fixed70_cost),
        "delta_cost_vs_fixed70": fmt(delta_cost),
        "deployable": deployable,
        "deployable_before_llm_prefill": deployable_before_llm_prefill,
        "requires_extra_generation": requires_extra_generation,
        "uses_generated_answer_signal": uses_generated_answer_signal,
        "uses_oracle_scores": uses_oracle_scores,
        "passes_near_fixed70_lower_cost": pass_contract,
        "note": note,
    }


def task_summary(
    task: str,
    rows: list[dict[str, Any]],
    fixed70_score: float,
    fixed70_cost: float,
) -> list[dict[str, Any]]:
    deployable = [row for row in rows if int(row["deployable"]) == 1]
    prefill = [row for row in rows if int(row["deployable_before_llm_prefill"]) == 1 and row["family"] != "fixed"]
    passing = [row for row in rows if int(row["passes_near_fixed70_lower_cost"]) == 1]
    best_deployable = max(deployable, key=lambda row: f(row["test_score"]))
    best_prefill = max(prefill, key=lambda row: f(row["test_score"])) if prefill else None
    cheapest_near = min(passing, key=lambda row: f(row["deployment_cost_proxy"])) if passing else None
    out = [
        {
            "task": task,
            "summary_item": "fixed70_reference",
            "policy": "fixed_0.70",
            "test_score": fmt(fixed70_score),
            "deployment_cost_proxy": fmt(fixed70_cost),
            "interpretation": "reference policy for near-quality/lower-cost gate",
        },
        {
            "task": task,
            "summary_item": "best_deployable_score",
            "policy": best_deployable["policy"],
            "test_score": best_deployable["test_score"],
            "deployment_cost_proxy": best_deployable["deployment_cost_proxy"],
            "interpretation": "best score among non-oracle candidates, regardless of cost",
        },
    ]
    if best_prefill:
        out.append(
            {
                "task": task,
                "summary_item": "best_prefill_deployable_score",
                "policy": best_prefill["policy"],
                "test_score": best_prefill["test_score"],
                "deployment_cost_proxy": best_prefill["deployment_cost_proxy"],
                "interpretation": "best pre-generation candidate; no generated answer signal",
            }
        )
    out.append(
        {
            "task": task,
            "summary_item": "near_fixed70_lower_cost_candidate",
            "policy": cheapest_near["policy"] if cheapest_near else "none",
            "test_score": cheapest_near["test_score"] if cheapest_near else "",
            "deployment_cost_proxy": cheapest_near["deployment_cost_proxy"] if cheapest_near else "",
            "interpretation": (
                "passes score >= fixed70 - 0.01 and lower deployment cost"
                if cheapest_near
                else "no deployable candidate passes score >= fixed70 - 0.01 with lower deployment cost"
            ),
        }
    )
    return out


def build_report(rows: list[dict[str, Any]], summary: list[dict[str, Any]]) -> str:
    key_cols = [
        "task",
        "policy",
        "deployment_contract",
        "test_score",
        "delta_score_vs_fixed70",
        "selected_keep",
        "deployment_cost_proxy",
        "delta_cost_vs_fixed70",
        "deployable_before_llm_prefill",
        "requires_extra_generation",
        "uses_oracle_scores",
        "passes_near_fixed70_lower_cost",
    ]
    lines = [
        "# Open OCR QA Deployment-Contract Audit",
        "",
        "This audit puts adaptive open-QA policies under a common deployment contract. The key distinction is between selected keep ratio and a deployment-cost proxy: policies that first generate a low-budget answer or run serial stability checks pay extra cost beyond the final selected prefix.",
        "",
        f"Near-fixed70 lower-cost gate: test score >= fixed70 - {TOLERANCE:.2f} and deployment-cost proxy < fixed70 cost.",
        "",
        "## Summary",
        "",
        table_md(summary, ["task", "summary_item", "policy", "test_score", "deployment_cost_proxy", "interpretation"]),
        "",
        "## Candidate Rows",
        "",
        table_md(rows, key_cols),
        "",
        "## Interpretation",
        "",
        "- Oracle rows quantify headroom but use per-sample full-prefix/gold score information and are not deployable.",
        "- Answer-stability cascades can improve quality, but their serial cost can exceed a fixed 70% pass even when their selected keep ratio looks small.",
        "- Pre-generation policies are the cleanest deployment candidates. A passing row is a scoped positive result; a missing passing row means the current signals should remain diagnostics rather than a solved adaptive controller.",
        "",
    ]
    return "\n".join(lines)


def find_one(rows: list[dict[str, Any]], **criteria: str) -> dict[str, Any]:
    found = []
    for row in rows:
        ok = True
        for key, value in criteria.items():
            if str(row.get(key, "")) != str(value):
                ok = False
                break
        if ok:
            found.append(row)
    if len(found) != 1:
        raise ValueError(f"Expected one row for {criteria}, found {len(found)}")
    return found[0]


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


def f(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def fmt(value: float) -> str:
    return f"{value:.3f}"


if __name__ == "__main__":
    main()
