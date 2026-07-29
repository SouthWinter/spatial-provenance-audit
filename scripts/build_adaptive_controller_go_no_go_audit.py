#!/usr/bin/env python3
"""Aggregate adaptive-controller evidence into a single go/no-go audit.

problem.md asks whether the work can move beyond hand-picked selectors toward
a unified adaptive risk controller. Many local audits already exist; this
script puts them under one decision contract so we do not accidentally promote
a scoped or oracle result into the main method.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "problem_optimization_audit" / "adaptive_controller_go_no_go"

DEPLOYMENT_CONTRACT = ROOT / "runs" / "paper_evidence" / "table_open_ocr_qa_deployment_contract.csv"
DEPLOYMENT_SUMMARY = ROOT / "runs" / "paper_evidence" / "table_open_ocr_qa_deployment_contract_summary.csv"
UNIFIED_TRANSFER = ROOT / "runs" / "paper_evidence" / "table_open_ocr_qa_unified_policy_transfer_summary.csv"
DETECTOR_AWARE = ROOT / "runs" / "paper_evidence" / "table_open_ocr_qa_detector_aware_policy_summary.csv"
BOX_AWARE = ROOT / "runs" / "paper_evidence" / "table_open_ocr_qa_box_aware_budget_summary.csv"
NOISY_LATENCY = ROOT / "runs" / "paper_evidence" / "table_open_ocr_qa_noisy_box_latency_key_summary.csv"
CONFORMAL_DECISION = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_conformal_risk_policy"
    / "conformal_risk_policy_decision.csv"
)
CONFORMAL_SELECTION = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_conformal_risk_policy"
    / "conformal_risk_policy_selection.csv"
)
DOMAIN_AWARE_PORTFOLIO_DECISION = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_domain_aware_portfolio"
    / "domain_aware_portfolio_decision.csv"
)
SATURATION_DIR = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_evidence_saturation_policy"
)
SATURATION_DECISION = SATURATION_DIR / "evidence_saturation_policy_decision.csv"
SATURATION_SUMMARY = SATURATION_DIR / "evidence_saturation_policy_summary.csv"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_go_no_go_rows()
    decision = build_decision(rows)
    write_csv(OUT_DIR / "adaptive_controller_go_no_go.csv", rows)
    write_csv(OUT_DIR / "adaptive_controller_decision.csv", [decision])
    (OUT_DIR / "adaptive_controller_go_no_go.md").write_text(build_markdown(rows, decision), encoding="utf-8")
    print(f"Wrote adaptive controller go/no-go audit to {OUT_DIR}")
    print(f"main_controller_status={decision['main_controller_status']}")


def build_go_no_go_rows() -> list[dict[str, Any]]:
    deployment = read_csv(DEPLOYMENT_CONTRACT)
    unified = read_csv(UNIFIED_TRANSFER)
    detector = read_csv(DETECTOR_AWARE)
    box = read_csv(BOX_AWARE)
    noisy = read_csv(NOISY_LATENCY)
    conformal_decision = read_csv(CONFORMAL_DECISION)
    conformal_selection = read_csv(CONFORMAL_SELECTION)
    domain_portfolio = read_csv(DOMAIN_AWARE_PORTFOLIO_DECISION)
    saturation_decision = read_csv(SATURATION_DECISION)
    saturation_summary = read_csv(SATURATION_SUMMARY)

    rows = []
    rows.append(deployment_gate(deployment))
    rows.append(domain_aware_portfolio_gate(domain_portfolio))
    rows.append(conformal_risk_gate(conformal_decision, conformal_selection))
    rows.append(evidence_saturation_gate(saturation_decision, saturation_summary))
    rows.append(unified_transfer_gate(unified))
    rows.append(detector_gate(detector))
    rows.append(box_aware_gate(box))
    rows.append(noisy_latency_gate(noisy))
    rows.append(oracle_headroom_gate(deployment, box))
    return rows


def evidence_saturation_gate(
    decision: list[dict[str, str]],
    summaries: list[dict[str, str]],
) -> dict[str, Any]:
    decision_row = decision[0] if decision else {}
    adaptive_test = [
        row
        for row in summaries
        if row.get("split") == "test" and row.get("selected_by") == "pooled_dev"
    ]
    candidate = "; ".join(
        f"{row.get('task','')} score={row.get('score','')} keep={row.get('mean_keep','')}"
        for row in adaptive_test
    )
    status = "pass" if decision_row.get("status") == "go_for_main_controller_claim" else "fail"
    return {
        "gate": "pregen_evidence_saturation",
        "required_for_main_controller": "yes",
        "status": status,
        "best_candidate": candidate or "missing",
        "evidence": (
            f"status={decision_row.get('status','missing')}; "
            f"passed_tasks={decision_row.get('passed_tasks','0')} of "
            f"{decision_row.get('required_tasks','2')}; "
            f"target_max_keep={decision_row.get('target_max_keep','')}"
        ),
        "reading": "The pooled-dev saturation policy preserves fixed70-level held-out quality, but selects 70% for nearly every sample and therefore misses the efficiency gate.",
    }


def domain_aware_portfolio_gate(rows: list[dict[str, str]]) -> dict[str, Any]:
    row = rows[0] if rows else {}
    status = "partial" if row.get("portfolio_status") == "partial_aggregate_portfolio_only" else (
        "pass" if row.get("portfolio_status") == "go_for_per_task_domain_aware_controller_claim" else "fail"
    )
    return {
        "gate": "domain_aware_portfolio",
        "required_for_main_controller": "diagnostic",
        "status": status,
        "best_candidate": f"aggregate score={row.get('aggregate_score','')} cost={row.get('aggregate_cost','')} fallback={row.get('fallback_tasks','')}",
        "evidence": (
            f"status={row.get('portfolio_status','missing')}; "
            f"delta_score={row.get('delta_score_vs_fixed70','')}; delta_cost={row.get('delta_cost_vs_fixed70','')}; "
            f"all_tasks_lower_cost={row.get('all_tasks_lower_cost','')}"
        ),
        "reading": "A task-aware portfolio slightly reduces aggregate cost but falls back to fixed70 for DocVQA, so it is not a solved per-task adaptive controller.",
    }


def conformal_risk_gate(decision: list[dict[str, str]], selection: list[dict[str, str]]) -> dict[str, Any]:
    row = decision[0] if decision else {}
    main_rows = [item for item in selection if item.get("epsilon") == "0.010"]
    best = max(main_rows, key=lambda item: fnum(item.get("test_delta_vs_fixed70")), default={})
    status = "pass" if row.get("conformal_policy_status") == "go_for_conformal_controller_claim" else "fail"
    return {
        "gate": "calibrated_conformal_risk_control",
        "required_for_main_controller": "yes",
        "status": status,
        "best_candidate": f"{best.get('task','')} threshold={best.get('threshold','')} delta={best.get('test_delta_vs_fixed70','')} serial_cost={best.get('test_serial_cost','')}",
        "evidence": f"status={row.get('conformal_policy_status','missing')}; go_tasks={row.get('go_tasks','missing')}; epsilon={row.get('epsilon_for_main_gate','')}",
        "reading": "A calibrated 30%->70% fallback using low-budget risk scores cannot certify fixed70-quality at lower serial cost across TextVQA-lite and DocVQA-lite.",
    }


def deployment_gate(rows: list[dict[str, str]]) -> dict[str, Any]:
    pregen = [
        row
        for row in rows
        if row.get("deployable_before_llm_prefill") == "1"
        and row.get("uses_oracle_scores") == "0"
        and row.get("family") not in {"fixed"}
    ]
    passing = [row for row in pregen if row.get("passes_near_fixed70_lower_cost") == "1"]
    by_task = {row["task"]: row for row in passing}
    best_by_task = {}
    for task in sorted({row.get("task", "") for row in pregen}):
        task_rows = [row for row in pregen if row.get("task") == task]
        best_by_task[task] = max(task_rows, key=lambda row: fnum(row.get("test_score")), default={})
    pass_both = {"TextVQA-lite", "DocVQA-lite"}.issubset(by_task)
    reading = (
        "TextVQA has one pre-generation lower-cost near-fixed70 candidate, but DocVQA has none."
        if passing
        else "No pre-generation deployable candidate passes the gate."
    )
    return {
        "gate": "deployment_contract_pregen",
        "required_for_main_controller": "yes",
        "status": "pass" if pass_both else "fail",
        "best_candidate": "; ".join(
            f"{task}:{row.get('policy','none')} score={row.get('test_score','')} cost={row.get('deployment_cost_proxy','')}"
            for task, row in sorted(best_by_task.items())
        ),
        "evidence": f"passing_tasks={','.join(sorted(by_task)) or 'none'}; passing_candidates={len(passing)}",
        "reading": reading,
    }


def unified_transfer_gate(rows: list[dict[str, str]]) -> dict[str, Any]:
    total_candidates = sum(int_or_zero(row.get("near_fixed70_lower_keep_candidates")) for row in rows)
    best = max(rows, key=lambda row: fnum(row.get("best_quality_delta_vs_fixed70")), default={})
    return {
        "gate": "cross_task_or_pooled_transfer",
        "required_for_main_controller": "yes",
        "status": "pass" if total_candidates > 0 else "fail",
        "best_candidate": f"{best.get('scope','')}:{best.get('best_quality_policy','')} delta={best.get('best_quality_delta_vs_fixed70','')} keep={best.get('best_quality_target_keep','')}",
        "evidence": f"near_fixed70_lower_keep_candidates_total={total_candidates}",
        "reading": "No cross-task or pooled pre-generation controller passes the near-fixed70/lower-keep gate.",
    }


def detector_gate(rows: list[dict[str, str]]) -> dict[str, Any]:
    passes = [row for row in rows if str(row.get("test_passes_near_fixed70_lower_keep", "")).strip() == "1"]
    best_quality = max(rows, key=lambda row: fnum(row.get("test_delta_vs_fixed70")), default={})
    return {
        "gate": "detector_evidence_aware_stress_pack",
        "required_for_main_controller": "yes",
        "status": "pass" if passes else "fail",
        "best_candidate": f"{best_quality.get('task','')}:{best_quality.get('selected_policy','')} delta={best_quality.get('test_delta_vs_fixed70','')} keep={best_quality.get('test_mean_keep','')}",
        "evidence": f"heldout_passes={len(passes)} of {len(rows)}; mean_detector_ms_range={range_text(row.get('mean_detector_ms') for row in rows)}",
        "reading": "EasyOCR detector statistics and selector-mask evidence coverage still fail held-out stress-pack gates.",
    }


def box_aware_gate(rows: list[dict[str, str]]) -> dict[str, Any]:
    test_rows = [row for row in rows if row.get("split") == "test" and row.get("selected_by") == "dev_best"]
    fixed70 = {
        row.get("scope"): row
        for row in rows
        if row.get("split") == "test" and row.get("policy") == "fixed_0.70"
    }
    passing = []
    for row in test_rows:
        ref = fixed70.get(row.get("scope"))
        if not ref:
            continue
        if fnum(row.get("score")) >= fnum(ref.get("score")) - 0.01 and fnum(row.get("mean_keep")) < fnum(ref.get("mean_keep")):
            passing.append(row)
    annotated_subset = [row for row in passing if row.get("scope") == "answer_or_gt_bbox"]
    doc_context = [row for row in passing if row.get("scope") == "docvqa_line_context"]
    return {
        "gate": "box_aware_annotated_subset",
        "required_for_main_controller": "no",
        "status": "partial" if annotated_subset and not doc_context else ("pass" if annotated_subset and doc_context else "fail"),
        "best_candidate": "; ".join(
            f"{row.get('scope')}:{row.get('policy')} score={row.get('score')} keep={row.get('mean_keep')}"
            for row in passing[:4]
        )
        or "none",
        "evidence": f"passing_test_policies={len(passing)}; answer_or_gt_bbox_passes={len(annotated_subset)}; docvqa_line_context_passes={len(doc_context)}",
        "reading": "Box-aware ECR fallback can match fixed70 on a scoped annotated subset, but it does not give a general controller and often relies on near-full fallback for document context.",
    }


def noisy_latency_gate(rows: list[dict[str, str]]) -> dict[str, Any]:
    detector_rows = [row for row in rows if row.get("requires_detector") == "yes"]
    no_detector_positive = [row for row in detector_rows if fnum(row.get("speedup_no_detector")) > 1.0]
    detector_positive = [row for row in detector_rows if fnum(row.get("speedup_with_mean_detector")) > 1.0]
    best = max(detector_rows, key=lambda row: fnum(row.get("speedup_with_mean_detector")), default={})
    return {
        "gate": "detector_inclusive_latency",
        "required_for_main_controller": "yes_for_box_aware",
        "status": "pass" if detector_positive else "fail",
        "best_candidate": f"{best.get('scope','')}:{best.get('policy','')} speedup_mean_detector={best.get('speedup_with_mean_detector','')} score={best.get('score','')}",
        "evidence": f"detector_rows={len(detector_rows)}; no_detector_speedup_rows={len(no_detector_positive)}; detector_inclusive_speedup_rows={len(detector_positive)}",
        "reading": "Detector-assisted selected-keep savings do not translate into single-sample speedup when mean EasyOCR latency is counted.",
    }


def oracle_headroom_gate(deployment: list[dict[str, str]], box: list[dict[str, str]]) -> dict[str, Any]:
    oracle_rows = [row for row in deployment if row.get("family") == "oracle"] + [
        row for row in box if row.get("policy") == "oracle_best_budget" and row.get("split") == "test"
    ]
    deployable_rows = [row for row in deployment if row.get("deployable") == "1" and row.get("uses_oracle_scores") == "0"]
    best_oracle = max(oracle_rows, key=lambda row: fnum(row.get("test_score") or row.get("score")), default={})
    best_deploy = max(deployable_rows, key=lambda row: fnum(row.get("test_score")), default={})
    return {
        "gate": "oracle_headroom_vs_deployable_gap",
        "required_for_main_controller": "diagnostic",
        "status": "headroom_exists",
        "best_candidate": f"oracle={best_oracle.get('policy','')} score={best_oracle.get('test_score') or best_oracle.get('score','')} keep={best_oracle.get('deployment_cost_proxy') or best_oracle.get('mean_keep','')}; deployable={best_deploy.get('policy','')} score={best_deploy.get('test_score','')} cost={best_deploy.get('deployment_cost_proxy','')}",
        "evidence": f"oracle_rows={len(oracle_rows)}; deployable_rows={len(deployable_rows)}",
        "reading": "There is real oracle headroom, so the negative result is about current deployable signals rather than impossibility in principle.",
    }


def build_decision(rows: list[dict[str, Any]]) -> dict[str, str]:
    required = [row for row in rows if row["required_for_main_controller"] in {"yes", "yes_for_box_aware"}]
    failed_required = [row for row in required if row["status"] != "pass"]
    partial = [row for row in rows if row["status"] == "partial"]
    status = "no_go_for_main_method_claim" if failed_required else "go_for_main_method_claim"
    return {
        "main_controller_status": status,
        "failed_required_gates": str(len(failed_required)),
        "partial_gates": str(len(partial)),
        "recommended_claim": (
            "Keep adaptive control as a diagnostic boundary and future-work opportunity; do not claim a solved unified deployable controller."
            if failed_required
            else "A unified deployable controller can be claimed under the audited contract."
        ),
        "next_design_requirement": "A viable controller must be pre-generation or amortized, pass both TextVQA-lite and DocVQA-lite near-fixed70/lower-cost gates, and preserve detector-inclusive speed when boxes are required.",
    }


def build_markdown(rows: list[dict[str, Any]], decision: dict[str, str]) -> str:
    lines = [
        "# Adaptive Controller Go/No-Go Audit",
        "",
        "This audit aggregates existing adaptive-budget, transfer, detector-aware, box-aware, noisy-box, and latency checks under one decision contract. It is intended to answer whether adaptive risk control is strong enough to become a main method claim.",
        "",
        "## Decision",
        "",
        f"- Main-controller status: `{decision['main_controller_status']}`",
        f"- Failed required gates: {decision['failed_required_gates']}",
        f"- Partial gates: {decision['partial_gates']}",
        f"- Recommended claim: {decision['recommended_claim']}",
        f"- Next design requirement: {decision['next_design_requirement']}",
        "",
        "## Gate Table",
        "",
        table_md(rows, ["gate", "required_for_main_controller", "status", "best_candidate", "evidence", "reading"]),
        "",
        "## Source Tables",
        "",
        f"- `{DEPLOYMENT_CONTRACT.relative_to(ROOT)}`",
        f"- `{DEPLOYMENT_SUMMARY.relative_to(ROOT)}`",
        f"- `{UNIFIED_TRANSFER.relative_to(ROOT)}`",
        f"- `{DETECTOR_AWARE.relative_to(ROOT)}`",
        f"- `{BOX_AWARE.relative_to(ROOT)}`",
        f"- `{NOISY_LATENCY.relative_to(ROOT)}`",
        f"- `{SATURATION_DECISION.relative_to(ROOT)}`",
        f"- `{SATURATION_SUMMARY.relative_to(ROOT)}`",
        "",
    ]
    return "\n".join(lines)


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
        out.append("| " + " | ".join(clean_cell(row.get(col, "")) for col in columns) + " |")
    return "\n".join(out)


def clean_cell(value: Any) -> str:
    return str(value).replace("\n", " ").replace("|", "/")


def fnum(value: Any) -> float:
    try:
        if value is None or value == "":
            return float("-inf")
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def int_or_zero(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def range_text(values: Any) -> str:
    nums = [fnum(value) for value in values]
    nums = [value for value in nums if value != float("-inf")]
    if not nums:
        return ""
    return f"{min(nums):.3f}-{max(nums):.3f}"


if __name__ == "__main__":
    main()
