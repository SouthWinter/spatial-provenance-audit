#!/usr/bin/env python3
"""Build a compact dashboard for remaining problem.md blockers.

The project now has many focused go/no-go and progress audits. This script
collects the reviewer-facing decision state into one table so manuscript
claims can stay aligned with what the evidence actually supports.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "runs" / "problem_optimization_audit"
PAPER_EVIDENCE_DIR = ROOT / "runs" / "paper_evidence"
OUT_DIR = AUDIT_DIR / "problem_md_remaining_blockers"

INPUTS = {
    "causal": AUDIT_DIR / "causal_evidence_go_no_go" / "causal_evidence_decision.csv",
    "adaptive": AUDIT_DIR / "adaptive_controller_go_no_go" / "adaptive_controller_decision.csv",
    "manual_annotation": AUDIT_DIR
    / "open_ocr_qa_manual_annotation_launch"
    / "progress_audit"
    / "manual_annotation_progress_decision.csv",
    "manual_final_package": AUDIT_DIR
    / "open_ocr_qa_manual_final_package"
    / "manual_final_package_status.csv",
    "hard_negative_qc": AUDIT_DIR
    / "hard_negative_human_qc_progress"
    / "hard_negative_human_qc_progress_decision.csv",
    "text_replacement_qc": AUDIT_DIR
    / "text_replacement_human_qc_progress"
    / "text_replacement_human_qc_progress_decision.csv",
    "text_replacement_human_valid": AUDIT_DIR
    / "text_replacement_control_pack_v3"
    / "human_valid_eval"
    / "human_valid_model_summary.csv",
    "human_qc_claim": AUDIT_DIR
    / "human_qc_claim_gate"
    / "human_qc_claim_gate_decision.csv",
    "human_qc_claim_rows": AUDIT_DIR
    / "human_qc_claim_gate"
    / "human_qc_claim_gate_rows.csv",
    "strong_accept": AUDIT_DIR
    / "problem_md_strong_accept_readiness"
    / "strong_accept_upgrade_decision.csv",
    "domain_portfolio": AUDIT_DIR
    / "open_ocr_qa_domain_aware_portfolio"
    / "domain_aware_portfolio_decision.csv",
    "conformal_policy": AUDIT_DIR
    / "open_ocr_qa_conformal_risk_policy"
    / "conformal_risk_policy_decision.csv",
    "requirements": AUDIT_DIR
    / "problem_md_requirement_audit"
    / "problem_md_requirement_audit.csv",
    "full_open_qa": ROOT / "runs" / "open_ocr_qa_full" / "report" / "full_open_ocr_qa_runs.csv",
    "strict_controller": AUDIT_DIR
    / "strict_remaining_plan_go_no_go"
    / "strict_remaining_plan_report.json",
    "method_summary": PAPER_EVIDENCE_DIR / "table_method_component_pareto_summary.csv",
    "external_baselines": PAPER_EVIDENCE_DIR / "table_external_baseline_fairness_matrix.csv",
    "efficiency": PAPER_EVIDENCE_DIR / "table_efficiency_decomposition.csv",
    "statistics": PAPER_EVIDENCE_DIR / "table_statistics.csv",
    "internvl_calibration": PAPER_EVIDENCE_DIR / "table_internvl_threshold_calibration_summary.csv",
    "robustness": PAPER_EVIDENCE_DIR / "table_hard_robustness_conflict_summary.csv",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    write_csv(OUT_DIR / "problem_md_remaining_blockers.csv", rows)
    (OUT_DIR / "problem_md_remaining_blockers.md").write_text(build_markdown(rows), encoding="utf-8")
    print(f"Wrote remaining-blocker dashboard to {OUT_DIR}")


def build_rows() -> list[dict[str, str]]:
    causal = first("causal")
    adaptive = first("adaptive")
    manual = first("manual_annotation")
    manual_final = read_csv(INPUTS["manual_final_package"])
    hard_neg = first("hard_negative_qc")
    text_replace = first("text_replacement_qc")
    text_replace_valid = read_csv(INPUTS["text_replacement_human_valid"])
    human_qc = first("human_qc_claim")
    human_qc_rows = read_csv(INPUTS["human_qc_claim_rows"])
    strong = first("strong_accept")
    domain = first("domain_portfolio")
    conformal = first("conformal_policy")
    requirements = read_csv(INPUTS["requirements"])
    full_open_qa = read_csv(INPUTS["full_open_qa"])
    strict_controller = read_json(INPUTS["strict_controller"])
    method_summary = read_csv(INPUTS["method_summary"])
    external_baselines = read_csv(INPUTS["external_baselines"])
    efficiency = read_csv(INPUTS["efficiency"])
    statistics = read_csv(INPUTS["statistics"])
    internvl_calibration = read_csv(INPUTS["internvl_calibration"])
    robustness = read_csv(INPUTS["robustness"])

    qwen_text70 = find_row(full_open_qa, model="Qwen3-VL-8B", task="TextVQA", budget="70%")
    qwen_doc70 = find_row(full_open_qa, model="Qwen3-VL-8B", task="DocVQA", budget="70%")
    llava_text70 = find_row(full_open_qa, model="LLaVA-1.5-7B", task="TextVQA", budget="70%")
    llava_doc70 = find_row(full_open_qa, model="LLaVA-1.5-7B", task="DocVQA", budget="70%")
    strict_contract = strict_controller.get("controller_contract", {})
    method_best_acc = find_row(method_summary, summary_item="best_accuracy")
    method_best_ecr = find_row(method_summary, summary_item="best_ECR")
    method_coverage_tradeoff = find_row(method_summary, summary_item="coverage_tradeoff_count")
    method_similar_risk = find_row(method_summary, summary_item="evidence_gain_similar_risk_count")
    done_baselines = [row for row in external_baselines if row.get("result_status") == "done"]
    unsupported_baselines = [row for row in external_baselines if row.get("result_status") == "unsupported"]
    qwen_target20_eff = find_row(efficiency, point="Qwen Target 20%")
    llava_protected_eff = find_row(efficiency, point="LLaVA Protected 40%")
    internvl_soft_eff = find_row(efficiency, point="InternVL Soft evidence 50%")
    qwen_target30_full_stats = find_row(statistics, comparison="qwen_target0p30_vs_full")
    internvl_full_cal = find_row(internvl_calibration, model_run="InternVL3.5-8B-full")
    robustness_statuses = sorted({row.get("status", "") for row in robustness if row.get("status", "")})
    qwen_semantic = find_row(
        text_replace_valid, model="Qwen3-VL-8B", metric="all_four_controls_correct"
    )
    internvl_semantic = find_row(
        text_replace_valid, model="InternVL3.5-8B", metric="all_four_controls_correct"
    )
    text_qc_complete = (
        text_replace.get("text_replacement_human_qc_status")
        == "ready_for_verified_semantic_counterfactual_claim"
    )

    return [
        {
            "gate": "causal evidence use",
            "problem_md_link": "R1",
            "current_status": causal.get("causal_claim_status", "missing"),
            "paper_status": "write as causal-style support with scope limits",
            "remaining_blocker": (
                "ECR is availability rather than proof of use; LLaVA remains weak; "
                "human-verified semantic replacement is strong only for Qwen."
            ),
            "next_action": "Keep triad diagnostics separate and report the Qwen-specific semantic result with the InternVL/LLaVA boundaries.",
            "key_numbers": (
                f"qwen_strong={causal.get('qwen_strong','')}; "
                f"internvl_strong={causal.get('internvl_strong','')}; "
                f"llava_weak={causal.get('llava_weak_boundary','')}; "
                f"semantic_strong={causal.get('semantic_counterfactual_strong','')}"
            ),
            "claim_boundary": "do not say ECR proves causal use",
        },
        {
            "gate": "semantic text-replacement QC",
            "problem_md_link": "R1",
            "current_status": (
                f"{text_replace.get('text_replacement_human_qc_status', 'missing')}; "
                f"aggregate={human_qc.get('human_qc_claim_status', 'missing')}"
            ),
            "paper_status": (
                "completed human-verified model-dependent diagnostic"
                if text_qc_complete
                else "boundary audit only"
            ),
            "remaining_blocker": (
                "InternVL does not reproduce Qwen's semantic-switch strength, so the result is not backbone-uniform."
                if text_qc_complete
                else "Human edit-validity QC has not been completed."
            ),
            "next_action": (
                "Report Qwen as human-verified semantic support and InternVL as the explicit cross-backbone boundary."
                if text_qc_complete
                else "Fill the QC template, then rerun the progress audit before promoting semantic-counterfactual claims."
            ),
            "key_numbers": (
                f"ready={text_replace.get('ready_rows','')}/{text_replace.get('rows','')}; "
                f"valid_rate={text_replace.get('valid_semantic_edit_rate','')}; "
                f"Qwen_strict={qwen_semantic.get('successes','')}/{qwen_semantic.get('n','')}; "
                f"InternVL_strict={internvl_semantic.get('successes','')}/{internvl_semantic.get('n','')}; "
                f"stale_tables_absent={gate_status(human_qc_rows, 'stale_human_qc_paper_tables_absent')}"
            ),
            "claim_boundary": (
                "do not claim backbone-uniform semantic causal use"
                if text_qc_complete
                else "do not say human-verified semantic edit"
            ),
        },
        {
            "gate": "native open-answer OCR/document QA",
            "problem_md_link": "R2",
            "current_status": req_status(requirements, "R2"),
            "paper_status": "full-validation cross-model boundary evidence",
            "remaining_blocker": "At 70% keep, both backbones still lose quality on both tasks; the runs do not establish lossless or universally successful open QA.",
            "next_action": "Use full-validation rows to answer the scale criticism; retain lite sets only for controlled selector/evidence ablations.",
            "key_numbers": (
                f"{format_open_qa(qwen_text70)}; {format_open_qa(qwen_doc70)}; "
                f"{format_open_qa(llava_text70)}; {format_open_qa(llava_doc70)}"
            ),
            "claim_boundary": "do not claim lossless or leaderboard-optimized open QA",
        },
        {
            "gate": "manual multi-region evidence",
            "problem_md_link": "R3",
            "current_status": (
                f"{first_value(manual, 'manual_annotation_progress_status', 'progress_status', default='missing')}; "
                f"paper_copy={gate_status(manual_final, 'paper_evidence_copy_allowed')}"
            ),
            "paper_status": "completed human-adjudicated stress audit",
            "remaining_blocker": "The audit is complete but remains a 96-sample stress subset rather than full-benchmark annotation.",
            "next_action": "Use the final ECR and quality-association tables with the stated scope boundary.",
            "key_numbers": (
                f"primary_ready={manual.get('primary_ready_rows','')}/{manual.get('total_primary_rows','')}; "
                f"calibration_ready={manual.get('calibration_ready_rows','')}/{manual.get('total_calibration_rows','')}; "
                f"stale_tables_absent={gate_status(manual_final, 'stale_paper_evidence_tables_absent')}"
            ),
            "claim_boundary": "do not generalize the 96-sample audit to full TextVQA/DocVQA or causal evidence use",
        },
        {
            "gate": "hard-negative human QC",
            "problem_md_link": "R8",
            "current_status": (
                f"{hard_neg.get('hard_negative_human_qc_status', 'missing')}; "
                f"aggregate={human_qc.get('human_qc_claim_status', 'missing')}"
            ),
            "paper_status": "automatic construction audit plus completed 100-row human QC",
            "remaining_blocker": "The human audit covers 100 sampled negatives rather than all 500.",
            "next_action": "Report 100/100 sampled validity while retaining the non-exhaustive scope qualifier.",
            "key_numbers": (
                f"ready={hard_neg.get('ready_rows','')}/{hard_neg.get('rows','')}; "
                f"valid_negative_rows={hard_neg.get('valid_negative_rows','')}; "
                f"invalid_rows={hard_neg.get('invalid_rows','')}; "
                f"stale_tables_absent={gate_status(human_qc_rows, 'stale_human_qc_paper_tables_absent')}"
            ),
            "claim_boundary": "do not say full human validation of hard negatives",
        },
        {
            "gate": "unified adaptive controller",
            "problem_md_link": "R4",
            "current_status": adaptive.get("main_controller_status", "missing"),
            "paper_status": "negative boundary result",
            "remaining_blocker": "Deployable policies do not pass cost-quality gates across TextVQA and DocVQA.",
            "next_action": "Report the no-go and stop incremental controller expansion unless a genuinely new pre-generation signal is available.",
            "key_numbers": (
                f"failed_required_gates={adaptive.get('failed_required_gates','')}; "
                f"partial_gates={adaptive.get('partial_gates','')}; "
                f"strict_status={strict_contract.get('status','missing')}; "
                f"strict_passing_candidates={len(strict_contract.get('passing_candidates', []))}"
            ),
            "claim_boundary": "do not claim solved unified adaptive risk control",
        },
        {
            "gate": "domain-aware portfolio",
            "problem_md_link": "R4",
            "current_status": domain.get("portfolio_status", "missing"),
            "paper_status": "partial diagnostic",
            "remaining_blocker": "Aggregate cost improves slightly, but the policy falls back on DocVQA and is not per-task adaptive control.",
            "next_action": "Use only as supporting evidence for scoped deployment trade-offs.",
            "key_numbers": (
                f"aggregate_score={domain.get('aggregate_score','')}; "
                f"fixed70_score={domain.get('fixed70_aggregate_score','')}; "
                f"aggregate_cost={domain.get('aggregate_cost','')}; "
                f"fixed70_cost={domain.get('fixed70_aggregate_cost','')}"
            ),
            "claim_boundary": "do not claim a universal controller",
        },
        {
            "gate": "calibrated/conformal fallback",
            "problem_md_link": "R4",
            "current_status": conformal.get("conformal_policy_status", "missing"),
            "paper_status": "diagnostic or no-go depending on task",
            "remaining_blocker": "Fallback policies still do not establish broad deployable adaptive control.",
            "next_action": "Use as one diagnostic in the adaptive-controller audit, not as a main algorithmic contribution.",
            "key_numbers": (
                f"go_tasks={conformal.get('go_tasks','')}; "
                f"epsilon={conformal.get('epsilon_for_main_gate','')}"
            ),
            "claim_boundary": "do not overstate calibrated fallback as solved risk control",
        },
        {
            "gate": "method principle and objective",
            "problem_md_link": "R5",
            "current_status": req_status(requirements, "R5"),
            "paper_status": "principled selector family plus negative/diagnostic component audit",
            "remaining_blocker": "The objective clarifies trade-offs but does not become a universally better optimizer.",
            "next_action": "Frame the method as evidence-risk design and Pareto audit; keep coverage-greedy as an ablation boundary.",
            "key_numbers": (
                f"best_accuracy={method_best_acc.get('value','')}; "
                f"best_ECR={method_best_ecr.get('value','')}; "
                f"coverage_tradeoff_count={method_coverage_tradeoff.get('value','')}; "
                f"similar_risk_evidence_gain={method_similar_risk.get('value','')}"
            ),
            "claim_boundary": "do not claim coverage maximization universally improves answers",
        },
        {
            "gate": "external baseline parity",
            "problem_md_link": "R6",
            "current_status": req_status(requirements, "R6"),
            "paper_status": "fair for claimed LLaVA ports and scoped Qwen3 VisionZip, not universal parity",
            "remaining_blocker": "Qwen3 FastV and InternVL external baseline ports remain unsupported.",
            "next_action": "Claim only the backbone-method pairs with source-level parity and matched-budget evidence.",
            "key_numbers": (
                f"done_rows={len(done_baselines)}; unsupported_rows={len(unsupported_baselines)}; "
                f"done_pairs={'; '.join(format_baseline_pair(row) for row in done_baselines[:4])}"
            ),
            "claim_boundary": "do not claim uniformly stronger than all prior pruning methods",
        },
        {
            "gate": "end-to-end efficiency scope",
            "problem_md_link": "R7",
            "current_status": req_status(requirements, "R7"),
            "paper_status": "measured prefill/TTFT and memory gains with detector-cost caveats",
            "remaining_blocker": "Long-output and online-detector settings shrink or erase some single-sample gains.",
            "next_action": "State batch-prefill, TTFT, detector-free, detector-inclusive, and decode-length results separately.",
            "key_numbers": (
                f"Qwen20 TTFT {qwen_target20_eff.get('TTFT_speedup_no_detector','')}x, "
                f"batch-prefill {qwen_target20_eff.get('batch_prefill_speedup','')}x; "
                f"LLaVA protected detector-inclusive {llava_protected_eff.get('TTFT_speedup_with_detector','')}x; "
                f"InternVL soft detector-inclusive {internvl_soft_eff.get('TTFT_speedup_with_detector','')}x"
            ),
            "claim_boundary": "do not present 4.32x batch-prefill as general end-to-end speedup",
        },
        {
            "gate": "statistics and calibration protocol",
            "problem_md_link": "R8",
            "current_status": req_status(requirements, "R8"),
            "paper_status": "mostly addressed with sampled hard-negative human validation",
            "remaining_blocker": "The 100-row human audit is sampled rather than exhaustive, and some auxiliary random rows are not multi-seed.",
            "next_action": "Keep image-cluster/statistical and InternVL threshold protocols explicit; report the 100/500 human-QC scope.",
            "key_numbers": (
                f"Qwen target30-vs-full acc_diff={qwen_target30_full_stats.get('acc_diff','')} "
                f"CI={qwen_target30_full_stats.get('acc_CI','')}; "
                f"InternVL full dev_best_threshold={internvl_full_cal.get('dev_best_threshold','')} "
                f"test_hFPR={internvl_full_cal.get('dev_best_test_hFPR','')}"
            ),
            "claim_boundary": "do not say all labels/negatives have full human validation",
        },
        {
            "gate": "hard robustness and conflict taxonomy",
            "problem_md_link": "R9",
            "current_status": req_status(requirements, "R9"),
            "paper_status": "stress taxonomy and boundary evidence",
            "remaining_blocker": "No exhaustive dense-conflict, multilingual, handwriting, rotated-text, or detector-misleading benchmark.",
            "next_action": "Use robustness slices as failure taxonomy; avoid claiming broad hard-case robustness.",
            "key_numbers": (
                f"rows={len(robustness)}; statuses={'; '.join(robustness_statuses)}"
            ),
            "claim_boundary": "do not claim exhaustive robustness",
        },
        {
            "gate": "strong-accept upgrade readiness",
            "problem_md_link": "global",
            "current_status": strong.get("strong_accept_upgrade_status", "missing"),
            "paper_status": "not ready for strong-accept upgrade claim",
            "remaining_blocker": "Full open-QA scale and manual multi-region evidence are complete; semantic effects remain model-dependent, adaptive control fails the strict contract, and full external-port parity remains limited.",
            "next_action": "Run final manuscript/submission integrity checks and preserve these limits rather than reopening stopped experiment tracks.",
            "key_numbers": (
                f"passed={first_value(strong, 'passed_gates', 'pass_gates')}; "
                f"failed={first_value(strong, 'failed_gates', 'fail_gates')}; "
                f"partial={strong.get('partial_gates','')}"
            ),
            "claim_boundary": "do not say all originally identified weaknesses are solved",
        },
    ]


def first(name: str) -> dict[str, str]:
    rows = read_csv(INPUTS[name])
    return rows[0] if rows else {}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def first_value(row: dict[str, str], *keys: str, default: str = "") -> str:
    for key in keys:
        value = row.get(key)
        if value not in {None, ""}:
            return value
    return default


def find_row(rows: list[dict[str, str]], **criteria: str) -> dict[str, str]:
    for row in rows:
        if all(row.get(key) == value for key, value in criteria.items()):
            return row
    return {}


def req_status(requirements: list[dict[str, str]], req_id: str) -> str:
    row = find_row(requirements, id=req_id)
    return row.get("status", "missing")


def gate_status(rows: list[dict[str, str]], gate: str) -> str:
    row = find_row(rows, gate=gate)
    return row.get("status", "missing")


def format_baseline_pair(row: dict[str, str]) -> str:
    backbone = row.get("backbone", "")
    baseline = row.get("baseline", "")
    return f"{backbone}/{baseline}"


def format_open_qa(row: dict[str, str]) -> str:
    if not row:
        return "missing full-open-QA row"
    return (
        f"{row.get('model')} {row.get('task')} n={row.get('n')} "
        f"{float(row.get('full_score', 0)):.4f}->{float(row.get('pruned_score', 0)):.4f} "
        f"(delta {float(row.get('paired_delta', 0)):+.4f}, "
        f"CI [{float(row.get('ci_low', 0)):+.4f},{float(row.get('ci_high', 0)):+.4f}])"
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_markdown(rows: list[dict[str, str]]) -> str:
    return "\n".join(
        [
            "# problem.md Remaining-Blocker Dashboard",
            "",
            "This dashboard summarizes which problem.md concerns support manuscript claims, which are closed negative boundaries, and which warrant only narrowly scoped follow-up.",
            "",
            table_md(
                rows,
                [
                    "gate",
                    "problem_md_link",
                    "current_status",
                    "paper_status",
                    "remaining_blocker",
                    "next_action",
                    "key_numbers",
                    "claim_boundary",
                ],
            ),
            "",
        ]
    )


def table_md(rows: list[dict[str, str]], columns: list[str]) -> str:
    out = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(clean(row.get(col, "")) for col in columns) + " |")
    return "\n".join(out)


def clean(value: Any) -> str:
    return str(value).replace("\n", " ").replace("|", "/")


if __name__ == "__main__":
    main()
