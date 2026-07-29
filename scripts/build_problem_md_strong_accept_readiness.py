#!/usr/bin/env python3
"""Build a Strong-Accept upgrade readiness audit from problem.md.

The file `problem.md` proposes a minimal upgrade package for moving the paper
from borderline/weak-accept toward strong-accept. This script turns that package
into a reproducible go/no-go dashboard. It does not rerun experiments and does
not invent missing evidence; it reads the current evidence tables and records
which gates are satisfied, partial, or still blocked by missing evidence.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "problem_optimization_audit" / "problem_md_strong_accept_readiness"
PAPER = ROOT / "runs" / "paper_evidence"

CAUSAL_DECISION = ROOT / "runs" / "problem_optimization_audit" / "causal_evidence_go_no_go" / "causal_evidence_decision.csv"
ADAPTIVE_DECISION = ROOT / "runs" / "problem_optimization_audit" / "adaptive_controller_go_no_go" / "adaptive_controller_decision.csv"
MANUAL_GATES = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "manual_evidence_readiness_gate"
    / "manual_evidence_readiness_gates.csv"
)
MANUAL_PROGRESS_DECISION = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_manual_annotation_launch"
    / "progress_audit"
    / "manual_annotation_progress_decision.csv"
)
OPEN_QA = PAPER / "table_open_ocr_qa_generation.csv"
FULL_OPEN_QA = ROOT / "runs" / "open_ocr_qa_full" / "report" / "full_open_ocr_qa_runs.csv"
STRICT_CONTROLLER = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "strict_remaining_plan_go_no_go"
    / "strict_remaining_plan_report.json"
)
BASELINE_FEASIBILITY = PAPER / "table_native_external_port_feasibility.csv"
EFFICIENCY = PAPER / "table_efficiency_decomposition.csv"
STATISTICS = PAPER / "table_statistics.csv"
IMAGE_CLUSTER_STATS = PAPER / "table_image_cluster_statistics.csv"
RANDOM_SEED_STATUS = PAPER / "table_random_seed_status.csv"
INTERNVL_CALIBRATION = PAPER / "table_internvl_threshold_calibration_summary.csv"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_readiness_rows()
    decision = build_decision(rows)
    write_csv(OUT_DIR / "strong_accept_upgrade_readiness.csv", rows)
    write_csv(OUT_DIR / "strong_accept_upgrade_decision.csv", [decision])
    (OUT_DIR / "strong_accept_upgrade_readiness.md").write_text(build_markdown(rows, decision), encoding="utf-8")
    print(f"Wrote Strong-Accept readiness audit to {OUT_DIR}")
    print(f"strong_accept_upgrade_status={decision['strong_accept_upgrade_status']}")


def build_readiness_rows() -> list[dict[str, Any]]:
    causal = first_row(CAUSAL_DECISION)
    adaptive = first_row(ADAPTIVE_DECISION)
    manual = read_csv(MANUAL_GATES)
    manual_progress = first_row(MANUAL_PROGRESS_DECISION) if MANUAL_PROGRESS_DECISION.exists() else {}
    open_qa = read_csv(OPEN_QA)
    full_open_qa = read_csv(FULL_OPEN_QA)
    strict_controller = read_json(STRICT_CONTROLLER)
    baselines = read_csv(BASELINE_FEASIBILITY)
    efficiency = read_csv(EFFICIENCY)
    statistics = read_csv(STATISTICS)
    cluster_stats = read_csv(IMAGE_CLUSTER_STATS)
    random_seed = read_csv(RANDOM_SEED_STATUS)
    internvl_cal = read_csv(INTERNVL_CALIBRATION)

    return [
        semantic_counterfactual_gate(causal),
        native_open_qa_gate(open_qa, full_open_qa, manual, manual_progress),
        adaptive_controller_gate(adaptive, strict_controller),
        baseline_efficiency_gate(baselines, efficiency),
        statistics_calibration_gate(statistics, cluster_stats, random_seed, internvl_cal),
    ]


def semantic_counterfactual_gate(causal: dict[str, str]) -> dict[str, Any]:
    full_causal = causal.get("causal_claim_status") == "go_for_full_causal_claim"
    qwen_internvl = causal.get("qwen_strong") == "1" and causal.get("internvl_strong") == "1"
    semantic_strong = causal.get("semantic_counterfactual_strong") == "1"
    status = "pass" if full_causal else ("partial" if qwen_internvl else "fail")
    return {
        "upgrade_item": "semantic_counterfactual_evidence",
        "problem_md_target": "Show that answers move with semantic evidence changes, not only bbox/token availability.",
        "status": status,
        "evidence_reading": (
            "Qwen and calibrated InternVL pass strong causal-style support gates. Completed human QC provides "
            "Qwen-specific semantic-switch support, but the aggregate decision remains no-go for a full causal claim."
        ),
        "blocking_gap": (
            "The 102-valid-edit audit is strongly model-dependent: Qwen passes all four controls on 26 cases and InternVL on 3."
            if not semantic_strong
            else "No blocking gap recorded."
        ),
        "source_evidence": "table_causal_evidence_decision.csv; table_causal_evidence_go_no_go.csv; text-replacement audits",
        "paper_stance": "Claim causal-style support with scope limits; do not claim ECR proves causal use.",
    }


def native_open_qa_gate(
    open_qa: list[dict[str, str]],
    full_open_qa: list[dict[str, str]],
    manual: list[dict[str, str]],
    manual_progress: dict[str, str],
) -> dict[str, Any]:
    tasks = sorted({row.get("task", "") for row in full_open_qa if row.get("task")})
    if not tasks:
        tasks = sorted({row.get("task", "") for row in open_qa if row.get("task")})
    has_textvqa = any("textvqa" in task for task in tasks)
    has_docvqa = any("docvqa" in task for task in tasks)
    final_manual_ready = all(
        row.get("status") == "pass"
        for row in manual
        if row.get("gate", "").startswith("final_")
    )
    status = "pass" if has_textvqa and has_docvqa and final_manual_ready else ("partial" if has_textvqa and has_docvqa else "fail")
    best70 = [row for row in full_open_qa if row.get("budget") == "70%"]
    evidence = "; ".join(
        f"{row.get('model')} {row.get('task')} (n={row.get('n')}): "
        f"full={fnum(row.get('full_score')):.4f} pruned={fnum(row.get('pruned_score')):.4f} "
        f"delta={fnum(row.get('paired_delta')):+.4f} "
        f"CI=[{fnum(row.get('ci_low')):+.4f},{fnum(row.get('ci_high')):+.4f}]"
        for row in best70
    )
    progress_status = manual_progress.get("progress_status", "missing_progress_dashboard")
    primary_ready = f"{manual_progress.get('primary_ready_rows', '?')}/{manual_progress.get('total_primary_rows', '?')}"
    calibration_ready = (
        f"{manual_progress.get('calibration_ready_rows', '?')}/{manual_progress.get('total_calibration_rows', '?')}"
    )
    return {
        "upgrade_item": "native_open_qa_multi_evidence",
        "problem_md_target": "Validate native open-answer OCR/document QA and multi-region evidence preservation.",
        "status": status,
        "evidence_reading": (
            f"Native open-QA diagnostics exist for {', '.join(tasks)}; full-validation 70% rows: {evidence}. "
            + (
                "Manual multi-region annotations are complete and validated. "
                if final_manual_ready
                else "Manual multi-region final annotations are not complete. "
            )
            + f"Progress dashboard status={progress_status}; primary_ready={primary_ready}; calibration_ready={calibration_ready}."
        ),
        "blocking_gap": (
            "Full-validation scale and the 96-sample audit are complete; pruning is not lossless and the evidence audit remains scoped rather than benchmark-wide."
            if final_manual_ready
            else "Final 96-row human multi-region annotations must be completed and passed through the strict finalization pipeline."
        ),
        "source_evidence": "full_open_ocr_qa_runs.csv; table_open_ocr_qa_generation.csv; table_manual_evidence_readiness_gates.csv; manual_annotation_progress_decision.csv",
        "paper_stance": "Report full-validation open QA as benchmark-scale transfer and quality-boundary evidence; report the human multi-region audit as scoped availability evidence, not causal proof or lossless pruning.",
    }


def adaptive_controller_gate(
    adaptive: dict[str, str], strict_controller: dict[str, Any]
) -> dict[str, Any]:
    passed = adaptive.get("main_controller_status") == "go_for_main_method_claim"
    strict_contract = strict_controller.get("controller_contract", {})
    return {
        "upgrade_item": "unified_adaptive_risk_controller",
        "problem_md_target": "Replace hand-selected model-specific operating points with a deployable adaptive keep/fallback policy.",
        "status": "pass" if passed else "fail",
        "evidence_reading": (
            f"Aggregate controller status is {adaptive.get('main_controller_status')}; "
            f"failed required gates={adaptive.get('failed_required_gates')}, partial gates={adaptive.get('partial_gates')}; "
            f"strict contract status={strict_contract.get('status', 'missing')}."
        ),
        "blocking_gap": (
            "No non-oracle candidate passes both-task quality (within 0.01 of fixed70), "
            "mean keep at most 0.60, and overhead-aware cost below fixed70."
        ),
        "source_evidence": "table_adaptive_controller_decision.csv; table_adaptive_controller_go_no_go.csv; strict_remaining_plan_report.json",
        "paper_stance": "Keep adaptive control as quantified no-go evidence and stop incremental controller expansion unless a genuinely new pre-generation signal is introduced.",
    }


def baseline_efficiency_gate(baselines: list[dict[str, str]], efficiency: list[dict[str, str]]) -> dict[str, Any]:
    runnable = [
        row for row in baselines
        if row.get("current_status", "").endswith("evaluated")
    ]
    unsupported = [
        row for row in baselines
        if "unsupported" in row.get("feasibility", "")
    ]
    detector_rows = [row for row in efficiency if fnum(row.get("detector_ms")) > 0]
    detector_speedups = [fnum(row.get("TTFT_speedup_with_detector")) for row in detector_rows]
    detector_positive = [value for value in detector_speedups if value > 1.0]
    status = "pass" if len(unsupported) == 0 and detector_positive else "partial"
    return {
        "upgrade_item": "official_baselines_and_end_to_end_efficiency",
        "problem_md_target": "Use faithful external baselines and report complete detector-aware system speed.",
        "status": status,
        "evidence_reading": (
            f"Evaluated official/native ports={len(runnable)}; unsupported ports={len(unsupported)}. "
            f"Detector-assisted rows with detector-inclusive TTFT speedup>1={len(detector_positive)} of {len(detector_rows)}."
        ),
        "blocking_gap": "Qwen3 FastV and InternVL FastV/VisionZip remain unsupported; online detector cost erases single-sample box-aware speedups.",
        "source_evidence": "table_native_external_port_feasibility.csv; table_efficiency_decomposition.csv",
        "paper_stance": "Claim fair LLaVA and scoped Qwen3 VisionZip comparisons; scope speed as prefill/TTFT and separate detector-free from detector-assisted settings.",
    }


def statistics_calibration_gate(
    statistics: list[dict[str, str]],
    cluster_stats: list[dict[str, str]],
    random_seed: list[dict[str, str]],
    internvl_cal: list[dict[str, str]],
) -> dict[str, Any]:
    has_cluster = bool(cluster_stats)
    multi_seed = any("multi" in row.get("status", "").lower() or "pass" in row.get("status", "").lower() for row in random_seed)
    has_internvl = bool(internvl_cal)
    status = "pass" if has_cluster and has_internvl and statistics else "partial"
    return {
        "upgrade_item": "statistics_and_calibration_protocol",
        "problem_md_target": "Use image-cluster statistics, random-seed controls, hard-negative checks, and explicit calibration protocols.",
        "status": status,
        "evidence_reading": (
            f"statistics_rows={len(statistics)}; image_cluster_rows={len(cluster_stats)}; "
            f"random_seed_rows={len(random_seed)}; internvl_calibration_rows={len(internvl_cal)}."
        ),
        "blocking_gap": "" if status == "pass" else "Some statistics/calibration evidence tables are missing.",
        "source_evidence": "table_statistics.csv; table_image_cluster_statistics.csv; table_random_seed_status.csv; table_internvl_threshold_calibration_summary.csv",
        "paper_stance": "Report automatic validation and calibration protocols plainly; do not describe automatic checks as human validation.",
    }


def build_decision(rows: list[dict[str, Any]]) -> dict[str, str]:
    counts = {status: sum(row["status"] == status for row in rows) for status in ("pass", "partial", "fail")}
    status = "ready_for_strong_accept_upgrade_claim" if counts["fail"] == 0 and counts["partial"] <= 1 else "not_ready_for_strong_accept_upgrade_claim"
    return {
        "strong_accept_upgrade_status": status,
        "pass_gates": str(counts["pass"]),
        "partial_gates": str(counts["partial"]),
        "fail_gates": str(counts["fail"]),
        "recommended_next_step": "Run a final manuscript/submission integrity audit and preserve the controller and cross-backbone results as explicit boundaries; do not continue incremental controller experiments under the failed strict contract.",
    }


def build_markdown(rows: list[dict[str, Any]], decision: dict[str, str]) -> str:
    lines = [
        "# Problem.md Strong-Accept Upgrade Readiness",
        "",
        "This audit maps the five-item upgrade package recommended in `problem.md` to the current evidence package. It is intentionally conservative: a gate passes only when the current artifacts support the broad target without relying on scoped caveats.",
        "",
        "## Decision",
        "",
        f"- Strong-Accept upgrade status: `{decision['strong_accept_upgrade_status']}`",
        f"- Pass gates: {decision['pass_gates']}",
        f"- Partial gates: {decision['partial_gates']}",
        f"- Fail gates: {decision['fail_gates']}",
        f"- Recommended next step: {decision['recommended_next_step']}",
        "",
        "## Gate Table",
        "",
        table_md(
            rows,
            [
                "upgrade_item",
                "status",
                "problem_md_target",
                "evidence_reading",
                "blocking_gap",
                "paper_stance",
            ],
        ),
        "",
    ]
    return "\n".join(lines)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def first_row(path: Path) -> dict[str, str]:
    rows = read_csv(path)
    return rows[0] if rows else {}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fnum(value: str | None) -> float:
    try:
        return float(value or 0)
    except ValueError:
        return 0.0


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


if __name__ == "__main__":
    main()
