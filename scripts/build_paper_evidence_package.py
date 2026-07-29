#!/usr/bin/env python
"""Build a paper-facing evidence package from cached experiment results."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "paper_evidence"
CLAIM_LANGUAGE_AUDIT_MD = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "claim_language_audit"
    / "claim_language_audit.md"
)
CLAIM_LANGUAGE_AUDIT_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "claim_language_audit"
    / "claim_language_audit.csv"
)
PROBLEM_MD_MANUSCRIPT_COVERAGE_MD = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "problem_md_manuscript_coverage"
    / "problem_md_manuscript_coverage.md"
)
PROBLEM_MD_MANUSCRIPT_COVERAGE_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "problem_md_manuscript_coverage"
    / "problem_md_manuscript_coverage.csv"
)
ADAPTIVE_CONTROLLER_GO_NO_GO_MD = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "adaptive_controller_go_no_go"
    / "adaptive_controller_go_no_go.md"
)
ADAPTIVE_CONTROLLER_GO_NO_GO_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "adaptive_controller_go_no_go"
    / "adaptive_controller_go_no_go.csv"
)
ADAPTIVE_CONTROLLER_DECISION_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "adaptive_controller_go_no_go"
    / "adaptive_controller_decision.csv"
)
CAUSAL_EVIDENCE_GO_NO_GO_MD = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "causal_evidence_go_no_go"
    / "causal_evidence_go_no_go.md"
)
CAUSAL_EVIDENCE_GO_NO_GO_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "causal_evidence_go_no_go"
    / "causal_evidence_go_no_go.csv"
)
CAUSAL_EVIDENCE_DECISION_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "causal_evidence_go_no_go"
    / "causal_evidence_decision.csv"
)
STRONG_ACCEPT_READINESS_MD = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "problem_md_strong_accept_readiness"
    / "strong_accept_upgrade_readiness.md"
)
STRONG_ACCEPT_READINESS_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "problem_md_strong_accept_readiness"
    / "strong_accept_upgrade_readiness.csv"
)
STRONG_ACCEPT_DECISION_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "problem_md_strong_accept_readiness"
    / "strong_accept_upgrade_decision.csv"
)
PROBLEM_MD_CLAIM_BOUNDARY_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "problem_md_requirement_audit"
    / "problem_md_claim_boundaries.csv"
)
PROBLEM_MD_REMAINING_BLOCKERS_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "problem_md_remaining_blockers"
    / "problem_md_remaining_blockers.csv"
)
CONFORMAL_RISK_SELECTION_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_conformal_risk_policy"
    / "conformal_risk_policy_selection.csv"
)
CONFORMAL_RISK_DECISION_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_conformal_risk_policy"
    / "conformal_risk_policy_decision.csv"
)
DOMAIN_AWARE_PORTFOLIO_ROWS_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_domain_aware_portfolio"
    / "domain_aware_portfolio_rows.csv"
)
DOMAIN_AWARE_PORTFOLIO_DECISION_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_domain_aware_portfolio"
    / "domain_aware_portfolio_decision.csv"
)

CROSS_CSV = ROOT / "runs" / "cross_model_textocr_hard" / "cross_model_summary.csv"
HARD_CSV = ROOT / "runs" / "hard_evidence" / "hard_evidence_summary.csv"
P0_CSV = ROOT / "runs" / "p0_stats" / "pairwise_stats.csv"
EFF_CSV = ROOT / "runs" / "efficiency" / "textocr_hard_real_efficiency_report.csv"
EFF_DECOMP_CSV = ROOT / "runs" / "efficiency" / "textocr_efficiency_decomposition.csv"
E2E_LENGTH_DIR = ROOT / "runs" / "problem_optimization_audit" / "end_to_end_efficiency_length"
E2E_MEASURED_DECODE_CSV = E2E_LENGTH_DIR / "measured_qwen_decode32_summary.csv"
E2E_LENGTH_KEY_CSV = E2E_LENGTH_DIR / "textocr_length_sensitivity_key_points.csv"
OCR_CSV = ROOT / "runs" / "ocrbench_generalization" / "ocrbench_generalization_summary.csv"
OPEN_ANSWER_METRICS = (
    ROOT
    / "runs"
    / "ocrbench_open_answer"
    / "qwen3_8b_open_answer_rank_target_grid0p30"
    / "metrics.json"
)
OPEN_GENERATION_METRICS = (
    (
        "Qwen Grid 30%",
        ROOT
        / "runs"
        / "ocrbench_open_answer"
        / "qwen3_8b_open_answer_generate_target_grid0p30"
        / "metrics.json",
    ),
    (
        "Qwen Grid 50%",
        ROOT
        / "runs"
        / "ocrbench_open_answer"
        / "qwen3_8b_open_answer_generate_target_grid0p50"
        / "metrics.json",
    ),
    (
        "Qwen Grid 70%",
        ROOT
        / "runs"
        / "ocrbench_open_answer"
        / "qwen3_8b_open_answer_generate_target_grid0p70"
        / "metrics.json",
    ),
)
OPEN_OCR_QA_GENERATION_METRICS = (
    (
        "TextVQA-lite Grid 30%",
        ROOT
        / "runs"
        / "open_ocr_qa"
        / "qwen3_8b_textvqa_lite_target_grid0p30_full500"
        / "metrics.json",
    ),
    (
        "TextVQA-lite Grid 50%",
        ROOT
        / "runs"
        / "open_ocr_qa"
        / "qwen3_8b_textvqa_lite_target_grid0p50_full500"
        / "metrics.json",
    ),
    (
        "TextVQA-lite Grid 70%",
        ROOT
        / "runs"
        / "open_ocr_qa"
        / "qwen3_8b_textvqa_lite_target_grid0p70_full500"
        / "metrics.json",
    ),
    (
        "DocVQA-lite Grid 30%",
        ROOT
        / "runs"
        / "open_ocr_qa"
        / "qwen3_8b_docvqa_lite_target_grid0p30_full500"
        / "metrics.json",
    ),
    (
        "DocVQA-lite Grid 50%",
        ROOT
        / "runs"
        / "open_ocr_qa"
        / "qwen3_8b_docvqa_lite_target_grid0p50_full500"
        / "metrics.json",
    ),
    (
        "DocVQA-lite Grid 70%",
        ROOT
        / "runs"
        / "open_ocr_qa"
        / "qwen3_8b_docvqa_lite_target_grid0p70_full500"
        / "metrics.json",
    ),
)
OPEN_OCR_QA_STRESS_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_stress"
    / "open_ocr_qa_stress_summary.csv"
)
OPEN_OCR_QA_STRESS_MANIFEST_SUMMARY_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_stress_manifest"
    / "open_ocr_qa_stress_manifest_summary.csv"
)
OPEN_OCR_QA_ANNOTATION_PACK_SUMMARY_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_annotation_pack"
    / "annotation_pack_summary.csv"
)
MANUAL_EVIDENCE_READINESS_GATES_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "manual_evidence_readiness_gate"
    / "manual_evidence_readiness_gates.csv"
)
MANUAL_FINAL_PACKAGE_STATUS_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_manual_final_package"
    / "manual_final_package_status.csv"
)
OPEN_OCR_QA_EVIDENCE_SOURCE_BOUNDARY_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_evidence_source_boundary"
    / "evidence_source_boundary_summary.csv"
)
OPEN_OCR_QA_EVIDENCE_PREFILL_SUMMARY_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_evidence_prefill"
    / "evidence_prefill_summary.csv"
)
OPEN_OCR_QA_BBOX_TOOL_SUMMARY_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_bbox_annotation_tool"
    / "annotation_tool_summary.csv"
)
OPEN_OCR_QA_BBOX_VALIDATION_SUMMARY_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_bbox_annotation_validation"
    / "bbox_annotation_validation_summary.csv"
)
OPEN_OCR_QA_BBOX_ECR_SUMMARY_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_bbox_ecr"
    / "bbox_ecr_summary.csv"
)
OPEN_OCR_QA_EXTERNAL_BBOX_ADAPTER_SUMMARY_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_external_bbox_annotations"
    / "external_bbox_annotation_summary.csv"
)
OPEN_OCR_QA_TEXTVQA_GT_BBOX_SUMMARY_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_textvqa_gt_bbox"
    / "external_bbox_annotation_summary.csv"
)
OPEN_OCR_QA_TEXTVQA_GT_BBOX_ECR_SUMMARY_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_textvqa_gt_bbox_ecr"
    / "bbox_ecr_summary.csv"
)
OPEN_OCR_QA_DOCVQA_OCR_BBOX_SUMMARY_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_docvqa_hxlinh_bbox_expanded"
    / "external_bbox_annotation_summary.csv"
)
OPEN_OCR_QA_DOCVQA_OCR_BBOX_ECR_SUMMARY_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_docvqa_hxlinh_bbox_expanded_ecr"
    / "bbox_ecr_summary.csv"
)
OPEN_OCR_QA_BBOX_QUALITY_SUMMARY_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_bbox_quality_audit_expanded"
    / "bbox_quality_summary.csv"
)
OPEN_OCR_QA_ADAPTIVE_BUDGET_SUMMARY_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_adaptive_budget"
    / "adaptive_budget_summary.csv"
)
OPEN_OCR_QA_ADAPTIVE_BUDGET_SELECTION_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_adaptive_budget"
    / "adaptive_budget_policy_selection.csv"
)
OPEN_OCR_QA_RISK_COVERAGE_FRONTIER_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_risk_coverage_frontier"
    / "risk_coverage_frontier_summary.csv"
)
OPEN_OCR_QA_LEARNED_RISK_POLICY_SUMMARY_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_learned_risk_policy"
    / "learned_risk_policy_summary.csv"
)
OPEN_OCR_QA_LEARNED_RISK_POLICY_SELECTION_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_learned_risk_policy"
    / "learned_risk_policy_selection.csv"
)
OPEN_OCR_QA_LEARNED_RISK_MODEL_WEIGHTS_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_learned_risk_policy"
    / "learned_risk_model_weights.csv"
)
OPEN_OCR_QA_ANSWER_STABILITY_CASCADE_SUMMARY_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_answer_stability_cascade"
    / "answer_stability_cascade_summary.csv"
)
OPEN_OCR_QA_ANSWER_STABILITY_CASCADE_SELECTION_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_answer_stability_cascade"
    / "answer_stability_cascade_selection.csv"
)
OPEN_OCR_QA_ANSWER_STABILITY_SIGNAL_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_answer_stability_signal"
    / "answer_stability_signal_summary.csv"
)
OPEN_OCR_QA_PREGEN_RISK_SIGNAL_MODEL_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_pregen_risk_signal"
    / "pregen_risk_signal_model_summary.csv"
)
OPEN_OCR_QA_PREGEN_RISK_SIGNAL_POLICY_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_pregen_risk_signal"
    / "pregen_risk_signal_policy_summary.csv"
)
OPEN_OCR_QA_DEPLOYMENT_CONTRACT_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_deployment_contract"
    / "deployment_contract_rows.csv"
)
OPEN_OCR_QA_DEPLOYMENT_CONTRACT_SUMMARY_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_deployment_contract"
    / "deployment_contract_summary.csv"
)
OPEN_OCR_QA_UNIFIED_POLICY_TRANSFER_SUMMARY_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_unified_policy_transfer"
    / "unified_policy_transfer_summary.csv"
)
OPEN_OCR_QA_UNIFIED_POLICY_CROSS_TASK_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_unified_policy_transfer"
    / "unified_policy_cross_task_rows.csv"
)
OPEN_OCR_QA_UNIFIED_POLICY_POOLED_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_unified_policy_transfer"
    / "unified_policy_pooled_rows.csv"
)
OPEN_OCR_QA_DETECTOR_AWARE_POLICY_SUMMARY_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_detector_aware_policy"
    / "detector_aware_policy_summary.csv"
)
OPEN_OCR_QA_DETECTOR_AWARE_POLICY_READOUT_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_detector_aware_policy"
    / "detector_aware_policy_readout.csv"
)
OPEN_OCR_QA_REPAIRABILITY_SUMMARY_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_repairability"
    / "repairability_summary.csv"
)
OPEN_OCR_QA_REPAIRABILITY_FEATURE_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_repairability"
    / "repairability_feature_summary.csv"
)
OPEN_OCR_QA_BOX_AWARE_BUDGET_SUMMARY_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_box_aware_budget"
    / "box_aware_budget_summary.csv"
)
OPEN_OCR_QA_BOX_AWARE_BUDGET_SELECTION_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_box_aware_budget"
    / "box_aware_budget_selection.csv"
)
OPEN_OCR_QA_ECR_QUALITY_BUCKET_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_ecr_quality_association"
    / "ecr_quality_bucket_summary.csv"
)
OPEN_OCR_QA_ECR_QUALITY_CORRELATION_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_ecr_quality_association"
    / "ecr_quality_correlation_summary.csv"
)
OPEN_OCR_QA_DOCVQA_LINE_CONTEXT_SUMMARY_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_docvqa_hxlinh_line_context_bbox"
    / "line_context_summary.csv"
)
OPEN_OCR_QA_DOCVQA_LINE_CONTEXT_ECR_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_docvqa_hxlinh_line_context_bbox_ecr"
    / "bbox_ecr_summary.csv"
)
OPEN_OCR_QA_BBOX_NOISE_SUMMARY_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_bbox_noise_audit"
    / "bbox_noise_summary.csv"
)
HARD_ROBUSTNESS_SUMMARY_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "hard_robustness_conflict"
    / "hard_robustness_conflict_summary.csv"
)
HARD_ROBUSTNESS_TEXTOCR_METHOD_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "hard_robustness_conflict"
    / "hard_robustness_textocr_method_slice.csv"
)
HARD_ROBUSTNESS_OPENQA_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "hard_robustness_conflict"
    / "hard_robustness_openqa_slice.csv"
)
HARD_ROBUSTNESS_DETECTOR_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "hard_robustness_conflict"
    / "hard_robustness_detector_slice.csv"
)
OPEN_OCR_QA_NOISY_BOX_FALLBACK_KEY_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_noisy_box_fallback"
    / "noisy_box_fallback_key_summary.csv"
)
OPEN_OCR_QA_NOISY_BOX_LATENCY_KEY_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_noisy_box_latency"
    / "noisy_box_latency_key_summary.csv"
)
DOCVQA_LINE_CONTEXT_QUALITY_BUCKET_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "docvqa_line_context_quality_association"
    / "line_context_quality_bucket_summary.csv"
)
DOCVQA_LINE_CONTEXT_QUALITY_CORRELATION_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "docvqa_line_context_quality_association"
    / "line_context_quality_correlation_summary.csv"
)
DOCVQA_DOCUMENT_RISK_BY_BUDGET_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "docvqa_document_evidence_risk_decomposition"
    / "docvqa_document_risk_by_budget.csv"
)
DOCVQA_DOCUMENT_RISK_BY_CONDITION_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "docvqa_document_evidence_risk_decomposition"
    / "docvqa_document_risk_by_condition.csv"
)
DOCVQA_DOCUMENT_RISK_TRAJECTORY_SUMMARY_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "docvqa_document_evidence_risk_decomposition"
    / "docvqa_document_risk_trajectory_summary.csv"
)
METHOD_OBJECTIVE_MAPPING_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "method_principle_audit"
    / "method_objective_mapping.csv"
)
COVERAGE_GREEDY_TRADEOFF_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "method_principle_audit"
    / "coverage_greedy_tradeoff.csv"
)
METHOD_COVERAGE_PAIRED_SUMMARY_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "method_coverage_paired_audit"
    / "method_coverage_paired_summary.csv"
)
METHOD_COVERAGE_PAIRED_GROUP_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "method_coverage_paired_audit"
    / "method_coverage_paired_group_summary.csv"
)
METHOD_COMPONENT_PARETO_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "method_component_pareto"
    / "method_component_pareto_rows.csv"
)
METHOD_COMPONENT_DELTA_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "method_component_pareto"
    / "method_component_pareto_delta_vs_target.csv"
)
METHOD_COMPONENT_SUMMARY_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "method_component_pareto"
    / "method_component_pareto_summary.csv"
)
EXTERNAL_BASELINE_FAIRNESS_MATRIX_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "external_baseline_fairness"
    / "baseline_fairness_matrix.csv"
)
EXTERNAL_BASELINE_BUDGET_CURVE_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "external_baseline_fairness"
    / "baseline_budget_curve_summary.csv"
)
EXTERNAL_BASELINE_PAIRED_READOUT_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "external_baseline_fairness"
    / "baseline_pairwise_readout.csv"
)
QWEN3_VISIONZIP_NATIVE_PORT_CSV = ROOT / "runs" / "qwen3_visionzip_textocr_hard" / "metrics.csv"
NATIVE_EXTERNAL_PORT_FEASIBILITY_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "native_external_port_feasibility"
    / "native_external_port_feasibility.csv"
)
BOX_ROBUSTNESS_CSV = ROOT / "runs" / "box_robustness" / "box_robustness_summary.csv"
EASYOCR_SUMMARY_JSON = ROOT / "data" / "textocr_easyocr" / "easyocr_input_summary.json"
PROBLEM_AUDIT_DIR = ROOT / "runs" / "problem_optimization_audit"
IMAGE_CLUSTER_STATS_CSV = PROBLEM_AUDIT_DIR / "image_cluster_pairwise_stats.csv"
HARD_NEGATIVE_QUALITY_CSV = PROBLEM_AUDIT_DIR / "hard_negative_quality_summary.csv"
HARD_NEGATIVE_LEXICAL_SUMMARY_CSV = (
    PROBLEM_AUDIT_DIR
    / "hard_negative_lexical_audit"
    / "hard_negative_lexical_summary.csv"
)
HARD_NEGATIVE_EDIT_CLASS_CSV = (
    PROBLEM_AUDIT_DIR
    / "hard_negative_lexical_audit"
    / "hard_negative_edit_class_summary.csv"
)
HARD_NEGATIVE_SUSPICIOUS_CSV = (
    PROBLEM_AUDIT_DIR
    / "hard_negative_lexical_audit"
    / "hard_negative_suspicious_examples.csv"
)
HARD_NEGATIVE_HUMAN_QC_LAUNCH_CSV = (
    PROBLEM_AUDIT_DIR
    / "hard_negative_human_qc_launch"
    / "hard_negative_human_qc_launch_summary.csv"
)
HARD_NEGATIVE_HUMAN_QC_PROGRESS_CSV = (
    PROBLEM_AUDIT_DIR
    / "hard_negative_human_qc_progress"
    / "hard_negative_human_qc_progress_decision.csv"
)
HUMAN_QC_CLAIM_GATE_ROWS_CSV = (
    PROBLEM_AUDIT_DIR
    / "human_qc_claim_gate"
    / "human_qc_claim_gate_rows.csv"
)
HUMAN_QC_CLAIM_GATE_DECISION_CSV = (
    PROBLEM_AUDIT_DIR
    / "human_qc_claim_gate"
    / "human_qc_claim_gate_decision.csv"
)
RANDOM_SEED_STATUS_CSV = PROBLEM_AUDIT_DIR / "random_seed_status.csv"
INTERNVL_OPERATING_POINTS_CSV = PROBLEM_AUDIT_DIR / "internvl_operating_point_notes.csv"
INTERNVL_CALIBRATION_SUMMARY_CSV = (
    PROBLEM_AUDIT_DIR
    / "internvl_calibration_audit"
    / "internvl_threshold_calibration_summary.csv"
)
INTERNVL_SOFT_OPERATING_AUDIT_CSV = (
    PROBLEM_AUDIT_DIR
    / "internvl_calibration_audit"
    / "internvl_soft_operating_point_audit.csv"
)
INTERNVL_SOFT_PAIR_IDENTITY_CSV = (
    PROBLEM_AUDIT_DIR
    / "internvl_calibration_audit"
    / "internvl_soft_pair_identity_check.csv"
)
DELETION_RESTORATION_CSV = (
    ROOT
    / "runs"
    / "textocr_deletion_restoration"
    / "qwen_target30_runs"
    / "deletion_restoration_summary.csv"
)
INTERNVL_DELETION_RESTORATION_CSV = (
    ROOT
    / "runs"
    / "textocr_deletion_restoration"
    / "internvl_soft50_runs_cal_hfprconstr"
    / "deletion_restoration_summary.csv"
)
CAUSAL_EVIDENCE_TRIAD_CSV = (
    PROBLEM_AUDIT_DIR
    / "causal_evidence_triad"
    / "causal_evidence_triad_summary.csv"
)
TEXT_REPLACEMENT_HQ49_SUMMARY_CSV = (
    PROBLEM_AUDIT_DIR
    / "text_replacement_counterfactual_joined"
    / "qwen_hq49_summary.csv"
)
TEXT_REPLACEMENT_PILOT100_SUMMARY_CSV = (
    PROBLEM_AUDIT_DIR
    / "text_replacement_counterfactual_joined"
    / "qwen_pilot100_summary.csv"
)
TEXT_REPLACEMENT_OCR_HQ49_SUMMARY_CSV = (
    PROBLEM_AUDIT_DIR
    / "text_replacement_ocr_quality"
    / "qwen_hq49_summary.csv"
)
TEXT_REPLACEMENT_OCR_PILOT100_SUMMARY_CSV = (
    PROBLEM_AUDIT_DIR
    / "text_replacement_ocr_quality"
    / "qwen_pilot100_summary.csv"
)
TEXT_REPLACEMENT_STRATIFIED_HQ49_CSV = (
    PROBLEM_AUDIT_DIR
    / "text_replacement_stratified"
    / "qwen_hq49_summary.csv"
)
TEXT_REPLACEMENT_STRATIFIED_PILOT100_CSV = (
    PROBLEM_AUDIT_DIR
    / "text_replacement_stratified"
    / "qwen_pilot100_summary.csv"
)
TEXT_REPLACEMENT_HUMAN_QC_LAUNCH_CSV = (
    PROBLEM_AUDIT_DIR
    / "text_replacement_control_pack_v3"
    / "human_qc_launch"
    / "text_replacement_human_qc_launch_summary.csv"
)
TEXT_REPLACEMENT_HUMAN_QC_PROGRESS_CSV = (
    PROBLEM_AUDIT_DIR
    / "text_replacement_human_qc_progress"
    / "text_replacement_human_qc_progress_decision.csv"
)
ADAPTIVE_POLICY_DIR = ROOT / "runs" / "textocr_adaptive_policy" / "qwen_target_risk_v1"
ADAPTIVE_POLICY_CONFIG = ADAPTIVE_POLICY_DIR / "policy_config.json"
ADAPTIVE_POLICY_FIXED_CSV = ADAPTIVE_POLICY_DIR / "fixed_budget_summary.csv"
ADAPTIVE_POLICY_CANDIDATES_CSV = ADAPTIVE_POLICY_DIR / "policy_candidates.csv"
MULTISIGNAL_ADAPTIVE_POLICY_DIR = ROOT / "runs" / "textocr_adaptive_policy" / "qwen_multisignal_risk_v2"
MULTISIGNAL_ADAPTIVE_BEST_CSV = MULTISIGNAL_ADAPTIVE_POLICY_DIR / "best_by_mode.csv"
MULTISIGNAL_ADAPTIVE_CANDIDATES_CSV = MULTISIGNAL_ADAPTIVE_POLICY_DIR / "policy_candidates.csv"
CALIBRATED_RISK_POLICY_DIR = ROOT / "runs" / "textocr_adaptive_policy" / "qwen_calibrated_risk_v1"
CALIBRATED_RISK_BEST_CSV = CALIBRATED_RISK_POLICY_DIR / "best_by_target.csv"
OCCLUSION_RUNS = (
    (
        "Qwen3-VL-8B",
        ROOT
        / "runs"
        / "bbox_occlusion_qwen_textocr_hard_200"
        / "qwen3_8b_direct_802816"
        / "occlusion_report"
        / "qwen3_8b_textocr_hard200_bbox_occlusion_summary.csv",
        ROOT
        / "runs"
        / "bbox_occlusion_qwen_textocr_hard_200"
        / "qwen3_8b_direct_802816"
        / "occlusion_report"
        / "qwen3_8b_textocr_hard200_bbox_occlusion_pairwise.csv",
    ),
    (
        "LLaVA-1.5-7B",
        ROOT
        / "runs"
        / "bbox_occlusion_cross_model_textocr_hard_100"
        / "llava15_7b_direct"
        / "occlusion_report"
        / "llava15_7b_textocr_hard100_bbox_occlusion_summary.csv",
        ROOT
        / "runs"
        / "bbox_occlusion_cross_model_textocr_hard_100"
        / "llava15_7b_direct"
        / "occlusion_report"
        / "llava15_7b_textocr_hard100_bbox_occlusion_pairwise.csv",
    ),
    (
        "InternVL3.5-8B calibrated",
        ROOT
        / "runs"
        / "bbox_occlusion_cross_model_textocr_hard_100"
        / "internvl35_8b_direct_calibrated"
        / "occlusion_report"
        / "internvl35_8b_textocr_hard100_bbox_occlusion_calibrated_summary.csv",
        ROOT
        / "runs"
        / "bbox_occlusion_cross_model_textocr_hard_100"
        / "internvl35_8b_direct_calibrated"
        / "occlusion_report"
        / "internvl35_8b_textocr_hard100_bbox_occlusion_calibrated_pairwise.csv",
    ),
)
REGION_RUNS = (
    (
        "Qwen3-VL-8B",
        ROOT / "runs" / "region_logit_drop" / "qwen_region_logit_drop_summary.csv",
        ROOT / "runs" / "region_logit_drop" / "qwen_region_logit_drop_pairwise.csv",
    ),
    (
        "LLaVA-1.5-7B",
        ROOT / "runs" / "region_logit_drop_llava" / "llava_region_logit_drop_summary.csv",
        ROOT / "runs" / "region_logit_drop_llava" / "llava_region_logit_drop_pairwise.csv",
    ),
    (
        "InternVL3.5-8B calibrated-test",
        ROOT / "runs" / "region_logit_drop_internvl_calibrated" / "internvl_calibrated_region_logit_drop_summary.csv",
        ROOT / "runs" / "region_logit_drop_internvl_calibrated" / "internvl_calibrated_region_logit_drop_pairwise.csv",
    ),
)
FUTURE_STATUS = ROOT / "runs" / "future_experiments" / "future_experiment_status.md"


def main() -> None:
    cross = read_csv(CROSS_CSV)
    hard = read_csv(HARD_CSV)
    stats = read_csv(P0_CSV)
    efficiency = read_csv(EFF_CSV)
    efficiency_decomposition = read_csv(EFF_DECOMP_CSV)
    e2e_measured_decode = read_csv(E2E_MEASURED_DECODE_CSV)
    e2e_length_key = read_csv(E2E_LENGTH_KEY_CSV)
    ocr = read_csv(OCR_CSV)
    open_answer = read_json(OPEN_ANSWER_METRICS)
    open_generation = [(label, read_json(path)) for label, path in OPEN_GENERATION_METRICS]
    open_ocr_qa_generation = [(label, read_json(path)) for label, path in OPEN_OCR_QA_GENERATION_METRICS]
    open_ocr_qa_stress = read_csv(OPEN_OCR_QA_STRESS_CSV)
    open_ocr_qa_stress_manifest = read_csv(OPEN_OCR_QA_STRESS_MANIFEST_SUMMARY_CSV)
    open_ocr_qa_annotation_pack = read_csv(OPEN_OCR_QA_ANNOTATION_PACK_SUMMARY_CSV)
    manual_evidence_readiness = read_csv(MANUAL_EVIDENCE_READINESS_GATES_CSV)
    manual_final_package_status = read_csv(MANUAL_FINAL_PACKAGE_STATUS_CSV)
    open_ocr_qa_evidence_source_boundary = read_csv(OPEN_OCR_QA_EVIDENCE_SOURCE_BOUNDARY_CSV)
    open_ocr_qa_evidence_prefill = read_csv(OPEN_OCR_QA_EVIDENCE_PREFILL_SUMMARY_CSV)
    open_ocr_qa_bbox_tool = read_csv(OPEN_OCR_QA_BBOX_TOOL_SUMMARY_CSV)
    open_ocr_qa_bbox_validation = read_csv(OPEN_OCR_QA_BBOX_VALIDATION_SUMMARY_CSV)
    open_ocr_qa_bbox_ecr = read_csv(OPEN_OCR_QA_BBOX_ECR_SUMMARY_CSV)
    open_ocr_qa_external_bbox_adapter = read_csv(OPEN_OCR_QA_EXTERNAL_BBOX_ADAPTER_SUMMARY_CSV)
    open_ocr_qa_textvqa_gt_bbox = read_csv(OPEN_OCR_QA_TEXTVQA_GT_BBOX_SUMMARY_CSV)
    open_ocr_qa_textvqa_gt_bbox_ecr = read_csv(OPEN_OCR_QA_TEXTVQA_GT_BBOX_ECR_SUMMARY_CSV)
    open_ocr_qa_docvqa_ocr_bbox = read_csv(OPEN_OCR_QA_DOCVQA_OCR_BBOX_SUMMARY_CSV)
    open_ocr_qa_docvqa_ocr_bbox_ecr = read_csv(OPEN_OCR_QA_DOCVQA_OCR_BBOX_ECR_SUMMARY_CSV)
    open_ocr_qa_bbox_quality = read_csv(OPEN_OCR_QA_BBOX_QUALITY_SUMMARY_CSV)
    open_ocr_qa_adaptive_budget = read_csv(OPEN_OCR_QA_ADAPTIVE_BUDGET_SUMMARY_CSV)
    open_ocr_qa_adaptive_budget_selection = read_csv(OPEN_OCR_QA_ADAPTIVE_BUDGET_SELECTION_CSV)
    open_ocr_qa_risk_coverage_frontier = read_csv(OPEN_OCR_QA_RISK_COVERAGE_FRONTIER_CSV)
    open_ocr_qa_learned_risk_policy = read_csv(OPEN_OCR_QA_LEARNED_RISK_POLICY_SUMMARY_CSV)
    open_ocr_qa_learned_risk_selection = read_csv(OPEN_OCR_QA_LEARNED_RISK_POLICY_SELECTION_CSV)
    open_ocr_qa_learned_risk_model_weights = read_csv(OPEN_OCR_QA_LEARNED_RISK_MODEL_WEIGHTS_CSV)
    open_ocr_qa_answer_stability_cascade = read_csv(OPEN_OCR_QA_ANSWER_STABILITY_CASCADE_SUMMARY_CSV)
    open_ocr_qa_answer_stability_selection = read_csv(OPEN_OCR_QA_ANSWER_STABILITY_CASCADE_SELECTION_CSV)
    open_ocr_qa_answer_stability_signal = read_csv(OPEN_OCR_QA_ANSWER_STABILITY_SIGNAL_CSV)
    open_ocr_qa_pregen_risk_signal_model = read_csv(OPEN_OCR_QA_PREGEN_RISK_SIGNAL_MODEL_CSV)
    open_ocr_qa_pregen_risk_signal_policy = read_csv(OPEN_OCR_QA_PREGEN_RISK_SIGNAL_POLICY_CSV)
    open_ocr_qa_deployment_contract = read_csv(OPEN_OCR_QA_DEPLOYMENT_CONTRACT_CSV)
    open_ocr_qa_deployment_contract_summary = read_csv(OPEN_OCR_QA_DEPLOYMENT_CONTRACT_SUMMARY_CSV)
    open_ocr_qa_unified_policy_transfer_summary = read_csv(
        OPEN_OCR_QA_UNIFIED_POLICY_TRANSFER_SUMMARY_CSV
    )
    open_ocr_qa_unified_policy_cross_task = read_csv(OPEN_OCR_QA_UNIFIED_POLICY_CROSS_TASK_CSV)
    open_ocr_qa_unified_policy_pooled = read_csv(OPEN_OCR_QA_UNIFIED_POLICY_POOLED_CSV)
    open_ocr_qa_detector_aware_policy_summary = read_csv(
        OPEN_OCR_QA_DETECTOR_AWARE_POLICY_SUMMARY_CSV
    )
    open_ocr_qa_detector_aware_policy_readout = read_csv(
        OPEN_OCR_QA_DETECTOR_AWARE_POLICY_READOUT_CSV
    )
    open_ocr_qa_repairability_summary = read_csv(OPEN_OCR_QA_REPAIRABILITY_SUMMARY_CSV)
    open_ocr_qa_repairability_feature = read_csv(OPEN_OCR_QA_REPAIRABILITY_FEATURE_CSV)
    open_ocr_qa_box_aware_budget = read_csv(OPEN_OCR_QA_BOX_AWARE_BUDGET_SUMMARY_CSV)
    open_ocr_qa_box_aware_budget_selection = read_csv(OPEN_OCR_QA_BOX_AWARE_BUDGET_SELECTION_CSV)
    open_ocr_qa_ecr_quality_bucket = read_csv(OPEN_OCR_QA_ECR_QUALITY_BUCKET_CSV)
    open_ocr_qa_ecr_quality_correlation = read_csv(OPEN_OCR_QA_ECR_QUALITY_CORRELATION_CSV)
    open_ocr_qa_docvqa_line_context = read_csv(OPEN_OCR_QA_DOCVQA_LINE_CONTEXT_SUMMARY_CSV)
    open_ocr_qa_docvqa_line_context_ecr = read_csv(OPEN_OCR_QA_DOCVQA_LINE_CONTEXT_ECR_CSV)
    open_ocr_qa_bbox_noise = read_csv(OPEN_OCR_QA_BBOX_NOISE_SUMMARY_CSV)
    hard_robustness_summary = read_csv(HARD_ROBUSTNESS_SUMMARY_CSV)
    hard_robustness_textocr_method = read_csv(HARD_ROBUSTNESS_TEXTOCR_METHOD_CSV)
    hard_robustness_openqa = read_csv(HARD_ROBUSTNESS_OPENQA_CSV)
    hard_robustness_detector = read_csv(HARD_ROBUSTNESS_DETECTOR_CSV)
    open_ocr_qa_noisy_box_fallback_key = read_csv(OPEN_OCR_QA_NOISY_BOX_FALLBACK_KEY_CSV)
    open_ocr_qa_noisy_box_latency_key = read_csv(OPEN_OCR_QA_NOISY_BOX_LATENCY_KEY_CSV)
    docvqa_line_context_quality_bucket = read_csv(DOCVQA_LINE_CONTEXT_QUALITY_BUCKET_CSV)
    docvqa_line_context_quality_correlation = read_csv(DOCVQA_LINE_CONTEXT_QUALITY_CORRELATION_CSV)
    docvqa_document_risk_by_budget = read_csv(DOCVQA_DOCUMENT_RISK_BY_BUDGET_CSV)
    docvqa_document_risk_by_condition = read_csv(DOCVQA_DOCUMENT_RISK_BY_CONDITION_CSV)
    docvqa_document_risk_trajectory_summary = read_csv(DOCVQA_DOCUMENT_RISK_TRAJECTORY_SUMMARY_CSV)
    method_objective_mapping = read_csv(METHOD_OBJECTIVE_MAPPING_CSV)
    coverage_greedy_tradeoff = read_csv(COVERAGE_GREEDY_TRADEOFF_CSV)
    method_coverage_paired_summary = read_csv(METHOD_COVERAGE_PAIRED_SUMMARY_CSV)
    method_coverage_paired_group = read_csv(METHOD_COVERAGE_PAIRED_GROUP_CSV)
    method_component_pareto = read_csv(METHOD_COMPONENT_PARETO_CSV)
    method_component_delta = read_csv(METHOD_COMPONENT_DELTA_CSV)
    method_component_summary = read_csv(METHOD_COMPONENT_SUMMARY_CSV)
    external_baseline_fairness = read_csv(EXTERNAL_BASELINE_FAIRNESS_MATRIX_CSV)
    external_baseline_budget_curve = read_csv(EXTERNAL_BASELINE_BUDGET_CURVE_CSV)
    external_baseline_paired = read_csv(EXTERNAL_BASELINE_PAIRED_READOUT_CSV)
    qwen3_visionzip_native_port = read_csv(QWEN3_VISIONZIP_NATIVE_PORT_CSV)
    native_external_port_feasibility = read_csv(NATIVE_EXTERNAL_PORT_FEASIBILITY_CSV)
    box_robustness = read_csv(BOX_ROBUSTNESS_CSV)
    easyocr_summary = read_json(EASYOCR_SUMMARY_JSON)
    image_cluster_stats = read_csv(IMAGE_CLUSTER_STATS_CSV)
    hard_negative_quality = read_csv(HARD_NEGATIVE_QUALITY_CSV)
    hard_negative_lexical = read_csv(HARD_NEGATIVE_LEXICAL_SUMMARY_CSV)
    hard_negative_edit_class = read_csv(HARD_NEGATIVE_EDIT_CLASS_CSV)
    hard_negative_suspicious = read_csv(HARD_NEGATIVE_SUSPICIOUS_CSV)
    hard_negative_human_qc_launch = read_csv(HARD_NEGATIVE_HUMAN_QC_LAUNCH_CSV)
    hard_negative_human_qc_progress = read_csv(HARD_NEGATIVE_HUMAN_QC_PROGRESS_CSV)
    human_qc_claim_gate_rows = read_csv(HUMAN_QC_CLAIM_GATE_ROWS_CSV)
    human_qc_claim_gate_decision = read_csv(HUMAN_QC_CLAIM_GATE_DECISION_CSV)
    random_seed_status = read_csv(RANDOM_SEED_STATUS_CSV)
    internvl_operating_points = read_csv(INTERNVL_OPERATING_POINTS_CSV)
    internvl_calibration_summary = read_csv(INTERNVL_CALIBRATION_SUMMARY_CSV)
    internvl_soft_operating_audit = read_csv(INTERNVL_SOFT_OPERATING_AUDIT_CSV)
    internvl_soft_pair_identity = read_csv(INTERNVL_SOFT_PAIR_IDENTITY_CSV)
    deletion_restoration = read_csv(DELETION_RESTORATION_CSV)
    internvl_deletion_restoration = read_csv(INTERNVL_DELETION_RESTORATION_CSV)
    causal_evidence_triad = read_csv(CAUSAL_EVIDENCE_TRIAD_CSV)
    text_replacement_counterfactual = build_text_replacement_counterfactual_table(
        read_csv(TEXT_REPLACEMENT_HQ49_SUMMARY_CSV),
        read_csv(TEXT_REPLACEMENT_PILOT100_SUMMARY_CSV),
    )
    text_replacement_ocr_quality = build_text_replacement_ocr_quality_table(
        read_csv(TEXT_REPLACEMENT_OCR_HQ49_SUMMARY_CSV),
        read_csv(TEXT_REPLACEMENT_OCR_PILOT100_SUMMARY_CSV),
    )
    text_replacement_stratified = build_text_replacement_stratified_table(
        read_csv(TEXT_REPLACEMENT_STRATIFIED_HQ49_CSV),
        read_csv(TEXT_REPLACEMENT_STRATIFIED_PILOT100_CSV),
    )
    text_replacement_human_qc_launch = read_csv(TEXT_REPLACEMENT_HUMAN_QC_LAUNCH_CSV)
    text_replacement_human_qc_progress = read_csv(TEXT_REPLACEMENT_HUMAN_QC_PROGRESS_CSV)
    adaptive_policy_config = read_json(ADAPTIVE_POLICY_CONFIG)
    adaptive_policy_fixed = read_csv(ADAPTIVE_POLICY_FIXED_CSV)
    adaptive_policy_candidates = read_csv(ADAPTIVE_POLICY_CANDIDATES_CSV)
    multisignal_adaptive_best = read_csv(MULTISIGNAL_ADAPTIVE_BEST_CSV)
    calibrated_risk_best = read_csv(CALIBRATED_RISK_BEST_CSV)
    adaptive_controller_go_no_go = read_csv(ADAPTIVE_CONTROLLER_GO_NO_GO_CSV)
    adaptive_controller_decision = read_csv(ADAPTIVE_CONTROLLER_DECISION_CSV)
    causal_evidence_go_no_go = read_csv(CAUSAL_EVIDENCE_GO_NO_GO_CSV)
    causal_evidence_decision = read_csv(CAUSAL_EVIDENCE_DECISION_CSV)
    strong_accept_readiness = read_csv(STRONG_ACCEPT_READINESS_CSV)
    strong_accept_decision = read_csv(STRONG_ACCEPT_DECISION_CSV)
    problem_md_claim_boundaries = read_csv(PROBLEM_MD_CLAIM_BOUNDARY_CSV)
    problem_md_remaining_blockers = read_csv(PROBLEM_MD_REMAINING_BLOCKERS_CSV)
    conformal_risk_selection = read_csv(CONFORMAL_RISK_SELECTION_CSV)
    conformal_risk_decision = read_csv(CONFORMAL_RISK_DECISION_CSV)
    domain_aware_portfolio_rows = read_csv(DOMAIN_AWARE_PORTFOLIO_ROWS_CSV)
    domain_aware_portfolio_decision = read_csv(DOMAIN_AWARE_PORTFOLIO_DECISION_CSV)
    occlusion_rows: list[dict[str, Any]] = []
    for model, summary_path, pairwise_path in OCCLUSION_RUNS:
        occlusion_rows.extend(
            build_bbox_occlusion_table(
                read_csv(summary_path),
                read_csv(pairwise_path),
                model=model,
            )
        )
    region_rows: list[dict[str, Any]] = []
    for model, summary_path, pairwise_path in REGION_RUNS:
        region_rows.extend(
            build_region_logit_drop_table(
                read_csv(summary_path),
                read_csv(pairwise_path),
                model=model,
            )
        )

    tables = {
        "main_textocr": build_main_textocr_table(cross),
        "evidence_baselines": build_evidence_baseline_table(hard),
        "efficiency": build_efficiency_table(efficiency),
        "efficiency_decomposition": build_efficiency_decomposition_table(efficiency_decomposition),
        "e2e_measured_decode": e2e_measured_decode,
        "e2e_length_key": e2e_length_key,
        "ocrbench": build_ocrbench_table(ocr),
        "open_answer": build_open_answer_table(open_answer),
        "open_generation": build_open_generation_table(open_generation),
        "open_ocr_qa_generation": build_open_ocr_qa_generation_table(open_ocr_qa_generation),
        "open_ocr_qa_stress": open_ocr_qa_stress,
        "open_ocr_qa_stress_manifest": open_ocr_qa_stress_manifest,
        "open_ocr_qa_annotation_pack": open_ocr_qa_annotation_pack,
        "manual_evidence_readiness": manual_evidence_readiness,
        "manual_final_package_status": manual_final_package_status,
        "open_ocr_qa_evidence_source_boundary": open_ocr_qa_evidence_source_boundary,
        "open_ocr_qa_evidence_prefill": open_ocr_qa_evidence_prefill,
        "open_ocr_qa_bbox_tool": open_ocr_qa_bbox_tool,
        "open_ocr_qa_bbox_validation": open_ocr_qa_bbox_validation,
        "open_ocr_qa_bbox_ecr": open_ocr_qa_bbox_ecr,
        "open_ocr_qa_external_bbox_adapter": open_ocr_qa_external_bbox_adapter,
        "open_ocr_qa_textvqa_gt_bbox": open_ocr_qa_textvqa_gt_bbox,
        "open_ocr_qa_textvqa_gt_bbox_ecr": open_ocr_qa_textvqa_gt_bbox_ecr,
        "open_ocr_qa_docvqa_ocr_bbox": open_ocr_qa_docvqa_ocr_bbox,
        "open_ocr_qa_docvqa_ocr_bbox_ecr": open_ocr_qa_docvqa_ocr_bbox_ecr,
        "open_ocr_qa_bbox_quality": open_ocr_qa_bbox_quality,
        "open_ocr_qa_adaptive_budget": open_ocr_qa_adaptive_budget,
        "open_ocr_qa_adaptive_budget_selection": open_ocr_qa_adaptive_budget_selection,
        "open_ocr_qa_risk_coverage_frontier": open_ocr_qa_risk_coverage_frontier,
        "open_ocr_qa_learned_risk_policy": open_ocr_qa_learned_risk_policy,
        "open_ocr_qa_learned_risk_selection": open_ocr_qa_learned_risk_selection,
        "open_ocr_qa_learned_risk_model_weights": open_ocr_qa_learned_risk_model_weights,
        "open_ocr_qa_answer_stability_cascade": open_ocr_qa_answer_stability_cascade,
        "open_ocr_qa_answer_stability_selection": open_ocr_qa_answer_stability_selection,
        "open_ocr_qa_answer_stability_signal": open_ocr_qa_answer_stability_signal,
        "open_ocr_qa_pregen_risk_signal_model": open_ocr_qa_pregen_risk_signal_model,
        "open_ocr_qa_pregen_risk_signal_policy": open_ocr_qa_pregen_risk_signal_policy,
        "open_ocr_qa_deployment_contract": open_ocr_qa_deployment_contract,
        "open_ocr_qa_deployment_contract_summary": open_ocr_qa_deployment_contract_summary,
        "open_ocr_qa_unified_policy_transfer_summary": open_ocr_qa_unified_policy_transfer_summary,
        "open_ocr_qa_unified_policy_cross_task": open_ocr_qa_unified_policy_cross_task,
        "open_ocr_qa_unified_policy_pooled": open_ocr_qa_unified_policy_pooled,
        "open_ocr_qa_detector_aware_policy_summary": open_ocr_qa_detector_aware_policy_summary,
        "open_ocr_qa_detector_aware_policy_readout": open_ocr_qa_detector_aware_policy_readout,
        "open_ocr_qa_repairability_summary": open_ocr_qa_repairability_summary,
        "open_ocr_qa_repairability_feature": open_ocr_qa_repairability_feature,
        "open_ocr_qa_box_aware_budget": open_ocr_qa_box_aware_budget,
        "open_ocr_qa_box_aware_budget_selection": open_ocr_qa_box_aware_budget_selection,
        "open_ocr_qa_ecr_quality_bucket": open_ocr_qa_ecr_quality_bucket,
        "open_ocr_qa_ecr_quality_correlation": open_ocr_qa_ecr_quality_correlation,
        "open_ocr_qa_docvqa_line_context": open_ocr_qa_docvqa_line_context,
        "open_ocr_qa_docvqa_line_context_ecr": open_ocr_qa_docvqa_line_context_ecr,
        "open_ocr_qa_bbox_noise": open_ocr_qa_bbox_noise,
        "hard_robustness_summary": hard_robustness_summary,
        "hard_robustness_textocr_method": hard_robustness_textocr_method,
        "hard_robustness_openqa": hard_robustness_openqa,
        "hard_robustness_detector": hard_robustness_detector,
        "open_ocr_qa_noisy_box_fallback_key": open_ocr_qa_noisy_box_fallback_key,
        "open_ocr_qa_noisy_box_latency_key": open_ocr_qa_noisy_box_latency_key,
        "docvqa_line_context_quality_bucket": docvqa_line_context_quality_bucket,
        "docvqa_line_context_quality_correlation": docvqa_line_context_quality_correlation,
        "docvqa_document_risk_by_budget": docvqa_document_risk_by_budget,
        "docvqa_document_risk_by_condition": docvqa_document_risk_by_condition,
        "docvqa_document_risk_trajectory_summary": docvqa_document_risk_trajectory_summary,
        "method_objective_mapping": method_objective_mapping,
        "coverage_greedy_tradeoff": coverage_greedy_tradeoff,
        "method_coverage_paired_summary": method_coverage_paired_summary,
        "method_coverage_paired_group": method_coverage_paired_group,
        "method_component_pareto": method_component_pareto,
        "method_component_delta": method_component_delta,
        "method_component_summary": method_component_summary,
        "external_baseline_fairness": external_baseline_fairness,
        "external_baseline_budget_curve": external_baseline_budget_curve,
        "external_baseline_paired": external_baseline_paired,
        "qwen3_visionzip_native_port": qwen3_visionzip_native_port,
        "native_external_port_feasibility": native_external_port_feasibility,
        "box_source_robustness": build_box_source_robustness_table(box_robustness, easyocr_summary),
        "statistics": build_statistics_table(stats),
        "region_logit_drop": region_rows,
        "bbox_occlusion": occlusion_rows,
        "image_cluster_statistics": image_cluster_stats,
        "hard_negative_quality": hard_negative_quality,
        "hard_negative_lexical": hard_negative_lexical,
        "hard_negative_edit_class": hard_negative_edit_class,
        "hard_negative_suspicious": hard_negative_suspicious,
        "hard_negative_human_qc_launch": hard_negative_human_qc_launch,
        "hard_negative_human_qc_progress": hard_negative_human_qc_progress,
        "human_qc_claim_gate_rows": human_qc_claim_gate_rows,
        "human_qc_claim_gate_decision": human_qc_claim_gate_decision,
        "random_seed_status": random_seed_status,
        "internvl_operating_points": internvl_operating_points,
        "internvl_calibration_summary": internvl_calibration_summary,
        "internvl_soft_operating_audit": internvl_soft_operating_audit,
        "internvl_soft_pair_identity": internvl_soft_pair_identity,
        "deletion_restoration": build_deletion_restoration_table(deletion_restoration),
        "internvl_deletion_restoration": build_deletion_restoration_table(internvl_deletion_restoration),
        "causal_evidence_triad": causal_evidence_triad,
        "text_replacement_counterfactual": text_replacement_counterfactual,
        "text_replacement_ocr_quality": text_replacement_ocr_quality,
        "text_replacement_stratified": text_replacement_stratified,
        "text_replacement_human_qc_launch": text_replacement_human_qc_launch,
        "text_replacement_human_qc_progress": text_replacement_human_qc_progress,
        "adaptive_policy": build_adaptive_policy_table(
            adaptive_policy_config,
            adaptive_policy_fixed,
            adaptive_policy_candidates,
            multisignal_adaptive_best,
            calibrated_risk_best,
        ),
        "adaptive_controller_go_no_go": adaptive_controller_go_no_go,
        "adaptive_controller_decision": adaptive_controller_decision,
        "causal_evidence_go_no_go": causal_evidence_go_no_go,
        "causal_evidence_decision": causal_evidence_decision,
        "strong_accept_readiness": strong_accept_readiness,
        "strong_accept_decision": strong_accept_decision,
        "problem_md_claim_boundaries": problem_md_claim_boundaries,
        "problem_md_remaining_blockers": problem_md_remaining_blockers,
        "conformal_risk_selection": conformal_risk_selection,
        "conformal_risk_decision": conformal_risk_decision,
        "domain_aware_portfolio_rows": domain_aware_portfolio_rows,
        "domain_aware_portfolio_decision": domain_aware_portfolio_decision,
        "unsupported": build_unsupported_table(),
    }
    claim_ledger = build_claim_ledger(tables, easyocr_summary)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "claim_ledger.csv", claim_ledger)
    write_csv(OUT_DIR / "table_main_textocr.csv", tables["main_textocr"])
    write_csv(OUT_DIR / "table_evidence_baselines.csv", tables["evidence_baselines"])
    write_csv(OUT_DIR / "table_efficiency.csv", tables["efficiency"])
    write_csv(OUT_DIR / "table_efficiency_decomposition.csv", tables["efficiency_decomposition"])
    write_csv(OUT_DIR / "table_e2e_measured_decode32.csv", tables["e2e_measured_decode"])
    write_csv(OUT_DIR / "table_e2e_length_sensitivity_key_points.csv", tables["e2e_length_key"])
    write_csv(OUT_DIR / "table_ocrbench_generalization.csv", tables["ocrbench"])
    write_csv(OUT_DIR / "table_ocrbench_open_answer_ranking.csv", tables["open_answer"])
    write_csv(OUT_DIR / "table_ocrbench_open_answer_generation.csv", tables["open_generation"])
    write_csv(OUT_DIR / "table_open_ocr_qa_generation.csv", tables["open_ocr_qa_generation"])
    write_csv(OUT_DIR / "table_open_ocr_qa_stress.csv", tables["open_ocr_qa_stress"])
    write_csv(OUT_DIR / "table_open_ocr_qa_stress_manifest_summary.csv", tables["open_ocr_qa_stress_manifest"])
    write_csv(OUT_DIR / "table_open_ocr_qa_annotation_pack_summary.csv", tables["open_ocr_qa_annotation_pack"])
    write_csv(OUT_DIR / "table_manual_evidence_readiness_gates.csv", tables["manual_evidence_readiness"])
    write_csv(OUT_DIR / "table_manual_final_package_status.csv", tables["manual_final_package_status"])
    write_csv(OUT_DIR / "table_open_ocr_qa_evidence_source_boundary.csv", tables["open_ocr_qa_evidence_source_boundary"])
    write_csv(OUT_DIR / "table_open_ocr_qa_evidence_prefill_summary.csv", tables["open_ocr_qa_evidence_prefill"])
    write_csv(OUT_DIR / "table_open_ocr_qa_bbox_tool_summary.csv", tables["open_ocr_qa_bbox_tool"])
    write_csv(OUT_DIR / "table_open_ocr_qa_bbox_validation_summary.csv", tables["open_ocr_qa_bbox_validation"])
    write_csv(OUT_DIR / "table_open_ocr_qa_bbox_ecr_summary.csv", tables["open_ocr_qa_bbox_ecr"])
    write_csv(OUT_DIR / "table_open_ocr_qa_external_bbox_adapter_summary.csv", tables["open_ocr_qa_external_bbox_adapter"])
    write_csv(OUT_DIR / "table_open_ocr_qa_textvqa_gt_bbox_summary.csv", tables["open_ocr_qa_textvqa_gt_bbox"])
    write_csv(OUT_DIR / "table_open_ocr_qa_textvqa_gt_bbox_ecr_summary.csv", tables["open_ocr_qa_textvqa_gt_bbox_ecr"])
    write_csv(OUT_DIR / "table_open_ocr_qa_docvqa_ocr_bbox_summary.csv", tables["open_ocr_qa_docvqa_ocr_bbox"])
    write_csv(OUT_DIR / "table_open_ocr_qa_docvqa_ocr_bbox_ecr_summary.csv", tables["open_ocr_qa_docvqa_ocr_bbox_ecr"])
    write_csv(OUT_DIR / "table_open_ocr_qa_bbox_quality_summary.csv", tables["open_ocr_qa_bbox_quality"])
    write_csv(OUT_DIR / "table_open_ocr_qa_adaptive_budget_summary.csv", tables["open_ocr_qa_adaptive_budget"])
    write_csv(OUT_DIR / "table_open_ocr_qa_adaptive_budget_selection.csv", tables["open_ocr_qa_adaptive_budget_selection"])
    write_csv(OUT_DIR / "table_open_ocr_qa_risk_coverage_frontier.csv", tables["open_ocr_qa_risk_coverage_frontier"])
    write_csv(OUT_DIR / "table_open_ocr_qa_learned_risk_policy.csv", tables["open_ocr_qa_learned_risk_policy"])
    write_csv(OUT_DIR / "table_open_ocr_qa_learned_risk_selection.csv", tables["open_ocr_qa_learned_risk_selection"])
    write_csv(OUT_DIR / "table_open_ocr_qa_learned_risk_model_weights.csv", tables["open_ocr_qa_learned_risk_model_weights"])
    write_csv(OUT_DIR / "table_open_ocr_qa_answer_stability_cascade.csv", tables["open_ocr_qa_answer_stability_cascade"])
    write_csv(OUT_DIR / "table_open_ocr_qa_answer_stability_selection.csv", tables["open_ocr_qa_answer_stability_selection"])
    write_csv(OUT_DIR / "table_open_ocr_qa_answer_stability_signal.csv", tables["open_ocr_qa_answer_stability_signal"])
    write_csv(OUT_DIR / "table_open_ocr_qa_pregen_risk_signal_model.csv", tables["open_ocr_qa_pregen_risk_signal_model"])
    write_csv(OUT_DIR / "table_open_ocr_qa_pregen_risk_signal_policy.csv", tables["open_ocr_qa_pregen_risk_signal_policy"])
    write_csv(OUT_DIR / "table_open_ocr_qa_deployment_contract.csv", tables["open_ocr_qa_deployment_contract"])
    write_csv(
        OUT_DIR / "table_open_ocr_qa_deployment_contract_summary.csv",
        tables["open_ocr_qa_deployment_contract_summary"],
    )
    write_csv(
        OUT_DIR / "table_open_ocr_qa_unified_policy_transfer_summary.csv",
        tables["open_ocr_qa_unified_policy_transfer_summary"],
    )
    write_csv(
        OUT_DIR / "table_open_ocr_qa_unified_policy_cross_task.csv",
        tables["open_ocr_qa_unified_policy_cross_task"],
    )
    write_csv(
        OUT_DIR / "table_open_ocr_qa_unified_policy_pooled.csv",
        tables["open_ocr_qa_unified_policy_pooled"],
    )
    write_csv(
        OUT_DIR / "table_open_ocr_qa_detector_aware_policy_summary.csv",
        tables["open_ocr_qa_detector_aware_policy_summary"],
    )
    write_csv(
        OUT_DIR / "table_open_ocr_qa_detector_aware_policy_readout.csv",
        tables["open_ocr_qa_detector_aware_policy_readout"],
    )
    write_csv(OUT_DIR / "table_open_ocr_qa_repairability_summary.csv", tables["open_ocr_qa_repairability_summary"])
    write_csv(OUT_DIR / "table_open_ocr_qa_repairability_feature_summary.csv", tables["open_ocr_qa_repairability_feature"])
    write_csv(OUT_DIR / "table_open_ocr_qa_box_aware_budget_summary.csv", tables["open_ocr_qa_box_aware_budget"])
    write_csv(OUT_DIR / "table_open_ocr_qa_box_aware_budget_selection.csv", tables["open_ocr_qa_box_aware_budget_selection"])
    write_csv(OUT_DIR / "table_open_ocr_qa_ecr_quality_bucket_summary.csv", tables["open_ocr_qa_ecr_quality_bucket"])
    write_csv(OUT_DIR / "table_open_ocr_qa_ecr_quality_correlation_summary.csv", tables["open_ocr_qa_ecr_quality_correlation"])
    write_csv(OUT_DIR / "table_open_ocr_qa_docvqa_line_context_summary.csv", tables["open_ocr_qa_docvqa_line_context"])
    write_csv(OUT_DIR / "table_open_ocr_qa_docvqa_line_context_ecr_summary.csv", tables["open_ocr_qa_docvqa_line_context_ecr"])
    write_csv(OUT_DIR / "table_open_ocr_qa_bbox_noise_summary.csv", tables["open_ocr_qa_bbox_noise"])
    write_csv(OUT_DIR / "table_hard_robustness_conflict_summary.csv", tables["hard_robustness_summary"])
    write_csv(OUT_DIR / "table_hard_robustness_textocr_method_slice.csv", tables["hard_robustness_textocr_method"])
    write_csv(OUT_DIR / "table_hard_robustness_openqa_slice.csv", tables["hard_robustness_openqa"])
    write_csv(OUT_DIR / "table_hard_robustness_detector_slice.csv", tables["hard_robustness_detector"])
    write_csv(OUT_DIR / "table_open_ocr_qa_noisy_box_fallback_key_summary.csv", tables["open_ocr_qa_noisy_box_fallback_key"])
    write_csv(OUT_DIR / "table_open_ocr_qa_noisy_box_latency_key_summary.csv", tables["open_ocr_qa_noisy_box_latency_key"])
    write_csv(OUT_DIR / "table_docvqa_line_context_quality_bucket_summary.csv", tables["docvqa_line_context_quality_bucket"])
    write_csv(OUT_DIR / "table_docvqa_line_context_quality_correlation_summary.csv", tables["docvqa_line_context_quality_correlation"])
    write_csv(OUT_DIR / "table_docvqa_document_risk_by_budget.csv", tables["docvqa_document_risk_by_budget"])
    write_csv(OUT_DIR / "table_docvqa_document_risk_by_condition.csv", tables["docvqa_document_risk_by_condition"])
    write_csv(OUT_DIR / "table_docvqa_document_risk_trajectory_summary.csv", tables["docvqa_document_risk_trajectory_summary"])
    write_csv(OUT_DIR / "table_method_objective_mapping.csv", tables["method_objective_mapping"])
    write_csv(OUT_DIR / "table_coverage_greedy_tradeoff.csv", tables["coverage_greedy_tradeoff"])
    write_csv(OUT_DIR / "table_method_coverage_paired_summary.csv", tables["method_coverage_paired_summary"])
    write_csv(OUT_DIR / "table_method_coverage_paired_group_summary.csv", tables["method_coverage_paired_group"])
    write_csv(OUT_DIR / "table_method_component_pareto.csv", tables["method_component_pareto"])
    write_csv(OUT_DIR / "table_method_component_delta_vs_target.csv", tables["method_component_delta"])
    write_csv(OUT_DIR / "table_method_component_pareto_summary.csv", tables["method_component_summary"])
    write_csv(OUT_DIR / "table_external_baseline_fairness_matrix.csv", tables["external_baseline_fairness"])
    write_csv(OUT_DIR / "table_external_baseline_budget_curve.csv", tables["external_baseline_budget_curve"])
    write_csv(OUT_DIR / "table_external_baseline_paired_readout.csv", tables["external_baseline_paired"])
    write_csv(OUT_DIR / "table_qwen3_visionzip_native_port.csv", tables["qwen3_visionzip_native_port"])
    write_csv(OUT_DIR / "table_native_external_port_feasibility.csv", tables["native_external_port_feasibility"])
    write_csv(OUT_DIR / "table_box_source_robustness.csv", tables["box_source_robustness"])
    write_csv(OUT_DIR / "table_statistics.csv", tables["statistics"])
    write_csv(OUT_DIR / "table_region_logit_drop.csv", tables["region_logit_drop"])
    write_csv(OUT_DIR / "table_bbox_occlusion.csv", tables["bbox_occlusion"])
    write_csv(OUT_DIR / "table_image_cluster_statistics.csv", tables["image_cluster_statistics"])
    write_csv(OUT_DIR / "table_hard_negative_quality.csv", tables["hard_negative_quality"])
    write_csv(OUT_DIR / "table_hard_negative_lexical_summary.csv", tables["hard_negative_lexical"])
    write_csv(OUT_DIR / "table_hard_negative_edit_class_summary.csv", tables["hard_negative_edit_class"])
    write_csv(OUT_DIR / "table_hard_negative_suspicious_examples.csv", tables["hard_negative_suspicious"])
    write_csv(OUT_DIR / "table_hard_negative_human_qc_launch_summary.csv", tables["hard_negative_human_qc_launch"])
    write_csv(OUT_DIR / "table_hard_negative_human_qc_progress_decision.csv", tables["hard_negative_human_qc_progress"])
    write_csv(OUT_DIR / "table_human_qc_claim_gate_rows.csv", tables["human_qc_claim_gate_rows"])
    write_csv(OUT_DIR / "table_human_qc_claim_gate_decision.csv", tables["human_qc_claim_gate_decision"])
    write_csv(OUT_DIR / "table_random_seed_status.csv", tables["random_seed_status"])
    write_csv(OUT_DIR / "table_internvl_operating_points.csv", tables["internvl_operating_points"])
    write_csv(OUT_DIR / "table_internvl_threshold_calibration_summary.csv", tables["internvl_calibration_summary"])
    write_csv(OUT_DIR / "table_internvl_soft_operating_audit.csv", tables["internvl_soft_operating_audit"])
    write_csv(OUT_DIR / "table_internvl_soft_pair_identity_check.csv", tables["internvl_soft_pair_identity"])
    write_csv(OUT_DIR / "table_deletion_restoration.csv", tables["deletion_restoration"])
    write_csv(OUT_DIR / "table_internvl_deletion_restoration.csv", tables["internvl_deletion_restoration"])
    write_csv(OUT_DIR / "table_causal_evidence_triad.csv", tables["causal_evidence_triad"])
    write_csv(OUT_DIR / "table_text_replacement_counterfactual.csv", tables["text_replacement_counterfactual"])
    write_csv(OUT_DIR / "table_text_replacement_ocr_quality.csv", tables["text_replacement_ocr_quality"])
    write_csv(OUT_DIR / "table_text_replacement_stratified.csv", tables["text_replacement_stratified"])
    write_csv(
        OUT_DIR / "table_text_replacement_human_qc_launch_summary.csv",
        tables["text_replacement_human_qc_launch"],
    )
    write_csv(
        OUT_DIR / "table_text_replacement_human_qc_progress_decision.csv",
        tables["text_replacement_human_qc_progress"],
    )
    write_csv(OUT_DIR / "table_adaptive_policy.csv", tables["adaptive_policy"])
    write_csv(OUT_DIR / "table_adaptive_controller_go_no_go.csv", tables["adaptive_controller_go_no_go"])
    write_csv(OUT_DIR / "table_adaptive_controller_decision.csv", tables["adaptive_controller_decision"])
    write_csv(OUT_DIR / "table_causal_evidence_go_no_go.csv", tables["causal_evidence_go_no_go"])
    write_csv(OUT_DIR / "table_causal_evidence_decision.csv", tables["causal_evidence_decision"])
    write_csv(OUT_DIR / "table_problem_md_strong_accept_readiness.csv", tables["strong_accept_readiness"])
    write_csv(OUT_DIR / "table_problem_md_strong_accept_decision.csv", tables["strong_accept_decision"])
    write_csv(OUT_DIR / "table_problem_md_claim_boundaries.csv", tables["problem_md_claim_boundaries"])
    write_csv(OUT_DIR / "table_problem_md_remaining_blockers.csv", tables["problem_md_remaining_blockers"])
    write_csv(OUT_DIR / "table_open_ocr_qa_conformal_risk_selection.csv", tables["conformal_risk_selection"])
    write_csv(OUT_DIR / "table_open_ocr_qa_conformal_risk_decision.csv", tables["conformal_risk_decision"])
    write_csv(OUT_DIR / "table_open_ocr_qa_domain_aware_portfolio_rows.csv", tables["domain_aware_portfolio_rows"])
    write_csv(
        OUT_DIR / "table_open_ocr_qa_domain_aware_portfolio_decision.csv",
        tables["domain_aware_portfolio_decision"],
    )
    write_csv(OUT_DIR / "unsupported_or_negative_claims.csv", tables["unsupported"])
    (OUT_DIR / "paper_evidence_package.md").write_text(markdown(claim_ledger, tables), encoding="utf-8")
    print(f"Wrote {OUT_DIR / 'paper_evidence_package.md'}")
    print(f"Wrote CSV tables to {OUT_DIR}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_main_textocr_table(cross: list[dict[str, str]]) -> list[dict[str, Any]]:
    wanted = [
        ("Qwen3-VL-8B", "full", "Full baseline"),
        ("Qwen3-VL-8B", "target0p20", "Ours, efficiency point"),
        ("Qwen3-VL-8B", "target0p30", "Ours, quality point"),
        ("Qwen3-VL-8B", "grid0p50", "Spatial baseline"),
        ("Qwen3-VL-8B", "random0p50", "Random baseline"),
        ("LLaVA-1.5-7B", "full", "Full baseline"),
        ("LLaVA-1.5-7B", "embed0p40", "Accuracy-efficiency point"),
        ("LLaVA-1.5-7B", "protected_embed0p40", "Evidence-protected point"),
        ("LLaVA-1.5-7B", "grid0p40", "Low-hallucination spatial point"),
        ("InternVL3.5-8B calibrated-test", "full_cal", "Calibrated full baseline"),
        ("InternVL3.5-8B calibrated-test", "target0p50_cal", "Low-hFPR point"),
        ("InternVL3.5-8B calibrated-test", "target_soft_evidence0p50_hfpr_cal", "Risk-constrained point"),
        ("InternVL3.5-8B calibrated-test", "grid0p50_cal", "Spatial baseline"),
    ]
    by_key = {(row["model"], row["run"]): row for row in cross}
    out = []
    for model, run, note in wanted:
        row = by_key[(model, run)]
        out.append(
            {
                "model": model,
                "method": run,
                "note": note,
                "n": row["n"],
                "acc": f(row["acc"]),
                "hFPR": f(row["hFPR"]),
                "keep_ratio": f(row["keep_ratio"]),
                "ECR": f(row["ECR"]),
                "CenterR": f(row["CenterR"]),
                "PatchR": f(row["PatchR"]),
                "source": row["path"],
            }
        )
    return out


def build_evidence_baseline_table(hard: list[dict[str, str]]) -> list[dict[str, Any]]:
    wanted_groups = {
        "same_budget_qwen_0p20",
        "same_budget_qwen_0p30",
        "same_budget_llava_0p40",
        "same_budget_internvl_0p50",
        "external_method_llava_0p40",
        "external_proxy_qwen",
        "external_proxy_internvl",
        "causal_qwen_0p30",
        "causal_internvl_0p50",
    }
    out = []
    for row in hard:
        if row["group"] not in wanted_groups:
            continue
        if row["status"] != "done":
            continue
        out.append(
            {
                "family": row["family"],
                "group": row["group"],
                "model": row["model"],
                "method": row["label"],
                "role": row["role"],
                "n": row["n"],
                "acc": f(row["acc"]),
                "hFPR": f(row["hFPR"]),
                "keep_ratio": f(row["keep_ratio"]),
                "ECR": f(row["ECR"]),
                "CenterR": f(row["CenterR"]),
                "PatchR": f(row["PatchR"]),
                "delta_acc": f(row["acc_delta_vs_baseline"]),
                "delta_hFPR": f(row["hFPR_delta_vs_baseline"]),
                "delta_ECR": f(row["ECR_delta_vs_baseline"]),
                "source": row["path"],
                "note": row["note"],
            }
        )
    return out


def build_efficiency_table(efficiency: list[dict[str, str]]) -> list[dict[str, Any]]:
    wanted = {
        ("Qwen3-VL-8B", "target0p20"),
        ("Qwen3-VL-8B", "target0p30"),
        ("LLaVA-1.5-7B", "embed0p40"),
        ("LLaVA-1.5-7B", "protected_embed0p40"),
        ("LLaVA-1.5-7B", "grid0p40"),
        ("InternVL3.5-8B", "target0p50"),
        ("InternVL3.5-8B", "soft_evidence0p50_hfpr"),
        ("InternVL3.5-8B", "grid0p50"),
    }
    out = []
    for row in efficiency:
        if (row["model"], row["point"]) not in wanted:
            continue
        out.append(
            {
                "model": row["model"],
                "point": row["point"],
                "acc": f(row["quality_acc"]),
                "hFPR": f(row["quality_hFPR"]),
                "keep_ratio": f(row["keep_ratio"]),
                "single_forward_speedup": f(row["single_forward_speedup"]),
                "batch_prefill_speedup": f(row["batch_prefill_speedup"]),
                "full_samples_per_s": f(row["full_samples_per_s"]),
                "pruned_samples_per_s": f(row["pruned_samples_per_s"]),
                "incremental_peak_reduction_pct": f(row["incremental_peak_allocated_reduction_pct"]),
                "batch_overhead_pct_saved_prefill": f(row["batch_prune_overhead_pct_saved_prefill"]),
                "source": row["batch_summary_path"],
            }
        )
    return out


def build_efficiency_decomposition_table(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        out.append(
            {
                "point": row["label"],
                "acc": row["quality_acc"],
                "hFPR": row["quality_hFPR"],
                "keep_ratio": row["keep_ratio"],
                "full_TTFT_ms": row["full_ttft_ms"],
                "pruned_TTFT_ms": row["pruned_ttft_ms"],
                "TTFT_speedup_no_detector": row["ttft_speedup_no_detector"],
                "detector_ms": row["detector_ms"],
                "detector_inclusive_TTFT_ms": row["detector_inclusive_ttft_ms"],
                "TTFT_speedup_with_detector": row["ttft_speedup_with_detector"],
                "batch_prefill_speedup": row["batch_prefill_speedup"],
                "incremental_peak_reduction_pct": row["incremental_peak_allocated_reduction_pct"],
                "note": row["note"],
            }
        )
    return out


def build_deletion_restoration_table(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if row.get("status") != "done":
            continue
        out.append(
            {
                "variant": row["variant"],
                "n": row["n"],
                "acc": f(row["acc"]),
                "hFPR": f(row["hFPR"]),
                "keep_ratio": f(row["keep_ratio"]),
                "ECR": f(row["ECR"]),
                "delta_acc_vs_removed": f(row.get("delta_acc_vs_removed", "")),
                "acc_recovery_pct": f"{100.0 * float(row.get('acc_recovery_frac', 0.0) or 0.0):.1f}",
            }
        )
    return out


def build_text_replacement_counterfactual_table(
    hq49_rows: list[dict[str, str]],
    pilot100_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for split, rows in [("HQ filtered", hq49_rows), ("Pilot", pilot100_rows)]:
        if not rows:
            continue
        row = rows[0]
        out.append(
            {
                "split": split,
                "n_pairs": row.get("n_pairs", ""),
                "original_pair_correct_rate": f(row.get("original_pair_correct_rate")),
                "edited_pair_correct_rate": f(row.get("edited_pair_correct_rate")),
                "full_four_way_semantic_switch_rate": f(row.get("full_four_way_semantic_switch_rate")),
                "edited_pair_correct_given_original_correct": f(
                    row.get("edited_pair_correct_given_original_correct")
                ),
                "source_absence_switch_rate": f(row.get("source_absence_switch_rate")),
                "replacement_presence_switch_rate": f(row.get("replacement_presence_switch_rate")),
                "mean_source_yes_support_drop": f(row.get("mean_source_yes_support_drop")),
                "mean_replacement_yes_support_gain": f(row.get("mean_replacement_yes_support_gain")),
                "mean_box_width_px": f(row.get("mean_box_width_px")),
                "mean_box_height_px": f(row.get("mean_box_height_px")),
                "note": "semantic text replacement diagnostic; not photorealistic and not a universal causal proof",
            }
        )
    return out


def build_text_replacement_ocr_quality_table(
    hq49_rows: list[dict[str, str]],
    pilot100_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for split, rows in [("HQ filtered", hq49_rows), ("Pilot", pilot100_rows)]:
        if not rows:
            continue
        row = rows[0]
        out.append(
            {
                "split": split,
                "n_pairs": row.get("n_pairs", ""),
                "source_original_detected_crop_rate": f(row.get("source_original_detected_crop_rate")),
                "edited_crop_replacement_detected_rate": f(row.get("edited_crop_replacement_detected_rate")),
                "edited_crop_source_absent_rate": f(row.get("edited_crop_source_absent_rate")),
                "edited_crop_ocr_success_rate": f(row.get("edited_crop_ocr_success_rate")),
                "edited_full_replacement_detected_rate": f(row.get("edited_full_replacement_detected_rate")),
                "edited_full_source_absent_rate": f(row.get("edited_full_source_absent_rate")),
                "edited_full_ocr_success_rate": f(row.get("edited_full_ocr_success_rate")),
                "mean_box_width_px": f(row.get("mean_box_width_px")),
                "mean_box_height_px": f(row.get("mean_box_height_px")),
                "mean_font_size": f(row.get("mean_font_size")),
                "note": "EasyOCR readability audit for renderer quality; not human verification",
            }
        )
    return out


def build_text_replacement_stratified_table(
    hq49_rows: list[dict[str, str]],
    pilot100_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    wanted_groups = [
        "all",
        "edited_crop_ocr_success",
        "edited_crop_ocr_failure",
    ]
    group_notes = {
        "all": "all joined before/after pairs",
        "edited_crop_ocr_success": "inserted replacement is detected and source is absent in edited crop",
        "edited_crop_ocr_failure": "EasyOCR readability check fails in edited crop",
    }
    for split, rows in [("HQ filtered", hq49_rows), ("Pilot", pilot100_rows)]:
        by_group = {row.get("group", ""): row for row in rows}
        for group in wanted_groups:
            row = by_group.get(group)
            if not row:
                continue
            out.append(
                {
                    "split": split,
                    "group": group,
                    "n_pairs": row.get("n_pairs", ""),
                    "full_four_way_semantic_switch_rate": f(
                        row.get("full_four_way_semantic_switch_rate")
                    ),
                    "edited_pair_correct_given_original_correct": f(
                        row.get("edited_pair_correct_given_original_correct")
                    ),
                    "source_absence_switch_rate": f(row.get("source_absence_switch_rate")),
                    "replacement_presence_switch_rate": f(
                        row.get("replacement_presence_switch_rate")
                    ),
                    "mean_source_yes_support_drop": f(row.get("mean_source_yes_support_drop")),
                    "mean_replacement_yes_support_gain": f(
                        row.get("mean_replacement_yes_support_gain")
                    ),
                    "ocr_success_rate": f(row.get("ocr_success_rate")),
                    "note": group_notes[group],
                }
            )
    return out


def build_adaptive_policy_table(
    config: dict[str, Any],
    fixed_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    multisignal_rows: list[dict[str, str]],
    calibrated_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    out = []
    wanted_fixed = {"Fixed 0.20", "Fixed 0.25", "Fixed 0.30", "Fixed 0.35", "Fixed 1.00"}
    for row in fixed_rows:
        if row["policy"] not in wanted_fixed:
            continue
        out.append(
            {
                "selection": "fixed",
                "policy": row["policy"],
                "feature_source": "",
                "dev_acc": f(row["dev_accuracy"]),
                "dev_hFPR": f(row["dev_hFPR"]),
                "dev_keep": f(row["dev_mean_keep_ratio"]),
                "test_acc": f(row["test_accuracy"]),
                "test_hFPR": f(row["test_hFPR"]),
                "test_keep": f(row["test_mean_keep_ratio"]),
                "test_cascade": f(row["test_cascade_keep_ratio"]),
                "test_ECR": f(row["test_mean_ecr"]),
                "note": "fixed budget baseline",
            }
        )
    out.append(
        {
            "selection": "dev-selected",
            "policy": config["policy"],
            "feature_source": config["feature_source"],
            "dev_acc": f(config["dev_accuracy"]),
            "dev_hFPR": f(config["dev_hFPR"]),
            "dev_keep": f(config["dev_mean_keep_ratio"]),
            "test_acc": f(config["test_accuracy"]),
            "test_hFPR": f(config["test_hFPR"]),
            "test_keep": f(config["test_mean_keep_ratio"]),
            "test_cascade": f(config["test_cascade_keep_ratio"]),
            "test_ECR": f(config["test_mean_ecr"]),
            "note": "legitimate split-safe adaptive result",
        }
    )
    if candidate_rows:
        best_test = max(candidate_rows, key=lambda row: float(row["test_accuracy"]))
        out.append(
            {
                "selection": "posthoc-test-best",
                "policy": best_test["policy"],
                "feature_source": best_test["feature_source"],
                "dev_acc": f(best_test["dev_accuracy"]),
                "dev_hFPR": f(best_test["dev_hFPR"]),
                "dev_keep": f(best_test["dev_mean_keep_ratio"]),
                "test_acc": f(best_test["test_accuracy"]),
                "test_hFPR": f(best_test["test_hFPR"]),
                "test_keep": f(best_test["test_mean_keep_ratio"]),
                "test_cascade": f(best_test["test_cascade_keep_ratio"]),
                "test_ECR": f(best_test["test_mean_ecr"]),
                "note": "diagnostic upper bound only; not dev-selected",
            }
        )
    for row in multisignal_rows:
        mode = row.get("mode", "")
        if mode == "deployable":
            note = "selector-side spatial/full-fallback search; lowers hFPR but still misses fixed-budget frontier"
        else:
            note = "oracle-audit upper-bound search with evidence features; diagnostic only"
        out.append(
            {
                "selection": f"multi-signal-{mode}",
                "policy": row["policy"],
                "feature_source": row.get("feature_sources", ""),
                "dev_acc": f(row["dev_accuracy"]),
                "dev_hFPR": f(row["dev_hFPR"]),
                "dev_keep": f(row["dev_mean_keep_ratio"]),
                "test_acc": f(row["test_accuracy"]),
                "test_hFPR": f(row["test_hFPR"]),
                "test_keep": f(row["test_mean_keep_ratio"]),
                "test_cascade": f(row["test_cascade_keep_ratio"]),
                "test_ECR": f(row["test_mean_ecr"]),
                "note": note,
            }
        )
    for row in calibrated_rows:
        target = row.get("target", "")
        out.append(
            {
                "selection": f"learned-risk-{target}",
                "policy": row["policy"],
                "feature_source": "deployable low-budget logistic risk predictor",
                "dev_acc": f(row["dev_accuracy"]),
                "dev_hFPR": f(row["dev_hFPR"]),
                "dev_keep": f(row["dev_mean_keep_ratio"]),
                "test_acc": f(row["test_accuracy"]),
                "test_hFPR": f(row["test_hFPR"]),
                "test_keep": f(row["test_mean_keep_ratio"]),
                "test_cascade": f(row["test_cascade_keep_ratio"]),
                "test_ECR": f(row["test_mean_ecr"]),
                "note": "learned calibrated risk predictor; reduces mean keep but does not beat fixed-budget frontier",
            }
        )
    return out


def build_ocrbench_table(ocr: list[dict[str, str]]) -> list[dict[str, Any]]:
    wanted = {
        ("Qwen3-VL-8B", "direct", "full"),
        ("Qwen3-VL-8B", "target_embed_topk", "0.3"),
        ("Qwen3-VL-8B", "target_embed_grid_topk", "0.3"),
        ("Qwen3-VL-8B", "random", "0.3"),
        ("InternVL3.5-8B", "direct", "full"),
        ("InternVL3.5-8B", "target_embed_topk", "0.5"),
        ("InternVL3.5-8B", "target_embed_grid_topk", "0.5"),
        ("InternVL3.5-8B", "target_embed_grid_topk", "0.6"),
        ("InternVL3.5-8B", "grid", "0.5"),
        ("LLaVA-1.5-7B", "direct", "full"),
        ("LLaVA-1.5-7B", "embed_protected_topk", "0.4"),
        ("LLaVA-1.5-7B", "VisionZip", "0.4"),
        ("LLaVA-1.5-7B", "FastV", "0.4"),
    }
    out = []
    for row in ocr:
        if (row["model"], row["method"], row["ratio"]) not in wanted:
            continue
        out.append(
            {
                "model": row["model"],
                "method": row["method"],
                "ratio": row["ratio"],
                "implementation": row["implementation"],
                "n": row["n"],
                "acc": f(row["acc"]),
                "hFPR": f(row["hFPR"]),
                "keep_ratio": f(row["keep_ratio"]),
                "source": row["path"],
                "note": row["note"],
            }
        )
    return out


def build_open_answer_table(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "scope": "overall",
            "type": "all",
            "n": metrics["n"],
            "full_rank_acc": f(metrics["full_rank_acc"]),
            "pruned_rank_acc": f(metrics["pruned_rank_acc"]),
            "delta_rank_acc": f(metrics["rank_acc_delta_pruned_minus_full"]),
            "full_margin": f(metrics["mean_full_margin"]),
            "pruned_margin": f(metrics["mean_pruned_margin"]),
            "delta_margin": f(metrics["mean_margin_delta_pruned_minus_full"]),
            "keep_ratio": f(metrics["mean_effective_keep_ratio"]),
            "selector_answer_tokens": f(metrics["mean_target_text_token_count"]),
            "note": "original OCRBench questions; gold-vs-decoy answer likelihood; not free-form leaderboard",
        }
    ]
    for question_type, item in sorted(metrics.get("by_type", {}).items()):
        rows.append(
            {
                "scope": "by_type",
                "type": question_type,
                "n": item["n"],
                "full_rank_acc": f(item["full_rank_acc"]),
                "pruned_rank_acc": f(item["pruned_rank_acc"]),
                "delta_rank_acc": f(float(item["pruned_rank_acc"]) - float(item["full_rank_acc"])),
                "full_margin": "",
                "pruned_margin": "",
                "delta_margin": f(item["mean_margin_delta"]),
                "keep_ratio": "",
                "selector_answer_tokens": "",
                "note": "",
            }
        )
    return rows


def build_open_generation_table(runs: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for label, metrics in runs:
        rows.append(
            {
                "method": label,
                "n": metrics["n"],
                "full_exact": f(metrics["full_exact"]),
                "pruned_exact": f(metrics["pruned_exact"]),
                "delta_exact": f(metrics["exact_delta_pruned_minus_full"]),
                "full_anls": f(metrics["full_anls"]),
                "pruned_anls": f(metrics["pruned_anls"]),
                "delta_anls": f(metrics["anls_delta_pruned_minus_full"]),
                "full_contains": f(metrics["full_contains"]),
                "pruned_contains": f(metrics["pruned_contains"]),
                "keep_ratio": f(metrics["mean_effective_keep_ratio"]),
                "selector_answer_tokens": "0.000",
                "note": "greedy free-form generation; selector uses question only",
            }
        )
    return rows


def build_open_ocr_qa_generation_table(runs: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for label, metrics in runs:
        rows.append(
            {
                "method": label,
                "task": metrics["task"],
                "n": metrics["n"],
                "primary_metric": metrics["primary_metric"],
                "full_score": f(metrics["full_score"]),
                "pruned_score": f(metrics["pruned_score"]),
                "delta_score": f(metrics["score_delta_pruned_minus_full"]),
                "full_exact": f(metrics["full_exact"]),
                "pruned_exact": f(metrics["pruned_exact"]),
                "full_anls": f(metrics["full_anls"]),
                "pruned_anls": f(metrics["pruned_anls"]),
                "keep_ratio": f(metrics["mean_effective_keep_ratio"]),
                "selector_question_tokens": f(metrics["mean_target_text_token_count"]),
                "note": "greedy free-form generation; selector uses question only; full 500-sample lite split",
            }
        )
    return rows


def build_box_source_robustness_table(
    rows: list[dict[str, str]],
    easyocr_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    wanted_sources = {
        "GT boxes",
        "simulated detected-like boxes",
        "EasyOCR detected boxes",
        "heavy-jitter boxes",
        "missing boxes",
    }
    detector_ms = float(easyocr_summary.get("mean_detector_ms_per_image", 0.0) or 0.0)
    detector_missing = float(easyocr_summary.get("oracle_missing_rate", 0.0) or 0.0)
    detector_iou = float(easyocr_summary.get("mean_best_iou_to_oracle", 0.0) or 0.0)
    out = []
    for row in rows:
        if row["variant"] not in wanted_sources:
            continue
        note = ""
        if row["variant"] == "EasyOCR detected boxes":
            note = (
                f"real detector; {detector_ms:.1f} ms/image; "
                f"oracle-missing {detector_missing:.3f}; mean IoU {detector_iou:.3f}"
            )
        elif row["variant"] == "GT boxes":
            note = "oracle box upper-bound"
        elif row["variant"] == "missing boxes":
            note = "no detector/evidence boxes visible to selector"
        else:
            note = "synthetic detector perturbation"
        out.append(
            {
                "model": row["model"],
                "box_source": row["variant"],
                "selector": row["selector"],
                "n": row["num_samples"],
                "acc": f(row["accuracy"]),
                "hFPR": f(row["hFPR"]),
                "keep_ratio": f(row["keep_ratio"]),
                "true_ECR": f(row["true_ECR"]),
                "true_CenterR": f(row["true_CenterR"]),
                "selector_box_ECR": f(row["selector_box_ECR"]),
                "note": note,
            }
        )
    return out


def build_statistics_table(stats: list[dict[str, str]]) -> list[dict[str, Any]]:
    wanted = [
        "qwen_target0p20_vs_full",
        "qwen_target0p20_vs_random",
        "qwen_target0p20_vs_grid",
        "qwen_target0p30_vs_full",
        "qwen_target0p30_vs_fastv_proxy",
        "qwen_target0p30_vs_visionzip_proxy",
        "llava_protected0p40_vs_full",
        "llava_protected0p40_vs_visionzip_official",
        "internvl_soft0p50_vs_full_cal",
        "internvl_soft0p50_vs_grid",
        "internvl_soft0p50_vs_embed_proxy",
    ]
    by_name = {row["comparison"]: row for row in stats}
    out = []
    for name in wanted:
        row = by_name[name]
        out.append(
            {
                "comparison": name,
                "family": row["family"],
                "claim": row["claim"],
                "n": row["n_overlap"],
                "left": row["left_method"],
                "right": row["right_method"],
                "acc_diff": f(row["acc_diff"]),
                "acc_CI": interval(row["acc_diff_ci_low"], row["acc_diff_ci_high"]),
                "acc_p": p(row["acc_sign_p"]),
                "hFPR_diff": f(row["hFPR_diff"]),
                "hFPR_CI": interval(row["hFPR_diff_ci_low"], row["hFPR_diff_ci_high"]),
                "hFPR_p": p(row["hFPR_sign_p"]),
            }
        )
    return out


def build_region_logit_drop_table(
    summary: list[dict[str, str]],
    pairwise: list[dict[str, str]],
    *,
    model: str,
) -> list[dict[str, Any]]:
    summary_by_arm = {row["arm"]: row for row in summary}
    wanted_pairs = [
        "evidence_kept_vs_removed",
        "full_vs_removed",
        "ours_vs_removed",
        "ours_vs_full",
        "ours_vs_evidence_kept",
    ]
    out = []
    for row in pairwise:
        if row["comparison"] not in wanted_pairs:
            continue
        left_arm = summary_by_arm.get(row["left"], {})
        right_arm = summary_by_arm.get(row["right"], {})
        out.append(
            {
                "model": model,
                "comparison": row["comparison"],
                "n": row["n"],
                "left": row["left"],
                "right": row["right"],
                "left_acc": f(left_arm.get("acc", "")),
                "right_acc": f(right_arm.get("acc", "")),
                "left_hFPR": f(left_arm.get("hFPR", "")),
                "right_hFPR": f(right_arm.get("hFPR", "")),
                "support_diff": f(row["mean_support_diff"]),
                "support_CI": interval(row["support_diff_ci_low"], row["support_diff_ci_high"]),
                "support_p": p(row["support_diff_p_two_sided"]),
                "pos_support_diff": f(row["pos_mean_support_diff"]),
                "pos_support_CI": interval(row["pos_support_diff_ci_low"], row["pos_support_diff_ci_high"]),
                "neg_support_diff": f(row["neg_mean_support_diff"]),
                "neg_support_CI": interval(row["neg_support_diff_ci_low"], row["neg_support_diff_ci_high"]),
                "correct_flip_L_R": f"{row['left_correct_right_wrong']}/{row['left_wrong_right_correct']}",
                "hFP_flip_L_R": f"{row['left_hfp_right_not']}/{row['right_hfp_left_not']}",
            }
        )
    return out


def build_bbox_occlusion_table(
    summary: list[dict[str, str]],
    pairwise: list[dict[str, str]],
    *,
    model: str,
) -> list[dict[str, Any]]:
    summary_by_view = {row["view"]: row for row in summary}
    wanted_pairs = [
        "orig_vs_evidence",
        "random_vs_evidence",
        "orig_vs_random",
    ]
    out = []
    for row in pairwise:
        if row["comparison"] not in wanted_pairs:
            continue
        left_view = summary_by_view.get(row["left"], {})
        right_view = summary_by_view.get(row["right"], {})
        out.append(
            {
                "model": model,
                "comparison": row["comparison"],
                "n": row["n"],
                "left": row["left"],
                "right": row["right"],
                "left_acc": f(left_view.get("acc", "")),
                "right_acc": f(right_view.get("acc", "")),
                "left_hFPR": f(left_view.get("hFPR", "")),
                "right_hFPR": f(right_view.get("hFPR", "")),
                "support_diff": f(row["mean_support_diff"]),
                "support_CI": interval(row["support_diff_ci_low"], row["support_diff_ci_high"]),
                "support_p": p(row["support_diff_p_two_sided"]),
                "pos_support_diff": f(row["pos_mean_support_diff"]),
                "pos_support_CI": interval(row["pos_support_diff_ci_low"], row["pos_support_diff_ci_high"]),
                "neg_support_diff": f(row["neg_mean_support_diff"]),
                "neg_support_CI": interval(row["neg_support_diff_ci_low"], row["neg_support_diff_ci_high"]),
                "correct_flip_L_R": f"{row['left_correct_right_wrong']}/{row['left_wrong_right_correct']}",
                "hFP_flip_L_R": f"{row['left_hfp_right_not']}/{row['right_hfp_left_not']}",
            }
        )
    return out


def build_unsupported_table() -> list[dict[str, str]]:
    return [
        {
            "claim": "A universal adaptive risk-aware budget policy beats fixed budgets.",
            "status": "unsupported",
            "reason": "Single-signal, multi-signal, selector-spatial, and full-fallback split-safe searches do not beat the fixed-budget frontier on TextOCR-Hard. Native open OCR/DocQA question-side and low-answer fallback rules also remain well below fixed 70% retention despite a strong oracle budget-selection upper bound.",
            "safe_wording": "Use fixed operating points as the main results; present adaptive budgeting as a diagnostic boundary and future-work direction, not as a solved contribution.",
        },
        {
            "claim": "The method is best on every model and task.",
            "status": "unsupported",
            "reason": "LLaVA target-conditioned pruning is a negative quality result; OCR-safe grid_topk helps OCRBench but not TextOCR-Hard.",
            "safe_wording": "Report model-specific operating points and limitations.",
        },
        {
            "claim": "Official FastV/VisionZip comparisons are available for every backbone and budget.",
            "status": "unsupported",
            "reason": "LLaVA has matched-budget FastV/VisionZip official-algorithm ports, and Qwen3 VisionZip now has a scoped native port at 0.30. Qwen3 FastV and InternVL FastV/VisionZip remain unsupported without new backend hooks.",
            "safe_wording": "Report LLaVA official ports across budgets and Qwen3 VisionZip as a single-budget scoped native port; keep Qwen3 FastV and InternVL external rows proxy, unsupported, or not claimed according to the baseline fairness matrix.",
        },
        {
            "claim": "Spatial-aware pruning fixes GSR/spatial hallucination.",
            "status": "negative",
            "reason": "GSR spatial transfer is a negative or tied result in P0 statistics.",
            "safe_wording": "Use as a limitation/failure analysis.",
        },
        {
            "claim": "Evidence preservation is fully causal in the strong logit-drop/oracle sense.",
            "status": "partial",
            "reason": "We now have region ECR, evidence topk/bottomk ablations, matched token logit-drop diagnostics, cross-model input-space bbox occlusion, a strong Qwen deletion-restoration curve, a calibrated InternVL deletion-restoration curve, and a semantic text-replacement audit. The text-replacement audit shows positive average support shifts but low full switch rates, so it remains a boundary result rather than a full causal oracle.",
            "safe_wording": "Claim region-evidence preservation plus causal-style token/input-space diagnostics, with Qwen as the cleanest deletion-restoration proof, calibrated InternVL as supporting evidence, and semantic text replacement as a limitation-aware boundary audit.",
        },
    ]


def open_adaptive_row(
    tables: dict[str, list[dict[str, Any]]],
    task: str,
    split: str,
    policy: str,
) -> dict[str, Any]:
    return next(
        row
        for row in tables["open_ocr_qa_adaptive_budget"]
        if row["task"] == task and row["split"] == split and row["policy"] == policy
    )


def build_claim_ledger(tables: dict[str, list[dict[str, Any]]], easyocr_summary: dict[str, Any]) -> list[dict[str, str]]:
    easyocr_ms = float(easyocr_summary.get("mean_detector_ms_per_image", 0.0) or 0.0)
    easyocr_missing = float(easyocr_summary.get("oracle_missing_rate", 0.0) or 0.0)
    open_overall = next(row for row in tables["open_answer"] if row["scope"] == "overall")
    open_gen_70 = next(row for row in tables["open_generation"] if row["method"] == "Qwen Grid 70%")
    textvqa_30 = next(row for row in tables["open_ocr_qa_generation"] if row["method"] == "TextVQA-lite Grid 30%")
    textvqa_50 = next(row for row in tables["open_ocr_qa_generation"] if row["method"] == "TextVQA-lite Grid 50%")
    textvqa_70 = next(row for row in tables["open_ocr_qa_generation"] if row["method"] == "TextVQA-lite Grid 70%")
    docvqa_30 = next(row for row in tables["open_ocr_qa_generation"] if row["method"] == "DocVQA-lite Grid 30%")
    docvqa_50 = next(row for row in tables["open_ocr_qa_generation"] if row["method"] == "DocVQA-lite Grid 50%")
    docvqa_70 = next(row for row in tables["open_ocr_qa_generation"] if row["method"] == "DocVQA-lite Grid 70%")
    stress_textvqa_numeric30 = next(
        row
        for row in tables["open_ocr_qa_stress"]
        if row["task"] == "TextVQA-lite" and row["ratio"] == "0.30" and row["stress_tag"] == "numeric_answer"
    )
    stress_textvqa_numeric70 = next(
        row
        for row in tables["open_ocr_qa_stress"]
        if row["task"] == "TextVQA-lite" and row["ratio"] == "0.70" and row["stress_tag"] == "numeric_answer"
    )
    stress_docvqa_long30 = next(
        row
        for row in tables["open_ocr_qa_stress"]
        if row["task"] == "DocVQA-lite" and row["ratio"] == "0.30" and row["stress_tag"] == "long_question_ge10"
    )
    stress_docvqa_long70 = next(
        row
        for row in tables["open_ocr_qa_stress"]
        if row["task"] == "DocVQA-lite" and row["ratio"] == "0.70" and row["stress_tag"] == "long_question_ge10"
    )
    hard_validity = next(
        row for row in tables["hard_robustness_summary"] if row["category"] == "near-miss benchmark validity"
    )
    hard_nearmiss_risk = next(
        row for row in tables["hard_robustness_summary"] if row["category"] == "near-miss answer risk"
    )
    hard_openqa_stress = next(
        row for row in tables["hard_robustness_summary"] if row["category"] == "open-QA stress"
    )
    hard_noisy_boxes = next(
        row for row in tables["hard_robustness_summary"] if row["category"] == "noisy and missing evidence boxes"
    )
    hard_detector_backbone = next(
        row for row in tables["hard_robustness_summary"] if row["category"] == "detector-in-loop backbone robustness"
    )
    hard_openqa_detector = next(
        row for row in tables["hard_robustness_summary"] if row["category"] == "open-QA detector boxes"
    )
    adaptive_dev = next(row for row in tables["adaptive_policy"] if row["selection"] == "dev-selected")
    adaptive_fixed30 = next(row for row in tables["adaptive_policy"] if row["policy"] == "Fixed 0.30")
    adaptive_fixed35 = next(row for row in tables["adaptive_policy"] if row["policy"] == "Fixed 0.35")
    adaptive_multisignal = next(
        row for row in tables["adaptive_policy"] if row["selection"] == "multi-signal-deployable"
    )
    adaptive_learned = next(
        row for row in tables["adaptive_policy"] if row["selection"] == "learned-risk-base_error"
    )
    open_adapt_text_fixed70 = open_adaptive_row(tables, "TextVQA-lite", "test", "fixed_0.70")
    open_adapt_text_fullfallback = open_adaptive_row(
        tables, "TextVQA-lite", "test", "q_cue_full_w2_1_3"
    )
    open_adapt_text_oracle = open_adaptive_row(tables, "TextVQA-lite", "test", "oracle_best_budget")
    open_adapt_doc_fixed70 = open_adaptive_row(tables, "DocVQA-lite", "test", "fixed_0.70")
    open_adapt_doc_fullfallback = open_adaptive_row(
        tables, "DocVQA-lite", "test", "q_cue_full_w1_1_3"
    )
    open_adapt_doc_oracle = open_adaptive_row(tables, "DocVQA-lite", "test", "oracle_best_budget")
    textvqa_frontier0 = next(
        row
        for row in tables["open_ocr_qa_risk_coverage_frontier"]
        if row["task"] == "TextVQA-lite" and row["split"] == "test" and row["tolerance"] == "0.000"
    )
    docvqa_frontier0 = next(
        row
        for row in tables["open_ocr_qa_risk_coverage_frontier"]
        if row["task"] == "DocVQA-lite" and row["split"] == "test" and row["tolerance"] == "0.000"
    )
    textvqa_deploy_contract = next(
        row
        for row in tables["open_ocr_qa_deployment_contract_summary"]
        if row["task"] == "TextVQA-lite" and row["summary_item"] == "near_fixed70_lower_cost_candidate"
    )
    docvqa_deploy_contract = next(
        row
        for row in tables["open_ocr_qa_deployment_contract_summary"]
        if row["task"] == "DocVQA-lite" and row["summary_item"] == "near_fixed70_lower_cost_candidate"
    )
    domain_portfolio = tables["domain_aware_portfolio_decision"][0]
    transfer_text_to_doc = next(
        row
        for row in tables["open_ocr_qa_unified_policy_transfer_summary"]
        if row["scope"] == "TextVQA-lite -> DocVQA-lite"
    )
    transfer_doc_to_text = next(
        row
        for row in tables["open_ocr_qa_unified_policy_transfer_summary"]
        if row["scope"] == "DocVQA-lite -> TextVQA-lite"
    )
    pooled_doc = next(
        row
        for row in tables["open_ocr_qa_unified_policy_transfer_summary"]
        if row["scope"] == "pooled dev -> DocVQA-lite"
    )
    pooled_text = next(
        row
        for row in tables["open_ocr_qa_unified_policy_transfer_summary"]
        if row["scope"] == "pooled dev -> TextVQA-lite"
    )
    learned_text_threeway = next(
        row
        for row in tables["open_ocr_qa_learned_risk_policy"]
        if row["task"] == "TextVQA-lite" and row["split"] == "test" and row["family"] == "learned_threeway"
    )
    learned_text_fullfallback = next(
        row
        for row in tables["open_ocr_qa_learned_risk_policy"]
        if row["task"] == "TextVQA-lite" and row["split"] == "test" and row["family"] == "learned_fullfallback"
    )
    learned_doc_threeway = next(
        row
        for row in tables["open_ocr_qa_learned_risk_policy"]
        if row["task"] == "DocVQA-lite" and row["split"] == "test" and row["family"] == "learned_threeway"
    )
    learned_doc_fullfallback = next(
        row
        for row in tables["open_ocr_qa_learned_risk_policy"]
        if row["task"] == "DocVQA-lite" and row["split"] == "test" and row["family"] == "learned_fullfallback"
    )
    learned_text_escalate_auc = next(
        row
        for row in tables["open_ocr_qa_learned_risk_model_weights"]
        if row["task"] == "TextVQA-lite" and row["label"] == "label_escalate"
    )
    learned_doc_escalate_auc = next(
        row
        for row in tables["open_ocr_qa_learned_risk_model_weights"]
        if row["task"] == "DocVQA-lite" and row["label"] == "label_escalate"
    )
    stability_text_stable30_70 = next(
        row
        for row in tables["open_ocr_qa_answer_stability_cascade"]
        if row["task"] == "TextVQA-lite" and row["split"] == "test" and row["policy"] == "stable30_else70"
    )
    stability_text_stable30_full = next(
        row
        for row in tables["open_ocr_qa_answer_stability_cascade"]
        if row["task"] == "TextVQA-lite" and row["split"] == "test" and row["policy"] == "stable30_elsefull"
    )
    stability_doc_stable30_70 = next(
        row
        for row in tables["open_ocr_qa_answer_stability_cascade"]
        if row["task"] == "DocVQA-lite" and row["split"] == "test" and row["policy"] == "stable30_else70"
    )
    stability_doc_stable30_full = next(
        row
        for row in tables["open_ocr_qa_answer_stability_cascade"]
        if row["task"] == "DocVQA-lite" and row["split"] == "test" and row["policy"] == "stable30_elsefull"
    )
    stability_signal_text_agree = next(
        row
        for row in tables["open_ocr_qa_answer_stability_signal"]
        if row["task"] == "TextVQA-lite" and row["split"] == "test" and row["group"] == "agree30_50"
    )
    stability_signal_text_disagree = next(
        row
        for row in tables["open_ocr_qa_answer_stability_signal"]
        if row["task"] == "TextVQA-lite" and row["split"] == "test" and row["group"] == "disagree30_50"
    )
    stability_signal_doc_agree = next(
        row
        for row in tables["open_ocr_qa_answer_stability_signal"]
        if row["task"] == "DocVQA-lite" and row["split"] == "test" and row["group"] == "agree30_50"
    )
    stability_signal_doc_disagree = next(
        row
        for row in tables["open_ocr_qa_answer_stability_signal"]
        if row["task"] == "DocVQA-lite" and row["split"] == "test" and row["group"] == "disagree30_50"
    )
    pregen_text_mask30_policy = next(
        row
        for row in tables["open_ocr_qa_pregen_risk_signal_policy"]
        if row["task"] == "TextVQA-lite" and row["feature_group"] == "mask30_only"
    )
    pregen_text_mask_stability_model = next(
        row
        for row in tables["open_ocr_qa_pregen_risk_signal_model"]
        if row["task"] == "TextVQA-lite"
        and row["target"] == "low_failure_ge0p25"
        and row["feature_group"] == "mask_stability_pregen"
        and row["split"] == "test"
    )
    pregen_doc_mask_stability_model = next(
        row
        for row in tables["open_ocr_qa_pregen_risk_signal_model"]
        if row["task"] == "DocVQA-lite"
        and row["target"] == "low_failure_ge0p25"
        and row["feature_group"] == "mask_stability_pregen"
        and row["split"] == "test"
    )
    pregen_doc_lowanswer_policy = next(
        row
        for row in tables["open_ocr_qa_pregen_risk_signal_policy"]
        if row["task"] == "DocVQA-lite" and row["feature_group"] == "question_mask_lowanswer"
    )
    repair_all = next(row for row in tables["open_ocr_qa_repairability_summary"] if row["task"] == "all")
    repair_doc = next(row for row in tables["open_ocr_qa_repairability_summary"] if row["task"] == "DocVQA-lite")
    repair_text = next(row for row in tables["open_ocr_qa_repairability_summary"] if row["task"] == "TextVQA-lite")
    repair_feature_all = next(
        row
        for row in tables["open_ocr_qa_repairability_feature"]
        if row["target"] == "low_failure_drop_ge_0p25"
        and row["scope"] == "all"
        and row["feature_group"] == "deployable"
        and row["feature"] == "low_answer_len"
    )
    repair_feature_doc_line = next(
        row
        for row in tables["open_ocr_qa_repairability_feature"]
        if row["target"] == "high_still_bad"
        and row["scope"] == "DocVQA-lite"
        and row["feature_group"] == "evidence_audit"
        and row["feature"] == "line_context_ECR_0p30"
    )
    box_fixed70 = next(
        row
        for row in tables["open_ocr_qa_box_aware_budget"]
        if row["scope"] == "answer_or_gt_bbox"
        and row["split"] == "test"
        and row["policy"] == "fixed_0.70"
    )
    box_to70 = next(
        row
        for row in tables["open_ocr_qa_box_aware_budget"]
        if row["scope"] == "answer_or_gt_bbox"
        and row["split"] == "test"
        and row["policy"] == "all_regions_ECR_ge_0p50_lt_0p05_to_70"
    )
    box_question = next(
        row
        for row in tables["open_ocr_qa_box_aware_budget"]
        if row["scope"] == "answer_or_gt_bbox"
        and row["split"] == "test"
        and row["family"] == "question_cue"
        and row["selected_by"] == "dev_best"
    )
    box_oracle = next(
        row
        for row in tables["open_ocr_qa_box_aware_budget"]
        if row["scope"] == "answer_or_gt_bbox"
        and row["split"] == "test"
        and row["policy"] == "oracle_best_budget"
    )
    box_fullish = next(
        row
        for row in tables["open_ocr_qa_box_aware_budget"]
        if row["scope"] == "answer_or_gt_bbox"
        and row["split"] == "test"
        and row["policy"] == "all_regions_ECR_ge_0p50_lt_0p05_to_full"
    )
    noisy_box_drop40_fixed70 = next(
        row
        for row in tables["open_ocr_qa_noisy_box_fallback_key"]
        if row["scope"] == "drop_40pct" and row["policy"] == "fixed_0.70"
    )
    noisy_box_drop40_missing = next(
        row
        for row in tables["open_ocr_qa_noisy_box_fallback_key"]
        if row["scope"] == "drop_40pct" and row["family"] == "missing_box_to_0p70"
    )
    noisy_box_drop40_allregions = next(
        row
        for row in tables["open_ocr_qa_noisy_box_fallback_key"]
        if row["scope"] == "drop_40pct"
        and row["family"] == "noisy_box_all_regions_ECR_ge_0p50_0p30_to_0p70"
    )
    noisy_box_mixed_allregions = next(
        row
        for row in tables["open_ocr_qa_noisy_box_fallback_key"]
        if row["scope"] == "mixed_light"
        and row["family"] == "noisy_box_all_regions_ECR_ge_0p50_0p30_to_0p70"
    )
    noisy_latency_drop40_allregions = next(
        row
        for row in tables["open_ocr_qa_noisy_box_latency_key"]
        if row["scope"] == "drop_40pct"
        and row["family"] == "noisy_box_all_regions_ECR_ge_0p50_0p30_to_0p70"
    )
    noisy_latency_mixed_allregions = next(
        row
        for row in tables["open_ocr_qa_noisy_box_latency_key"]
        if row["scope"] == "mixed_light"
        and row["family"] == "noisy_box_all_regions_ECR_ge_0p50_0p30_to_0p70"
    )
    coverage_target = next(row for row in tables["coverage_greedy_tradeoff"] if row["variant"] == "Target 0.30")
    coverage_greedy = next(
        row for row in tables["coverage_greedy_tradeoff"] if row["variant"] == "Coverage-greedy 0.30"
    )
    coverage_delta = next(
        row for row in tables["coverage_greedy_tradeoff"] if row["variant"] == "Coverage-greedy minus Target"
    )
    method_paired_all = next(row for row in tables["method_coverage_paired_summary"] if row["group"] == "all")
    method_paired_low_improved = next(
        row for row in tables["method_coverage_paired_summary"] if row["group"] == "target_low_ECR_and_ECR_improved"
    )
    method_paired_negative = next(row for row in tables["method_coverage_paired_summary"] if row["group"] == "negative_probes")
    method_paired_target_only = next(
        row for row in tables["method_coverage_paired_group"] if row["group"] == "target_only_correct"
    )
    method_paired_coverage_only = next(
        row for row in tables["method_coverage_paired_group"] if row["group"] == "coverage_only_correct"
    )
    method_component_best_acc = next(
        row for row in tables["method_component_summary"] if row["summary_item"] == "best_accuracy"
    )
    method_component_best_ecr = next(
        row for row in tables["method_component_summary"] if row["summary_item"] == "best_ECR"
    )
    method_component_soft_delta = next(
        row for row in tables["method_component_delta"] if row["variant"] == "Soft evidence 0.30"
    )
    method_component_coverage_count = next(
        row for row in tables["method_component_summary"] if row["summary_item"] == "coverage_tradeoff_count"
    )
    method_component_similar_count = next(
        row for row in tables["method_component_summary"] if row["summary_item"] == "evidence_gain_similar_risk_count"
    )
    llava_fastv_fairness = next(
        row
        for row in tables["external_baseline_fairness"]
        if row["backbone"] == "LLaVA-1.5-7B"
        and row["baseline"] == "FastV"
        and row["implementation_class"] == "official-algorithm port"
    )
    llava_visionzip_fairness = next(
        row
        for row in tables["external_baseline_fairness"]
        if row["backbone"] == "LLaVA-1.5-7B"
        and row["baseline"] == "VisionZip"
        and row["implementation_class"] == "official-algorithm port"
    )
    qwen_fastv_gap = next(
        row
        for row in tables["external_baseline_fairness"]
        if row["backbone"] == "Qwen3-VL-8B"
        and row["baseline"] == "FastV"
        and row["implementation_class"] == "not claimed"
    )
    qwen_visionzip_fairness = next(
        row
        for row in tables["external_baseline_fairness"]
        if row["backbone"] == "Qwen3-VL-8B"
        and row["baseline"] == "VisionZip"
        and row["implementation_class"] == "official-algorithm port"
    )
    qwen_visionzip_native = next(
        row
        for row in tables["qwen3_visionzip_native_port"]
        if row["comparison_type"] == "official_algorithm_port"
    )
    qwen_target_native = next(
        row
        for row in tables["qwen3_visionzip_native_port"]
        if row["comparison_type"] == "ours"
    )
    qwen_random_native = next(
        row
        for row in tables["qwen3_visionzip_native_port"]
        if row["comparison_type"] == "sanity_baseline"
    )
    internvl_fastv_gap = next(
        row
        for row in tables["external_baseline_fairness"]
        if row["backbone"] == "InternVL3.5-8B"
        and row["baseline"] == "FastV"
        and row["implementation_class"] == "not claimed"
    )
    qwen_visionzip_feasibility = next(
        row
        for row in tables["native_external_port_feasibility"]
        if row["backbone"] == "Qwen3-VL-8B" and row["method"] == "VisionZip"
    )
    qwen_fastv_feasibility = next(
        row
        for row in tables["native_external_port_feasibility"]
        if row["backbone"] == "Qwen3-VL-8B" and row["method"] == "FastV"
    )
    internvl_visionzip_feasibility = next(
        row
        for row in tables["native_external_port_feasibility"]
        if row["backbone"] == "InternVL3.5-8B" and row["method"] == "VisionZip"
    )
    internvl_fastv_feasibility = next(
        row
        for row in tables["native_external_port_feasibility"]
        if row["backbone"] == "InternVL3.5-8B" and row["method"] == "FastV"
    )
    llava_budget_040 = {
        row["method"]: row
        for row in tables["external_baseline_budget_curve"]
        if row["ratio"] == "0.400"
    }
    open_ecr_low = next(
        row
        for row in tables["open_ocr_qa_ecr_quality_bucket"]
        if row["scope"] == "all" and row["group"] == "ECR [0,0.25)"
    )
    open_ecr_high = next(
        row
        for row in tables["open_ocr_qa_ecr_quality_bucket"]
        if row["scope"] == "all" and row["group"] == "ECR [0.75,1.00]"
    )
    textvqa_ecr_pass = next(
        row
        for row in tables["open_ocr_qa_ecr_quality_bucket"]
        if row["scope"] == "TextVQA-lite" and row["group"] == "all_regions_ECR_ge_0p50=1"
    )
    textvqa_ecr_fail = next(
        row
        for row in tables["open_ocr_qa_ecr_quality_bucket"]
        if row["scope"] == "TextVQA-lite" and row["group"] == "all_regions_ECR_ge_0p50=0"
    )
    docvqa_ecr_pass = next(
        row
        for row in tables["open_ocr_qa_ecr_quality_bucket"]
        if row["scope"] == "DocVQA-lite" and row["group"] == "all_regions_ECR_ge_0p50=1"
    )
    docvqa_ecr_fail = next(
        row
        for row in tables["open_ocr_qa_ecr_quality_bucket"]
        if row["scope"] == "DocVQA-lite" and row["group"] == "all_regions_ECR_ge_0p50=0"
    )
    open_ecr_corr = next(
        row
        for row in tables["open_ocr_qa_ecr_quality_correlation"]
        if row["scope"] == "all" and row["x"] == "ECR" and row["y"] == "score_delta"
    )
    docvqa_line_boxes = next(
        row for row in tables["open_ocr_qa_docvqa_line_context"] if row["metric"] == "line_context_boxes"
    )
    docvqa_line_added = next(
        row for row in tables["open_ocr_qa_docvqa_line_context"] if row["metric"] == "added_context_boxes"
    )
    docvqa_line_multi = next(
        row for row in tables["open_ocr_qa_docvqa_line_context"] if row["metric"] == "multi_box_rows"
    )
    docvqa_line_70 = {
        row["metric"]: row["value"]
        for row in tables["open_ocr_qa_docvqa_line_context_ecr"]
        if row["scope"] == "DocVQA-lite@0.70"
    }
    docvqa_line_quality_pass = next(
        row
        for row in tables["docvqa_line_context_quality_bucket"]
        if row["scope"] == "DocVQA-line-context" and row["group"] == "all_regions_ECR_ge_0p50=1"
    )
    docvqa_line_quality_fail = next(
        row
        for row in tables["docvqa_line_context_quality_bucket"]
        if row["scope"] == "DocVQA-line-context" and row["group"] == "all_regions_ECR_ge_0p50=0"
    )
    docvqa_line_quality_corr = next(
        row
        for row in tables["docvqa_line_context_quality_correlation"]
        if row["scope"] == "DocVQA-line-context" and row["x"] == "worst_region_ECR" and row["y"] == "score_delta"
    )
    docvqa_doc_risk_70 = next(row for row in tables["docvqa_document_risk_by_budget"] if row["budget"] == "0.700")
    docvqa_doc_risk_low_failure = next(
        row for row in tables["docvqa_document_risk_trajectory_summary"] if row["group"] == "low_failure30"
    )
    docvqa_doc_risk_persistent = next(
        row
        for row in tables["docvqa_document_risk_trajectory_summary"]
        if row["group"] == "persistent70_among_low_failure30"
    )
    docvqa_doc_risk_persistent_high = next(
        row
        for row in tables["docvqa_document_risk_trajectory_summary"]
        if row["group"] == "persistent70_with_ECR70_ge0p75"
    )
    textvqa_noise_clean70 = next(
        row
        for row in tables["open_ocr_qa_bbox_noise"]
        if row["variant"] == "clean" and row["task"] == "TextVQA-lite" and row["budget_keep_ratio"] == "0.70"
    )
    textvqa_noise_jitter25_70 = next(
        row
        for row in tables["open_ocr_qa_bbox_noise"]
        if row["variant"] == "jitter_25pct" and row["task"] == "TextVQA-lite" and row["budget_keep_ratio"] == "0.70"
    )
    textvqa_noise_drop40_70 = next(
        row
        for row in tables["open_ocr_qa_bbox_noise"]
        if row["variant"] == "drop_40pct" and row["task"] == "TextVQA-lite" and row["budget_keep_ratio"] == "0.70"
    )
    docvqa_noise_clean70 = next(
        row
        for row in tables["open_ocr_qa_bbox_noise"]
        if row["variant"] == "clean" and row["task"] == "DocVQA-lite" and row["budget_keep_ratio"] == "0.70"
    )
    docvqa_noise_jitter25_70 = next(
        row
        for row in tables["open_ocr_qa_bbox_noise"]
        if row["variant"] == "jitter_25pct" and row["task"] == "DocVQA-lite" and row["budget_keep_ratio"] == "0.70"
    )
    docvqa_noise_drop40_70 = next(
        row
        for row in tables["open_ocr_qa_bbox_noise"]
        if row["variant"] == "drop_40pct" and row["task"] == "DocVQA-lite" and row["budget_keep_ratio"] == "0.70"
    )
    qwen_random30 = next(row for row in tables["random_seed_status"] if row["run"] == "qwen_random30")
    llava_random40 = next(row for row in tables["random_seed_status"] if row["run"] == "llava_random40")
    internvl_random50 = next(row for row in tables["random_seed_status"] if row["run"] == "internvl_random50")
    hard_lex = {row["metric"]: row["value"] for row in tables["hard_negative_lexical"]}
    hard_qc_launch = {row["metric"]: row["value"] for row in tables["hard_negative_human_qc_launch"]}
    hard_qc_progress = tables["hard_negative_human_qc_progress"][0]
    qwen_decode32_measured = next(
        row for row in tables["e2e_measured_decode"] if row["run"] == "qwen_highres_policy_decode32"
    )
    qwen_target20_len32 = next(
        row
        for row in tables["e2e_length_key"]
        if row["point"] == "target0p20"
        and row["detector_mode"] == "no_detector"
        and row["generated_tokens"] == "32"
    )
    qwen_target20_len128 = next(
        row
        for row in tables["e2e_length_key"]
        if row["point"] == "target0p20"
        and row["detector_mode"] == "no_detector"
        and row["generated_tokens"] == "128"
    )
    llava_protected_online32 = next(
        row
        for row in tables["e2e_length_key"]
        if row["point"] == "protected_embed0p40"
        and row["detector_mode"] == "online_detector"
        and row["generated_tokens"] == "32"
    )
    internvl_soft_online32 = next(
        row
        for row in tables["e2e_length_key"]
        if row["point"] == "soft_evidence0p50_hfpr"
        and row["detector_mode"] == "online_detector"
        and row["generated_tokens"] == "32"
    )
    triad_qwen_restore = next(
        row
        for row in tables["causal_evidence_triad"]
        if row["model"] == "Qwen3-VL-8B" and row["category"] == "necessity_and_restoration"
    )
    triad_qwen_occlusion = next(
        row
        for row in tables["causal_evidence_triad"]
        if row["model"] == "Qwen3-VL-8B" and row["category"] == "input_space_specificity"
    )
    triad_llava_region = next(
        row
        for row in tables["causal_evidence_triad"]
        if row["model"] == "LLaVA-1.5-7B" and row["category"] == "region_logit_necessity"
    )
    triad_llava_occlusion = next(
        row
        for row in tables["causal_evidence_triad"]
        if row["model"] == "LLaVA-1.5-7B" and row["category"] == "input_space_specificity"
    )
    triad_internvl_restore = next(
        row
        for row in tables["causal_evidence_triad"]
        if row["model"] == "InternVL3.5-8B calibrated"
        and row["category"] == "necessity_and_restoration"
    )
    text_replace_hq = next(row for row in tables["text_replacement_counterfactual"] if row["split"] == "HQ filtered")
    text_replace_pilot = next(row for row in tables["text_replacement_counterfactual"] if row["split"] == "Pilot")
    text_replace_ocr_hq = next(row for row in tables["text_replacement_ocr_quality"] if row["split"] == "HQ filtered")
    text_replace_ocr_pilot = next(row for row in tables["text_replacement_ocr_quality"] if row["split"] == "Pilot")
    text_replace_strat_hq_readable = next(
        row
        for row in tables["text_replacement_stratified"]
        if row["split"] == "HQ filtered" and row["group"] == "edited_crop_ocr_success"
    )
    text_replace_strat_pilot_readable = next(
        row
        for row in tables["text_replacement_stratified"]
        if row["split"] == "Pilot" and row["group"] == "edited_crop_ocr_success"
    )
    text_replace_qc_launch = {row["metric"]: row["value"] for row in tables["text_replacement_human_qc_launch"]}
    text_replace_qc_progress = tables["text_replacement_human_qc_progress"][0]
    causal_decision = tables["causal_evidence_decision"][0]
    causal_qwen_gate = next(
        row
        for row in tables["causal_evidence_go_no_go"]
        if row["scope"] == "Qwen3-VL-8B"
    )
    causal_internvl_gate = next(
        row
        for row in tables["causal_evidence_go_no_go"]
        if row["scope"] == "InternVL3.5-8B calibrated"
    )
    causal_llava_gate = next(
        row
        for row in tables["causal_evidence_go_no_go"]
        if row["scope"] == "LLaVA-1.5-7B"
    )
    causal_semantic_gate = next(
        row
        for row in tables["causal_evidence_go_no_go"]
        if row["gate"] == "semantic_text_replacement_counterfactual"
    )
    internvl_full_cal = next(
        row for row in tables["internvl_calibration_summary"] if row["model_run"] == "InternVL3.5-8B-full"
    )
    internvl_soft_default_audit = next(
        row
        for row in tables["internvl_soft_operating_audit"]
        if row["cross_model_run"] == "target_soft_evidence0p50_cal"
    )
    internvl_soft_hfpr_audit = next(
        row
        for row in tables["internvl_soft_operating_audit"]
        if row["cross_model_run"] == "target_soft_evidence0p50_hfpr_cal"
    )
    internvl_soft_identity = tables["internvl_soft_pair_identity"][0]
    main_random_multiseed = all(
        int(row.get("cached_random_realizations", 0) or 0) >= 2
        for row in (qwen_random30, llava_random40, internvl_random50)
    )
    return [
        {
            "claim_id": "C1",
            "claim": "Training-free prefill-only visual token pruning can keep 20-30% Qwen visual tokens on TextOCR-Hard while matching or slightly improving full-token quality.",
            "status": "supported",
            "safe_wording": "Qwen target@0.20/0.30 matches full-token accuracy within paired CIs and slightly improves the point estimates.",
            "primary_evidence": "cross_model_summary.csv; pairwise_stats.csv",
            "key_numbers": "full 0.787/0.236; target0p20 0.793/0.224; target0p30 0.798/0.222",
            "avoid_saying": "Do not claim statistically significant improvement over full; paired CIs include 0.",
        },
        {
            "claim_id": "C2",
            "claim": "The result is not explained only by the keep ratio.",
            "status": "supported_with_caveat",
            "safe_wording": "Same-budget random/grid/shuffled baselines lose substantial accuracy and ECR, though some have lower hFPR by becoming conservative.",
            "primary_evidence": "hard_evidence_summary.csv; p0_stats/pairwise_stats.csv",
            "key_numbers": (
                f"Qwen target0p30 acc 0.798/ECR 0.604 vs random30 mean "
                f"{qwen_random30.get('acc_mean', qwen_random30.get('acc', ''))}/"
                f"{qwen_random30.get('ECR_mean', qwen_random30.get('ECR', ''))} "
                f"over {qwen_random30.get('cached_random_realizations', '1')} realization(s), and grid 0.711/0.297."
            ),
            "avoid_saying": "Do not claim universal hFPR dominance over random/grid.",
        },
        {
            "claim_id": "C3",
            "claim": "The selected tokens preserve annotated OCR/bbox evidence better than simple baselines.",
            "status": "supported",
            "safe_wording": "Evidence-aware variants improve ECR/CenterR/PatchR over random, grid, shuffled, and selected external baselines at matched budgets.",
            "primary_evidence": "hard_evidence_summary.csv; table_causal_evidence_go_no_go.csv; table_causal_evidence_decision.csv",
            "key_numbers": "LLaVA protected ECR 1.000 vs grid 0.391/random 0.419; InternVL soft ECR 0.893 vs grid 0.695/random 0.741.",
            "avoid_saying": f"Do not claim full causal proof of evidence usage; the causal go/no-go audit returns {causal_decision['causal_claim_status']}.",
        },
        {
            "claim_id": "C4",
            "claim": "The method gives real CUDA efficiency gains, not just token-count reductions.",
            "status": "supported",
            "safe_wording": "Measured CUDA results show substantial shortened-prefix prefill and memory gains, while detector-inclusive single-sample latency must be reported separately for box-aware settings.",
            "primary_evidence": "textocr_hard_real_efficiency_report.csv; textocr_efficiency_decomposition.csv; table_open_ocr_qa_noisy_box_latency_key_summary.csv; problem_optimization_audit/open_ocr_qa_noisy_box_latency/noisy_box_latency_report.md",
            "key_numbers": (
                "Qwen target0p20: 1.45x single-sample TTFT, 4.32x batch prefill, incremental peak -76.4%; "
                "LLaVA/InternVL box-aware rows lose single-sample speedup if EasyOCR is run online per image; "
                f"for open-QA noisy all-region fallback under 40% box dropout, estimated TTFT is "
                f"{noisy_latency_drop40_allregions['estimated_no_detector_ttft_ms']} ms "
                f"({noisy_latency_drop40_allregions['speedup_no_detector']}x) without detector cost, but "
                f"{noisy_latency_drop40_allregions['estimated_detector_mean_ttft_ms']} ms "
                f"({noisy_latency_drop40_allregions['speedup_with_mean_detector']}x) when mean EasyOCR latency is added online."
            ),
            "avoid_saying": "Do not imply 4.32x is end-to-end generation speedup or that external OCR detection is free.",
        },
        {
            "claim_id": "C4b",
            "claim": "End-to-end generation speedup remains large for long outputs.",
            "status": "boundary_result",
            "safe_wording": "The strongest efficiency claim is shortened-prefix prefill/TTFT and memory reduction; measured Qwen decode32 and a conservative length sweep show that full-generation speedups shrink as output length grows.",
            "primary_evidence": "table_e2e_measured_decode32.csv; table_e2e_length_sensitivity_key_points.csv; problem_optimization_audit/end_to_end_efficiency_length/end_to_end_efficiency_length_report.md",
            "key_numbers": (
                f"Measured Qwen high-resolution policy: TTFT speedup {qwen_decode32_measured['mean_TTFT_speedup']} but 32-token generation speedup "
                f"{qwen_decode32_measured['mean_generation_speedup']} with full/pruned decode totals "
                f"{qwen_decode32_measured['mean_decode_full_ms_total']}/{qwen_decode32_measured['mean_decode_pruned_ms_total']} ms. "
                f"TextOCR Qwen Target20 no-detector length sweep: 32-token speedup {qwen_target20_len32['speedup']} and 128-token speedup "
                f"{qwen_target20_len128['speedup']} under equal decode latency. "
                f"With online detector included, LLaVA Protected40 32-token speedup is {llava_protected_online32['speedup']} and InternVL Soft50 32-token speedup is "
                f"{internvl_soft_online32['speedup']}."
            ),
            "avoid_saying": "Do not describe batch-prefill speedup as end-to-end generation speedup. The TextOCR length sweep uses an equal per-token decode assumption from measured Qwen decode32, so it is a conservative latency-bound estimate, not a new cross-model decode benchmark.",
        },
        {
            "claim_id": "C5",
            "claim": "The approach generalizes beyond one backbone.",
            "status": "partially_supported",
            "safe_wording": "Qwen, LLaVA, and InternVL each have a defensible operating point, but the positive mechanism differs by backbone.",
            "primary_evidence": "cross_model_summary.csv; textocr_hard_real_efficiency_report.csv",
            "key_numbers": "Qwen target; LLaVA embed/protected/grid; InternVL calibrated soft/target/grid.",
            "avoid_saying": "Do not claim one selector is uniformly best across all backbones.",
        },
        {
            "claim_id": "C6",
            "claim": "OCR-safe pruning improves domain-shift robustness on OCRBench-style verification.",
            "status": "partially_supported",
            "safe_wording": "Grid-floor target pruning is safer than pure target pruning on OCRBench subsets, especially for Qwen/InternVL.",
            "primary_evidence": "ocrbench_generalization_summary.csv",
            "key_numbers": "Qwen target0p30 0.890/0.080 vs grid_topk0p30 0.920/0.050; InternVL target0p50 0.900/0.160 vs grid_topk0p50 0.925/0.120.",
            "avoid_saying": "Do not claim OCRBench leaderboard performance or that grid_topk is best on TextOCR-Hard.",
        },
        {
            "claim_id": "C7",
            "claim": "The OCRBench evidence is not limited to binary yes/no probes.",
            "status": "supported_with_scope",
            "safe_wording": "On original OCRBench questions, Qwen target-grid pruning supports answer ranking at 30% visual tokens and native greedy generation at higher retention; 70% retention nearly matches the full-prefix exact score on the 100-question subset.",
            "primary_evidence": "table_ocrbench_open_answer_ranking.csv; table_ocrbench_open_answer_generation.csv",
            "key_numbers": (
                f"full rank acc {open_overall['full_rank_acc']}; pruned rank acc {open_overall['pruned_rank_acc']}; "
                f"delta {open_overall['delta_rank_acc']}; generation exact at 70% {open_gen_70['pruned_exact']} "
                f"vs full {open_gen_70['full_exact']}; generation ANLS {open_gen_70['pruned_anls']} "
                f"vs full {open_gen_70['full_anls']}."
            ),
            "avoid_saying": "Do not claim OCRBench leaderboard performance; the generation check is a 100-question subset and 30-50% retention remains fragile.",
        },
        {
            "claim_id": "C7b",
            "claim": "The native open-answer check extends beyond OCRBench.",
            "status": "supported_with_scope",
            "safe_wording": "On the full 500-sample TextVQA-lite native generation check, question-conditioned pruning remains usable at higher retention but aggressive 30% pruning causes a large drop.",
            "primary_evidence": "table_open_ocr_qa_generation.csv",
            "key_numbers": (
                f"full TextVQA acc {textvqa_50['full_score']}; 30% {textvqa_30['pruned_score']} "
                f"(delta {textvqa_30['delta_score']}), 50% {textvqa_50['pruned_score']} "
                f"(delta {textvqa_50['delta_score']}), 70% {textvqa_70['pruned_score']} "
                f"(delta {textvqa_70['delta_score']})."
            ),
            "avoid_saying": "Do not claim full TextVQA leaderboard performance; this is the 500-sample lite split, not the official full validation set.",
        },
        {
            "claim_id": "C7c",
            "claim": "Document-centric native open-answer QA is a harder boundary case.",
            "status": "supported_with_scope",
            "safe_wording": "On the full 500-sample DocVQA-lite native generation check, Qwen question-conditioned pruning requires high retention: 70% retains most ANLS, while 30-50% is too aggressive.",
            "primary_evidence": "table_open_ocr_qa_generation.csv",
            "key_numbers": (
                f"full DocVQA ANLS {docvqa_70['full_score']}; 30% {docvqa_30['pruned_score']} "
                f"(delta {docvqa_30['delta_score']}), 50% {docvqa_50['pruned_score']} "
                f"(delta {docvqa_50['delta_score']}), 70% {docvqa_70['pruned_score']} "
                f"(delta {docvqa_70['delta_score']}); exact {docvqa_70['pruned_exact']} vs full {docvqa_70['full_exact']} at 70%."
            ),
            "avoid_saying": "Do not claim DocVQA leaderboard performance or that low-budget pruning is safe for document QA.",
        },
        {
            "claim_id": "C7d",
            "claim": "Open OCR/document QA failures concentrate on harder answer/question subgroups.",
            "status": "supported_with_scope",
            "safe_wording": "A cached stress audit shows that aggressive pruning is most fragile for long-question, numeric, and multi-token-answer subgroups; these are heuristic stress tags, not ground-truth multi-evidence annotations. A source-boundary audit separates external TextVQA GT boxes, DocVQA OCR-derived answer-token boxes, DocVQA line-context expansion, and the not-yet-completed human annotation pack.",
            "primary_evidence": "table_open_ocr_qa_stress.csv; table_open_ocr_qa_stress_manifest_summary.csv; table_open_ocr_qa_annotation_pack_summary.csv; table_manual_evidence_readiness_gates.csv; table_manual_final_package_status.csv; table_open_ocr_qa_evidence_source_boundary.csv; table_open_ocr_qa_evidence_prefill_summary.csv; table_open_ocr_qa_bbox_tool_summary.csv; table_open_ocr_qa_bbox_validation_summary.csv; table_open_ocr_qa_bbox_ecr_summary.csv; table_open_ocr_qa_external_bbox_adapter_summary.csv; table_open_ocr_qa_textvqa_gt_bbox_summary.csv; table_open_ocr_qa_textvqa_gt_bbox_ecr_summary.csv; table_open_ocr_qa_docvqa_ocr_bbox_summary.csv; table_open_ocr_qa_docvqa_ocr_bbox_ecr_summary.csv; table_open_ocr_qa_bbox_quality_summary.csv; table_open_ocr_qa_bbox_noise_summary.csv; problem_optimization_audit/open_ocr_qa_stress/open_ocr_qa_stress_report.md; problem_optimization_audit/open_ocr_qa_stress_manifest/open_ocr_qa_stress_manifest.md; problem_optimization_audit/open_ocr_qa_annotation_pack/annotation_pack.md; problem_optimization_audit/manual_evidence_readiness_gate/manual_evidence_readiness_report.md; problem_optimization_audit/open_ocr_qa_manual_final_package/manual_final_package_report.md; problem_optimization_audit/open_ocr_qa_evidence_source_boundary/evidence_source_boundary_report.md; problem_optimization_audit/open_ocr_qa_evidence_prefill/evidence_prefill_report.md; problem_optimization_audit/open_ocr_qa_bbox_annotation_tool/index.html; problem_optimization_audit/open_ocr_qa_bbox_ecr/bbox_ecr_report.md; problem_optimization_audit/open_ocr_qa_external_bbox_annotations/external_bbox_annotation_report.md; problem_optimization_audit/open_ocr_qa_textvqa_gt_bbox_ecr/bbox_ecr_report.md; problem_optimization_audit/open_ocr_qa_docvqa_hxlinh_bbox_expanded_ecr/bbox_ecr_report.md; problem_optimization_audit/open_ocr_qa_bbox_quality_audit_expanded/bbox_quality_report.md; problem_optimization_audit/open_ocr_qa_bbox_noise_audit/bbox_noise_report.md",
            "key_numbers": (
                f"DocVQA long-question subgroup n={stress_docvqa_long30['n']}: 30% "
                f"{stress_docvqa_long30['full_score']}->{stress_docvqa_long30['pruned_score']} "
                f"(delta {stress_docvqa_long30['delta_score']}), 70% "
                f"{stress_docvqa_long70['full_score']}->{stress_docvqa_long70['pruned_score']} "
                f"(delta {stress_docvqa_long70['delta_score']}); TextVQA numeric subgroup n={stress_textvqa_numeric30['n']}: "
                f"30% {stress_textvqa_numeric30['full_score']}->{stress_textvqa_numeric30['pruned_score']} "
                f"(delta {stress_textvqa_numeric30['delta_score']}), 70% "
                f"{stress_textvqa_numeric70['full_score']}->{stress_textvqa_numeric70['pruned_score']} "
                f"(delta {stress_textvqa_numeric70['delta_score']}); external TextVQA GT bbox covers 47/48 stress samples, with mean ECR "
                f"0.232/0.457/0.677 at 30%/50%/70% retention; public DocVQA OCR expanded answer-token boxes cover 47/48 stress samples and 106 OCR boxes, with mean ECR "
                f"0.247/0.418/0.687 at 30%/50%/70% retention; bbox quality audit finds 19 multi-box DocVQA rows and, among 32 annotated multi-token DocVQA rows, 16 have full label recall while 15 are partial; "
                "source-boundary audit at 70%: TextVQA GT boxes ECR/worst/all-region-pass 0.677/0.677/0.851; "
                "DocVQA answer-token boxes 0.687/0.581/0.511; DocVQA line-context boxes 0.773/0.423/0.298; "
                f"bbox-noise audit at 70% shows 25% coordinate jitter leaves TextVQA/DocVQA ECR near clean "
                f"({textvqa_noise_clean70['mean_ECR']}->{textvqa_noise_jitter25_70['mean_ECR']}, "
                f"{docvqa_noise_clean70['mean_ECR']}->{docvqa_noise_jitter25_70['mean_ECR']}), but 40% box dropout lowers retention-adjusted ECR to "
                f"{textvqa_noise_drop40_70['retention_adjusted_ECR']} and {docvqa_noise_drop40_70['retention_adjusted_ECR']}."
            ),
            "avoid_saying": "Do not describe stress tags as manually verified multi-evidence boxes, do not report synthetic smoke annotation rows as evidence, and do not claim final human multi-region annotation is complete. TextVQA boxes are external GT boxes, while DocVQA boxes are OCR-derived expanded answer-token spans and may still under-cover layout relations or non-answer context. Do not call the bbox-noise audit an end-to-end noisy-detector pruning run; it is an audit-signal sensitivity test over cached masks.",
        },
        {
            "claim_id": "C7e",
            "claim": "Open-QA evidence coverage is associated with answer preservation.",
            "status": "supported_as_association",
            "safe_wording": "On annotated TextVQA/DocVQA stress rows, higher bbox ECR is associated with smaller native-generation score drops, but counterexamples show that ECR remains an availability/risk signal rather than proof of causal evidence use.",
            "primary_evidence": "table_open_ocr_qa_ecr_quality_bucket_summary.csv; table_open_ocr_qa_ecr_quality_correlation_summary.csv; problem_optimization_audit/open_ocr_qa_ecr_quality_association/ecr_quality_association_report.md",
            "key_numbers": (
                f"all annotated rows: ECR<0.25 has pruned score {open_ecr_low['mean_pruned_score']} and delta {open_ecr_low['mean_score_delta']} "
                f"vs ECR>=0.75 score {open_ecr_high['mean_pruned_score']} and delta {open_ecr_high['mean_score_delta']}; "
                f"ECR-vs-delta Pearson/Spearman {open_ecr_corr['pearson']}/{open_ecr_corr['spearman']}; "
                f"TextVQA all-regions pass/fail pruned score {textvqa_ecr_pass['mean_pruned_score']}/{textvqa_ecr_fail['mean_pruned_score']}; "
                f"DocVQA all-regions pass/fail pruned score {docvqa_ecr_pass['mean_pruned_score']}/{docvqa_ecr_fail['mean_pruned_score']}."
            ),
            "avoid_saying": "Do not call this causal evidence usage; the report includes high-ECR large-drop and low-ECR correct-answer counterexamples.",
        },
        {
            "claim_id": "C7f",
            "claim": "Document QA evidence extends beyond the answer token itself.",
            "status": "supported_as_boundary_audit",
            "safe_wording": "A deterministic DocVQA OCR line-context audit expands answer-token boxes with nearby same-line OCR context and shows that multi-token context preservation is stricter than answer-token availability alone.",
            "primary_evidence": "table_open_ocr_qa_docvqa_line_context_summary.csv; table_open_ocr_qa_docvqa_line_context_ecr_summary.csv; table_docvqa_line_context_quality_bucket_summary.csv; table_docvqa_line_context_quality_correlation_summary.csv; table_docvqa_document_risk_by_budget.csv; table_docvqa_document_risk_by_condition.csv; table_docvqa_document_risk_trajectory_summary.csv; problem_optimization_audit/open_ocr_qa_docvqa_hxlinh_line_context_bbox/line_context_report.md; problem_optimization_audit/open_ocr_qa_docvqa_hxlinh_line_context_bbox_ecr/bbox_ecr_report.md; problem_optimization_audit/docvqa_line_context_quality_association/line_context_quality_report.md; problem_optimization_audit/docvqa_document_evidence_risk_decomposition/docvqa_document_risk_decomposition_report.md",
            "key_numbers": (
                f"DocVQA line-context audit expands 106 answer-token boxes to {docvqa_line_boxes['value']} OCR boxes "
                f"with {docvqa_line_added['value']} added context boxes and {docvqa_line_multi['value']} multi-box rows; "
                f"at 70% retention mean ECR {docvqa_line_70['mean_ECR']}, worst-region ECR {docvqa_line_70['mean_worst_region_ECR']}, "
                f"and all-regions ECR>=0.50 fraction {docvqa_line_70['mean_all_regions_ECR_ge_0p50']}; "
                f"line-context all-regions pass/fail rows have pruned score {docvqa_line_quality_pass['mean_pruned_score']}/{docvqa_line_quality_fail['mean_pruned_score']}, "
                f"with worst-region-ECR-vs-delta Pearson/Spearman {docvqa_line_quality_corr['pearson']}/{docvqa_line_quality_corr['spearman']}; "
                f"document-risk decomposition at 70% has all-region pass {docvqa_doc_risk_70['all_regions_pass_rate']}, pruned-good {docvqa_doc_risk_70['pruned_good_rate']}, "
                f"low-failure rate {docvqa_doc_risk_70['low_failure_rate_ge0p25']}, and high-mean-ECR-but-bad rate {docvqa_doc_risk_70['high_mean_ECR_but_bad_rate']}; "
                f"among 30% low-failure trajectories, 70% repairs {docvqa_doc_risk_low_failure['repaired70_rate']} but persistent failure remains {docvqa_doc_risk_low_failure['persistent70_rate']}, "
                f"with {docvqa_doc_risk_persistent_high['n']} persistent cases still having ECR70>=0.75."
            ),
            "avoid_saying": "Do not describe the line-context boxes as manual layout annotations or complete document reasoning evidence; they are deterministic OCR-neighborhood context boxes. Do not use high ECR as proof of answer use: the decomposition explicitly includes persistent failures with high mean ECR and even all-region pass.",
        },
        {
            "claim_id": "C8",
            "claim": "Official external-method comparisons are covered where technically valid.",
            "status": "supported_with_scope",
            "safe_wording": "LLaVA official-algorithm VisionZip/FastV ports are evaluated across matched budgets; Qwen3 VisionZip has a scoped native official-algorithm port at the 0.30 TextOCR-Hard budget; Qwen3 FastV and InternVL external rows remain proxy or unsupported.",
            "primary_evidence": "table_external_baseline_fairness_matrix.csv; table_external_baseline_budget_curve.csv; table_external_baseline_paired_readout.csv; table_qwen3_visionzip_native_port.csv; table_native_external_port_feasibility.csv; problem_optimization_audit/qwen3_visionzip_native_port_gate/qwen3_visionzip_native_port_gate_report.md; problem_optimization_audit/native_external_port_feasibility/native_external_port_feasibility_report.md; official_baseline_extension_report.md; external_baseline_parity_audit.md; hard_evidence_summary.csv",
            "key_numbers": (
                f"LLaVA FastV official budgets {llava_fastv_fairness['budgets_evaluated']} and VisionZip official budgets {llava_visionzip_fairness['budgets_evaluated']}; "
                f"at LLaVA 0.40, ours {llava_budget_040['ours_protected_embed']['acc']}/{llava_budget_040['ours_protected_embed']['hFPR']} ECR {llava_budget_040['ours_protected_embed']['ECR']} "
                f"vs VisionZip {llava_budget_040['VisionZip']['acc']}/{llava_budget_040['VisionZip']['hFPR']} ECR {llava_budget_040['VisionZip']['ECR']} "
                f"and FastV {llava_budget_040['FastV']['acc']}/{llava_budget_040['FastV']['hFPR']} ECR {llava_budget_040['FastV']['ECR']}; "
                f"Qwen3 VisionZip native port budget {qwen_visionzip_fairness['budgets_evaluated']} has "
                f"{qwen_visionzip_native['accuracy']} accuracy / {qwen_visionzip_native['hFPR']} hFPR / "
                f"{qwen_visionzip_native['ECR']} ECR at mean keep {qwen_visionzip_native['mean_actual_keep_ratio']}, "
                f"versus Qwen Target 0.30 {qwen_target_native['accuracy']} / {qwen_target_native['hFPR']} / {qwen_target_native['ECR']} "
                f"and Random 0.30 {qwen_random_native['accuracy']} / {qwen_random_native['hFPR']} / {qwen_random_native['ECR']}; "
                f"Qwen FastV official status {qwen_fastv_gap['result_status']}; InternVL FastV official status {internvl_fastv_gap['result_status']}; "
                f"native feasibility audit labels Qwen3 VisionZip as {qwen_visionzip_feasibility['feasibility']}, Qwen3 FastV as {qwen_fastv_feasibility['feasibility']}, "
                f"InternVL VisionZip as {internvl_visionzip_feasibility['feasibility']}, and InternVL FastV as {internvl_fastv_feasibility['feasibility']}."
            ),
            "avoid_saying": "Do not claim official FastV/VisionZip results are available on all three backbones, do not claim Qwen3 VisionZip beats our Qwen Target 0.30 accuracy, and do not treat FastV hFPR=0 as evidence safety because it is an all-no collapse on LLaVA.",
        },
        {
            "claim_id": "C9",
            "claim": "Risk-aware dynamic budgeting is solved.",
            "status": "partially_supported_as_diagnostic",
            "safe_wording": "Split-safe adaptive searches, including learned calibrated risk predictors, conformal-style calibrated fallback, answer-stability cascades, domain-aware portfolios, and open-QA selective fallback rules, are useful diagnostics but do not solve universal risk-aware budgeting. A deployment-contract audit finds a scoped TextVQA pre-generation policy that nearly matches fixed 70% quality at slightly lower cost, while DocVQA still has no deployable lower-cost near-fixed70 policy. A task-aware portfolio slightly lowers aggregate cost by falling back to fixed70 on DocVQA, but this is only a partial portfolio result; oracle budget selection remains the main headroom signal.",
            "primary_evidence": "table_adaptive_policy.csv; table_adaptive_controller_go_no_go.csv; table_adaptive_controller_decision.csv; table_open_ocr_qa_conformal_risk_selection.csv; table_open_ocr_qa_conformal_risk_decision.csv; table_open_ocr_qa_domain_aware_portfolio_rows.csv; table_open_ocr_qa_domain_aware_portfolio_decision.csv; table_open_ocr_qa_adaptive_budget_summary.csv; table_open_ocr_qa_adaptive_budget_selection.csv; table_open_ocr_qa_risk_coverage_frontier.csv; table_open_ocr_qa_learned_risk_policy.csv; table_open_ocr_qa_learned_risk_selection.csv; table_open_ocr_qa_learned_risk_model_weights.csv; table_open_ocr_qa_answer_stability_cascade.csv; table_open_ocr_qa_answer_stability_selection.csv; table_open_ocr_qa_answer_stability_signal.csv; table_open_ocr_qa_pregen_risk_signal_model.csv; table_open_ocr_qa_pregen_risk_signal_policy.csv; table_open_ocr_qa_deployment_contract.csv; table_open_ocr_qa_deployment_contract_summary.csv; table_open_ocr_qa_detector_aware_policy_summary.csv; table_open_ocr_qa_detector_aware_policy_readout.csv; table_open_ocr_qa_repairability_summary.csv; table_open_ocr_qa_repairability_feature_summary.csv; table_open_ocr_qa_box_aware_budget_summary.csv; table_open_ocr_qa_box_aware_budget_selection.csv; table_open_ocr_qa_noisy_box_fallback_key_summary.csv; table_open_ocr_qa_noisy_box_latency_key_summary.csv; textocr_adaptive_policy/qwen_target_risk_v1/policy_report.md; textocr_adaptive_policy/qwen_multisignal_risk_v2/policy_report.md; textocr_adaptive_policy/qwen_calibrated_risk_v1/policy_report.md; textocr_adaptive_policy/qwen_oracle_budget_frontier/oracle_budget_frontier.md; problem_optimization_audit/adaptive_controller_go_no_go/adaptive_controller_go_no_go.md; problem_optimization_audit/open_ocr_qa_conformal_risk_policy/conformal_risk_policy_report.md; problem_optimization_audit/open_ocr_qa_domain_aware_portfolio/domain_aware_portfolio_report.md; problem_optimization_audit/open_ocr_qa_adaptive_budget/adaptive_budget_report.md; problem_optimization_audit/open_ocr_qa_risk_coverage_frontier/risk_coverage_frontier_report.md; problem_optimization_audit/open_ocr_qa_learned_risk_policy/learned_risk_policy_report.md; problem_optimization_audit/open_ocr_qa_answer_stability_cascade/answer_stability_cascade_report.md; problem_optimization_audit/open_ocr_qa_answer_stability_signal/answer_stability_signal_report.md; problem_optimization_audit/open_ocr_qa_pregen_risk_signal/pregen_risk_signal_report.md; problem_optimization_audit/open_ocr_qa_deployment_contract/deployment_contract_report.md; problem_optimization_audit/open_ocr_qa_detector_aware_policy/detector_aware_policy_report.md; problem_optimization_audit/open_ocr_qa_repairability/repairability_report.md; problem_optimization_audit/open_ocr_qa_box_aware_budget/box_aware_budget_report.md; problem_optimization_audit/open_ocr_qa_noisy_box_fallback/noisy_box_fallback_report.md; problem_optimization_audit/open_ocr_qa_noisy_box_latency/noisy_box_latency_report.md",
            "key_numbers": (
                f"single-signal adaptive test {adaptive_dev['test_acc']}/{adaptive_dev['test_hFPR']} "
                f"at keep {adaptive_dev['test_keep']} and cascade {adaptive_dev['test_cascade']}; "
                f"multi-signal deployable test {adaptive_multisignal['test_acc']}/{adaptive_multisignal['test_hFPR']} "
                f"at keep {adaptive_multisignal['test_keep']}, cascade {adaptive_multisignal['test_cascade']}, and ECR {adaptive_multisignal['test_ECR']}; "
                f"learned-risk test {adaptive_learned['test_acc']}/{adaptive_learned['test_hFPR']} "
                f"at keep {adaptive_learned['test_keep']}, cascade {adaptive_learned['test_cascade']}, and ECR {adaptive_learned['test_ECR']}; "
                f"fixed0.30 test {adaptive_fixed30['test_acc']}/{adaptive_fixed30['test_hFPR']} "
                f"at keep {adaptive_fixed30['test_keep']} and ECR {adaptive_fixed30['test_ECR']}; "
                f"fixed0.35 test {adaptive_fixed35['test_acc']}/{adaptive_fixed35['test_hFPR']} "
                f"at keep {adaptive_fixed35['test_keep']} and ECR {adaptive_fixed35['test_ECR']}; "
                "oracle min-correct upper bound reaches 0.852/0.158 at mean keep 0.336; "
                f"open TextVQA test fixed70 {open_adapt_text_fixed70['score']} at keep {open_adapt_text_fixed70['mean_keep']} vs best full-fallback {open_adapt_text_fullfallback['score']} at keep {open_adapt_text_fullfallback['mean_keep']} and oracle {open_adapt_text_oracle['score']} at keep {open_adapt_text_oracle['mean_keep']}; "
                f"open DocVQA test fixed70 {open_adapt_doc_fixed70['score']} at keep {open_adapt_doc_fixed70['mean_keep']} vs best full-fallback {open_adapt_doc_fullfallback['score']} at keep {open_adapt_doc_fullfallback['mean_keep']} and oracle {open_adapt_doc_oracle['score']} at keep {open_adapt_doc_oracle['mean_keep']}; "
                f"risk-coverage frontier at zero per-sample loss uses mean keep {textvqa_frontier0['mean_keep']} on TextVQA test with full fallback {textvqa_frontier0['full_fallback_rate']} and mean keep {docvqa_frontier0['mean_keep']} on DocVQA test with full fallback {docvqa_frontier0['full_fallback_rate']}; "
                f"learned deployable open-QA risk scores have dev escalation AUC {learned_text_escalate_auc['dev_auc']} on TextVQA and {learned_doc_escalate_auc['dev_auc']} on DocVQA; "
                f"TextVQA learned three-way policy scores {learned_text_threeway['score']} at keep {learned_text_threeway['mean_keep']} and learned full-fallback scores {learned_text_fullfallback['score']} at keep {learned_text_fullfallback['mean_keep']}; "
                f"DocVQA learned three-way policy scores {learned_doc_threeway['score']} at keep {learned_doc_threeway['mean_keep']} and learned full-fallback scores {learned_doc_fullfallback['score']} at keep {learned_doc_fullfallback['mean_keep']}; "
                f"answer-stability cascade: TextVQA stable30_else70 scores {stability_text_stable30_70['score']} at selected keep {stability_text_stable30_70['selected_mean_keep']} but serial cascade cost {stability_text_stable30_70['cascade_mean_cost']} "
                f"and stable30_elsefull scores {stability_text_stable30_full['score']} at selected keep {stability_text_stable30_full['selected_mean_keep']} with full fallback {stability_text_stable30_full['full_fallback_rate']}; "
                f"DocVQA stable30_else70 scores {stability_doc_stable30_70['score']} at selected keep {stability_doc_stable30_70['selected_mean_keep']} and serial cost {stability_doc_stable30_70['cascade_mean_cost']}, while stable30_elsefull scores {stability_doc_stable30_full['score']} at selected keep {stability_doc_stable30_full['selected_mean_keep']} but serial cost {stability_doc_stable30_full['cascade_mean_cost']}; "
                f"answer-stability signal: TextVQA agree30_50 covers {stability_signal_text_agree['fraction']} of test cases with safe30 {stability_signal_text_agree['safe30_within0p10_rate']} and low-failure {stability_signal_text_agree['low_failure_rate_ge0p25']}, while disagree30_50 has low-failure {stability_signal_text_disagree['low_failure_rate_ge0p25']} and 70%-repair-among-low-fail {stability_signal_text_disagree['repair70_rate_among_low_fail']}; "
                f"DocVQA agree30_50 covers {stability_signal_doc_agree['fraction']} with safe30 {stability_signal_doc_agree['safe30_within0p10_rate']}, while disagree30_50 has low-failure {stability_signal_doc_disagree['low_failure_rate_ge0p25']} and 70%-repair-among-low-fail {stability_signal_doc_disagree['repair70_rate_among_low_fail']}; "
                f"pre-generation risk signal: TextVQA mask-stability low-failure AUC is {pregen_text_mask_stability_model['auc']} and the mask30-only 30->70 policy scores {pregen_text_mask30_policy['test_score']} at keep {pregen_text_mask30_policy['test_mean_keep']} versus fixed70 {pregen_text_mask30_policy['fixed70_test_score']}; "
                f"DocVQA mask-stability low-failure AUC is {pregen_doc_mask_stability_model['auc']}, and even the low-answer-assisted 30->70 policy scores {pregen_doc_lowanswer_policy['test_score']} at keep {pregen_doc_lowanswer_policy['test_mean_keep']} versus fixed70 {pregen_doc_lowanswer_policy['fixed70_test_score']}; "
                f"deployment-contract audit: TextVQA near-fixed70 lower-cost candidate is {textvqa_deploy_contract['policy']} with score {textvqa_deploy_contract['test_score']} and cost proxy {textvqa_deploy_contract['deployment_cost_proxy']}, while DocVQA has {docvqa_deploy_contract['policy']} passing candidate under the same gate; "
                f"domain-aware portfolio audit returns {domain_portfolio['portfolio_status']} with aggregate score {domain_portfolio['aggregate_score']} vs fixed70 {domain_portfolio['fixed70_aggregate_score']} "
                f"and aggregate cost {domain_portfolio['aggregate_cost']} vs fixed70 {domain_portfolio['fixed70_aggregate_cost']}, falling back on {domain_portfolio['fallback_tasks']}; "
                f"repairability audit: 30% low-budget failure rates are {repair_text['low_failure_rate']} TextVQA and {repair_doc['low_failure_rate']} DocVQA, and 70% retention repairs {repair_all['repaired_by_70_rate_among_low_fail']} of low-budget failures overall; "
                f"the best simple deployable low-failure feature here has AUC {repair_feature_all['auc_best_direction']}, while DocVQA line-context ECR for still-bad-after-70 cases reaches AUC {repair_feature_doc_line['auc_best_direction']} on annotated rows; "
                f"box-aware annotated-subset audit: question-cue fallback scores {box_question['score']} at keep {box_question['mean_keep']}, while all-region-ECR 30->70 escalation scores {box_to70['score']} at keep {box_to70['mean_keep']} versus fixed70 {box_fixed70['score']} at keep {box_fixed70['mean_keep']} and oracle {box_oracle['score']} at keep {box_oracle['mean_keep']}; "
                f"full-fallback ECR scoring reaches {box_fullish['score']} but with keep {box_fullish['mean_keep']}, showing near-full fallback rather than fine-grained budget control; "
                f"under simulated 40% box dropout, missing-box-only fallback scores {noisy_box_drop40_missing['score']} at keep {noisy_box_drop40_missing['mean_keep']}, while noisy all-region ECR fallback scores {noisy_box_drop40_allregions['score']} at keep {noisy_box_drop40_allregions['mean_keep']} versus fixed70 {noisy_box_drop40_fixed70['score']} at keep {noisy_box_drop40_fixed70['mean_keep']}; "
                f"under mixed light noise, noisy all-region ECR fallback scores {noisy_box_mixed_allregions['score']} at keep {noisy_box_mixed_allregions['mean_keep']}; "
                f"detector-aware latency estimate: drop40 noisy all-region fallback is {noisy_latency_drop40_allregions['estimated_no_detector_ttft_ms']} ms "
                f"({noisy_latency_drop40_allregions['speedup_no_detector']}x) without detector cost but {noisy_latency_drop40_allregions['estimated_detector_mean_ttft_ms']} ms "
                f"({noisy_latency_drop40_allregions['speedup_with_mean_detector']}x) with mean EasyOCR cost, and mixed-light all-region fallback is "
                f"{noisy_latency_mixed_allregions['estimated_detector_mean_ttft_ms']} ms detector-inclusive; "
                f"the aggregate adaptive-controller go/no-go audit returns {tables['adaptive_controller_decision'][0]['main_controller_status']} "
                f"with {tables['adaptive_controller_decision'][0]['failed_required_gates']} failed required gates; "
                f"the calibrated fallback audit returns {tables['conformal_risk_decision'][0]['conformal_policy_status']} "
                f"with go tasks {tables['conformal_risk_decision'][0]['go_tasks']} at epsilon {tables['conformal_risk_decision'][0]['epsilon_for_main_gate']}."
            ),
            "avoid_saying": "Do not claim unified adaptive risk control is solved or stronger than fixed-budget rows. Answer-stability agreement is a diagnostic signal, not a free controller, because observing it requires extra serial generations and its reliability differs between TextVQA and DocVQA. Pre-generation selector-mask and detector/evidence risk signals are useful diagnostics but do not pass the held-out stress-pack gate as a universal controller. The noisy-box fallback and detector-aware policy audits are cached-budget simulations, not end-to-end detector-in-the-loop MLLM runs; do not claim detector-assisted fallback improves single-sample TTFT unless evidence boxes are already available, precomputed, or amortized.",
        },
        {
            "claim_id": "C9b",
            "claim": "A unified adaptive controller transfers across open OCR/document QA tasks.",
            "status": "not_supported_boundary_quantified",
            "safe_wording": "A cross-task and pooled-dev transfer audit shows that current deployable pre-generation risk policies do not satisfy the near-fixed70/lower-keep gate on both TextVQA-lite and DocVQA-lite. This turns the hand-selected-policy concern into an explicit empirical boundary rather than an unstated assumption.",
            "primary_evidence": "table_open_ocr_qa_unified_policy_transfer_summary.csv; table_open_ocr_qa_unified_policy_cross_task.csv; table_open_ocr_qa_unified_policy_pooled.csv; problem_optimization_audit/open_ocr_qa_unified_policy_transfer/unified_policy_transfer_report.md",
            "key_numbers": (
                f"TextVQA->DocVQA best transferred policy {transfer_text_to_doc['best_quality_policy']} scores "
                f"{transfer_text_to_doc['best_quality_target_score']} at keep {transfer_text_to_doc['best_quality_target_keep']} "
                f"(delta vs fixed70 {transfer_text_to_doc['best_quality_delta_vs_fixed70']}) with "
                f"{transfer_text_to_doc['near_fixed70_lower_keep_candidates']} passing candidates; "
                f"DocVQA->TextVQA best transferred policy {transfer_doc_to_text['best_quality_policy']} scores "
                f"{transfer_doc_to_text['best_quality_target_score']} at keep {transfer_doc_to_text['best_quality_target_keep']} "
                f"(delta {transfer_doc_to_text['best_quality_delta_vs_fixed70']}) with "
                f"{transfer_doc_to_text['near_fixed70_lower_keep_candidates']} passing candidates; "
                f"pooled-dev best DocVQA policy {pooled_doc['best_quality_policy']} scores {pooled_doc['best_quality_target_score']} "
                f"at keep {pooled_doc['best_quality_target_keep']} (delta {pooled_doc['best_quality_delta_vs_fixed70']}); "
                f"pooled-dev best TextVQA policy {pooled_text['best_quality_policy']} scores {pooled_text['best_quality_target_score']} "
                f"at keep {pooled_text['best_quality_target_keep']} (delta {pooled_text['best_quality_delta_vs_fixed70']})."
            ),
            "avoid_saying": "Do not claim a solved unified adaptive controller. The audit should be framed as transparent boundary evidence and motivation for future risk calibration, not as a positive universal-control result.",
        },
        {
            "claim_id": "C10",
            "claim": "Annotated OCR evidence regions causally affect the model's yes/no decision margins under pruning.",
            "status": "supported_with_scope",
            "safe_wording": "Matched region logit-drop diagnostics show strong causal-style evidence on Qwen and cross-backbone diagnostic support with backbone-specific trade-offs.",
            "primary_evidence": "region_logit_drop*/*region_logit_drop_pairwise.csv; table_region_logit_drop.csv",
            "key_numbers": "Qwen ours-vs-removed +1.089 overall/+3.316 positive and ours-vs-full +0.006; LLaVA ours-vs-full +0.019 overall/+0.053 positive; InternVL-cal evidence-kept-vs-removed +0.218 overall.",
            "avoid_saying": "Do not call this a full causal oracle across all models or say evidence removal behaves identically across backbones.",
        },
        {
            "claim_id": "C11",
            "claim": "Input-space OCR evidence occlusion confirms that annotated text regions affect OCR decisions.",
            "status": "supported_with_scope",
            "safe_wording": "Input-space bbox masking strongly reduces positive yes-support on Qwen and InternVL, with a smaller same-direction LLaVA effect; same-size random masking is near-neutral.",
            "primary_evidence": "table_bbox_occlusion.csv; qwen/llava/internvl bbox occlusion reports",
            "key_numbers": "Qwen orig-vs-evidence +2.190 overall/+6.261 positive; InternVL +0.786 positive; LLaVA +0.025 positive; orig-vs-random near zero on all three.",
            "avoid_saying": "Do not claim identical behavior across backbones; negative probes can move in the opposite direction under evidence masking.",
        },
        {
            "claim_id": "C11b",
            "claim": "Retained evidence tokens are behaviorally important rather than merely annotated as evidence.",
            "status": "supported_with_scope",
            "safe_wording": "Qwen gives the cleanest deletion-restoration proof; calibrated InternVL shows the same direction under its risk-constrained threshold.",
            "primary_evidence": "table_deletion_restoration.csv; table_internvl_deletion_restoration.csv; qwen_target30_runs/deletion_restoration_report.md; internvl_soft50_runs_cal_hfprconstr/deletion_restoration_report.md",
            "key_numbers": "Qwen selected/remove/restore-all/random-all acc 0.798/0.769/0.798/0.770; InternVL calibrated selected/remove/restore-all/random-all acc 0.647/0.586/0.647/0.597.",
            "avoid_saying": "Do not claim a universal causal proof across all backbones; InternVL depends on the calibrated yes/no threshold and LLaVA still relies on complementary diagnostics.",
        },
        {
            "claim_id": "C11c",
            "claim": "The causal evidence can be separated into necessity, sufficiency/restoration, and specificity diagnostics.",
            "status": "supported_with_scope",
            "safe_wording": "A triad audit shows strong Qwen support, calibrated InternVL support, and weak or mixed LLaVA causal-style evidence; this supports evidence-preservation as a risk-aware diagnostic while preserving backbone-specific caveats.",
            "primary_evidence": "table_causal_evidence_triad.csv; table_causal_evidence_go_no_go.csv; table_causal_evidence_decision.csv; problem_optimization_audit/causal_evidence_triad/causal_evidence_triad_report.md; problem_optimization_audit/causal_evidence_go_no_go/causal_evidence_go_no_go.md; table_deletion_restoration.csv; table_internvl_deletion_restoration.csv; table_region_logit_drop.csv; table_bbox_occlusion.csv",
            "key_numbers": (
                f"Causal go/no-go status {causal_decision['causal_claim_status']}: "
                f"Qwen gate {causal_qwen_gate['status']}, InternVL gate {causal_internvl_gate['status']}, "
                f"LLaVA gate {causal_llava_gate['status']}, semantic replacement gate {causal_semantic_gate['status']}; "
                f"Qwen deletion accuracy drop {triad_qwen_restore['primary_effect']} with evidence/random full-restore recovery "
                f"{triad_qwen_restore['supporting_effect']}/{triad_qwen_restore['control_effect']}; "
                f"Qwen input-space positive-support drop evidence/random {triad_qwen_occlusion['primary_effect']}/{triad_qwen_occlusion['control_effect']}; "
                f"InternVL calibrated deletion drop {triad_internvl_restore['primary_effect']} with evidence/random full-restore recovery "
                f"{triad_internvl_restore['supporting_effect']}/{triad_internvl_restore['control_effect']}; "
                f"LLaVA region margin kept-minus-removed {triad_llava_region['primary_effect']} and input-space positive-support drop evidence/random "
                f"{triad_llava_occlusion['primary_effect']}/{triad_llava_occlusion['control_effect']}."
            ),
            "avoid_saying": "Do not collapse ECR, token deletion, region logit-drop, input occlusion, and text replacement into a single causal proof. The triad audit explicitly marks LLaVA as weak or mixed, while semantic text replacement remains a boundary result.",
        },
        {
            "claim_id": "C11d",
            "claim": "Semantic text replacement provides a stronger counterfactual than bbox masking.",
            "status": "supported_as_boundary_audit",
            "safe_wording": "A TextOCR text-replacement diagnostic changes the local word to its paired near-miss string and joins original/edited predictions; Qwen shows positive average support shifts but low full four-way semantic switch rates. Conditioning on OCR-readable edits improves switch rates, but the rates remain modest. A human-QC package for edit validity is launch-ready but not complete, so this remains a renderer-quality and causal-scope boundary audit rather than a strong causal proof.",
            "primary_evidence": "table_text_replacement_counterfactual.csv; table_text_replacement_ocr_quality.csv; table_text_replacement_stratified.csv; table_text_replacement_human_qc_launch_summary.csv; table_text_replacement_human_qc_progress_decision.csv; table_human_qc_claim_gate_rows.csv; table_human_qc_claim_gate_decision.csv; problem_optimization_audit/text_replacement_counterfactual_joined/qwen_hq49_report.md; problem_optimization_audit/text_replacement_counterfactual_joined/qwen_pilot100_report.md; problem_optimization_audit/text_replacement_ocr_quality/qwen_hq49_report.md; problem_optimization_audit/text_replacement_ocr_quality/qwen_pilot100_report.md; problem_optimization_audit/text_replacement_stratified/qwen_hq49_report.md; problem_optimization_audit/text_replacement_stratified/qwen_pilot100_report.md; problem_optimization_audit/text_replacement_human_qc_launch/annotator_handbook.md; problem_optimization_audit/text_replacement_human_qc_progress/text_replacement_human_qc_progress_report.md; problem_optimization_audit/human_qc_claim_gate/human_qc_claim_gate_report.md",
            "key_numbers": (
                f"HQ filtered n={text_replace_hq['n_pairs']}: original pair-correct {text_replace_hq['original_pair_correct_rate']}, "
                f"edited pair-correct {text_replace_hq['edited_pair_correct_rate']}, full four-way switch {text_replace_hq['full_four_way_semantic_switch_rate']}, "
                f"mean source yes-support drop {text_replace_hq['mean_source_yes_support_drop']} and replacement yes-support gain {text_replace_hq['mean_replacement_yes_support_gain']}; "
                f"pilot n={text_replace_pilot['n_pairs']}: full four-way switch {text_replace_pilot['full_four_way_semantic_switch_rate']}; "
                f"EasyOCR edited-crop replacement readability is {text_replace_ocr_hq['edited_crop_replacement_detected_rate']} on HQ and {text_replace_ocr_pilot['edited_crop_replacement_detected_rate']} on pilot; "
                f"within OCR-readable edits, full four-way switch rises to {text_replace_strat_hq_readable['full_four_way_semantic_switch_rate']} on HQ "
                f"and {text_replace_strat_pilot_readable['full_four_way_semantic_switch_rate']} on pilot, with edited-pair-correct-given-original-correct "
                f"{text_replace_strat_hq_readable['edited_pair_correct_given_original_correct']} and "
                f"{text_replace_strat_pilot_readable['edited_pair_correct_given_original_correct']}; "
                f"human QC launch rows {text_replace_qc_launch.get('qc_manifest_rows', '')}, ready rows "
                f"{text_replace_qc_progress.get('ready_rows', '')}, status "
                f"{text_replace_qc_progress.get('text_replacement_human_qc_status', '')}."
            ),
            "avoid_saying": "Do not call the current text replacement experiment photorealistic, human-verified, a human-verified semantic edit, or a universal causal proof; the renderer is local, OCR readability is limited, the switch rate is low, and human QC is not complete.",
        },
        {
            "claim_id": "C12",
            "claim": "BBox-assisted variants are not purely oracle-box results.",
            "status": "supported_with_scope",
            "safe_wording": "A real EasyOCR detector source preserves the InternVL soft-evidence result and exposes detector sensitivity for LLaVA protected pruning.",
            "primary_evidence": "table_box_source_robustness.csv; easyocr_input_summary.json",
            "key_numbers": f"EasyOCR detector {easyocr_ms:.1f} ms/image with oracle-missing rate {easyocr_missing:.3f}; InternVL EasyOCR 0.649/0.157 true ECR 0.888; LLaVA EasyOCR 0.639/0.322 true ECR 0.655.",
            "avoid_saying": "Do not imply the detector is free or that every backbone is equally robust to detector misses.",
        },
        {
            "claim_id": "C13",
            "claim": "TextOCR-Hard paired statistics remain stable when probes are clustered by image.",
            "status": "supported",
            "safe_wording": "Image-cluster bootstrap intervals preserve the main paired conclusions while respecting that each image contributes both positive and negative probes.",
            "primary_evidence": "table_image_cluster_statistics.csv",
            "key_numbers": "Qwen target30-full acc diff +0.011 CI [-0.005,+0.028]; LLaVA protected-VisionZip acc diff +0.081 CI [+0.054,+0.107]; InternVL soft-full hFPR diff -0.131 CI [-0.190,-0.071].",
            "avoid_saying": "Do not treat the 1000 TextOCR-Hard probes as fully independent examples.",
        },
        {
            "claim_id": "C14",
            "claim": "TextOCR-Hard near-miss negatives are not accidental positives in the same image OCR annotations.",
            "status": "supported",
            "safe_wording": "All 500 near-miss negative targets are absent from the same-image OCR token set, with mostly single-substitution or single-deletion edits.",
            "primary_evidence": "table_hard_negative_quality.csv; table_hard_negative_human_qc_launch_summary.csv; table_hard_negative_human_qc_progress_decision.csv; table_human_qc_claim_gate_rows.csv; table_human_qc_claim_gate_decision.csv; hard_negative_quality_examples.csv; problem_optimization_audit/human_qc_claim_gate/human_qc_claim_gate_report.md",
            "key_numbers": (
                "500 negatives; same-image OCR collision 0/500; mean edit distance 1.012; "
                "73.0% single substitution and 25.8% single deletion; "
                f"human QC launch rows {hard_qc_launch.get('qc_manifest_rows', '')}, ready rows {hard_qc_progress.get('ready_rows', '')}, "
                f"status {hard_qc_progress.get('hard_negative_human_qc_status', '')}."
            ),
            "avoid_saying": "Do not claim manual human verification unless it is actually performed.",
        },
        {
            "claim_id": "C14b",
            "claim": "TextOCR-Hard near-miss negatives remain valid under common lexical normalizers.",
            "status": "supported_with_scope",
            "safe_wording": "A stricter automatic lexical audit finds no same-image negative-target collisions after NFC/NFKC, case-folding, whitespace removal, alphanumeric stripping, or ASCII-folding normalizers.",
            "primary_evidence": "table_hard_negative_lexical_summary.csv; table_hard_negative_edit_class_summary.csv; table_hard_negative_suspicious_examples.csv; table_hard_negative_human_qc_launch_summary.csv; table_hard_negative_human_qc_progress_decision.csv; table_human_qc_claim_gate_rows.csv; table_human_qc_claim_gate_decision.csv; problem_optimization_audit/hard_negative_lexical_audit/hard_negative_lexical_report.md; problem_optimization_audit/hard_negative_human_qc_launch/annotator_handbook.md; problem_optimization_audit/hard_negative_human_qc_progress/hard_negative_human_qc_progress_report.md; problem_optimization_audit/human_qc_claim_gate/human_qc_claim_gate_report.md",
            "key_numbers": (
                f"{hard_lex.get('negative_probe_count', '')} negatives over {hard_lex.get('image_count', '')} images; "
                f"same-image collision any-normalizer {hard_lex.get('same_image_collision_any_normalizer_count', '')}/"
                f"{hard_lex.get('negative_probe_count', '')} (rate {hard_lex.get('same_image_collision_any_normalizer_rate', '')}); "
                f"NFKC-casefold collision {hard_lex.get('same_image_collision_nfkc_casefold_count', '')}; "
                f"alnum collision {hard_lex.get('same_image_collision_nfkc_casefold_alnum_count', '')}; "
                f"source-target same after any normalizer {hard_lex.get('source_target_same_any_normalizer_count', '')}; "
                f"edit-distance>2 count {hard_lex.get('edit_distance_gt2_count', '')}."
            ),
            "avoid_saying": "Do not claim semantic or human-verified negative validity. Other-image global OCR collisions are not construction errors; they only show that some near-miss strings are plausible OCR tokens elsewhere.",
        },
        {
            "claim_id": "C15",
            "claim": "Random baseline variance is characterized for the main matched-budget random controls.",
            "status": "supported_with_scope" if main_random_multiseed else "partially_supported",
            "safe_wording": (
                "The main matched-budget random controls for Qwen, LLaVA, and InternVL now have multi-seed mean/std summaries; Qwen random20 remains a secondary single-seed row."
                if main_random_multiseed
                else "Only part of the random-control table has multi-seed summaries; remaining single-seed rows should be treated as cached controls."
            ),
            "primary_evidence": "table_random_seed_status.csv",
            "key_numbers": (
                f"Qwen random30 n={qwen_random30.get('cached_random_realizations', '')}: "
                f"acc {qwen_random30.get('acc_mean', '')}±{qwen_random30.get('acc_std', '')}, "
                f"hFPR {qwen_random30.get('hFPR_mean', '')}±{qwen_random30.get('hFPR_std', '')}, "
                f"ECR {qwen_random30.get('ECR_mean', '')}±{qwen_random30.get('ECR_std', '')}; "
                f"LLaVA random40 n={llava_random40.get('cached_random_realizations', '')}: "
                f"acc {llava_random40.get('acc_mean', '')}±{llava_random40.get('acc_std', '')}, "
                f"ECR {llava_random40.get('ECR_mean', '')}±{llava_random40.get('ECR_std', '')}; "
                f"InternVL random50 n={internvl_random50.get('cached_random_realizations', '')}: "
                f"acc {internvl_random50.get('acc_mean', '')}±{internvl_random50.get('acc_std', '')}, "
                f"hFPR {internvl_random50.get('hFPR_mean', '')}±{internvl_random50.get('hFPR_std', '')}, "
                f"ECR {internvl_random50.get('ECR_mean', '')}±{internvl_random50.get('ECR_std', '')}."
            ),
            "avoid_saying": "Do not imply every auxiliary random row has multi-seed variance; Qwen random20 remains a secondary single cached control.",
        },
        {
            "claim_id": "C15b",
            "claim": "InternVL soft-evidence operating points are different threshold choices, not different token-selection results.",
            "status": "supported",
            "safe_wording": "InternVL is yes-biased at the default threshold and all main InternVL rows use dev-split threshold calibration; the two soft-evidence rows share the same selector/raw margins/ECR but use different yes/no thresholds.",
            "primary_evidence": "table_internvl_threshold_calibration_summary.csv; table_internvl_soft_operating_audit.csv; table_internvl_soft_pair_identity_check.csv; problem_optimization_audit/internvl_calibration_audit/internvl_calibration_audit_report.md",
            "key_numbers": (
                f"default full-prefix InternVL hFPR {internvl_full_cal['default_all_hFPR']} with yes-rate {internvl_full_cal['default_all_yes_rate']}; "
                f"soft default threshold {internvl_soft_default_audit['yesno_threshold']} gives acc/hFPR "
                f"{internvl_soft_default_audit['acc']}/{internvl_soft_default_audit['hFPR']}; "
                f"soft hFPR-constrained threshold {internvl_soft_hfpr_audit['yesno_threshold']} gives acc/hFPR "
                f"{internvl_soft_hfpr_audit['acc']}/{internvl_soft_hfpr_audit['hFPR']}; "
                f"identity check over {internvl_soft_identity['shared_rows']} probes has raw/ECR/selector mismatches "
                f"{internvl_soft_identity['raw_margin_mismatches']}/{internvl_soft_identity['ECR_mismatches']}/{internvl_soft_identity['selector_mismatches']} "
                f"and {internvl_soft_identity['yes_to_no']} yes->no threshold-driven prediction changes."
            ),
            "avoid_saying": "Do not imply the 0.175 and 0.328 hFPR soft-evidence rows differ in ECR or token selection; the difference is a calibrated decision-threshold operating point.",
        },
        {
            "claim_id": "C16",
            "claim": "The method is more than an unprincipled score-plus-top-k heuristic.",
            "status": "partially_supported_as_design_and_diagnostic",
            "safe_wording": "The selector family is organized by an explicit evidence-risk objective, and component/Pareto audits show which objective terms help. Soft evidence gives a small ECR gain at similar answer risk, while hard evidence protection and coverage-greedy expose the boundary: maximizing evidence availability alone can raise hFPR or lower accuracy.",
            "primary_evidence": "table_method_objective_mapping.csv; table_coverage_greedy_tradeoff.csv; table_method_coverage_paired_summary.csv; table_method_coverage_paired_group_summary.csv; table_method_component_pareto.csv; table_method_component_delta_vs_target.csv; table_method_component_pareto_summary.csv; problem_optimization_audit/method_principle_audit/method_principle_report.md; problem_optimization_audit/method_coverage_paired_audit/method_coverage_paired_report.md; problem_optimization_audit/method_component_pareto/method_component_pareto_report.md",
            "key_numbers": (
                f"component Pareto audit identifies {method_component_best_acc['value']} as the best-accuracy row and {method_component_best_ecr['value']} as the best-ECR row; "
                f"Soft evidence 0.30 changes accuracy/hFPR/ECR by {method_component_soft_delta['delta_accuracy_vs_target']}/{method_component_soft_delta['delta_hFPR_vs_target']}/{method_component_soft_delta['delta_ECR_vs_target']} versus Target 0.30; "
                f"{method_component_coverage_count['value']} component variants show coverage gain with answer-risk trade-off, while {method_component_similar_count['value']} shows evidence gain with similar answer risk; "
                f"coverage-greedy at 30% keep reaches ECR {coverage_greedy['ECR']} vs Target 0.30 ECR {coverage_target['ECR']} "
                f"(delta {coverage_delta['ECR']}), but accuracy changes by {coverage_delta['accuracy']} and hFPR by {coverage_delta['hFPR']}; "
                f"coverage-greedy selector/overhead costs are {coverage_greedy['selector_ms']}/{coverage_greedy['overhead_ms']} ms; "
                f"paired audit over {method_paired_all['n']} probes shows target-only correct {method_paired_target_only['n']} vs coverage-only correct {method_paired_coverage_only['n']}, "
                f"negative-probe hFPR {method_paired_negative['target_hFPR']}->{method_paired_negative['coverage_hFPR']}, and on low-target-ECR probes with ECR improved, "
                f"accuracy changes {method_paired_low_improved['target_acc']}->{method_paired_low_improved['coverage_acc']} despite ECR {method_paired_low_improved['mean_target_ECR']}->{method_paired_low_improved['mean_coverage_ECR']}."
            ),
            "avoid_saying": "Do not promote coverage-greedy as the main method or claim the objective solves answer-risk control. The paired audit shows evidence coverage can rise while negative-probe false positives and target-only-correct losses also rise.",
        },
        {
            "claim_id": "C17",
            "claim": "The method is robust to hard OCR, conflict, and detector-failure cases.",
            "status": "partially_supported_as_boundary_audit",
            "safe_wording": "Hard-case audits support a narrower claim: the near-miss negative benchmark is lexically clean, detector and noisy-box experiments identify which failure modes are mild or severe, and automatic OCR boxes can help in scoped settings. The same audits also show that answer risk can rise when evidence protection is too hard, DocVQA-style open document QA remains difficult, and detector dropout requires fallback.",
            "primary_evidence": "table_hard_robustness_conflict_summary.csv; table_hard_robustness_textocr_method_slice.csv; table_hard_robustness_openqa_slice.csv; table_hard_robustness_detector_slice.csv; problem_optimization_audit/hard_robustness_conflict/hard_robustness_conflict_report.md",
            "key_numbers": (
                f"{hard_validity['key_result']} "
                f"{hard_nearmiss_risk['key_result']} "
                f"{hard_openqa_stress['key_result']} "
                f"{hard_noisy_boxes['key_result']} "
                f"{hard_detector_backbone['key_result']} "
                f"{hard_openqa_detector['key_result']}"
            ),
            "avoid_saying": "Do not claim exhaustive robustness to all dense, multilingual, handwritten, rotated, or conflict-heavy documents. Do not claim detector-assisted pruning improves latency unless OCR boxes are already available, precomputed, amortized, or explicitly included in the timing budget.",
        },
    ]


def markdown(claims: list[dict[str, str]], tables: dict[str, list[dict[str, Any]]]) -> str:
    lines = [
        "# Paper Evidence Package",
        "",
        "This package freezes the current safe claim boundary. All tables are generated from cached experiment outputs; no values are invented.",
        "",
        "## Claim Ledger",
        "",
        "| id | status | safe claim | primary evidence | key numbers | avoid saying |",
        "|---|---|---|---|---|---|",
    ]
    for row in claims:
        lines.append(
            f"| {row['claim_id']} | {row['status']} | {row['safe_wording']} | "
            f"{row['primary_evidence']} | {row['key_numbers']} | {row['avoid_saying']} |"
        )
    lines.extend(
        [
            "",
            "## Table 1: Main TextOCR-Hard Results",
            "",
            table_md(
                tables["main_textocr"],
                ["model", "method", "n", "acc", "hFPR", "keep_ratio", "ECR", "CenterR", "PatchR", "note"],
            ),
            "",
            "## Table 2: Same-Budget / Evidence / External Baselines",
            "",
            table_md(
                tables["evidence_baselines"],
                [
                    "family",
                    "group",
                    "model",
                    "method",
                    "role",
                    "acc",
                    "hFPR",
                    "keep_ratio",
                    "ECR",
                    "delta_acc",
                    "delta_hFPR",
                    "delta_ECR",
                ],
            ),
            "",
            "## Table 2b: External Baseline Fairness Matrix",
            "",
            table_md(
                tables["external_baseline_fairness"],
                [
                    "backbone",
                    "baseline",
                    "implementation_class",
                    "result_status",
                    "budgets_evaluated",
                    "same_protocol",
                    "same_budget_curve",
                    "safe_claim",
                    "remaining_gap",
                ],
            ),
            "",
            "## Table 2c: LLaVA Official-Port Budget Curve",
            "",
            table_md(
                tables["external_baseline_budget_curve"],
                [
                    "method",
                    "implementation",
                    "ratio",
                    "acc",
                    "hFPR",
                    "yes_rate",
                    "keep_ratio",
                    "ECR",
                    "note",
                ],
            ),
            "",
            "## Table 2d: LLaVA Official-Port Paired Readout",
            "",
            table_md(
                tables["external_baseline_paired"],
                [
                    "left",
                    "right",
                    "ratio",
                    "n",
                    "acc_diff_left_minus_right",
                    "acc_p",
                    "hFPR_diff_left_minus_right",
                    "hFPR_p",
                    "interpretation",
                ],
            ),
            "",
            "## Table 2e: Qwen3 VisionZip Native-Port Check",
            "",
            table_md(
                tables["qwen3_visionzip_native_port"],
                [
                    "method",
                    "comparison_type",
                    "n",
                    "accuracy",
                    "hFPR",
                    "mean_actual_keep_ratio",
                    "ECR",
                    "center_recall",
                    "patch_recall",
                    "score_source",
                    "matched_budget_to_target",
                    "accuracy_delta_vs_ours",
                    "hfpr_delta_vs_ours",
                    "ecr_delta_vs_ours",
                ],
            ),
            "",
            "## Table 2f: Native External-Port Feasibility",
            "",
            table_md(
                tables["native_external_port_feasibility"],
                [
                    "backbone",
                    "method",
                    "feasibility",
                    "current_status",
                    "main_blocker",
                    "required_work",
                    "paper_action",
                ],
            ),
            "",
            "## Table 3: Method Objective Mapping",
            "",
            table_md(
                tables["method_objective_mapping"],
                [
                    "selector_family",
                    "objective_terms",
                    "mechanism_tested",
                    "main_role",
                    "known_limit",
                ],
            ),
            "",
            "## Table 4: Coverage-Greedy Trade-Off",
            "",
            table_md(
                tables["coverage_greedy_tradeoff"],
                [
                    "variant",
                    "role",
                    "n",
                    "accuracy",
                    "hFPR",
                    "keep_ratio",
                    "ECR",
                    "CenterR",
                    "PatchR",
                    "selector_ms",
                    "overhead_ms",
                ],
            ),
            "",
            "## Table 4b: Coverage-Greedy Paired Summary",
            "",
            table_md(
                tables["method_coverage_paired_summary"],
                [
                    "group",
                    "n",
                    "target_acc",
                    "coverage_acc",
                    "target_hFPR",
                    "coverage_hFPR",
                    "mean_target_ECR",
                    "mean_coverage_ECR",
                    "mean_ECR_delta",
                    "target_only_rate",
                    "coverage_only_rate",
                ],
            ),
            "",
            "## Table 4c: Coverage-Greedy Paired Outcome Groups",
            "",
            table_md(
                tables["method_coverage_paired_group"],
                [
                    "group",
                    "n",
                    "mean_ECR_delta",
                    "mean_margin_abs_delta",
                    "both_correct",
                    "target_only_correct",
                    "coverage_only_correct",
                    "both_wrong",
                ],
            ),
            "",
            "## Table 4d: Method Component Pareto Audit",
            "",
            table_md(
                tables["method_component_pareto"],
                [
                    "variant",
                    "family",
                    "accuracy",
                    "hFPR",
                    "mean_keep",
                    "mean_ECR",
                    "mean_center_recall",
                    "mean_patch_recall",
                    "pareto_front",
                    "dominated_by",
                ],
            ),
            "",
            "## Table 4e: Method Component Delta Versus Target",
            "",
            table_md(
                tables["method_component_delta"],
                [
                    "variant",
                    "delta_accuracy_vs_target",
                    "delta_hFPR_vs_target",
                    "delta_ECR_vs_target",
                    "interpretation",
                ],
            ),
            "",
            "## Table 5: Real Efficiency",
            "",
            table_md(
                tables["efficiency"],
                [
                    "model",
                    "point",
                    "acc",
                    "hFPR",
                    "keep_ratio",
                    "single_forward_speedup",
                    "batch_prefill_speedup",
                    "full_samples_per_s",
                    "pruned_samples_per_s",
                    "incremental_peak_reduction_pct",
                    "batch_overhead_pct_saved_prefill",
                ],
            ),
            "",
            "## Table 6: Efficiency Decomposition",
            "",
            table_md(
                tables["efficiency_decomposition"],
                [
                    "point",
                    "acc",
                    "hFPR",
                    "keep_ratio",
                    "full_TTFT_ms",
                    "pruned_TTFT_ms",
                    "TTFT_speedup_no_detector",
                    "detector_ms",
                    "detector_inclusive_TTFT_ms",
                    "TTFT_speedup_with_detector",
                    "batch_prefill_speedup",
                    "incremental_peak_reduction_pct",
                    "note",
                ],
            ),
            "",
            "## Table 6b: Measured Qwen Decode32 Boundary",
            "",
            table_md(
                tables["e2e_measured_decode"],
                [
                    "run",
                    "num_samples",
                    "decode_steps",
                    "mean_visual_keep_ratio",
                    "mean_ECR",
                    "mean_TTFT_speedup",
                    "mean_generation_speedup",
                    "mean_decode_full_ms_total",
                    "mean_decode_pruned_ms_total",
                ],
            ),
            "",
            "## Table 6c: End-to-End Length Sensitivity Key Points",
            "",
            table_md(
                tables["e2e_length_key"],
                [
                    "label",
                    "detector_mode",
                    "generated_tokens",
                    "full_TTFT_ms",
                    "pruned_base_TTFT_ms",
                    "speedup",
                    "latency_reduction_pct",
                    "assumption",
                ],
            ),
            "",
            "## Table 7a: OCRBench Generalization Check",
            "",
            table_md(
                tables["ocrbench"],
                ["model", "method", "ratio", "implementation", "n", "acc", "hFPR", "keep_ratio", "note"],
            ),
            "",
            "## Table 7b: OCRBench Native Open-Question Answer Ranking",
            "",
            table_md(
                tables["open_answer"],
                [
                    "scope",
                    "type",
                    "n",
                    "full_rank_acc",
                    "pruned_rank_acc",
                    "delta_rank_acc",
                    "full_margin",
                    "pruned_margin",
                    "delta_margin",
                    "keep_ratio",
                    "selector_answer_tokens",
                    "note",
                ],
            ),
            "",
            "## Table 7c: OCRBench Native Open-Question Generation",
            "",
            table_md(
                tables["open_generation"],
                [
                    "method",
                    "n",
                    "full_exact",
                    "pruned_exact",
                    "delta_exact",
                    "full_anls",
                    "pruned_anls",
                    "delta_anls",
                    "full_contains",
                    "pruned_contains",
                    "keep_ratio",
                    "selector_answer_tokens",
                    "note",
                ],
            ),
            "",
            "## Table 8a: TextVQA Native Open-Answer Generation",
            "",
            table_md(
                tables["open_ocr_qa_generation"],
                [
                    "method",
                    "task",
                    "n",
                    "primary_metric",
                    "full_score",
                    "pruned_score",
                    "delta_score",
                    "full_exact",
                    "pruned_exact",
                    "full_anls",
                    "pruned_anls",
                    "keep_ratio",
                    "selector_question_tokens",
                    "note",
                ],
            ),
            "",
            "## Table 8b: Open OCR/Document QA Stress Subgroups",
            "",
            table_md(
                tables["open_ocr_qa_stress"],
                [
                    "task",
                    "ratio",
                    "stress_tag",
                    "n",
                    "full_score",
                    "pruned_score",
                    "delta_score",
                    "full_exact",
                    "pruned_exact",
                    "delta_exact",
                    "note",
                ],
            ),
            "",
            "## Table 8c: Open QA Stress Manifest Summary",
            "",
            table_md(tables["open_ocr_qa_stress_manifest"], ["scope", "group", "count"]),
            "",
            "## Table 8d: Open QA Annotation Pack Summary",
            "",
            table_md(tables["open_ocr_qa_annotation_pack"], ["scope", "group", "count"]),
            "",
            "## Table 8d2: Manual Evidence Readiness Gates",
            "",
            table_md(tables["manual_evidence_readiness"], ["gate", "status", "requirement", "evidence"]),
            "",
            "## Table 8d3: Manual Final Package Status",
            "",
            table_md(tables["manual_final_package_status"], ["gate", "status", "evidence"]),
            "",
            "## Table 8d4: Open QA Evidence Source Boundary",
            "",
            table_md(
                tables["open_ocr_qa_evidence_source_boundary"],
                [
                    "source",
                    "task_scope",
                    "source_type",
                    "evidence_scope",
                    "samples_with_boxes",
                    "coverage_rate",
                    "total_boxes",
                    "multi_box_sample_rate",
                    "mean_ecr_0p70",
                    "mean_worst_region_ecr_0p70",
                    "all_regions_ge_0p50_rate_0p70",
                    "readiness",
                    "claim_use",
                    "caveat",
                ],
            ),
            "",
            "## Table 8e: Open QA Evidence Prefill Summary",
            "",
            table_md(tables["open_ocr_qa_evidence_prefill"], ["scope", "metric", "value"]),
            "",
            "## Table 8f: Open QA BBox Annotation Tool Summary",
            "",
            table_md(tables["open_ocr_qa_bbox_tool"], ["scope", "metric", "value"]),
            "",
            "## Table 8g: Open QA BBox Annotation Seed Validation",
            "",
            table_md(tables["open_ocr_qa_bbox_validation"], ["scope", "metric", "value"]),
            "",
            "## Table 8h: Open QA BBox ECR Evaluator Summary",
            "",
            table_md(tables["open_ocr_qa_bbox_ecr"], ["scope", "metric", "value"]),
            "",
            "## Table 8i: Open QA External BBox Adapter Summary",
            "",
            table_md(tables["open_ocr_qa_external_bbox_adapter"], ["scope", "metric", "value"]),
            "",
            "## Table 8j: TextVQA GT BBox Match Summary",
            "",
            table_md(tables["open_ocr_qa_textvqa_gt_bbox"], ["scope", "metric", "value"]),
            "",
            "## Table 8k: TextVQA GT BBox ECR Summary",
            "",
            table_md(tables["open_ocr_qa_textvqa_gt_bbox_ecr"], ["scope", "metric", "value"]),
            "",
            "## Table 8l: DocVQA Expanded OCR Answer-Token BBox Match Summary",
            "",
            table_md(tables["open_ocr_qa_docvqa_ocr_bbox"], ["scope", "metric", "value"]),
            "",
            "## Table 8m: DocVQA Expanded OCR Answer-Token BBox ECR Summary",
            "",
            table_md(tables["open_ocr_qa_docvqa_ocr_bbox_ecr"], ["scope", "metric", "value"]),
            "",
            "## Table 8n: DocVQA OCR Line-Context BBox Summary",
            "",
            table_md(tables["open_ocr_qa_docvqa_line_context"], ["scope", "metric", "value"]),
            "",
            "## Table 8o: DocVQA OCR Line-Context BBox ECR Summary",
            "",
            table_md(tables["open_ocr_qa_docvqa_line_context_ecr"], ["scope", "metric", "value"]),
            "",
            "## Table 8p: Open QA BBox Noise Sensitivity Summary",
            "",
            table_md(
                tables["open_ocr_qa_bbox_noise"],
                [
                    "variant",
                    "task",
                    "budget_keep_ratio",
                    "scored_rows",
                    "sample_has_box_rate",
                    "mean_box_retention_rate",
                    "mean_ECR",
                    "retention_adjusted_ECR",
                    "mean_all_regions_ECR_ge_0p50",
                    "sample_adjusted_all_regions_ECR_ge_0p50",
                ],
            ),
            "",
            "## Table 8p2: Hard Robustness Conflict Summary",
            "",
            table_md(
                tables["hard_robustness_summary"],
                ["category", "source", "key_result", "reviewer_reading", "claim_boundary", "status"],
            ),
            "",
            "## Table 8p3: Hard Robustness TextOCR Method Slices",
            "",
            table_md(
                tables["hard_robustness_textocr_method"],
                [
                    "method",
                    "slice",
                    "n",
                    "accuracy",
                    "hFPR",
                    "yes_rate",
                    "mean_ECR",
                    "mean_margin",
                ],
            ),
            "",
            "## Table 8p4: Hard Robustness Open-QA Stress Slices",
            "",
            table_md(
                tables["hard_robustness_openqa"],
                [
                    "task",
                    "ratio",
                    "stress_tag",
                    "n",
                    "full_score",
                    "pruned_score",
                    "delta_score",
                    "reading",
                ],
            ),
            "",
            "## Table 8p5: Hard Robustness Detector and Noisy-Box Slices",
            "",
            table_md(
                tables["hard_robustness_detector"],
                [
                    "source",
                    "scope",
                    "condition",
                    "n_or_rows",
                    "primary_metric",
                    "primary_value",
                    "secondary_metric",
                    "secondary_value",
                    "cost_or_latency",
                ],
            ),
            "",
            "## Table 8q: DocVQA Line-Context ECR/Quality Bucket Summary",
            "",
            table_md(
                tables["docvqa_line_context_quality_bucket"],
                [
                    "scope",
                    "group",
                    "n",
                    "mean_ECR",
                    "mean_worst_region_ECR",
                    "mean_pruned_score",
                    "mean_score_delta",
                    "pruned_good_rate",
                ],
            ),
            "",
            "## Table 8r: DocVQA Line-Context ECR/Quality Correlation Summary",
            "",
            table_md(
                tables["docvqa_line_context_quality_correlation"],
                ["scope", "x", "y", "n", "pearson", "spearman"],
            ),
            "",
            "## Table 8r2: DocVQA Document Evidence-Risk Budget Decomposition",
            "",
            table_md(
                tables["docvqa_document_risk_by_budget"],
                [
                    "budget",
                    "n",
                    "mean_box_count",
                    "mean_ECR",
                    "mean_worst_region_ECR",
                    "all_regions_pass_rate",
                    "pruned_good_rate",
                    "low_failure_rate_ge0p25",
                    "high_mean_ECR_but_bad_rate",
                    "all_regions_pass_but_bad_rate",
                ],
            ),
            "",
            "## Table 8r3: DocVQA Document Evidence-Risk Condition Decomposition",
            "",
            table_md(
                tables["docvqa_document_risk_by_condition"],
                [
                    "budget",
                    "condition",
                    "n",
                    "fraction",
                    "mean_ECR",
                    "mean_worst_region_ECR",
                    "mean_pruned_score",
                    "pruned_good_rate",
                    "low_failure_rate_ge0p25",
                ],
            ),
            "",
            "## Table 8r4: DocVQA Document Evidence-Risk Trajectory Decomposition",
            "",
            table_md(
                tables["docvqa_document_risk_trajectory_summary"],
                [
                    "group",
                    "n",
                    "mean_box_count",
                    "mean_score30",
                    "mean_score70",
                    "mean_ECR70",
                    "mean_worst70",
                    "all_pass70_rate",
                    "repaired70_rate",
                    "persistent70_rate",
                    "high_ECR70_bad_rate",
                    "all_pass70_bad_rate",
                ],
            ),
            "",
            "## Table 8r: Open QA External BBox Quality Audit",
            "",
            table_md(tables["open_ocr_qa_bbox_quality"], ["scope", "metric", "value"]),
            "",
            "## Table 8s: Open QA Adaptive Budget Summary",
            "",
            table_md(
                tables["open_ocr_qa_adaptive_budget"],
                [
                    "task",
                    "split",
                    "policy",
                    "family",
                    "selected_by",
                    "n",
                    "score",
                    "exact",
                    "mean_keep",
                    "fallback_rate_ge_0p50",
                    "full_fallback_rate",
                    "note",
                ],
            ),
            "",
            "## Table 8t: Open QA Adaptive Budget Policy Selection",
            "",
            table_md(
                tables["open_ocr_qa_adaptive_budget_selection"],
                [
                    "task",
                    "family",
                    "selected_policy",
                    "dev_score",
                    "dev_exact",
                    "dev_mean_keep",
                    "dev_fallback_rate",
                    "note",
                ],
            ),
            "",
            "## Table 8u: Open QA Oracle Risk-Coverage Frontier",
            "",
            table_md(
                tables["open_ocr_qa_risk_coverage_frontier"],
                [
                    "task",
                    "split",
                    "tolerance",
                    "n",
                    "oracle_score",
                    "full_score",
                    "score_delta_vs_full",
                    "mean_keep",
                    "escalate_rate_gt_0p30",
                    "fallback_rate_ge_0p70",
                    "full_fallback_rate",
                    "choose_0p30",
                    "choose_0p50",
                    "choose_0p70",
                    "choose_1p00",
                ],
            ),
            "",
            "## Table 8v: Open QA Learned Risk Policy Summary",
            "",
            table_md(
                tables["open_ocr_qa_learned_risk_policy"],
                [
                    "task",
                    "split",
                    "policy",
                    "family",
                    "selected_by",
                    "score",
                    "exact",
                    "mean_keep",
                    "fallback_rate_ge_0p70",
                    "full_fallback_rate",
                    "note",
                ],
            ),
            "",
            "## Table 8w: Open QA Learned Risk Policy Selection",
            "",
            table_md(
                tables["open_ocr_qa_learned_risk_selection"],
                [
                    "task",
                    "family",
                    "selected_policy",
                    "dev_score",
                    "dev_mean_keep",
                    "dev_full_fallback_rate",
                    "note",
                ],
            ),
            "",
            "## Table 8x: Open QA Learned Risk Model Weights",
            "",
            table_md(
                tables["open_ocr_qa_learned_risk_model_weights"],
                [
                    "task",
                    "label",
                    "dev_auc",
                    "top_positive_features",
                    "top_negative_features",
                ],
            ),
            "",
            "## Table 8x2: Open QA Answer-Stability Cascade Summary",
            "",
            table_md(
                tables["open_ocr_qa_answer_stability_cascade"],
                [
                    "task",
                    "split",
                    "policy",
                    "family",
                    "score",
                    "full_score",
                    "delta_vs_full",
                    "selected_mean_keep",
                    "cascade_mean_cost",
                    "fallback_rate_ge_0p70",
                    "full_fallback_rate",
                    "answer30_50_agreement_rate",
                    "note",
                ],
            ),
            "",
            "## Table 8x3: Open QA Answer-Stability Dev-Selected Policies",
            "",
            table_md(
                tables["open_ocr_qa_answer_stability_selection"],
                [
                    "task",
                    "family",
                    "selected_policy",
                    "dev_score",
                    "dev_selected_keep",
                    "dev_cascade_cost",
                    "test_score",
                    "test_selected_keep",
                    "test_cascade_cost",
                    "test_full_fallback_rate",
                    "note",
                ],
            ),
            "",
            "## Table 8x4: Open QA Answer-Stability Signal Summary",
            "",
            table_md(
                tables["open_ocr_qa_answer_stability_signal"],
                [
                    "task",
                    "split",
                    "group",
                    "n",
                    "fraction",
                    "mean_score30",
                    "mean_score70",
                    "mean_drop30",
                    "low_failure_rate_ge0p25",
                    "safe30_within0p10_rate",
                    "safe70_within0p10_rate",
                    "repair70_rate_among_low_fail",
                    "note",
                ],
            ),
            "",
            "## Table 8x5: Open QA Pre-Generation Risk Signal Models",
            "",
            table_md(
                tables["open_ocr_qa_pregen_risk_signal_model"],
                [
                    "task",
                    "target",
                    "feature_group",
                    "split",
                    "n",
                    "positive_rate",
                    "auc",
                    "top20_positive_rate",
                    "bottom20_positive_rate",
                    "note",
                ],
            ),
            "",
            "## Table 8x6: Open QA Pre-Generation 30-to-70 Policy",
            "",
            table_md(
                tables["open_ocr_qa_pregen_risk_signal_policy"],
                [
                    "task",
                    "feature_group",
                    "selected_threshold",
                    "test_score",
                    "test_mean_keep",
                    "test_fallback70_rate",
                    "test_delta_vs_fixed30",
                    "test_delta_vs_fixed70",
                    "note",
                ],
            ),
            "",
            "## Table 8x7: Open QA Adaptive Deployment Contract",
            "",
            table_md(
                tables["open_ocr_qa_deployment_contract_summary"],
                [
                    "task",
                    "summary_item",
                    "policy",
                    "test_score",
                    "deployment_cost_proxy",
                    "interpretation",
                ],
            ),
            "",
            "## Table 8x8: Open QA Domain-Aware Portfolio Diagnostic",
            "",
            table_md(
                tables["domain_aware_portfolio_decision"],
                [
                    "portfolio_status",
                    "aggregate_score",
                    "fixed70_aggregate_score",
                    "delta_score_vs_fixed70",
                    "aggregate_cost",
                    "fixed70_aggregate_cost",
                    "delta_cost_vs_fixed70",
                    "fallback_tasks",
                    "safe_claim",
                ],
            ),
            "",
            table_md(
                tables["domain_aware_portfolio_rows"],
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
            "## Table 8x9: Open QA Unified Policy Transfer Summary",
            "",
            table_md(
                tables["open_ocr_qa_unified_policy_transfer_summary"],
                [
                    "scope",
                    "best_quality_policy",
                    "best_quality_target_score",
                    "best_quality_target_keep",
                    "best_quality_delta_vs_fixed70",
                    "near_fixed70_lower_keep_candidates",
                    "reading",
                ],
            ),
            "",
            "## Table 8x10: Open QA Cross-Task Policy Transfer",
            "",
            table_md(
                tables["open_ocr_qa_unified_policy_cross_task"],
                [
                    "source_task",
                    "target_task",
                    "feature_group",
                    "source_test_score",
                    "source_test_keep",
                    "source_delta_vs_fixed70",
                    "target_test_score",
                    "target_test_keep",
                    "target_delta_vs_fixed70",
                    "target_passes_near_fixed70_lower_keep",
                ],
            ),
            "",
            "## Table 8x11: Open QA Pooled Policy Transfer",
            "",
            table_md(
                tables["open_ocr_qa_unified_policy_pooled"],
                [
                    "task",
                    "feature_group",
                    "test_score",
                    "test_keep",
                    "test_fallback70_rate",
                    "delta_vs_fixed70",
                    "passes_near_fixed70_lower_keep",
                    "decision_features",
                ],
            ),
            "",
            "## Table 8x12: Open QA Detector-Aware Adaptive Policy",
            "",
            table_md(
                tables["open_ocr_qa_detector_aware_policy_summary"],
                [
                    "task",
                    "selected_policy",
                    "feature",
                    "threshold",
                    "test_score",
                    "test_mean_keep",
                    "test_delta_vs_fixed70",
                    "test_passes_near_fixed70_lower_keep",
                    "mean_detector_ms",
                    "reading",
                ],
            ),
            "",
            "## Table 8y: Open QA Repairability Summary",
            "",
            table_md(
                tables["open_ocr_qa_repairability_summary"],
                [
                    "task",
                    "n",
                    "low_failure_rate",
                    "repaired_by_70_rate_among_low_fail",
                    "high_still_bad_rate_among_low_fail",
                    "mean_low_drop",
                    "mean_gain_30_to_70_low_fail",
                ],
            ),
            "",
            "## Table 8z: Open QA Repairability Feature Summary",
            "",
            table_md(
                tables["open_ocr_qa_repairability_feature"],
                [
                    "target",
                    "scope",
                    "feature_group",
                    "feature",
                    "n",
                    "positive_rate",
                    "auc_best_direction",
                    "direction",
                    "spearman_with_target",
                ],
            ),
            "",
            "## Table 8aa: Open QA Box-Aware Budget Summary",
            "",
            table_md(
                tables["open_ocr_qa_box_aware_budget"],
                [
                    "scope",
                    "split",
                    "policy",
                    "family",
                    "selected_by",
                    "n",
                    "score",
                    "mean_keep",
                    "delta_vs_full",
                    "fallback_rate_ge_0p70",
                    "full_fallback_rate",
                    "note",
                ],
            ),
            "",
            "## Table 8ab: Open QA Box-Aware Budget Policy Selection",
            "",
            table_md(
                tables["open_ocr_qa_box_aware_budget_selection"],
                [
                    "scope",
                    "family",
                    "selected_policy",
                    "dev_score",
                    "dev_mean_keep",
                    "dev_delta_vs_full",
                    "dev_fallback_rate_ge_0p70",
                    "note",
                ],
            ),
            "",
            "## Table 8ac: Open QA Noisy-Box Fallback Key Summary",
            "",
            table_md(
                tables["open_ocr_qa_noisy_box_fallback_key"],
                [
                    "scope",
                    "split",
                    "policy",
                    "family",
                    "score",
                    "mean_keep",
                    "delta_vs_full",
                    "missing_box_rate",
                    "fallback_rate_ge_0p70",
                    "full_fallback_rate",
                    "note",
                ],
            ),
            "",
            "## Table 8ad: Open QA Noisy-Box Latency Key Summary",
            "",
            table_md(
                tables["open_ocr_qa_noisy_box_latency_key"],
                [
                    "scope",
                    "policy",
                    "family",
                    "score",
                    "mean_keep",
                    "requires_detector",
                    "estimated_no_detector_ttft_ms",
                    "estimated_detector_mean_ttft_ms",
                    "speedup_no_detector",
                    "speedup_with_mean_detector",
                    "estimate_note",
                ],
            ),
            "",
            "## Table 8ae: Open QA ECR/Quality Bucket Summary",
            "",
            table_md(
                tables["open_ocr_qa_ecr_quality_bucket"],
                [
                    "scope",
                    "group",
                    "n",
                    "mean_ECR",
                    "mean_pruned_score",
                    "mean_score_delta",
                    "mean_score_drop",
                    "pruned_good_rate",
                ],
            ),
            "",
            "## Table 8af: Open QA ECR/Quality Correlation Summary",
            "",
            table_md(
                tables["open_ocr_qa_ecr_quality_correlation"],
                ["scope", "x", "y", "n", "pearson", "spearman"],
            ),
            "",
            "## Table 9: Key Paired Statistics",
            "",
            table_md(
                tables["statistics"],
                ["comparison", "n", "left", "right", "acc_diff", "acc_CI", "acc_p", "hFPR_diff", "hFPR_CI", "hFPR_p"],
            ),
            "",
            "## Table 10: Image-Cluster Paired Statistics",
            "",
            table_md(
                tables["image_cluster_statistics"],
                [
                    "comparison",
                    "claim",
                    "model",
                    "left",
                    "right",
                    "n_probes",
                    "n_images",
                    "acc_diff",
                    "acc_cluster_ci",
                    "hFPR_diff",
                    "hFPR_cluster_ci",
                    "ECR_diff",
                    "ECR_cluster_ci",
                    "left_mean_keep",
                ],
            ),
            "",
            "## Table 11: Real/Noisy Box-Source Robustness",
            "",
            table_md(
                tables["box_source_robustness"],
                [
                    "model",
                    "box_source",
                    "selector",
                    "n",
                    "acc",
                    "hFPR",
                    "keep_ratio",
                    "true_ECR",
                    "true_CenterR",
                    "selector_box_ECR",
                    "note",
                ],
            ),
            "",
            "## Table 12: Region Logit-Drop Diagnostics",
            "",
            table_md(
                tables["region_logit_drop"],
                [
                    "model",
                    "comparison",
                    "n",
                    "left",
                    "right",
                    "left_acc",
                    "right_acc",
                    "left_hFPR",
                    "right_hFPR",
                    "support_diff",
                    "support_CI",
                    "support_p",
                    "pos_support_diff",
                    "pos_support_CI",
                    "neg_support_diff",
                    "neg_support_CI",
                    "correct_flip_L_R",
                    "hFP_flip_L_R",
                ],
            ),
            "",
            "## Table 13: Input-Space BBox Occlusion Diagnostics",
            "",
            table_md(
                tables["bbox_occlusion"],
                [
                    "model",
                    "comparison",
                    "n",
                    "left",
                    "right",
                    "left_acc",
                    "right_acc",
                    "left_hFPR",
                    "right_hFPR",
                    "support_diff",
                    "support_CI",
                    "support_p",
                    "pos_support_diff",
                    "pos_support_CI",
                    "neg_support_diff",
                    "neg_support_CI",
                    "correct_flip_L_R",
                    "hFP_flip_L_R",
                ],
            ),
            "",
            "## Table 14: Qwen Deletion-Restoration Causal Audit",
            "",
            table_md(
                tables["deletion_restoration"],
                [
                    "variant",
                    "n",
                    "acc",
                    "hFPR",
                    "keep_ratio",
                    "ECR",
                    "delta_acc_vs_removed",
                    "acc_recovery_pct",
                ],
            ),
            "",
            "## Table 15: InternVL Calibrated Deletion-Restoration Causal Audit",
            "",
            table_md(
                tables["internvl_deletion_restoration"],
                [
                    "variant",
                    "n",
                    "acc",
                    "hFPR",
                    "keep_ratio",
                    "ECR",
                    "delta_acc_vs_removed",
                    "acc_recovery_pct",
                ],
            ),
            "",
            "## Table 16: Causal Evidence Triad Summary",
            "",
            table_md(
                tables["causal_evidence_triad"],
                [
                    "model",
                    "category",
                    "primary_metric",
                    "primary_effect",
                    "control_metric",
                    "control_effect",
                    "supporting_metric",
                    "supporting_effect",
                    "strength",
                    "caveat",
                ],
            ),
            "",
            "## Table 16b: Semantic Text-Replacement Counterfactual",
            "",
            table_md(
                tables["text_replacement_counterfactual"],
                [
                    "split",
                    "n_pairs",
                    "original_pair_correct_rate",
                    "edited_pair_correct_rate",
                    "full_four_way_semantic_switch_rate",
                    "edited_pair_correct_given_original_correct",
                    "source_absence_switch_rate",
                    "replacement_presence_switch_rate",
                    "mean_source_yes_support_drop",
                    "mean_replacement_yes_support_gain",
                    "note",
                ],
            ),
            "",
            "## Table 16c: Text-Replacement OCR Quality Audit",
            "",
            table_md(
                tables["text_replacement_ocr_quality"],
                [
                    "split",
                    "n_pairs",
                    "source_original_detected_crop_rate",
                    "edited_crop_replacement_detected_rate",
                    "edited_crop_source_absent_rate",
                    "edited_crop_ocr_success_rate",
                    "edited_full_replacement_detected_rate",
                    "edited_full_source_absent_rate",
                    "edited_full_ocr_success_rate",
                    "note",
                ],
            ),
            "",
            "## Table 16d: Text-Replacement OCR-Conditioned Behavior",
            "",
            table_md(
                tables["text_replacement_stratified"],
                [
                    "split",
                    "group",
                    "n_pairs",
                    "full_four_way_semantic_switch_rate",
                    "edited_pair_correct_given_original_correct",
                    "source_absence_switch_rate",
                    "replacement_presence_switch_rate",
                    "mean_source_yes_support_drop",
                    "mean_replacement_yes_support_gain",
                    "ocr_success_rate",
                    "note",
                ],
            ),
            "",
            "## Table 16e: Text-Replacement Human QC Launch Summary",
            "",
            table_md(
                tables["text_replacement_human_qc_launch"],
                [
                    "metric",
                    "value",
                ],
            ),
            "",
            "## Table 16f: Text-Replacement Human QC Progress Decision",
            "",
            table_md(
                tables["text_replacement_human_qc_progress"],
                [
                    "text_replacement_human_qc_status",
                    "rows",
                    "ready_rows",
                    "blocker_rows",
                    "valid_semantic_edit_rows",
                    "valid_semantic_edit_rate",
                    "safe_claim",
                ],
            ),
            "",
            "## Table 16f2: Human-QC Claim Gate",
            "",
            table_md(tables["human_qc_claim_gate_decision"], ["human_qc_claim_status", "text_replacement_ready", "hard_negative_ready", "stale_tables_absent", "safe_claim"]),
            "",
            table_md(tables["human_qc_claim_gate_rows"], ["gate", "status", "evidence"]),
            "",
            "## Table 16g: Causal Evidence Go/No-Go Decision",
            "",
            table_md(
                tables["causal_evidence_decision"],
                [
                    "causal_claim_status",
                    "qwen_strong",
                    "internvl_strong",
                    "llava_weak_boundary",
                    "semantic_counterfactual_strong",
                    "recommended_claim",
                    "avoid_claim",
                ],
            ),
            "",
            "## Table 16h: Causal Evidence Go/No-Go Gates",
            "",
            table_md(
                tables["causal_evidence_go_no_go"],
                [
                    "gate",
                    "scope",
                    "status",
                    "strong_rows",
                    "mixed_rows",
                    "weak_rows",
                    "evidence",
                    "safe_claim",
                ],
            ),
            "",
            "## Table 17: Hard-Negative Construction Quality",
            "",
            table_md(tables["hard_negative_quality"], ["metric", "value"]),
            "",
            "## Table 17b: Hard-Negative Lexical Audit Summary",
            "",
            table_md(tables["hard_negative_lexical"], ["metric", "value"]),
            "",
            "## Table 17c: Hard-Negative Edit And Shape Summary",
            "",
            table_md(tables["hard_negative_edit_class"], ["group", "bucket", "count", "rate"]),
            "",
            "## Table 17d: Hard-Negative Lexical Suspicious Examples",
            "",
            table_md(
                tables["hard_negative_suspicious"][:50],
                [
                    "sample_id",
                    "image_id",
                    "source_text",
                    "target_text",
                    "reason",
                    "edit_distance",
                    "edit_type",
                    "same_image_collision_normalizers",
                    "source_target_same_normalizers",
                ],
            ),
            "",
            "## Table 17e: Hard-Negative Human QC Launch Summary",
            "",
            table_md(tables["hard_negative_human_qc_launch"], ["metric", "value"]),
            "",
            "## Table 17f: Hard-Negative Human QC Progress Decision",
            "",
            table_md(
                tables["hard_negative_human_qc_progress"],
                [
                    "hard_negative_human_qc_status",
                    "rows",
                    "ready_rows",
                    "blocker_rows",
                    "valid_negative_rows",
                    "invalid_rows",
                    "unclear_rows",
                    "safe_claim",
                ],
            ),
            "",
            "## Table 18: Random Baseline Seed Status",
            "",
            table_md(
                tables["random_seed_status"],
                [
                    "run",
                    "cached_random_realizations",
                    "n",
                    "acc_mean",
                    "acc_std",
                    "hFPR_mean",
                    "hFPR_std",
                    "keep_mean",
                    "ECR_mean",
                    "ECR_std",
                    "status",
                ],
            ),
            "",
            "## Table 19: InternVL Soft-Evidence Operating Points",
            "",
            table_md(tables["internvl_operating_points"], ["run", "method", "n", "acc", "hFPR", "keep_ratio", "ECR", "note"]),
            "",
            "## Table 20: InternVL Threshold Calibration Summary",
            "",
            table_md(
                tables["internvl_calibration_summary"],
                [
                    "model_run",
                    "default_all_hFPR",
                    "default_all_yes_rate",
                    "dev_best_threshold",
                    "dev_best_test_acc",
                    "dev_best_test_hFPR",
                    "all_best_threshold",
                    "all_best_all_acc",
                    "all_best_all_hFPR",
                ],
            ),
            "",
            "## Table 21: InternVL Soft Operating-Point Audit",
            "",
            table_md(
                tables["internvl_soft_operating_audit"],
                [
                    "name",
                    "yesno_threshold",
                    "selection_protocol",
                    "selector",
                    "acc",
                    "hFPR",
                    "keep_ratio",
                    "ECR",
                    "path",
                ],
            ),
            "",
            "## Table 22: InternVL Soft Pair Identity Check",
            "",
            table_md(
                tables["internvl_soft_pair_identity"],
                [
                    "comparison",
                    "shared_rows",
                    "raw_margin_mismatches",
                    "ECR_mismatches",
                    "selector_mismatches",
                    "pred_changed",
                    "yes_to_no",
                    "no_to_yes",
                    "interpretation",
                ],
            ),
            "",
            "## Table 23: Adaptive Risk Policy Search",
            "",
            table_md(
                tables["adaptive_policy"],
                [
                    "selection",
                    "policy",
                    "feature_source",
                    "dev_acc",
                    "dev_hFPR",
                    "dev_keep",
                    "test_acc",
                    "test_hFPR",
                    "test_keep",
                    "test_cascade",
                    "test_ECR",
                    "note",
                ],
            ),
            "",
            "## Table 23b: Calibrated Risk-Control Fallback Audit",
            "",
            table_md(
                tables["conformal_risk_decision"],
                [
                    "conformal_policy_status",
                    "epsilon_for_main_gate",
                    "go_tasks",
                    "required_tasks",
                    "recommended_claim",
                ],
            ),
            "",
            table_md(
                tables["conformal_risk_selection"],
                [
                    "task",
                    "epsilon",
                    "selection_status",
                    "threshold",
                    "dev_loss_upper95",
                    "dev_serial_cost",
                    "test_delta_vs_fixed70",
                    "test_fallback_rate",
                    "test_serial_cost",
                    "test_go_under_contract",
                    "reading",
                ],
            ),
            "",
            "## Adaptive Controller Go/No-Go Audit",
            "",
            table_md(
                tables["adaptive_controller_decision"],
                [
                    "main_controller_status",
                    "failed_required_gates",
                    "partial_gates",
                    "recommended_claim",
                    "next_design_requirement",
                ],
            ),
            "",
            table_md(
                tables["adaptive_controller_go_no_go"],
                [
                    "gate",
                    "required_for_main_controller",
                    "status",
                    "best_candidate",
                    "evidence",
                    "reading",
                ],
            ),
            "",
            "## Problem.md Strong-Accept Upgrade Readiness",
            "",
            table_md(
                tables["strong_accept_decision"],
                [
                    "strong_accept_upgrade_status",
                    "pass_gates",
                    "partial_gates",
                    "fail_gates",
                    "recommended_next_step",
                ],
            ),
            "",
            table_md(
                tables["strong_accept_readiness"],
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
            "## Problem.md Claim Boundaries",
            "",
            "This table records the safest manuscript boundary for each major `problem.md` concern. "
            "Allowed claims still require the listed scope limits; forbidden claims should not appear unless new evidence is added.",
            "",
            table_md(
                tables["problem_md_claim_boundaries"],
                [
                    "boundary_id",
                    "problem_md_link",
                    "allowed_claim",
                    "forbidden_claim",
                    "current_status",
                ],
            ),
            "",
            "## Problem.md Remaining-Blocker Dashboard",
            "",
            "This table converts each major `problem.md` risk into a current manuscript stance, a remaining blocker, and a concrete claim boundary.",
            "",
            table_md(
                tables["problem_md_remaining_blockers"],
                [
                    "gate",
                    "problem_md_link",
                    "current_status",
                    "paper_status",
                    "remaining_blocker",
                    "key_numbers",
                    "claim_boundary",
                ],
            ),
            "",
            "## Unsupported Or Negative Claims",
            "",
            table_md(tables["unsupported"], ["claim", "status", "reason", "safe_wording"]),
            "",
            "## Claim-Language Guardrail",
            "",
            "The manuscript language is separately checked against the current `problem.md` evidence boundary. "
            "This guardrail scans for overclaims around ECR-as-causality, solved adaptive control, detector-assisted speed scope, "
            "pending manual multi-region annotation, all-backbone external-baseline parity, and broad open-QA/leaderboard claims. "
            f"The latest report is `{CLAIM_LANGUAGE_AUDIT_MD.relative_to(ROOT)}`.",
            "",
            "## Problem.md Manuscript Coverage Audit",
            "",
            "A separate coverage audit checks that each major `problem.md` concern is visible in the manuscript and tied to paper-evidence artifacts. "
            "It covers causal evidence, native open tasks, multi-evidence document context, adaptive-control boundaries, method principle, external-baseline scope, efficiency scope, statistics/data quality, and hard robustness. "
            f"The latest report is `{PROBLEM_MD_MANUSCRIPT_COVERAGE_MD.relative_to(ROOT)}`.",
            "",
            "## Source Files",
            "",
            f"- `{CLAIM_LANGUAGE_AUDIT_MD.relative_to(ROOT)}`",
            f"- `{CLAIM_LANGUAGE_AUDIT_CSV.relative_to(ROOT)}`",
            f"- `{PROBLEM_MD_MANUSCRIPT_COVERAGE_MD.relative_to(ROOT)}`",
            f"- `{PROBLEM_MD_MANUSCRIPT_COVERAGE_CSV.relative_to(ROOT)}`",
            f"- `{ADAPTIVE_CONTROLLER_GO_NO_GO_MD.relative_to(ROOT)}`",
            f"- `{ADAPTIVE_CONTROLLER_GO_NO_GO_CSV.relative_to(ROOT)}`",
            f"- `{ADAPTIVE_CONTROLLER_DECISION_CSV.relative_to(ROOT)}`",
            f"- `{CAUSAL_EVIDENCE_GO_NO_GO_MD.relative_to(ROOT)}`",
            f"- `{CAUSAL_EVIDENCE_GO_NO_GO_CSV.relative_to(ROOT)}`",
            f"- `{CAUSAL_EVIDENCE_DECISION_CSV.relative_to(ROOT)}`",
            f"- `{STRONG_ACCEPT_READINESS_MD.relative_to(ROOT)}`",
            f"- `{STRONG_ACCEPT_READINESS_CSV.relative_to(ROOT)}`",
            f"- `{STRONG_ACCEPT_DECISION_CSV.relative_to(ROOT)}`",
            f"- `{PROBLEM_MD_CLAIM_BOUNDARY_CSV.relative_to(ROOT)}`",
            f"- `{PROBLEM_MD_REMAINING_BLOCKERS_CSV.relative_to(ROOT)}`",
            f"- `{CROSS_CSV.relative_to(ROOT)}`",
            f"- `{HARD_CSV.relative_to(ROOT)}`",
            f"- `{P0_CSV.relative_to(ROOT)}`",
            f"- `{EFF_CSV.relative_to(ROOT)}`",
            f"- `{EFF_DECOMP_CSV.relative_to(ROOT)}`",
            f"- `{OCR_CSV.relative_to(ROOT)}`",
            f"- `{OPEN_ANSWER_METRICS.relative_to(ROOT)}`",
            *open_generation_source_lines(),
            *open_ocr_qa_source_lines(),
            f"- `{OPEN_OCR_QA_STRESS_CSV.relative_to(ROOT)}`",
            f"- `{OPEN_OCR_QA_STRESS_MANIFEST_SUMMARY_CSV.relative_to(ROOT)}`",
            f"- `{OPEN_OCR_QA_ANNOTATION_PACK_SUMMARY_CSV.relative_to(ROOT)}`",
            f"- `{OPEN_OCR_QA_EVIDENCE_PREFILL_SUMMARY_CSV.relative_to(ROOT)}`",
            f"- `{OPEN_OCR_QA_BBOX_TOOL_SUMMARY_CSV.relative_to(ROOT)}`",
            f"- `{OPEN_OCR_QA_BBOX_VALIDATION_SUMMARY_CSV.relative_to(ROOT)}`",
            f"- `{OPEN_OCR_QA_BBOX_ECR_SUMMARY_CSV.relative_to(ROOT)}`",
            f"- `{OPEN_OCR_QA_EXTERNAL_BBOX_ADAPTER_SUMMARY_CSV.relative_to(ROOT)}`",
            f"- `{OPEN_OCR_QA_TEXTVQA_GT_BBOX_SUMMARY_CSV.relative_to(ROOT)}`",
            f"- `{OPEN_OCR_QA_TEXTVQA_GT_BBOX_ECR_SUMMARY_CSV.relative_to(ROOT)}`",
            f"- `{OPEN_OCR_QA_DOCVQA_OCR_BBOX_SUMMARY_CSV.relative_to(ROOT)}`",
            f"- `{OPEN_OCR_QA_DOCVQA_OCR_BBOX_ECR_SUMMARY_CSV.relative_to(ROOT)}`",
            f"- `{OPEN_OCR_QA_BBOX_QUALITY_SUMMARY_CSV.relative_to(ROOT)}`",
            f"- `{OPEN_OCR_QA_ADAPTIVE_BUDGET_SUMMARY_CSV.relative_to(ROOT)}`",
            f"- `{OPEN_OCR_QA_ADAPTIVE_BUDGET_SELECTION_CSV.relative_to(ROOT)}`",
            f"- `{OPEN_OCR_QA_RISK_COVERAGE_FRONTIER_CSV.relative_to(ROOT)}`",
            f"- `{OPEN_OCR_QA_LEARNED_RISK_POLICY_SUMMARY_CSV.relative_to(ROOT)}`",
            f"- `{OPEN_OCR_QA_LEARNED_RISK_POLICY_SELECTION_CSV.relative_to(ROOT)}`",
            f"- `{OPEN_OCR_QA_LEARNED_RISK_MODEL_WEIGHTS_CSV.relative_to(ROOT)}`",
            f"- `{OPEN_OCR_QA_ANSWER_STABILITY_CASCADE_SUMMARY_CSV.relative_to(ROOT)}`",
            f"- `{OPEN_OCR_QA_ANSWER_STABILITY_CASCADE_SELECTION_CSV.relative_to(ROOT)}`",
            f"- `{OPEN_OCR_QA_ANSWER_STABILITY_SIGNAL_CSV.relative_to(ROOT)}`",
            f"- `{OPEN_OCR_QA_PREGEN_RISK_SIGNAL_MODEL_CSV.relative_to(ROOT)}`",
            f"- `{OPEN_OCR_QA_PREGEN_RISK_SIGNAL_POLICY_CSV.relative_to(ROOT)}`",
            f"- `{OPEN_OCR_QA_DEPLOYMENT_CONTRACT_CSV.relative_to(ROOT)}`",
            f"- `{OPEN_OCR_QA_DEPLOYMENT_CONTRACT_SUMMARY_CSV.relative_to(ROOT)}`",
            f"- `{DOMAIN_AWARE_PORTFOLIO_ROWS_CSV.relative_to(ROOT)}`",
            f"- `{DOMAIN_AWARE_PORTFOLIO_DECISION_CSV.relative_to(ROOT)}`",
            f"- `{OPEN_OCR_QA_DETECTOR_AWARE_POLICY_SUMMARY_CSV.relative_to(ROOT)}`",
            f"- `{OPEN_OCR_QA_DETECTOR_AWARE_POLICY_READOUT_CSV.relative_to(ROOT)}`",
            f"- `{OPEN_OCR_QA_REPAIRABILITY_SUMMARY_CSV.relative_to(ROOT)}`",
            f"- `{OPEN_OCR_QA_REPAIRABILITY_FEATURE_CSV.relative_to(ROOT)}`",
            f"- `{OPEN_OCR_QA_BOX_AWARE_BUDGET_SUMMARY_CSV.relative_to(ROOT)}`",
            f"- `{OPEN_OCR_QA_BOX_AWARE_BUDGET_SELECTION_CSV.relative_to(ROOT)}`",
            f"- `{OPEN_OCR_QA_ECR_QUALITY_BUCKET_CSV.relative_to(ROOT)}`",
            f"- `{OPEN_OCR_QA_ECR_QUALITY_CORRELATION_CSV.relative_to(ROOT)}`",
            f"- `{OPEN_OCR_QA_DOCVQA_LINE_CONTEXT_SUMMARY_CSV.relative_to(ROOT)}`",
            f"- `{OPEN_OCR_QA_DOCVQA_LINE_CONTEXT_ECR_CSV.relative_to(ROOT)}`",
            f"- `{DOCVQA_LINE_CONTEXT_QUALITY_BUCKET_CSV.relative_to(ROOT)}`",
            f"- `{DOCVQA_LINE_CONTEXT_QUALITY_CORRELATION_CSV.relative_to(ROOT)}`",
            f"- `{METHOD_OBJECTIVE_MAPPING_CSV.relative_to(ROOT)}`",
            f"- `{COVERAGE_GREEDY_TRADEOFF_CSV.relative_to(ROOT)}`",
            f"- `{METHOD_COVERAGE_PAIRED_SUMMARY_CSV.relative_to(ROOT)}`",
            f"- `{METHOD_COVERAGE_PAIRED_GROUP_CSV.relative_to(ROOT)}`",
            f"- `{METHOD_COMPONENT_PARETO_CSV.relative_to(ROOT)}`",
            f"- `{METHOD_COMPONENT_DELTA_CSV.relative_to(ROOT)}`",
            f"- `{METHOD_COMPONENT_SUMMARY_CSV.relative_to(ROOT)}`",
            f"- `{EXTERNAL_BASELINE_FAIRNESS_MATRIX_CSV.relative_to(ROOT)}`",
            f"- `{EXTERNAL_BASELINE_BUDGET_CURVE_CSV.relative_to(ROOT)}`",
            f"- `{EXTERNAL_BASELINE_PAIRED_READOUT_CSV.relative_to(ROOT)}`",
            f"- `{NATIVE_EXTERNAL_PORT_FEASIBILITY_CSV.relative_to(ROOT)}`",
            f"- `{BOX_ROBUSTNESS_CSV.relative_to(ROOT)}`",
            f"- `{EASYOCR_SUMMARY_JSON.relative_to(ROOT)}`",
            f"- `{IMAGE_CLUSTER_STATS_CSV.relative_to(ROOT)}`",
            f"- `{HARD_NEGATIVE_QUALITY_CSV.relative_to(ROOT)}`",
            f"- `{HARD_NEGATIVE_LEXICAL_SUMMARY_CSV.relative_to(ROOT)}`",
            f"- `{HARD_NEGATIVE_EDIT_CLASS_CSV.relative_to(ROOT)}`",
            f"- `{HARD_NEGATIVE_SUSPICIOUS_CSV.relative_to(ROOT)}`",
            f"- `{HARD_NEGATIVE_HUMAN_QC_LAUNCH_CSV.relative_to(ROOT)}`",
            f"- `{HARD_NEGATIVE_HUMAN_QC_PROGRESS_CSV.relative_to(ROOT)}`",
            f"- `{RANDOM_SEED_STATUS_CSV.relative_to(ROOT)}`",
            f"- `{INTERNVL_OPERATING_POINTS_CSV.relative_to(ROOT)}`",
            f"- `{DELETION_RESTORATION_CSV.relative_to(ROOT)}`",
            f"- `{ADAPTIVE_POLICY_CONFIG.relative_to(ROOT)}`",
            f"- `{ADAPTIVE_POLICY_FIXED_CSV.relative_to(ROOT)}`",
            f"- `{ADAPTIVE_POLICY_CANDIDATES_CSV.relative_to(ROOT)}`",
            f"- `{MULTISIGNAL_ADAPTIVE_BEST_CSV.relative_to(ROOT)}`",
            f"- `{MULTISIGNAL_ADAPTIVE_CANDIDATES_CSV.relative_to(ROOT)}`",
            *occlusion_source_lines(),
            *region_source_lines(),
            f"- `{FUTURE_STATUS.relative_to(ROOT)}`",
            "",
        ]
    )
    return "\n".join(lines)


def region_source_lines() -> list[str]:
    out = []
    for _, summary_path, pairwise_path in REGION_RUNS:
        out.append(f"- `{summary_path.relative_to(ROOT)}`")
        out.append(f"- `{pairwise_path.relative_to(ROOT)}`")
    return out


def open_generation_source_lines() -> list[str]:
    return [f"- `{path.relative_to(ROOT)}`" for _, path in OPEN_GENERATION_METRICS]


def open_ocr_qa_source_lines() -> list[str]:
    return [f"- `{path.relative_to(ROOT)}`" for _, path in OPEN_OCR_QA_GENERATION_METRICS]


def occlusion_source_lines() -> list[str]:
    out = []
    for _, summary_path, pairwise_path in OCCLUSION_RUNS:
        out.append(f"- `{summary_path.relative_to(ROOT)}`")
        out.append(f"- `{pairwise_path.relative_to(ROOT)}`")
    return out


def table_md(rows: list[dict[str, Any]], columns: list[str]) -> str:
    out = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(fmt_cell(row.get(col, "")) for col in columns) + " |")
    return "\n".join(out)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def f(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def p(value: Any) -> str:
    if value is None or value == "":
        return ""
    value_f = float(value)
    if value_f < 1e-4:
        return "<1e-4"
    return f"{value_f:.4f}"


def interval(low: Any, high: Any) -> str:
    if low is None or high is None or low == "" or high == "":
        return ""
    return f"[{float(low):+.3f}, {float(high):+.3f}]"


def fmt_cell(value: Any) -> str:
    return str(value).replace("|", "/")


if __name__ == "__main__":
    main()
