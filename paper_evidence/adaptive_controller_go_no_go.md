# Adaptive Controller Go/No-Go Audit

This audit aggregates existing adaptive-budget, transfer, detector-aware, box-aware, noisy-box, and latency checks under one decision contract. It is intended to answer whether adaptive risk control is strong enough to become a main method claim.

## Decision

- Main-controller status: `no_go_for_main_method_claim`
- Failed required gates: 6
- Partial gates: 2
- Recommended claim: Keep adaptive control as a diagnostic boundary and future-work opportunity; do not claim a solved unified deployable controller.
- Next design requirement: A viable controller must be pre-generation or amortized, pass both TextVQA-lite and DocVQA-lite near-fixed70/lower-cost gates, and preserve detector-inclusive speed when boxes are required.

## Gate Table

| gate | required_for_main_controller | status | best_candidate | evidence | reading |
| --- | --- | --- | --- | --- | --- |
| deployment_contract_pregen | yes | fail | DocVQA-lite:pregen_question_only score=0.849 cost=0.642; TextVQA-lite:pregen_mask30_only score=0.830 cost=0.663 | passing_tasks=TextVQA-lite; passing_candidates=1 | TextVQA has one pre-generation lower-cost near-fixed70 candidate, but DocVQA has none. |
| domain_aware_portfolio | diagnostic | partial | aggregate score=0.860 cost=0.681 fallback=DocVQA-lite | status=partial_aggregate_portfolio_only; delta_score=-0.002; delta_cost=-0.018; all_tasks_lower_cost=0 | A task-aware portfolio slightly reduces aggregate cost but falls back to fixed70 for DocVQA, so it is not a solved per-task adaptive controller. |
| calibrated_conformal_risk_control | yes | fail | TextVQA-lite threshold=0.259 delta=-0.083 serial_cost=0.632 | status=no_go_for_conformal_controller_claim; go_tasks=none; epsilon=0.010 | A calibrated 30%->70% fallback using low-budget risk scores cannot certify fixed70-quality at lower serial cost across TextVQA-lite and DocVQA-lite. |
| pregen_evidence_saturation | yes | fail | TextVQA-lite score=0.834818 keep=0.699190; DocVQA-lite score=0.886157 keep=0.695984 | status=no_go_for_main_controller_claim; passed_tasks=0 of 2; target_max_keep=0.600000 | The pooled-dev saturation policy preserves fixed70-level held-out quality, but selects 70% for nearly every sample and therefore misses the efficiency gate. |
| cross_task_or_pooled_transfer | yes | fail | DocVQA-lite -> TextVQA-lite:mask30_only delta=-0.013 keep=0.637 | near_fixed70_lower_keep_candidates_total=0 | No cross-task or pooled pre-generation controller passes the near-fixed70/lower-keep gate. |
| detector_evidence_aware_stress_pack | yes | fail | pooled->TextVQA-lite:missing_or_low_ecr30_tofull delta=0.065 keep=0.785 | heldout_passes=0 of 4; mean_detector_ms_range=202.797-1190.805 | EasyOCR detector statistics and selector-mask evidence coverage still fail held-out stress-pack gates. |
| box_aware_annotated_subset | no | partial | answer_or_gt_bbox:all_regions_ECR_ge_0p50_lt_0p05_to_70 score=0.765 keep=0.684; answer_or_gt_bbox:worst_region_ECR_lt_0p50_to_70 score=0.765 keep=0.684; answer_or_gt_bbox:TextVQA-lite:ECR_lt_0p50_to_70 score=0.728 keep=0.684; answer_or_gt_bbox:TextVQA-lite:all_regions_ECR_ge_0p50_lt_0p05_to_70 score=0.728 keep=0.684 | passing_test_policies=5; answer_or_gt_bbox_passes=2; docvqa_line_context_passes=0 | Box-aware ECR fallback can match fixed70 on a scoped annotated subset, but it does not give a general controller and often relies on near-full fallback for document context. |
| detector_inclusive_latency | yes_for_box_aware | fail | mixed_light:missing_box_to_0p70 speedup_mean_detector=0.602 score=0.373 | detector_rows=6; no_detector_speedup_rows=6; detector_inclusive_speedup_rows=0 | Detector-assisted selected-keep savings do not translate into single-sample speedup when mean EasyOCR latency is counted. |
| oracle_headroom_vs_deployable_gap | diagnostic | headroom_exists | oracle=oracle_best_budget score=1.000 keep=0.654; deployable=full_prefix score=0.949 cost=1.000 | oracle_rows=7; deployable_rows=30 | There is real oracle headroom, so the negative result is about current deployable signals rather than impossibility in principle. |

## Source Tables

- `runs/paper_evidence/table_open_ocr_qa_deployment_contract.csv`
- `runs/paper_evidence/table_open_ocr_qa_deployment_contract_summary.csv`
- `runs/paper_evidence/table_open_ocr_qa_unified_policy_transfer_summary.csv`
- `runs/paper_evidence/table_open_ocr_qa_detector_aware_policy_summary.csv`
- `runs/paper_evidence/table_open_ocr_qa_box_aware_budget_summary.csv`
- `runs/paper_evidence/table_open_ocr_qa_noisy_box_latency_key_summary.csv`
- `runs/problem_optimization_audit/open_ocr_qa_evidence_saturation_policy/evidence_saturation_policy_decision.csv`
- `runs/problem_optimization_audit/open_ocr_qa_evidence_saturation_policy/evidence_saturation_policy_summary.csv`
