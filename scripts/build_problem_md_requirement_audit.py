#!/usr/bin/env python3
"""Requirement-level audit for the reviewer concerns in problem.md.

This script does not rerun experiments. It turns the strategic concerns in
`problem.md` into a concrete evidence ledger: what has been addressed, what is
only partially addressed, and what should be done next.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "problem_optimization_audit" / "problem_md_requirement_audit"


REQUIREMENTS: list[dict[str, Any]] = [
    {
        "id": "R1",
        "section": "Causal evidence",
        "reviewer_question": "Does retaining evidence-covered tokens mean the model actually uses the evidence?",
        "requested_evidence": "Necessity, sufficiency/restoration, specificity, and semantic counterfactuals.",
        "status": "mostly_addressed_with_scope_limit",
        "evidence_artifacts": "causal_evidence_triad; causal_evidence_go_no_go; Qwen and InternVL deletion-restoration; region logit-drop; bbox occlusion; text-replacement counterfactual audits; completed 106-row text-replacement human QC and 102-valid-edit model audit.",
        "key_reading": "The aggregate causal go/no-go audit returns no_go_for_full_causal_claim: Qwen and calibrated InternVL pass strong causal-style support gates, LLaVA is a weak-backbone boundary, and completed human QC supports a Qwen-specific semantic-switch diagnostic but not a backbone-uniform semantic-causal claim.",
        "remaining_gap": "ECR still measures availability, not full causal use. Among 102 human-validated edits, Qwen passes all four controls on 26 cases whereas InternVL does so on 3, and LLaVA does not show strong causal-style effects.",
        "paper_action": "Report necessity/restoration, specificity, and semantic counterfactuals separately. Use causal-style support language for Qwen/InternVL, treat LLaVA/text replacement as boundaries, and avoid saying ECR proves causal use.",
        "next_priority": "medium",
    },
    {
        "id": "R2",
        "section": "Native open tasks",
        "reviewer_question": "Are the conclusions only true on a constructed yes/no probe benchmark?",
        "requested_evidence": "Original open-answer OCR/document QA with native metrics, not only yes/no or ranking.",
        "status": "addressed_as_full_validation_boundary",
        "evidence_artifacts": "Full Qwen and LLaVA TextVQA validation (n=5,000) and DocVQA validation (n=5,349) generation; OCRBench open generation; lite-scale selector/evidence ablations; open OCR/document stress audit.",
        "key_reading": "Native generation is question-only for the selector. At 70% retention, Qwen drops by 0.0335 on TextVQA and 0.0933 ANLS on DocVQA; LLaVA drops by 0.1259 and 0.0538, respectively. The full-validation runs remove the scale objection while establishing a clear cross-model quality boundary rather than lossless transfer.",
        "remaining_gap": "Full-validation evidence does not establish lossless pruning, leaderboard-optimized performance, or universal open-document QA gains; the 96-row multi-region evidence audit remains a scoped subset.",
        "paper_action": "Use the full-validation results as benchmark-scale transfer and boundary evidence; retain lite-scale sets for controlled selector/evidence ablations and avoid lossless or universal open-QA claims.",
        "next_priority": "low",
    },
    {
        "id": "R3",
        "section": "Multi-evidence",
        "reviewer_question": "Can the method preserve multiple fields, table context, and worst-case evidence regions?",
        "requested_evidence": "Multi-box ECR, worst-region coverage, all-region pass rate, reading/context preservation.",
        "status": "mostly_addressed_with_scoped_human_audit",
        "evidence_artifacts": "TextVQA GT-box audit; DocVQA OCR answer-token audit; DocVQA line-context audit; 96-row final human-adjudicated multi-region annotations; validation, ECR, worst-region, all-region, and quality-association tables.",
        "key_reading": "The completed 96-sample audit contains 194 geometrically distinct evidence regions. At 70% Qwen Target+Grid retention, mean ECR is 0.729, worst-region ECR is 0.613, and the all-region pass rate is 0.729; at 30%, the corresponding values are 0.297, 0.216, and 0.105.",
        "remaining_gap": "The audit is a deliberately difficult 96-sample TextVQA/DocVQA subset rather than full-benchmark annotation, and ECR remains evidence availability rather than causal use.",
        "paper_action": "Report the final human-adjudicated audit with its subset and availability boundaries; retain external/OCR-derived rows as separate source comparisons.",
        "next_priority": "low",
    },
    {
        "id": "R4",
        "section": "Unified adaptive policy",
        "reviewer_question": "Is this a unified method, or hand-selected policies for Qwen, LLaVA, and InternVL?",
        "requested_evidence": "Adaptive keep ratio, adaptive evidence strength, selective fallback, and transfer to a held-out model.",
        "status": "not_solved_boundary_quantified",
        "evidence_artifacts": "TextOCR adaptive policies; oracle budget frontiers; open-QA risk-coverage frontier; learned risk policies; answer-stability cascade/signal audits; pre-generation risk-signal audit; deployment-contract audit; unified policy transfer audit; detector/evidence-aware policy audit; strict remaining-plan go/no-go audit.",
        "key_reading": "There is real oracle headroom and answer agreement is useful. Under a deployment-cost contract, TextVQA has a scoped pre-generation mask30-only policy that nearly matches fixed70 quality at lower cost, but DocVQA has no deployable near-fixed70 lower-cost candidate. A conservative domain-aware portfolio slightly lowers aggregate cost by using the TextVQA pre-generation policy and fixed70 fallback on DocVQA, but it does not solve per-task adaptive control. Cross-task and pooled-dev transfer audits find zero near-fixed70/lower-keep candidates across TextVQA and DocVQA. Adding EasyOCR detector statistics and selector-mask ECR signals still fails held-out stress-pack gates: TextVQA trails fixed70 by -0.077 at keep 0.654, while DocVQA trails by -0.029 at keep 0.738. The aggregate adaptive-controller audit records six failed required gates and two partial gates. Under the final strict contract, no non-oracle candidate simultaneously stays within 0.01 of fixed70 on both tasks, keeps at most 0.60, and remains cheaper after controller overhead, so controller expansion is stopped.",
        "remaining_gap": "No deployable unified controller yet matches high-budget quality at clearly lower real compute across both TextVQA and DocVQA. Serial cascades and low-answer policies often understate cost if reported only by selected keep, online detector-aware policies must count OCR latency, and scoped box-aware ECR fallback does not generalize to DocVQA line-context evidence.",
        "paper_action": "Do not claim universal adaptive control. Report the TextVQA scoped positive, the DocVQA failure, the unified-transfer negative result, the detector-aware stress-pack negative result, and the aggregate go/no-go audit under the same deployment contract.",
        "next_priority": "closed_negative",
    },
    {
        "id": "R5",
        "section": "Method principle",
        "reviewer_question": "Is the algorithm more than score plus top-k and tuned weights?",
        "requested_evidence": "A principled risk objective or coverage/diversity optimizer that improves answer behavior, not only ECR.",
        "status": "partially_addressed_negative",
        "evidence_artifacts": "method-principle audit; coverage-greedy selector; method coverage paired audit; component/Pareto audit.",
        "key_reading": "Component-level Pareto analysis shows Target 0.30 is the best-accuracy row, Protected evidence is the best-ECR row, and Soft evidence is the only audited component with ECR gain at similar answer risk. Coverage-greedy and hard protection raise ECR but degrade accuracy or hFPR.",
        "remaining_gap": "The method has a clearer evidence-risk design and mechanism audit, but still does not provide a universally better answer-risk optimizer.",
        "paper_action": "Claim a principled evidence-risk selector family and component-level trade-off audit. Do not promote coverage maximization as the final algorithm.",
        "next_priority": "high",
    },
    {
        "id": "R6",
        "section": "External baselines",
        "reviewer_question": "Are FastV/VisionZip and other pruning comparisons fair across backbones?",
        "requested_evidence": "Official or faithful ports, matched budgets, same decoding/timing path, and Pareto curves.",
        "status": "mostly_addressed_with_scoped_ports",
        "evidence_artifacts": "LLaVA FastV/VisionZip/SCOPE ports; pinned AnchorPrune official-algorithm port with exact selector parity; CoIn paper-algorithm port; external-baseline fairness matrix; Qwen3 VisionZip native port gate and matched-budget run; selector-inclusive repeated timing.",
        "key_reading": "LLaVA comparisons now include five external selectors at matched budgets. The closest available AnchorPrune selector is pinned, adapted only at the Hugging Face model interface, index-identical on 12 deterministic parity cases, and evaluated on both development and locked confirmation splits with selector-inclusive timing. Qwen3 VisionZip passes native-port gates and is fair at matched budget. Qwen3 FastV and native InternVL external ports remain unsupported.",
        "remaining_gap": "No universal external-baseline parity over every backbone and method.",
        "paper_action": "Claim fair, scoped LLaVA external-method and Qwen VisionZip comparisons only. Distinguish official-algorithm, paper-algorithm, and unsupported ports; avoid blanket superiority over all prior pruning methods.",
        "next_priority": "low",
    },
    {
        "id": "R7",
        "section": "End-to-end efficiency",
        "reviewer_question": "Does the reported speedup hold for real inference, detector cost, and long decoding?",
        "requested_evidence": "Pipeline decomposition: vision encoder, projector, OCR detector, selector, prefill, decode, total latency, p50/p95.",
        "status": "mostly_addressed_with_scope_limit",
        "evidence_artifacts": "real CUDA prefill/memory; cross-model batch prefill; cross-model actual overhead; detector-in-loop latency; length-sensitive end-to-end audits; noisy-box latency estimate.",
        "key_reading": "Shortened visual prefixes accelerate prefill and memory; end-to-end gains shrink for long output, and online EasyOCR can erase single-sample detector-assisted gains.",
        "remaining_gap": "Some cross-model long-output numbers are latency-bound estimates rather than fresh full decode benchmarks.",
        "paper_action": "State speedups as batch-prefill/TTFT where measured, and separate detector-free from detector-assisted settings.",
        "next_priority": "low",
    },
    {
        "id": "R8",
        "section": "Statistics and data rigor",
        "reviewer_question": "Are benchmark labels, random baselines, clusters, and calibration protocols rigorous?",
        "requested_evidence": "Image-cluster bootstrap, multi-seed random, hard-negative validation, and explicit InternVL threshold protocol.",
        "status": "addressed",
        "evidence_artifacts": "image-cluster stats; multi-seed random audit; hard-negative construction and lexical audits; separate exhaustive 500-row development and locked-confirmation human QC, each with a frozen 100-row independent overlap; InternVL calibration audit.",
        "key_reading": "Core TextOCR-Hard conclusions survive image-level resampling. Random baselines are multi-seed for main budgets. Development QC changes any hFPR by at most 0.013. Locked-confirmation QC finds 465 strict-valid negatives; filtering changes any hFPR by at most 0.009 and preserves the Qwen Target--Full conclusion. InternVL hFPR differences are threshold operating points, not selector differences.",
        "remaining_gap": "Qwen random20 remains an auxiliary single-seed row; this does not affect the locked 30% primary comparison.",
        "paper_action": "Keep the prespecified 500-image confirmation table primary and report the 465-pair human-valid readout as a post-QC sensitivity analysis.",
        "next_priority": "low",
    },
    {
        "id": "R9",
        "section": "Hard robustness",
        "reviewer_question": "Does the method survive dense, noisy, conflicting, multi-language, or detector-failure cases?",
        "requested_evidence": "Stress slices, detector noise/dropout, conflict cases, dense tables, similar strings, small/rotated text, and prompt variations.",
        "status": "partially_addressed",
        "evidence_artifacts": "hard-robustness/conflict-slice audit; open-QA stress audit; bbox-noise audit; TextOCR-Hard EasyOCR detector-in-loop; open-QA EasyOCR detector-in-loop; GSR-Bench boundary result.",
        "key_reading": "Near-miss negatives are lexically clean, but evidence protection can raise hFPR on decoys. Coordinate jitter is less harmful than detector dropout; InternVL soft evidence is more robust to EasyOCR boxes than LLaVA hard protection; TextVQA is more recoverable than DocVQA under open-QA stress.",
        "remaining_gap": "The audit is a consolidated boundary analysis, not an exhaustive robustness suite. There is still no dedicated dense-conflict benchmark or broad multilingual/handwriting/rotated-text evaluation.",
        "paper_action": "Use the hard robustness/conflict audit as a reviewer-facing failure taxonomy. Claim scoped robustness evidence and explicit boundaries, not exhaustive hard-case coverage.",
        "next_priority": "medium",
    },
]


CLAIM_BOUNDARIES: list[dict[str, Any]] = [
    {
        "boundary_id": "B1",
        "problem_md_link": "R1 causal evidence",
        "allowed_claim": "The audits provide causal-style support that retained OCR evidence can matter, especially for Qwen and calibrated InternVL, plus human-verified Qwen-specific semantic-switch evidence.",
        "forbidden_claim": "ECR proves that the model causally used every retained evidence region, or that text replacement provides backbone-uniform semantic causal proof.",
        "evidence_required": "causal go/no-go, deletion/restoration, specificity, occlusion, logit-drop, semantic counterfactual boundary audits, and completed text-replacement human QC for any stronger semantic-causal claim.",
        "current_status": "allowed only with scope limits",
    },
    {
        "boundary_id": "B2",
        "problem_md_link": "R2 native open tasks",
        "allowed_claim": "Full Qwen/LLaVA TextVQA and DocVQA validation runs test benchmark-scale transfer beyond TextOCR-Hard, while lite sets support controlled selector/evidence ablations.",
        "forbidden_claim": "Visual pruning is lossless on open OCR/document QA or establishes leaderboard superiority.",
        "evidence_required": "Full-validation native generation metrics, paired intervals, and scoped stress/evidence audits support the boundary claim; lossless or superiority claims would require materially different results and broader baseline coverage.",
        "current_status": "full-validation cross-model boundary complete",
    },
    {
        "boundary_id": "B3",
        "problem_md_link": "R3 multi-evidence",
        "allowed_claim": "A human-corrected 96-row stress audit measures multi-region and worst-region evidence availability; 32 disjointly selected rows have independent pre-adjudication agreement measurements.",
        "forbidden_claim": "The 96-row audit is full-benchmark annotation, proves causal use, or has uniformly high geometric/context-role agreement.",
        "evidence_required": "The completed annotations, ECR and quality tables, and 32-row task/type agreement report support the scoped claim; stronger claims require a tighter evidence-role ontology and broader adjudication.",
        "current_status": "scoped human audit and 32-row reliability audit complete",
    },
    {
        "boundary_id": "B4",
        "problem_md_link": "R4 unified adaptive policy",
        "allowed_claim": "Adaptive control has quantified headroom but current deployable controllers fail cross-task/cost-quality gates.",
        "forbidden_claim": "The paper proposes a solved unified adaptive controller that automatically replaces model-specific operating points.",
        "evidence_required": "deployment-contract, unified-transfer, detector-aware, calibrated fallback, and aggregate go/no-go audits.",
        "current_status": "negative boundary result",
    },
    {
        "boundary_id": "B5",
        "problem_md_link": "R5 method principle",
        "allowed_claim": "The selector family is organized by evidence-risk terms and component/Pareto audits reveal the trade-offs.",
        "forbidden_claim": "Coverage maximization or the current objective universally improves answer behavior.",
        "evidence_required": "method-principle, coverage-greedy, paired coverage, and component Pareto audits.",
        "current_status": "principled framing, not solved optimizer",
    },
    {
        "boundary_id": "B6",
        "problem_md_link": "R6 external baselines",
        "allowed_claim": "The paper has fair matched-budget LLaVA comparisons including a pinned, parity-tested AnchorPrune official-algorithm port, plus a scoped Qwen3 VisionZip native-port comparison.",
        "forbidden_claim": "The method is uniformly stronger than all external pruning baselines on every backbone.",
        "evidence_required": "source-level parity and matched-budget native runs for each claimed backbone-method pair.",
        "current_status": "scoped parity only",
    },
    {
        "boundary_id": "B7",
        "problem_md_link": "R7 efficiency",
        "allowed_claim": "Shorter visual prefixes improve measured prefill/TTFT and memory under stated settings.",
        "forbidden_claim": "The maximum batch-prefill speedup is a general end-to-end inference speedup including OCR and long decoding.",
        "evidence_required": "CUDA prefill, TTFT, detector latency, decode-length sensitivity, and detector-inclusive measurements.",
        "current_status": "scope speed claims by pipeline component",
    },
    {
        "boundary_id": "B8",
        "problem_md_link": "R8 statistics and data rigor",
        "allowed_claim": "The paper includes image-cluster statistics, multi-seed random controls, lexical audits, separate exhaustive human QC of 500 development and 500 locked-confirmation negatives with frozen 100-row independent overlaps, a scoped 96-row human-corrected evidence audit, 32 independent multi-region agreement rows, and calibration protocols.",
        "forbidden_claim": "The full OCR benchmarks have been exhaustively human annotated, or moderate geometry agreement establishes a unique evidence ontology.",
        "evidence_required": "Retain the final adjudicated confirmation labels, pre-adjudication agreement record, post-QC sensitivity outputs, and the unchanged prespecified primary table.",
        "current_status": "development and locked-confirmation hard-negative QC complete",
    },
    {
        "boundary_id": "B9",
        "problem_md_link": "R9 hard robustness",
        "allowed_claim": "The paper reports a stress/failure taxonomy for detector noise, open-QA slices, near-miss negatives, and spatial/OCR boundaries.",
        "forbidden_claim": "The method is exhaustively robust to dense, multilingual, handwritten, rotated, or detector-misleading documents.",
        "evidence_required": "dedicated broad hard-case benchmark beyond the current stress audits.",
        "current_status": "boundary taxonomy, not exhaustive robustness",
    },
]


NEXT_ACTIONS: list[dict[str, Any]] = [
    {
        "priority": 1,
        "action": "Freeze adaptive control as a documented no-go boundary and run a final manuscript/submission integrity audit.",
        "closes_requirements": "R2,R4,global",
        "expected_gain": "Prevents stale lite-scale wording and already-rejected controller plans from weakening the completed evidence package.",
        "blocker": "None; the strict stop rule has fired.",
        "stop_condition": "All current status artifacts and manuscript variants agree on full-validation open-QA results and the controller no-go boundary.",
    },
    {
        "priority": 2,
        "action": "Keep completed semantic replacement as Qwen-specific support and InternVL as a cross-backbone boundary.",
        "closes_requirements": "R1",
        "expected_gain": "Preserves the value of 102 human-validated edits without overstating a non-uniform result.",
        "blocker": "InternVL has only 3/102 strict all-control successes versus Qwen's 26/102.",
        "stop_condition": "No further semantic-edit experiment unless a new editing method plausibly changes the cross-backbone signal.",
    },
    {
        "priority": 3,
        "action": "Add one more native external baseline port only if implementation hooks are clean.",
        "closes_requirements": "R6",
        "expected_gain": "Reduces baseline asymmetry.",
        "blocker": "Qwen3 FastV and InternVL hooks may be costly or unsupported.",
        "stop_condition": "Matched-budget run passes source-level parity gates; otherwise report unsupported transparently.",
    },
    {
        "priority": 4,
        "action": "Polish manuscript claims around full open QA, speed, ECR, and adaptive control.",
        "closes_requirements": "R1,R2,R4,R7",
        "expected_gain": "Prevents overclaiming or stale underclaiming from becoming a review liability.",
        "blocker": "None.",
        "stop_condition": "Abstract/introduction/results use scoped language: benchmark-scale boundary, prefill/TTFT, evidence availability, and controller no-go.",
    },
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "problem_md_requirement_audit.csv", REQUIREMENTS)
    write_csv(OUT_DIR / "problem_md_claim_boundaries.csv", CLAIM_BOUNDARIES)
    write_csv(OUT_DIR / "problem_md_next_action_queue.csv", NEXT_ACTIONS)
    (OUT_DIR / "problem_md_requirement_audit.md").write_text(build_markdown(), encoding="utf-8")
    print(f"Wrote requirement audit to {OUT_DIR}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_markdown() -> str:
    lines = [
        "# Problem.md Requirement Audit",
        "",
        "This file translates the strategic concerns in `problem.md` into a claim-evidence-action ledger. It is a planning artifact, not manuscript prose.",
        "",
        "## Bottom Line",
        "",
        "The paper is now much stronger than the original borderline version on data rigor, full-validation native open QA, causal-style deletion/restoration, completed human semantic-edit QC, detector-source auditing, scoped human multi-region evidence, and efficiency caveats. Full Qwen/LLaVA TextVQA and DocVQA runs close the scale objection but reveal non-negligible pruning losses. The strict controller contract triggers a no-go stop rule; adaptive control must remain a negative boundary, and the method novelty story should be framed as an evidence-risk framework plus audits rather than a universal optimizer.",
        "",
        "## Requirement Matrix",
        "",
        table_md(
            REQUIREMENTS,
            [
                "id",
                "section",
                "status",
                "reviewer_question",
                "key_reading",
                "remaining_gap",
                "paper_action",
                "next_priority",
            ],
        ),
        "",
        "## Claim Boundaries",
        "",
        table_md(
            CLAIM_BOUNDARIES,
            [
                "boundary_id",
                "problem_md_link",
                "allowed_claim",
                "forbidden_claim",
                "current_status",
            ],
        ),
        "",
        "## Immediate Action Queue",
        "",
        table_md(
            NEXT_ACTIONS,
            [
                "priority",
                "action",
                "closes_requirements",
                "expected_gain",
                "blocker",
                "stop_condition",
            ],
        ),
        "",
        "## Recommended Manuscript Stance",
        "",
        "- Strongly claim: evidence-risk-aware pruning exposes failures that accuracy-only evaluation hides; the proposed training-free selectors preserve OCR evidence better than matched controls on TextOCR-Hard; the evidence package includes causal-style, detector, native-open-QA, and real-efficiency audits.",
        "- Carefully scope: ECR is evidence availability, not proof of causal use; open-QA gains are Qwen-specific while LLaVA supplies a negative transfer boundary; detector-assisted speedups depend on whether boxes are already available or must be run online.",
        "- Do not claim yet: a solved unified adaptive controller, exhaustive document-layout robustness, or full external-baseline parity on every backbone.",
        "",
    ]
    return "\n".join(lines)


def table_md(rows: list[dict[str, Any]], columns: list[str]) -> str:
    out = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(clean_cell(row.get(col, "")) for col in columns) + " |")
    return "\n".join(out)


def clean_cell(value: Any) -> str:
    text = str(value)
    return text.replace("\n", " ").replace("|", "/")


if __name__ == "__main__":
    main()
