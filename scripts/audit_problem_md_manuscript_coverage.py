#!/usr/bin/env python3
"""Audit whether problem.md risks are visible in the manuscript.

This is a manuscript-coverage audit, not a scientific score. It checks that
each major concern from problem.md has (1) a corresponding manuscript signal,
(2) a boundary/scope statement where needed, and (3) a paper-evidence artifact.
The goal is to prevent a common failure mode: the experiment directory contains
the right caveat, but the submitted paper does not make it visible to reviewers.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "problem_optimization_audit" / "problem_md_manuscript_coverage"

MANUSCRIPT_FILES = [
    ROOT / "paper_aaai2027" / "main.tex",
    ROOT / "paper_aaai2027" / "SupplementaryMaterial.tex",
]
EVIDENCE_FILES = [
    ROOT / "runs" / "paper_evidence" / "paper_evidence_package.md",
    ROOT / "runs" / "paper_evidence" / "claim_ledger.csv",
    ROOT / "runs" / "problem_optimization_audit" / "problem_md_requirement_audit" / "problem_md_requirement_audit.md",
]


@dataclass(frozen=True)
class CoverageSpec:
    req_id: str
    concern: str
    manuscript_signals: tuple[str, ...]
    boundary_signals: tuple[str, ...]
    evidence_signals: tuple[str, ...]
    severity_if_missing: str


SPECS = [
    CoverageSpec(
        "R1",
        "ECR is evidence availability, not proof of causal use.",
        (
            r"\bECR\b",
            r"deletion--restoration|deletion-restoration",
            r"bbox occlusion|logit-drop|evidence-only|anti-evidence",
        ),
        (
            r"ECR is an availability metric",
            r"not a causal guarantee",
            r"causal-style",
            r"spatial-provenance availability rather than .*causal use",
            r"proof of causal use",
        ),
        (
            r"Causal evidence",
            r"table_causal_evidence_triad",
            r"table_deletion_restoration",
        ),
        "error",
    ),
    CoverageSpec(
        "R2",
        "Constructed yes/no probes need native open-task boundary checks.",
        (
            r"TextOCR-Hard",
            r"OCRBench",
            r"TextVQA-lite",
            r"DocVQA-lite",
        ),
        (
            r"scoped original-question transfer",
            r"not claim OCRBench, TextVQA, or DocVQA leaderboard",
            r"external-validity diagnostic",
            r"rather than official .* leaderboard",
            r"not leaderboard-tuned runs",
            r"rather than the native OCRBench leaderboard metric",
        ),
        (
            r"table_ocrbench_open_answer_generation",
            r"table_open_ocr_qa_generation",
            r"table_open_ocr_qa_stress",
        ),
        "error",
    ),
    CoverageSpec(
        "R3",
        "Multi-evidence and document context are stricter than single answer boxes.",
        (
            r"line-context",
            r"multi-region|multi-box|all-region|all regions",
            r"worst-region",
        ),
        (
            r"OCR-derived",
            r"external TextVQA GT boxes",
            r"not manual layout annotations",
            r"evidence-availability .* diagnostics",
        ),
        (
            r"table_open_ocr_qa_docvqa_line_context",
            r"table_docvqa_document_risk",
            r"Manual multi-region evidence boxes are launch-ready but not complete",
        ),
        "warn",
    ),
    CoverageSpec(
        "R4",
        "Unified adaptive risk control is not solved.",
        (
            r"adaptive control",
            r"unified.*policy|pooled-controller|cross-task",
            r"detector-aware",
        ),
        (
            r"not claim to solve adaptive risk control",
            r"unified adaptive control remains a quantified boundary",
            r"no policy that simultaneously matches fixed-70",
            r"do not yield a unified efficient controller",
            r"fixed 70\\% remains the qualified operating point",
            r"Fixed 70\\% therefore remains the qualified operating point",
            r"no open-QA candidate satisfies both-task quality",
            r"zero-candidate outcomes identify the transfer gap",
        ),
        (
            r"table_open_ocr_qa_unified_policy_transfer_summary",
            r"table_open_ocr_qa_detector_aware_policy_summary",
            r"C9b",
        ),
        "error",
    ),
    CoverageSpec(
        "R5",
        "Method principle is an evidence-risk framework, not a solved optimizer.",
        (
            r"evidence-risk",
            r"Target top-\$k\$",
            r"Soft-evidence|Soft evidence",
        ),
        (
            r"not to prove that cosine similarity is a universal salience estimator",
            r"not a claim that one salience score is universally optimal",
            r"not a universal recipe",
            r"not a claim that relevance--norm scoring or coverage-aware selection is new",
            r"hard protection raises coverage but can reduce accuracy or increase hFPR",
            r"quality--risk--coverage frontier",
            r"calibrated to each backbone",
        ),
        (
            r"table_method_component_pareto",
            r"table_method_coverage_paired",
            r"coverage maximization",
        ),
        "warn",
    ),
    CoverageSpec(
        "R6",
        "External baseline parity is scoped, not complete across all backbones.",
        (
            r"FastV",
            r"VisionZip",
            r"official-algorithm port",
        ),
        (
            r"Qwen and InternVL comparisons .* proxies",
            r"not official FastV results",
            r"No official InternVL",
            r"do not claim official FastV/VisionZip results are available on all three backbones",
            r"cross-backbone comparisons are restricted",
            r"outside native-port scope",
        ),
        (
            r"table_external_baseline_fairness_matrix",
            r"table_qwen3_visionzip_native_port",
            r"table_native_external_port_feasibility",
        ),
        "error",
    ),
    CoverageSpec(
        "R7",
        "Efficiency claims must distinguish prefill/TTFT/decode and detector cost.",
        (
            r"4\.32",
            r"batch-prefill",
            r"TTFT|decode32|32-token",
        ),
        (
            r"not claims about full end-to-end generation speed",
            r"online EasyOCR can erase",
            r"decode dilutes",
            r"detector column adds the measured EasyOCR mean latency",
        ),
        (
            r"table_efficiency",
            r"table_e2e_measured_decode32",
            r"table_e2e_length_sensitivity_key_points",
        ),
        "error",
    ),
    CoverageSpec(
        "R8",
        "Statistics and data quality need cluster/random/negative/calibration checks.",
        (
            r"paired confidence intervals|paired intervals",
            r"hard near-miss negatives",
            r"InternVL.*threshold|calibrated",
        ),
        (
            r"image-disjoint",
            r"Unicode-normalized",
            r"not statistically significant superiority",
            r"risk-selected and default rows",
        ),
        (
            r"table_image_cluster_statistics",
            r"table_random_seed_status",
            r"table_hard_negative_lexical_summary",
            r"table_internvl_threshold_calibration_summary",
        ),
        "warn",
    ),
    CoverageSpec(
        "R9",
        "Hard robustness is a boundary taxonomy, not exhaustive coverage.",
        (
            r"detector dropout",
            r"coordinate jitter",
            r"dense|multilingual|handwritten|rotated|conflict-heavy",
        ),
        (
            r"do not cover all dense",
            r"not exhaustive",
            r"boundary case rather than a positive",
            r"not official benchmark leaderboard results",
        ),
        (
            r"table_hard_robustness_conflict_summary",
            r"table_open_ocr_qa_bbox_noise_summary",
            r"table_box_source_robustness",
        ),
        "warn",
    ),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manuscript = joined_text(MANUSCRIPT_FILES)
    evidence = joined_text(EVIDENCE_FILES)
    rows = [audit_spec(spec, manuscript, evidence) for spec in SPECS]
    write_csv(OUT_DIR / "problem_md_manuscript_coverage.csv", rows)
    (OUT_DIR / "problem_md_manuscript_coverage.md").write_text(build_markdown(rows), encoding="utf-8")
    errors = [row for row in rows if row["status"] == "fail" and row["severity_if_missing"] == "error"]
    warnings = [row for row in rows if row["status"] == "fail" and row["severity_if_missing"] == "warn"]
    print(f"Wrote {OUT_DIR / 'problem_md_manuscript_coverage.md'}")
    print(f"errors={len(errors)} warnings={len(warnings)}")
    if errors:
        raise SystemExit(1)


def joined_text(paths: Iterable[Path]) -> str:
    chunks = []
    for path in paths:
        chunks.append(f"\n\n### {path.relative_to(ROOT)}\n")
        chunks.append(path.read_text(encoding="utf-8"))
    return "\n".join(chunks)


def audit_spec(spec: CoverageSpec, manuscript: str, evidence: str) -> dict[str, str]:
    manuscript_signal_hits = hit_count(spec.manuscript_signals, manuscript)
    boundary_hits = hit_count(spec.boundary_signals, manuscript)
    evidence_hits = hit_count(spec.evidence_signals, evidence)
    missing = []
    if manuscript_signal_hits == 0:
        missing.append("manuscript_signal")
    if boundary_hits == 0:
        missing.append("boundary_signal")
    if evidence_hits == 0:
        missing.append("evidence_signal")
    status = "pass" if not missing else "fail"
    return {
        "req_id": spec.req_id,
        "concern": spec.concern,
        "status": status,
        "severity_if_missing": spec.severity_if_missing,
        "manuscript_signal_hits": str(manuscript_signal_hits),
        "boundary_hits": str(boundary_hits),
        "evidence_hits": str(evidence_hits),
        "missing": ",".join(missing),
    }


def hit_count(patterns: tuple[str, ...], text: str) -> int:
    return sum(1 for pattern in patterns if re.search(pattern, text, flags=re.I | re.S))


def build_markdown(rows: list[dict[str, str]]) -> str:
    lines = [
        "# Problem.md Manuscript Coverage Audit",
        "",
        "This audit checks whether the manuscript and paper-evidence package visibly cover each major risk from `problem.md`. A pass means the current files contain at least one manuscript signal, one boundary/scope signal, and one evidence artifact signal for the requirement; it does not mean the scientific concern is fully solved.",
        "",
        "| id | status | severity | manuscript | boundary | evidence | missing | concern |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {req_id} | {status} | {severity_if_missing} | {manuscript_signal_hits} | {boundary_hits} | {evidence_hits} | {missing} | {concern} |".format(
                **{key: clean(value) for key, value in row.items()}
            )
        )
    return "\n".join(lines) + "\n"


def clean(text: str) -> str:
    return str(text).replace("|", "/").replace("\n", " ")


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
