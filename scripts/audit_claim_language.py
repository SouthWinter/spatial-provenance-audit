#!/usr/bin/env python3
"""Audit manuscript wording against the current evidence boundary.

The checks are intentionally conservative: they flag statements that can sound
stronger than the evidence package supports, especially around causality,
official baselines, adaptive control, leaderboard claims, and speedup scope.
This is a language-risk audit, not a grammar checker.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "problem_optimization_audit" / "claim_language_audit"

TARGET_FILES = [
    ROOT / "paper_aaai2027" / "main.tex",
    ROOT / "paper_aaai2027" / "SupplementaryMaterial.tex",
    ROOT / "arxiv_submission" / "main.tex",
    ROOT / "arxiv_upload" / "main.tex",
]


@dataclass(frozen=True)
class Rule:
    rule_id: str
    severity: str
    pattern: re.Pattern[str]
    rationale: str
    safe_context: tuple[str, ...] = ()


RULES = [
    Rule(
        "strong_causal_language",
        "warn",
        re.compile(r"\b(causal audit|causal audits|causal support|causal proof|causally proves?)\b", re.I),
        "Deletion/restoration, logit-drop, and occlusion are causal-style diagnostics, not full causal proof.",
        safe_context=("causal-style", "not a causal", "do not claim", "not claim", "rather than universal causal proof"),
    ),
    Rule(
        "causal_guarantee_claim",
        "error",
        re.compile(r"\bcausal guarantee\b", re.I),
        "The manuscript may mention that we do not have a causal guarantee, but must not claim one.",
        safe_context=(
            "not a causal guarantee",
            "not a backbone-uniform causal guarantee",
            "no causal guarantee",
            "do not claim",
        ),
    ),
    Rule(
        "ecr_causal_use_overclaim",
        "error",
        re.compile(
            r"\bECR\b[^\n]{0,100}\b(proves?|demonstrates?|establishes?|guarantees?|confirms?)\b[^\n]{0,100}\b(uses?|relies?|causal|reasoning)\b|"
            r"\b(uses?|relies?)\b[^\n]{0,100}\b(proved|demonstrated|established|guaranteed|confirmed)\b[^\n]{0,100}\bECR\b",
            re.I,
        ),
        "ECR is an operational evidence-availability metric, not proof that the model uses the evidence.",
        safe_context=("availability", "not", "does not", "cannot", "proxy", "operational", "boundary"),
    ),
    Rule(
        "evidence_preserving_guarantee",
        "error",
        re.compile(r"\bevidence-preserving\b[^\n]{0,100}\b(guarantees?|ensures?|always|causally)\b", re.I),
        "Evidence-preserving should be presented as an operational pruning/audit goal, not a guarantee on every example.",
        safe_context=("not", "does not", "cannot", "operational", "audit", "boundary"),
    ),
    Rule(
        "end_to_end_speedup_overclaim",
        "error",
        re.compile(r"\b(end-to-end|end to end)\b.*\b(4\.32|speedup|throughput gain|latency reduction)\b", re.I),
        "The 4.32x result is batch-prefill throughput, not full end-to-end generation speed.",
        safe_context=("not claims", "not claim", "not full", "separately", "can erase", "detector-inclusive"),
    ),
    Rule(
        "four_point_three_two_scope",
        "error",
        re.compile(r"4\.32\s*\\?\$?\\?times|4\.32x", re.I),
        "The 4.32x result must be scoped as batch-prefill throughput, not generic inference or generation speedup.",
        safe_context=("batch-prefill", "batch prefill", "prefill"),
    ),
    Rule(
        "generation_speedup_scope",
        "warn",
        re.compile(r"\b(generation speedup|generation gain|full-generation speedup|full generation speedup)\b", re.I),
        "Generation-speed claims should mention decode dilution or the measured decode32/length-sensitivity boundary.",
        safe_context=("decode", "dilut", "32-token", "length", "not claim"),
    ),
    Rule(
        "detector_assisted_speed_scope",
        "error",
        re.compile(r"\b(box-aware|OCR-assisted|detector-assisted|detector-aware)\b[^\n]{0,100}\b(speedup|latency reduction|faster|TTFT|throughput)\b", re.I),
        "Detector-assisted efficiency claims must state whether boxes are precomputed/application-provided or online detector cost is included.",
        safe_context=("precomputed", "application-provided", "online", "detector cost", "ocr cost", "latency", "counted", "amortized", "can erase", "must be counted", "separately"),
    ),
    Rule(
        "real_detector_box_scope",
        "warn",
        re.compile(r"\breal[- ]detector boxes?\b|\breal detector\b[^\n]{0,80}\bbox", re.I),
        "Detector experiments use scoped EasyOCR/source-robustness and detector-in-loop diagnostics, not a full deployment claim.",
        safe_context=("EasyOCR", "detector-source", "source robustness", "source-robustness", "diagnostic", "scoped", "online", "latency"),
    ),
    Rule(
        "official_baseline_overclaim",
        "error",
        re.compile(r"\b(Qwen|InternVL)[^\n]{0,80}\bofficial(?:-algorithm)?[^\n]{0,80}\b(FastV|VisionZip)\b|\bofficial(?:-algorithm)?[^\n]{0,80}\b(FastV|VisionZip)[^\n]{0,80}\b(Qwen|InternVL)\b", re.I),
        "Qwen and InternVL external-method rows are proxies or unsupported, not official FastV/VisionZip ports.",
        safe_context=("not", "proxy", "proxies", "unsupported", "not claimed", "no official"),
    ),
    Rule(
        "qwen_visionzip_stale_proxy_claim",
        "error",
        re.compile(
            r"\bQwen(?:3|-?VL)?[^\n]{0,100}\bVisionZip[^\n]{0,100}\b(native port required|proxy|proxies|not official)\b|"
            r"\bQwen rows are VisionZip-style proxies\b|"
            r"\bQwen and InternVL external comparisons are treated as mechanism proxies\b|"
            r"\bQwen and InternVL comparisons to external mechanisms are reported only as budget-matched proxies\b",
            re.I,
        ),
        "Qwen3 VisionZip now has a scoped native official-algorithm port at the matched 30% TextOCR-Hard budget; only legacy diversity rows should be called proxies.",
        safe_context=("scoped native", "official-algorithm row", "official-algorithm port", "older diversity row remains a proxy", "legacy", "not all budgets"),
    ),
    Rule(
        "adaptive_solved_overclaim",
        "error",
        re.compile(r"\b(adaptive|risk-aware|dynamic)\b[^\n]{0,80}\b(solved|dominates?|beats fixed|unified controller)\b", re.I),
        "Adaptive risk control remains diagnostic/negative; it does not beat the fixed-budget frontier.",
        safe_context=("not", "do not claim", "does not", "diagnostic", "boundary"),
    ),
    Rule(
        "adaptive_transfer_overclaim",
        "error",
        re.compile(r"\b(adaptive|dynamic|risk)\b[^\n]{0,120}\b(transfers?|generalizes?|pools?)\b[^\n]{0,120}\b(TextVQA|DocVQA|tasks?|datasets?)\b", re.I),
        "Cross-task and pooled adaptive policies fail the near-fixed70/lower-keep gate and should be described as a negative/boundary result.",
        safe_context=("not", "no ", "fail", "fails", "negative", "boundary", "does not", "zero"),
    ),
    Rule(
        "leaderboard_overclaim",
        "error",
        re.compile(r"\b(TextVQA|DocVQA|OCRBench)[^\n]{0,80}\bleaderboard\b", re.I),
        "TextVQA/DocVQA/OCRBench rows are scoped diagnostics, not leaderboard submissions.",
        safe_context=("not", "rather than", "do not claim"),
    ),
    Rule(
        "textocr_hard_benchmark_scope",
        "warn",
        re.compile(r"\bTextOCR-Hard\b[^\n]{0,100}\bbenchmark\b", re.I),
        "TextOCR-Hard is a 1000-probe yes/no diagnostic benchmark, not a broad OCR/document-QA benchmark.",
        safe_context=("diagnostic", "1000-probe", "yes/no", "scope", "boundary"),
    ),
    Rule(
        "openqa_broad_claim",
        "warn",
        re.compile(r"\b(TextVQA|DocVQA|OCRBench|open-QA|open QA|document QA)\b[^\n]{0,120}\b(proves?|establishes?|solves?|demonstrates? broad|generalizes? broadly)\b", re.I),
        "Open-QA results are full-validation and fixed-subset boundary checks, not broad document-QA proof.",
        safe_context=("diagnostic", "full-validation", "fixed subset", "scoped", "boundary", "not", "rather than"),
    ),
    Rule(
        "openqa_oracle_free_deployment_overclaim",
        "error",
        re.compile(
            r"\b(open-QA|open QA|TextVQA|DocVQA|document-QA|document QA)\b[^\n]{0,140}\b(oracle-free|deployment claim|deployable evidence|deployment result)\b|"
            r"\b(oracle-free|deployment claim|deployable evidence|deployment result)\b[^\n]{0,140}\b(open-QA|open QA|TextVQA|DocVQA|document-QA|document QA)\b",
            re.I,
        ),
        "Open-QA evidence rows mix external GT boxes, OCR-derived boxes, and deterministic context; they are diagnostics, not oracle-free deployment evidence.",
        safe_context=("not", "prevents", "prevent", "rather than", "boundary", "diagnostic", "not a", "cannot"),
    ),
    Rule(
        "universal_selector_overclaim",
        "error",
        re.compile(r"\b(universal|single best|uniformly best)\b[^\n]{0,80}\b(selector|keep ratio|policy)\b", re.I),
        "The evidence supports a selector family with backbone-dependent operating points, not one universal selector.",
        safe_context=("not", "no single", "do not claim"),
    ),
    Rule(
        "method_objective_guarantee_overclaim",
        "error",
        re.compile(
            r"\b(evidence-risk objective|method objective|objective)\b[^\n]{0,120}\b(guarantees?|ensures?|solves?|dominates?|optimal|near-optimal)\b|"
            r"\b(guarantees?|ensures?|solves?|dominates?|optimal|near-optimal)\b[^\n]{0,120}\b(evidence-risk objective|method objective|objective)\b",
            re.I,
        ),
        "The evidence-risk objective is a design contract and audit lens, not a guarantee or learned optimizer.",
        safe_context=("not", "does not", "cannot", "diagnostic", "boundary", "audit lens", "design contract"),
    ),
    Rule(
        "coverage_greedy_main_overclaim",
        "error",
        re.compile(r"\bcoverage-greedy\b[^\n]{0,120}\b(main selector|main method|dominates?|best|solves?|optimal|near-optimal)\b", re.I),
        "Coverage-greedy is a diagnostic/boundary case; it improves availability in some settings but can hurt answer risk.",
        safe_context=("not", "negative diagnostic", "diagnostic", "boundary", "does not"),
    ),
    Rule(
        "full_benchmark_evidence_annotation_overclaim",
        "error",
        re.compile(
            r"\b(all|entire|full[- ]benchmark|every)\b[^\n]{0,100}\b(TextVQA|DocVQA|open[- ]QA)\b[^\n]{0,120}\b(human[- ]annotated|human[- ]validated|manually annotated)\b|"
            r"\b(human[- ]annotated|human[- ]validated|manually annotated)\b[^\n]{0,120}\b(all|entire|full[- ]benchmark|every)\b[^\n]{0,100}\b(TextVQA|DocVQA|open[- ]QA)\b",
            re.I,
        ),
        "Human-corrected multi-region annotation covers 96 fixed cases, not every TextVQA/DocVQA validation example.",
        safe_context=("not", "96", "fixed", "subset", "stress audit", "boundary"),
    ),
    Rule(
        "all_backbone_external_baseline_overclaim",
        "error",
        re.compile(r"\b(all|every|each)\b[^\n]{0,80}\b(backbone|model family|model)\b[^\n]{0,100}\b(FastV|VisionZip|external baseline)\b", re.I),
        "External-method parity is strongest for LLaVA and Qwen VisionZip; it is not complete for all backbones.",
        safe_context=("not", "unsupported", "proxy", "proxies", "cannot", "do not claim"),
    ),
    Rule(
        "exhaustive_robustness_overclaim",
        "error",
        re.compile(
            r"\b(exhaustive|comprehensive|fully robust|robust to all|covers all)\b[^\n]{0,120}\b(dense|multilingual|handwrit|rotated|detector|document|layout|robustness|hard cases?)\b",
            re.I,
        ),
        "The hard-case experiments are a stress/failure taxonomy, not exhaustive robustness over all document layouts and OCR conditions.",
        safe_context=("not", "not exhaustive", "boundary", "taxonomy", "future", "does not"),
    ),
    Rule(
        "cross_backbone_full_prefix_parity_overclaim",
        "warn",
        re.compile(
            r"\b(across|over)\b[^\n]{0,80}\b(three|3)\b[^\n]{0,80}\b(MLLMs|models|families|backbones)\b[^\n]{0,140}\b(matches?|preserves?|maintains?)\b[^\n]{0,80}\bfull-prefix quality\b",
            re.I,
        ),
        "Full-prefix parity is cleanest for Qwen and calibrated InternVL; LLaVA is better framed as an evidence-protection/baseline-boundary case.",
        safe_context=("Qwen and calibrated InternVL", "explicit evidence protection", "operating point", "case where", "backbone-dependent"),
    ),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = list(scan_files(TARGET_FILES))
    write_csv(OUT_DIR / "claim_language_audit.csv", rows)
    (OUT_DIR / "claim_language_audit.md").write_text(build_markdown(rows), encoding="utf-8")
    errors = [row for row in rows if row["severity"] == "error"]
    warnings = [row for row in rows if row["severity"] == "warn"]
    print(f"Wrote {OUT_DIR / 'claim_language_audit.md'}")
    print(f"errors={len(errors)} warnings={len(warnings)}")
    if errors:
        raise SystemExit(1)


def scan_files(paths: Iterable[Path]) -> Iterable[dict[str, str]]:
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        for lineno, line in enumerate(lines, start=1):
            normalized = line.strip()
            if not normalized or normalized.startswith("%"):
                continue
            window = "\n".join(lines[max(0, lineno - 3) : min(len(lines), lineno + 2)])
            for rule in RULES:
                if not rule.pattern.search(normalized):
                    continue
                if has_safe_context(window, rule.safe_context):
                    continue
                yield {
                    "file": str(path.relative_to(ROOT)),
                    "line": str(lineno),
                    "severity": rule.severity,
                    "rule_id": rule.rule_id,
                    "rationale": rule.rationale,
                    "text": normalized,
                }


def has_safe_context(line: str, safe_terms: tuple[str, ...]) -> bool:
    lower = line.lower()
    return any(term.lower() in lower for term in safe_terms)


def build_markdown(rows: list[dict[str, str]]) -> str:
    lines = [
        "# Claim-Language Audit",
        "",
        "This audit scans manuscript files for wording that can exceed the current evidence boundary in `problem.md`.",
        "",
    ]
    if not rows:
        lines.extend(
            [
                "No risky manuscript-language matches were found under the current rules.",
                "",
                "Current guarded scopes: ECR-as-availability, causal-style diagnostics, batch-prefill speedup, detector-assisted latency scope, LLaVA/Qwen-specific external ports, diagnostic adaptive control, a scoped 96-sample human-corrected multi-region audit, exhaustive 500-pair human QC on each TextOCR-Hard split, non-leaderboard open-QA checks, and non-exhaustive hard-case robustness.",
            ]
        )
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "| severity | rule | file:line | rationale | text |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        location = f"{row['file']}:{row['line']}"
        text = row["text"].replace("|", "\\|")
        rationale = row["rationale"].replace("|", "\\|")
        lines.append(f"| {row['severity']} | {row['rule_id']} | {location} | {rationale} | {text} |")
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["file", "line", "severity", "rule_id", "rationale", "text"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
