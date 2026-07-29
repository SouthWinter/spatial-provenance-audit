#!/usr/bin/env python3
"""Aggregate evidence-use diagnostics into a causal-evidence go/no-go audit.

problem.md's strongest conceptual concern is that evidence coverage does not
prove evidence use. This script aggregates the cached causal-style diagnostics
and decides what level of causal language the paper can safely use.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "problem_optimization_audit" / "causal_evidence_go_no_go"

TRIAD_CSV = ROOT / "runs" / "paper_evidence" / "table_causal_evidence_triad.csv"
TEXT_REPLACEMENT_CSV = ROOT / "runs" / "paper_evidence" / "table_text_replacement_counterfactual.csv"
TEXT_REPLACEMENT_OCR_CSV = ROOT / "runs" / "paper_evidence" / "table_text_replacement_ocr_quality.csv"
TEXT_REPLACEMENT_STRATIFIED_CSV = ROOT / "runs" / "paper_evidence" / "table_text_replacement_stratified.csv"
TEXT_REPLACEMENT_HUMAN_QC_PROGRESS_CSV = (
    ROOT / "runs" / "paper_evidence" / "table_text_replacement_human_qc_progress_decision.csv"
)
TEXT_REPLACEMENT_HUMAN_VALID_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "text_replacement_control_pack_v3"
    / "human_valid_eval"
    / "human_valid_model_summary.csv"
)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    decision = build_decision(rows)
    write_csv(OUT_DIR / "causal_evidence_go_no_go.csv", rows)
    write_csv(OUT_DIR / "causal_evidence_decision.csv", [decision])
    (OUT_DIR / "causal_evidence_go_no_go.md").write_text(build_markdown(rows, decision), encoding="utf-8")
    print(f"Wrote causal evidence go/no-go audit to {OUT_DIR}")
    print(f"causal_claim_status={decision['causal_claim_status']}")


def build_rows() -> list[dict[str, Any]]:
    triad = read_csv(TRIAD_CSV)
    replacement = read_csv(TEXT_REPLACEMENT_CSV)
    replacement_ocr = read_csv(TEXT_REPLACEMENT_OCR_CSV)
    replacement_stratified = read_csv(TEXT_REPLACEMENT_STRATIFIED_CSV)
    replacement_qc = read_csv(TEXT_REPLACEMENT_HUMAN_QC_PROGRESS_CSV)
    replacement_human_valid = read_csv(TEXT_REPLACEMENT_HUMAN_VALID_CSV)
    rows = []
    rows.extend(model_rows(triad))
    rows.append(
        semantic_counterfactual_row(
            replacement,
            replacement_ocr,
            replacement_stratified,
            replacement_qc,
            replacement_human_valid,
        )
    )
    rows.append(cross_backbone_row(triad))
    return rows


def model_rows(triad: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in triad:
        by_model[row["model"]].append(row)
    out = []
    for model, rows in sorted(by_model.items()):
        strong = sum(row.get("strength") == "strong" for row in rows)
        mixed = sum(row.get("strength") in {"mixed", "weak_or_mixed"} for row in rows)
        weak = sum(row.get("strength") == "weak" for row in rows)
        has_restoration = any(row.get("category") == "necessity_and_restoration" and row.get("strength") == "strong" for row in rows)
        has_specificity = any(row.get("category") == "input_space_specificity" and row.get("strength") == "strong" for row in rows)
        status = "strong_causal_style_support" if strong >= 2 and has_specificity else "boundary_or_weak_support"
        if "LLaVA" in model:
            status = "weak_backbone_boundary"
        out.append(
            {
                "gate": f"model_causal_style_support:{model}",
                "scope": model,
                "status": status,
                "strong_rows": strong,
                "mixed_rows": mixed,
                "weak_rows": weak,
                "evidence": summarize_model(rows),
                "safe_claim": safe_claim_for_model(model, status),
            }
        )
    return out


def semantic_counterfactual_row(
    replacement: list[dict[str, str]],
    replacement_ocr: list[dict[str, str]],
    replacement_stratified: list[dict[str, str]],
    replacement_qc: list[dict[str, str]],
    replacement_human_valid: list[dict[str, str]],
) -> dict[str, Any]:
    hq = next((row for row in replacement if row.get("split") == "HQ filtered"), {})
    hq_ocr = next((row for row in replacement_ocr if row.get("split") == "HQ filtered"), {})
    hq_success = next(
        (row for row in replacement_stratified if row.get("split") == "HQ filtered" and row.get("group") == "edited_crop_ocr_success"),
        {},
    )
    full_switch = fnum(hq.get("full_four_way_semantic_switch_rate"))
    ocr_success = fnum(hq_ocr.get("edited_crop_ocr_success_rate"))
    success_switch = fnum(hq_success.get("full_four_way_semantic_switch_rate"))
    qc_row = replacement_qc[0] if replacement_qc else {}
    human_qc_status = qc_row.get("text_replacement_human_qc_status", "missing")
    human_qc_ready = f"{qc_row.get('ready_rows', '')}/{qc_row.get('rows', '')}"
    human_valid_rate = fnum(qc_row.get("valid_semantic_edit_rate"))
    qwen_switch = find_metric(replacement_human_valid, "Qwen3-VL-8B", "full_semantic_switch")
    qwen_strict = find_metric(replacement_human_valid, "Qwen3-VL-8B", "all_four_controls_correct")
    internvl_switch = find_metric(replacement_human_valid, "InternVL3.5-8B", "full_semantic_switch")
    internvl_strict = find_metric(replacement_human_valid, "InternVL3.5-8B", "all_four_controls_correct")
    status = "boundary_not_full_causal_proof"
    if human_valid_rate >= 0.8 and human_qc_status == "ready_for_verified_semantic_counterfactual_claim":
        status = "human_verified_model_dependent_support"
    return {
        "gate": "semantic_text_replacement_counterfactual",
        "scope": "Qwen/InternVL edited TextOCR subset",
        "status": status,
        "strong_rows": int(status == "strong_semantic_counterfactual"),
        "mixed_rows": int(status != "strong_semantic_counterfactual"),
        "weak_rows": 0,
        "evidence": (
            f"HQ full four-way switch={hq.get('full_four_way_semantic_switch_rate','')}; "
            f"HQ edited crop OCR success={hq_ocr.get('edited_crop_ocr_success_rate','')}; "
            f"OCR-success subgroup switch={hq_success.get('full_four_way_semantic_switch_rate','')}; "
            f"source support drop={hq.get('mean_source_yes_support_drop','')}; replacement support gain={hq.get('mean_replacement_yes_support_gain','')}; "
            f"human_qc_status={human_qc_status}; human_qc_ready={human_qc_ready}; human_valid_semantic_edit_rate={qc_row.get('valid_semantic_edit_rate','')}; "
            f"human-valid Qwen switch/strict={qwen_switch.get('rate','')}/{qwen_strict.get('rate','')}; "
            f"InternVL switch/strict={internvl_switch.get('rate','')}/{internvl_strict.get('rate','')}"
        ),
        "safe_claim": "Use the completed human-verified edit audit as model-dependent semantic support for Qwen; InternVL's weak response prevents a backbone-uniform causal claim.",
    }


def cross_backbone_row(triad: list[dict[str, str]]) -> dict[str, Any]:
    models = sorted({row.get("model", "") for row in triad})
    strong_models = []
    weak_models = []
    for model in models:
        rows = [row for row in triad if row.get("model") == model]
        strong = sum(row.get("strength") == "strong" for row in rows)
        weakish = sum(row.get("strength") in {"weak", "weak_or_mixed"} for row in rows)
        if strong >= 2:
            strong_models.append(model)
        if weakish >= 2:
            weak_models.append(model)
    status = "not_uniform_across_backbones" if weak_models else "directionally_consistent"
    return {
        "gate": "cross_backbone_causal_uniformity",
        "scope": "Qwen/LLaVA/InternVL",
        "status": status,
        "strong_rows": len(strong_models),
        "mixed_rows": 0,
        "weak_rows": len(weak_models),
        "evidence": f"strong_models={'; '.join(strong_models)}; weak_or_mixed_models={'; '.join(weak_models)}",
        "safe_claim": "Claim causal-style support with backbone differences; do not claim uniform causal evidence use across all evaluated MLLMs.",
    }


def build_decision(rows: list[dict[str, Any]]) -> dict[str, str]:
    qwen_strong = any(row["scope"].startswith("Qwen") and row["status"] == "strong_causal_style_support" for row in rows)
    internvl_strong = any(row["scope"].startswith("InternVL") and row["status"] == "strong_causal_style_support" for row in rows)
    llava_weak = any(row["scope"].startswith("LLaVA") and row["status"] == "weak_backbone_boundary" for row in rows)
    semantic_strong = any(row["gate"] == "semantic_text_replacement_counterfactual" and row["status"] == "strong_semantic_counterfactual" for row in rows)
    if qwen_strong and internvl_strong and not llava_weak and semantic_strong:
        status = "go_for_strong_causal_use_claim"
    else:
        status = "no_go_for_full_causal_claim"
    return {
        "causal_claim_status": status,
        "qwen_strong": str(int(qwen_strong)),
        "internvl_strong": str(int(internvl_strong)),
        "llava_weak_boundary": str(int(llava_weak)),
        "semantic_counterfactual_strong": str(int(semantic_strong)),
        "recommended_claim": "Use causal-style diagnostics and evidence-use support for Qwen/InternVL, keep ECR as availability, and present human-verified semantic replacement as Qwen-specific support with an InternVL boundary.",
        "avoid_claim": "Do not claim that ECR proves causal use or that all retained evidence patches are causally used across all backbones.",
    }


def summarize_model(rows: list[dict[str, str]]) -> str:
    parts = []
    for row in rows:
        parts.append(
            f"{row.get('category')}:{row.get('strength')} primary={row.get('primary_metric')}={row.get('primary_effect')}"
        )
    return "; ".join(parts)


def find_metric(rows: list[dict[str, str]], model: str, metric: str) -> dict[str, str]:
    return next(
        (row for row in rows if row.get("model") == model and row.get("metric") == metric),
        {},
    )


def safe_claim_for_model(model: str, status: str) -> str:
    if status == "strong_causal_style_support":
        return f"{model} has strong causal-style support from token/input-space interventions, not a formal causal guarantee."
    if status == "weak_backbone_boundary":
        return f"{model} preserves annotated evidence under some selectors, but causal-style effects are weak or mixed; report as a backbone boundary."
    return f"{model} has partial or mixed causal-style support; keep claims scoped."


def build_markdown(rows: list[dict[str, Any]], decision: dict[str, str]) -> str:
    return "\n".join(
        [
            "# Causal Evidence Go/No-Go Audit",
            "",
            "This audit answers how far the paper can go beyond evidence availability. It aggregates deletion/restoration, evidence-only/anti-evidence, region logit-drop, bbox occlusion, and semantic text-replacement diagnostics.",
            "",
            "## Decision",
            "",
            f"- Causal-claim status: `{decision['causal_claim_status']}`",
            f"- Qwen strong: {decision['qwen_strong']}",
            f"- InternVL strong: {decision['internvl_strong']}",
            f"- LLaVA weak boundary: {decision['llava_weak_boundary']}",
            f"- Semantic counterfactual strong: {decision['semantic_counterfactual_strong']}",
            f"- Recommended claim: {decision['recommended_claim']}",
            f"- Avoid claim: {decision['avoid_claim']}",
            "",
            "## Gate Table",
            "",
            table_md(rows, ["gate", "scope", "status", "strong_rows", "mixed_rows", "weak_rows", "evidence", "safe_claim"]),
            "",
            "## Source Tables",
            "",
            f"- `{TRIAD_CSV.relative_to(ROOT)}`",
            f"- `{TEXT_REPLACEMENT_CSV.relative_to(ROOT)}`",
            f"- `{TEXT_REPLACEMENT_OCR_CSV.relative_to(ROOT)}`",
            f"- `{TEXT_REPLACEMENT_STRATIFIED_CSV.relative_to(ROOT)}`",
            f"- `{TEXT_REPLACEMENT_HUMAN_QC_PROGRESS_CSV.relative_to(ROOT)}`",
            f"- `{TEXT_REPLACEMENT_HUMAN_VALID_CSV.relative_to(ROOT)}`",
            "",
        ]
    )


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
        out.append("| " + " | ".join(clean(row.get(col, "")) for col in columns) + " |")
    return "\n".join(out)


def clean(value: Any) -> str:
    return str(value).replace("\n", " ").replace("|", "/")


def fnum(value: Any) -> float:
    try:
        if value is None or value == "":
            return float("-inf")
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


if __name__ == "__main__":
    main()
