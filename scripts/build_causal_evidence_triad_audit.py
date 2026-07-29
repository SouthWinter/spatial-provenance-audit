#!/usr/bin/env python
"""Summarize necessity, sufficiency, and specificity evidence from cached audits.

This script does not create new model outputs. It reorganizes existing
deletion-restoration, evidence-only/anti-evidence, region logit-drop, and
input-occlusion results into the causal-evidence categories requested by
problem.md.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "problem_optimization_audit" / "causal_evidence_triad"

HARD_EVIDENCE_CSV = ROOT / "runs" / "hard_evidence" / "hard_evidence_summary.csv"
QWEN_DELETION_CSV = (
    ROOT
    / "runs"
    / "textocr_deletion_restoration"
    / "qwen_target30_runs"
    / "deletion_restoration_summary.csv"
)
INTERNVL_DELETION_CSV = (
    ROOT
    / "runs"
    / "textocr_deletion_restoration"
    / "internvl_soft50_runs_cal_hfprconstr"
    / "deletion_restoration_summary.csv"
)
REGION_SUMMARIES = {
    "Qwen3-VL-8B": ROOT / "runs" / "region_logit_drop" / "qwen_region_logit_drop_summary.csv",
    "LLaVA-1.5-7B": ROOT / "runs" / "region_logit_drop_llava" / "llava_region_logit_drop_summary.csv",
    "InternVL3.5-8B calibrated": ROOT
    / "runs"
    / "region_logit_drop_internvl_calibrated"
    / "internvl_calibrated_region_logit_drop_summary.csv",
}
OCCLUSION_SUMMARIES = {
    "Qwen3-VL-8B": ROOT
    / "runs"
    / "bbox_occlusion_qwen_textocr_hard_200"
    / "qwen3_8b_direct_802816"
    / "occlusion_report"
    / "qwen3_8b_textocr_hard200_bbox_occlusion_summary.csv",
    "LLaVA-1.5-7B": ROOT
    / "runs"
    / "bbox_occlusion_cross_model_textocr_hard_100"
    / "llava15_7b_direct"
    / "occlusion_report"
    / "llava15_7b_textocr_hard100_bbox_occlusion_summary.csv",
    "InternVL3.5-8B calibrated": ROOT
    / "runs"
    / "bbox_occlusion_cross_model_textocr_hard_100"
    / "internvl35_8b_direct_calibrated"
    / "occlusion_report"
    / "internvl35_8b_textocr_hard100_bbox_occlusion_calibrated_summary.csv",
}


def main() -> None:
    hard = read_csv(HARD_EVIDENCE_CSV)
    rows: list[dict[str, Any]] = []
    rows.extend(build_restoration_rows("Qwen3-VL-8B", QWEN_DELETION_CSV))
    rows.extend(build_restoration_rows("InternVL3.5-8B calibrated", INTERNVL_DELETION_CSV))
    rows.extend(build_token_evidence_rows(hard))
    rows.extend(build_region_rows())
    rows.extend(build_occlusion_rows())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "causal_evidence_triad_summary.csv", rows)
    (OUT_DIR / "causal_evidence_triad_report.md").write_text(build_report(rows), encoding="utf-8")
    print(f"Wrote {OUT_DIR / 'causal_evidence_triad_report.md'}")


def build_restoration_rows(model: str, path: Path) -> list[dict[str, Any]]:
    rows = read_csv(path)
    selected = find(rows, variant="selected")
    removed = find(rows, variant="remove_evidence")
    restore_e = find(rows, variant="restore_evidence_1p00")
    restore_r = find(rows, variant="restore_random_1p00")
    drop = fnum(selected["acc"]) - fnum(removed["acc"])
    evidence_recovery = fnum(restore_e["acc_recovery_frac"])
    random_recovery = fnum(restore_r["acc_recovery_frac"])
    return [
        {
            "model": model,
            "category": "necessity_and_restoration",
            "diagnostic": "delete selected evidence tokens, then restore evidence vs random tokens",
            "n": selected["n"],
            "primary_metric": "accuracy_drop_selected_minus_removed",
            "primary_effect": fmt(drop),
            "control_metric": "random_restore_all_recovery_fraction",
            "control_effect": fmt(random_recovery),
            "supporting_metric": "evidence_restore_all_recovery_fraction",
            "supporting_effect": fmt(evidence_recovery),
            "strength": "strong" if drop >= 0.025 and evidence_recovery - random_recovery >= 0.5 else "moderate",
            "caveat": caveat_for_model(model, "restoration"),
            "source": str(path.relative_to(ROOT)),
        }
    ]


def build_token_evidence_rows(hard: list[dict[str, str]]) -> list[dict[str, Any]]:
    specs = [
        ("Qwen3-VL-8B", "causal_qwen_0p30", "evidence_topk0p30", "anti_evidence_bottomk0p30"),
        ("LLaVA-1.5-7B", "causal_llava_0p40", "evidence_topk0p40", "anti_evidence_bottomk0p40"),
        (
            "InternVL3.5-8B calibrated",
            "causal_internvl_0p50",
            "evidence_topk0p50_cal",
            "anti_evidence_bottomk0p50_cal",
        ),
    ]
    out = []
    for model, group, kept_label, removed_label in specs:
        kept = find(hard, group=group, label=kept_label)
        removed = find(hard, group=group, label=removed_label)
        acc_gap = fnum(kept["acc"]) - fnum(removed["acc"])
        hFPR_gap = fnum(kept["hFPR"]) - fnum(removed["hFPR"])
        out.append(
            {
                "model": model,
                "category": "token_sufficiency_vs_anti_evidence",
                "diagnostic": "keep evidence-overlap tokens only vs keep anti-evidence tokens",
                "n": kept["n"],
                "primary_metric": "accuracy_kept_minus_removed",
                "primary_effect": fmt(acc_gap),
                "control_metric": "hFPR_kept_minus_removed",
                "control_effect": fmt(hFPR_gap),
                "supporting_metric": "ECR_kept_minus_removed",
                "supporting_effect": fmt(fnum(kept["ECR"]) - fnum(removed["ECR"])),
                "strength": classify_token_gap(model, acc_gap, hFPR_gap),
                "caveat": caveat_for_model(model, "token"),
                "source": str(HARD_EVIDENCE_CSV.relative_to(ROOT)),
            }
        )
    return out


def build_region_rows() -> list[dict[str, Any]]:
    out = []
    for model, path in REGION_SUMMARIES.items():
        rows = read_csv(path)
        kept = find(rows, arm="evidence_kept")
        removed = find(rows, arm="evidence_removed")
        margin_gap = fnum(kept["mean_yes_margin"]) - fnum(removed["mean_yes_margin"])
        support_gap = fnum(kept["mean_target_support"]) - fnum(removed["mean_target_support"])
        out.append(
            {
                "model": model,
                "category": "region_logit_necessity",
                "diagnostic": "compare target support/margins when evidence tokens are kept vs removed",
                "n": kept["n"],
                "primary_metric": "mean_yes_margin_kept_minus_removed",
                "primary_effect": fmt(margin_gap),
                "control_metric": "mean_target_support_kept_minus_removed",
                "control_effect": fmt(support_gap),
                "supporting_metric": "accuracy_kept_minus_removed",
                "supporting_effect": fmt(fnum(kept["acc"]) - fnum(removed["acc"])),
                "strength": classify_signed_effect(model, margin_gap),
                "caveat": caveat_for_model(model, "region"),
                "source": str(path.relative_to(ROOT)),
            }
        )
    return out


def build_occlusion_rows() -> list[dict[str, Any]]:
    out = []
    for model, path in OCCLUSION_SUMMARIES.items():
        rows = read_csv(path)
        orig = find(rows, view="orig")
        evidence = find(rows, view="evidence_masked")
        random = find(rows, view="random_masked")
        pos_drop = fnum(orig["mean_pos_support"]) - fnum(evidence["mean_pos_support"])
        random_drop = fnum(orig["mean_pos_support"]) - fnum(random["mean_pos_support"])
        out.append(
            {
                "model": model,
                "category": "input_space_specificity",
                "diagnostic": "mask evidence bbox vs same-size random bbox",
                "n": orig["n"],
                "primary_metric": "positive_support_drop_evidence_mask",
                "primary_effect": fmt(pos_drop),
                "control_metric": "positive_support_drop_random_mask",
                "control_effect": fmt(random_drop),
                "supporting_metric": "accuracy_orig_minus_evidence_masked",
                "supporting_effect": fmt(fnum(orig["acc"]) - fnum(evidence["acc"])),
                "strength": classify_occlusion(model, pos_drop, random_drop),
                "caveat": caveat_for_model(model, "occlusion"),
                "source": str(path.relative_to(ROOT)),
            }
        )
    return out


def classify_token_gap(model: str, acc_gap: float, hFPR_gap: float) -> str:
    if model.startswith("LLaVA"):
        return "weak_or_mixed"
    if acc_gap >= 0.08:
        return "strong"
    return "moderate"


def classify_signed_effect(model: str, value: float) -> str:
    if model.startswith("LLaVA"):
        return "weak_or_mixed"
    if value >= 0.5:
        return "strong"
    if value > 0:
        return "moderate"
    return "mixed"


def classify_occlusion(model: str, evidence_drop: float, random_drop: float) -> str:
    if model.startswith("LLaVA"):
        return "weak"
    if evidence_drop - random_drop >= 0.5:
        return "strong"
    if evidence_drop > random_drop:
        return "moderate"
    return "mixed"


def caveat_for_model(model: str, kind: str) -> str:
    if model.startswith("LLaVA"):
        return "LLaVA evidence effects are weak or mixed; use as boundary evidence, not as a strong causal proof."
    if model.startswith("InternVL") and kind in {"restoration", "token"}:
        return "InternVL result uses the calibrated hFPR-constrained operating point; report threshold protocol."
    return "Causal-style diagnostic; still not a full semantic text-replacement counterfactual."


def build_report(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Causal Evidence Triad Audit",
        "",
        "This audit reorganizes cached diagnostics around the problem.md request to separate necessity, sufficiency/restoration, and specificity. It does not rerun any model.",
        "",
        "## Main Reading",
        "",
        "- Qwen has the cleanest causal-style support: deleting selected evidence tokens reduces accuracy, restoring evidence recovers the loss, and input-space evidence masking has a much larger effect than random masking.",
        "- InternVL shows the same broad direction under the calibrated hFPR-constrained setting, but the evidence should be reported with the calibration caveat.",
        "- LLaVA remains the weakest causal case: protected pruning preserves annotated evidence, but logit-drop, token-only, and occlusion effects are small or mixed. This should be framed as a backbone limitation rather than hidden.",
        "",
        "## Summary Table",
        "",
        markdown_table(
            rows,
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
    ]
    return "\n".join(lines)


def find(rows: list[dict[str, str]], **criteria: str) -> dict[str, str]:
    return next(row for row in rows if all(row.get(k) == v for k, v in criteria.items()))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(str(row.get(col, "")).replace("|", "/") for col in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def fnum(value: Any) -> float:
    if value in {None, ""}:
        return 0.0
    return float(value)


def fmt(value: Any) -> str:
    return f"{float(value):.3f}"


if __name__ == "__main__":
    main()
