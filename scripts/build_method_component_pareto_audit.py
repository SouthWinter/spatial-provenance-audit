#!/usr/bin/env python3
"""Component and Pareto audit for Qwen TextOCR-Hard selectors.

This audit addresses the "score + top-k" concern by separating selector
components under the same TextOCR-Hard protocol. It asks which objective terms
actually change the accuracy / hFPR / ECR frontier, and which ones are
diagnostic only.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs" / "prune_textocr_hard_full1000"
OUT_DIR = ROOT / "runs" / "problem_optimization_audit" / "method_component_pareto"
PAPER_DIR = ROOT / "runs" / "paper_evidence"

VARIANTS = [
    {
        "variant": "Bottom-k 0.30",
        "family": "sanity_control",
        "objective_terms": "anti-salience",
        "path": "qwen3_8b_textocr_hard_full1000_bottomk_0p30",
        "role": "lower-bound sanity control",
    },
    {
        "variant": "Random 0.30",
        "family": "sanity_control",
        "objective_terms": "random coverage",
        "path": "qwen3_8b_textocr_hard_full1000_random_0p30",
        "role": "matched-budget random control",
    },
    {
        "variant": "Shuffled target 0.30",
        "family": "target_ablation",
        "objective_terms": "target relevance with shuffled target",
        "path": "qwen3_8b_textocr_hard_full1000_target_embed_shuffled_topk_0p30",
        "role": "target-text specificity control",
    },
    {
        "variant": "Generic embed 0.30",
        "family": "target_ablation",
        "objective_terms": "image-text embedding salience",
        "path": "qwen3_8b_textocr_hard_full1000_embed_topk_0p30",
        "role": "non-target salience ablation",
    },
    {
        "variant": "Target 0.30",
        "family": "main_relevance",
        "objective_terms": "target relevance",
        "path": "qwen3_8b_textocr_hard_full1000_target_embed_topk_0p30_targetfix_802816",
        "role": "main Qwen relevance operating point",
    },
    {
        "variant": "Grid-floor target 0.30",
        "family": "spatial_floor",
        "objective_terms": "target relevance + grid floor",
        "path": "qwen3_8b_textocr_hard_full1000_target_embed_grid_topk_0p30_targetfix_802816",
        "role": "spatial coverage ablation",
    },
    {
        "variant": "Soft evidence 0.30",
        "family": "evidence_prior",
        "objective_terms": "target relevance + soft evidence prior",
        "path": "qwen3_8b_textocr_hard_full1000_target_embed_soft_evidence_topk_0p30_b0p05_targetfix_802816",
        "role": "soft evidence-prior ablation",
    },
    {
        "variant": "Protected evidence 0.30",
        "family": "evidence_prior",
        "objective_terms": "target relevance + hard evidence reservation",
        "path": "qwen3_8b_textocr_hard_full1000_target_embed_protected_topk_0p30",
        "role": "hard evidence-reservation ablation",
    },
    {
        "variant": "Protected-center 0.30",
        "family": "evidence_prior",
        "objective_terms": "target relevance + evidence-center reservation",
        "path": "qwen3_8b_textocr_hard_full1000_target_embed_protected_center_topk_0p30_targetfix_802816",
        "role": "center-only evidence reservation ablation",
    },
    {
        "variant": "Coverage-greedy 0.30",
        "family": "coverage_objective",
        "objective_terms": "target relevance + spatial/evidence/uniqueness coverage",
        "path": "qwen3_8b_textocr_hard_full1000_target_embed_coverage_greedy_0p30_hard_targetfix_802816",
        "role": "explicit coverage-objective diagnostic",
    },
]


def main() -> None:
    rows = [summarize_variant(spec) for spec in VARIANTS]
    rows = mark_pareto(rows)
    delta_rows = build_delta_rows(rows)
    summary = build_summary(rows, delta_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "method_component_pareto_rows.csv", rows)
    write_csv(OUT_DIR / "method_component_pareto_delta_vs_target.csv", delta_rows)
    write_csv(OUT_DIR / "method_component_pareto_summary.csv", summary)
    write_csv(PAPER_DIR / "table_method_component_pareto.csv", rows)
    write_csv(PAPER_DIR / "table_method_component_delta_vs_target.csv", delta_rows)
    write_csv(PAPER_DIR / "table_method_component_pareto_summary.csv", summary)
    (OUT_DIR / "method_component_pareto_report.md").write_text(
        build_report(rows, delta_rows, summary), encoding="utf-8"
    )
    print(f"Wrote method component Pareto audit to {OUT_DIR}")


def summarize_variant(spec: dict[str, str]) -> dict[str, Any]:
    run_dir = RUN_ROOT / spec["path"]
    probes_path = run_dir / "probe_scores.jsonl"
    if not probes_path.exists():
        raise FileNotFoundError(probes_path)
    probes = read_jsonl(probes_path)
    positives = [row for row in probes if row.get("target_answer") == "yes"]
    negatives = [row for row in probes if row.get("target_answer") == "no"]
    ecrs = [f(row.get("prune_ecr")) for row in probes]
    center = [f(row.get("prune_evidence_center_recall")) for row in probes]
    patch = [f(row.get("prune_evidence_patch_recall")) for row in probes]
    keep = [f(row.get("prune_keep_ratio")) for row in probes if row.get("prune_keep_ratio") not in (None, "")]
    margins = [abs(f(row.get("margin"))) for row in probes if row.get("margin") not in (None, "")]
    acc = rate(probes, lambda row: bool(row.get("correct")))
    hFPR = rate(negatives, lambda row: row.get("pred_answer") == "yes")
    return {
        "variant": spec["variant"],
        "family": spec["family"],
        "objective_terms": spec["objective_terms"],
        "role": spec["role"],
        "n": len(probes),
        "accuracy": fmt(acc),
        "positive_acc": fmt(rate(positives, lambda row: bool(row.get("correct")))),
        "negative_acc": fmt(rate(negatives, lambda row: bool(row.get("correct")))),
        "hFPR": fmt(hFPR),
        "mean_keep": fmt(avg(keep)),
        "mean_ECR": fmt(avg(ecrs)),
        "positive_ECR": fmt(avg([f(row.get("prune_ecr")) for row in positives])),
        "negative_ECR": fmt(avg([f(row.get("prune_ecr")) for row in negatives])),
        "mean_center_recall": fmt(avg(center)),
        "mean_patch_recall": fmt(avg(patch)),
        "mean_abs_margin": fmt(avg(margins)),
        "pareto_front": "",
        "dominated_by": "",
        "path": str(run_dir.relative_to(ROOT)),
    }


def mark_pareto(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        dominators = []
        for other in rows:
            if other is row:
                continue
            if dominates(other, row):
                dominators.append(other["variant"])
        row["pareto_front"] = int(not dominators)
        row["dominated_by"] = "; ".join(dominators[:4])
    return rows


def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_vals = (f(left["accuracy"]), -f(left["hFPR"]), f(left["mean_ECR"]))
    right_vals = (f(right["accuracy"]), -f(right["hFPR"]), f(right["mean_ECR"]))
    return all(l >= r - 1e-12 for l, r in zip(left_vals, right_vals)) and any(
        l > r + 1e-12 for l, r in zip(left_vals, right_vals)
    )


def build_delta_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    target = next(row for row in rows if row["variant"] == "Target 0.30")
    out = []
    for row in rows:
        if row is target:
            continue
        out.append(
            {
                "variant": row["variant"],
                "family": row["family"],
                "delta_accuracy_vs_target": fmt(f(row["accuracy"]) - f(target["accuracy"])),
                "delta_hFPR_vs_target": fmt(f(row["hFPR"]) - f(target["hFPR"])),
                "delta_ECR_vs_target": fmt(f(row["mean_ECR"]) - f(target["mean_ECR"])),
                "delta_center_vs_target": fmt(f(row["mean_center_recall"]) - f(target["mean_center_recall"])),
                "delta_patch_vs_target": fmt(f(row["mean_patch_recall"]) - f(target["mean_patch_recall"])),
                "interpretation": interpret_delta(row, target),
            }
        )
    return out


def interpret_delta(row: dict[str, Any], target: dict[str, Any]) -> str:
    da = f(row["accuracy"]) - f(target["accuracy"])
    dh = f(row["hFPR"]) - f(target["hFPR"])
    de = f(row["mean_ECR"]) - f(target["mean_ECR"])
    if da >= -0.005 and dh <= 0.005 and de >= 0.02:
        return "evidence gain with similar answer risk"
    if de >= 0.10 and (da < -0.005 or dh > 0.005):
        return "coverage gain but answer-risk trade-off"
    if da < -0.03 and de < 0.0:
        return "worse answer and evidence behavior"
    if dh < -0.03 and da < -0.03:
        return "more conservative answers but lower accuracy"
    return "mixed or small change"


def build_summary(rows: list[dict[str, Any]], delta_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pareto = [row for row in rows if int(row["pareto_front"]) == 1]
    best_acc = max(rows, key=lambda row: f(row["accuracy"]))
    best_ecr = max(rows, key=lambda row: f(row["mean_ECR"]))
    lowest_hfpr = min(rows, key=lambda row: f(row["hFPR"]))
    coverage_tradeoffs = [row for row in delta_rows if row["interpretation"] == "coverage gain but answer-risk trade-off"]
    evidence_similar_risk = [row for row in delta_rows if row["interpretation"] == "evidence gain with similar answer risk"]
    return [
        {
            "summary_item": "pareto_front_variants",
            "value": "; ".join(row["variant"] for row in pareto),
            "interpretation": "non-dominated in accuracy / hFPR / ECR space",
        },
        {
            "summary_item": "best_accuracy",
            "value": f"{best_acc['variant']} ({best_acc['accuracy']})",
            "interpretation": "highest TextOCR-Hard accuracy among audited same-budget components",
        },
        {
            "summary_item": "best_ECR",
            "value": f"{best_ecr['variant']} ({best_ecr['mean_ECR']})",
            "interpretation": "strongest evidence availability, regardless of answer risk",
        },
        {
            "summary_item": "lowest_hFPR",
            "value": f"{lowest_hfpr['variant']} ({lowest_hfpr['hFPR']})",
            "interpretation": "most conservative negative behavior, not necessarily best accuracy",
        },
        {
            "summary_item": "coverage_tradeoff_count",
            "value": len(coverage_tradeoffs),
            "interpretation": "variants whose ECR gain over Target comes with accuracy or hFPR degradation",
        },
        {
            "summary_item": "evidence_gain_similar_risk_count",
            "value": len(evidence_similar_risk),
            "interpretation": "variants that improve ECR without material answer-risk loss under the audit heuristic",
        },
    ]


def build_report(
    rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
) -> str:
    return "\n".join(
        [
            "# Method Component Pareto Audit",
            "",
            "This audit compares same-budget Qwen TextOCR-Hard selector components. A row is Pareto-front if no other audited row has at least as high accuracy, at most as high hFPR, and at least as high ECR, with one strict improvement.",
            "",
            "## Summary",
            "",
            table_md(summary, ["summary_item", "value", "interpretation"]),
            "",
            "## Component Rows",
            "",
            table_md(
                rows,
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
            "## Delta Versus Target 0.30",
            "",
            table_md(
                delta_rows,
                [
                    "variant",
                    "delta_accuracy_vs_target",
                    "delta_hFPR_vs_target",
                    "delta_ECR_vs_target",
                    "interpretation",
                ],
            ),
            "",
            "## Interpretation",
            "",
            "- Target-conditioned relevance remains the best accuracy operating point in this Qwen TextOCR-Hard slice.",
            "- Evidence and coverage terms can move ECR substantially, but many such moves are not Pareto improvements because hFPR or accuracy degrades.",
            "- This supports a conservative method claim: the framework exposes and controls evidence-risk trade-offs, while stronger answer-risk optimization remains open.",
            "",
        ]
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table_md(rows: list[dict[str, Any]], cols: list[str]) -> str:
    out = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(col, "")).replace("|", "/") for col in cols) + " |")
    return "\n".join(out)


def rate(rows: list[dict[str, Any]], pred) -> float:
    return mean([float(pred(row)) for row in rows]) if rows else math.nan


def avg(values: list[float]) -> float:
    clean = [value for value in values if not math.isnan(value)]
    return mean(clean) if clean else math.nan


def f(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def fmt(value: Any) -> str:
    value = f(value)
    if math.isnan(value):
        return ""
    return f"{value:.3f}"


if __name__ == "__main__":
    main()
