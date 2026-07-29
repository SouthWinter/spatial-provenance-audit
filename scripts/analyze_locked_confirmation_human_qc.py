#!/usr/bin/env python3
"""Audit locked-confirmation hard negatives after exhaustive human QC."""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
QC_DIR = (
    ROOT
    / "runs/problem_optimization_audit/hard_negative_confirmation_full_qc"
)
QC_FILE = QC_DIR / "locked_confirmation_500_adjudicated.csv"
AGREEMENT_FILE = QC_DIR / "secondary_100/pre_adjudication_agreement_summary.csv"
OUTPUT = QC_DIR / "integrated_audit"

RUNS = {
    "Qwen3-VL-8B": {
        "Full": "runs/textocr_confirmation/qwen3_8b_full/probe_scores.jsonl",
        "Target (30%)": (
            "runs/textocr_confirmation/qwen3_8b_target_0p30/probe_scores.jsonl"
        ),
        "Random (30%)": (
            "runs/textocr_confirmation/qwen3_8b_random_0p30/probe_scores.jsonl"
        ),
        "Grid (30%)": (
            "runs/textocr_confirmation/qwen3_8b_grid_0p30/probe_scores.jsonl"
        ),
        "VisionZip (30%)": (
            "runs/textocr_confirmation/qwen3_8b_visionzip_0p30/probe_scores.jsonl"
        ),
    },
    "LLaVA-1.5-7B": {
        "Full": "runs/textocr_confirmation/llava15_7b_full/probe_scores.jsonl",
        "Protected (40%)": (
            "runs/textocr_confirmation/llava15_7b_protected_0p40/probe_scores.jsonl"
        ),
        "Random (40%)": (
            "runs/textocr_confirmation/llava15_7b_random_0p40/probe_scores.jsonl"
        ),
        "Target (40%)": (
            "runs/textocr_confirmation/llava15_7b_target_0p40/probe_scores.jsonl"
        ),
        "SCOPE (40%)": (
            "runs/textocr_confirmation/llava15_7b_scope_0p40/probe_scores.jsonl"
        ),
        "AnchorPrune (40%)": (
            "runs/anchorprune_textocr/confirmation_llava15_anchorprune_0p40/"
            "probe_scores.jsonl"
        ),
        "CoIn (40%)": (
            "runs/textocr_confirmation/llava15_7b_coin_0p40/probe_scores.jsonl"
        ),
        "VisionZip (40%)": (
            "runs/textocr_confirmation/llava15_7b_visionzip_0p40/probe_scores.jsonl"
        ),
    },
}

SCOPES: dict[str, Callable[[dict[str, str]], bool]] = {
    "all_locked_negatives": lambda row: True,
    "target_absence_confirmed": lambda row: (
        row["target_absent_after_case_punct_normalization"].strip().lower() == "yes"
    ),
    "strict_valid_near_miss": lambda row: (
        row["qc_decision"].strip().lower() == "valid_negative"
    ),
}

CORE_FIELDS = (
    "human_source_text_visible",
    "human_target_text_visible_same_image",
    "target_absent_after_case_punct_normalization",
    "source_bbox_matches_source_text",
    "qc_decision",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    fieldnames.extend(
        key for row in rows[1:] for key in row if key not in fieldnames
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def wilson(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(
        rate * (1.0 - rate) / total + z * z / (4.0 * total * total)
    )
    half /= denominator
    return center - half, center + half


def lineage_coverage(row: dict[str, Any]) -> float:
    if str(row.get("prune_selector", "")).strip().lower() in {
        "visionzip",
        "official_visionzip",
        "qwen3_visionzip",
    }:
        return 1.0
    return float(row.get("prune_ecr", 1.0))


def anchor_coverage(row: dict[str, Any]) -> float:
    """Return representative-position coverage for deletion or merging."""
    return float(row.get("prune_anchor_ecr", row.get("prune_ecr", 1.0)))


def binary_auroc(rows: list[dict[str, Any]]) -> float:
    """Compute rank-based AUROC with average ranks for tied margins."""
    ranked = sorted(
        (
            float(row["margin"]),
            1 if row["binary_polarity"] == "positive" else 0,
        )
        for row in rows
    )
    positive_n = sum(label for _, label in ranked)
    negative_n = len(ranked) - positive_n
    if positive_n == 0 or negative_n == 0:
        raise ValueError("AUROC requires both positive and negative examples.")

    positive_rank_sum = 0.0
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(
            label for _, label in ranked[index:end]
        )
        index = end

    return (
        positive_rank_sum - positive_n * (positive_n + 1) / 2.0
    ) / (positive_n * negative_n)


def validate_qc() -> list[dict[str, str]]:
    rows = read_csv(QC_FILE)
    ids = [row["sample_id"] for row in rows]
    incomplete = [
        row["sample_id"]
        for row in rows
        if any(not row.get(field, "").strip() for field in CORE_FIELDS)
    ]
    if len(rows) != 500 or len(set(ids)) != 500:
        raise RuntimeError(
            f"Expected 500 unique adjudicated rows, found {len(rows)}/{len(set(ids))}."
        )
    if incomplete:
        raise RuntimeError(f"Incomplete adjudicated rows: {incomplete[:5]}")
    if any(row.get("secondary_reviewed") == "yes" for row in rows) is False:
        raise RuntimeError("No independently reviewed rows found.")
    return rows


def summarize_qc(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    decisions = Counter(row["qc_decision"] for row in rows)
    target_absence = Counter(
        row["target_absent_after_case_punct_normalization"] for row in rows
    )
    agreement = {
        (row["field"], row["metric"]): row["value"]
        for row in read_csv(AGREEMENT_FILE)
    }
    summary = [
        {"scope": "all_500", "metric": "rows", "value": len(rows)},
        {
            "scope": "all_500",
            "metric": "strict_valid_near_miss",
            "value": decisions["valid_negative"],
        },
        {
            "scope": "all_500",
            "metric": "invalid_source_not_visible",
            "value": decisions["invalid_source_not_visible"],
        },
        {
            "scope": "all_500",
            "metric": "invalid_target_present",
            "value": decisions["invalid_target_present"],
        },
        {
            "scope": "all_500",
            "metric": "target_absence_confirmed",
            "value": target_absence["yes"],
        },
        {
            "scope": "secondary_100_pre_adjudication",
            "metric": "rows",
            "value": 100,
        },
    ]
    for field, metric in (
        ("qc_decision", "agreement"),
        ("qc_decision", "cohen_kappa"),
        ("validity_binary", "agreement"),
        ("validity_binary", "cohen_kappa"),
        ("all_core_fields", "exact_agreement"),
        ("all_core_fields", "disagreement_rows"),
    ):
        summary.append(
            {
                "scope": "secondary_100_pre_adjudication",
                "metric": f"{field}_{metric}",
                "value": agreement[(field, metric)],
            }
        )
    return summary


def is_correct(row: dict[str, Any]) -> bool:
    expected = "no" if row["binary_polarity"] == "negative" else "yes"
    return str(row["pred_answer"]).strip().lower() == expected


def analyze_runs(
    qc_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], list[dict[str, Any]]]]:
    qc = {row["sample_id"]: row for row in qc_rows}
    output = []
    cached: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for model, methods in RUNS.items():
        for method, relative in methods.items():
            score_rows = read_jsonl(ROOT / relative)
            if len(score_rows) != 1000:
                raise RuntimeError(
                    f"{model}/{method}: expected 1000 scores, found {len(score_rows)}."
                )
            cached[(model, method)] = score_rows
            negatives = {
                row["sample_id"]: row
                for row in score_rows
                if row["binary_polarity"] == "negative"
            }
            if set(negatives) != set(qc):
                raise RuntimeError(
                    f"{model}/{method}: prediction/QC sample IDs do not match."
                )
            all_hfpr: float | None = None
            for scope, include in SCOPES.items():
                selected_qc = [row for row in qc_rows if include(row)]
                selected = [negatives[row["sample_id"]] for row in selected_qc]
                false_positives = sum(
                    row["pred_answer"].strip().lower() == "yes" for row in selected
                )
                hfpr = false_positives / len(selected)
                if all_hfpr is None:
                    all_hfpr = hfpr
                low, high = wilson(false_positives, len(selected))
                image_ids = {row["image_id"] for row in selected_qc}
                paired = [row for row in score_rows if row["image_id"] in image_ids]
                if len(paired) != 2 * len(image_ids):
                    raise RuntimeError(
                        f"{model}/{method}/{scope}: incomplete positive-negative pairs."
                    )
                positives = [
                    row for row in paired if row["binary_polarity"] == "positive"
                ]
                paired_negatives = [
                    row for row in paired if row["binary_polarity"] == "negative"
                ]
                accuracy = sum(is_correct(row) for row in paired) / len(paired)
                auroc = binary_auroc(paired)
                pos_ecr = sum(lineage_coverage(row) for row in positives) / len(
                    positives
                )
                anchor_ecr = sum(anchor_coverage(row) for row in positives) / len(
                    positives
                )
                neg_src = sum(
                    lineage_coverage(row) for row in paired_negatives
                ) / len(paired_negatives)
                output.append(
                    {
                        "model": model,
                        "method": method,
                        "scope": scope,
                        "negative_n": len(selected),
                        "false_positives": false_positives,
                        "hFPR": f"{hfpr:.6f}",
                        "hFPR_ci95_low": f"{low:.6f}",
                        "hFPR_ci95_high": f"{high:.6f}",
                        "delta_hFPR_vs_all": f"{hfpr - all_hfpr:.6f}",
                        "paired_probe_n": len(paired),
                        "paired_accuracy": f"{accuracy:.6f}",
                        "delta_accuracy_vs_all": "",
                        "AUROC": f"{auroc:.6f}",
                        "delta_AUROC_vs_all": "",
                        "PosECR": f"{pos_ecr:.6f}",
                        "delta_PosECR_vs_all": "",
                        "AncECR": f"{anchor_ecr:.6f}",
                        "delta_AncECR_vs_all": "",
                        "NegSRC": f"{neg_src:.6f}",
                        "delta_NegSRC_vs_all": "",
                    }
                )
            method_rows = output[-len(SCOPES) :]
            all_metrics = {
                key: float(method_rows[0][key])
                for key in ("paired_accuracy", "AUROC", "PosECR", "AncECR", "NegSRC")
            }
            for row in method_rows:
                row["delta_accuracy_vs_all"] = (
                    f"{float(row['paired_accuracy']) - all_metrics['paired_accuracy']:.6f}"
                )
                for metric in ("AUROC", "PosECR", "AncECR", "NegSRC"):
                    row[f"delta_{metric}_vs_all"] = (
                        f"{float(row[metric]) - all_metrics[metric]:.6f}"
                    )
    return output, cached


def paired_bootstrap(
    qc_rows: list[dict[str, str]],
    cached: dict[tuple[str, str], list[dict[str, Any]]],
    draws: int = 10_000,
) -> list[dict[str, Any]]:
    full = cached[("Qwen3-VL-8B", "Full")]
    target = cached[("Qwen3-VL-8B", "Target (30%)")]
    full_by_id = {row["sample_id"]: row for row in full}
    target_by_id = {row["sample_id"]: row for row in target}
    rows = []
    rng = random.Random(20260724)
    for scope, include in SCOPES.items():
        image_ids = [row["image_id"] for row in qc_rows if include(row)]
        differences = []
        for image_id in image_ids:
            sample_ids = (
                f"textocr-{image_id}:hard-small-pos",
                f"textocr-{image_id}:hard-nearmiss-neg",
            )
            target_acc = sum(is_correct(target_by_id[sid]) for sid in sample_ids) / 2
            full_acc = sum(is_correct(full_by_id[sid]) for sid in sample_ids) / 2
            differences.append(target_acc - full_acc)
        observed = sum(differences) / len(differences)
        draws_out = []
        for _ in range(draws):
            draws_out.append(
                sum(rng.choice(differences) for _ in differences) / len(differences)
            )
        draws_out.sort()
        low = draws_out[int(0.025 * draws)]
        high = draws_out[int(0.975 * draws)]
        rows.append(
            {
                "comparison": "Qwen Target (30%) - Full",
                "scope": scope,
                "image_n": len(image_ids),
                "accuracy_difference": f"{observed:.6f}",
                "ci95_low": f"{low:.6f}",
                "ci95_high": f"{high:.6f}",
                "bootstrap_draws": draws,
                "seed": 20260724,
            }
        )
    return rows


def markdown(
    summary: list[dict[str, Any]],
    sensitivity: list[dict[str, Any]],
    paired: list[dict[str, Any]],
) -> str:
    values = {(row["scope"], row["metric"]): row["value"] for row in summary}
    lines = [
        "# Locked-Confirmation Human-QC Audit",
        "",
        "All 500 locked-confirmation hard negatives received final human QC. "
        "A frozen 100-row subset was independently labeled before adjudication.",
        "",
        "## Label Quality",
        "",
        f"- Strictly valid near-miss probes: {values[('all_500', 'strict_valid_near_miss')]}/500.",
        f"- Target absence confirmed after normalization: {values[('all_500', 'target_absence_confirmed')]}/500.",
        f"- Source not visually identifiable: {values[('all_500', 'invalid_source_not_visible')]}/500.",
        f"- Target present in the same image: {values[('all_500', 'invalid_target_present')]}/500.",
        "- Frozen secondary subset: "
        f"{float(values[('secondary_100_pre_adjudication', 'validity_binary_agreement')]):.3f} "
        "binary-validity agreement, "
        f"$\\kappa={float(values[('secondary_100_pre_adjudication', 'validity_binary_cohen_kappa')]):.3f}$; "
        f"{values[('secondary_100_pre_adjudication', 'all_core_fields_disagreement_rows')]} "
        "rows adjudicated.",
        "",
        "## hFPR and Paired-Accuracy Sensitivity",
        "",
        "| Model | Method | Original hFPR | Human-valid hFPR | $\\Delta$hFPR | Original acc. | Human-valid acc. | $\\Delta$acc. |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in sensitivity:
        grouped.setdefault((row["model"], row["method"]), {})[row["scope"]] = row
    for (model, method), scoped in grouped.items():
        original = scoped["all_locked_negatives"]
        valid = scoped["strict_valid_near_miss"]
        lines.append(
            f"| {model} | {method} | {float(original['hFPR']):.3f} | "
            f"{float(valid['hFPR']):.3f} | {float(valid['delta_hFPR_vs_all']):+.3f} | "
            f"{float(original['paired_accuracy']):.3f} | "
            f"{float(valid['paired_accuracy']):.3f} | "
            f"{float(valid['delta_accuracy_vs_all']):+.3f} |"
        )
    lines.extend(
        [
            "",
            "The original 500-image locked table remains the prespecified primary readout. "
            "The 465-image human-valid subset is a post-QC sensitivity analysis.",
            "",
            "## Complete Valid-465 Readout",
            "",
            "| Model | Method | Accuracy | hFPR | AUROC | PosECR | AncECR | NegSRC |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for (model, method), scoped in grouped.items():
        valid = scoped["strict_valid_near_miss"]
        lines.append(
            f"| {model} | {method} | {float(valid['paired_accuracy']):.3f} | "
            f"{float(valid['hFPR']):.3f} | {float(valid['AUROC']):.3f} | "
            f"{float(valid['PosECR']):.3f} | {float(valid['AncECR']):.3f} | "
            f"{float(valid['NegSRC']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Primary Paired Comparison",
            "",
        ]
    )
    for row in paired:
        lines.append(
            f"- {row['scope']}: $n={row['image_n']}$, Target--Full accuracy "
            f"{float(row['accuracy_difference']):+.3f} "
            f"[{float(row['ci95_low']):+.3f},{float(row['ci95_high']):+.3f}]."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    qc_rows = validate_qc()
    summary = summarize_qc(qc_rows)
    sensitivity, cached = analyze_runs(qc_rows)
    paired = paired_bootstrap(qc_rows, cached)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "locked_confirmation_human_qc_summary.csv", summary)
    write_csv(OUTPUT / "locked_confirmation_hfpr_sensitivity.csv", sensitivity)
    write_csv(
        OUTPUT / "locked_confirmation_valid465_full_metrics.csv",
        [row for row in sensitivity if row["scope"] == "strict_valid_near_miss"],
    )
    write_csv(OUTPUT / "locked_confirmation_paired_bootstrap.csv", paired)
    (OUTPUT / "locked_confirmation_human_qc_report.md").write_text(
        markdown(summary, sensitivity, paired), encoding="utf-8"
    )
    print(f"Wrote locked-confirmation human-QC audit to {OUTPUT}")


if __name__ == "__main__":
    main()
