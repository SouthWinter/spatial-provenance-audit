#!/usr/bin/env python3
"""Integrate all 500 development hard-negative QC decisions and audit sensitivity."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
QC_ROOT = ROOT / "runs/problem_optimization_audit"
ORIGINAL_QC = QC_ROOT / "hard_negative_human_qc_launch/hard_negative_human_qc_template.csv"
EXTENSION_QC = (
    QC_ROOT
    / "hard_negative_full_qc_extension/hard_negative_remaining_400_adjudicated.csv"
)
AGREEMENT = (
    QC_ROOT
    / "hard_negative_full_qc_extension/secondary_100/pre_adjudication_agreement_summary.csv"
)
THRESHOLDS = (
    ROOT
    / "runs/internvl_textocr_hard/shared_threshold_audit/internvl_threshold_decomposition.csv"
)
OUTPUT = QC_ROOT / "hard_negative_full_qc_extension/integrated_audit"

RUNS = {
    "Qwen3-VL-8B": {
        "Full": (
            "runs/prune_textocr_hard_full1000/"
            "qwen3_8b_textocr_hard_full1000_topk_1p00/probe_scores.jsonl"
        ),
        "Target (30%)": (
            "runs/prune_textocr_hard_full1000/"
            "qwen3_8b_textocr_hard_full1000_target_embed_topk_0p30_targetfix_802816/"
            "probe_scores.jsonl"
        ),
        "Random (30%)": (
            "runs/prune_textocr_hard_full1000/"
            "qwen3_8b_textocr_hard_full1000_random_0p30/probe_scores.jsonl"
        ),
        "Grid (30%)": (
            "runs/prune_textocr_hard_full1000/"
            "qwen3_8b_textocr_hard_full1000_grid_0p30/probe_scores.jsonl"
        ),
        "VisionZip (30%)": (
            "runs/qwen3_visionzip_textocr_hard/"
            "qwen3_8b_textocr_hard_full1000_visionzip_0p30_minmax802816/"
            "probe_scores.jsonl"
        ),
    },
    "LLaVA-1.5-7B": {
        "Full": (
            "runs/llava_textocr_hard/"
            "llava15_7b_textocr_hard_full1000_direct/probe_scores.jsonl"
        ),
        "Protected (40%)": (
            "runs/llava_textocr_hard/"
            "llava15_7b_textocr_hard_full1000_embed_protected_topk_0p40/"
            "probe_scores.jsonl"
        ),
        "Random (40%)": (
            "runs/llava_textocr_hard/"
            "llava15_7b_textocr_hard_full1000_random_0p40/probe_scores.jsonl"
        ),
        "Target (40%)": (
            "runs/llava_textocr_hard/"
            "llava15_7b_textocr_hard_full1000_target_embed_topk_0p40/probe_scores.jsonl"
        ),
        "SCOPE (40%)": (
            "runs/llava_textocr_hard/"
            "llava15_7b_textocr_hard_full1000_scope_0p40/probe_scores.jsonl"
        ),
        "AnchorPrune (40%)": (
            "runs/anchorprune_textocr/development_llava15_anchorprune_0p40/probe_scores.jsonl"
        ),
        "CoIn (40%)": (
            "runs/llava_textocr_hard/"
            "llava15_7b_textocr_hard_full1000_coin_0p40/probe_scores.jsonl"
        ),
        "VisionZip (40%)": (
            "runs/llava_textocr_hard/"
            "llava15_7b_textocr_hard_full1000_visionzip_0p40/probe_scores.jsonl"
        ),
    },
    "InternVL3.5-8B": {
        "Full": (
            "runs/internvl_textocr_hard/"
            "internvl35_8b_textocr_hard_full1000_direct/probe_scores.jsonl"
        ),
        "Soft evidence (50%)": (
            "runs/internvl_textocr_hard/"
            "internvl35_8b_textocr_hard_full1000_target_embed_soft_evidence_topk_0p50_b0p05/"
            "probe_scores.jsonl"
        ),
        "Random (50%)": (
            "runs/internvl_textocr_hard/"
            "internvl35_8b_textocr_hard_full1000_random_0p50/probe_scores.jsonl"
        ),
        "Grid (50%)": (
            "runs/internvl_textocr_hard/"
            "internvl35_8b_textocr_hard_full1000_grid_0p50/probe_scores.jsonl"
        ),
    },
}

SCOPES: dict[str, Callable[[dict[str, str]], bool]] = {
    "all_development_negatives": lambda row: True,
    "target_absence_confirmed": lambda row: (
        row["target_absent_after_case_punct_normalization"].strip().lower() == "yes"
    ),
    "strict_valid_near_miss": lambda row: row["qc_decision"].strip().lower() == "valid_negative",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
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
    if total == 0:
        return 0.0, 0.0
    z = 1.959963984540054
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total))
    half /= denominator
    return center - half, center + half


def full_shared_threshold() -> float:
    for row in read_csv(THRESHOLDS):
        if row["method"] == "Full":
            return float(row["full_shared_threshold"])
    raise RuntimeError("Full shared threshold is missing.")


def lineage_coverage(row: dict[str, Any]) -> float:
    if str(row.get("prune_selector", "")).strip().lower() in {
        "visionzip",
        "official_visionzip",
        "qwen3_visionzip",
    }:
        return 1.0
    return float(row.get("prune_ecr", 1.0))


def integrate_qc() -> list[dict[str, str]]:
    original = read_csv(ORIGINAL_QC)
    extension = read_csv(EXTENSION_QC)
    rows = original + extension
    ids = [row["sample_id"] for row in rows]
    if len(original) != 100 or len(extension) != 400:
        raise RuntimeError(f"Expected 100+400 QC rows, found {len(original)}+{len(extension)}.")
    if len(set(ids)) != 500:
        raise RuntimeError("The combined QC rows are not 500 unique sample IDs.")
    core = (
        "human_source_text_visible",
        "human_target_text_visible_same_image",
        "target_absent_after_case_punct_normalization",
        "source_bbox_matches_source_text",
        "qc_decision",
    )
    incomplete = [row["sample_id"] for row in rows if any(not row[field].strip() for field in core)]
    if incomplete:
        raise RuntimeError(f"Incomplete final QC rows: {incomplete[:5]}")
    return rows


def summarize_qc(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    decisions = Counter(row["qc_decision"] for row in rows)
    target_visible = Counter(row["human_target_text_visible_same_image"] for row in rows)
    target_absent = Counter(
        row["target_absent_after_case_punct_normalization"] for row in rows
    )
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
        {"scope": "all_500", "metric": "unclear", "value": decisions["unclear"]},
        {
            "scope": "all_500",
            "metric": "target_visible_yes",
            "value": target_visible["yes"],
        },
        {
            "scope": "all_500",
            "metric": "target_absence_confirmed",
            "value": target_absent["yes"],
        },
    ]
    agreement = {
        (row["field"], row["metric"]): row["value"] for row in read_csv(AGREEMENT)
    }
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


def analyze_runs(qc_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    qc = {row["sample_id"]: row for row in qc_rows}
    threshold = full_shared_threshold()
    output = []
    for model, methods in RUNS.items():
        for method, relative in methods.items():
            score_rows = [
                row
                for row in read_jsonl(ROOT / relative)
                if row.get("binary_polarity") == "negative"
            ]
            missing = [row["sample_id"] for row in score_rows if row["sample_id"] not in qc]
            if missing:
                raise RuntimeError(f"{model}/{method} has QC-missing IDs: {missing[:5]}")
            all_hfpr = None
            for scope, include in SCOPES.items():
                selected = [row for row in score_rows if include(qc[row["sample_id"]])]
                if model == "InternVL3.5-8B":
                    predictions = [
                        float(row["no_loss"]) - float(row["yes_loss"]) >= threshold
                        for row in selected
                    ]
                else:
                    predictions = [row["pred_answer"] == "yes" for row in selected]
                false_positives = sum(predictions)
                hfpr = false_positives / len(selected)
                if all_hfpr is None:
                    all_hfpr = hfpr
                low, high = wilson(false_positives, len(selected))
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
                        "NegSRC": f"{sum(lineage_coverage(row) for row in selected) / len(selected):.6f}",
                        "internvl_threshold": (
                            f"{threshold:.12f}" if model == "InternVL3.5-8B" else ""
                        ),
                    }
                )
    return output


def markdown(qc_summary: list[dict[str, Any]], sensitivity: list[dict[str, Any]]) -> str:
    values = {(row["scope"], row["metric"]): row["value"] for row in qc_summary}
    lines = [
        "# Full Development Hard-Negative Human-QC and Sensitivity Audit",
        "",
        "All 500 development-set hard negatives received final human QC. "
        "The original 100 rows and the adjudicated 400-row extension are disjoint.",
        "",
        "## Label and Source-Region Quality",
        "",
        f"- Strictly valid near-miss probes: {values[('all_500', 'strict_valid_near_miss')]}/500.",
        f"- Target absence confirmed after normalization: {values[('all_500', 'target_absence_confirmed')]}/500.",
        f"- Target observed in the same image: {values[('all_500', 'target_visible_yes')]}/500.",
        f"- Source not visually identifiable: {values[('all_500', 'invalid_source_not_visible')]}/500.",
        f"- Unclear: {values[('all_500', 'unclear')]}/500.",
        "",
        "The target-absence scope audits negative-label validity. The strict scope additionally "
        "requires a visible source and matching source box, which is necessary for interpreting NegSRC.",
        "",
        "## Independent Reliability",
        "",
        "- A second non-author annotator independently reviewed a frozen 100-row subset before adjudication.",
        "- QC-decision agreement/kappa: "
        f"{float(values[('secondary_100_pre_adjudication', 'qc_decision_agreement')]):.3f}/"
        f"{float(values[('secondary_100_pre_adjudication', 'qc_decision_cohen_kappa')]):.3f}.",
        "- Valid-vs-non-valid agreement/kappa: "
        f"{float(values[('secondary_100_pre_adjudication', 'validity_binary_agreement')]):.3f}/"
        f"{float(values[('secondary_100_pre_adjudication', 'validity_binary_cohen_kappa')]):.3f}.",
        "- Exact five-field agreement: "
        f"{float(values[('secondary_100_pre_adjudication', 'all_core_fields_exact_agreement')]):.3f}; "
        f"{values[('secondary_100_pre_adjudication', 'all_core_fields_disagreement_rows')]} rows were adjudicated.",
        "",
        "## hFPR Sensitivity",
        "",
        "| Model | Method | All $n$/hFPR | Absence-confirmed $n$/hFPR | Strict $n$/hFPR | Max $|\\Delta|$ |",
        "|---|---|---:|---:|---:|---:|",
    ]
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in sensitivity:
        grouped.setdefault((row["model"], row["method"]), {})[row["scope"]] = row
    for (model, method), rows in grouped.items():
        all_row = rows["all_development_negatives"]
        absent = rows["target_absence_confirmed"]
        strict = rows["strict_valid_near_miss"]
        max_delta = max(
            abs(float(absent["delta_hFPR_vs_all"])),
            abs(float(strict["delta_hFPR_vs_all"])),
        )
        lines.append(
            f"| {model} | {method} | {all_row['negative_n']}/{float(all_row['hFPR']):.3f} | "
            f"{absent['negative_n']}/{float(absent['hFPR']):.3f} | "
            f"{strict['negative_n']}/{float(strict['hFPR']):.3f} | {max_delta:.3f} |"
        )
    lines.extend(
        [
            "",
            "The all-development rows remain the original readout. The two filtered columns are "
            "post-QC sensitivity analyses; they do not replace or validate the locked confirmation set.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    qc_rows = integrate_qc()
    qc_summary = summarize_qc(qc_rows)
    sensitivity = analyze_runs(qc_rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "hard_negative_all500_adjudicated.csv", qc_rows)
    write_csv(OUTPUT / "hard_negative_all500_qc_summary.csv", qc_summary)
    write_csv(OUTPUT / "hard_negative_result_sensitivity.csv", sensitivity)
    (OUTPUT / "hard_negative_full_qc_report.md").write_text(
        markdown(qc_summary, sensitivity), encoding="utf-8"
    )
    print(f"Wrote full hard-negative QC audit to {OUTPUT}")


if __name__ == "__main__":
    main()
