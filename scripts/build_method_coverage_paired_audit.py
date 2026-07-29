#!/usr/bin/env python3
"""Paired audit for target top-k versus coverage-greedy selection.

This audit explains the method-principle boundary: coverage-greedy improves
evidence coverage at the same budget, but does not necessarily improve answer
behavior. It joins cached TextOCR-Hard probe-level outputs and does not rerun
the model.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TARGET_RUN = (
    ROOT
    / "runs"
    / "prune_textocr_hard_full1000"
    / "qwen3_8b_textocr_hard_full1000_target_embed_topk_0p30_targetfix_802816"
)
COVERAGE_RUN = (
    ROOT
    / "runs"
    / "prune_textocr_hard_full1000"
    / "qwen3_8b_textocr_hard_full1000_target_embed_coverage_greedy_0p30_hard_targetfix_802816"
)
OUT_DIR = ROOT / "runs" / "problem_optimization_audit" / "method_coverage_paired_audit"


def main() -> None:
    target = {row["sample_id"]: row for row in read_jsonl(TARGET_RUN / "probe_scores.jsonl")}
    coverage = {row["sample_id"]: row for row in read_jsonl(COVERAGE_RUN / "probe_scores.jsonl")}
    rows = [join_row(sample_id, target[sample_id], coverage[sample_id]) for sample_id in sorted(target.keys() & coverage.keys())]
    summary = build_summary(rows)
    group_summary = build_group_summary(rows)
    examples = build_examples(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "method_coverage_paired_rows.csv", rows)
    write_csv(OUT_DIR / "method_coverage_paired_summary.csv", summary)
    write_csv(OUT_DIR / "method_coverage_paired_group_summary.csv", group_summary)
    write_csv(OUT_DIR / "method_coverage_paired_examples.csv", examples)
    (OUT_DIR / "method_coverage_paired_report.md").write_text(
        build_report(summary, group_summary, examples),
        encoding="utf-8",
    )
    print(f"Wrote method coverage paired audit to {OUT_DIR}")


def join_row(sample_id: str, target: dict[str, Any], coverage: dict[str, Any]) -> dict[str, Any]:
    target_correct = bool(target.get("correct"))
    coverage_correct = bool(coverage.get("correct"))
    target_is_negative = target.get("target_answer") == "no"
    return {
        "sample_id": sample_id,
        "image_id": target.get("image_id", ""),
        "target_text": target.get("target_text", ""),
        "source_text": target.get("source_text", ""),
        "hard_type": target.get("hard_type") or coverage.get("hard_type", ""),
        "binary_polarity": target.get("binary_polarity") or coverage.get("binary_polarity", ""),
        "target_answer": target.get("target_answer", ""),
        "target_pred": target.get("pred_answer", ""),
        "coverage_pred": coverage.get("pred_answer", ""),
        "target_correct": int(target_correct),
        "coverage_correct": int(coverage_correct),
        "outcome": outcome(target_correct, coverage_correct),
        "target_ecr": fmt(f(target.get("prune_ecr"))),
        "coverage_ecr": fmt(f(coverage.get("prune_ecr"))),
        "ecr_delta": fmt(f(coverage.get("prune_ecr")) - f(target.get("prune_ecr"))),
        "target_center": fmt(f(target.get("prune_evidence_center_recall"))),
        "coverage_center": fmt(f(coverage.get("prune_evidence_center_recall"))),
        "target_patch": fmt(f(target.get("prune_evidence_patch_recall"))),
        "coverage_patch": fmt(f(coverage.get("prune_evidence_patch_recall"))),
        "target_margin": fmt(f(target.get("margin"))),
        "coverage_margin": fmt(f(coverage.get("margin"))),
        "margin_abs_delta": fmt(abs(f(coverage.get("margin"))) - abs(f(target.get("margin")))),
        "target_false_positive": int(target_is_negative and target.get("pred_answer") == "yes"),
        "coverage_false_positive": int(target_is_negative and coverage.get("pred_answer") == "yes"),
        "question": target.get("question", ""),
    }


def outcome(target_correct: bool, coverage_correct: bool) -> str:
    if target_correct and coverage_correct:
        return "both_correct"
    if target_correct and not coverage_correct:
        return "target_only_correct"
    if not target_correct and coverage_correct:
        return "coverage_only_correct"
    return "both_wrong"


def build_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    positives = [row for row in rows if row["target_answer"] == "yes"]
    negatives = [row for row in rows if row["target_answer"] == "no"]
    ecr_improved = [row for row in rows if f(row["ecr_delta"]) > 0.25]
    target_low_ecr = [row for row in rows if f(row["target_ecr"]) < 0.50]
    return [
        summarize("all", rows),
        summarize("positive_probes", positives),
        summarize("negative_probes", negatives),
        summarize("target_ECR_lt0p50", target_low_ecr),
        summarize("ECR_delta_gt0p25", ecr_improved),
        summarize("target_low_ECR_and_ECR_improved", [row for row in target_low_ecr if f(row["ecr_delta"]) > 0.25]),
    ]


def build_group_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["outcome"], []).append(row)
        groups.setdefault(f"hard_type={row['hard_type']}", []).append(row)
        groups.setdefault(f"polarity={row['binary_polarity']}", []).append(row)
    order = [
        "both_correct",
        "target_only_correct",
        "coverage_only_correct",
        "both_wrong",
        "polarity=positive",
        "polarity=negative",
        "hard_type=small_positive",
        "hard_type=near_miss_negative",
    ]
    return [summarize(name, groups[name]) for name in order if name in groups]


def summarize(group: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    outcomes = Counter(row["outcome"] for row in rows)
    return {
        "group": group,
        "n": len(rows),
        "target_acc": fmt(rate(rows, lambda row: int(row["target_correct"]) == 1)),
        "coverage_acc": fmt(rate(rows, lambda row: int(row["coverage_correct"]) == 1)),
        "target_hFPR": fmt(false_positive_rate(rows, "target_false_positive")),
        "coverage_hFPR": fmt(false_positive_rate(rows, "coverage_false_positive")),
        "mean_target_ECR": fmt(avg(rows, "target_ecr")),
        "mean_coverage_ECR": fmt(avg(rows, "coverage_ecr")),
        "mean_ECR_delta": fmt(avg(rows, "ecr_delta")),
        "mean_margin_abs_delta": fmt(avg(rows, "margin_abs_delta")),
        "both_correct": outcomes.get("both_correct", 0),
        "target_only_correct": outcomes.get("target_only_correct", 0),
        "coverage_only_correct": outcomes.get("coverage_only_correct", 0),
        "both_wrong": outcomes.get("both_wrong", 0),
        "target_only_rate": fmt(rate(rows, lambda row: row["outcome"] == "target_only_correct")),
        "coverage_only_rate": fmt(rate(rows, lambda row: row["outcome"] == "coverage_only_correct")),
    }


def build_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    categories = [
        ("coverage_hurts_despite_ECR_gain", lambda row: row["outcome"] == "target_only_correct" and f(row["ecr_delta"]) > 0.25),
        ("coverage_repairs_low_ECR", lambda row: row["outcome"] == "coverage_only_correct" and f(row["target_ecr"]) < 0.50),
        ("both_wrong_high_ECR_gain", lambda row: row["outcome"] == "both_wrong" and f(row["ecr_delta"]) > 0.25),
        ("coverage_false_positive", lambda row: int(row["coverage_false_positive"]) == 1 and int(row["target_false_positive"]) == 0),
    ]
    for category, pred in categories:
        selected = [row for row in rows if pred(row)]
        selected.sort(key=lambda row: (-f(row["ecr_delta"]), f(row["margin_abs_delta"]), row["sample_id"]))
        for row in selected[:6]:
            item = {
                "category": category,
                "sample_id": row["sample_id"],
                "hard_type": row["hard_type"],
                "target_answer": row["target_answer"],
                "target_pred": row["target_pred"],
                "coverage_pred": row["coverage_pred"],
                "target_ecr": row["target_ecr"],
                "coverage_ecr": row["coverage_ecr"],
                "ecr_delta": row["ecr_delta"],
                "target_margin": row["target_margin"],
                "coverage_margin": row["coverage_margin"],
                "question": row["question"],
            }
            examples.append(item)
    return examples


def false_positive_rate(rows: list[dict[str, Any]], key: str) -> float:
    negatives = [row for row in rows if row["target_answer"] == "no"]
    if not negatives:
        return math.nan
    return mean(float(int(row[key]) == 1) for row in negatives)


def build_report(
    summary: list[dict[str, Any]],
    group_summary: list[dict[str, Any]],
    examples: list[dict[str, Any]],
) -> str:
    lines = [
        "# Method Coverage Paired Audit",
        "",
        "This audit joins Target 0.30 and Coverage-greedy 0.30 on the same TextOCR-Hard probes. It asks whether explicit evidence coverage repairs answer errors, or whether it mostly increases ECR without improving answer behavior.",
        "",
        "## Summary",
        "",
        table_md(
            summary,
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
        "## Outcome Groups",
        "",
        table_md(
            group_summary,
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
        "## Examples",
        "",
        table_md(
            examples,
            [
                "category",
                "sample_id",
                "hard_type",
                "target_answer",
                "target_pred",
                "coverage_pred",
                "target_ecr",
                "coverage_ecr",
                "ecr_delta",
                "question",
            ],
        ),
        "",
        "## Interpretation",
        "",
        "Coverage-greedy is useful as a mechanism test: it shows that the evidence term can raise ECR at a fixed token budget. Its paired failures explain why the paper should not claim that coverage maximization alone solves answer-risk control.",
    ]
    return "\n".join(lines) + "\n"


def avg(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return math.nan
    return mean(f(row[key]) for row in rows)


def rate(rows: list[dict[str, Any]], pred) -> float:
    if not rows:
        return math.nan
    return mean(float(pred(row)) for row in rows)


def f(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def table_md(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "(empty)"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")).replace("|", "/") for col in columns) + " |")
    return "\n".join(lines)


def fmt(value: Any) -> str:
    x = f(value)
    if math.isnan(x):
        return ""
    return f"{x:.3f}"


if __name__ == "__main__":
    main()
