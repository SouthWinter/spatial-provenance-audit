#!/usr/bin/env python3
"""Audit whether one adaptive open-QA policy transfers across tasks.

This script tests the strongest remaining R4 question from problem.md: whether
the adaptive budget policy is a unified deployable controller or a task-specific
diagnostic. It reuses cached open-QA generations and selector traces; no model
inference is run.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import mean
from typing import Any

from build_open_ocr_qa_pregen_risk_signal_audit import (
    FEATURE_GROUPS,
    ROOT,
    fixed_summary,
    fmt,
    load_rows,
    low_failure_label,
    policy_objective,
    policy_summary,
    predict,
    quantile_grid,
    train_probe,
)


OUT_DIR = ROOT / "runs" / "problem_optimization_audit" / "open_ocr_qa_unified_policy_transfer"
PAPER_DIR = ROOT / "runs" / "paper_evidence"

DEPLOYABLE_GROUPS = (
    "question_only",
    "mask30_only",
    "mask_stability_pregen",
    "question_mask_pregen",
)


def main() -> None:
    rows_by_task = load_rows()
    transfer_rows = build_cross_task_transfer(rows_by_task)
    pooled_rows = build_pooled_policy(rows_by_task)
    summary_rows = build_summary(transfer_rows, pooled_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {
        "unified_policy_transfer_summary.csv": summary_rows,
        "unified_policy_cross_task_rows.csv": transfer_rows,
        "unified_policy_pooled_rows.csv": pooled_rows,
    }
    for name, rows in outputs.items():
        write_csv(OUT_DIR / name, rows)
        write_csv(PAPER_DIR / f"table_open_ocr_qa_{name}", rows)

    (OUT_DIR / "unified_policy_transfer_report.md").write_text(
        build_report(summary_rows, transfer_rows, pooled_rows),
        encoding="utf-8",
    )
    print(f"Wrote unified policy transfer audit to {OUT_DIR}")


def build_cross_task_transfer(rows_by_task: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tasks = sorted(rows_by_task)
    for source in tasks:
        for target in tasks:
            if source == target:
                continue
            for feature_group in DEPLOYABLE_GROUPS:
                probe = train_probe(
                    source,
                    "low_failure_ge0p25",
                    feature_group,
                    FEATURE_GROUPS[feature_group],
                    rows_by_task[source],
                    low_failure_label,
                )
                threshold, source_dev = select_threshold(probe, rows_by_task[source])
                source_test = policy_summary(rows_by_task[source], split="test", probe=probe, threshold=threshold)
                target_test = policy_summary(rows_by_task[target], split="test", probe=probe, threshold=threshold)
                source_fixed70 = fixed_summary(rows_by_task[source], "test", 0.70)
                target_fixed70 = fixed_summary(rows_by_task[target], "test", 0.70)
                rows.append(
                    {
                        "source_task": source,
                        "target_task": target,
                        "feature_group": feature_group,
                        "threshold_source_dev": fmt(threshold),
                        "source_dev_score": fmt(source_dev["score"]),
                        "source_dev_keep": fmt(source_dev["mean_keep"]),
                        "source_test_score": fmt(source_test["score"]),
                        "source_test_keep": fmt(source_test["mean_keep"]),
                        "source_delta_vs_fixed70": fmt(source_test["score"] - source_fixed70["score"]),
                        "target_test_score": fmt(target_test["score"]),
                        "target_test_keep": fmt(target_test["mean_keep"]),
                        "target_delta_vs_fixed70": fmt(target_test["score"] - target_fixed70["score"]),
                        "target_fixed70_score": fmt(target_fixed70["score"]),
                        "target_fixed70_keep": fmt(target_fixed70["mean_keep"]),
                        "target_passes_near_fixed70_lower_keep": pass_near_fixed70_lower_keep(
                            target_test, target_fixed70
                        ),
                        "decision_features": feature_group_note(feature_group),
                    }
                )
    return rows


def build_pooled_policy(rows_by_task: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    pooled_rows: list[dict[str, Any]] = []
    all_rows = [row for rows in rows_by_task.values() for row in rows]
    for feature_group in DEPLOYABLE_GROUPS:
        probe = train_probe(
            "pooled_TextVQA_DocVQA",
            "low_failure_ge0p25",
            feature_group,
            FEATURE_GROUPS[feature_group],
            all_rows,
            low_failure_label,
        )
        threshold, pooled_dev = select_pooled_threshold(probe, rows_by_task)
        for task, task_rows in sorted(rows_by_task.items()):
            test_summary = policy_summary(task_rows, split="test", probe=probe, threshold=threshold)
            fixed30 = fixed_summary(task_rows, "test", 0.30)
            fixed70 = fixed_summary(task_rows, "test", 0.70)
            pooled_rows.append(
                {
                    "policy": "pooled_dev_threshold",
                    "task": task,
                    "feature_group": feature_group,
                    "threshold_pooled_dev": fmt(threshold),
                    "pooled_dev_objective": fmt(pooled_dev["objective"]),
                    "test_score": fmt(test_summary["score"]),
                    "test_keep": fmt(test_summary["mean_keep"]),
                    "test_fallback70_rate": fmt(test_summary["fallback70_rate"]),
                    "delta_vs_fixed30": fmt(test_summary["score"] - fixed30["score"]),
                    "delta_vs_fixed70": fmt(test_summary["score"] - fixed70["score"]),
                    "fixed30_score": fmt(fixed30["score"]),
                    "fixed70_score": fmt(fixed70["score"]),
                    "passes_near_fixed70_lower_keep": pass_near_fixed70_lower_keep(test_summary, fixed70),
                    "decision_features": feature_group_note(feature_group),
                }
            )
    return pooled_rows


def build_summary(
    transfer_rows: list[dict[str, Any]], pooled_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, target in (("TextVQA-lite", "DocVQA-lite"), ("DocVQA-lite", "TextVQA-lite")):
        candidates = [
            row
            for row in transfer_rows
            if row["source_task"] == source and row["target_task"] == target
        ]
        best_target_quality = max(candidates, key=lambda row: float(row["target_test_score"]))
        best_target_cost = min(candidates, key=lambda row: float(row["target_test_keep"]))
        passing = [row for row in candidates if row["target_passes_near_fixed70_lower_keep"] == "1"]
        rows.append(
            {
                "scope": f"{source} -> {target}",
                "question": "Does a dev-selected pre-generation policy transfer across tasks?",
                "best_quality_policy": best_target_quality["feature_group"],
                "best_quality_target_score": best_target_quality["target_test_score"],
                "best_quality_target_keep": best_target_quality["target_test_keep"],
                "best_quality_delta_vs_fixed70": best_target_quality["target_delta_vs_fixed70"],
                "lowest_keep_policy": best_target_cost["feature_group"],
                "lowest_keep_target_score": best_target_cost["target_test_score"],
                "lowest_keep_target_keep": best_target_cost["target_test_keep"],
                "near_fixed70_lower_keep_candidates": len(passing),
                "reading": transfer_reading(best_target_quality, passing),
            }
        )

    for task in sorted({row["task"] for row in pooled_rows}):
        candidates = [row for row in pooled_rows if row["task"] == task]
        best_quality = max(candidates, key=lambda row: float(row["test_score"]))
        lowest_keep = min(candidates, key=lambda row: float(row["test_keep"]))
        passing = [row for row in candidates if row["passes_near_fixed70_lower_keep"] == "1"]
        rows.append(
            {
                "scope": f"pooled dev -> {task}",
                "question": "Does one pooled controller satisfy both tasks?",
                "best_quality_policy": best_quality["feature_group"],
                "best_quality_target_score": best_quality["test_score"],
                "best_quality_target_keep": best_quality["test_keep"],
                "best_quality_delta_vs_fixed70": best_quality["delta_vs_fixed70"],
                "lowest_keep_policy": lowest_keep["feature_group"],
                "lowest_keep_target_score": lowest_keep["test_score"],
                "lowest_keep_target_keep": lowest_keep["test_keep"],
                "near_fixed70_lower_keep_candidates": len(passing),
                "reading": pooled_reading(best_quality, passing),
            }
        )
    return rows


def select_threshold(probe: Any, rows: list[dict[str, Any]]) -> tuple[float, dict[str, float]]:
    dev = [row for row in rows if row["split"] == "dev"]
    scored: list[tuple[float, float, dict[str, float]]] = []
    for threshold in quantile_grid([predict(row, probe) for row in dev]):
        summary = policy_summary(rows, split="dev", probe=probe, threshold=threshold)
        scored.append((policy_objective(summary), threshold, summary))
    scored.sort(key=lambda item: (item[0], item[2]["score"], -item[2]["mean_keep"]), reverse=True)
    _, threshold, summary = scored[0]
    return threshold, summary


def select_pooled_threshold(
    probe: Any, rows_by_task: dict[str, list[dict[str, Any]]]
) -> tuple[float, dict[str, float]]:
    dev_scores = [
        predict(row, probe)
        for task_rows in rows_by_task.values()
        for row in task_rows
        if row["split"] == "dev"
    ]
    scored: list[tuple[float, float, dict[str, float]]] = []
    for threshold in quantile_grid(dev_scores):
        task_summaries = [
            policy_summary(task_rows, split="dev", probe=probe, threshold=threshold)
            for task_rows in rows_by_task.values()
        ]
        objective = mean(policy_objective(summary) for summary in task_summaries)
        score = mean(summary["score"] for summary in task_summaries)
        keep = mean(summary["mean_keep"] for summary in task_summaries)
        scored.append((objective, threshold, {"objective": objective, "score": score, "mean_keep": keep}))
    scored.sort(key=lambda item: (item[0], item[2]["score"], -item[2]["mean_keep"]), reverse=True)
    _, threshold, summary = scored[0]
    return threshold, summary


def pass_near_fixed70_lower_keep(candidate: dict[str, float], fixed70: dict[str, float]) -> str:
    score_ok = candidate["score"] >= fixed70["score"] - 0.01
    keep_ok = candidate["mean_keep"] < fixed70["mean_keep"]
    return "1" if score_ok and keep_ok else "0"


def transfer_reading(best_quality: dict[str, Any], passing: list[dict[str, Any]]) -> str:
    if passing:
        return "A source-task policy transfers under the near-fixed70/lower-keep gate."
    return (
        "No source-task policy transfers under the near-fixed70/lower-keep gate; "
        f"best target delta vs fixed70 is {best_quality['target_delta_vs_fixed70']}."
    )


def pooled_reading(best_quality: dict[str, Any], passing: list[dict[str, Any]]) -> str:
    if passing:
        return "The pooled controller passes the near-fixed70/lower-keep gate for this task."
    return (
        "The pooled controller does not pass the near-fixed70/lower-keep gate for this task; "
        f"best delta vs fixed70 is {best_quality['delta_vs_fixed70']}."
    )


def feature_group_note(group: str) -> str:
    return {
        "question_only": "question metadata only",
        "mask30_only": "30% selector-mask geometry only",
        "mask_stability_pregen": "30/50/70 selector-mask geometry and overlap",
        "question_mask_pregen": "question metadata plus selector-mask features",
    }.get(group, group)


def build_report(
    summary_rows: list[dict[str, Any]],
    transfer_rows: list[dict[str, Any]],
    pooled_rows: list[dict[str, Any]],
) -> str:
    return "\n".join(
        [
            "# Open OCR QA Unified Policy Transfer Audit",
            "",
            "This audit tests whether a pre-generation budget controller selected on one open-QA task transfers to the other, and whether a pooled dev controller can satisfy both tasks. It uses cached scores and selector traces only.",
            "",
            "## Summary",
            "",
            table_md(
                summary_rows,
                [
                    "scope",
                    "question",
                    "best_quality_policy",
                    "best_quality_target_score",
                    "best_quality_target_keep",
                    "best_quality_delta_vs_fixed70",
                    "near_fixed70_lower_keep_candidates",
                    "reading",
                ],
            ),
            "",
            "## Cross-Task Transfer Rows",
            "",
            table_md(
                transfer_rows,
                [
                    "source_task",
                    "target_task",
                    "feature_group",
                    "source_test_score",
                    "source_test_keep",
                    "source_delta_vs_fixed70",
                    "target_test_score",
                    "target_test_keep",
                    "target_delta_vs_fixed70",
                    "target_passes_near_fixed70_lower_keep",
                ],
            ),
            "",
            "## Pooled Dev Controller Rows",
            "",
            table_md(
                pooled_rows,
                [
                    "policy",
                    "task",
                    "feature_group",
                    "test_score",
                    "test_keep",
                    "delta_vs_fixed70",
                    "passes_near_fixed70_lower_keep",
                    "decision_features",
                ],
            ),
            "",
            "## Safe Reading",
            "",
            "The transfer audit should be used as a claim-boundary result: it can support a scoped unified controller only if the same deployable policy passes both tasks. Otherwise, it strengthens the paper by showing that adaptive control is an explicit open problem rather than an unstated hand-selection artifact.",
            "",
        ]
    )


def table_md(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "(empty)"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
