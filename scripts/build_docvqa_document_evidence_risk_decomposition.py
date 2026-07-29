#!/usr/bin/env python3
"""Decompose DocVQA line-context evidence risk across budgets.

This audit turns the existing line-context ECR/quality rows into reviewer-facing
diagnostics: mean coverage, worst-region coverage, all-region coverage, and
repair/persistent-failure trajectories. It does not rerun any MLLM.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
IN_ROWS = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "docvqa_line_context_quality_association"
    / "line_context_quality_rows.csv"
)
OUT_DIR = ROOT / "runs" / "problem_optimization_audit" / "docvqa_document_evidence_risk_decomposition"


def main() -> None:
    rows = read_csv(IN_ROWS)
    budget_rows = build_budget_rows(rows)
    condition_rows = build_condition_rows(rows)
    trajectory_rows, trajectory_summary = build_trajectory_rows(rows)
    example_rows = build_example_rows(trajectory_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "docvqa_document_risk_by_budget.csv", budget_rows)
    write_csv(OUT_DIR / "docvqa_document_risk_by_condition.csv", condition_rows)
    write_csv(OUT_DIR / "docvqa_document_risk_trajectory_rows.csv", trajectory_rows)
    write_csv(OUT_DIR / "docvqa_document_risk_trajectory_summary.csv", trajectory_summary)
    write_csv(OUT_DIR / "docvqa_document_risk_examples.csv", example_rows)
    (OUT_DIR / "docvqa_document_risk_decomposition_report.md").write_text(
        build_report(budget_rows, condition_rows, trajectory_summary, example_rows),
        encoding="utf-8",
    )
    print(f"Wrote DocVQA document evidence-risk decomposition to {OUT_DIR}")


def build_budget_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for budget in (0.30, 0.50, 0.70):
        scoped = [row for row in rows if f(row["budget_keep_ratio"]) == budget]
        out.append(
            {
                "budget": fmt(budget),
                "n": len(scoped),
                "mean_box_count": fmt(avg(scoped, "box_count")),
                "mean_ECR": fmt(avg(scoped, "ECR")),
                "mean_worst_region_ECR": fmt(avg(scoped, "worst_region_ECR")),
                "all_regions_pass_rate": fmt(rate(scoped, lambda row: f(row["all_regions_ECR_ge_0p50"]) >= 0.5)),
                "worst_region_ge0p50_rate": fmt(rate(scoped, lambda row: f(row["worst_region_ECR"]) >= 0.5)),
                "ECR_ge0p75_rate": fmt(rate(scoped, lambda row: f(row["ECR"]) >= 0.75)),
                "pruned_good_rate": fmt(rate(scoped, lambda row: f(row["pruned_good"]) >= 0.5)),
                "low_failure_rate_ge0p25": fmt(rate(scoped, lambda row: f(row["score_drop"]) >= 0.25)),
                "severe_failure_rate_score_le0p10": fmt(rate(scoped, lambda row: f(row["pruned_score"]) <= 0.10)),
                "high_mean_ECR_but_bad_rate": fmt(rate(scoped, lambda row: f(row["ECR"]) >= 0.75 and f(row["pruned_good"]) < 0.5)),
                "all_regions_pass_but_bad_rate": fmt(rate(scoped, lambda row: f(row["all_regions_ECR_ge_0p50"]) >= 0.5 and f(row["pruned_good"]) < 0.5)),
            }
        )
    return out


def build_condition_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    conditions = [
        ("all", lambda row: True),
        ("ECR_ge0p75", lambda row: f(row["ECR"]) >= 0.75),
        ("ECR_lt0p75", lambda row: f(row["ECR"]) < 0.75),
        ("worst_region_ge0p50", lambda row: f(row["worst_region_ECR"]) >= 0.50),
        ("worst_region_lt0p50", lambda row: f(row["worst_region_ECR"]) < 0.50),
        ("all_regions_pass", lambda row: f(row["all_regions_ECR_ge_0p50"]) >= 0.5),
        ("all_regions_fail", lambda row: f(row["all_regions_ECR_ge_0p50"]) < 0.5),
        ("box_count_le2", lambda row: f(row["box_count"]) <= 2),
        ("box_count_ge3", lambda row: f(row["box_count"]) >= 3),
        ("box_count_ge5", lambda row: f(row["box_count"]) >= 5),
    ]
    out = []
    for budget in (0.30, 0.50, 0.70):
        budget_rows = [row for row in rows if f(row["budget_keep_ratio"]) == budget]
        for name, pred in conditions:
            scoped = [row for row in budget_rows if pred(row)]
            out.append(
                {
                    "budget": fmt(budget),
                    "condition": name,
                    "n": len(scoped),
                    "fraction": fmt(len(scoped) / len(budget_rows) if budget_rows else math.nan),
                    "mean_ECR": fmt(avg(scoped, "ECR")),
                    "mean_worst_region_ECR": fmt(avg(scoped, "worst_region_ECR")),
                    "mean_pruned_score": fmt(avg(scoped, "pruned_score")),
                    "pruned_good_rate": fmt(rate(scoped, lambda row: f(row["pruned_good"]) >= 0.5)),
                    "low_failure_rate_ge0p25": fmt(rate(scoped, lambda row: f(row["score_drop"]) >= 0.25)),
                    "severe_failure_rate_score_le0p10": fmt(rate(scoped, lambda row: f(row["pruned_score"]) <= 0.10)),
                }
            )
    return out


def build_trajectory_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_sample: dict[str, dict[float, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_sample[row["sample_id"]][f(row["budget_keep_ratio"])] = row

    trajectory_rows = []
    for sample_id, by_budget in sorted(by_sample.items()):
        if not all(budget in by_budget for budget in (0.30, 0.50, 0.70)):
            continue
        row30 = by_budget[0.30]
        row50 = by_budget[0.50]
        row70 = by_budget[0.70]
        low_failure30 = f(row30["score_drop"]) >= 0.25
        repaired70 = low_failure30 and f(row70["score_drop"]) <= 0.10
        persistent70 = low_failure30 and f(row70["score_drop"]) >= 0.25
        trajectory_rows.append(
            {
                "sample_id": sample_id,
                "box_count": row70["box_count"],
                "score30": fmt(f(row30["pruned_score"])),
                "score50": fmt(f(row50["pruned_score"])),
                "score70": fmt(f(row70["pruned_score"])),
                "drop30": fmt(f(row30["score_drop"])),
                "drop70": fmt(f(row70["score_drop"])),
                "ECR30": fmt(f(row30["ECR"])),
                "ECR50": fmt(f(row50["ECR"])),
                "ECR70": fmt(f(row70["ECR"])),
                "worst70": fmt(f(row70["worst_region_ECR"])),
                "all_pass70": fmt(f(row70["all_regions_ECR_ge_0p50"])),
                "low_failure30": int(low_failure30),
                "repaired70": int(repaired70),
                "persistent70": int(persistent70),
                "high_ECR70_bad": int(f(row70["ECR"]) >= 0.75 and f(row70["pruned_good"]) < 0.5),
                "all_pass70_bad": int(f(row70["all_regions_ECR_ge_0p50"]) >= 0.5 and f(row70["pruned_good"]) < 0.5),
                "raw_question": row70["raw_question"],
                "gold_answers": row70["gold_answers"],
                "full_answer": row70["full_answer"],
                "answer70": row70["pruned_answer"],
            }
        )

    low_fail = [row for row in trajectory_rows if int(row["low_failure30"])]
    persistent = [row for row in trajectory_rows if int(row["persistent70"])]
    summary = [
        summarize_trajectory_group("all_samples", trajectory_rows),
        summarize_trajectory_group("low_failure30", low_fail),
        summarize_trajectory_group("repaired70_among_low_failure30", [row for row in low_fail if int(row["repaired70"])]),
        summarize_trajectory_group("persistent70_among_low_failure30", persistent),
        summarize_trajectory_group("persistent70_with_ECR70_ge0p75", [row for row in persistent if f(row["ECR70"]) >= 0.75]),
        summarize_trajectory_group("persistent70_with_all_pass70", [row for row in persistent if f(row["all_pass70"]) >= 0.5]),
        summarize_trajectory_group("low_failure30_all_pass70", [row for row in low_fail if f(row["all_pass70"]) >= 0.5]),
        summarize_trajectory_group("low_failure30_all_fail70", [row for row in low_fail if f(row["all_pass70"]) < 0.5]),
        summarize_trajectory_group("low_failure30_box_count_ge3", [row for row in low_fail if f(row["box_count"]) >= 3]),
        summarize_trajectory_group("low_failure30_box_count_ge5", [row for row in low_fail if f(row["box_count"]) >= 5]),
    ]
    return trajectory_rows, summary


def summarize_trajectory_group(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "group": name,
        "n": len(rows),
        "mean_box_count": fmt(avg(rows, "box_count")),
        "mean_score30": fmt(avg(rows, "score30")),
        "mean_score70": fmt(avg(rows, "score70")),
        "mean_ECR70": fmt(avg(rows, "ECR70")),
        "mean_worst70": fmt(avg(rows, "worst70")),
        "all_pass70_rate": fmt(rate(rows, lambda row: f(row["all_pass70"]) >= 0.5)),
        "repaired70_rate": fmt(rate(rows, lambda row: int(row["repaired70"]) == 1)),
        "persistent70_rate": fmt(rate(rows, lambda row: int(row["persistent70"]) == 1)),
        "high_ECR70_bad_rate": fmt(rate(rows, lambda row: int(row["high_ECR70_bad"]) == 1)),
        "all_pass70_bad_rate": fmt(rate(rows, lambda row: int(row["all_pass70_bad"]) == 1)),
    }


def build_example_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    categories = [
        ("persistent_high_ECR70", lambda row: int(row["persistent70"]) and f(row["ECR70"]) >= 0.75),
        ("persistent_all_pass70", lambda row: int(row["persistent70"]) and f(row["all_pass70"]) >= 0.5),
        ("repaired_low_ECR70", lambda row: int(row["repaired70"]) and f(row["ECR70"]) < 0.75),
        ("high_ECR70_bad", lambda row: int(row["high_ECR70_bad"])),
    ]
    out = []
    for category, pred in categories:
        selected = [row for row in rows if pred(row)]
        selected.sort(key=lambda row: (-f(row["drop70"]), -f(row["ECR70"]), row["sample_id"]))
        for row in selected[:5]:
            item = dict(row)
            item["category"] = category
            out.append(item)
    return out


def build_report(
    budget_rows: list[dict[str, Any]],
    condition_rows: list[dict[str, Any]],
    trajectory_summary: list[dict[str, Any]],
    example_rows: list[dict[str, Any]],
) -> str:
    key_conditions = [
        row
        for row in condition_rows
        if row["condition"]
        in {"all", "ECR_ge0p75", "worst_region_ge0p50", "worst_region_lt0p50", "all_regions_pass", "all_regions_fail", "box_count_ge3"}
    ]
    lines = [
        "# DocVQA Document Evidence-Risk Decomposition",
        "",
        "This audit decomposes the DocVQA line-context evidence signal into mean coverage, worst-region coverage, all-region coverage, and budget trajectories. It reuses cached line-context ECR and generation scores, so it is an association and failure-analysis audit rather than a new model run.",
        "",
        "## Budget Summary",
        "",
        table_md(
            budget_rows,
            [
                "budget",
                "n",
                "mean_box_count",
                "mean_ECR",
                "mean_worst_region_ECR",
                "all_regions_pass_rate",
                "pruned_good_rate",
                "low_failure_rate_ge0p25",
                "high_mean_ECR_but_bad_rate",
                "all_regions_pass_but_bad_rate",
            ],
        ),
        "",
        "## Condition Summary",
        "",
        table_md(
            key_conditions,
            [
                "budget",
                "condition",
                "n",
                "fraction",
                "mean_ECR",
                "mean_worst_region_ECR",
                "mean_pruned_score",
                "pruned_good_rate",
                "low_failure_rate_ge0p25",
            ],
        ),
        "",
        "## Trajectory Summary",
        "",
        table_md(
            trajectory_summary,
            [
                "group",
                "n",
                "mean_box_count",
                "mean_score30",
                "mean_score70",
                "mean_ECR70",
                "mean_worst70",
                "all_pass70_rate",
                "repaired70_rate",
                "persistent70_rate",
                "high_ECR70_bad_rate",
                "all_pass70_bad_rate",
            ],
        ),
        "",
        "## Examples",
        "",
        table_md(
            example_rows[:12],
            [
                "category",
                "sample_id",
                "box_count",
                "score30",
                "score70",
                "ECR70",
                "worst70",
                "all_pass70",
                "raw_question",
                "full_answer",
                "answer70",
            ],
        ),
        "",
        "## Interpretation",
        "",
        "DocVQA failures are not explained by mean ECR alone. Worst-region and all-region coverage expose missing necessary regions, while high-ECR bad examples show that coverage is still an availability signal rather than proof of answer use.",
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
