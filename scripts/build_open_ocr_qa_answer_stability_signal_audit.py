#!/usr/bin/env python3
"""Audit answer agreement as a risk signal for open OCR/DocQA pruning."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import Any

from build_open_ocr_qa_answer_stability_cascade import ROOT, load_rows, norm_answer


OUT_DIR = ROOT / "runs" / "problem_optimization_audit" / "open_ocr_qa_answer_stability_signal"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    rows_by_task = load_rows()
    signal_rows: list[dict[str, Any]] = []
    for task in sorted(rows_by_task):
        task_rows = rows_by_task[task]
        for split in ("dev", "test", "all"):
            scoped = [row for row in task_rows if split == "all" or row["split"] == split]
            signal_rows.extend(build_signal_rows(task, split, scoped))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "answer_stability_signal_summary.csv", signal_rows)
    (out_dir / "answer_stability_signal_report.md").write_text(build_report(signal_rows), encoding="utf-8")
    print(f"Wrote answer-stability signal audit to {out_dir}")


def build_signal_rows(task: str, split: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = {
        "all": rows,
        "agree30_50": [row for row in rows if agree(row, 0.30, 0.50)],
        "disagree30_50": [row for row in rows if not agree(row, 0.30, 0.50)],
        "agree50_70": [row for row in rows if agree(row, 0.50, 0.70)],
        "disagree50_70": [row for row in rows if not agree(row, 0.50, 0.70)],
        "all_equal_30_50_70": [row for row in rows if agree(row, 0.30, 0.50) and agree(row, 0.50, 0.70)],
        "any_pair_agree": [row for row in rows if agree(row, 0.30, 0.50) or agree(row, 0.50, 0.70) or agree(row, 0.30, 0.70)],
        "no_pair_agree": [row for row in rows if not (agree(row, 0.30, 0.50) or agree(row, 0.50, 0.70) or agree(row, 0.30, 0.70))],
    }
    return [summarize_group(task, split, name, subset, len(rows)) for name, subset in groups.items()]


def agree(row: dict[str, Any], left: float, right: float) -> bool:
    return norm_answer(row["budgets"][left]["answer"]) == norm_answer(row["budgets"][right]["answer"])


def summarize_group(
    task: str,
    split: str,
    group: str,
    rows: list[dict[str, Any]],
    total: int,
) -> dict[str, Any]:
    if not rows:
        return {
            "task": task,
            "split": split,
            "group": group,
            "n": 0,
            "fraction": "0.000",
        }
    score30 = [score(row, 0.30) for row in rows]
    score50 = [score(row, 0.50) for row in rows]
    score70 = [score(row, 0.70) for row in rows]
    full = [float(row["full_score"]) for row in rows]
    low_fail = [f - s30 >= 0.25 for f, s30 in zip(full, score30)]
    safe30 = [f - s30 <= 0.10 for f, s30 in zip(full, score30)]
    safe50 = [f - s50 <= 0.10 for f, s50 in zip(full, score50)]
    safe70 = [f - s70 <= 0.10 for f, s70 in zip(full, score70)]
    repaired70 = [
        (f - s30 >= 0.25) and (f - s70 <= 0.10)
        for f, s30, s70 in zip(full, score30, score70)
    ]
    return {
        "task": task,
        "split": split,
        "group": group,
        "n": len(rows),
        "fraction": fmt(len(rows) / total if total else 0.0),
        "mean_full_score": fmt(mean(full)),
        "mean_score30": fmt(mean(score30)),
        "mean_score50": fmt(mean(score50)),
        "mean_score70": fmt(mean(score70)),
        "mean_drop30": fmt(mean(f - s30 for f, s30 in zip(full, score30))),
        "mean_gain30_to70": fmt(mean(s70 - s30 for s30, s70 in zip(score30, score70))),
        "low_failure_rate_ge0p25": fmt(mean(low_fail)),
        "safe30_within0p10_rate": fmt(mean(safe30)),
        "safe50_within0p10_rate": fmt(mean(safe50)),
        "safe70_within0p10_rate": fmt(mean(safe70)),
        "repair70_rate_among_all": fmt(mean(repaired70)),
        "repair70_rate_among_low_fail": fmt(sum(repaired70) / max(1, sum(low_fail))),
        "note": group_note(group),
    }


def group_note(group: str) -> str:
    return {
        "all": "all examples",
        "agree30_50": "30% and 50% normalized answers match; candidate accept signal",
        "disagree30_50": "30% and 50% answers differ; candidate escalation signal",
        "agree50_70": "50% and 70% normalized answers match",
        "disagree50_70": "50% and 70% answers differ",
        "all_equal_30_50_70": "all three pruned budgets agree",
        "any_pair_agree": "at least two pruned budgets agree",
        "no_pair_agree": "all three pruned answers differ",
    }.get(group, "")


def score(row: dict[str, Any], budget: float) -> float:
    return float(row["budgets"][budget]["score"])


def fmt(value: float) -> str:
    return f"{float(value):.3f}"


def build_report(rows: list[dict[str, Any]]) -> str:
    key_rows = [
        row
        for row in rows
        if row["split"] == "test"
        and row["group"] in {"all", "agree30_50", "disagree30_50", "all_equal_30_50_70", "no_pair_agree"}
    ]
    cols = [
        "task",
        "group",
        "n",
        "fraction",
        "mean_score30",
        "mean_score70",
        "mean_drop30",
        "low_failure_rate_ge0p25",
        "safe30_within0p10_rate",
        "safe70_within0p10_rate",
        "repair70_rate_among_low_fail",
        "note",
    ]
    lines = [
        "# Open OCR QA Answer-Stability Signal Audit",
        "",
        "This audit measures answer agreement as a risk signal, without choosing a new budget policy. It asks whether agreement between low/mid-budget answers identifies safe low-budget cases and whether disagreement identifies examples repaired by higher retention.",
        "",
        "## Test Signal Summary",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in key_rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in cols) + " |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Agreement is useful but asymmetric: when 30% and 50% agree, TextVQA is often safe at 30%, while DocVQA remains much riskier. Disagreement is an escalation signal, but running multiple generations to observe it has real cost.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
