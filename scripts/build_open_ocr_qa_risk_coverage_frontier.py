#!/usr/bin/env python3
"""Build oracle risk-coverage frontiers for native open OCR/DocQA runs.

This audit asks a sharper version of the adaptive-budget question: if a perfect
risk controller could select among cached 30%, 50%, 70%, and full-prefix runs,
how many tokens would be needed to stay within a per-sample loss tolerance from
the full-prefix answer? The result is an upper bound, not a deployable policy.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "problem_optimization_audit" / "open_ocr_qa_risk_coverage_frontier"

RUNS = {
    "TextVQA-lite": {
        0.30: ROOT / "runs/open_ocr_qa/qwen3_8b_textvqa_lite_target_grid0p30_full500/open_ocr_qa_generation.jsonl",
        0.50: ROOT / "runs/open_ocr_qa/qwen3_8b_textvqa_lite_target_grid0p50_full500/open_ocr_qa_generation.jsonl",
        0.70: ROOT / "runs/open_ocr_qa/qwen3_8b_textvqa_lite_target_grid0p70_full500/open_ocr_qa_generation.jsonl",
    },
    "DocVQA-lite": {
        0.30: ROOT / "runs/open_ocr_qa/qwen3_8b_docvqa_lite_target_grid0p30_full500/open_ocr_qa_generation.jsonl",
        0.50: ROOT / "runs/open_ocr_qa/qwen3_8b_docvqa_lite_target_grid0p50_full500/open_ocr_qa_generation.jsonl",
        0.70: ROOT / "runs/open_ocr_qa/qwen3_8b_docvqa_lite_target_grid0p70_full500/open_ocr_qa_generation.jsonl",
    },
}
BUDGETS = (0.30, 0.50, 0.70, 1.00)
TOLERANCES = (0.00, 0.01, 0.02, 0.05, 0.10, 0.20)


def main() -> None:
    rows = load_rows()
    summary_rows: list[dict[str, Any]] = []
    choice_rows: list[dict[str, Any]] = []
    for task in sorted(rows):
        for tolerance in TOLERANCES:
            for split in ("dev", "test", "all"):
                scoped = [row for row in rows[task] if split == "all" or row["split"] == split]
                choices = [choose_cheapest_within_tolerance(row, tolerance) for row in scoped]
                summary_rows.append(summarize(task, split, tolerance, choices))
                if split == "all":
                    for choice in choices:
                        choice_rows.append(flatten_choice(task, tolerance, choice))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "risk_coverage_frontier_summary.csv", summary_rows)
    write_csv(OUT_DIR / "risk_coverage_frontier_choices.csv", choice_rows)
    (OUT_DIR / "risk_coverage_frontier_report.md").write_text(
        build_report(summary_rows),
        encoding="utf-8",
    )
    print(f"Wrote open OCR QA risk-coverage frontier to {OUT_DIR}")


def load_rows() -> dict[str, list[dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for task, by_budget in RUNS.items():
        task_rows: dict[str, dict[str, Any]] = {}
        for budget, path in by_budget.items():
            for row in read_jsonl(path):
                sample_id = row["sample_id"]
                record = task_rows.setdefault(
                    sample_id,
                    {
                        "sample_id": sample_id,
                        "question_id": row.get("question_id", ""),
                        "split": split_for_id(sample_id),
                        "raw_question": row.get("raw_question") or row.get("question", ""),
                        "metric": row.get("metric", ""),
                        "gold_answers": row.get("gold_answers", []),
                        "full_answer": row.get("full_answer", ""),
                        "full_score": float(row.get("full_score", 0.0)),
                        "full_exact": float(row.get("full_exact", 0.0)),
                        "full_anls": float(row.get("full_anls", 0.0)),
                        "full_textvqa_accuracy": float(row.get("full_textvqa_accuracy", 0.0)),
                        "budgets": {},
                    },
                )
                record["budgets"][budget] = {
                    "answer": row.get("pruned_answer", ""),
                    "score": float(row.get("pruned_score", 0.0)),
                    "exact": float(row.get("pruned_exact", 0.0)),
                    "anls": float(row.get("pruned_anls", 0.0)),
                    "textvqa_accuracy": float(row.get("pruned_textvqa_accuracy", 0.0)),
                    "effective_keep": float(row.get("effective_keep_ratio", budget) or budget),
                }
        for record in task_rows.values():
            record["budgets"][1.00] = {
                "answer": record["full_answer"],
                "score": record["full_score"],
                "exact": record["full_exact"],
                "anls": record["full_anls"],
                "textvqa_accuracy": record["full_textvqa_accuracy"],
                "effective_keep": 1.0,
            }
        out[task] = list(task_rows.values())
    return out


def choose_cheapest_within_tolerance(row: dict[str, Any], tolerance: float) -> dict[str, Any]:
    full_score = float(row["full_score"])
    target = max(0.0, full_score - tolerance)
    qualifying = [
        budget
        for budget in BUDGETS
        if float(row["budgets"][budget]["score"]) + 1e-12 >= target
    ]
    chosen_budget = min(qualifying) if qualifying else 1.00
    chosen = row["budgets"][chosen_budget]
    low_budget = row["budgets"][0.30]
    return {
        "sample_id": row["sample_id"],
        "question_id": row["question_id"],
        "split": row["split"],
        "raw_question": row["raw_question"],
        "metric": row["metric"],
        "full_score": full_score,
        "full_exact": row["full_exact"],
        "full_anls": row["full_anls"],
        "full_textvqa_accuracy": row["full_textvqa_accuracy"],
        "full_answer": row["full_answer"],
        "low_score": low_budget["score"],
        "low_answer": low_budget["answer"],
        "chosen_budget": chosen_budget,
        "chosen_score": chosen["score"],
        "chosen_exact": chosen["exact"],
        "chosen_anls": chosen["anls"],
        "chosen_textvqa_accuracy": chosen["textvqa_accuracy"],
        "chosen_answer": chosen["answer"],
        "chosen_loss_vs_full": full_score - float(chosen["score"]),
        "required_escalation": chosen_budget > 0.30,
        "requires_70_or_full": chosen_budget >= 0.70,
        "requires_full": chosen_budget >= 1.00,
    }


def summarize(task: str, split: str, tolerance: float, choices: list[dict[str, Any]]) -> dict[str, Any]:
    if not choices:
        return {
            "task": task,
            "split": split,
            "tolerance": fmt(tolerance),
            "n": 0,
        }
    budget_counts = {budget: sum(1 for choice in choices if choice["chosen_budget"] == budget) for budget in BUDGETS}
    return {
        "task": task,
        "split": split,
        "tolerance": fmt(tolerance),
        "n": len(choices),
        "oracle_score": fmt(mean(choice["chosen_score"] for choice in choices)),
        "full_score": fmt(mean(choice["full_score"] for choice in choices)),
        "score_delta_vs_full": fmt(mean(choice["chosen_score"] - choice["full_score"] for choice in choices)),
        "oracle_exact": fmt(mean(choice["chosen_exact"] for choice in choices)),
        "oracle_anls": fmt(mean(choice["chosen_anls"] for choice in choices)),
        "oracle_textvqa_accuracy": fmt(mean(choice["chosen_textvqa_accuracy"] for choice in choices)),
        "mean_keep": fmt(mean(choice["chosen_budget"] for choice in choices)),
        "escalate_rate_gt_0p30": fmt(mean(float(choice["required_escalation"]) for choice in choices)),
        "fallback_rate_ge_0p70": fmt(mean(float(choice["requires_70_or_full"]) for choice in choices)),
        "full_fallback_rate": fmt(mean(float(choice["requires_full"]) for choice in choices)),
        "choose_0p30": budget_counts[0.30],
        "choose_0p50": budget_counts[0.50],
        "choose_0p70": budget_counts[0.70],
        "choose_1p00": budget_counts[1.00],
        "note": "oracle cheapest budget within per-sample full-prefix score tolerance; upper bound only",
    }


def flatten_choice(task: str, tolerance: float, choice: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": task,
        "tolerance": fmt(tolerance),
        "sample_id": choice["sample_id"],
        "question_id": choice["question_id"],
        "split": choice["split"],
        "metric": choice["metric"],
        "full_score": fmt(choice["full_score"]),
        "low_score": fmt(choice["low_score"]),
        "chosen_budget": fmt(choice["chosen_budget"]),
        "chosen_score": fmt(choice["chosen_score"]),
        "chosen_loss_vs_full": fmt(choice["chosen_loss_vs_full"]),
        "required_escalation": int(choice["required_escalation"]),
        "requires_70_or_full": int(choice["requires_70_or_full"]),
        "requires_full": int(choice["requires_full"]),
        "raw_question": choice["raw_question"],
        "full_answer": choice["full_answer"],
        "low_answer": choice["low_answer"],
        "chosen_answer": choice["chosen_answer"],
    }


def build_report(summary_rows: list[dict[str, Any]]) -> str:
    selected = [
        row
        for row in summary_rows
        if row.get("split") in {"test", "all"} and row.get("tolerance") in {"0.000", "0.020", "0.050", "0.100"}
    ]
    lines = [
        "# Open OCR QA Risk-Coverage Frontier",
        "",
        "This audit reports an oracle upper bound for selective fallback on native open-answer TextVQA-lite and DocVQA-lite. For each sample and tolerance, it chooses the cheapest cached budget among 30%, 50%, 70%, and full prefix whose score is within the tolerance of that sample's full-prefix score. It is not a deployable policy because it uses gold evaluation scores after generation.",
        "",
        "## Key Frontier Rows",
        "",
        table_md(
            selected,
            [
                "task",
                "split",
                "tolerance",
                "n",
                "oracle_score",
                "full_score",
                "score_delta_vs_full",
                "mean_keep",
                "escalate_rate_gt_0p30",
                "fallback_rate_ge_0p70",
                "full_fallback_rate",
                "choose_0p30",
                "choose_0p50",
                "choose_0p70",
                "choose_1p00",
            ],
        ),
        "",
        "## Interpretation",
        "",
        "- The frontier quantifies the best possible selective-fallback controller under the already-computed budgets.",
        "- A low mean keep at tight tolerance would support an adaptive-risk strategy; a high full-fallback rate means that quality recovery depends on near-full prefixes.",
        "- These rows should be compared against deployable question-cue and box-aware policies, not presented as method results.",
    ]
    return "\n".join(lines) + "\n"


def split_for_id(sample_id: str) -> str:
    digest = hashlib.md5(sample_id.encode("utf-8")).hexdigest()
    return "dev" if int(digest[:8], 16) % 2 == 0 else "test"


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
    return f"{float(value):.3f}"


if __name__ == "__main__":
    main()
