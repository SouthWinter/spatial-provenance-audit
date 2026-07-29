#!/usr/bin/env python3
"""Audit answer-stability cascades for open OCR/DocQA budget fallback."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "problem_optimization_audit" / "open_ocr_qa_answer_stability_cascade"
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


@dataclass(frozen=True)
class Decision:
    chosen_budget: float
    cascade_cost: float
    reason: str


@dataclass(frozen=True)
class Policy:
    name: str
    family: str
    decide: Callable[[dict[str, Any]], Decision]
    note: str


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    rows = load_rows()
    policies = build_policies()
    summary_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []

    for task in sorted(rows):
        task_rows = rows[task]
        for policy in policies:
            for split in ("dev", "test", "all"):
                scoped = [row for row in task_rows if split == "all" or row["split"] == split]
                summary_rows.append(summarize(task, split, policy, scoped, selected_by="preset"))
        for family in sorted({p.family for p in policies if p.family not in {"fixed", "oracle"}}):
            candidates = [p for p in policies if p.family == family]
            ranked = []
            for policy in candidates:
                dev = summarize(task, "dev", policy, [r for r in task_rows if r["split"] == "dev"], selected_by="dev_candidate")
                ranked.append((objective(dev), dev, policy))
            ranked.sort(key=lambda item: item[0], reverse=True)
            _, dev_summary, best_policy = ranked[0]
            test_summary = summarize(task, "test", best_policy, [r for r in task_rows if r["split"] == "test"], selected_by="dev_selected")
            selection_rows.append(
                {
                    "task": task,
                    "family": family,
                    "selected_policy": best_policy.name,
                    "dev_score": dev_summary["score"],
                    "dev_selected_keep": dev_summary["selected_mean_keep"],
                    "dev_cascade_cost": dev_summary["cascade_mean_cost"],
                    "test_score": test_summary["score"],
                    "test_selected_keep": test_summary["selected_mean_keep"],
                    "test_cascade_cost": test_summary["cascade_mean_cost"],
                    "test_full_fallback_rate": test_summary["full_fallback_rate"],
                    "note": best_policy.note,
                }
            )
            example_rows.extend(build_examples(task, task_rows, best_policy))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "answer_stability_cascade_summary.csv", summary_rows)
    write_csv(out_dir / "answer_stability_cascade_selection.csv", selection_rows)
    write_csv(out_dir / "answer_stability_cascade_examples.csv", example_rows)
    (out_dir / "answer_stability_cascade_report.md").write_text(
        build_report(summary_rows, selection_rows, example_rows), encoding="utf-8"
    )
    print(f"Wrote answer-stability cascade audit to {out_dir}")


def load_rows() -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for task, by_budget in RUNS.items():
        by_sample: dict[str, dict[str, Any]] = {}
        for budget, path in by_budget.items():
            for row in read_jsonl(path):
                sid = row["sample_id"]
                record = by_sample.setdefault(
                    sid,
                    {
                        "task": task,
                        "sample_id": sid,
                        "question_id": row.get("question_id", ""),
                        "split": split_for_id(sid),
                        "raw_question": row.get("raw_question") or row.get("question", ""),
                        "gold_answers": row.get("gold_answers", []),
                        "metric": row.get("metric", ""),
                        "full_answer": row.get("full_answer", ""),
                        "full_score": float(row.get("full_score", 0.0)),
                        "full_exact": float(row.get("full_exact", 0.0)),
                        "budgets": {},
                    },
                )
                record["budgets"][budget] = {
                    "answer": row.get("pruned_answer", ""),
                    "score": float(row.get("pruned_score", 0.0)),
                    "exact": float(row.get("pruned_exact", 0.0)),
                    "effective_keep": float(row.get("effective_keep_ratio", budget) or budget),
                }
        for record in by_sample.values():
            record["budgets"][1.00] = {
                "answer": record["full_answer"],
                "score": record["full_score"],
                "exact": record["full_exact"],
                "effective_keep": 1.0,
            }
        out[task] = list(by_sample.values())
    return out


def build_policies() -> list[Policy]:
    policies: list[Policy] = []
    for budget in BUDGETS:
        policies.append(
            Policy(
                name=f"fixed_{budget:.2f}",
                family="fixed",
                decide=lambda _row, b=budget: Decision(b, b, "fixed"),
                note="fixed budget baseline",
            )
        )
    policies.append(
        Policy(
            name="oracle_best_budget",
            family="oracle",
            decide=oracle_decision,
            note="uses gold evaluation scores; diagnostic upper bound only",
        )
    )
    policies.extend(
        [
            Policy(
                name="stable30_else70",
                family="two_pass_stability",
                decide=lambda row: stable_pair(row, accept_budget=0.30, fallback_budget=0.70),
                note="run 30% and 50%; accept 30% only if normalized answers agree, otherwise use 70%",
            ),
            Policy(
                name="stable50_else70",
                family="two_pass_stability",
                decide=lambda row: stable_pair(row, accept_budget=0.50, fallback_budget=0.70),
                note="run 30% and 50%; accept 50% if answers agree, otherwise use 70%",
            ),
            Policy(
                name="stable30_elsefull",
                family="two_pass_stability_full",
                decide=lambda row: stable_pair(row, accept_budget=0.30, fallback_budget=1.00),
                note="run 30% and 50%; accept 30% if answers agree, otherwise use full prefix",
            ),
            Policy(
                name="stable50_elsefull",
                family="two_pass_stability_full",
                decide=lambda row: stable_pair(row, accept_budget=0.50, fallback_budget=1.00),
                note="run 30% and 50%; accept 50% if answers agree, otherwise use full prefix",
            ),
            Policy(
                name="staged_30_50_then_70_full",
                family="three_pass_stability",
                decide=staged_stability,
                note="accept 30% if 30/50 agree; else accept 70% if 50/70 agree; otherwise full",
            ),
            Policy(
                name="majority_30_50_70_elsefull",
                family="three_pass_majority",
                decide=majority_stability,
                note="run 30/50/70; choose cheapest normalized answer with a majority, otherwise full",
            ),
        ]
    )
    return policies


def stable_pair(row: dict[str, Any], accept_budget: float, fallback_budget: float) -> Decision:
    agree = norm_answer(row["budgets"][0.30]["answer"]) == norm_answer(row["budgets"][0.50]["answer"])
    if agree:
        return Decision(accept_budget, 0.30 + 0.50, "answer30_eq_answer50")
    return Decision(fallback_budget, 0.30 + 0.50 + fallback_budget, "answer30_ne_answer50")


def staged_stability(row: dict[str, Any]) -> Decision:
    a30 = norm_answer(row["budgets"][0.30]["answer"])
    a50 = norm_answer(row["budgets"][0.50]["answer"])
    if a30 == a50:
        return Decision(0.30, 0.80, "answer30_eq_answer50")
    a70 = norm_answer(row["budgets"][0.70]["answer"])
    if a50 == a70:
        return Decision(0.70, 1.50, "answer50_eq_answer70")
    return Decision(1.00, 2.50, "unstable_30_50_70")


def majority_stability(row: dict[str, Any]) -> Decision:
    answers = {
        0.30: norm_answer(row["budgets"][0.30]["answer"]),
        0.50: norm_answer(row["budgets"][0.50]["answer"]),
        0.70: norm_answer(row["budgets"][0.70]["answer"]),
    }
    for budget in (0.30, 0.50, 0.70):
        if answers[budget] and sum(answers[budget] == ans for ans in answers.values()) >= 2:
            return Decision(budget, 1.50, "majority_answer")
    return Decision(1.00, 2.50, "no_majority_answer")


def oracle_decision(row: dict[str, Any]) -> Decision:
    full_score = float(row["full_score"])
    target = full_score
    for budget in BUDGETS:
        if float(row["budgets"][budget]["score"]) + 1e-12 >= target:
            return Decision(budget, budget, "oracle_within_0tol")
    return Decision(1.00, 1.00, "oracle_default_full")


def summarize(
    task: str,
    split: str,
    policy: Policy,
    rows: list[dict[str, Any]],
    *,
    selected_by: str,
) -> dict[str, Any]:
    if not rows:
        return {
            "task": task,
            "split": split,
            "policy": policy.name,
            "family": policy.family,
            "selected_by": selected_by,
            "n": 0,
        }
    chosen = [(row, policy.decide(row)) for row in rows]
    scores = [row["budgets"][decision.chosen_budget]["score"] for row, decision in chosen]
    exacts = [row["budgets"][decision.chosen_budget]["exact"] for row, decision in chosen]
    selected_keeps = [row["budgets"][decision.chosen_budget]["effective_keep"] for row, decision in chosen]
    cascade_costs = [decision.cascade_cost for _row, decision in chosen]
    full_scores = [row["full_score"] for row, _decision in chosen]
    agreement_30_50 = [
        norm_answer(row["budgets"][0.30]["answer"]) == norm_answer(row["budgets"][0.50]["answer"])
        for row, _decision in chosen
    ]
    return {
        "task": task,
        "split": split,
        "policy": policy.name,
        "family": policy.family,
        "selected_by": selected_by,
        "n": len(rows),
        "score": fmt(mean(scores)),
        "full_score": fmt(mean(full_scores)),
        "delta_vs_full": fmt(mean(scores) - mean(full_scores)),
        "exact": fmt(mean(exacts)),
        "selected_mean_keep": fmt(mean(selected_keeps)),
        "cascade_mean_cost": fmt(mean(cascade_costs)),
        "cascade_cost_vs_full": fmt(mean(cascade_costs) / 1.0),
        "fallback_rate_ge_0p70": fmt(mean(decision.chosen_budget >= 0.70 for _row, decision in chosen)),
        "full_fallback_rate": fmt(mean(decision.chosen_budget >= 1.00 for _row, decision in chosen)),
        "answer30_50_agreement_rate": fmt(mean(agreement_30_50)),
        "choose_0p30": sum(decision.chosen_budget == 0.30 for _row, decision in chosen),
        "choose_0p50": sum(decision.chosen_budget == 0.50 for _row, decision in chosen),
        "choose_0p70": sum(decision.chosen_budget == 0.70 for _row, decision in chosen),
        "choose_1p00": sum(decision.chosen_budget == 1.00 for _row, decision in chosen),
        "note": policy.note,
    }


def objective(summary: dict[str, Any]) -> float:
    if not summary or int(summary.get("n", 0) or 0) == 0:
        return -1e9
    score = float(summary["score"])
    selected_keep = float(summary["selected_mean_keep"])
    cascade_cost = float(summary["cascade_mean_cost"])
    # Favor quality first, then selected prefix, while penalizing serial reruns.
    return score - 0.08 * selected_keep - 0.03 * max(0.0, cascade_cost - 1.0)


def build_examples(task: str, rows: list[dict[str, Any]], policy: Policy) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in rows:
        if row["split"] != "test":
            continue
        decision = policy.decide(row)
        chosen = row["budgets"][decision.chosen_budget]
        full = row["full_score"]
        low = row["budgets"][0.30]["score"]
        if len(examples) < 80 and (decision.chosen_budget >= 0.70 or chosen["score"] + 0.25 < full or low + 0.25 < full <= chosen["score"] + 0.10):
            examples.append(
                {
                    "task": task,
                    "policy": policy.name,
                    "sample_id": row["sample_id"],
                    "reason": decision.reason,
                    "chosen_budget": f"{decision.chosen_budget:.2f}",
                    "cascade_cost": fmt(decision.cascade_cost),
                    "full_score": fmt(row["full_score"]),
                    "score30": fmt(row["budgets"][0.30]["score"]),
                    "score50": fmt(row["budgets"][0.50]["score"]),
                    "score70": fmt(row["budgets"][0.70]["score"]),
                    "chosen_score": fmt(chosen["score"]),
                    "answer30": row["budgets"][0.30]["answer"],
                    "answer50": row["budgets"][0.50]["answer"],
                    "answer70": row["budgets"][0.70]["answer"],
                    "full_answer": row["full_answer"],
                    "question": row["raw_question"],
                }
            )
    return examples


def norm_answer(text: Any) -> str:
    raw = str(text).lower()
    raw = raw.replace(",", "")
    return " ".join(re.findall(r"[a-z0-9]+", raw))


def split_for_id(sample_id: str) -> str:
    digest = hashlib.md5(sample_id.encode("utf-8")).hexdigest()
    return "dev" if int(digest[:8], 16) % 2 == 0 else "test"


def fmt(value: float) -> str:
    return f"{float(value):.3f}"


def build_report(
    summary_rows: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
    example_rows: list[dict[str, Any]],
) -> str:
    key_rows = [
        row
        for row in summary_rows
        if row.get("split") == "test"
        and row.get("policy") in {
            "fixed_0.30",
            "fixed_0.50",
            "fixed_0.70",
            "fixed_1.00",
            "oracle_best_budget",
            "stable30_else70",
            "stable50_else70",
            "stable30_elsefull",
            "staged_30_50_then_70_full",
            "majority_30_50_70_elsefull",
        }
    ]
    lines = [
        "# Open OCR QA Answer-Stability Cascade",
        "",
        "This audit simulates selective fallback policies that use agreement between cached 30%, 50%, and 70% answers. It is deployability-adjacent rather than free: the report distinguishes the selected visual-prefix keep ratio from the cumulative serial cascade cost.",
        "",
        "## Test Summary",
        "",
        "| Task | Policy | Score | Delta | Selected keep | Cascade cost | >=70 fallback | Full fallback | 30/50 agree |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in key_rows:
        lines.append(
            f"| {row['task']} | {row['policy']} | {row['score']} | {row['delta_vs_full']} | "
            f"{row['selected_mean_keep']} | {row['cascade_mean_cost']} | {row['fallback_rate_ge_0p70']} | "
            f"{row['full_fallback_rate']} | {row['answer30_50_agreement_rate']} |"
        )
    lines.extend(
        [
            "",
            "## Dev-Selected Policies",
            "",
            "| Task | Family | Policy | Dev score | Dev keep | Dev cascade | Test score | Test keep | Test cascade | Full fallback |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in selection_rows:
        lines.append(
            f"| {row['task']} | {row['family']} | {row['selected_policy']} | {row['dev_score']} | "
            f"{row['dev_selected_keep']} | {row['dev_cascade_cost']} | {row['test_score']} | "
            f"{row['test_selected_keep']} | {row['test_cascade_cost']} | {row['test_full_fallback_rate']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Safe claim: answer stability is an informative fallback signal, but serial cascades can cost more cumulative prefill work than a single full-prefix pass. Unsafe claim: this is a solved efficient adaptive controller.",
            "",
            "## Example Rows",
            "",
            "| Task | Policy | Sample | Reason | Chosen | Cost | Full | 30 | 50 | 70 | Chosen score |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in example_rows[:80]:
        lines.append(
            f"| {row['task']} | {row['policy']} | {row['sample_id']} | {row['reason']} | "
            f"{row['chosen_budget']} | {row['cascade_cost']} | {row['full_score']} | "
            f"{row['score30']} | {row['score50']} | {row['score70']} | {row['chosen_score']} |"
        )
    return "\n".join(lines) + "\n"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


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
