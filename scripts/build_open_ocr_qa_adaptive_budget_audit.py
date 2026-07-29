#!/usr/bin/env python3
"""Build split-safe adaptive-budget diagnostics for native open OCR/DocQA runs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "problem_optimization_audit" / "open_ocr_qa_adaptive_budget"
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
class Policy:
    name: str
    family: str
    choose_budget: Callable[[dict[str, Any]], float]
    note: str


def main() -> None:
    rows = load_all_rows()
    policies = build_policies()
    summary_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []

    for task in sorted(rows):
        task_rows = rows[task]
        fixed = [policy for policy in policies if policy.family == "fixed"]
        oracle = [policy for policy in policies if policy.family == "oracle"]
        candidates = [policy for policy in policies if policy.family not in {"fixed", "oracle"}]
        for policy in fixed + oracle:
            summary_rows.extend(evaluate_policy(task_rows, policy, task=task, selected_by="preset"))

        for family in sorted({policy.family for policy in candidates}):
            family_policies = [policy for policy in candidates if policy.family == family]
            dev_scores = []
            for policy in family_policies:
                dev_summary = summarize(task_rows, policy, split="dev")
                dev_scores.append((objective(dev_summary), dev_summary, policy))
            dev_scores.sort(key=lambda item: item[0], reverse=True)
            best_obj, best_dev, best_policy = dev_scores[0]
            selection_rows.append(
                {
                    "task": task,
                    "family": family,
                    "selected_policy": best_policy.name,
                    "dev_objective": f"{best_obj:.6f}",
                    "dev_score": fmt(best_dev["score"]),
                    "dev_exact": fmt(best_dev["exact"]),
                    "dev_mean_keep": fmt(best_dev["mean_keep"]),
                    "dev_fallback_rate": fmt(best_dev["fallback_rate"]),
                    "note": best_policy.note,
                }
            )
            summary_rows.extend(evaluate_policy(task_rows, best_policy, task=task, selected_by="dev_best"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "adaptive_budget_summary.csv", summary_rows)
    write_csv(OUT_DIR / "adaptive_budget_policy_selection.csv", selection_rows)
    (OUT_DIR / "adaptive_budget_report.md").write_text(
        build_report(summary_rows, selection_rows),
        encoding="utf-8",
    )
    print(f"Wrote open OCR QA adaptive-budget audit to {OUT_DIR}")


def load_all_rows() -> dict[str, list[dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for task, by_budget in RUNS.items():
        task_out: dict[str, dict[str, Any]] = {}
        for budget, path in by_budget.items():
            for row in read_jsonl(path):
                sid = row["sample_id"]
                record = task_out.setdefault(
                    sid,
                    {
                        "task": task,
                        "sample_id": sid,
                        "question_id": row.get("question_id", ""),
                        "raw_question": row.get("raw_question") or row.get("question", ""),
                        "question": row.get("question", ""),
                        "gold_answers": row.get("gold_answers", []),
                        "metric": row.get("metric", ""),
                        "full_answer": row.get("full_answer", ""),
                        "full_score": float(row.get("full_score", 0.0)),
                        "full_exact": float(row.get("full_exact", 0.0)),
                        "full_anls": float(row.get("full_anls", 0.0)),
                        "full_textvqa_accuracy": float(row.get("full_textvqa_accuracy", 0.0)),
                        "target_text_token_count": int(row.get("target_text_token_count", 0) or 0),
                        "split": split_for_id(sid),
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
        for record in task_out.values():
            record["budgets"][1.00] = {
                "answer": record["full_answer"],
                "score": record["full_score"],
                "exact": record["full_exact"],
                "anls": record["full_anls"],
                "textvqa_accuracy": record["full_textvqa_accuracy"],
                "effective_keep": 1.0,
            }
        out[task] = list(task_out.values())
    return out


def split_for_id(sample_id: str) -> str:
    digest = hashlib.md5(sample_id.encode("utf-8")).hexdigest()
    return "dev" if int(digest[:8], 16) % 2 == 0 else "test"


def build_policies() -> list[Policy]:
    policies: list[Policy] = []
    for budget in BUDGETS:
        policies.append(
            Policy(
                name=f"fixed_{budget:.2f}",
                family="fixed",
                choose_budget=lambda _row, b=budget: b,
                note="fixed budget baseline",
            )
        )
    policies.append(
        Policy(
            name="oracle_best_budget",
            family="oracle",
            choose_budget=oracle_budget,
            note="uses gold score; diagnostic upper bound only",
        )
    )
    for mid_len in (8, 10, 12, 14, 16):
        for high_len in (14, 18, 22, 26, 30):
            if high_len <= mid_len:
                continue
            policies.append(
                Policy(
                    name=f"q_len_{mid_len}_{high_len}",
                    family="question_length",
                    choose_budget=lambda row, m=mid_len, h=high_len: length_budget(row, m, h),
                    note="question-only length threshold; 0.30/0.50/0.70 budgets",
                )
            )
    for cue_weight in (1, 2):
        for mid_score in (1, 2, 3):
            for high_score in (3, 4, 5):
                if high_score <= mid_score:
                    continue
                policies.append(
                    Policy(
                        name=f"q_cue_w{cue_weight}_{mid_score}_{high_score}",
                        family="question_cue",
                        choose_budget=lambda row, w=cue_weight, m=mid_score, h=high_score: cue_budget(row, w, m, h),
                        note="question-only cue score from length/numeric/date/layout/multi-constraint terms",
                    )
                )
    for cue_weight in (1, 2):
        for mid_score in (1, 2, 3):
            for high_score in (3, 4, 5):
                if high_score <= mid_score:
                    continue
                policies.append(
                    Policy(
                        name=f"q_cue_full_w{cue_weight}_{mid_score}_{high_score}",
                        family="question_cue_fullfallback",
                        choose_budget=lambda row, w=cue_weight, m=mid_score, h=high_score: cue_full_budget(row, w, m, h),
                        note="question-only cue score with full-prefix fallback for highest-risk questions",
                    )
                )
    for length_thr in (10, 14, 18, 22):
        for answer_thr in (4, 8, 12):
            policies.append(
                Policy(
                    name=f"low_answer_len{length_thr}_out{answer_thr}",
                    family="question_plus_low_answer",
                    choose_budget=lambda row, l=length_thr, a=answer_thr: low_answer_budget(row, l, a),
                    note="starts at 0.30 and escalates using question cues plus 0.30 answer-shape checks",
                )
            )
            policies.append(
                Policy(
                    name=f"low_answer_full_len{length_thr}_out{answer_thr}",
                    family="question_plus_low_answer_fullfallback",
                    choose_budget=lambda row, l=length_thr, a=answer_thr: low_answer_full_budget(row, l, a),
                    note="starts at 0.30 and can fall back to full prefix using question cues plus 0.30 answer-shape checks",
                )
            )
    return policies


def length_budget(row: dict[str, Any], mid_len: int, high_len: int) -> float:
    q_len = len(norm_tokens(row["raw_question"]))
    if q_len >= high_len:
        return 0.70
    if q_len >= mid_len:
        return 0.50
    return 0.30


def cue_budget(row: dict[str, Any], cue_weight: int, mid_score: int, high_score: int) -> float:
    score = question_risk_score(row, cue_weight=cue_weight)
    if score >= high_score:
        return 0.70
    if score >= mid_score:
        return 0.50
    return 0.30


def cue_full_budget(row: dict[str, Any], cue_weight: int, mid_score: int, high_score: int) -> float:
    score = question_risk_score(row, cue_weight=cue_weight)
    if score >= high_score:
        return 1.00
    if score >= mid_score:
        return 0.70
    return 0.30


def low_answer_budget(row: dict[str, Any], length_thr: int, answer_thr: int) -> float:
    q_score = question_risk_score(row, cue_weight=1)
    low_answer = str(row["budgets"][0.30]["answer"])
    q_text = row["raw_question"].lower()
    low_tokens = norm_tokens(low_answer)
    repeated = has_repetition(low_tokens)
    too_long = len(low_tokens) >= answer_thr
    missing_digit = asks_numeric(q_text) and not re.search(r"\d", low_answer)
    empty = not low_answer.strip()
    q_len = len(norm_tokens(row["raw_question"]))
    if repeated or too_long or missing_digit or empty or (q_score >= 4 and q_len >= length_thr):
        return 0.70
    if q_score >= 2 or q_len >= length_thr:
        return 0.50
    return 0.30


def low_answer_full_budget(row: dict[str, Any], length_thr: int, answer_thr: int) -> float:
    q_score = question_risk_score(row, cue_weight=1)
    low_answer = str(row["budgets"][0.30]["answer"])
    q_text = row["raw_question"].lower()
    low_tokens = norm_tokens(low_answer)
    repeated = has_repetition(low_tokens)
    too_long = len(low_tokens) >= answer_thr
    missing_digit = asks_numeric(q_text) and not re.search(r"\d", low_answer)
    empty = not low_answer.strip()
    q_len = len(norm_tokens(row["raw_question"]))
    if repeated or too_long or empty or (q_score >= 4 and q_len >= length_thr):
        return 1.00
    if missing_digit or q_score >= 2 or q_len >= length_thr:
        return 0.70
    return 0.30


def oracle_budget(row: dict[str, Any]) -> float:
    best_budget = 0.30
    best_score = -1.0
    for budget in BUDGETS:
        item = row["budgets"][budget]
        score = float(item["score"])
        if score > best_score or (score == best_score and budget < best_budget):
            best_budget = budget
            best_score = score
    return best_budget


def question_risk_score(row: dict[str, Any], *, cue_weight: int) -> int:
    text = row["raw_question"].lower()
    tokens = norm_tokens(text)
    score = 0
    if len(tokens) >= 10:
        score += 1
    if len(tokens) >= 18:
        score += 1
    if asks_numeric(text):
        score += cue_weight
    if any(cue in text for cue in ("date", "year", "phone", "number", "amount", "total", "how many", "percent", "value")):
        score += cue_weight
    if any(cue in text for cue in ("according to", "during", "between", "from", "per", "under", "which", "where")):
        score += 1
    if any(cue in text for cue in (" and ", " or ", " with ", " of the ", " in the ")):
        score += 1
    return score


def asks_numeric(text: str) -> bool:
    return bool(re.search(r"\b(how many|number|date|year|amount|total|percent|value|phone|per 1000)\b", text))


def has_repetition(tokens: list[str]) -> bool:
    if len(tokens) < 6:
        return False
    for n in (1, 2, 3):
        chunks = [tuple(tokens[i : i + n]) for i in range(0, min(len(tokens) - n + 1, 12), n)]
        if len(chunks) >= 3 and len(set(chunks[:3])) == 1:
            return True
    return False


def evaluate_policy(rows: list[dict[str, Any]], policy: Policy, *, task: str, selected_by: str) -> list[dict[str, Any]]:
    out = []
    for split in ("dev", "test", "all"):
        summary = summarize(rows, policy, split=split)
        out.append(
            {
                "task": task,
                "split": split,
                "policy": policy.name,
                "family": policy.family,
                "selected_by": selected_by,
                "n": summary["n"],
                "score": fmt(summary["score"]),
                "exact": fmt(summary["exact"]),
                "anls": fmt(summary["anls"]),
                "textvqa_accuracy": fmt(summary["textvqa_accuracy"]),
                "mean_keep": fmt(summary["mean_keep"]),
                "fallback_rate_ge_0p50": fmt(summary["fallback_rate"]),
                "full_fallback_rate": fmt(summary["full_fallback_rate"]),
                "score_per_keep_objective": fmt(objective(summary)),
                "note": policy.note,
            }
        )
    return out


def summarize(rows: list[dict[str, Any]], policy: Policy, *, split: str) -> dict[str, Any]:
    subset = [row for row in rows if split == "all" or row["split"] == split]
    scores = []
    exacts = []
    anls = []
    textvqa_acc = []
    keeps = []
    for row in subset:
        budget = policy.choose_budget(row)
        item = row["budgets"][budget]
        scores.append(float(item["score"]))
        exacts.append(float(item["exact"]))
        anls.append(float(item["anls"]))
        textvqa_acc.append(float(item["textvqa_accuracy"]))
        keeps.append(float(item["effective_keep"]))
    return {
        "n": len(subset),
        "score": mean(scores) if scores else math.nan,
        "exact": mean(exacts) if exacts else math.nan,
        "anls": mean(anls) if anls else math.nan,
        "textvqa_accuracy": mean(textvqa_acc) if textvqa_acc else math.nan,
        "mean_keep": mean(keeps) if keeps else math.nan,
        "fallback_rate": mean([k >= 0.50 for k in keeps]) if keeps else math.nan,
        "full_fallback_rate": mean([k >= 1.0 for k in keeps]) if keeps else math.nan,
    }


def objective(summary: dict[str, Any]) -> float:
    if math.isnan(float(summary["score"])):
        return -1e9
    # Prefer quality, but mildly penalize token use so policies that only choose
    # the largest budget do not dominate the deployable search.
    return float(summary["score"]) - 0.10 * float(summary["mean_keep"])


def build_report(summary_rows: list[dict[str, Any]], selection_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Open OCR QA Adaptive Budget Audit",
        "",
        "This split-safe diagnostic reuses cached Qwen native generation runs at 30%, 50%, and 70% visual-token retention. Full-prefix scores are used as a 100% budget row. Oracle rows use gold scores and are upper bounds only.",
        "",
        "## Selected Deployable Policies",
        "",
        "| Task | Family | Selected policy | Dev score | Dev keep | Note |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in selection_rows:
        lines.append(
            f"| {row['task']} | {row['family']} | {row['selected_policy']} | "
            f"{row['dev_score']} | {row['dev_mean_keep']} | {row['note']} |"
        )
    lines.extend(
        [
            "",
            "## Test Summary",
            "",
            "| Task | Policy | Family | Score | Exact | Mean keep | Fallback >=0.50 | Note |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in summary_rows:
        if row["split"] != "test":
            continue
        if row["family"] == "fixed" and row["policy"] not in {"fixed_0.30", "fixed_0.50", "fixed_0.70", "fixed_1.00"}:
            continue
        lines.append(
            f"| {row['task']} | {row['policy']} | {row['family']} | {row['score']} | "
            f"{row['exact']} | {row['mean_keep']} | {row['fallback_rate_ge_0p50']} | {row['note']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Fixed high retention remains the strongest deployable option in this audit for open DocVQA-style QA.",
            "- Question-side and low-answer-shape policies test whether a cheap risk rule can recover quality without using gold labels.",
            "- The oracle row quantifies budget-selection headroom and must not be reported as a deployable method.",
        ]
    )
    return "\n".join(lines) + "\n"


def norm_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


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


def fmt(value: Any) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(x):
        return ""
    return f"{x:.3f}"


if __name__ == "__main__":
    main()
