#!/usr/bin/env python3
"""Audit whether open-QA pruning failures are repairable by larger budgets.

This script is a diagnostic for the adaptive-control gap in problem.md. It does
not run models. It reuses cached Qwen TextVQA/DocVQA open-answer generations and
asks:

1. Which low-budget (30%) drops are repaired by 70% or full-prefix budgets?
2. How well do cheap deployable features identify risky or repairable samples?
3. How much stronger are evidence-audit features such as ECR on annotated rows?
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "problem_optimization_audit" / "open_ocr_qa_repairability"

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

ECR_ROWS = (
    ROOT
    / "runs/problem_optimization_audit/open_ocr_qa_ecr_quality_association/ecr_quality_rows.csv"
)
LINE_CONTEXT_ECR_ROWS = (
    ROOT
    / "runs/problem_optimization_audit/docvqa_line_context_quality_association/line_context_quality_rows.csv"
)

LOW = 0.30
MID = 0.50
HIGH = 0.70
FULL = 1.00
DROP_EPS = 0.25
RECOVER_CLOSE_EPS = 0.10


def main() -> None:
    rows = load_generation_rows()
    ecr_by_sample_budget = load_ecr_features()
    repair_rows = [build_repair_row(row, ecr_by_sample_budget) for row in rows]
    summary_rows = build_summary(repair_rows)
    feature_rows = build_feature_summary(repair_rows)
    example_rows = build_examples(repair_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "repairability_rows.csv", repair_rows)
    write_csv(OUT_DIR / "repairability_summary.csv", summary_rows)
    write_csv(OUT_DIR / "repairability_feature_summary.csv", feature_rows)
    write_csv(OUT_DIR / "repairability_examples.csv", example_rows)
    (OUT_DIR / "repairability_report.md").write_text(
        build_report(summary_rows, feature_rows, example_rows),
        encoding="utf-8",
    )
    print(f"Wrote open-QA repairability audit to {OUT_DIR}")


def load_generation_rows() -> list[dict[str, Any]]:
    by_task_sample: dict[tuple[str, str], dict[str, Any]] = {}
    for task, by_budget in RUNS.items():
        for budget, path in by_budget.items():
            for row in read_jsonl(path):
                key = (task, row["sample_id"])
                record = by_task_sample.setdefault(
                    key,
                    {
                        "task": task,
                        "sample_id": row["sample_id"],
                        "question_id": row.get("question_id", ""),
                        "raw_question": row.get("raw_question") or row.get("question", ""),
                        "gold_answers": row.get("gold_answers", []),
                        "metric": row.get("metric", ""),
                        "full_answer": row.get("full_answer", ""),
                        "full_score": float(row.get("full_score", 0.0)),
                        "full_exact": float(row.get("full_exact", 0.0)),
                        "target_text_token_count": int(row.get("target_text_token_count", 0) or 0),
                        "budgets": {},
                    },
                )
                record["budgets"][budget] = {
                    "answer": row.get("pruned_answer", ""),
                    "score": float(row.get("pruned_score", 0.0)),
                    "exact": float(row.get("pruned_exact", 0.0)),
                    "effective_keep": float(row.get("effective_keep_ratio", budget) or budget),
                }
    out = []
    for record in by_task_sample.values():
        record["budgets"][FULL] = {
            "answer": record["full_answer"],
            "score": record["full_score"],
            "exact": record["full_exact"],
            "effective_keep": 1.0,
        }
        if all(b in record["budgets"] for b in (LOW, MID, HIGH, FULL)):
            out.append(record)
    return sorted(out, key=lambda r: (r["task"], r["sample_id"]))


def load_ecr_features() -> dict[tuple[str, str, float], dict[str, float]]:
    out: dict[tuple[str, str, float], dict[str, float]] = {}
    for path, source in ((ECR_ROWS, "answer_or_gt_bbox"), (LINE_CONTEXT_ECR_ROWS, "docvqa_line_context")):
        if not path.exists():
            continue
        for row in read_csv(path):
            task = row.get("task") or ("DocVQA-line-context" if source == "docvqa_line_context" else "")
            if source == "docvqa_line_context":
                task = "DocVQA-lite"
            sample_id = row.get("sample_id", "")
            budget = parse_float(row.get("budget_keep_ratio", "nan"))
            if not sample_id or math.isnan(budget):
                continue
            key = (task, sample_id, budget)
            prefix = "line_context_" if source == "docvqa_line_context" else ""
            item = out.setdefault(key, {})
            for name in ("ECR", "worst_region_ECR", "all_regions_ECR_ge_0p50", "box_count"):
                item[prefix + name] = parse_float(row.get(name, "nan"))
    return out


def build_repair_row(row: dict[str, Any], ecr_by_sample_budget: dict[tuple[str, str, float], dict[str, float]]) -> dict[str, Any]:
    full_score = float(row["budgets"][FULL]["score"])
    low_score = float(row["budgets"][LOW]["score"])
    mid_score = float(row["budgets"][MID]["score"])
    high_score = float(row["budgets"][HIGH]["score"])
    low_drop = full_score - low_score
    high_gain = high_score - low_score
    mid_gain = mid_score - low_score
    full_gain = full_score - low_score
    low_failure = low_drop >= DROP_EPS
    repaired_by_50 = low_failure and mid_score >= full_score - RECOVER_CLOSE_EPS
    repaired_by_70 = low_failure and high_score >= full_score - RECOVER_CLOSE_EPS
    repaired_by_full = low_failure and full_score >= low_score + DROP_EPS
    high_still_bad = low_failure and not repaired_by_70

    q = row["raw_question"]
    low_answer = str(row["budgets"][LOW]["answer"])
    q_tokens = norm_tokens(q)
    low_tokens = norm_tokens(low_answer)
    q_risk = question_risk_score(q)
    ecr = ecr_by_sample_budget.get((row["task"], row["sample_id"], LOW), {})
    ecr_high = ecr_by_sample_budget.get((row["task"], row["sample_id"], HIGH), {})

    out = {
        "task": row["task"],
        "sample_id": row["sample_id"],
        "question_id": row["question_id"],
        "split": split_for_id(row["sample_id"]),
        "metric": row["metric"],
        "full_score": fmt(full_score),
        "score_0p30": fmt(low_score),
        "score_0p50": fmt(mid_score),
        "score_0p70": fmt(high_score),
        "low_drop": fmt(low_drop),
        "gain_30_to_50": fmt(mid_gain),
        "gain_30_to_70": fmt(high_gain),
        "gain_30_to_full": fmt(full_gain),
        "low_failure_drop_ge_0p25": int(low_failure),
        "repaired_by_50_close_to_full": int(repaired_by_50),
        "repaired_by_70_close_to_full": int(repaired_by_70),
        "repaired_by_full": int(repaired_by_full),
        "high_still_bad": int(high_still_bad),
        "question_len": len(q_tokens),
        "question_risk_score": q_risk,
        "target_text_token_count": row["target_text_token_count"],
        "low_answer_len": len(low_tokens),
        "low_answer_empty": int(not low_answer.strip()),
        "low_answer_has_digit": int(bool(re.search(r"\d", low_answer))),
        "low_answer_repetition": int(has_repetition(low_tokens)),
        "asks_numeric": int(asks_numeric(q.lower())),
        "question_multi_constraint": int(any(cue in q.lower() for cue in (" and ", " or ", " with ", " of the ", " in the "))),
        "ECR_0p30": fmt(ecr.get("ECR", math.nan)),
        "worst_region_ECR_0p30": fmt(ecr.get("worst_region_ECR", math.nan)),
        "all_regions_ECR_ge_0p50_0p30": fmt(ecr.get("all_regions_ECR_ge_0p50", math.nan)),
        "box_count": fmt(ecr.get("box_count", math.nan)),
        "ECR_0p70": fmt(ecr_high.get("ECR", math.nan)),
        "worst_region_ECR_0p70": fmt(ecr_high.get("worst_region_ECR", math.nan)),
        "line_context_ECR_0p30": fmt(ecr.get("line_context_ECR", math.nan)),
        "line_context_worst_region_ECR_0p30": fmt(ecr.get("line_context_worst_region_ECR", math.nan)),
        "line_context_ECR_0p70": fmt(ecr_high.get("line_context_ECR", math.nan)),
        "line_context_worst_region_ECR_0p70": fmt(ecr_high.get("line_context_worst_region_ECR", math.nan)),
        "raw_question": q,
        "gold_answers": " | ".join(map(str, row["gold_answers"])),
        "full_answer": row["full_answer"],
        "answer_0p30": low_answer,
        "answer_0p70": row["budgets"][HIGH]["answer"],
    }
    return out


def build_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for task in sorted({r["task"] for r in rows}) + ["all"]:
        subset = rows if task == "all" else [r for r in rows if r["task"] == task]
        low_fail = [r for r in subset if int(r["low_failure_drop_ge_0p25"])]
        out.append(
            {
                "task": task,
                "n": len(subset),
                "low_failure_n": len(low_fail),
                "low_failure_rate": fmt(rate(len(low_fail), len(subset))),
                "repaired_by_50_n": sum_int(low_fail, "repaired_by_50_close_to_full"),
                "repaired_by_50_rate_among_low_fail": fmt(rate(sum_int(low_fail, "repaired_by_50_close_to_full"), len(low_fail))),
                "repaired_by_70_n": sum_int(low_fail, "repaired_by_70_close_to_full"),
                "repaired_by_70_rate_among_low_fail": fmt(rate(sum_int(low_fail, "repaired_by_70_close_to_full"), len(low_fail))),
                "high_still_bad_n": sum_int(low_fail, "high_still_bad"),
                "high_still_bad_rate_among_low_fail": fmt(rate(sum_int(low_fail, "high_still_bad"), len(low_fail))),
                "mean_low_drop": fmt(mean_float(subset, "low_drop")),
                "mean_gain_30_to_70": fmt(mean_float(subset, "gain_30_to_70")),
                "mean_gain_30_to_70_low_fail": fmt(mean_float(low_fail, "gain_30_to_70")),
            }
        )
    return out


def build_feature_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deployable = [
        "question_len",
        "question_risk_score",
        "target_text_token_count",
        "low_answer_len",
        "low_answer_empty",
        "low_answer_has_digit",
        "low_answer_repetition",
        "asks_numeric",
        "question_multi_constraint",
    ]
    evidence = [
        "ECR_0p30",
        "worst_region_ECR_0p30",
        "all_regions_ECR_ge_0p50_0p30",
        "box_count",
        "line_context_ECR_0p30",
        "line_context_worst_region_ECR_0p30",
    ]
    targets = [
        ("low_failure_drop_ge_0p25", rows),
        ("repaired_by_70_close_to_full", [r for r in rows if int(r["low_failure_drop_ge_0p25"])]),
        ("high_still_bad", [r for r in rows if int(r["low_failure_drop_ge_0p25"])]),
    ]
    out = []
    for target, target_rows in targets:
        for scope in ("all", "TextVQA-lite", "DocVQA-lite"):
            scoped = target_rows if scope == "all" else [r for r in target_rows if r["task"] == scope]
            for feature in deployable + evidence:
                xs, ys = paired_feature_target(scoped, feature, target)
                if len(set(ys)) < 2 or len(xs) < 4:
                    continue
                auc_pos = auc(xs, ys)
                auc_neg = auc([-x for x in xs], ys)
                best_auc = max(auc_pos, auc_neg)
                direction = "higher_risk" if auc_pos >= auc_neg else "lower_risk"
                out.append(
                    {
                        "target": target,
                        "scope": scope,
                        "feature_group": "deployable" if feature in deployable else "evidence_audit",
                        "feature": feature,
                        "n": len(xs),
                        "positive_rate": fmt(mean(ys)),
                        "auc_best_direction": fmt(best_auc),
                        "direction": direction,
                        "pearson_with_target": fmt(pearson(xs, ys)),
                        "spearman_with_target": fmt(spearman(xs, ys)),
                    }
                )
    return out


def build_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    groups = [
        ("repaired_by_70", [r for r in rows if int(r["repaired_by_70_close_to_full"])], "gain_30_to_70", True),
        ("not_repaired_by_70", [r for r in rows if int(r["high_still_bad"])], "low_drop", True),
        ("low_risk_good_30", [r for r in rows if not int(r["low_failure_drop_ge_0p25"])], "score_0p30", True),
    ]
    for label, subset, key, reverse in groups:
        subset = sorted(subset, key=lambda r: parse_float(r[key]), reverse=reverse)
        for row in subset[:8]:
            examples.append(
                {
                    "case_type": label,
                    "task": row["task"],
                    "sample_id": row["sample_id"],
                    "full_score": row["full_score"],
                    "score_0p30": row["score_0p30"],
                    "score_0p70": row["score_0p70"],
                    "low_drop": row["low_drop"],
                    "gain_30_to_70": row["gain_30_to_70"],
                    "question_risk_score": row["question_risk_score"],
                    "ECR_0p30": row["ECR_0p30"],
                    "ECR_0p70": row["ECR_0p70"],
                    "raw_question": row["raw_question"],
                    "full_answer": row["full_answer"],
                    "answer_0p30": row["answer_0p30"],
                    "answer_0p70": row["answer_0p70"],
                }
            )
    return examples


def build_report(summary_rows: list[dict[str, Any]], feature_rows: list[dict[str, Any]], example_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Open OCR QA Repairability Audit",
        "",
        "This diagnostic reuses cached Qwen3-VL open-answer runs on TextVQA-lite and DocVQA-lite. It treats 30% visual-token retention as the low-budget policy, then asks whether 50%, 70%, or full-prefix inference repairs low-budget quality drops. No model outputs are generated by this script.",
        "",
        f"Definitions: a low-budget failure means full score minus 30% score is at least {DROP_EPS:.2f}. A sample is repaired by 70% if its 70% score is within {RECOVER_CLOSE_EPS:.2f} of the full-prefix score.",
        "",
        "## Repairability Summary",
        "",
        "| Task | n | Low-failure rate | Repaired by 70% / low failures | Still bad at 70% / low failures | Mean 30% drop | Mean 30->70 gain on low failures |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['task']} | {row['n']} | {row['low_failure_rate']} | "
            f"{row['repaired_by_70_rate_among_low_fail']} | {row['high_still_bad_rate_among_low_fail']} | "
            f"{row['mean_low_drop']} | {row['mean_gain_30_to_70_low_fail']} |"
        )

    lines.extend(
        [
            "",
            "## Strongest Feature Signals",
            "",
            "The table below lists the best feature signals by target and scope. Deployable features are based on the question and the low-budget answer only; evidence-audit features use annotated boxes/ECR and are not deployable unless such boxes are available at inference.",
            "",
            "| Target | Scope | Group | Feature | n | Positive rate | AUC | Direction | Spearman |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: |",
        ]
    )
    for row in top_feature_rows(feature_rows):
        lines.append(
            f"| {row['target']} | {row['scope']} | {row['feature_group']} | {row['feature']} | "
            f"{row['n']} | {row['positive_rate']} | {row['auc_best_direction']} | {row['direction']} | {row['spearman_with_target']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- High-budget inference can repair many, but not all, 30% failures; the non-repaired cases explain why a naive fallback policy cannot simply escalate every uncertain sample cheaply.",
            "- Cheap question/answer-shape features are useful diagnostics but only weak risk controls. This supports the existing negative result that hand-written adaptive rules do not dominate fixed high-retention budgets.",
            "- Annotated ECR features are evidence-audit signals rather than free deployable signals. On this repairability target they are informative but still imperfect, which reinforces that a future controller needs better online detector/evidence features instead of simply thresholding current question cues.",
            "- The audit should be cited as a repairability and signal-quality analysis, not as a solved adaptive method.",
            "",
            "## Example Rows",
            "",
            "| Case | Task | Sample | Full | 30% | 70% | Question | Full answer | 30% answer | 70% answer |",
            "| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |",
        ]
    )
    for row in example_rows[:18]:
        lines.append(
            f"| {row['case_type']} | {row['task']} | {row['sample_id']} | {row['full_score']} | "
            f"{row['score_0p30']} | {row['score_0p70']} | {clip_md(row['raw_question'])} | "
            f"{clip_md(row['full_answer'])} | {clip_md(row['answer_0p30'])} | {clip_md(row['answer_0p70'])} |"
        )
    return "\n".join(lines) + "\n"


def top_feature_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["target"], row["scope"], row["feature_group"])].append(row)
    selected = []
    for key in sorted(grouped):
        best = sorted(grouped[key], key=lambda r: parse_float(r["auc_best_direction"]), reverse=True)[:3]
        selected.extend(best)
    return selected


def paired_feature_target(rows: list[dict[str, Any]], feature: str, target: str) -> tuple[list[float], list[int]]:
    xs: list[float] = []
    ys: list[int] = []
    for row in rows:
        x = parse_float(row.get(feature, "nan"))
        if math.isnan(x):
            continue
        xs.append(x)
        ys.append(int(row[target]))
    return xs, ys


def auc(scores: list[float], labels: list[int]) -> float:
    pairs = sorted(zip(scores, labels), key=lambda p: p[0])
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return math.nan
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        rank_sum += avg_rank * sum(label for _, label in pairs[i:j])
        i = j
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def pearson(xs: list[float], ys: list[float | int]) -> float:
    if len(xs) < 2:
        return math.nan
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (float(y) - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((float(y) - my) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return math.nan
    return num / (den_x * den_y)


def spearman(xs: list[float], ys: list[float | int]) -> float:
    return pearson(ranks(xs), ranks([float(y) for y in ys]))


def ranks(values: list[float]) -> list[float]:
    order = sorted(enumerate(values), key=lambda item: item[1])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and order[j][1] == order[i][1]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for idx, _ in order[i:j]:
            out[idx] = avg
        i = j
    return out


def question_risk_score(text: str) -> int:
    low = text.lower()
    tokens = norm_tokens(low)
    score = 0
    if len(tokens) >= 10:
        score += 1
    if len(tokens) >= 18:
        score += 1
    if asks_numeric(low):
        score += 1
    if any(cue in low for cue in ("date", "year", "phone", "number", "amount", "total", "how many", "percent", "value")):
        score += 1
    if any(cue in low for cue in ("according to", "during", "between", "from", "per", "under", "which", "where")):
        score += 1
    if any(cue in low for cue in (" and ", " or ", " with ", " of the ", " in the ")):
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


def split_for_id(sample_id: str) -> str:
    digest = hashlib.md5(sample_id.encode("utf-8")).hexdigest()
    return "dev" if int(digest[:8], 16) % 2 == 0 else "test"


def norm_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def sum_int(rows: list[dict[str, Any]], key: str) -> int:
    return sum(int(row[key]) for row in rows)


def mean_float(rows: list[dict[str, Any]], key: str) -> float:
    vals = [parse_float(row[key]) for row in rows if not math.isnan(parse_float(row[key]))]
    return mean(vals) if vals else math.nan


def rate(num: int, den: int) -> float:
    return num / den if den else math.nan


def parse_float(value: Any) -> float:
    try:
        if value == "":
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def fmt(value: Any) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(x):
        return ""
    return f"{x:.3f}"


def clip_md(value: Any, max_len: int = 80) -> str:
    text = str(value).replace("|", "/").replace("\n", " ")
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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
