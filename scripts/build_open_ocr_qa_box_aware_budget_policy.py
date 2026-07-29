#!/usr/bin/env python3
"""Evaluate box-aware selective escalation policies for open OCR/DocQA.

This is a diagnostic for the adaptive-control gap in problem.md. It asks
whether evidence-risk signals available after a low-budget selector has run
(ECR, worst-region ECR, all-regions pass) can choose when to escalate from a
30% visual prefix to a larger budget.

The policies are only deployable when evidence boxes are available at inference
time from an OCR/layout pipeline. They are not box-free policies.
"""

from __future__ import annotations

import csv
import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "problem_optimization_audit" / "open_ocr_qa_box_aware_budget"
ECR_ROWS = (
    ROOT
    / "runs/problem_optimization_audit/open_ocr_qa_ecr_quality_association/ecr_quality_rows.csv"
)
LINE_CONTEXT_ROWS = (
    ROOT
    / "runs/problem_optimization_audit/docvqa_line_context_quality_association/line_context_quality_rows.csv"
)

BUDGETS = (0.30, 0.50, 0.70, 1.00)
THRESHOLDS = (0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.85, 0.90)


@dataclass(frozen=True)
class Policy:
    name: str
    family: str
    choose_budget: Callable[[dict[str, Any]], float]
    note: str


def main() -> None:
    rows = load_rows()
    policies = build_policies()
    summary_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []

    for scope, scope_rows in iter_scopes(rows):
        fixed = [p for p in policies if p.family == "fixed"]
        oracle = [p for p in policies if p.family == "oracle"]
        for policy in fixed + oracle:
            summary_rows.extend(evaluate_policy(scope_rows, policy, scope=scope, selected_by="preset"))

        candidates = [p for p in policies if p.family not in {"fixed", "oracle"}]
        for family in sorted({p.family for p in candidates}):
            fam = [p for p in candidates if p.family == family]
            scored = []
            for policy in fam:
                dev = summarize(scope_rows, policy, split="dev")
                scored.append((objective(dev), dev, policy))
            scored = [item for item in scored if not math.isnan(item[0])]
            if not scored:
                continue
            scored.sort(key=lambda item: item[0], reverse=True)
            best_obj, best_dev, best_policy = scored[0]
            selection_rows.append(
                {
                    "scope": scope,
                    "family": family,
                    "selected_policy": best_policy.name,
                    "dev_objective": fmt(best_obj),
                    "dev_score": fmt(best_dev["score"]),
                    "dev_mean_keep": fmt(best_dev["mean_keep"]),
                    "dev_delta_vs_full": fmt(best_dev["delta_vs_full"]),
                    "dev_fallback_rate_ge_0p70": fmt(best_dev["fallback_rate_ge_0p70"]),
                    "note": best_policy.note,
                }
            )
            summary_rows.extend(evaluate_policy(scope_rows, best_policy, scope=scope, selected_by="dev_best"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "box_aware_budget_summary.csv", summary_rows)
    write_csv(OUT_DIR / "box_aware_budget_selection.csv", selection_rows)
    (OUT_DIR / "box_aware_budget_report.md").write_text(
        build_report(summary_rows, selection_rows),
        encoding="utf-8",
    )
    print(f"Wrote open-QA box-aware budget audit to {OUT_DIR}")


def load_rows() -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for source, path in (("answer_or_gt_bbox", ECR_ROWS), ("docvqa_line_context", LINE_CONTEXT_ROWS)):
        for row in read_csv(path):
            task = row.get("task", "")
            if source == "docvqa_line_context":
                task = "DocVQA-lite"
            sample_id = row.get("sample_id", "")
            budget = parse_float(row.get("budget_keep_ratio", "nan"))
            if not sample_id or math.isnan(budget):
                continue
            key = (source, task, sample_id)
            rec = grouped.setdefault(
                key,
                {
                    "source": source,
                    "task": task,
                    "sample_id": sample_id,
                    "question_id": row.get("question_id", ""),
                    "raw_question": row.get("raw_question", ""),
                    "gold_answers": row.get("gold_answers", ""),
                    "full_answer": row.get("full_answer", ""),
                    "full_score": parse_float(row.get("full_score", "nan")),
                    "split": split_for_id(sample_id),
                    "budgets": {},
                },
            )
            if not rec.get("raw_question") and row.get("raw_question"):
                rec["raw_question"] = row.get("raw_question", "")
            rec["budgets"][budget] = {
                "score": parse_float(row.get("pruned_score", "nan")),
                "answer": row.get("pruned_answer", ""),
                "ECR": parse_float(row.get("ECR", "nan")),
                "worst_region_ECR": parse_float(row.get("worst_region_ECR", "nan")),
                "all_regions_ECR_ge_0p50": parse_float(row.get("all_regions_ECR_ge_0p50", "nan")),
                "box_count": parse_float(row.get("box_count", "nan")),
            }
    out = []
    for rec in grouped.values():
        if all(b in rec["budgets"] for b in (0.30, 0.50, 0.70)) and not math.isnan(rec["full_score"]):
            rec["budgets"][1.00] = {
                "score": rec["full_score"],
                "answer": rec["full_answer"],
                "ECR": 1.0,
                "worst_region_ECR": 1.0,
                "all_regions_ECR_ge_0p50": 1.0,
                "box_count": rec["budgets"][0.30].get("box_count", math.nan),
            }
            out.append(rec)
    return sorted(out, key=lambda r: (r["source"], r["task"], r["sample_id"]))


def iter_scopes(rows: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    scopes: list[tuple[str, list[dict[str, Any]]]] = []
    by_source: dict[str, list[dict[str, Any]]] = {}
    by_source_task: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_source.setdefault(row["source"], []).append(row)
        by_source_task.setdefault((row["source"], row["task"]), []).append(row)
    for source in sorted(by_source):
        scopes.append((source, by_source[source]))
    for source, task in sorted(by_source_task):
        scopes.append((f"{source}:{task}", by_source_task[(source, task)]))
    return scopes


def build_policies() -> list[Policy]:
    policies: list[Policy] = []
    for budget in BUDGETS:
        policies.append(
            Policy(
                name=f"fixed_{budget:.2f}",
                family="fixed",
                choose_budget=lambda _row, b=budget: b,
                note="fixed-budget reference",
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
    for feature in ("ECR", "worst_region_ECR", "all_regions_ECR_ge_0p50"):
        for threshold in THRESHOLDS:
            policies.append(
                Policy(
                    name=f"{feature}_lt_{tag(threshold)}_to_70",
                    family=f"box_ecr_to70:{feature}",
                    choose_budget=lambda row, f=feature, t=threshold: 0.70
                    if low_feature(row, f) < t
                    else 0.30,
                    note="box-aware: escalate 30% to 70% when low-budget evidence coverage is below threshold",
                )
            )
            policies.append(
                Policy(
                    name=f"{feature}_lt_{tag(threshold)}_to_full",
                    family=f"box_ecr_tofull:{feature}",
                    choose_budget=lambda row, f=feature, t=threshold: 1.00
                    if low_feature(row, f) < t
                    else 0.30,
                    note="box-aware: escalate 30% to full prefix when low-budget evidence coverage is below threshold",
                )
            )
    for feature in ("ECR", "worst_region_ECR"):
        for low_threshold in THRESHOLDS:
            for mid_threshold in THRESHOLDS:
                if low_threshold >= mid_threshold:
                    continue
                policies.append(
                    Policy(
                        name=f"{feature}_lt_{tag(low_threshold)}_full_lt_{tag(mid_threshold)}_70",
                        family=f"box_ecr_threeway:{feature}",
                        choose_budget=lambda row, f=feature, lo=low_threshold, mid=mid_threshold: threeway_budget(
                            row, f, lo, mid
                        ),
                        note="box-aware: full fallback for very low evidence coverage, 70% for moderate risk",
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
                        note="question-only cue baseline on the same annotated subset",
                    )
                )
    return policies


def low_feature(row: dict[str, Any], feature: str) -> float:
    value = float(row["budgets"][0.30].get(feature, math.nan))
    if math.isnan(value):
        return -1.0
    return value


def threeway_budget(row: dict[str, Any], feature: str, low_threshold: float, mid_threshold: float) -> float:
    value = low_feature(row, feature)
    if value < low_threshold:
        return 1.00
    if value < mid_threshold:
        return 0.70
    return 0.30


def cue_budget(row: dict[str, Any], cue_weight: int, mid_score: int, high_score: int) -> float:
    score = question_risk_score(row.get("raw_question", ""), cue_weight=cue_weight)
    if score >= high_score:
        return 0.70
    if score >= mid_score:
        return 0.50
    return 0.30


def oracle_budget(row: dict[str, Any]) -> float:
    best_budget = 0.30
    best_score = -1.0
    for budget in BUDGETS:
        score = float(row["budgets"][budget]["score"])
        if score > best_score or (score == best_score and budget < best_budget):
            best_budget = budget
            best_score = score
    return best_budget


def evaluate_policy(rows: list[dict[str, Any]], policy: Policy, *, scope: str, selected_by: str) -> list[dict[str, Any]]:
    out = []
    for split in ("dev", "test", "all"):
        summary = summarize(rows, policy, split=split)
        out.append(
            {
                "scope": scope,
                "split": split,
                "policy": policy.name,
                "family": policy.family,
                "selected_by": selected_by,
                "n": summary["n"],
                "score": fmt(summary["score"]),
                "mean_keep": fmt(summary["mean_keep"]),
                "delta_vs_full": fmt(summary["delta_vs_full"]),
                "fallback_rate_ge_0p50": fmt(summary["fallback_rate_ge_0p50"]),
                "fallback_rate_ge_0p70": fmt(summary["fallback_rate_ge_0p70"]),
                "full_fallback_rate": fmt(summary["full_fallback_rate"]),
                "objective": fmt(objective(summary)),
                "note": policy.note,
            }
        )
    return out


def summarize(rows: list[dict[str, Any]], policy: Policy, *, split: str) -> dict[str, Any]:
    subset = [row for row in rows if split == "all" or row["split"] == split]
    scores = []
    full_scores = []
    keeps = []
    for row in subset:
        budget = policy.choose_budget(row)
        scores.append(float(row["budgets"][budget]["score"]))
        full_scores.append(float(row["full_score"]))
        keeps.append(budget)
    return {
        "n": len(subset),
        "score": mean(scores) if scores else math.nan,
        "mean_keep": mean(keeps) if keeps else math.nan,
        "delta_vs_full": mean([s - f for s, f in zip(scores, full_scores)]) if scores else math.nan,
        "fallback_rate_ge_0p50": mean([k >= 0.50 for k in keeps]) if keeps else math.nan,
        "fallback_rate_ge_0p70": mean([k >= 0.70 for k in keeps]) if keeps else math.nan,
        "full_fallback_rate": mean([k >= 1.0 for k in keeps]) if keeps else math.nan,
    }


def objective(summary: dict[str, Any]) -> float:
    if summary["n"] == 0 or math.isnan(float(summary["score"])):
        return math.nan
    return float(summary["score"]) - 0.10 * float(summary["mean_keep"])


def build_report(summary_rows: list[dict[str, Any]], selection_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Open OCR QA Box-Aware Budget Audit",
        "",
        "This diagnostic evaluates selective escalation policies on annotated open-QA stress rows. It uses low-budget ECR features computed from cached pruning traces and external evidence boxes. These policies are only deployable when OCR/layout evidence boxes are available at inference time.",
        "",
        "## Dev-Selected Policies",
        "",
        "| Scope | Family | Selected policy | Dev score | Dev keep | Dev delta vs full | Note |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in selection_rows:
        lines.append(
            f"| {row['scope']} | {row['family']} | {row['selected_policy']} | "
            f"{row['dev_score']} | {row['dev_mean_keep']} | {row['dev_delta_vs_full']} | {row['note']} |"
        )
    lines.extend(
        [
            "",
            "## Test Summary",
            "",
            "| Scope | Policy | Family | Score | Keep | Delta vs full | Fallback >=0.70 | Full fallback | Note |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in summary_rows:
        if row["split"] != "test":
            continue
        if row["family"] == "fixed" and row["policy"] not in {"fixed_0.30", "fixed_0.50", "fixed_0.70", "fixed_1.00"}:
            continue
        if row["family"] != "fixed" and row["selected_by"] != "dev_best" and row["family"] != "oracle":
            continue
        lines.append(
            f"| {row['scope']} | {row['policy']} | {row['family']} | {row['score']} | "
            f"{row['mean_keep']} | {row['delta_vs_full']} | {row['fallback_rate_ge_0p70']} | "
            f"{row['full_fallback_rate']} | {row['note']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Box-aware evidence-risk signals are stronger than question-only cues when reliable boxes exist, but they are not box-free and should not be mixed with the main training-free selector claim.",
            "- The useful rows are the 30%->70% escalation policies: they test whether low-budget evidence coverage can recover quality without simply falling back to the full prefix.",
            "- Full-fallback policies are diagnostic only. When dev selection drives mean keep close to 1.0, the result shows that the signal detects severe evidence risk but does not yet provide fine-grained budget control.",
            "- The relevant comparison is against fixed 70% and the oracle row on the same annotated subset, not against the full 500-sample open-QA curves.",
            "- If a box-aware policy improves quality/keep trade-offs, it supports a future detector-assisted risk controller. If it still trails fixed budgets, it shows that ECR alone is not enough for adaptive control.",
        ]
    )
    return "\n".join(lines) + "\n"


def question_risk_score(text: str, *, cue_weight: int) -> int:
    low = text.lower()
    tokens = norm_tokens(low)
    score = 0
    if len(tokens) >= 10:
        score += 1
    if len(tokens) >= 18:
        score += 1
    if asks_numeric(low):
        score += cue_weight
    if any(cue in low for cue in ("date", "year", "phone", "number", "amount", "total", "how many", "percent", "value")):
        score += cue_weight
    if any(cue in low for cue in ("according to", "during", "between", "from", "per", "under", "which", "where")):
        score += 1
    if any(cue in low for cue in (" and ", " or ", " with ", " of the ", " in the ")):
        score += 1
    return score


def asks_numeric(text: str) -> bool:
    return bool(re.search(r"\b(how many|number|date|year|amount|total|percent|value|phone|per 1000)\b", text))


def norm_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def split_for_id(sample_id: str) -> str:
    digest = hashlib.md5(sample_id.encode("utf-8")).hexdigest()
    return "dev" if int(digest[:8], 16) % 2 == 0 else "test"


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


def tag(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


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
