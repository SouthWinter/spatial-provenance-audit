#!/usr/bin/env python3
"""Train split-safe deployable risk policies for open OCR/DocQA budgets.

The policy may use question text, selector metadata, and the 30% low-budget
answer/selector-mask shape. The selector-mask features may use masks for
candidate budgets because these are available before the LLM prefill once the
visual-token scores are computed. The policy may not use gold answers,
full-prefix answers, or evaluation scores at test time. Dev labels are derived
from the oracle risk-coverage frontier, making this a calibrated diagnostic
rather than an oracle method.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "problem_optimization_audit" / "open_ocr_qa_learned_risk_policy"
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
TRACE_RUNS = {
    "TextVQA-lite": {
        0.30: ROOT / "runs/open_ocr_qa/qwen3_8b_textvqa_lite_target_grid0p30_full500/prune_traces.jsonl",
        0.50: ROOT / "runs/open_ocr_qa/qwen3_8b_textvqa_lite_target_grid0p50_full500/prune_traces.jsonl",
        0.70: ROOT / "runs/open_ocr_qa/qwen3_8b_textvqa_lite_target_grid0p70_full500/prune_traces.jsonl",
    },
    "DocVQA-lite": {
        0.30: ROOT / "runs/open_ocr_qa/qwen3_8b_docvqa_lite_target_grid0p30_full500/prune_traces.jsonl",
        0.50: ROOT / "runs/open_ocr_qa/qwen3_8b_docvqa_lite_target_grid0p50_full500/prune_traces.jsonl",
        0.70: ROOT / "runs/open_ocr_qa/qwen3_8b_docvqa_lite_target_grid0p70_full500/prune_traces.jsonl",
    },
}
BUDGETS = (0.30, 0.50, 0.70, 1.00)
FEATURES = (
    "question_len",
    "question_chars",
    "question_risk_score",
    "target_text_token_count",
    "asks_numeric",
    "asks_date",
    "asks_layout",
    "question_multi_constraint",
    "low_answer_len",
    "low_answer_chars",
    "low_answer_empty",
    "low_answer_has_digit",
    "low_answer_repetition",
    "numeric_question_digit_missing",
    "low_answer_many_numbers",
    "low_answer_punctuation",
    "mask_full_visual_tokens",
    "mask_kept_visual_tokens",
    "mask_keep_ratio",
    "mask_mean_index",
    "mask_std_index",
    "mask_span",
    "mask_gap_mean",
    "mask_gap_max",
    "mask_run_count",
    "mask_run_count_norm",
    "mask_decile_entropy",
    "mask_decile_coverage",
    "mask_edge_fraction",
    "mask_center_fraction",
    "mask_largest_decile_fraction",
    "mask50_std_index",
    "mask50_decile_entropy",
    "mask50_edge_fraction",
    "mask50_center_fraction",
    "mask70_std_index",
    "mask70_decile_entropy",
    "mask70_edge_fraction",
    "mask70_center_fraction",
    "mask_jaccard_30_50",
    "mask_jaccard_50_70",
    "mask_jaccard_30_70",
    "mask_new_fraction_30_50",
    "mask_new_fraction_50_70",
    "mask_entropy_gain_30_50",
    "mask_entropy_gain_50_70",
    "mask_edge_delta_30_70",
    "mask_center_delta_30_70",
)


@dataclass(frozen=True)
class LogisticModel:
    label: str
    feature_names: tuple[str, ...]
    means: dict[str, float]
    scales: dict[str, float]
    weights: dict[str, float]
    bias: float
    dev_auc: float


def main() -> None:
    rows = load_rows()
    summary_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for task in sorted(rows):
        task_rows = rows[task]
        add_fixed_and_oracle_rows(summary_rows, task, task_rows)
        models = train_models(task_rows)
        for model in models.values():
            model_rows.append(model_to_row(task, model))
        scored_rows = add_model_scores(task_rows, models)
        policies = build_policy_candidates(scored_rows)
        selected = select_policies(task, scored_rows, policies)
        selection_rows.extend(selected["selection_rows"])
        summary_rows.extend(selected["summary_rows"])
        prediction_rows.extend(build_prediction_rows(task, scored_rows, selected["best_policies"]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "learned_risk_policy_summary.csv", summary_rows)
    write_csv(OUT_DIR / "learned_risk_policy_selection.csv", selection_rows)
    write_csv(OUT_DIR / "learned_risk_model_weights.csv", model_rows)
    write_csv(OUT_DIR / "learned_risk_policy_predictions.csv", prediction_rows)
    (OUT_DIR / "learned_risk_policy_report.md").write_text(
        build_report(summary_rows, selection_rows, model_rows),
        encoding="utf-8",
    )
    print(f"Wrote open OCR QA learned-risk policy audit to {OUT_DIR}")


def load_rows() -> dict[str, list[dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for task, by_budget in RUNS.items():
        traces = {
            budget: {row["sample_id"]: row for row in read_jsonl(path)}
            for budget, path in TRACE_RUNS[task].items()
        }
        task_rows: dict[str, dict[str, Any]] = {}
        for budget, path in by_budget.items():
            for row in read_jsonl(path):
                sid = row["sample_id"]
                record = task_rows.setdefault(
                    sid,
                    {
                        "task": task,
                        "sample_id": sid,
                        "question_id": row.get("question_id", ""),
                        "split": split_for_id(sid),
                        "raw_question": row.get("raw_question") or row.get("question", ""),
                        "metric": row.get("metric", ""),
                        "full_score": float(row.get("full_score", 0.0)),
                        "full_exact": float(row.get("full_exact", 0.0)),
                        "full_anls": float(row.get("full_anls", 0.0)),
                        "full_textvqa_accuracy": float(row.get("full_textvqa_accuracy", 0.0)),
                        "target_text_token_count": int(row.get("target_text_token_count", 0) or 0),
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
                "answer": "",
                "score": record["full_score"],
                "exact": record["full_exact"],
                "anls": record["full_anls"],
                "textvqa_accuracy": record["full_textvqa_accuracy"],
                "effective_keep": 1.0,
            }
            trace30 = traces[0.30].get(record["sample_id"], {})
            trace50 = traces[0.50].get(record["sample_id"], {})
            trace70 = traces[0.70].get(record["sample_id"], {})
            record.update(mask_features(trace30, prefix="mask_"))
            record.update(selected_mask_features(trace50, prefix="mask50_"))
            record.update(selected_mask_features(trace70, prefix="mask70_"))
            record.update(cross_budget_mask_features(trace30, trace50, trace70))
            record.update(deployable_features(record))
            oracle = cheapest_oracle_budget(record, tolerance=0.0)
            record["oracle_budget_0tol"] = oracle
            record["label_escalate"] = float(oracle > 0.30)
            record["label_high"] = float(oracle >= 0.70)
            record["label_full"] = float(oracle >= 1.00)
        out[task] = list(task_rows.values())
    return out


def mask_features(trace: dict[str, Any], *, prefix: str) -> dict[str, float]:
    full = int(trace.get("full_visual_tokens", 0) or 0)
    kept = [int(x) for x in trace.get("kept_indices", []) if isinstance(x, int) or str(x).isdigit()]
    kept = sorted(x for x in kept if full <= 0 or 0 <= x < full)
    if full <= 0 or not kept:
        return {
            f"{prefix}full_visual_tokens": float(full),
            f"{prefix}kept_visual_tokens": 0.0,
            f"{prefix}keep_ratio": 0.0,
            f"{prefix}mean_index": 0.0,
            f"{prefix}std_index": 0.0,
            f"{prefix}span": 0.0,
            f"{prefix}gap_mean": 0.0,
            f"{prefix}gap_max": 0.0,
            f"{prefix}run_count": 0.0,
            f"{prefix}run_count_norm": 0.0,
            f"{prefix}decile_entropy": 0.0,
            f"{prefix}decile_coverage": 0.0,
            f"{prefix}edge_fraction": 0.0,
            f"{prefix}center_fraction": 0.0,
            f"{prefix}largest_decile_fraction": 0.0,
        }

    denom = max(1, full - 1)
    norm_indices = [idx / denom for idx in kept]
    mu = mean(norm_indices)
    var = mean([(idx - mu) ** 2 for idx in norm_indices])
    gaps = [b - a for a, b in zip(kept, kept[1:])]
    run_count = 1 + sum(1 for gap in gaps if gap > 1) if kept else 0
    decile_counts = [0] * 10
    for idx in kept:
        decile_counts[min(9, int(idx / full * 10))] += 1
    probs = [count / len(kept) for count in decile_counts if count > 0]
    entropy = -sum(p * math.log(p + 1e-12) for p in probs) / math.log(10)
    return {
        f"{prefix}full_visual_tokens": float(full),
        f"{prefix}kept_visual_tokens": float(len(kept)),
        f"{prefix}keep_ratio": float(len(kept) / full),
        f"{prefix}mean_index": float(mu),
        f"{prefix}std_index": float(math.sqrt(var)),
        f"{prefix}span": float((kept[-1] - kept[0]) / denom),
        f"{prefix}gap_mean": float((mean(gaps) / denom) if gaps else 0.0),
        f"{prefix}gap_max": float((max(gaps) / denom) if gaps else 0.0),
        f"{prefix}run_count": float(run_count),
        f"{prefix}run_count_norm": float(run_count / max(1, len(kept))),
        f"{prefix}decile_entropy": float(entropy),
        f"{prefix}decile_coverage": float(sum(1 for count in decile_counts if count > 0) / 10.0),
        f"{prefix}edge_fraction": float(sum(1 for idx in norm_indices if idx <= 0.10 or idx >= 0.90) / len(norm_indices)),
        f"{prefix}center_fraction": float(sum(1 for idx in norm_indices if 0.40 <= idx <= 0.60) / len(norm_indices)),
        f"{prefix}largest_decile_fraction": float(max(decile_counts) / len(kept)),
    }


def selected_mask_features(trace: dict[str, Any], *, prefix: str) -> dict[str, float]:
    all_features = mask_features(trace, prefix=prefix)
    keep = {
        "std_index",
        "decile_entropy",
        "edge_fraction",
        "center_fraction",
    }
    return {f"{prefix}{name}": all_features[f"{prefix}{name}"] for name in keep}


def cross_budget_mask_features(
    trace30: dict[str, Any],
    trace50: dict[str, Any],
    trace70: dict[str, Any],
) -> dict[str, float]:
    kept30 = kept_set(trace30)
    kept50 = kept_set(trace50)
    kept70 = kept_set(trace70)
    m30 = mask_features(trace30, prefix="mask_")
    m50 = mask_features(trace50, prefix="tmp50_")
    m70 = mask_features(trace70, prefix="tmp70_")
    return {
        "mask_jaccard_30_50": jaccard(kept30, kept50),
        "mask_jaccard_50_70": jaccard(kept50, kept70),
        "mask_jaccard_30_70": jaccard(kept30, kept70),
        "mask_new_fraction_30_50": new_fraction(kept30, kept50),
        "mask_new_fraction_50_70": new_fraction(kept50, kept70),
        "mask_entropy_gain_30_50": m50["tmp50_decile_entropy"] - m30["mask_decile_entropy"],
        "mask_entropy_gain_50_70": m70["tmp70_decile_entropy"] - m50["tmp50_decile_entropy"],
        "mask_edge_delta_30_70": m70["tmp70_edge_fraction"] - m30["mask_edge_fraction"],
        "mask_center_delta_30_70": m70["tmp70_center_fraction"] - m30["mask_center_fraction"],
    }


def kept_set(trace: dict[str, Any]) -> set[int]:
    return {int(x) for x in trace.get("kept_indices", []) if isinstance(x, int) or str(x).isdigit()}


def jaccard(left: set[int], right: set[int]) -> float:
    if not left and not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def new_fraction(smaller: set[int], larger: set[int]) -> float:
    if not larger:
        return 0.0
    return len(larger - smaller) / len(larger)


def deployable_features(row: dict[str, Any]) -> dict[str, float]:
    question = str(row["raw_question"])
    low_answer = str(row["budgets"][0.30]["answer"])
    q_tokens = norm_tokens(question)
    low_tokens = norm_tokens(low_answer)
    q_lower = question.lower()
    numeric = asks_numeric(q_lower)
    return {
        "question_len": float(len(q_tokens)),
        "question_chars": float(len(question)),
        "question_risk_score": float(question_risk_score(q_lower)),
        "target_text_token_count": float(row.get("target_text_token_count", 0) or 0),
        "asks_numeric": float(numeric),
        "asks_date": float(bool(re.search(r"\b(date|year|month|day|when|time)\b", q_lower))),
        "asks_layout": float(bool(re.search(r"\b(row|column|table|under|above|below|left|right|total|amount|value|per)\b", q_lower))),
        "question_multi_constraint": float(any(cue in q_lower for cue in (" and ", " or ", " during ", " between ", " per ", " of the "))),
        "low_answer_len": float(len(low_tokens)),
        "low_answer_chars": float(len(low_answer)),
        "low_answer_empty": float(not low_answer.strip()),
        "low_answer_has_digit": float(bool(re.search(r"\d", low_answer))),
        "low_answer_repetition": float(has_repetition(low_tokens)),
        "numeric_question_digit_missing": float(numeric and not re.search(r"\d", low_answer)),
        "low_answer_many_numbers": float(len(re.findall(r"\d+", low_answer)) >= 2),
        "low_answer_punctuation": float(len(re.findall(r"[^A-Za-z0-9\s]", low_answer))),
    }


def train_models(rows: list[dict[str, Any]]) -> dict[str, LogisticModel]:
    dev = [row for row in rows if row["split"] == "dev"]
    return {
        "escalate": train_logistic(dev, "label_escalate"),
        "high": train_logistic(dev, "label_high"),
        "full": train_logistic(dev, "label_full"),
    }


def train_logistic(rows: list[dict[str, Any]], label: str) -> LogisticModel:
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    for feature in FEATURES:
        values = [float(row.get(feature, 0.0)) for row in rows]
        mu = mean(values) if values else 0.0
        var = mean([(value - mu) ** 2 for value in values]) if values else 0.0
        means[feature] = mu
        scales[feature] = math.sqrt(var) if var > 1e-12 else 1.0

    weights = {feature: 0.0 for feature in FEATURES}
    bias = logit((mean(float(row[label]) for row in rows) if rows else 0.5) * 0.98 + 0.01)
    lr = 0.08
    l2 = 0.01
    for _ in range(500):
        grad_w = {feature: 0.0 for feature in FEATURES}
        grad_b = 0.0
        for row in rows:
            xs = standardized(row, means, scales)
            pred = sigmoid(bias + sum(weights[feature] * xs[feature] for feature in FEATURES))
            err = pred - float(row[label])
            grad_b += err
            for feature in FEATURES:
                grad_w[feature] += err * xs[feature]
        denom = max(1, len(rows))
        bias -= lr * grad_b / denom
        for feature in FEATURES:
            weights[feature] -= lr * ((grad_w[feature] / denom) + l2 * weights[feature])

    scores = [predict_prob(row, means, scales, weights, bias) for row in rows]
    labels = [float(row[label]) for row in rows]
    return LogisticModel(
        label=label,
        feature_names=FEATURES,
        means=means,
        scales=scales,
        weights=weights,
        bias=bias,
        dev_auc=auc(scores, labels),
    )


def add_model_scores(rows: list[dict[str, Any]], models: dict[str, LogisticModel]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        new = dict(row)
        for key, model in models.items():
            new[f"risk_{key}"] = predict_model(row, model)
        out.append(new)
    return out


def build_policy_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dev = [row for row in rows if row["split"] == "dev"]
    q_escalate = quantile_grid([row["risk_escalate"] for row in dev])
    q_high = quantile_grid([row["risk_high"] for row in dev])
    q_full = quantile_grid([row["risk_full"] for row in dev])
    candidates = []
    for t_e in q_escalate:
        candidates.append(
            {
                "policy": f"learned_escalate_to70_e{t_e:.3f}",
                "family": "learned_to70",
                "threshold_escalate": t_e,
                "threshold_high": "",
                "threshold_full": "",
                "note": "learned risk score: choose 70% when low-budget failure risk is high",
            }
        )
        for t_h in q_high:
            candidates.append(
                {
                    "policy": f"learned_threeway_e{t_e:.3f}_h{t_h:.3f}",
                    "family": "learned_threeway",
                    "threshold_escalate": t_e,
                    "threshold_high": t_h,
                    "threshold_full": "",
                    "note": "learned risk scores: choose 30/50/70%",
                }
            )
            for t_f in q_full:
                candidates.append(
                    {
                        "policy": f"learned_full_e{t_e:.3f}_h{t_h:.3f}_f{t_f:.3f}",
                        "family": "learned_fullfallback",
                        "threshold_escalate": t_e,
                        "threshold_high": t_h,
                        "threshold_full": t_f,
                        "note": "learned risk scores: choose 30/50/70/full",
                    }
                )
    return candidates


def select_policies(task: str, rows: list[dict[str, Any]], policies: list[dict[str, Any]]) -> dict[str, Any]:
    selection_rows = []
    summary_rows = []
    best_policies: dict[str, dict[str, Any]] = {}
    for family in sorted({policy["family"] for policy in policies}):
        scored = []
        for policy in policies:
            if policy["family"] != family:
                continue
            dev_summary = summarize(rows, policy, split="dev")
            scored.append((objective(dev_summary), dev_summary, policy))
        scored.sort(key=lambda item: (item[0], item[1]["score"], -item[1]["mean_keep"]), reverse=True)
        best_obj, best_dev, best_policy = scored[0]
        best_policies[family] = best_policy
        selection_rows.append(
            {
                "task": task,
                "family": family,
                "selected_policy": best_policy["policy"],
                "dev_objective": fmt(best_obj),
                "dev_score": fmt(best_dev["score"]),
                "dev_exact": fmt(best_dev["exact"]),
                "dev_mean_keep": fmt(best_dev["mean_keep"]),
                "dev_full_fallback_rate": fmt(best_dev["full_fallback_rate"]),
                "note": best_policy["note"],
            }
        )
        for split in ("dev", "test", "all"):
            summary_rows.append(summary_row(task, split, best_policy, selected_by="dev_best", rows=rows))
    return {"selection_rows": selection_rows, "summary_rows": summary_rows, "best_policies": best_policies}


def add_fixed_and_oracle_rows(summary_rows: list[dict[str, Any]], task: str, rows: list[dict[str, Any]]) -> None:
    for budget in BUDGETS:
        policy = {
            "policy": f"fixed_{budget:.2f}",
            "family": "fixed",
            "fixed_budget": budget,
            "note": "fixed budget baseline",
        }
        for split in ("dev", "test", "all"):
            summary_rows.append(summary_row(task, split, policy, selected_by="preset", rows=rows))
    oracle = {"policy": "oracle_0tol", "family": "oracle", "note": "uses gold scores; diagnostic upper bound only"}
    for split in ("dev", "test", "all"):
        summary_rows.append(summary_row(task, split, oracle, selected_by="preset", rows=rows))


def choose_budget(row: dict[str, Any], policy: dict[str, Any]) -> float:
    family = policy["family"]
    if family == "fixed":
        return float(policy["fixed_budget"])
    if family == "oracle":
        return float(row["oracle_budget_0tol"])
    if family == "learned_to70":
        return 0.70 if row["risk_escalate"] >= float(policy["threshold_escalate"]) else 0.30
    if family == "learned_threeway":
        if row["risk_high"] >= float(policy["threshold_high"]):
            return 0.70
        if row["risk_escalate"] >= float(policy["threshold_escalate"]):
            return 0.50
        return 0.30
    if family == "learned_fullfallback":
        if row["risk_full"] >= float(policy["threshold_full"]):
            return 1.00
        if row["risk_high"] >= float(policy["threshold_high"]):
            return 0.70
        if row["risk_escalate"] >= float(policy["threshold_escalate"]):
            return 0.50
        return 0.30
    raise ValueError(f"Unknown policy family: {family}")


def summarize(rows: list[dict[str, Any]], policy: dict[str, Any], *, split: str) -> dict[str, float]:
    subset = [row for row in rows if split == "all" or row["split"] == split]
    if not subset:
        return {"n": 0, "score": math.nan, "exact": math.nan, "anls": math.nan, "textvqa_accuracy": math.nan, "mean_keep": math.nan, "fallback_rate_ge_0p50": math.nan, "fallback_rate_ge_0p70": math.nan, "full_fallback_rate": math.nan}
    scores = []
    exacts = []
    anls = []
    textvqa_acc = []
    keeps = []
    for row in subset:
        budget = choose_budget(row, policy)
        item = row["budgets"][budget]
        scores.append(float(item["score"]))
        exacts.append(float(item["exact"]))
        anls.append(float(item["anls"]))
        textvqa_acc.append(float(item["textvqa_accuracy"]))
        keeps.append(float(item["effective_keep"]))
    return {
        "n": len(subset),
        "score": mean(scores),
        "exact": mean(exacts),
        "anls": mean(anls),
        "textvqa_accuracy": mean(textvqa_acc),
        "mean_keep": mean(keeps),
        "fallback_rate_ge_0p50": mean(float(k >= 0.50) for k in keeps),
        "fallback_rate_ge_0p70": mean(float(k >= 0.70) for k in keeps),
        "full_fallback_rate": mean(float(k >= 1.00) for k in keeps),
    }


def summary_row(task: str, split: str, policy: dict[str, Any], *, selected_by: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize(rows, policy, split=split)
    return {
        "task": task,
        "split": split,
        "policy": policy["policy"],
        "family": policy["family"],
        "selected_by": selected_by,
        "n": summary["n"],
        "score": fmt(summary["score"]),
        "exact": fmt(summary["exact"]),
        "anls": fmt(summary["anls"]),
        "textvqa_accuracy": fmt(summary["textvqa_accuracy"]),
        "mean_keep": fmt(summary["mean_keep"]),
        "fallback_rate_ge_0p50": fmt(summary["fallback_rate_ge_0p50"]),
        "fallback_rate_ge_0p70": fmt(summary["fallback_rate_ge_0p70"]),
        "full_fallback_rate": fmt(summary["full_fallback_rate"]),
        "objective": fmt(objective(summary)),
        "note": policy.get("note", ""),
    }


def objective(summary: dict[str, Any]) -> float:
    if math.isnan(float(summary["score"])):
        return -1e9
    return float(summary["score"]) - 0.10 * float(summary["mean_keep"])


def build_prediction_rows(task: str, rows: list[dict[str, Any]], best_policies: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for family, policy in sorted(best_policies.items()):
        for row in rows:
            budget = choose_budget(row, policy)
            out.append(
                {
                    "task": task,
                    "family": family,
                    "sample_id": row["sample_id"],
                    "split": row["split"],
                    "chosen_budget": fmt(budget),
                    "chosen_score": fmt(row["budgets"][budget]["score"]),
                    "full_score": fmt(row["full_score"]),
                    "oracle_budget_0tol": fmt(row["oracle_budget_0tol"]),
                    "risk_escalate": fmt(row["risk_escalate"]),
                    "risk_high": fmt(row["risk_high"]),
                    "risk_full": fmt(row["risk_full"]),
                }
            )
    return out


def build_report(summary_rows: list[dict[str, Any]], selection_rows: list[dict[str, Any]], model_rows: list[dict[str, Any]]) -> str:
    test_rows = [
        row
        for row in summary_rows
        if row["split"] == "test"
        and (
            row["family"] in {"learned_to70", "learned_threeway", "learned_fullfallback", "oracle"}
            or row["policy"] in {"fixed_0.30", "fixed_0.50", "fixed_0.70", "fixed_1.00"}
        )
    ]
    lines = [
        "# Open OCR QA Learned Risk Policy",
        "",
        "This audit trains split-safe linear risk scores on the dev split. Test-time features are deployable: question text, selector target-token count, 30% answer shape, and selector-mask geometry/stability across 30%, 50%, and 70% candidate budgets. Gold/full-prefix scores are used only to create dev labels and oracle diagnostic rows.",
        "",
        "## Selected Policies",
        "",
        table_md(selection_rows, ["task", "family", "selected_policy", "dev_score", "dev_mean_keep", "dev_full_fallback_rate", "note"]),
        "",
        "## Test Summary",
        "",
        table_md(test_rows, ["task", "policy", "family", "score", "exact", "mean_keep", "fallback_rate_ge_0p70", "full_fallback_rate", "note"]),
        "",
        "## Risk Model AUC",
        "",
        table_md(model_rows, ["task", "label", "dev_auc", "bias", "top_positive_features", "top_negative_features"]),
        "",
        "## Interpretation",
        "",
        "- A learned policy that beats the hand-written question/answer-shape rules would support a deployable risk-controller direction.",
        "- If it still trails fixed high-retention rows, the result is useful negative evidence: the current deployable features are not strong enough to close the oracle frontier.",
        "- These learned policies still require a 30% first pass before answer-shape features are available, while mask-stability features are available before the LLM prefill once visual-token scores are computed.",
    ]
    return "\n".join(lines) + "\n"


def model_to_row(task: str, model: LogisticModel) -> dict[str, Any]:
    ranked = sorted(model.weights.items(), key=lambda item: item[1], reverse=True)
    return {
        "task": task,
        "label": model.label,
        "dev_auc": fmt(model.dev_auc),
        "bias": fmt(model.bias),
        "top_positive_features": "; ".join(f"{k}:{v:.3f}" for k, v in ranked[:5]),
        "top_negative_features": "; ".join(f"{k}:{v:.3f}" for k, v in ranked[-5:]),
    }


def cheapest_oracle_budget(row: dict[str, Any], *, tolerance: float) -> float:
    target = max(0.0, float(row["full_score"]) - tolerance)
    for budget in BUDGETS:
        if float(row["budgets"][budget]["score"]) + 1e-12 >= target:
            return budget
    return 1.00


def standardized(row: dict[str, Any], means: dict[str, float], scales: dict[str, float]) -> dict[str, float]:
    return {feature: (float(row.get(feature, 0.0)) - means[feature]) / scales[feature] for feature in FEATURES}


def predict_model(row: dict[str, Any], model: LogisticModel) -> float:
    return predict_prob(row, model.means, model.scales, model.weights, model.bias)


def predict_prob(row: dict[str, Any], means: dict[str, float], scales: dict[str, float], weights: dict[str, float], bias: float) -> float:
    xs = standardized(row, means, scales)
    return sigmoid(bias + sum(weights[feature] * xs[feature] for feature in FEATURES))


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def logit(value: float) -> float:
    value = min(0.999, max(0.001, value))
    return math.log(value / (1.0 - value))


def auc(scores: list[float], labels: list[float]) -> float:
    pos = [(score, label) for score, label in zip(scores, labels) if label >= 0.5]
    neg = [(score, label) for score, label in zip(scores, labels) if label < 0.5]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for p_score, _ in pos:
        for n_score, _ in neg:
            if p_score > n_score:
                wins += 1.0
            elif p_score == n_score:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def quantile_grid(values: list[float]) -> list[float]:
    if not values:
        return [0.5]
    values = sorted(values)
    quantiles = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
    return sorted({values[min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))] for q in quantiles})


def question_risk_score(text: str) -> int:
    tokens = norm_tokens(text)
    score = 0
    if len(tokens) >= 10:
        score += 1
    if len(tokens) >= 18:
        score += 1
    if asks_numeric(text):
        score += 2
    if any(cue in text for cue in ("date", "year", "phone", "number", "amount", "total", "how many", "percent", "value")):
        score += 2
    if any(cue in text for cue in ("according to", "during", "between", "from", "per", "under", "which", "where")):
        score += 1
    if any(cue in text for cue in (" and ", " or ", " with ", " of the ", " in the ")):
        score += 1
    return score


def asks_numeric(text: str) -> bool:
    return bool(re.search(r"\b(how many|number|date|year|amount|total|percent|value|phone|per 1000|time)\b", text))


def has_repetition(tokens: list[str]) -> bool:
    if len(tokens) < 6:
        return False
    return len(set(tokens[:6])) <= 2


def norm_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


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
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(x):
        return ""
    return f"{x:.3f}"


if __name__ == "__main__":
    main()
