#!/usr/bin/env python3
"""Audit pre-generation risk signals for open OCR/DocQA pruning.

This audit separates features available before LLM prefill from features that
require a low-budget generation. It tests whether selector geometry and
cross-budget mask stability can predict low-budget failure or answer
instability before spending extra decoding passes.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from build_open_ocr_qa_answer_stability_cascade import norm_answer
from build_open_ocr_qa_learned_risk_policy import ROOT, load_rows


OUT_DIR = ROOT / "runs" / "problem_optimization_audit" / "open_ocr_qa_pregen_risk_signal"

QUESTION_FEATURES = (
    "question_len",
    "question_chars",
    "question_risk_score",
    "target_text_token_count",
    "asks_numeric",
    "asks_date",
    "asks_layout",
    "question_multi_constraint",
)
MASK30_FEATURES = (
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
)
MASK_STABILITY_FEATURES = MASK30_FEATURES + (
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
LOW_ANSWER_FEATURES = (
    "low_answer_len",
    "low_answer_chars",
    "low_answer_empty",
    "low_answer_has_digit",
    "low_answer_repetition",
    "numeric_question_digit_missing",
    "low_answer_many_numbers",
    "low_answer_punctuation",
)

FEATURE_GROUPS = {
    "question_only": QUESTION_FEATURES,
    "mask30_only": MASK30_FEATURES,
    "mask_stability_pregen": MASK_STABILITY_FEATURES,
    "question_mask_pregen": QUESTION_FEATURES + MASK_STABILITY_FEATURES,
    "question_mask_lowanswer": QUESTION_FEATURES + MASK_STABILITY_FEATURES + LOW_ANSWER_FEATURES,
}


@dataclass(frozen=True)
class LinearProbe:
    task: str
    target: str
    feature_group: str
    features: tuple[str, ...]
    means: dict[str, float]
    scales: dict[str, float]
    weights: dict[str, float]
    bias: float


TargetFn = Callable[[dict[str, Any]], float | None]


def main() -> None:
    rows_by_task = load_rows()
    model_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []

    targets: dict[str, TargetFn] = {
        "low_failure_ge0p25": low_failure_label,
        "unsafe30_gt0p10": unsafe30_label,
        "answer_disagree30_50": answer_disagree_label,
        "repair70_among_low_failure": repair70_among_low_failure_label,
    }

    for task in sorted(rows_by_task):
        task_rows = rows_by_task[task]
        for target, target_fn in targets.items():
            for feature_group, features in FEATURE_GROUPS.items():
                probe = train_probe(task, target, feature_group, features, task_rows, target_fn)
                for split in ("dev", "test", "all"):
                    model_rows.append(evaluate_probe(probe, task_rows, target_fn, split))
                if target == "low_failure_ge0p25" and feature_group in {
                    "question_only",
                    "mask30_only",
                    "mask_stability_pregen",
                    "question_mask_pregen",
                    "question_mask_lowanswer",
                }:
                    policy_rows.append(select_and_eval_30_to_70_policy(probe, task_rows))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "pregen_risk_signal_model_summary.csv", model_rows)
    write_csv(OUT_DIR / "pregen_risk_signal_policy_summary.csv", policy_rows)
    (OUT_DIR / "pregen_risk_signal_report.md").write_text(
        build_report(model_rows, policy_rows),
        encoding="utf-8",
    )
    print(f"Wrote pre-generation risk signal audit to {OUT_DIR}")


def low_failure_label(row: dict[str, Any]) -> float:
    return float(float(row["full_score"]) - score(row, 0.30) >= 0.25)


def unsafe30_label(row: dict[str, Any]) -> float:
    return float(float(row["full_score"]) - score(row, 0.30) > 0.10)


def answer_disagree_label(row: dict[str, Any]) -> float:
    a30 = norm_answer(row["budgets"][0.30]["answer"])
    a50 = norm_answer(row["budgets"][0.50]["answer"])
    return float(a30 != a50)


def repair70_among_low_failure_label(row: dict[str, Any]) -> float | None:
    if not low_failure_label(row):
        return None
    return float(float(row["full_score"]) - score(row, 0.70) <= 0.10)


def score(row: dict[str, Any], budget: float) -> float:
    return float(row["budgets"][budget]["score"])


def train_probe(
    task: str,
    target: str,
    feature_group: str,
    features: tuple[str, ...],
    rows: list[dict[str, Any]],
    target_fn: TargetFn,
) -> LinearProbe:
    train_rows = [row for row in rows if row["split"] == "dev" and target_fn(row) is not None]
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    for feature in features:
        values = [float(row.get(feature, 0.0)) for row in train_rows]
        mu = mean(values) if values else 0.0
        var = mean((value - mu) ** 2 for value in values) if values else 0.0
        means[feature] = mu
        scales[feature] = math.sqrt(var) if var > 1e-12 else 1.0

    labels = [float(target_fn(row) or 0.0) for row in train_rows]
    base = mean(labels) if labels else 0.5
    weights = {feature: 0.0 for feature in features}
    bias = logit(base * 0.98 + 0.01)
    lr = 0.08
    l2 = 0.02
    for _ in range(500):
        grad_w = {feature: 0.0 for feature in features}
        grad_b = 0.0
        for row in train_rows:
            label = float(target_fn(row) or 0.0)
            xs = standardized(row, features, means, scales)
            pred = sigmoid(bias + sum(weights[feature] * xs[feature] for feature in features))
            err = pred - label
            grad_b += err
            for feature in features:
                grad_w[feature] += err * xs[feature]
        denom = max(1, len(train_rows))
        bias -= lr * grad_b / denom
        for feature in features:
            weights[feature] -= lr * ((grad_w[feature] / denom) + l2 * weights[feature])

    return LinearProbe(task, target, feature_group, features, means, scales, weights, bias)


def evaluate_probe(
    probe: LinearProbe,
    rows: list[dict[str, Any]],
    target_fn: TargetFn,
    split: str,
) -> dict[str, Any]:
    scoped = [row for row in rows if (split == "all" or row["split"] == split) and target_fn(row) is not None]
    labels = [float(target_fn(row) or 0.0) for row in scoped]
    probs = [predict(row, probe) for row in scoped]
    return {
        "task": probe.task,
        "target": probe.target,
        "feature_group": probe.feature_group,
        "split": split,
        "n": len(scoped),
        "positive_rate": fmt(mean(labels) if labels else math.nan),
        "auc": fmt(auc(probs, labels)),
        "brier": fmt(mean((prob - label) ** 2 for prob, label in zip(probs, labels)) if labels else math.nan),
        "top20_positive_rate": fmt(tail_rate(probs, labels, top=True, fraction=0.20)),
        "bottom20_positive_rate": fmt(tail_rate(probs, labels, top=False, fraction=0.20)),
        "top_features": top_features(probe, positive=True),
        "negative_features": top_features(probe, positive=False),
        "note": feature_group_note(probe.feature_group),
    }


def select_and_eval_30_to_70_policy(probe: LinearProbe, rows: list[dict[str, Any]]) -> dict[str, Any]:
    dev = [row for row in rows if row["split"] == "dev"]
    thresholds = quantile_grid([predict(row, probe) for row in dev])
    scored = []
    for threshold in thresholds:
        dev_summary = policy_summary(rows, split="dev", probe=probe, threshold=threshold)
        scored.append((policy_objective(dev_summary), threshold, dev_summary))
    scored.sort(key=lambda item: (item[0], item[2]["score"], -item[2]["mean_keep"]), reverse=True)
    _, threshold, dev_summary = scored[0]
    test_summary = policy_summary(rows, split="test", probe=probe, threshold=threshold)
    fixed30 = fixed_summary(rows, "test", 0.30)
    fixed70 = fixed_summary(rows, "test", 0.70)
    return {
        "task": probe.task,
        "feature_group": probe.feature_group,
        "selected_threshold": fmt(threshold),
        "dev_score": fmt(dev_summary["score"]),
        "dev_mean_keep": fmt(dev_summary["mean_keep"]),
        "dev_fallback70_rate": fmt(dev_summary["fallback70_rate"]),
        "test_score": fmt(test_summary["score"]),
        "test_mean_keep": fmt(test_summary["mean_keep"]),
        "test_fallback70_rate": fmt(test_summary["fallback70_rate"]),
        "test_delta_vs_fixed30": fmt(test_summary["score"] - fixed30["score"]),
        "test_delta_vs_fixed70": fmt(test_summary["score"] - fixed70["score"]),
        "fixed30_test_score": fmt(fixed30["score"]),
        "fixed70_test_score": fmt(fixed70["score"]),
        "note": feature_group_note(probe.feature_group),
    }


def policy_summary(rows: list[dict[str, Any]], *, split: str, probe: LinearProbe, threshold: float) -> dict[str, float]:
    scoped = [row for row in rows if split == "all" or row["split"] == split]
    chosen = [0.70 if predict(row, probe) >= threshold else 0.30 for row in scoped]
    scores = [score(row, budget) for row, budget in zip(scoped, chosen)]
    keeps = [float(row["budgets"][budget]["effective_keep"]) for row, budget in zip(scoped, chosen)]
    return {
        "score": mean(scores) if scores else math.nan,
        "mean_keep": mean(keeps) if keeps else math.nan,
        "fallback70_rate": mean(float(budget >= 0.70) for budget in chosen) if chosen else math.nan,
    }


def fixed_summary(rows: list[dict[str, Any]], split: str, budget: float) -> dict[str, float]:
    scoped = [row for row in rows if split == "all" or row["split"] == split]
    return {
        "score": mean(score(row, budget) for row in scoped),
        "mean_keep": mean(float(row["budgets"][budget]["effective_keep"]) for row in scoped),
    }


def policy_objective(summary: dict[str, float]) -> float:
    return float(summary["score"]) - 0.10 * float(summary["mean_keep"])


def standardized(
    row: dict[str, Any],
    features: tuple[str, ...],
    means: dict[str, float],
    scales: dict[str, float],
) -> dict[str, float]:
    return {feature: (float(row.get(feature, 0.0)) - means[feature]) / scales[feature] for feature in features}


def predict(row: dict[str, Any], probe: LinearProbe) -> float:
    xs = standardized(row, probe.features, probe.means, probe.scales)
    return sigmoid(probe.bias + sum(probe.weights[feature] * xs[feature] for feature in probe.features))


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
    pos = [score for score, label in zip(scores, labels) if label >= 0.5]
    neg = [score for score, label in zip(scores, labels) if label < 0.5]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for p_score in pos:
        for n_score in neg:
            if p_score > n_score:
                wins += 1.0
            elif p_score == n_score:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def tail_rate(scores: list[float], labels: list[float], *, top: bool, fraction: float) -> float:
    if not labels:
        return math.nan
    pairs = sorted(zip(scores, labels), key=lambda item: item[0], reverse=top)
    k = max(1, int(round(len(pairs) * fraction)))
    return mean(label for _score, label in pairs[:k])


def quantile_grid(values: list[float]) -> list[float]:
    if not values:
        return [0.5]
    values = sorted(values)
    quantiles = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
    return sorted({values[min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))] for q in quantiles})


def top_features(probe: LinearProbe, *, positive: bool) -> str:
    items = sorted(probe.weights.items(), key=lambda item: item[1], reverse=positive)
    return "; ".join(f"{name}:{weight:.3f}" for name, weight in items[:5])


def feature_group_note(group: str) -> str:
    return {
        "question_only": "question metadata only; available before image scoring",
        "mask30_only": "30% selector-mask geometry only; available before LLM prefill",
        "mask_stability_pregen": "30/50/70 selector-mask geometry and overlap; available before LLM prefill after scoring",
        "question_mask_pregen": "question plus selector-mask features; no generated answer",
        "question_mask_lowanswer": "adds 30% answer-shape features; requires a low-budget generation",
    }.get(group, "")


def build_report(model_rows: list[dict[str, Any]], policy_rows: list[dict[str, Any]]) -> str:
    key_model_rows = [
        row
        for row in model_rows
        if row["split"] == "test"
        and row["target"] in {"low_failure_ge0p25", "answer_disagree30_50", "repair70_among_low_failure"}
        and row["feature_group"] in {"question_only", "mask_stability_pregen", "question_mask_pregen", "question_mask_lowanswer"}
    ]
    lines = [
        "# Open OCR QA Pre-Generation Risk Signal Audit",
        "",
        "This audit tests whether selector-side signals available before LLM prefill can predict low-budget risk. The low-answer feature group is included as a deployability-adjacent upper comparison because it requires a 30% generation.",
        "",
        "## Test Risk-Signal Summary",
        "",
        table_md(
            key_model_rows,
            [
                "task",
                "target",
                "feature_group",
                "n",
                "positive_rate",
                "auc",
                "top20_positive_rate",
                "bottom20_positive_rate",
                "note",
            ],
        ),
        "",
        "## Dev-Selected 30% to 70% Policy",
        "",
        table_md(
            policy_rows,
            [
                "task",
                "feature_group",
                "selected_threshold",
                "test_score",
                "test_mean_keep",
                "test_fallback70_rate",
                "test_delta_vs_fixed30",
                "test_delta_vs_fixed70",
                "note",
            ],
        ),
        "",
        "## Interpretation",
        "",
        "If pre-generation selector-mask features approach the low-answer feature group, they are promising for a reusable controller. If they lag fixed 70% quality, the result supports the current boundary: stability is predictable only weakly before generation and should remain a diagnostic rather than a solved adaptive method.",
    ]
    return "\n".join(lines) + "\n"


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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
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
