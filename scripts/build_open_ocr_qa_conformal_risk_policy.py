#!/usr/bin/env python3
"""Calibrated risk-control audit for open OCR QA pruning.

This script tests a more principled version of adaptive fallback than a raw
threshold search. It uses a split dev/test protocol: dev samples calibrate a
risk threshold for falling back from 30% to 70% visual-token retention, and test
samples only evaluate the selected rule. The risk score is the existing learned
low-budget risk estimate, so the policy is post-low-budget-answer and its
serial cost is reported explicitly.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "problem_optimization_audit" / "open_ocr_qa_conformal_risk_policy"
PREDICTIONS = ROOT / "runs" / "problem_optimization_audit" / "open_ocr_qa_learned_risk_policy" / "learned_risk_policy_predictions.csv"
RUNS = {
    "TextVQA-lite": {
        0.30: ROOT / "runs/open_ocr_qa/qwen3_8b_textvqa_lite_target_grid0p30_full500/open_ocr_qa_generation.jsonl",
        0.70: ROOT / "runs/open_ocr_qa/qwen3_8b_textvqa_lite_target_grid0p70_full500/open_ocr_qa_generation.jsonl",
    },
    "DocVQA-lite": {
        0.30: ROOT / "runs/open_ocr_qa/qwen3_8b_docvqa_lite_target_grid0p30_full500/open_ocr_qa_generation.jsonl",
        0.70: ROOT / "runs/open_ocr_qa/qwen3_8b_docvqa_lite_target_grid0p70_full500/open_ocr_qa_generation.jsonl",
    },
}
EPSILONS = (0.01, 0.03, 0.05)
CONFIDENCE_Z = 1.96


def main() -> None:
    risk = read_risk_scores()
    rows = load_rows(risk)
    candidates: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    for task in sorted(rows):
        task_candidates = build_candidates(task, rows[task])
        candidates.extend(task_candidates)
        selections.extend(select_candidates(task, task_candidates))
    decision = build_decision(selections)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "conformal_risk_policy_candidates.csv", candidates)
    write_csv(OUT_DIR / "conformal_risk_policy_selection.csv", selections)
    write_csv(OUT_DIR / "conformal_risk_policy_decision.csv", [decision])
    (OUT_DIR / "conformal_risk_policy_report.md").write_text(build_markdown(candidates, selections, decision), encoding="utf-8")
    print(f"Wrote conformal risk-policy audit to {OUT_DIR}")
    print(f"conformal_policy_status={decision['conformal_policy_status']}")


def read_risk_scores() -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    for row in read_csv(PREDICTIONS):
        key = (row["task"], row["sample_id"])
        out.setdefault(key, fnum(row.get("risk_escalate")))
    return out


def load_rows(risk: dict[tuple[str, str], float]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for task, paths in RUNS.items():
        low = {row["sample_id"]: row for row in read_jsonl(paths[0.30])}
        high = {row["sample_id"]: row for row in read_jsonl(paths[0.70])}
        task_rows: list[dict[str, Any]] = []
        for sid, low_row in low.items():
            if sid not in high:
                continue
            high_row = high[sid]
            risk_score = risk.get((task, sid))
            if risk_score is None:
                continue
            score30 = fnum(low_row.get("pruned_score"))
            score70 = fnum(high_row.get("pruned_score"))
            task_rows.append(
                {
                    "task": task,
                    "sample_id": sid,
                    "split": split_for_id(sid),
                    "score30": score30,
                    "score70": score70,
                    "full_score": fnum(low_row.get("full_score")),
                    "risk_score": risk_score,
                    "loss30_vs70": max(0.0, score70 - score30),
                }
            )
        out[task] = task_rows
    return out


def build_candidates(task: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dev = [row for row in rows if row["split"] == "dev"]
    test = [row for row in rows if row["split"] == "test"]
    thresholds = threshold_grid([row["risk_score"] for row in dev])
    out: list[dict[str, Any]] = []
    for threshold in thresholds:
        dev_metrics = evaluate_policy(dev, threshold)
        test_metrics = evaluate_policy(test, threshold)
        for epsilon in EPSILONS:
            out.append(
                {
                    "task": task,
                    "epsilon": fmt(epsilon),
                    "threshold": fmt(threshold),
                    "dev_n": len(dev),
                    "test_n": len(test),
                    "dev_score": fmt(dev_metrics["score"]),
                    "dev_fixed70_score": fmt(dev_metrics["fixed70_score"]),
                    "dev_delta_vs_fixed70": fmt(dev_metrics["score"] - dev_metrics["fixed70_score"]),
                    "dev_mean_loss_vs_fixed70": fmt(dev_metrics["mean_loss"]),
                    "dev_loss_upper95": fmt(dev_metrics["loss_upper95"]),
                    "dev_fallback_rate": fmt(dev_metrics["fallback_rate"]),
                    "dev_selected_keep": fmt(dev_metrics["selected_keep"]),
                    "dev_serial_cost": fmt(dev_metrics["serial_cost"]),
                    "dev_passes_risk_bound": int(dev_metrics["loss_upper95"] <= epsilon),
                    "dev_passes_lower_serial_cost": int(dev_metrics["serial_cost"] < 0.70),
                    "test_score": fmt(test_metrics["score"]),
                    "test_fixed70_score": fmt(test_metrics["fixed70_score"]),
                    "test_delta_vs_fixed70": fmt(test_metrics["score"] - test_metrics["fixed70_score"]),
                    "test_mean_loss_vs_fixed70": fmt(test_metrics["mean_loss"]),
                    "test_fallback_rate": fmt(test_metrics["fallback_rate"]),
                    "test_selected_keep": fmt(test_metrics["selected_keep"]),
                    "test_serial_cost": fmt(test_metrics["serial_cost"]),
                    "test_passes_quality_gate": int(test_metrics["score"] >= test_metrics["fixed70_score"] - epsilon),
                    "test_passes_lower_serial_cost": int(test_metrics["serial_cost"] < 0.70),
                    "contract": "post_low_budget_answer_30_then_optional_70",
                }
            )
    return out


def evaluate_policy(rows: list[dict[str, Any]], threshold: float) -> dict[str, float]:
    if not rows:
        return {
            "score": 0.0,
            "fixed70_score": 0.0,
            "mean_loss": 0.0,
            "loss_upper95": 0.0,
            "fallback_rate": 0.0,
            "selected_keep": 0.0,
            "serial_cost": 0.0,
        }
    chosen_scores = []
    fixed70_scores = []
    losses = []
    fallbacks = []
    for row in rows:
        fallback = row["risk_score"] >= threshold
        chosen = row["score70"] if fallback else row["score30"]
        chosen_scores.append(chosen)
        fixed70_scores.append(row["score70"])
        losses.append(max(0.0, row["score70"] - chosen))
        fallbacks.append(float(fallback))
    fallback_rate = mean(fallbacks)
    selected_keep = mean([0.70 if f else 0.30 for f in fallbacks])
    # Serial two-pass cost: every sample pays 30%; fallback samples then pay an
    # additional 70% pass because this risk score uses the low-budget answer.
    serial_cost = 0.30 + 0.70 * fallback_rate
    return {
        "score": mean(chosen_scores),
        "fixed70_score": mean(fixed70_scores),
        "mean_loss": mean(losses),
        "loss_upper95": mean_upper95(losses),
        "fallback_rate": fallback_rate,
        "selected_keep": selected_keep,
        "serial_cost": serial_cost,
    }


def select_candidates(task: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for epsilon in EPSILONS:
        eps = fmt(epsilon)
        task_eps = [row for row in candidates if row["task"] == task and row["epsilon"] == eps]
        dev_pass = [
            row for row in task_eps
            if row["dev_passes_risk_bound"] == 1 and row["dev_passes_lower_serial_cost"] == 1
        ]
        if dev_pass:
            selected = min(dev_pass, key=lambda row: (fnum(row["dev_serial_cost"]), -fnum(row["dev_score"])))
            selection_status = "dev_risk_bound_pass"
        else:
            lower_cost = [row for row in task_eps if row["dev_passes_lower_serial_cost"] == 1]
            selected = min(lower_cost or task_eps, key=lambda row: (fnum(row["dev_loss_upper95"]), fnum(row["dev_serial_cost"])))
            selection_status = "no_dev_policy_passed_risk_and_cost"
        test_go = (
            selected["test_passes_quality_gate"] == 1
            and selected["test_passes_lower_serial_cost"] == 1
            and selection_status == "dev_risk_bound_pass"
        )
        rows.append(
            {
                "task": task,
                "epsilon": eps,
                "selection_status": selection_status,
                "threshold": selected["threshold"],
                "dev_loss_upper95": selected["dev_loss_upper95"],
                "dev_serial_cost": selected["dev_serial_cost"],
                "test_score": selected["test_score"],
                "test_fixed70_score": selected["test_fixed70_score"],
                "test_delta_vs_fixed70": selected["test_delta_vs_fixed70"],
                "test_fallback_rate": selected["test_fallback_rate"],
                "test_selected_keep": selected["test_selected_keep"],
                "test_serial_cost": selected["test_serial_cost"],
                "test_go_under_contract": int(test_go),
                "reading": (
                    "Selected by dev risk bound and lower serial cost."
                    if selection_status == "dev_risk_bound_pass"
                    else "No threshold satisfied the dev risk bound while keeping serial cost below fixed70."
                ),
            }
        )
    return rows


def build_decision(selection_rows: list[dict[str, Any]]) -> dict[str, Any]:
    required = [
        row for row in selection_rows
        if row["epsilon"] == "0.010"
    ]
    go_tasks = [row["task"] for row in required if row["test_go_under_contract"] == 1]
    status = "go_for_conformal_controller_claim" if {"TextVQA-lite", "DocVQA-lite"}.issubset(go_tasks) else "no_go_for_conformal_controller_claim"
    return {
        "conformal_policy_status": status,
        "epsilon_for_main_gate": "0.010",
        "go_tasks": ",".join(sorted(go_tasks)) or "none",
        "required_tasks": "TextVQA-lite,DocVQA-lite",
        "recommended_claim": (
            "The calibrated fallback is a deployable controller under the audited contract."
            if status.startswith("go")
            else "The calibrated fallback is a principled negative audit: current low-budget risk scores cannot certify fixed70-quality at lower serial cost across TextVQA-lite and DocVQA-lite."
        ),
    }


def build_markdown(candidates: list[dict[str, Any]], selections: list[dict[str, Any]], decision: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Open OCR QA Calibrated Risk-Policy Audit",
            "",
            "This audit calibrates a nested 30%->70% fallback policy on dev samples and evaluates the selected rule on test samples. The risk score uses the low-budget answer, so serial cost is counted as 0.30 plus another 0.70 for fallback samples.",
            "",
            "## Decision",
            "",
            f"- Status: `{decision['conformal_policy_status']}`",
            f"- Main epsilon: {decision['epsilon_for_main_gate']}",
            f"- Go tasks: {decision['go_tasks']}",
            f"- Recommended claim: {decision['recommended_claim']}",
            "",
            "## Selected Policies",
            "",
            table_md(
                selections,
                [
                    "task",
                    "epsilon",
                    "selection_status",
                    "threshold",
                    "dev_loss_upper95",
                    "dev_serial_cost",
                    "test_score",
                    "test_fixed70_score",
                    "test_delta_vs_fixed70",
                    "test_fallback_rate",
                    "test_serial_cost",
                    "test_go_under_contract",
                    "reading",
                ],
            ),
            "",
            "## Candidate Count",
            "",
            f"- candidates: {len(candidates)}",
            "",
        ]
    )


def threshold_grid(values: list[float]) -> list[float]:
    if not values:
        return [0.0, 1.0]
    vals = sorted(values)
    quantiles = [0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00]
    thresholds = {0.0, 1.1}
    for q in quantiles:
        thresholds.add(vals[min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))])
    return sorted(thresholds)


def mean_upper95(values: list[float]) -> float:
    if not values:
        return 0.0
    mu = mean(values)
    if len(values) < 2:
        return mu
    var = sum((value - mu) ** 2 for value in values) / (len(values) - 1)
    return min(1.0, mu + CONFIDENCE_Z * math.sqrt(var / len(values)))


def split_for_id(sample_id: str) -> str:
    import hashlib

    digest = hashlib.md5(sample_id.encode("utf-8")).hexdigest()
    return "dev" if int(digest[:8], 16) % 2 == 0 else "test"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fnum(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def fmt(value: float) -> str:
    return f"{value:.3f}"


def table_md(rows: list[dict[str, Any]], columns: list[str]) -> str:
    out = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(clean_cell(row.get(col, "")) for col in columns) + " |")
    return "\n".join(out)


def clean_cell(value: Any) -> str:
    return str(value).replace("\n", " ").replace("|", "/")


if __name__ == "__main__":
    main()
