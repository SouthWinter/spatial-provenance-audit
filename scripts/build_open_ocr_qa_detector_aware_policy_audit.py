#!/usr/bin/env python3
"""Audit detector/evidence-aware adaptive policies for open OCR QA.

The policy family uses only pre-generation signals: question metadata already
in the stress pack, EasyOCR detector statistics, and selector-mask coverage of
EasyOCR boxes. It selects between 30%, 70%, and optionally full visual prefixes
using thresholds chosen on a development split and evaluated on a held-out
split. This is a stress-pack audit, not a leaderboard result.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "problem_optimization_audit" / "open_ocr_qa_detector_aware_policy"
PAPER_DIR = ROOT / "runs" / "paper_evidence"

DETECTOR_ROWS = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_easyocr_detector_boxes"
    / "easyocr_detector_rows.csv"
)
ECR_ROWS = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_easyocr_detector_boxes_ecr"
    / "bbox_ecr_rows.csv"
)
GEN_PATHS = {
    ("TextVQA-lite", 0.30): ROOT
    / "runs/open_ocr_qa/qwen3_8b_textvqa_lite_target_grid0p30_full500/open_ocr_qa_generation.jsonl",
    ("TextVQA-lite", 0.70): ROOT
    / "runs/open_ocr_qa/qwen3_8b_textvqa_lite_target_grid0p70_full500/open_ocr_qa_generation.jsonl",
    ("DocVQA-lite", 0.30): ROOT
    / "runs/open_ocr_qa/qwen3_8b_docvqa_lite_target_grid0p30_full500/open_ocr_qa_generation.jsonl",
    ("DocVQA-lite", 0.70): ROOT
    / "runs/open_ocr_qa/qwen3_8b_docvqa_lite_target_grid0p70_full500/open_ocr_qa_generation.jsonl",
}
TASKS = ("TextVQA-lite", "DocVQA-lite")
TOL = 0.01


def main() -> None:
    rows = build_rows()
    policy_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for task in TASKS:
        task_rows = [row for row in rows if row["task"] == task]
        dev_rows = [row for row in task_rows if row["split"] == "dev"]
        test_rows = [row for row in task_rows if row["split"] == "test"]
        policies = build_policy_candidates(dev_rows)
        selected = select_policy(dev_rows, policies)
        dev_eval = evaluate_policy(dev_rows, selected)
        test_eval = evaluate_policy(test_rows, selected)
        policy_rows.extend(policy_readout(task, selected, "dev", dev_eval))
        policy_rows.extend(policy_readout(task, selected, "test", test_eval))
        summary_rows.append(task_summary(task, selected, dev_eval, test_eval))

    pooled_dev = [row for row in rows if row["split"] == "dev"]
    pooled_test_by_task = {task: [row for row in rows if row["task"] == task and row["split"] == "test"] for task in TASKS}
    pooled_policy = select_policy(pooled_dev, build_policy_candidates(pooled_dev))
    pooled_dev_eval = evaluate_policy(pooled_dev, pooled_policy)
    for task, test_rows in pooled_test_by_task.items():
        test_eval = evaluate_policy(test_rows, pooled_policy)
        policy_rows.extend(policy_readout(f"pooled->{task}", pooled_policy, "test", test_eval))
        summary_rows.append(task_summary(f"pooled->{task}", pooled_policy, pooled_dev_eval, test_eval))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "detector_aware_policy_rows.csv", rows)
    write_csv(OUT_DIR / "detector_aware_policy_readout.csv", policy_rows)
    write_csv(OUT_DIR / "detector_aware_policy_summary.csv", summary_rows)
    write_csv(PAPER_DIR / "table_open_ocr_qa_detector_aware_policy_summary.csv", summary_rows)
    write_csv(PAPER_DIR / "table_open_ocr_qa_detector_aware_policy_readout.csv", policy_rows)
    (OUT_DIR / "detector_aware_policy_report.md").write_text(
        build_report(summary_rows, policy_rows), encoding="utf-8"
    )
    print(f"Wrote detector-aware policy audit for {len(rows)} rows to {OUT_DIR}")


def build_rows() -> list[dict[str, Any]]:
    detector = {row["sample_id"]: row for row in read_csv(DETECTOR_ROWS)}
    ecr = load_ecr()
    generation = load_generation()
    rows: list[dict[str, Any]] = []
    for sample_id, det in detector.items():
        task = det["task"]
        gen30 = generation.get((sample_id, 0.30))
        gen70 = generation.get((sample_id, 0.70))
        if task not in TASKS or not gen30 or not gen70:
            continue
        ecr30 = ecr.get((sample_id, 0.30), {})
        ecr70 = ecr.get((sample_id, 0.70), {})
        full_score = f(gen30["full_score"])
        score30 = f(gen30["pruned_score"])
        score70 = f(gen70["pruned_score"])
        rows.append(
            {
                "sample_id": sample_id,
                "task": task,
                "split": split_for_sample(sample_id),
                "full_score": fmt(full_score),
                "score30": fmt(score30),
                "score70": fmt(score70),
                "detector_box_count": fmt(f(det.get("box_count"))),
                "detector_has_boxes": int(f(det.get("box_count")) > 0),
                "detector_mean_confidence": fmt(f(det.get("mean_confidence"))),
                "detector_elapsed_ms": fmt(f(det.get("detector_elapsed_ms"))),
                "ecr30": fmt(f(ecr30.get("ECR"))),
                "worst_ecr30": fmt(f(ecr30.get("worst_region_ECR"))),
                "all_regions30": fmt(f(ecr30.get("all_regions_ECR_ge_0p50"))),
                "ecr70": fmt(f(ecr70.get("ECR"))),
                "worst_ecr70": fmt(f(ecr70.get("worst_region_ECR"))),
                "low30_failure": int(score30 < full_score - 0.1),
                "repair_by70": int(score30 < full_score - 0.1 and score70 >= full_score - 0.1),
            }
        )
    return rows


def load_ecr() -> dict[tuple[str, float], dict[str, str]]:
    out: dict[tuple[str, float], dict[str, str]] = {}
    for row in read_csv(ECR_ROWS):
        if row.get("metric_status") != "scored":
            continue
        out[(row["sample_id"], round(f(row["budget_keep_ratio"]), 2))] = row
    return out


def load_generation() -> dict[tuple[str, float], dict[str, Any]]:
    out: dict[tuple[str, float], dict[str, Any]] = {}
    for (_task, budget), path in GEN_PATHS.items():
        for row in read_jsonl(path):
            out[(row["sample_id"], budget)] = row
    return out


def build_policy_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs = [
        ("missing_or_low_ecr30_to70", "ecr30", "<=", 0.70, True),
        ("missing_or_low_worst_ecr30_to70", "worst_ecr30", "<=", 0.70, True),
        ("low_all_regions30_to70", "all_regions30", "<=", 0.70, False),
        ("many_boxes_to70", "detector_box_count", ">=", 0.70, False),
        ("low_confidence_to70", "detector_mean_confidence", "<=", 0.70, False),
        ("missing_or_low_ecr30_tofull", "ecr30", "<=", 1.00, True),
        ("many_boxes_tofull", "detector_box_count", ">=", 1.00, False),
    ]
    policies: list[dict[str, Any]] = [
        {"policy": "fixed30", "feature": "none", "op": "fixed", "threshold": "", "risk_keep": 0.30, "missing_is_risk": False},
        {"policy": "fixed70", "feature": "none", "op": "fixed", "threshold": "", "risk_keep": 0.70, "missing_is_risk": False},
        {"policy": "full", "feature": "none", "op": "fixed", "threshold": "", "risk_keep": 1.00, "missing_is_risk": False},
    ]
    for name, feature, op, risk_keep, missing_is_risk in specs:
        values = sorted({f(row.get(feature)) for row in rows if row.get(feature) not in ("", None)})
        if not values:
            continue
        thresholds = quantile_thresholds(values)
        for threshold in thresholds:
            policies.append(
                {
                    "policy": name,
                    "feature": feature,
                    "op": op,
                    "threshold": threshold,
                    "risk_keep": risk_keep,
                    "missing_is_risk": missing_is_risk,
                }
            )
    return policies


def select_policy(rows: list[dict[str, Any]], policies: list[dict[str, Any]]) -> dict[str, Any]:
    fixed70 = evaluate_policy(rows, {"policy": "fixed70", "op": "fixed", "risk_keep": 0.70})
    eligible = []
    evaluated = []
    for policy in policies:
        ev = evaluate_policy(rows, policy)
        evaluated.append((policy, ev))
        if ev["score"] >= fixed70["score"] - TOL and ev["mean_keep"] < fixed70["mean_keep"]:
            eligible.append((policy, ev))
    if eligible:
        return max(eligible, key=lambda item: (item[1]["score"], -item[1]["mean_keep"]))[0]
    return max(evaluated, key=lambda item: (item[1]["score"], -item[1]["mean_keep"]))[0]


def evaluate_policy(rows: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    if not rows:
        return empty_eval()
    scores = []
    keeps = []
    risk_flags = []
    detector_ms = []
    for row in rows:
        keep = choose_keep(row, policy)
        keeps.append(keep)
        risk_flags.append(int(keep > 0.30))
        detector_ms.append(f(row.get("detector_elapsed_ms")))
        if keep >= 1.0:
            scores.append(f(row["full_score"]))
        elif keep >= 0.70:
            scores.append(f(row["score70"]))
        else:
            scores.append(f(row["score30"]))
    fixed70_score = mean(f(row["score70"]) for row in rows)
    return {
        "n": len(rows),
        "score": mean(scores),
        "fixed70_score": fixed70_score,
        "delta_vs_fixed70": mean(scores) - fixed70_score,
        "mean_keep": mean(keeps),
        "risk_rate": mean(risk_flags),
        "mean_detector_ms": mean(detector_ms),
        "passes_near_fixed70_lower_keep": int(mean(scores) >= fixed70_score - TOL and mean(keeps) < 0.70),
    }


def choose_keep(row: dict[str, Any], policy: dict[str, Any]) -> float:
    name = policy.get("policy", "")
    if name == "fixed30":
        return 0.30
    if name == "fixed70":
        return 0.70
    if name == "full":
        return 1.00
    value_raw = row.get(str(policy.get("feature", "")), "")
    if value_raw in ("", None):
        risk = bool(policy.get("missing_is_risk"))
    else:
        value = f(value_raw)
        threshold = f(policy.get("threshold"))
        if policy.get("op") == "<=":
            risk = value <= threshold
        elif policy.get("op") == ">=":
            risk = value >= threshold
        else:
            risk = False
    return f(policy.get("risk_keep", 0.70)) if risk else 0.30


def policy_readout(task: str, policy: dict[str, Any], split: str, ev: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "task": task,
            "split": split,
            "policy": policy.get("policy", ""),
            "feature": policy.get("feature", ""),
            "op": policy.get("op", ""),
            "threshold": fmt(policy.get("threshold")) if policy.get("threshold") != "" else "",
            "risk_keep": fmt(policy.get("risk_keep", "")),
            "n": ev["n"],
            "score": fmt(ev["score"]),
            "fixed70_score": fmt(ev["fixed70_score"]),
            "delta_vs_fixed70": fmt(ev["delta_vs_fixed70"]),
            "mean_keep": fmt(ev["mean_keep"]),
            "risk_rate": fmt(ev["risk_rate"]),
            "mean_detector_ms": fmt(ev["mean_detector_ms"]),
            "passes_near_fixed70_lower_keep": ev["passes_near_fixed70_lower_keep"],
        }
    ]


def task_summary(task: str, policy: dict[str, Any], dev: dict[str, Any], test: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": task,
        "selected_policy": policy.get("policy", ""),
        "feature": policy.get("feature", ""),
        "op": policy.get("op", ""),
        "threshold": fmt(policy.get("threshold")) if policy.get("threshold") != "" else "",
        "risk_keep": fmt(policy.get("risk_keep", "")),
        "dev_score": fmt(dev["score"]),
        "dev_mean_keep": fmt(dev["mean_keep"]),
        "dev_delta_vs_fixed70": fmt(dev["delta_vs_fixed70"]),
        "test_score": fmt(test["score"]),
        "test_mean_keep": fmt(test["mean_keep"]),
        "test_delta_vs_fixed70": fmt(test["delta_vs_fixed70"]),
        "test_passes_near_fixed70_lower_keep": test["passes_near_fixed70_lower_keep"],
        "mean_detector_ms": fmt(test["mean_detector_ms"]),
        "reading": reading(test),
    }


def reading(ev: dict[str, Any]) -> str:
    if ev["passes_near_fixed70_lower_keep"]:
        return "passes stress-pack near-fixed70/lower-keep gate, but detector cost must be counted"
    if ev["mean_keep"] < 0.70:
        return "saves selected keep but trails fixed70 quality on held-out stress rows"
    return "matches or exceeds fixed70 cost, so it is not an efficiency-improving controller"


def empty_eval() -> dict[str, Any]:
    return {
        "n": 0,
        "score": 0.0,
        "fixed70_score": 0.0,
        "delta_vs_fixed70": 0.0,
        "mean_keep": 0.0,
        "risk_rate": 0.0,
        "mean_detector_ms": 0.0,
        "passes_near_fixed70_lower_keep": 0,
    }


def build_report(summary_rows: list[dict[str, Any]], policy_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Open OCR QA Detector-Aware Policy Audit",
        "",
        "This audit asks whether deployment-visible detector/evidence signals improve adaptive pruning on the 96-row open-QA stress pack. Policies use EasyOCR box statistics and selector-mask coverage of EasyOCR boxes before answer generation; thresholds are selected on a dev split and reported on held-out test rows.",
        "",
        "## Summary",
        "",
        "| Task | Policy | Feature | Threshold | Test score | Keep | Delta vs fixed70 | Pass | Detector ms | Reading |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['task']} | {row['selected_policy']} | {row['feature']} | {row['threshold']} | "
            f"{row['test_score']} | {row['test_mean_keep']} | {row['test_delta_vs_fixed70']} | "
            f"{row['test_passes_near_fixed70_lower_keep']} | {row['mean_detector_ms']} | {row['reading']} |"
        )
    lines.extend(
        [
            "",
            "## Readout",
            "",
            "| Task | Split | Policy | Score | Fixed70 | Keep | Risk rate | Pass |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in policy_rows:
        lines.append(
            f"| {row['task']} | {row['split']} | {row['policy']} | {row['score']} | "
            f"{row['fixed70_score']} | {row['mean_keep']} | {row['risk_rate']} | "
            f"{row['passes_near_fixed70_lower_keep']} |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "Detector-aware features are deployable only when OCR/layout boxes are already available or their latency is included. A positive selected-keep result does not imply end-to-end speedup under online EasyOCR. This audit is limited to the 96-row stress pack and should not be reported as a full TextVQA/DocVQA controller.",
        ]
    )
    return "\n".join(lines) + "\n"


def quantile_thresholds(values: list[float]) -> list[float]:
    if not values:
        return []
    out = {values[0], values[-1]}
    for q in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        idx = min(len(values) - 1, max(0, round(q * (len(values) - 1))))
        out.add(values[idx])
    return sorted(out)


def split_for_sample(sample_id: str) -> str:
    digest = hashlib.md5(sample_id.encode("utf-8")).hexdigest()
    return "dev" if int(digest[:8], 16) % 2 == 0 else "test"


def f(value: Any) -> float:
    try:
        if value in ("", None):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def fmt(value: Any) -> str:
    if value == "":
        return ""
    return f"{f(value):.3f}"


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


if __name__ == "__main__":
    main()
