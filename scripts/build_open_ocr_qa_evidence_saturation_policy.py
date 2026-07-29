#!/usr/bin/env python3
"""Calibrate and audit a unified evidence-saturation budget policy.

The script joins one pre-generation diagnostic trace per task with cached
30/50/70% generation outputs. Gold scores are used only on the deterministic
development split to choose a single pair of saturation thresholds. Test-time
budget decisions use selector-score diagnostics only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
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
DEFAULT_TRACES = {
    "TextVQA-lite": ROOT / "runs/open_ocr_qa/qwen3_8b_textvqa_lite_evidence_saturation_probe_full500/prune_traces.jsonl",
    "DocVQA-lite": ROOT / "runs/open_ocr_qa/qwen3_8b_docvqa_lite_evidence_saturation_probe_full500/prune_traces.jsonl",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "runs/problem_optimization_audit/open_ocr_qa_evidence_saturation_policy"),
    )
    parser.add_argument("--textvqa-trace", default=str(DEFAULT_TRACES["TextVQA-lite"]))
    parser.add_argument("--docvqa-trace", default=str(DEFAULT_TRACES["DocVQA-lite"]))
    parser.add_argument("--quality-tolerance", type=float, default=0.015)
    parser.add_argument("--target-max-keep", type=float, default=0.60)
    args = parser.parse_args()

    traces = {
        "TextVQA-lite": Path(args.textvqa_trace),
        "DocVQA-lite": Path(args.docvqa_trace),
    }
    rows = {task: load_task_rows(task, path) for task, path in traces.items()}
    policies = candidate_policies()
    selected, selection = select_unified_policy(
        rows,
        policies,
        quality_tolerance=args.quality_tolerance,
    )
    summaries = baseline_summaries(rows)
    summaries.extend(policy_summaries(rows, selected, selected_by="pooled_dev"))
    decision = build_decision(
        rows,
        selected,
        quality_tolerance=args.quality_tolerance,
        target_max_keep=args.target_max_keep,
    )
    prediction_rows = build_prediction_rows(rows, selected)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "evidence_saturation_policy_selection.csv", [selection])
    write_csv(output_dir / "evidence_saturation_policy_summary.csv", summaries)
    write_csv(output_dir / "evidence_saturation_policy_predictions.csv", prediction_rows)
    write_csv(output_dir / "evidence_saturation_policy_decision.csv", [decision])
    (output_dir / "evidence_saturation_policy_report.md").write_text(
        render_report(selection, summaries, decision),
        encoding="utf-8",
    )
    print(f"Wrote evidence-saturation policy audit to {output_dir}")


def load_task_rows(task: str, trace_path: Path) -> list[dict[str, Any]]:
    if not trace_path.is_file():
        raise FileNotFoundError(f"Missing diagnostic trace for {task}: {trace_path}")
    traces = {str(row["sample_id"]): row for row in read_jsonl(trace_path)}
    by_sample: dict[str, dict[str, Any]] = {}
    for budget, path in RUNS[task].items():
        for item in read_jsonl(path):
            sid = str(item["sample_id"])
            row = by_sample.setdefault(
                sid,
                {
                    "task": task,
                    "sample_id": sid,
                    "split": split_for_id(sid),
                    "budgets": {},
                },
            )
            row["budgets"][budget] = {
                "score": float(item.get("pruned_score", 0.0)),
                "exact": float(item.get("pruned_exact", 0.0)),
                "keep": float(item.get("effective_keep_ratio", budget) or budget),
            }
    missing = sorted(set(by_sample) - set(traces))
    if missing:
        raise ValueError(f"Diagnostic trace for {task} misses {len(missing)} samples")
    rows: list[dict[str, Any]] = []
    for sid, row in by_sample.items():
        diagnostics = traces[sid].get("saturation_candidate_diagnostics", [])
        row["candidate_diagnostics"] = {
            round(float(item["ratio"]), 2): item for item in diagnostics
        }
        if set(row["candidate_diagnostics"]) != {0.30, 0.50, 0.70}:
            raise ValueError(f"Incomplete saturation diagnostics for {task}:{sid}")
        rows.append(row)
    return rows


def candidate_policies() -> list[dict[str, float]]:
    mass_targets = [value / 100.0 for value in range(50, 91, 2)]
    cell_targets = (0.50, 0.625, 0.75, 0.875, 1.00)
    return [
        {"mass_target": mass_target, "cell_target": cell_target}
        for mass_target in mass_targets
        for cell_target in cell_targets
    ]


def select_unified_policy(
    rows: dict[str, list[dict[str, Any]]],
    policies: list[dict[str, float]],
    *,
    quality_tolerance: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    fixed70 = {
        task: summarize(task_rows, split="dev", fixed_budget=0.70)
        for task, task_rows in rows.items()
    }
    scored: list[tuple[bool, float, float, dict[str, float], dict[str, dict[str, float]]]] = []
    for policy in policies:
        task_summaries = {
            task: summarize(task_rows, split="dev", policy=policy)
            for task, task_rows in rows.items()
        }
        pass_all = all(
            task_summaries[task]["score"] >= fixed70[task]["score"] - quality_tolerance
            and task_summaries[task]["mean_keep"] < fixed70[task]["mean_keep"]
            for task in rows
        )
        aggregate_score = mean(summary["score"] for summary in task_summaries.values())
        aggregate_keep = mean(summary["mean_keep"] for summary in task_summaries.values())
        worst_gap = min(task_summaries[task]["score"] - fixed70[task]["score"] for task in rows)
        scored.append((pass_all, aggregate_keep, worst_gap, policy, task_summaries))
    passing = [item for item in scored if item[0]]
    if passing:
        passing.sort(key=lambda item: (item[1], -item[2]))
        chosen = passing[0]
        status = "dev_gate_pass"
    else:
        scored.sort(key=lambda item: (item[2], -item[1]), reverse=True)
        chosen = scored[0]
        status = "dev_gate_fail_best_available"
    _, aggregate_keep, worst_gap, policy, task_summaries = chosen
    selection = {
        "status": status,
        "mass_target": fmt(policy["mass_target"]),
        "cell_target": fmt(policy["cell_target"]),
        "dev_aggregate_mean_keep": fmt(aggregate_keep),
        "dev_worst_score_gap_vs_fixed70": fmt(worst_gap),
        "textvqa_dev_score": fmt(task_summaries["TextVQA-lite"]["score"]),
        "docvqa_dev_score": fmt(task_summaries["DocVQA-lite"]["score"]),
    }
    return policy, selection


def choose_budget(row: dict[str, Any], policy: dict[str, float]) -> float:
    for budget in (0.30, 0.50, 0.70):
        diagnostic = row["candidate_diagnostics"][budget]
        if (
            float(diagnostic["mass_coverage"]) >= policy["mass_target"]
            and float(diagnostic["active_cell_coverage"]) >= policy["cell_target"]
        ):
            return budget
    return 0.70


def summarize(
    rows: list[dict[str, Any]],
    *,
    split: str,
    policy: dict[str, float] | None = None,
    fixed_budget: float | None = None,
) -> dict[str, float]:
    subset = [row for row in rows if split == "all" or row["split"] == split]
    budgets = [fixed_budget if fixed_budget is not None else choose_budget(row, policy or {}) for row in subset]
    return {
        "n": len(subset),
        "score": mean(row["budgets"][budget]["score"] for row, budget in zip(subset, budgets)),
        "exact": mean(row["budgets"][budget]["exact"] for row, budget in zip(subset, budgets)),
        "mean_keep": mean(row["budgets"][budget]["keep"] for row, budget in zip(subset, budgets)),
        "budget30_rate": mean(float(budget == 0.30) for budget in budgets),
        "budget50_rate": mean(float(budget == 0.50) for budget in budgets),
        "budget70_rate": mean(float(budget == 0.70) for budget in budgets),
    }


def baseline_summaries(rows: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    output = []
    for task, task_rows in rows.items():
        for budget in (0.30, 0.50, 0.70):
            for split in ("dev", "test", "all"):
                output.append(summary_row(task, split, f"fixed_{budget:.2f}", summarize(task_rows, split=split, fixed_budget=budget)))
    return output


def policy_summaries(
    rows: dict[str, list[dict[str, Any]]],
    policy: dict[str, float],
    *,
    selected_by: str,
) -> list[dict[str, Any]]:
    name = f"saturation_m{policy['mass_target']:.2f}_c{policy['cell_target']:.3f}"
    output = []
    for task, task_rows in rows.items():
        for split in ("dev", "test", "all"):
            row = summary_row(task, split, name, summarize(task_rows, split=split, policy=policy))
            row["selected_by"] = selected_by
            output.append(row)
    return output


def build_decision(
    rows: dict[str, list[dict[str, Any]]],
    policy: dict[str, float],
    *,
    quality_tolerance: float,
    target_max_keep: float,
) -> dict[str, Any]:
    gates = []
    task_values: dict[str, dict[str, float]] = {}
    for task, task_rows in rows.items():
        fixed = summarize(task_rows, split="test", fixed_budget=0.70)
        adaptive = summarize(task_rows, split="test", policy=policy)
        task_values[task] = adaptive
        gates.append(
            adaptive["score"] >= fixed["score"] - quality_tolerance
            and adaptive["mean_keep"] <= target_max_keep
        )
    return {
        "status": "go_for_main_controller_claim" if all(gates) else "no_go_for_main_controller_claim",
        "passed_tasks": sum(gates),
        "required_tasks": len(gates),
        "quality_tolerance": fmt(quality_tolerance),
        "target_max_keep": fmt(target_max_keep),
        "textvqa_test_score": fmt(task_values["TextVQA-lite"]["score"]),
        "textvqa_test_keep": fmt(task_values["TextVQA-lite"]["mean_keep"]),
        "docvqa_test_score": fmt(task_values["DocVQA-lite"]["score"]),
        "docvqa_test_keep": fmt(task_values["DocVQA-lite"]["mean_keep"]),
    }


def build_prediction_rows(
    rows: dict[str, list[dict[str, Any]]],
    policy: dict[str, float],
) -> list[dict[str, Any]]:
    output = []
    for task, task_rows in rows.items():
        for row in task_rows:
            budget = choose_budget(row, policy)
            output.append(
                {
                    "task": task,
                    "sample_id": row["sample_id"],
                    "split": row["split"],
                    "selected_budget": fmt(budget),
                    "score": fmt(row["budgets"][budget]["score"]),
                    "effective_keep": fmt(row["budgets"][budget]["keep"]),
                }
            )
    return output


def summary_row(task: str, split: str, policy: str, values: dict[str, float]) -> dict[str, Any]:
    return {
        "task": task,
        "split": split,
        "policy": policy,
        **{key: fmt(value) if isinstance(value, float) else value for key, value in values.items()},
    }


def render_report(selection: dict[str, Any], summaries: list[dict[str, Any]], decision: dict[str, Any]) -> str:
    test_rows = [row for row in summaries if row["split"] == "test"]
    return "\n".join(
        [
            "# Open OCR QA Evidence-Saturation Controller",
            "",
            "The controller selects 30%, 50%, or 70% visual-token retention before LLM prefill from target-conditioned relevance mass and spatial coverage. One threshold pair is calibrated on pooled development splits and evaluated unchanged on both held-out tasks.",
            "",
            "## Selection",
            "",
            table_md([selection]),
            "",
            "## Held-Out Results",
            "",
            table_md(test_rows),
            "",
            "## Decision",
            "",
            table_md([decision]),
            "",
            "Safe claim: the policy is deployable only if both held-out task gates pass. Gold answers are used for development calibration and evaluation, never for test-time budget selection.",
            "",
        ]
    )


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
    columns = ordered_columns(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def table_md(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "(empty)"
    columns = ordered_columns(rows)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def ordered_columns(rows: list[dict[str, Any]]) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for column in row:
            if column not in seen:
                seen.add(column)
                columns.append(column)
    return columns


def fmt(value: float) -> str:
    if math.isnan(float(value)):
        return "nan"
    return f"{float(value):.6f}"


if __name__ == "__main__":
    main()
