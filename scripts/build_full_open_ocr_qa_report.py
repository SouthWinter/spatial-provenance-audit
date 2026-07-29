#!/usr/bin/env python3
"""Validate and summarize the two-model full-validation open-QA matrix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


RUNS = (
    ("Qwen3-VL-8B", "TextVQA", "Target-Grid", "30%", "qwen3_8b_textvqa_val_target_grid0p30"),
    ("Qwen3-VL-8B", "TextVQA", "Target-Grid", "70%", "qwen3_8b_textvqa_val_target_grid0p70"),
    ("Qwen3-VL-8B", "DocVQA", "Target-Grid", "30%", "qwen3_8b_docvqa_val_target_grid0p30"),
    ("Qwen3-VL-8B", "DocVQA", "Target-Grid", "70%", "qwen3_8b_docvqa_val_target_grid0p70"),
    ("LLaVA-1.5-7B", "TextVQA", "Target", "40%", "llava15_7b_textvqa_val_target0p40"),
    ("LLaVA-1.5-7B", "TextVQA", "Target", "70%", "llava15_7b_textvqa_val_target0p70"),
    ("LLaVA-1.5-7B", "DocVQA", "Target", "40%", "llava15_7b_docvqa_val_target0p40"),
    ("LLaVA-1.5-7B", "DocVQA", "Target", "70%", "llava15_7b_docvqa_val_target0p70"),
)

MATCHED_BASELINES = (
    ("Qwen3-VL-8B", "TextVQA", "Random", "70%", "qwen3_8b_textvqa_val_random0p70"),
    ("Qwen3-VL-8B", "TextVQA", "Grid", "70%", "qwen3_8b_textvqa_val_grid0p70"),
    ("Qwen3-VL-8B", "DocVQA", "Random", "70%", "qwen3_8b_docvqa_val_random0p70"),
    ("Qwen3-VL-8B", "DocVQA", "Grid", "70%", "qwen3_8b_docvqa_val_grid0p70"),
    ("LLaVA-1.5-7B", "TextVQA", "Random", "70%", "llava15_7b_textvqa_val_random0p70"),
    ("LLaVA-1.5-7B", "TextVQA", "Grid", "70%", "llava15_7b_textvqa_val_grid0p70"),
    ("LLaVA-1.5-7B", "DocVQA", "Random", "70%", "llava15_7b_docvqa_val_random0p70"),
    ("LLaVA-1.5-7B", "DocVQA", "Grid", "70%", "llava15_7b_docvqa_val_grid0p70"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/open_ocr_qa_full"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()

    loaded: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    summaries: list[dict[str, Any]] = []
    for offset, (model, task, method, budget, relative) in enumerate(RUNS + MATCHED_BASELINES):
        run_dir = args.root / relative
        rows = load_and_validate(run_dir)
        loaded[(model, task, method, budget)] = rows
        summaries.append(
            summarize(
                model,
                task,
                method,
                budget,
                rows,
                bootstrap=args.bootstrap,
                seed=args.seed + offset,
            )
        )

    recoveries: list[dict[str, Any]] = []
    for offset, (model, task, aggressive) in enumerate(
        (
            ("Qwen3-VL-8B", "TextVQA", "30%"),
            ("Qwen3-VL-8B", "DocVQA", "30%"),
            ("LLaVA-1.5-7B", "TextVQA", "40%"),
            ("LLaVA-1.5-7B", "DocVQA", "40%"),
        )
    ):
        method = "Target-Grid" if model == "Qwen3-VL-8B" else "Target"
        aggressive_rows = loaded[(model, task, method, aggressive)]
        safe_rows = loaded[(model, task, method, "70%")]
        verify_reused_full(aggressive_rows, safe_rows, model=model, task=task)
        recoveries.append(
            summarize_recovery(
                model,
                task,
                aggressive,
                aggressive_rows,
                safe_rows,
                bootstrap=args.bootstrap,
                seed=args.seed + 100 + offset,
            )
        )

    matched_comparisons: list[dict[str, Any]] = []
    for offset, (model, task) in enumerate(
        (
            ("Qwen3-VL-8B", "TextVQA"),
            ("Qwen3-VL-8B", "DocVQA"),
            ("LLaVA-1.5-7B", "TextVQA"),
            ("LLaVA-1.5-7B", "DocVQA"),
        )
    ):
        method = "Target-Grid" if model == "Qwen3-VL-8B" else "Target"
        target_rows = loaded[(model, task, method, "70%")]
        for baseline_index, baseline in enumerate(("Random", "Grid")):
            baseline_rows = loaded[(model, task, baseline, "70%")]
            verify_reused_full(target_rows, baseline_rows, model=model, task=task)
            matched_comparisons.append(
                summarize_method_delta(
                    model,
                    task,
                    method,
                    target_rows,
                    baseline,
                    baseline_rows,
                    bootstrap=args.bootstrap,
                    seed=args.seed + 200 + offset * 10 + baseline_index,
                )
            )

    output_dir = args.output_dir or args.root / "report"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "Full TextVQA/DocVQA validation; original-question greedy generation; question-only selectors",
        "bootstrap_samples": args.bootstrap,
        "seed": args.seed,
        "runs": summaries,
        "safe_budget_recovery": recoveries,
        "matched_budget_comparisons": matched_comparisons,
    }
    (output_dir / "full_open_ocr_qa_report.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(output_dir / "full_open_ocr_qa_runs.csv", summaries)
    write_csv(output_dir / "full_open_ocr_qa_recovery.csv", recoveries)
    write_csv(output_dir / "full_open_ocr_qa_matched_comparisons.csv", matched_comparisons)
    (output_dir / "full_open_ocr_qa_report.md").write_text(markdown(payload), encoding="utf-8")
    print(f"Wrote full open-QA report to {output_dir}")


def load_and_validate(run_dir: Path) -> list[dict[str, Any]]:
    rows_path = run_dir / "open_ocr_qa_generation.jsonl"
    traces_path = run_dir / "prune_traces.jsonl"
    metrics_path = run_dir / "metrics.json"
    for path in (rows_path, traces_path, metrics_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    rows = load_jsonl(rows_path)
    traces = load_jsonl(traces_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if len(rows) != len(traces) or len(rows) != int(metrics.get("n", -1)):
        raise ValueError(f"Count mismatch in {run_dir}")
    row_ids = [str(row.get("sample_id", "")) for row in rows]
    trace_ids = [str(trace.get("sample_id", "")) for trace in traces]
    if not row_ids or row_ids != trace_ids or len(set(row_ids)) != len(row_ids):
        raise ValueError(f"Missing, duplicated, or misaligned sample IDs in {run_dir}")
    if any(row.get("selector_target_source") not in {"question", "question_only"} for row in rows):
        raise ValueError(f"Selector is not question-only in {run_dir}")
    return rows


def summarize(
    model: str,
    task: str,
    method: str,
    budget: str,
    rows: list[dict[str, Any]],
    *,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    full = np.asarray([float(row["full_score"]) for row in rows], dtype=np.float64)
    pruned = np.asarray([float(row["pruned_score"]) for row in rows], dtype=np.float64)
    delta = pruned - full
    low, high = bootstrap_mean_ci(delta, bootstrap=bootstrap, seed=seed)
    return {
        "model": model,
        "task": task,
        "method": method,
        "budget": budget,
        "n": len(rows),
        "metric": "TextVQA accuracy" if task == "TextVQA" else "ANLS",
        "full_score": float(full.mean()),
        "pruned_score": float(pruned.mean()),
        "paired_delta": float(delta.mean()),
        "ci_low": low,
        "ci_high": high,
        "wins": int(np.count_nonzero(delta > 1e-12)),
        "losses": int(np.count_nonzero(delta < -1e-12)),
        "ties": int(np.count_nonzero(np.abs(delta) <= 1e-12)),
        "mean_keep": float(np.mean([float(row["effective_keep_ratio"]) for row in rows])),
    }


def summarize_method_delta(
    model: str,
    task: str,
    method: str,
    method_rows: list[dict[str, Any]],
    baseline: str,
    baseline_rows: list[dict[str, Any]],
    *,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    method_by_id = {str(row["sample_id"]): row for row in method_rows}
    baseline_by_id = {str(row["sample_id"]): row for row in baseline_rows}
    if set(method_by_id) != set(baseline_by_id):
        raise ValueError(f"Sample sets differ for {model} {task}: {method} vs {baseline}")
    ids = [str(row["sample_id"]) for row in method_rows]
    delta = np.asarray(
        [
            float(method_by_id[sample_id]["pruned_score"])
            - float(baseline_by_id[sample_id]["pruned_score"])
            for sample_id in ids
        ],
        dtype=np.float64,
    )
    method_keep = np.asarray(
        [float(method_by_id[sample_id]["effective_keep_ratio"]) for sample_id in ids],
        dtype=np.float64,
    )
    baseline_keep = np.asarray(
        [float(baseline_by_id[sample_id]["effective_keep_ratio"]) for sample_id in ids],
        dtype=np.float64,
    )
    low, high = bootstrap_mean_ci(delta, bootstrap=bootstrap, seed=seed)
    return {
        "model": model,
        "task": task,
        "budget": "70%",
        "comparison": f"{method} minus {baseline}",
        "n": len(ids),
        "paired_delta": float(delta.mean()),
        "ci_low": low,
        "ci_high": high,
        "wins": int(np.count_nonzero(delta > 1e-12)),
        "losses": int(np.count_nonzero(delta < -1e-12)),
        "ties": int(np.count_nonzero(np.abs(delta) <= 1e-12)),
        "method_keep": float(method_keep.mean()),
        "baseline_keep": float(baseline_keep.mean()),
        "keep_delta": float((method_keep - baseline_keep).mean()),
    }


def summarize_recovery(
    model: str,
    task: str,
    aggressive_budget: str,
    aggressive_rows: list[dict[str, Any]],
    safe_rows: list[dict[str, Any]],
    *,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    aggressive_by_id = {str(row["sample_id"]): row for row in aggressive_rows}
    safe_by_id = {str(row["sample_id"]): row for row in safe_rows}
    ids = [str(row["sample_id"]) for row in safe_rows]
    delta = np.asarray(
        [float(safe_by_id[sample_id]["pruned_score"]) - float(aggressive_by_id[sample_id]["pruned_score"]) for sample_id in ids],
        dtype=np.float64,
    )
    low, high = bootstrap_mean_ci(delta, bootstrap=bootstrap, seed=seed)
    return {
        "model": model,
        "task": task,
        "comparison": f"70% minus {aggressive_budget}",
        "n": len(ids),
        "paired_recovery": float(delta.mean()),
        "ci_low": low,
        "ci_high": high,
        "improved": int(np.count_nonzero(delta > 1e-12)),
        "degraded": int(np.count_nonzero(delta < -1e-12)),
        "tied": int(np.count_nonzero(np.abs(delta) <= 1e-12)),
    }


def verify_reused_full(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    model: str,
    task: str,
) -> None:
    left_by_id = {str(row["sample_id"]): row for row in left}
    right_by_id = {str(row["sample_id"]): row for row in right}
    if set(left_by_id) != set(right_by_id):
        raise ValueError(f"Sample sets differ for {model} {task}")
    mismatches = [
        sample_id
        for sample_id in left_by_id
        if left_by_id[sample_id]["full_answer"] != right_by_id[sample_id]["full_answer"]
        or float(left_by_id[sample_id]["full_score"]) != float(right_by_id[sample_id]["full_score"])
    ]
    if mismatches:
        raise ValueError(f"Full-answer reuse mismatch for {model} {task}: {len(mismatches)} rows")


def bootstrap_mean_ci(values: np.ndarray, *, bootstrap: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(bootstrap, dtype=np.float64)
    for start in range(0, bootstrap, 500):
        count = min(500, bootstrap - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means[start : start + count] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Full-Validation Open OCR/Document QA",
        "",
        "All selectors receive the original question only. Confidence intervals are paired sample bootstrap intervals.",
        "",
        "| Model | Task | Method | Budget | n | Metric | Full | Pruned | Delta | 95% CI | Win/Loss/Tie | Keep |",
        "|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["runs"]:
        lines.append(
            f"| {row['model']} | {row['task']} | {row['method']} | {row['budget']} | {row['n']} | {row['metric']} | "
            f"{row['full_score']:.4f} | {row['pruned_score']:.4f} | {row['paired_delta']:+.4f} | "
            f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | "
            f"{row['wins']}/{row['losses']}/{row['ties']} | {row['mean_keep']:.4f} |"
        )
    lines.extend(
        [
            "",
            "| Model | Task | Comparison | Recovery | 95% CI | Improved/Degraded/Tied |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in payload["safe_budget_recovery"]:
        lines.append(
            f"| {row['model']} | {row['task']} | {row['comparison']} | {row['paired_recovery']:+.4f} | "
            f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | "
            f"{row['improved']}/{row['degraded']}/{row['tied']} |"
        )
    lines.extend(
        [
            "",
            "| Model | Task | Budget | Comparison | Delta | 95% CI | Win/Loss/Tie | Target/Base keep |",
            "|---|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in payload["matched_budget_comparisons"]:
        lines.append(
            f"| {row['model']} | {row['task']} | {row['budget']} | {row['comparison']} | "
            f"{row['paired_delta']:+.4f} | [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | "
            f"{row['wins']}/{row['losses']}/{row['ties']} | "
            f"{row['method_keep']:.4f}/{row['baseline_keep']:.4f} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
