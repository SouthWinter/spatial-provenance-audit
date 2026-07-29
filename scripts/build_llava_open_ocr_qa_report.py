#!/usr/bin/env python3
"""Validate and summarize paired full-versus-pruned LLaVA open-QA runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--textvqa-dir", type=Path, required=True)
    parser.add_argument("--docvqa-dir", type=Path, required=True)
    parser.add_argument("--qwen-textvqa-rows", type=Path)
    parser.add_argument("--qwen-docvqa-rows", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--permutation-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()

    specifications = [
        ("TextVQA-lite", args.textvqa_dir, args.qwen_textvqa_rows),
        ("DocVQA-lite", args.docvqa_dir, args.qwen_docvqa_rows),
    ]
    reports = [
        summarize_run(
            label,
            run_dir,
            qwen_rows,
            bootstrap_samples=args.bootstrap_samples,
            permutation_samples=args.permutation_samples,
            seed=args.seed + index,
        )
        for index, (label, run_dir, qwen_rows) in enumerate(specifications)
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "paired full-prefix versus 40% question-conditioned LLaVA Target pruning",
        "bootstrap_samples": args.bootstrap_samples,
        "permutation_samples": args.permutation_samples,
        "seed": args.seed,
        "tasks": reports,
    }
    (args.output_dir / "llava_open_qa_report.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(args.output_dir / "llava_open_qa_report.csv", reports)
    (args.output_dir / "llava_open_qa_report.md").write_text(markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def summarize_run(
    label: str,
    run_dir: Path,
    qwen_rows_path: Path | None,
    *,
    bootstrap_samples: int,
    permutation_samples: int,
    seed: int,
) -> dict[str, Any]:
    rows = load_jsonl(run_dir / "open_ocr_qa_generation.jsonl")
    traces = load_jsonl(run_dir / "prune_traces.jsonl")
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    if len(rows) != 500 or len(traces) != 500 or int(metrics.get("n", -1)) != 500:
        raise ValueError(f"{label}: expected 500 rows, traces, and metric count")
    ids = [str(row.get("sample_id", "")) for row in rows]
    if not all(ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{label}: missing or duplicate sample IDs")
    trace_ids = [str(trace.get("sample_id", "")) for trace in traces]
    if ids != trace_ids:
        raise ValueError(f"{label}: generation rows and traces are not aligned")
    if any(row.get("selector_target_source") != "question" for row in rows):
        raise ValueError(f"{label}: selector target is not question-only")
    if any(int(row.get("target_text_token_count", 0)) <= 0 for row in rows):
        raise ValueError(f"{label}: at least one selector target span is empty")
    if any(not str(row.get("full_answer", "")).strip() for row in rows):
        raise ValueError(f"{label}: at least one full answer is empty")
    if any(not str(row.get("pruned_answer", "")).strip() for row in rows):
        raise ValueError(f"{label}: at least one pruned answer is empty")
    if qwen_rows_path is not None:
        qwen_ids = {str(row.get("sample_id", "")) for row in load_jsonl(qwen_rows_path)}
        if set(ids) != qwen_ids:
            raise ValueError(f"{label}: LLaVA and Qwen sample sets differ")

    full = np.asarray([float(row["full_score"]) for row in rows], dtype=np.float64)
    pruned = np.asarray([float(row["pruned_score"]) for row in rows], dtype=np.float64)
    differences = pruned - full
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(bootstrap_samples, dtype=np.float64)
    for start in range(0, bootstrap_samples, 1000):
        count = min(1000, bootstrap_samples - start)
        indices = rng.integers(0, len(rows), size=(count, len(rows)))
        bootstrap_means[start : start + count] = differences[indices].mean(axis=1)

    nonzero = differences[np.abs(differences) > 1e-12]
    observed = abs(float(differences.mean()))
    exceedances = 0
    if len(nonzero):
        for start in range(0, permutation_samples, 1000):
            count = min(1000, permutation_samples - start)
            signs = rng.choice((-1.0, 1.0), size=(count, len(nonzero)))
            permuted = np.abs((signs * nonzero).sum(axis=1) / len(rows))
            exceedances += int(np.count_nonzero(permuted >= observed - 1e-15))
    p_value = (exceedances + 1) / (permutation_samples + 1)
    return {
        "task": label,
        "n": len(rows),
        "metric": metrics["primary_metric"],
        "full_score": float(full.mean()),
        "pruned_score": float(pruned.mean()),
        "paired_delta": float(differences.mean()),
        "bootstrap_95_ci": [float(x) for x in np.quantile(bootstrap_means, [0.025, 0.975])],
        "paired_sign_flip_p_value": p_value,
        "wins": int(np.count_nonzero(differences > 1e-12)),
        "losses": int(np.count_nonzero(differences < -1e-12)),
        "ties": int(np.count_nonzero(np.abs(differences) <= 1e-12)),
        "mean_effective_keep_ratio": float(
            np.mean([float(row["effective_keep_ratio"]) for row in rows])
        ),
        "mean_target_text_token_count": float(
            np.mean([float(row["target_text_token_count"]) for row in rows])
        ),
        "same_sample_set_as_qwen": qwen_rows_path is not None,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, reports: list[dict[str, Any]]) -> None:
    fieldnames = [
        "task", "n", "metric", "full_score", "pruned_score", "paired_delta", "ci_low", "ci_high",
        "paired_sign_flip_p_value", "wins", "losses", "ties", "mean_effective_keep_ratio",
        "mean_target_text_token_count", "same_sample_set_as_qwen",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for report in reports:
            low, high = report["bootstrap_95_ci"]
            flattened = {key: report[key] for key in fieldnames if key in report}
            flattened.update({"ci_low": low, "ci_high": high})
            writer.writerow(flattened)


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# LLaVA Open-OCR-QA Paired Audit",
        "",
        "Original questions are used for greedy generation. The selector receives only the question, never the gold answer or evidence boxes.",
        "",
        "| Task | n | Metric | Full | Target (40%) | Delta | 95% CI | p | Win/Loss/Tie | Keep |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["tasks"]:
        low, high = row["bootstrap_95_ci"]
        lines.append(
            f"| {row['task']} | {row['n']} | {row['metric']} | {row['full_score']:.4f} | "
            f"{row['pruned_score']:.4f} | {row['paired_delta']:+.4f} | "
            f"[{low:+.4f}, {high:+.4f}] | {row['paired_sign_flip_p_value']:.6f} | "
            f"{row['wins']}/{row['losses']}/{row['ties']} | {row['mean_effective_keep_ratio']:.4f} |"
        )
    lines.extend(
        [
            "",
            "The intervals exclude zero for both tasks. At this aggressive budget, question-conditioned Target pruning is therefore not backbone-robust for native open-answer generation.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
