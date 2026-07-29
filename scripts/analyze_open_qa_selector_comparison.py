#!/usr/bin/env python
"""Compare two matched open-QA pruning runs with paired uncertainty estimates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-rows", required=True)
    parser.add_argument("--candidate-rows", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--permutation-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    result = compare_runs(
        load_rows(Path(args.baseline_rows)),
        load_rows(Path(args.candidate_rows)),
        bootstrap_samples=args.bootstrap_samples,
        permutation_samples=args.permutation_samples,
        seed=args.seed,
    )
    result.update(
        {
            "baseline_rows": args.baseline_rows,
            "candidate_rows": args.candidate_rows,
            "bootstrap_samples": args.bootstrap_samples,
            "permutation_samples": args.permutation_samples,
            "seed": args.seed,
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(markdown_report(result), encoding="utf-8")
    print(json.dumps(result, indent=2))


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compare_runs(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    permutation_samples: int,
    seed: int,
) -> dict[str, Any]:
    baseline = index_rows(baseline_rows, label="baseline")
    candidate = index_rows(candidate_rows, label="candidate")
    if baseline.keys() != candidate.keys():
        missing_candidate = sorted(baseline.keys() - candidate.keys())
        missing_baseline = sorted(candidate.keys() - baseline.keys())
        raise ValueError(
            "Runs contain different sample IDs: "
            f"missing_candidate={missing_candidate[:5]}, missing_baseline={missing_baseline[:5]}"
        )

    sample_ids = [str(row["sample_id"]) for row in baseline_rows]
    for sample_id in sample_ids:
        left = baseline[sample_id]
        right = candidate[sample_id]
        for field in ("dataset", "question_id", "keep_ratio"):
            if left.get(field) != right.get(field):
                raise ValueError(f"Protocol mismatch for {sample_id}: field {field}")

    baseline_scores = np.asarray([float(baseline[s]["pruned_score"]) for s in sample_ids])
    candidate_scores = np.asarray([float(candidate[s]["pruned_score"]) for s in sample_ids])
    differences = candidate_scores - baseline_scores
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(bootstrap_samples, dtype=np.float64)
    for start in range(0, bootstrap_samples, 1000):
        count = min(1000, bootstrap_samples - start)
        indices = rng.integers(0, len(differences), size=(count, len(differences)))
        bootstrap_means[start : start + count] = differences[indices].mean(axis=1)

    nonzero = differences[np.abs(differences) > 1e-12]
    observed = abs(float(differences.mean()))
    exceedances = 0
    if len(nonzero):
        for start in range(0, permutation_samples, 1000):
            count = min(1000, permutation_samples - start)
            signs = rng.choice((-1.0, 1.0), size=(count, len(nonzero)))
            permuted = np.abs((signs * nonzero).sum(axis=1) / len(differences))
            exceedances += int(np.count_nonzero(permuted >= observed - 1e-15))
    permutation_p = (exceedances + 1) / (permutation_samples + 1)

    return {
        "n": len(sample_ids),
        "dataset": baseline_rows[0].get("dataset") if baseline_rows else None,
        "keep_ratio": baseline_rows[0].get("keep_ratio") if baseline_rows else None,
        "baseline_selector_target_source": baseline_rows[0].get("selector_target_source") if baseline_rows else None,
        "candidate_selector_target_source": candidate_rows[0].get("selector_target_source") if candidate_rows else None,
        "baseline_score": float(baseline_scores.mean()),
        "candidate_score": float(candidate_scores.mean()),
        "mean_paired_difference": float(differences.mean()),
        "bootstrap_95_ci": [float(value) for value in np.quantile(bootstrap_means, [0.025, 0.975])],
        "paired_sign_flip_p_value": permutation_p,
        "wins": int(np.count_nonzero(differences > 1e-12)),
        "losses": int(np.count_nonzero(differences < -1e-12)),
        "ties": int(np.count_nonzero(np.abs(differences) <= 1e-12)),
    }


def index_rows(rows: list[dict[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id:
            raise ValueError(f"{label} row has no sample_id")
        if sample_id in indexed:
            raise ValueError(f"{label} contains duplicate sample_id: {sample_id}")
        indexed[sample_id] = row
    if not indexed:
        raise ValueError(f"{label} run is empty")
    return indexed


def markdown_report(result: dict[str, Any]) -> str:
    low, high = result["bootstrap_95_ci"]
    return (
        "# Open-QA Selector Comparison\n\n"
        "| Quantity | Value |\n"
        "|---|---:|\n"
        f"| Samples | {result['n']} |\n"
        f"| Keep ratio | {result['keep_ratio']:.2f} |\n"
        f"| Baseline score | {result['baseline_score']:.4f} |\n"
        f"| Candidate score | {result['candidate_score']:.4f} |\n"
        f"| Paired difference | {result['mean_paired_difference']:+.4f} |\n"
        f"| Bootstrap 95% CI | [{low:+.4f}, {high:+.4f}] |\n"
        f"| Sign-flip p-value | {result['paired_sign_flip_p_value']:.6f} |\n"
        f"| Wins / losses / ties | {result['wins']} / {result['losses']} / {result['ties']} |\n"
    )


if __name__ == "__main__":
    main()
