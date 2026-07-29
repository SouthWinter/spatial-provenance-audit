#!/usr/bin/env python
"""Paired significance summaries for pruning runs.

The script compares cached ``sample_scores.jsonl`` files. It uses paired
bootstrap over sample ids for accuracy and hallucination FPR differences, plus
exact McNemar tests for direct correctness and negative-sample false positives.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="runs/prune_main")
    parser.add_argument("--out-csv", default="runs/prune_main/paired_stats.csv")
    parser.add_argument("--out-md", default="runs/prune_main/paired_stats.md")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--comparison",
        action="append",
        required=True,
        help="Comparison in label:left_run:right_run form. Diff is left minus right.",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    rows = []
    for spec in args.comparison:
        label, left_run, right_run = parse_comparison(spec)
        left_scores = load_run_scores(resolve_run(runs_dir, left_run))
        right_scores = load_run_scores(resolve_run(runs_dir, right_run))
        rows.append(
            compare_runs(
                label=label,
                left_name=left_run,
                right_name=right_run,
                left=left_scores,
                right=right_scores,
                n_bootstrap=args.bootstrap,
                seed=args.seed + len(rows),
            )
        )

    write_csv(Path(args.out_csv), rows)
    write_markdown(Path(args.out_md), rows, n_bootstrap=args.bootstrap, seed=args.seed)
    print(f"Wrote {len(rows)} paired comparisons to {args.out_csv} and {args.out_md}")


def parse_comparison(spec: str) -> tuple[str, str, str]:
    parts = spec.split(":")
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"Invalid comparison {spec!r}; expected label:left_run:right_run")
    return parts[0], parts[1], parts[2]


def resolve_run(runs_dir: Path, run: str) -> Path:
    path = Path(run)
    if path.exists():
        return path
    return runs_dir / run


def load_run_scores(run_dir: Path) -> dict[str, dict[str, Any]]:
    path = run_dir / "sample_scores.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing sample_scores.jsonl in {run_dir}")
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row["sample_id"])
            rows[sample_id] = {
                "direct_correct": bool(row.get("direct_correct", False)),
                "hallucination": bool(row.get("hallucination", 0.0)),
                "target_is_negative": bool(row.get("target_is_negative", False)),
            }
    return rows


def compare_runs(
    *,
    label: str,
    left_name: str,
    right_name: str,
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    sample_ids = sorted(set(left) & set(right))
    if not sample_ids:
        raise ValueError(f"No overlapping sample ids for {left_name} vs {right_name}")
    missing_left = len(set(right) - set(left))
    missing_right = len(set(left) - set(right))
    if missing_left or missing_right:
        raise ValueError(
            f"Sample id mismatch for {left_name} vs {right_name}: "
            f"missing_left={missing_left}, missing_right={missing_right}"
        )

    pairs = [(left[sid], right[sid]) for sid in sample_ids]
    negative_pairs = [(l, r) for l, r in pairs if l["target_is_negative"] or r["target_is_negative"]]

    acc_left = mean([1.0 if l["direct_correct"] else 0.0 for l, _ in pairs])
    acc_right = mean([1.0 if r["direct_correct"] else 0.0 for _, r in pairs])
    hfpr_left = mean([1.0 if l["hallucination"] else 0.0 for l, _ in negative_pairs])
    hfpr_right = mean([1.0 if r["hallucination"] else 0.0 for _, r in negative_pairs])

    draws_acc: list[float] = []
    draws_hfpr: list[float] = []
    rng = random.Random(seed)
    for _ in range(n_bootstrap):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        draws_acc.append(
            mean([(1.0 if l["direct_correct"] else 0.0) - (1.0 if r["direct_correct"] else 0.0) for l, r in sample])
        )
        negative_sample = [(l, r) for l, r in sample if l["target_is_negative"] or r["target_is_negative"]]
        draws_hfpr.append(
            mean([(1.0 if l["hallucination"] else 0.0) - (1.0 if r["hallucination"] else 0.0) for l, r in negative_sample])
        )

    corr_table = discordance_table(pairs, "direct_correct")
    hall_table = discordance_table(negative_pairs, "hallucination")
    return {
        "label": label,
        "left": left_name,
        "right": right_name,
        "n": len(pairs),
        "n_negative": len(negative_pairs),
        "acc_left": acc_left,
        "acc_right": acc_right,
        "acc_diff": acc_left - acc_right,
        "acc_diff_ci_low": percentile(draws_acc, 0.025),
        "acc_diff_ci_high": percentile(draws_acc, 0.975),
        "acc_boot_p_two_sided": bootstrap_two_sided_p(draws_acc),
        "acc_mcnemar_b": corr_table["b"],
        "acc_mcnemar_c": corr_table["c"],
        "acc_mcnemar_p": exact_mcnemar_p(corr_table["b"], corr_table["c"]),
        "hfpr_left": hfpr_left,
        "hfpr_right": hfpr_right,
        "hfpr_diff": hfpr_left - hfpr_right,
        "hfpr_diff_ci_low": percentile(draws_hfpr, 0.025),
        "hfpr_diff_ci_high": percentile(draws_hfpr, 0.975),
        "hfpr_boot_p_two_sided": bootstrap_two_sided_p(draws_hfpr),
        "hfpr_mcnemar_b": hall_table["b"],
        "hfpr_mcnemar_c": hall_table["c"],
        "hfpr_mcnemar_p": exact_mcnemar_p(hall_table["b"], hall_table["c"]),
    }


def discordance_table(pairs: list[tuple[dict[str, Any], dict[str, Any]]], key: str) -> dict[str, int]:
    b = c = 0
    for left, right in pairs:
        left_value = bool(left[key])
        right_value = bool(right[key])
        if left_value and not right_value:
            b += 1
        elif right_value and not left_value:
            c += 1
    return {"b": b, "c": c}


def exact_mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2.0**n)
    return min(1.0, 2.0 * tail)


def bootstrap_two_sided_p(draws: list[float]) -> float:
    if not draws:
        return 1.0
    le_zero = sum(1 for value in draws if value <= 0.0) / len(draws)
    ge_zero = sum(1 for value in draws if value >= 0.0) / len(draws)
    return min(1.0, 2.0 * min(le_zero, ge_zero))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "left",
        "right",
        "n",
        "n_negative",
        "acc_left",
        "acc_right",
        "acc_diff",
        "acc_diff_ci_low",
        "acc_diff_ci_high",
        "acc_boot_p_two_sided",
        "acc_mcnemar_b",
        "acc_mcnemar_c",
        "acc_mcnemar_p",
        "hfpr_left",
        "hfpr_right",
        "hfpr_diff",
        "hfpr_diff_ci_low",
        "hfpr_diff_ci_high",
        "hfpr_boot_p_two_sided",
        "hfpr_mcnemar_b",
        "hfpr_mcnemar_c",
        "hfpr_mcnemar_p",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]], *, n_bootstrap: int, seed: int) -> None:
    lines = [
        "# Paired pruning statistics",
        "",
        f"Bootstrap samples: {n_bootstrap}",
        f"Seed: {seed}",
        "",
        "| comparison | acc left | acc right | acc diff | acc 95% CI | McNemar p | hFPR left | hFPR right | hFPR diff | hFPR 95% CI | hFPR McNemar p |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['label']} | "
            f"{row['acc_left']:.4f} | {row['acc_right']:.4f} | {row['acc_diff']:+.4f} | "
            f"[{row['acc_diff_ci_low']:+.4f}, {row['acc_diff_ci_high']:+.4f}] | "
            f"{row['acc_mcnemar_p']:.4g} | "
            f"{row['hfpr_left']:.4f} | {row['hfpr_right']:.4f} | {row['hfpr_diff']:+.4f} | "
            f"[{row['hfpr_diff_ci_low']:+.4f}, {row['hfpr_diff_ci_high']:+.4f}] | "
            f"{row['hfpr_mcnemar_p']:.4g} |"
        )
    lines.extend(
        [
            "",
            "Positive diff means the left run is higher than the right run. Lower hFPR is better.",
            "McNemar b/c counts are available in the CSV for exact discordance auditing.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
