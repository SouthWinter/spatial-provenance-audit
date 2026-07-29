#!/usr/bin/env python
"""Build an ECR-threshold visual-token budget policy from fixed-budget runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Callable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="runs/prune_main")
    parser.add_argument("--trace-run", required=True, help="Run whose prune_traces.jsonl provides ECR.")
    parser.add_argument(
        "--score-run",
        action="append",
        required=True,
        help="Budget label and run name, for example 0.30:qwen3_8b_..._0p30.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-budget", default="0.30")
    parser.add_argument("--mid-budget", default="0.40")
    parser.add_argument("--high-budget", default="0.50")
    parser.add_argument("--low-threshold-grid", default="0.40,0.50,0.60,0.70,0.75,0.80,0.85")
    parser.add_argument("--mid-threshold-grid", default="0.70,0.75,0.80,0.85,0.90,0.95,1.00")
    parser.add_argument("--dev-buckets", type=int, default=5)
    parser.add_argument("--hfpr-target", type=float, default=0.013636)
    parser.add_argument("--hfpr-penalty", type=float, default=0.5)
    parser.add_argument("--keep-target", type=float, default=0.40)
    parser.add_argument("--keep-penalty", type=float, default=0.02)
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    score_runs = parse_score_runs(args.score_run)
    traces = {sample_id(row): row for row in read_jsonl(runs_dir / args.trace_run / "prune_traces.jsonl")}
    scores = {
        budget: {sample_id(row): row for row in read_jsonl(runs_dir / run_name / "sample_scores.jsonl")}
        for budget, run_name in score_runs.items()
    }
    required = {args.base_budget, args.mid_budget, args.high_budget}
    missing = required.difference(scores)
    if missing:
        raise ValueError(f"Missing required score budgets: {sorted(missing)}")

    ids = sorted(set(traces).intersection(*(set(rows) for rows in scores.values())))
    if not ids:
        raise ValueError("No overlapping sample ids found across traces and score runs.")

    low_grid = parse_float_grid(args.low_threshold_grid)
    mid_grid = parse_float_grid(args.mid_threshold_grid)
    candidates = []
    for low_t in low_grid:
        for mid_t in mid_grid:
            if low_t > mid_t:
                continue
            policy = make_policy(
                traces,
                low_threshold=low_t,
                mid_threshold=mid_t,
                base_budget=args.base_budget,
                mid_budget=args.mid_budget,
                high_budget=args.high_budget,
            )
            dev = evaluate(ids, scores, policy, split_name="dev", dev_buckets=args.dev_buckets)
            test = evaluate(ids, scores, policy, split_name="test", dev_buckets=args.dev_buckets)
            all_metrics = evaluate(ids, scores, policy, split_name="all", dev_buckets=args.dev_buckets)
            objective = (
                dev["accuracy"]
                - args.hfpr_penalty * max(0.0, dev["hallucination_fpr"] - args.hfpr_target)
                - args.keep_penalty * max(0.0, dev["mean_keep_ratio"] - args.keep_target)
            )
            candidates.append(
                {
                    "low_threshold": low_t,
                    "mid_threshold": mid_t,
                    "objective": objective,
                    **prefixed("dev", dev),
                    **prefixed("test", test),
                    **prefixed("all", all_metrics),
                }
            )
    candidates.sort(
        key=lambda row: (
            row["objective"],
            -abs(row["dev_mean_keep_ratio"] - args.keep_target),
            row["dev_accuracy"],
            -row["dev_hallucination_fpr"],
        ),
        reverse=True,
    )
    best = candidates[0]

    policy_config = {
        "policy_type": "ecr_two_threshold",
        "trace_run": args.trace_run,
        "score_runs": score_runs,
        "base_budget": args.base_budget,
        "mid_budget": args.mid_budget,
        "high_budget": args.high_budget,
        "low_threshold": best["low_threshold"],
        "mid_threshold": best["mid_threshold"],
        "dev_buckets": args.dev_buckets,
        "hfpr_target": args.hfpr_target,
        "hfpr_penalty": args.hfpr_penalty,
        "keep_target": args.keep_target,
        "keep_penalty": args.keep_penalty,
    }
    write_json(out_dir / "policy_config.json", policy_config)
    write_csv(out_dir / "policy_candidates.csv", candidates)

    best_policy = make_policy(
        traces,
        low_threshold=float(best["low_threshold"]),
        mid_threshold=float(best["mid_threshold"]),
        base_budget=args.base_budget,
        mid_budget=args.mid_budget,
        high_budget=args.high_budget,
    )
    budget_rows = []
    for sid in ids:
        budget = best_policy(sid)
        budget_rows.append(
            {
                "sample_id": sid,
                "keep_ratio": float(budget),
                "ecr": float(traces[sid].get("ecr", 0.0)),
                "split": split_for_id(sid, args.dev_buckets),
            }
        )
    write_jsonl(out_dir / "budget_ratios.jsonl", budget_rows)
    write_report(out_dir / "policy_report.md", policy_config, candidates[:20], len(ids))

    print(f"Wrote policy to {out_dir / 'policy_config.json'}")
    print(f"Wrote budgets to {out_dir / 'budget_ratios.jsonl'}")
    print(
        "Best ECR policy "
        f"low={best['low_threshold']:.3f} mid={best['mid_threshold']:.3f} "
        f"all_acc={best['all_accuracy']:.4f} all_hfpr={best['all_hallucination_fpr']:.4f} "
        f"all_keep={best['all_mean_keep_ratio']:.4f}"
    )


def parse_score_runs(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if ":" not in item:
            raise ValueError(f"--score-run must be BUDGET:RUN_NAME, got {item!r}")
        budget, run_name = item.split(":", 1)
        out[budget.strip()] = run_name.strip()
    return out


def parse_float_grid(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, config: dict[str, Any], candidates: list[dict[str, Any]], n: int) -> None:
    lines = [
        "# ECR Budget Policy",
        "",
        f"Samples: {n}",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(config, indent=2, ensure_ascii=True),
        "```",
        "",
        "## Top Candidates",
        "",
        "| low t | mid t | dev acc | dev hFPR | dev keep | test acc | test hFPR | test keep | all acc | all hFPR | all keep |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in candidates:
        lines.append(
            "| "
            f"{row['low_threshold']:.3f} | {row['mid_threshold']:.3f} | "
            f"{row['dev_accuracy']:.4f} | {row['dev_hallucination_fpr']:.4f} | {row['dev_mean_keep_ratio']:.4f} | "
            f"{row['test_accuracy']:.4f} | {row['test_hallucination_fpr']:.4f} | {row['test_mean_keep_ratio']:.4f} | "
            f"{row['all_accuracy']:.4f} | {row['all_hallucination_fpr']:.4f} | {row['all_mean_keep_ratio']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def sample_id(row: dict[str, Any]) -> str:
    return str(row.get("sample_id", row.get("id", "")))


def split_for_id(sid: str, dev_buckets: int) -> str:
    bucket = int(hashlib.md5(sid.encode("utf-8")).hexdigest()[:8], 16) % 10
    return "dev" if bucket < dev_buckets else "test"


def make_policy(
    traces: dict[str, dict[str, Any]],
    *,
    low_threshold: float,
    mid_threshold: float,
    base_budget: str,
    mid_budget: str,
    high_budget: str,
) -> Callable[[str], str]:
    def policy(sid: str) -> str:
        ecr = float(traces[sid].get("ecr", 0.0))
        if ecr < low_threshold:
            return high_budget
        if ecr < mid_threshold:
            return mid_budget
        return base_budget

    return policy


def evaluate(
    ids: list[str],
    scores: dict[str, dict[str, dict[str, Any]]],
    policy: Callable[[str], str],
    *,
    split_name: str,
    dev_buckets: int,
) -> dict[str, float]:
    selected = [sid for sid in ids if split_name == "all" or split_for_id(sid, dev_buckets) == split_name]
    if not selected:
        return {"n": 0.0, "accuracy": 0.0, "hallucination_fpr": 0.0, "mean_keep_ratio": 0.0}
    correct = 0
    negative = 0
    hallucinated_negative = 0
    keep_total = 0.0
    for sid in selected:
        budget = policy(sid)
        row = scores[budget][sid]
        correct += int(bool(row.get("direct_correct", False)))
        is_negative = bool(row.get("target_is_negative", False))
        negative += int(is_negative)
        hallucinated_negative += int(is_negative and bool(row.get("hallucination", False)))
        keep_total += 1.0 if budget == "full" else float(budget)
    return {
        "n": float(len(selected)),
        "accuracy": correct / float(len(selected)),
        "hallucination_fpr": hallucinated_negative / float(negative) if negative else 0.0,
        "mean_keep_ratio": keep_total / float(len(selected)),
    }


def prefixed(prefix: str, row: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in row.items()}


if __name__ == "__main__":
    main()
