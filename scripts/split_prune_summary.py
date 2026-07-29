#!/usr/bin/env python
"""Summarize pruning runs by the policy dev/test split.

The benchmark records are all from the dataset test split, but learned pruning
policies use a stable hash split over sample ids for policy selection. This
script reports cached run metrics over that policy split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="runs/prune_main")
    parser.add_argument("--out-csv", default="runs/prune_main/policy_split_summary.csv")
    parser.add_argument("--out-md", default="runs/prune_main/policy_split_summary.md")
    parser.add_argument("--dev-buckets", type=int, default=5)
    parser.add_argument("--run", action="append", required=True, help="Run name or path to summarize.")
    args = parser.parse_args()

    rows = []
    runs_dir = Path(args.runs_dir)
    for run in args.run:
        rows.extend(summarize_run(resolve_run(runs_dir, run), run_name=Path(run).name, dev_buckets=args.dev_buckets))

    write_csv(Path(args.out_csv), rows)
    write_markdown(Path(args.out_md), rows, dev_buckets=args.dev_buckets)
    print(f"Wrote {len(rows)} split rows to {args.out_csv} and {args.out_md}")


def resolve_run(runs_dir: Path, run: str) -> Path:
    path = Path(run)
    if path.exists():
        return path
    return runs_dir / run


def summarize_run(run_dir: Path, *, run_name: str, dev_buckets: int) -> list[dict[str, Any]]:
    scores_path = run_dir / "sample_scores.jsonl"
    traces_path = run_dir / "prune_traces.jsonl"
    if not scores_path.exists():
        raise FileNotFoundError(f"Missing sample_scores.jsonl in {run_dir}")
    if not traces_path.exists():
        raise FileNotFoundError(f"Missing prune_traces.jsonl in {run_dir}")

    traces = {str(row["sample_id"]): row for row in read_jsonl(traces_path)}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for score in read_jsonl(scores_path):
        sample_id = str(score["sample_id"])
        trace = traces.get(sample_id, {})
        row = {
            "sample_id": sample_id,
            "direct_correct": bool(score.get("direct_correct", False)),
            "hallucination": bool(score.get("hallucination", 0.0)),
            "target_is_negative": bool(score.get("target_is_negative", False)),
            "keep": keep_ratio(trace),
            "ecr": as_float(trace.get("ecr")),
            "lang_ms": as_float(trace.get("mean_language_ms", trace.get("language_ms"))),
        }
        groups["all"].append(row)
        groups[policy_split(sample_id, dev_buckets=dev_buckets)].append(row)

    out = []
    for split in ("all", "dev", "test"):
        split_rows = groups.get(split, [])
        out.append(
            {
                "run": run_name,
                "policy_split": split,
                "n": len(split_rows),
                "n_negative": sum(1 for row in split_rows if row["target_is_negative"]),
                "acc": mean([1.0 if row["direct_correct"] else 0.0 for row in split_rows]),
                "hFPR": hallucination_fpr(split_rows),
                "keep": mean([row["keep"] for row in split_rows if row["keep"] is not None]),
                "ECR": mean([row["ecr"] for row in split_rows if row["ecr"] is not None]),
                "lang_ms": mean([row["lang_ms"] for row in split_rows if row["lang_ms"] is not None]),
            }
        )
    return out


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def policy_split(sample_id: str, *, dev_buckets: int) -> str:
    bucket = int(hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:8], 16) % 10
    return "dev" if bucket < dev_buckets else "test"


def keep_ratio(trace: dict[str, Any]) -> float | None:
    kept = trace.get("kept_visual_tokens")
    total = trace.get("full_visual_tokens")
    if isinstance(kept, (int, float)) and isinstance(total, (int, float)) and total:
        return float(kept) / float(total)
    return None


def as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def hallucination_fpr(rows: list[dict[str, Any]]) -> float:
    negatives = [row for row in rows if row["target_is_negative"]]
    return mean([1.0 if row["hallucination"] else 0.0 for row in negatives])


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run", "policy_split", "n", "n_negative", "acc", "hFPR", "keep", "ECR", "lang_ms"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]], *, dev_buckets: int) -> None:
    lines = [
        "# Policy split pruning summary",
        "",
        f"Split rule: SHA1(sample_id) modulo 10, dev buckets < {dev_buckets}.",
        "",
        "| run | split | n | acc | hFPR | keep | ECR | lang_ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['run']} | {row['policy_split']} | {row['n']} | {row['acc']:.4f} | "
            f"{row['hFPR']:.4f} | {row['keep']:.4f} | {row['ECR']:.4f} | {row['lang_ms']:.2f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
