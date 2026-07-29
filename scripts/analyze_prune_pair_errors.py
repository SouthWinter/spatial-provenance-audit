#!/usr/bin/env python
"""Analyze paired wins/losses between two pruning runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="runs/prune_main")
    parser.add_argument("--left", required=True, help="Left run name or path, usually the proposed method.")
    parser.add_argument("--right", required=True, help="Right run name or path, usually the baseline.")
    parser.add_argument("--label", default="", help="Human-readable comparison label.")
    parser.add_argument("--out-prefix", required=True, help="Output path prefix without extension.")
    parser.add_argument("--top-examples", type=int, default=20)
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    left_dir = resolve_run(runs_dir, args.left)
    right_dir = resolve_run(runs_dir, args.right)
    label = args.label or f"{left_dir.name} vs {right_dir.name}"

    left = load_run(left_dir)
    right = load_run(right_dir)
    sample_ids = sorted(set(left["scores"]) & set(right["scores"]))
    if not sample_ids:
        raise ValueError(f"No overlapping sample ids for {left_dir} and {right_dir}")
    if set(left["scores"]) != set(right["scores"]):
        raise ValueError(
            "Sample id mismatch: "
            f"left_only={len(set(left['scores']) - set(right['scores']))}, "
            f"right_only={len(set(right['scores']) - set(left['scores']))}"
        )

    paired = [build_pair_row(sid, left, right) for sid in sample_ids]
    detail_rows = [row for row in paired if row["outcome"] in {"win", "loss"}]

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out_prefix.with_suffix(".details.csv"), detail_rows)
    write_csv(out_prefix.with_suffix(".by_relation.csv"), summarize_by(paired, "base_relation"))
    write_csv(out_prefix.with_suffix(".by_polarity.csv"), summarize_by(paired, "binary_polarity"))
    write_csv(out_prefix.with_suffix(".by_ecr_bin.csv"), summarize_by(paired, "left_ecr_bin"))
    write_markdown(
        out_prefix.with_suffix(".md"),
        label=label,
        left_name=left_dir.name,
        right_name=right_dir.name,
        paired=paired,
        top_examples=args.top_examples,
    )
    print(f"Wrote paired error analysis to {out_prefix}.md and CSV sidecars")


def resolve_run(runs_dir: Path, run: str) -> Path:
    path = Path(run)
    if path.exists():
        return path
    return runs_dir / run


def load_run(run_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        "scores": load_jsonl_by_id(run_dir / "sample_scores.jsonl", "sample_id"),
        "probes": load_jsonl_by_id(run_dir / "probes.jsonl", "sample_id"),
        "traces": load_jsonl_by_id(run_dir / "prune_traces.jsonl", "sample_id"),
    }


def load_jsonl_by_id(path: Path, key: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row[key])] = row
    return rows


def build_pair_row(sid: str, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    ls = left["scores"][sid]
    rs = right["scores"][sid]
    probe = left["probes"].get(sid) or right["probes"].get(sid) or {}
    lt = left["traces"].get(sid, {})
    rt = right["traces"].get(sid, {})

    left_correct = bool(ls.get("direct_correct", False))
    right_correct = bool(rs.get("direct_correct", False))
    if left_correct and not right_correct:
        outcome = "win"
    elif right_correct and not left_correct:
        outcome = "loss"
    elif left_correct and right_correct:
        outcome = "both_correct"
    else:
        outcome = "both_wrong"

    ecr = as_float(lt.get("ecr"))
    return {
        "sample_id": sid,
        "outcome": outcome,
        "base_relation": probe.get("base_relation", ls.get("base_relation", "")),
        "relation_family": ls.get("relation_family", ""),
        "binary_polarity": probe.get("binary_polarity", ""),
        "target": ls.get("direct_target", ""),
        "left_pred": ls.get("direct_pred", ""),
        "right_pred": rs.get("direct_pred", ""),
        "left_correct": left_correct,
        "right_correct": right_correct,
        "left_hallucination": bool(ls.get("hallucination", False)),
        "right_hallucination": bool(rs.get("hallucination", False)),
        "target_is_negative": bool(ls.get("target_is_negative", False)),
        "question": probe.get("question", ""),
        "image": probe.get("image", ""),
        "image_id": probe.get("image_id", ""),
        "subject": probe.get("subject", ""),
        "object": probe.get("object", ""),
        "bbox_source": probe.get("bbox_source", ""),
        "evidence_region_count": probe.get("evidence_region_count", ""),
        "left_ecr": ecr,
        "left_ecr_bin": ecr_bin(ecr),
        "left_full_visual_tokens": lt.get("full_visual_tokens", ""),
        "left_kept_visual_tokens": lt.get("kept_visual_tokens", ""),
        "right_full_visual_tokens": rt.get("full_visual_tokens", ""),
        "right_kept_visual_tokens": rt.get("kept_visual_tokens", ""),
        "left_support_orig": as_float(ls.get("support_orig")),
        "right_support_orig": as_float(rs.get("support_orig")),
        "left_confidence_risk": as_float(ls.get("confidence_risk")),
        "right_confidence_risk": as_float(rs.get("confidence_risk")),
    }


def as_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def ecr_bin(value: float) -> str:
    if value >= 0.999:
        return "1.00"
    if value >= 0.75:
        return "0.75-1.00"
    if value >= 0.50:
        return "0.50-0.75"
    if value > 0.0:
        return "0.00-0.50"
    return "0.00"


def summarize_by(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, ""))].append(row)

    out = []
    for value, group in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        wins = sum(1 for row in group if row["outcome"] == "win")
        losses = sum(1 for row in group if row["outcome"] == "loss")
        both_correct = sum(1 for row in group if row["outcome"] == "both_correct")
        both_wrong = sum(1 for row in group if row["outcome"] == "both_wrong")
        out.append(
            {
                key: value,
                "n": len(group),
                "wins": wins,
                "losses": losses,
                "net_wins": wins - losses,
                "both_correct": both_correct,
                "both_wrong": both_wrong,
                "left_acc": (wins + both_correct) / len(group),
                "right_acc": (losses + both_correct) / len(group),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path,
    *,
    label: str,
    left_name: str,
    right_name: str,
    paired: list[dict[str, Any]],
    top_examples: int,
) -> None:
    counts = Counter(row["outcome"] for row in paired)
    lines = [
        f"# Paired error analysis: {label}",
        "",
        f"Left: `{left_name}`",
        f"Right: `{right_name}`",
        "",
        "## Outcome counts",
        "",
        "| outcome | count |",
        "|---|---:|",
    ]
    for outcome in ["win", "loss", "both_correct", "both_wrong"]:
        lines.append(f"| {outcome} | {counts.get(outcome, 0)} |")
    lines.append("")
    lines.append(f"Net wins: {counts.get('win', 0) - counts.get('loss', 0)}")
    lines.append("")

    for title, key in [
        ("By Relation", "base_relation"),
        ("By Polarity", "binary_polarity"),
        ("By ECR Bin", "left_ecr_bin"),
    ]:
        lines.extend([f"## {title}", "", "| group | n | wins | losses | net | left acc | right acc |", "|---|---:|---:|---:|---:|---:|---:|"])
        for row in summarize_by(paired, key):
            lines.append(
                f"| {row[key]} | {row['n']} | {row['wins']} | {row['losses']} | "
                f"{row['net_wins']} | {row['left_acc']:.4f} | {row['right_acc']:.4f} |"
            )
        lines.append("")

    for outcome, title in [("win", "Example Wins"), ("loss", "Example Losses")]:
        rows = [row for row in paired if row["outcome"] == outcome][:top_examples]
        lines.extend([f"## {title}", "", "| sample | relation | target | left | right | ECR | question |", "|---|---|---|---|---|---:|---|"])
        for row in rows:
            question = str(row["question"]).replace("|", "/")
            lines.append(
                f"| {row['sample_id']} | {row['base_relation']} | {row['target']} | "
                f"{row['left_pred']} | {row['right_pred']} | {row['left_ecr']:.3f} | {question} |"
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
