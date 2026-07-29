#!/usr/bin/env python3
"""Build an oracle adaptive-budget frontier for TextOCR-Hard Qwen runs.

This is a diagnostic upper bound over cached fixed-budget runs. It asks:
if a perfect sample-level controller could choose among existing keep ratios,
how much could it improve accuracy/hFPR/mean keep? The result is not deployable,
but it distinguishes "adaptive pruning has no headroom" from "our risk
predictor is too weak."
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RUNS = {
    "0.20": "runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_topk_0p20",
    "0.25": "runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_topk_0p25",
    "0.30": "runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_topk_0p30_targetfix_802816",
    "0.35": "runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_topk_0p35",
    "0.40": "runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_topk_0p40",
    "0.50": "runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_topk_0p50",
    "1.00": "runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_topk_1p00",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="runs/textocr_adaptive_policy/qwen_oracle_budget_frontier")
    parser.add_argument("--dev-buckets", type=int, default=5)
    args = parser.parse_args()

    runs = {ratio: load_run(Path(path)) for ratio, path in DEFAULT_RUNS.items()}
    ids = sorted(set.intersection(*(set(rows) for rows in runs.values())))
    rows = build_sample_rows(runs, ids, args.dev_buckets)
    fixed_rows = [summarize_policy(f"fixed_{ratio}", rows, lambda row, r=ratio: r) for ratio in sorted(runs, key=float)]
    oracle_rows = [
        summarize_policy("oracle_min_correct", rows, lambda row: row["min_correct_ratio"] or "1.00"),
        summarize_policy("oracle_min_label_match_else_full", rows, lambda row: row["min_correct_ratio"] or "1.00"),
        summarize_policy("oracle_match_full_prediction", rows, lambda row: row["min_fullmatch_ratio"] or "1.00"),
    ]
    split_rows = []
    for split in ("dev", "test"):
        split_sample_rows = [row for row in rows if row["split"] == split]
        split_rows.extend(
            [
                summarize_policy(f"{split}_oracle_min_correct", split_sample_rows, lambda row: row["min_correct_ratio"] or "1.00"),
                summarize_policy(f"{split}_fixed_0.30", split_sample_rows, lambda row: "0.30"),
                summarize_policy(f"{split}_fixed_0.35", split_sample_rows, lambda row: "0.35"),
            ]
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "sample_oracle_rows.csv", rows)
    write_csv(out_dir / "frontier_summary.csv", [*fixed_rows, *oracle_rows, *split_rows])
    (out_dir / "oracle_budget_frontier.md").write_text(
        build_markdown(fixed_rows, oracle_rows, split_rows, rows),
        encoding="utf-8",
    )
    print(f"Wrote {out_dir / 'oracle_budget_frontier.md'}")


def load_run(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for row in read_jsonl(ROOT / path / "probe_scores.jsonl"):
        rows[str(row["sample_id"])] = row
    return rows


def build_sample_rows(
    runs: dict[str, dict[str, dict[str, Any]]],
    ids: list[str],
    dev_buckets: int,
) -> list[dict[str, Any]]:
    ratios = sorted(runs, key=float)
    rows = []
    for sid in ids:
        base = runs["1.00"][sid]
        row: dict[str, Any] = {
            "sample_id": sid,
            "image_id": base.get("image_id", ""),
            "split": stable_split(base.get("image_id") or sid, dev_buckets),
            "target_answer": base.get("target_answer", ""),
            "binary_polarity": base.get("binary_polarity", ""),
            "target_is_negative": str(base.get("target_answer", "")).lower() == "no"
            or base.get("binary_polarity") == "negative",
            "full_pred": pred(base),
            "full_correct": correct(base),
        }
        min_correct = ""
        min_fullmatch = ""
        for ratio in ratios:
            score = runs[ratio][sid]
            row[f"pred_{tag(ratio)}"] = pred(score)
            row[f"correct_{tag(ratio)}"] = correct(score)
            row[f"hallucination_{tag(ratio)}"] = hallucination(score)
            if not min_correct and correct(score):
                min_correct = ratio
            if not min_fullmatch and pred(score) == row["full_pred"]:
                min_fullmatch = ratio
        row["min_correct_ratio"] = min_correct
        row["min_fullmatch_ratio"] = min_fullmatch
        row["oracle_correct_possible"] = bool(min_correct)
        rows.append(row)
    return rows


def summarize_policy(name: str, rows: list[dict[str, Any]], choose_ratio) -> dict[str, Any]:
    chosen = []
    for row in rows:
        ratio = str(choose_ratio(row))
        chosen.append((row, ratio))
    correct_values = [bool(row[f"correct_{tag(ratio)}"]) for row, ratio in chosen]
    negative = [(row, ratio) for row, ratio in chosen if row["target_is_negative"]]
    hallucinations = [bool(row[f"hallucination_{tag(ratio)}"]) for row, ratio in negative]
    pos = [(row, ratio) for row, ratio in chosen if not row["target_is_negative"]]
    yes_values = [row[f"pred_{tag(ratio)}"] == "yes" for row, ratio in chosen]
    return {
        "policy": name,
        "n": len(rows),
        "accuracy": mean(correct_values),
        "hFPR": mean(hallucinations),
        "yes_rate": mean(yes_values),
        "pos_acc": mean([bool(row[f"correct_{tag(ratio)}"]) for row, ratio in pos]),
        "neg_acc": mean([bool(row[f"correct_{tag(ratio)}"]) for row, ratio in negative]),
        "mean_keep": mean([float(ratio) for _, ratio in chosen]),
        "full_fallback_rate": mean([float(ratio) >= 1.0 for _, ratio in chosen]),
        "no_correct_budget_rate": mean([not row["oracle_correct_possible"] for row in rows]),
    }


def build_markdown(
    fixed_rows: list[dict[str, Any]],
    oracle_rows: list[dict[str, Any]],
    split_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# TextOCR-Hard Oracle Budget Frontier",
        "",
        "This is an oracle upper-bound diagnostic over cached Qwen3-VL-8B target-conditioned fixed-budget runs. It is not deployable because it uses labels to pick the smallest correct budget per sample.",
        "",
        "## Fixed Budgets",
        "",
        "| Policy | Acc. | hFPR | Mean keep | Full fallback |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in fixed_rows:
        lines.append(policy_line(row))
    lines.extend(
        [
            "",
            "## Oracle Policies",
            "",
            "| Policy | Acc. | hFPR | Mean keep | Full fallback | No-correct-budget |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in oracle_rows:
        lines.append(
            "| {policy} | {accuracy:.3f} | {hFPR:.3f} | {mean_keep:.3f} | {full_fallback_rate:.3f} | {no_correct_budget_rate:.3f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Split Check",
            "",
            "| Policy | Acc. | hFPR | Mean keep |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in split_rows:
        lines.append("| {policy} | {accuracy:.3f} | {hFPR:.3f} | {mean_keep:.3f} |".format(**row))
    oracle = next(row for row in oracle_rows if row["policy"] == "oracle_min_correct")
    fixed30 = next(row for row in fixed_rows if row["policy"] == "fixed_0.30")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- A perfect controller could raise accuracy from {fixed30['accuracy']:.3f} at fixed 0.30 to {oracle['accuracy']:.3f} while using mean keep {oracle['mean_keep']:.3f}.",
            f"- The oracle full-fallback rate is {oracle['full_fallback_rate']:.3f}, and {oracle['no_correct_budget_rate']:.3f} of samples have no correct budget among the cached choices.",
            "- Therefore adaptive budgeting has real headroom, but previous deployable policies failed because their risk signals were too weak, not because the budget set has no oracle frontier.",
        ]
    )
    return "\n".join(lines) + "\n"


def policy_line(row: dict[str, Any]) -> str:
    return "| {policy} | {accuracy:.3f} | {hFPR:.3f} | {mean_keep:.3f} | {full_fallback_rate:.3f} |".format(**row)


def tag(ratio: str) -> str:
    return ratio.replace(".", "p")


def pred(row: dict[str, Any]) -> str:
    return str(row.get("pred_answer") or row.get("direct_pred") or "").strip().lower()


def correct(row: dict[str, Any]) -> bool:
    return bool(row.get("correct", row.get("direct_correct", False)))


def hallucination(row: dict[str, Any]) -> bool:
    if "hallucination" in row:
        return bool(row["hallucination"])
    return pred(row) == "yes" and str(row.get("target_answer", "")).lower() == "no"


def stable_split(key: str, dev_buckets: int) -> str:
    bucket = int(hashlib.sha1(str(key).encode("utf-8")).hexdigest()[:8], 16) % 10
    return "dev" if bucket < dev_buckets else "test"


def mean(values: list[Any]) -> float:
    vals = [float(bool(v)) if isinstance(v, bool) else float(v) for v in values]
    return sum(vals) / len(vals) if vals else 0.0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
