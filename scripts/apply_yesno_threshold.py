#!/usr/bin/env python
"""Apply a calibrated yes/no threshold to cached probe scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recap.aggregate import aggregate_scores
from recap.io import write_json, write_jsonl
from recap.scoring import bool_to_answer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score", required=True, help="Input probe_scores.jsonl.")
    parser.add_argument("--out-dir", required=True, help="Output run directory.")
    parser.add_argument("--threshold", type=float, default=None, help="Threshold to apply. If omitted, selected from --threshold-split.")
    parser.add_argument("--threshold-split", default="dev", choices=["dev", "all"], help="Rows used to choose the threshold when --threshold is omitted.")
    parser.add_argument("--eval-split", default="all", choices=["dev", "test", "all"], help="Rows to write/evaluate.")
    parser.add_argument("--dev-buckets", type=int, default=5)
    args = parser.parse_args()

    rows = load_rows(Path(args.score), dev_buckets=args.dev_buckets)
    threshold = args.threshold
    if threshold is None:
        threshold_rows = rows if args.threshold_split == "all" else [row for row in rows if row["_split"] == "dev"]
        threshold = best_threshold(threshold_rows)

    eval_rows = rows if args.eval_split == "all" else [row for row in rows if row["_split"] == args.eval_split]
    calibrated = [apply_threshold(row, threshold=threshold) for row in eval_rows]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "probe_scores.jsonl", calibrated)
    result = aggregate_scores(calibrated)
    metrics = result["metrics"]
    source_metrics = load_source_metrics(Path(args.score))
    if source_metrics:
        for key in (
            "pruning",
            "prune_selector",
            "prune_score_source",
            "prune_position_mode",
            "prune_budget_mode",
            "prune_target_keep_ratio",
            "prune_min_patches",
            "prune_max_patches",
            "prune_kept_indices_file",
            "prune_kept_indices_key",
        ):
            if key in source_metrics:
                metrics[key] = source_metrics[key]
    if calibrated and calibrated[0].get("prune_budget_mode"):
        metrics["prune_budget_mode"] = calibrated[0]["prune_budget_mode"]
    metrics["yesno_threshold"] = float(threshold)
    metrics["yesno_threshold_source"] = "provided" if args.threshold is not None else args.threshold_split
    metrics["yesno_eval_split"] = args.eval_split
    metrics["yesno_input_score"] = str(args.score)
    write_json(out_dir / "metrics.json", metrics)
    write_jsonl(out_dir / "sample_scores.jsonl", result["samples"])
    write_jsonl(out_dir / "probes.jsonl", calibrated)
    print(
        f"Wrote calibrated {args.eval_split} run with {len(calibrated)} rows, "
        f"threshold={threshold:.6f}, acc={metrics['direct_accuracy']:.4f}, "
        f"hFPR={metrics['direct_hallucination_fpr']:.4f} to {out_dir}"
    )


def load_rows(path: Path, *, dev_buckets: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            group_id = str(row.get("image_id") or image_group_from_sample_id(str(row.get("sample_id", ""))))
            row["_split"] = stable_split(group_id, dev_buckets=dev_buckets)
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows loaded from {path}")
    return rows


def load_source_metrics(score_path: Path) -> dict[str, Any]:
    metrics_path = score_path.parent / "metrics.json"
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def image_group_from_sample_id(sample_id: str) -> str:
    return sample_id.split(":", 1)[0] if ":" in sample_id else sample_id


def stable_split(value: str, *, dev_buckets: int) -> str:
    bucket = int(hashlib.sha1(value.encode("utf-8")).hexdigest()[:8], 16) % 10
    return "dev" if bucket < dev_buckets else "test"


def best_threshold(rows: list[dict[str, Any]]) -> float:
    margins = sorted({raw_margin(row) for row in rows})
    if not margins:
        return 0.0
    candidates = [margins[0] - 1.0]
    candidates.extend((left + right) / 2.0 for left, right in zip(margins, margins[1:]))
    candidates.append(margins[-1] + 1.0)
    best_key: tuple[float, float, float] | None = None
    best_threshold_value = 0.0
    for threshold in candidates:
        metrics = evaluate(rows, threshold=threshold)
        key = (metrics["acc"], -metrics["hFPR"], -abs(metrics["yes_rate"] - 0.5))
        if best_key is None or key > best_key:
            best_key = key
            best_threshold_value = float(threshold)
    return best_threshold_value


def evaluate(rows: list[dict[str, Any]], *, threshold: float) -> dict[str, float]:
    if not rows:
        return {"acc": 0.0, "hFPR": 0.0, "yes_rate": 0.0}
    correct = 0
    yes_count = 0
    neg_total = 0
    false_yes = 0
    for row in rows:
        pred_yes = raw_margin(row) >= threshold
        target_yes = is_target_yes(row)
        correct += int(pred_yes == target_yes)
        yes_count += int(pred_yes)
        if not target_yes:
            neg_total += 1
            false_yes += int(pred_yes)
    return {
        "acc": correct / len(rows),
        "hFPR": false_yes / neg_total if neg_total else 0.0,
        "yes_rate": yes_count / len(rows),
    }


def apply_threshold(row: dict[str, Any], *, threshold: float) -> dict[str, Any]:
    out = {key: value for key, value in row.items() if key != "_split"}
    raw = raw_margin(row)
    calibrated_margin = raw - threshold
    pred_yes = calibrated_margin >= 0.0
    target_yes = is_target_yes(row)
    out["raw_margin"] = raw
    out["margin"] = calibrated_margin
    out["yesno_threshold"] = float(threshold)
    out["yesno_calibrated"] = True
    out["pred_answer"] = bool_to_answer(pred_yes)
    out["correct"] = pred_yes == target_yes
    out["support_pred"] = abs(calibrated_margin)
    return out


def raw_margin(row: dict[str, Any]) -> float:
    if "raw_margin" in row:
        return float(row["raw_margin"])
    if "yes_loss" in row and "no_loss" in row:
        return float(row["no_loss"]) - float(row["yes_loss"])
    return float(row["margin"])


def is_target_yes(row: dict[str, Any]) -> bool:
    return str(row.get("target_answer", row.get("answer", ""))).lower() == "yes"


if __name__ == "__main__":
    main()
