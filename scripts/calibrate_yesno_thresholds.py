#!/usr/bin/env python
"""Calibrate yes/no likelihood thresholds for direct VLM probe scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recap.metrics import roc_auc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score", action="append", required=True, help="Path to probe_scores.jsonl. Can be repeated.")
    parser.add_argument("--name", action="append", default=[], help="Display name for each --score path.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--dev-buckets", type=int, default=5, help="Number of hash buckets out of 10 used for dev.")
    args = parser.parse_args()

    names = args.name or []
    if names and len(names) != len(args.score):
        raise ValueError("--name must be supplied once per --score, or omitted entirely.")
    while len(names) < len(args.score):
        names.append(Path(args.score[len(names)]).parent.name)

    summaries = []
    for name, path in zip(names, args.score):
        rows = load_rows(Path(path), dev_buckets=args.dev_buckets)
        summaries.append(summarize(name=name, path=path, rows=rows))

    write_json(Path(args.output_json), {"summaries": summaries})
    write_markdown(Path(args.output_md), summaries)
    print(f"Wrote calibration JSON to {args.output_json}")
    print(f"Wrote calibration report to {args.output_md}")


def load_rows(path: Path, *, dev_buckets: int) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            margin = float(row["no_loss"]) - float(row["yes_loss"])
            target_yes = str(row.get("target_answer", "")).lower() == "yes"
            group_id = str(row.get("image_id") or image_group_from_sample_id(str(row.get("sample_id", ""))))
            rows.append(
                {
                    "sample_id": str(row.get("sample_id", "")),
                    "group_id": group_id,
                    "split": stable_split(group_id, dev_buckets=dev_buckets),
                    "margin": margin,
                    "target_yes": target_yes,
                }
            )
    if not rows:
        raise ValueError(f"No rows loaded from {path}")
    return rows


def image_group_from_sample_id(sample_id: str) -> str:
    if ":" in sample_id:
        return sample_id.split(":", 1)[0]
    return sample_id


def stable_split(value: str, *, dev_buckets: int) -> str:
    bucket = int(hashlib.sha1(value.encode("utf-8")).hexdigest()[:8], 16) % 10
    return "dev" if bucket < dev_buckets else "test"


def summarize(*, name: str, path: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    dev = [row for row in rows if row["split"] == "dev"]
    test = [row for row in rows if row["split"] == "test"]
    default_all = evaluate(rows, threshold=0.0)
    default_dev = evaluate(dev, threshold=0.0)
    default_test = evaluate(test, threshold=0.0)
    dev_threshold = best_threshold(dev)
    all_threshold = best_threshold(rows)
    return {
        "name": name,
        "path": path,
        "n": len(rows),
        "dev_n": len(dev),
        "test_n": len(test),
        "margin_auroc_all": roc_auc([1.0 if r["target_yes"] else 0.0 for r in rows], [r["margin"] for r in rows]),
        "default_threshold": 0.0,
        "default_all": default_all,
        "default_dev": default_dev,
        "default_test": default_test,
        "dev_best_threshold": dev_threshold,
        "dev_best_dev": evaluate(dev, threshold=dev_threshold),
        "dev_best_test": evaluate(test, threshold=dev_threshold),
        "all_best_threshold": all_threshold,
        "all_best_all": evaluate(rows, threshold=all_threshold),
    }


def best_threshold(rows: list[dict[str, Any]]) -> float:
    margins = sorted({float(row["margin"]) for row in rows})
    if not margins:
        return 0.0
    candidates = [margins[0] - 1.0]
    candidates.extend((left + right) / 2.0 for left, right in zip(margins, margins[1:]))
    candidates.append(margins[-1] + 1.0)
    best = None
    best_metrics = None
    for threshold in candidates:
        metrics = evaluate(rows, threshold=threshold)
        key = (metrics["acc"], -metrics["hFPR"], -abs(metrics["yes_rate"] - 0.5))
        if best is None or key > best:
            best = key
            best_metrics = (threshold, metrics)
    assert best_metrics is not None
    return float(best_metrics[0])


def evaluate(rows: list[dict[str, Any]], *, threshold: float) -> dict[str, float]:
    if not rows:
        return {"acc": 0.0, "hFPR": 0.0, "pos_acc": 0.0, "neg_acc": 0.0, "yes_rate": 0.0}
    correct = 0
    yes_count = 0
    pos_total = 0
    pos_correct = 0
    neg_total = 0
    neg_correct = 0
    false_yes = 0
    for row in rows:
        pred_yes = float(row["margin"]) >= threshold
        target_yes = bool(row["target_yes"])
        correct += int(pred_yes == target_yes)
        yes_count += int(pred_yes)
        if target_yes:
            pos_total += 1
            pos_correct += int(pred_yes)
        else:
            neg_total += 1
            neg_correct += int(not pred_yes)
            false_yes += int(pred_yes)
    return {
        "acc": correct / len(rows),
        "hFPR": false_yes / neg_total if neg_total else 0.0,
        "pos_acc": pos_correct / pos_total if pos_total else 0.0,
        "neg_acc": neg_correct / neg_total if neg_total else 0.0,
        "yes_rate": yes_count / len(rows),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_markdown(path: Path, summaries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Yes/No Threshold Calibration",
        "",
        "Prediction rule: `yes` iff `no_loss - yes_loss >= threshold`.",
        "Thresholds are selected on a hash split by image/group id, so paired positive/negative probes stay together.",
        "",
        "| model | n | AUROC | default acc | default hFPR | dev threshold | test acc | test hFPR | test yes_rate | all-best acc | all-best hFPR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            "| {name} | {n} | {auc:.3f} | {d_acc:.3f} | {d_hfpr:.3f} | {thr:.3f} | "
            "{t_acc:.3f} | {t_hfpr:.3f} | {t_yes:.3f} | {a_acc:.3f} | {a_hfpr:.3f} |".format(
                name=item["name"],
                n=item["n"],
                auc=item["margin_auroc_all"],
                d_acc=item["default_all"]["acc"],
                d_hfpr=item["default_all"]["hFPR"],
                thr=item["dev_best_threshold"],
                t_acc=item["dev_best_test"]["acc"],
                t_hfpr=item["dev_best_test"]["hFPR"],
                t_yes=item["dev_best_test"]["yes_rate"],
                a_acc=item["all_best_all"]["acc"],
                a_hfpr=item["all_best_all"]["hFPR"],
            )
        )
    lines.extend(["", "## Paths", ""])
    for item in summaries:
        lines.append(f"- {item['name']}: `{item['path']}`")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
