#!/usr/bin/env python
"""Analyze orig/text_only/crop/boxed probe scores with a deterministic dev/test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("score_files", nargs="+", type=Path)
    parser.add_argument("--dev-mod", type=int, default=5, help="Use sha1(sample_id) %% dev_mod == 0 as dev.")
    args = parser.parse_args()

    for score_file in args.score_files:
        rows = [json.loads(line) for line in score_file.open(encoding="utf-8")]
        print(f"\n== {score_file} ==")
        analyze(rows, dev_mod=args.dev_mod)


def analyze(rows: list[dict], *, dev_mod: int) -> None:
    by_probe: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_probe[str(row["probe"])].append(row)

    paired = _paired_rows(rows, dev_mod=dev_mod)
    dev = [row for row in paired if row["split"] == "dev"]
    test = [row for row in paired if row["split"] == "test"]
    print(f"n_samples={len(paired)} dev={len(dev)} test={len(test)} targets={dict(Counter(row['target_answer'] for row in paired))}")

    visual_probes = [probe for probe in ("orig", "crop", "boxed") if probe in by_probe]
    report_probes = [probe for probe in ("orig", "text_only", "crop", "boxed") if probe in by_probe]

    for probe in report_probes:
        if probe not in by_probe:
            continue
        all_metrics = _metrics(by_probe[probe], lambda row: float(row["margin"]), 0.0)
        test_rows = [row for row in by_probe[probe] if _split(str(row["sample_id"]), dev_mod=dev_mod) == "test"]
        test_metrics = _metrics(test_rows, lambda row: float(row["margin"]), 0.0)
        dev_rows = [row for row in by_probe[probe] if _split(str(row["sample_id"]), dev_mod=dev_mod) == "dev"]
        threshold, dev_metrics = _best_threshold(dev_rows, lambda row: float(row["margin"]))
        raw_cal_metrics = _metrics(test_rows, lambda row: float(row["margin"]), threshold)
        print(
            f"raw {probe:9s} all acc={all_metrics['acc']:.4f} hFPR={all_metrics['hFPR']:.4f} yes={all_metrics['yes_rate']:.4f} "
            f"| test@0 acc={test_metrics['acc']:.4f} hFPR={test_metrics['hFPR']:.4f} yes={test_metrics['yes_rate']:.4f} "
            f"| rawcal dev_t={threshold:.4f} dev_acc={dev_metrics['acc']:.4f} "
            f"test_acc={raw_cal_metrics['acc']:.4f} hFPR={raw_cal_metrics['hFPR']:.4f} yes={raw_cal_metrics['yes_rate']:.4f}"
        )

    labels = []
    for probe in visual_probes:
        if probe != "orig":
            labels.append(f"{probe}_minus_orig")
        labels.append(f"{probe}_minus_text")
    if "crop" in visual_probes and "boxed" in visual_probes:
        labels.extend(["boxed_minus_crop", "crop_minus_boxed"])

    seen_labels = set()
    labels = [label for label in labels if not (label in seen_labels or seen_labels.add(label))]
    for label in labels:
        if not paired:
            continue
        zero_metrics = _metrics(paired, lambda row, key=label: float(row[key]), 0.0)
        threshold, dev_metrics = _best_threshold(dev, lambda row, key=label: float(row[key]))
        test_metrics = _metrics(test, lambda row, key=label: float(row[key]), threshold)
        oracle_threshold, oracle_metrics = _best_threshold(paired, lambda row, key=label: float(row[key]))
        print(
            f"cal {label:15s} all@0 acc={zero_metrics['acc']:.4f} hFPR={zero_metrics['hFPR']:.4f} yes={zero_metrics['yes_rate']:.4f} "
            f"| dev_t={threshold:.4f} dev_acc={dev_metrics['acc']:.4f} "
            f"test_acc={test_metrics['acc']:.4f} hFPR={test_metrics['hFPR']:.4f} yes={test_metrics['yes_rate']:.4f} "
            f"| oracle_t={oracle_threshold:.4f} oracle_acc={oracle_metrics['acc']:.4f}"
        )

    for probe in report_probes:
        if probe not in by_probe:
            continue
        positives = [float(row["margin"]) for row in by_probe[probe] if row["target_answer"] == "yes"]
        negatives = [float(row["margin"]) for row in by_probe[probe] if row["target_answer"] == "no"]
        if positives and negatives:
            pos_avg = sum(positives) / len(positives)
            neg_avg = sum(negatives) / len(negatives)
            print(f"gap {probe:9s} pos={pos_avg:.4f} neg={neg_avg:.4f} gap={pos_avg - neg_avg:.4f}")


def _paired_rows(rows: Iterable[dict], *, dev_mod: int) -> list[dict]:
    by_sample: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_sample[str(row["sample_id"])][str(row["probe"])] = row

    paired = []
    for sample_id, probes in by_sample.items():
        if not {"orig", "text_only"}.issubset(probes):
            continue
        row = dict(probes["orig"])
        row["split"] = _split(sample_id, dev_mod=dev_mod)
        for probe_name, key in (("orig", "orig_margin"), ("text_only", "text_margin"), ("crop", "crop_margin"), ("boxed", "boxed_margin")):
            if probe_name in probes:
                row[key] = float(probes[probe_name]["margin"])
        if "orig_margin" in row and "text_margin" in row:
            row["orig_minus_text"] = row["orig_margin"] - row["text_margin"]
        if "crop_margin" in row and "text_margin" in row:
            row["crop_minus_text"] = row["crop_margin"] - row["text_margin"]
        if "boxed_margin" in row and "text_margin" in row:
            row["boxed_minus_text"] = row["boxed_margin"] - row["text_margin"]
        if "crop_margin" in row and "orig_margin" in row:
            row["crop_minus_orig"] = row["crop_margin"] - row["orig_margin"]
        if "boxed_margin" in row and "orig_margin" in row:
            row["boxed_minus_orig"] = row["boxed_margin"] - row["orig_margin"]
        if "boxed_margin" in row and "crop_margin" in row:
            row["boxed_minus_crop"] = row["boxed_margin"] - row["crop_margin"]
            row["crop_minus_boxed"] = row["crop_margin"] - row["boxed_margin"]
        paired.append(row)
    return paired


def _split(sample_id: str, *, dev_mod: int) -> str:
    digest = int(hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:8], 16)
    return "dev" if digest % dev_mod == 0 else "test"


def _metrics(rows: list[dict], score_fn: Callable[[dict], float], threshold: float) -> dict[str, float]:
    if not rows:
        return {"n": 0, "acc": math.nan, "hFPR": math.nan, "yes_rate": math.nan}
    correct = yes_count = true_pos = false_pos = true_neg = false_neg = 0
    for row in rows:
        target = str(row["target_answer"]).lower() == "yes"
        pred = score_fn(row) > threshold
        yes_count += int(pred)
        correct += int(pred == target)
        if target and pred:
            true_pos += 1
        elif target and not pred:
            false_neg += 1
        elif not target and pred:
            false_pos += 1
        else:
            true_neg += 1
    neg_count = false_pos + true_neg
    return {
        "n": len(rows),
        "acc": correct / len(rows),
        "hFPR": false_pos / neg_count if neg_count else math.nan,
        "yes_rate": yes_count / len(rows),
        "tp": true_pos,
        "fp": false_pos,
        "tn": true_neg,
        "fn": false_neg,
    }


def _best_threshold(rows: list[dict], score_fn: Callable[[dict], float]) -> tuple[float, dict[str, float]]:
    if not rows:
        return 0.0, _metrics(rows, score_fn, 0.0)
    scores = sorted({score_fn(row) for row in rows})
    candidates = [scores[0] - 1e-6]
    candidates.extend((left + right) / 2.0 for left, right in zip(scores, scores[1:]))
    candidates.append(scores[-1] + 1e-6)

    best_key: tuple[float, float, float] | None = None
    best_threshold = 0.0
    best_metrics: dict[str, float] = {}
    for threshold in candidates:
        metrics = _metrics(rows, score_fn, threshold)
        key = (metrics["acc"], -metrics["hFPR"], -abs(metrics["yes_rate"] - 0.5))
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = threshold
            best_metrics = metrics
    return best_threshold, best_metrics


if __name__ == "__main__":
    main()
