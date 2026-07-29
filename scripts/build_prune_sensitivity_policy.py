#!/usr/bin/env python
"""Build sample-level pruning sensitivity labels and a simple budget policy."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recap.prune.metrics import box_area, evidence_regions_from_sample, union_area


RATIOS = (0.25, 0.35, 0.50, 0.70)
RATIO_TAGS = {0.25: "0p25", 0.35: "0p35", 0.50: "0p50", 0.70: "0p70", 1.00: "1p00"}
DEFAULT_FEATURES = (
    "support_abs",
    "support_orig",
    "confidence_risk",
    "entropy_risk",
    "recap_evidence_risk",
    "recap_confidence_risk",
    "rice_recap_selector",
    "rice_recap_mean",
    "rice_profile_mean",
    "rice_profile_max",
    "rice_profile_top2",
    "rice_profile_lift",
    "u_conf_rank",
    "c_contra_rank",
    "g_prior_rank",
    "r_struct_profile_rank",
    "recap_evidence_risk_recap_rank",
    "evidence_area",
    "subject_area",
    "object_area",
    "min_object_area",
    "max_object_area",
    "area_imbalance",
    "center_distance",
    "horizontal_gap",
    "vertical_gap",
    "evidence_region_count",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-tag", required=True, choices=["2b", "8b"])
    parser.add_argument("--selector", default="hybrid")
    parser.add_argument("--canonical", default="data/recap_gsrbench_coco_spatial_two.jsonl")
    parser.add_argument("--risk-scores", required=True)
    parser.add_argument("--runs-dir", default="runs/prune_main")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-mean-keep", type=float, default=0.50)
    parser.add_argument("--dev-buckets", type=int, default=5, help="Number of hash buckets out of 10 used as dev.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = Path(args.runs_dir)

    canonical = {str(row.get("id", row.get("sample_id", ""))): row for row in read_jsonl(Path(args.canonical))}
    risk_rows = {str(row.get("sample_id", row.get("id", ""))): row for row in read_jsonl(Path(args.risk_scores))}
    correctness = load_correctness_grid(runs_dir, args.model_tag, args.selector)
    full_correctness = load_sample_correctness(runs_dir / f"qwen3_{args.model_tag}_gsr_orig_topk_1p00" / "sample_scores.jsonl")

    labels = build_labels(
        sample_ids=sorted(set(canonical) | set(risk_rows) | set(full_correctness)),
        canonical=canonical,
        risk_rows=risk_rows,
        correctness=correctness,
        full_correctness=full_correctness,
        dev_buckets=args.dev_buckets,
    )
    write_csv(out_dir / "prune_sensitivity_labels.csv", labels)
    write_jsonl(out_dir / "prune_sensitivity_labels.jsonl", labels)

    feature_rows = score_features(labels)
    write_csv(out_dir / "feature_auc.csv", feature_rows)

    candidates = search_policies(labels, feature_rows, target_mean_keep=args.target_mean_keep)
    write_csv(out_dir / "policy_candidates.csv", candidates)
    best = candidates[0]
    policy = {
        "model_tag": args.model_tag,
        "selector": args.selector,
        "feature": best["feature"],
        "direction": int(best["direction"]),
        "thresholds": [float(best["threshold_1"]), float(best["threshold_2"]), float(best["threshold_3"])],
        "budgets": [float(best["budget_1"]), float(best["budget_2"]), float(best["budget_3"]), float(best["budget_4"])],
        "target_mean_keep": float(args.target_mean_keep),
        "dev_accuracy": float(best["dev_accuracy"]),
        "dev_mean_keep": float(best["dev_mean_keep"]),
        "test_accuracy": float(best["test_accuracy"]),
        "test_mean_keep": float(best["test_mean_keep"]),
        "all_accuracy": float(best["all_accuracy"]),
        "all_mean_keep": float(best["all_mean_keep"]),
    }
    write_json(out_dir / "policy_config.json", policy)

    predictions = []
    for row in labels:
        keep_ratio = apply_policy(row, policy)
        predictions.append(
            {
                "sample_id": row["sample_id"],
                "keep_ratio": keep_ratio,
                "split": row["split"],
                "policy_feature": policy["feature"],
                "policy_score": float(row.get(policy["feature"], 0.0)),
                "min_keep": row["min_keep"],
                "oracle_correct": bool(row["oracle_correct"]),
            }
        )
    write_jsonl(out_dir / "budget_ratios.jsonl", predictions)
    write_report(out_dir / "policy_report.md", labels, feature_rows, candidates, policy)

    print(f"Wrote {len(labels)} labels to {out_dir / 'prune_sensitivity_labels.csv'}")
    print(f"Wrote policy to {out_dir / 'policy_config.json'}")
    print(f"Wrote budgets to {out_dir / 'budget_ratios.jsonl'}")
    print(
        "Best policy "
        f"feature={policy['feature']} direction={policy['direction']} "
        f"dev_acc={policy['dev_accuracy']:.4f} test_acc={policy['test_accuracy']:.4f} "
        f"all_acc={policy['all_accuracy']:.4f} all_keep={policy['all_mean_keep']:.4f}"
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_sample_correctness(path: Path) -> dict[str, bool]:
    out: dict[str, bool] = {}
    if not path.exists():
        return out
    for row in read_jsonl(path):
        sample_id = str(row.get("sample_id", row.get("id", "")))
        if not sample_id:
            continue
        if "direct_correct" in row:
            out[sample_id] = bool(row["direct_correct"])
        else:
            out[sample_id] = bool(row.get("correct", False))
    return out


def load_correctness_grid(runs_dir: Path, model_tag: str, selector: str) -> dict[float, dict[str, bool]]:
    out: dict[float, dict[str, bool]] = {}
    for ratio in RATIOS:
        run = runs_dir / f"qwen3_{model_tag}_gsr_orig_{selector}_{RATIO_TAGS[ratio]}" / "sample_scores.jsonl"
        out[ratio] = load_sample_correctness(run)
    return out


def build_labels(
    *,
    sample_ids: list[str],
    canonical: dict[str, dict[str, Any]],
    risk_rows: dict[str, dict[str, Any]],
    correctness: dict[float, dict[str, bool]],
    full_correctness: dict[str, bool],
    dev_buckets: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        sample = canonical.get(sample_id, {})
        risk = risk_rows.get(sample_id, {})
        correct_by_ratio = {ratio: bool(correctness.get(ratio, {}).get(sample_id, False)) for ratio in RATIOS}
        full_correct = bool(full_correctness.get(sample_id, False))
        min_keep = ""
        for ratio in (*RATIOS, 1.0):
            ok = full_correct if ratio == 1.0 else correct_by_ratio[ratio]
            if ok:
                min_keep = ratio
                break
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "split": stable_split(sample_id, dev_buckets=dev_buckets),
            "min_keep": min_keep,
            "oracle_correct": min_keep != "",
            "full_correct": full_correct,
            "needs_gt_0p25": bool(min_keep != "" and float(min_keep) > 0.25),
            "needs_gt_0p50": bool(min_keep != "" and float(min_keep) > 0.50),
            "correct_0p25": correct_by_ratio[0.25],
            "correct_0p35": correct_by_ratio[0.35],
            "correct_0p50": correct_by_ratio[0.50],
            "correct_0p70": correct_by_ratio[0.70],
            "correct_1p00": full_correct,
            "relation": sample.get("relation", risk.get("base_relation", "")),
            "relation_family": risk.get("relation_family", ""),
            "binary_polarity": sample.get("binary_polarity", "negative" if risk.get("target_is_negative") else "positive"),
        }
        row.update(numeric_risk_features(risk))
        row.update(geometry_features(sample))
        rows.append(row)
    return rows


def stable_split(sample_id: str, *, dev_buckets: int) -> str:
    bucket = int(hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:8], 16) % 10
    return "dev" if bucket < dev_buckets else "test"


def numeric_risk_features(row: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in DEFAULT_FEATURES:
        if key in {"support_abs"}:
            continue
        value = row.get(key)
        if isinstance(value, (int, float)):
            out[key] = float(value)
    support = row.get("support_orig")
    out["support_abs"] = abs(float(support)) if isinstance(support, (int, float)) else 0.0
    return out


def geometry_features(sample: dict[str, Any]) -> dict[str, float]:
    regions = evidence_regions_from_sample(sample) if sample else []
    subject = regions[0] if regions else None
    obj = regions[1] if len(regions) > 1 else None
    subject_area = box_area(subject) if subject else 0.0
    object_area = box_area(obj) if obj else 0.0
    min_area = min([area for area in (subject_area, object_area) if area > 0.0], default=0.0)
    max_area = max(subject_area, object_area)
    if subject and obj:
        scx, scy = center(subject)
        ocx, ocy = center(obj)
        center_distance = ((scx - ocx) ** 2 + (scy - ocy) ** 2) ** 0.5
        horizontal_gap = max(0.0, max(subject[0], obj[0]) - min(subject[2], obj[2]))
        vertical_gap = max(0.0, max(subject[1], obj[1]) - min(subject[3], obj[3]))
    else:
        center_distance = 0.0
        horizontal_gap = 0.0
        vertical_gap = 0.0
    return {
        "evidence_area": union_area(regions),
        "subject_area": subject_area,
        "object_area": object_area,
        "min_object_area": min_area,
        "max_object_area": max_area,
        "area_imbalance": (max_area / max(1e-12, min_area)) if min_area > 0.0 else 0.0,
        "center_distance": center_distance,
        "horizontal_gap": horizontal_gap,
        "vertical_gap": vertical_gap,
        "evidence_region_count": float(len(regions)),
    }


def center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def score_features(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for feature in DEFAULT_FEATURES:
        values: list[float] = []
        labels_25: list[float] = []
        labels_50: list[float] = []
        for row in rows:
            value = row.get(feature)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                continue
            if not row.get("oracle_correct"):
                continue
            values.append(float(value))
            labels_25.append(1.0 if row["needs_gt_0p25"] else 0.0)
            labels_50.append(1.0 if row["needs_gt_0p50"] else 0.0)
        if len(values) < 2:
            continue
        auc25 = roc_auc(labels_25, values)
        auc50 = roc_auc(labels_50, values)
        out.append(
            {
                "feature": feature,
                "n": len(values),
                "auc_needs_gt_0p25": auc25,
                "auc_needs_gt_0p25_abs": max(auc25, 1.0 - auc25),
                "direction_needs_gt_0p25": 1 if auc25 >= 0.5 else -1,
                "auc_needs_gt_0p50": auc50,
                "auc_needs_gt_0p50_abs": max(auc50, 1.0 - auc50),
                "direction_needs_gt_0p50": 1 if auc50 >= 0.5 else -1,
            }
        )
    return sorted(out, key=lambda r: (-float(r["auc_needs_gt_0p25_abs"]), r["feature"]))


def roc_auc(labels: list[float], scores: list[float]) -> float:
    positives = sum(1 for label in labels if label > 0.5)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            if pairs[k][1] > 0.5:
                rank_sum += avg_rank
        i = j
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def search_policies(rows: list[dict[str, Any]], feature_rows: list[dict[str, Any]], *, target_mean_keep: float) -> list[dict[str, Any]]:
    features = [row["feature"] for row in feature_rows]
    budgets_sets = [
        (0.25, 0.35, 0.50, 0.70),
        (0.25, 0.35, 0.50, 1.00),
        (0.25, 0.50, 0.70, 1.00),
    ]
    quantile_sets = [
        (0.50, 0.75, 0.90),
        (0.55, 0.75, 0.90),
        (0.60, 0.80, 0.92),
        (0.65, 0.82, 0.94),
        (0.70, 0.85, 0.95),
        (0.75, 0.88, 0.96),
    ]
    candidates: list[dict[str, Any]] = []
    dev_rows = [row for row in rows if row["split"] == "dev"]
    for feature in features:
        for direction in (1, -1):
            scored_dev = [float(row.get(feature, 0.0)) * direction for row in dev_rows if isinstance(row.get(feature), (int, float))]
            if len(scored_dev) < 10:
                continue
            for q1, q2, q3 in quantile_sets:
                thresholds = [quantile(scored_dev, q) for q in (q1, q2, q3)]
                for budgets in budgets_sets:
                    policy = {"feature": feature, "direction": direction, "thresholds": thresholds, "budgets": budgets}
                    dev_acc, dev_keep = eval_policy(rows, policy, split="dev")
                    if dev_keep > target_mean_keep:
                        continue
                    test_acc, test_keep = eval_policy(rows, policy, split="test")
                    all_acc, all_keep = eval_policy(rows, policy, split="")
                    candidates.append(
                        {
                            "feature": feature,
                            "direction": direction,
                            "threshold_1": thresholds[0],
                            "threshold_2": thresholds[1],
                            "threshold_3": thresholds[2],
                            "budget_1": budgets[0],
                            "budget_2": budgets[1],
                            "budget_3": budgets[2],
                            "budget_4": budgets[3],
                            "dev_accuracy": dev_acc,
                            "dev_mean_keep": dev_keep,
                            "test_accuracy": test_acc,
                            "test_mean_keep": test_keep,
                            "all_accuracy": all_acc,
                            "all_mean_keep": all_keep,
                        }
                    )
    if not candidates:
        raise RuntimeError("No policy candidates met the mean-keep target.")
    return sorted(candidates, key=lambda r: (-float(r["dev_accuracy"]), float(r["dev_mean_keep"]), -float(r["test_accuracy"]), r["feature"]))


def quantile(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    pos = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
    return values[pos]


def apply_policy(row: dict[str, Any], policy: dict[str, Any]) -> float:
    score = float(row.get(policy["feature"], 0.0)) * int(policy["direction"])
    thresholds = [float(x) for x in policy["thresholds"]]
    budgets = [float(x) for x in policy["budgets"]]
    if score <= thresholds[0]:
        return budgets[0]
    if score <= thresholds[1]:
        return budgets[1]
    if score <= thresholds[2]:
        return budgets[2]
    return budgets[3]


def eval_policy(rows: list[dict[str, Any]], policy: dict[str, Any], *, split: str) -> tuple[float, float]:
    selected = [row for row in rows if not split or row["split"] == split]
    if not selected:
        return 0.0, 0.0
    correct = 0
    keep_sum = 0.0
    for row in selected:
        keep = apply_policy(row, policy)
        keep_sum += keep
        correct += 1 if row.get(f"correct_{RATIO_TAGS[keep]}", False) else 0
    return correct / len(selected), keep_sum / len(selected)


def write_report(
    path: Path,
    labels: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    policy: dict[str, Any],
) -> None:
    split_counts = {split: sum(1 for row in labels if row["split"] == split) for split in ("dev", "test")}
    with path.open("w") as f:
        f.write("# Pruning Sensitivity Policy\n\n")
        f.write(f"Samples: {len(labels)} dev={split_counts['dev']} test={split_counts['test']}\n\n")
        f.write("## Selected Policy\n\n")
        f.write(json.dumps(policy, indent=2) + "\n\n")
        f.write("## Top Features\n\n")
        f.write("| feature | AUC needs >0.25 | direction | AUC needs >0.50 |\n")
        f.write("|---|---:|---:|---:|\n")
        for row in feature_rows[:12]:
            f.write(
                f"| {row['feature']} | {row['auc_needs_gt_0p25_abs']:.4f} | "
                f"{row['direction_needs_gt_0p25']} | {row['auc_needs_gt_0p50_abs']:.4f} |\n"
            )
        f.write("\n## Top Policies\n\n")
        f.write("| feature | dir | budgets | dev acc | dev keep | test acc | test keep |\n")
        f.write("|---|---:|---|---:|---:|---:|---:|\n")
        for row in candidates[:12]:
            budgets = ",".join(str(row[f"budget_{i}"]) for i in range(1, 5))
            f.write(
                f"| {row['feature']} | {row['direction']} | {budgets} | "
                f"{row['dev_accuracy']:.4f} | {row['dev_mean_keep']:.4f} | "
                f"{row['test_accuracy']:.4f} | {row['test_mean_keep']:.4f} |\n"
            )


if __name__ == "__main__":
    main()
