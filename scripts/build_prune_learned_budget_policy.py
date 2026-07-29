#!/usr/bin/env python
"""Build split-safe sample-level pruning budgets with a small monotone policy."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recap.prune.metrics import box_area, evidence_regions_from_sample, union_area


DEFAULT_RATIOS = (0.25, 0.35, 0.50, 0.70, 1.00)
RATIO_TAGS = {0.25: "0p25", 0.35: "0p35", 0.50: "0p50", 0.70: "0p70", 1.00: "1p00"}
FEATURES = (
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
    "is_horizontal",
    "is_vertical",
    "is_left",
    "is_right",
    "is_above",
    "is_below",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-tag", required=True, choices=["2b", "8b"])
    parser.add_argument("--selector", default="hybrid")
    parser.add_argument("--canonical", default="data/recap_gsrbench_coco_spatial_two.jsonl")
    parser.add_argument("--risk-scores", required=True)
    parser.add_argument("--runs-dir", default="runs/prune_main")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allowed-budgets", default="0.25,0.35,0.50,0.70,1.00")
    parser.add_argument("--target-mean-keep", type=float, default=0.50)
    parser.add_argument("--lambda-keep", type=float, default=0.02)
    parser.add_argument("--beta-hfpr", type=float, default=0.05)
    parser.add_argument("--dev-buckets", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=80, help="How many top policies to write to the report.")
    args = parser.parse_args()

    budgets = parse_budgets(args.allowed_budgets)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = build_rows(
        model_tag=args.model_tag,
        selector=args.selector,
        canonical_path=Path(args.canonical),
        risk_path=Path(args.risk_scores),
        runs_dir=Path(args.runs_dir),
        budgets=budgets,
        dev_buckets=args.dev_buckets,
    )
    write_jsonl(out_dir / "policy_rows.jsonl", rows)
    write_csv(out_dir / "policy_rows.csv", rows)

    candidates = search_policies(
        rows,
        budgets=budgets,
        target_mean_keep=args.target_mean_keep,
        lambda_keep=args.lambda_keep,
        beta_hfpr=args.beta_hfpr,
    )
    write_csv(out_dir / "policy_candidates.csv", candidates)

    best = candidates[0]
    policy = {
        "model_tag": args.model_tag,
        "selector": args.selector,
        "policy_type": "monotone_bins",
        "feature": best["feature"],
        "direction": int(best["direction"]),
        "thresholds": [float(x) for x in json.loads(best["thresholds"])],
        "budgets": [float(x) for x in json.loads(best["budgets"])],
        "target_mean_keep": float(args.target_mean_keep),
        "lambda_keep": float(args.lambda_keep),
        "beta_hfpr": float(args.beta_hfpr),
        "dev_accuracy": float(best["dev_accuracy"]),
        "dev_hfpr": float(best["dev_hfpr"]),
        "dev_mean_keep": float(best["dev_mean_keep"]),
        "dev_objective": float(best["dev_objective"]),
        "test_accuracy": float(best["test_accuracy"]),
        "test_hfpr": float(best["test_hfpr"]),
        "test_mean_keep": float(best["test_mean_keep"]),
        "all_accuracy": float(best["all_accuracy"]),
        "all_hfpr": float(best["all_hfpr"]),
        "all_mean_keep": float(best["all_mean_keep"]),
    }
    write_json(out_dir / "policy_config.json", policy)

    predictions = []
    for row in rows:
        keep_ratio = apply_policy(row, policy)
        predictions.append(
            {
                "sample_id": row["sample_id"],
                "keep_ratio": keep_ratio,
                "split": row["split"],
                "policy_type": policy["policy_type"],
                "policy_feature": policy["feature"],
                "policy_score": float(row.get(policy["feature"], 0.0)),
                "min_keep": row["min_keep"],
                "oracle_correct": bool(row["oracle_correct"]),
            }
        )
    write_jsonl(out_dir / "budget_ratios.jsonl", predictions)
    write_report(out_dir / "policy_report.md", rows, candidates[: args.top_k], policy)

    print(f"Wrote {len(rows)} rows to {out_dir / 'policy_rows.jsonl'}")
    print(f"Wrote policy to {out_dir / 'policy_config.json'}")
    print(f"Wrote budgets to {out_dir / 'budget_ratios.jsonl'}")
    print(
        "Best policy "
        f"feature={policy['feature']} direction={policy['direction']} budgets={policy['budgets']} "
        f"dev_acc={policy['dev_accuracy']:.4f} dev_hfpr={policy['dev_hfpr']:.4f} dev_keep={policy['dev_mean_keep']:.4f} "
        f"test_acc={policy['test_accuracy']:.4f} test_hfpr={policy['test_hfpr']:.4f} test_keep={policy['test_mean_keep']:.4f}"
    )


def parse_budgets(value: str) -> tuple[float, ...]:
    budgets = tuple(sorted({round(float(item.strip()), 2) for item in value.split(",") if item.strip()}))
    if not budgets:
        raise ValueError("At least one budget is required.")
    for budget in budgets:
        if budget <= 0.0 or budget > 1.0:
            raise ValueError(f"Budget must be in (0, 1], got {budget}.")
        if budget not in RATIO_TAGS:
            raise ValueError(f"Budget {budget} is not available. Use one of {sorted(RATIO_TAGS)}.")
    return budgets


def build_rows(
    *,
    model_tag: str,
    selector: str,
    canonical_path: Path,
    risk_path: Path,
    runs_dir: Path,
    budgets: tuple[float, ...],
    dev_buckets: int,
) -> list[dict[str, Any]]:
    canonical = {sample_id(row): row for row in read_jsonl(canonical_path)}
    risk_rows = {sample_id(row): row for row in read_jsonl(risk_path)}
    score_grid = {budget: load_scores_for_budget(runs_dir, model_tag, selector, budget) for budget in budgets}
    sample_ids = sorted(set(canonical) | set(risk_rows) | set().union(*(set(x) for x in score_grid.values())))

    rows = []
    for sid in sample_ids:
        sample = canonical.get(sid, {})
        risk = risk_rows.get(sid, {})
        relation = str(sample.get("relation", risk.get("base_relation", "")))
        row: dict[str, Any] = {
            "sample_id": sid,
            "split": stable_split(sid, dev_buckets=dev_buckets),
            "relation": relation,
            "binary_polarity": sample.get("binary_polarity", "negative" if risk.get("target_is_negative") else "positive"),
            "target_is_negative": bool(any(score_grid[b].get(sid, {}).get("target_is_negative", False) for b in budgets)),
        }
        for budget in budgets:
            tag = RATIO_TAGS[budget]
            score = score_grid[budget].get(sid, {})
            row[f"correct_{tag}"] = bool(score.get("direct_correct", False))
            row[f"hallucination_{tag}"] = bool(score.get("hallucination", False))

        min_keep = ""
        for budget in budgets:
            if row.get(f"correct_{RATIO_TAGS[budget]}", False):
                min_keep = budget
                break
        row["min_keep"] = min_keep
        row["oracle_correct"] = min_keep != ""
        row.update(numeric_features(risk))
        row.update(geometry_features(sample))
        row.update(relation_features(relation))
        rows.append(row)
    return rows


def load_scores_for_budget(runs_dir: Path, model_tag: str, selector: str, budget: float) -> dict[str, dict[str, Any]]:
    if budget == 1.0:
        path = runs_dir / f"qwen3_{model_tag}_gsr_orig_topk_1p00" / "sample_scores.jsonl"
    else:
        path = runs_dir / f"qwen3_{model_tag}_gsr_orig_{selector}_{RATIO_TAGS[budget]}" / "sample_scores.jsonl"
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        out[sample_id(row)] = row
    return out


def sample_id(row: dict[str, Any]) -> str:
    return str(row.get("sample_id", row.get("id", "")))


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


def stable_split(sid: str, *, dev_buckets: int) -> str:
    bucket = int(hashlib.sha1(sid.encode("utf-8")).hexdigest()[:8], 16) % 10
    return "dev" if bucket < dev_buckets else "test"


def numeric_features(row: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in FEATURES:
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


def relation_features(relation: str) -> dict[str, float]:
    relation = relation.lower()
    return {
        "is_horizontal": 1.0 if relation in {"left_of", "right_of", "left", "right"} else 0.0,
        "is_vertical": 1.0 if relation in {"above", "below", "over", "under"} else 0.0,
        "is_left": 1.0 if relation in {"left_of", "left"} else 0.0,
        "is_right": 1.0 if relation in {"right_of", "right"} else 0.0,
        "is_above": 1.0 if relation in {"above", "over"} else 0.0,
        "is_below": 1.0 if relation in {"below", "under"} else 0.0,
    }


def center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def search_policies(
    rows: list[dict[str, Any]],
    *,
    budgets: tuple[float, ...],
    target_mean_keep: float,
    lambda_keep: float,
    beta_hfpr: float,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    feature_names = [feature for feature in FEATURES if any(isinstance(row.get(feature), (int, float)) for row in rows)]
    quantile_sets = [
        (0.50,),
        (0.33, 0.67),
        (0.25, 0.50, 0.75),
        (0.50, 0.75, 0.90),
        (0.60, 0.80, 0.92),
        (0.70, 0.85, 0.95),
    ]
    for feature in feature_names:
        for direction in (1, -1):
            dev_scores = [
                direction * float(row.get(feature, 0.0))
                for row in rows
                if row["split"] == "dev" and isinstance(row.get(feature), (int, float)) and math.isfinite(float(row.get(feature)))
            ]
            if len(set(dev_scores)) < 2:
                continue
            for qs in quantile_sets:
                thresholds = sorted({quantile(dev_scores, q) for q in qs})
                if not thresholds:
                    continue
                for budget_tuple in monotone_budget_tuples(budgets, len(thresholds) + 1):
                    policy = {
                        "feature": feature,
                        "direction": direction,
                        "thresholds": thresholds,
                        "budgets": budget_tuple,
                    }
                    dev = eval_policy(rows, policy, split="dev", lambda_keep=lambda_keep, beta_hfpr=beta_hfpr)
                    if dev["mean_keep"] > target_mean_keep:
                        continue
                    test = eval_policy(rows, policy, split="test", lambda_keep=lambda_keep, beta_hfpr=beta_hfpr)
                    all_scores = eval_policy(rows, policy, split="", lambda_keep=lambda_keep, beta_hfpr=beta_hfpr)
                    candidates.append(
                        {
                            "feature": feature,
                            "direction": direction,
                            "thresholds": json.dumps(thresholds),
                            "budgets": json.dumps(list(budget_tuple)),
                            "dev_accuracy": dev["accuracy"],
                            "dev_hfpr": dev["hfpr"],
                            "dev_mean_keep": dev["mean_keep"],
                            "dev_objective": dev["objective"],
                            "test_accuracy": test["accuracy"],
                            "test_hfpr": test["hfpr"],
                            "test_mean_keep": test["mean_keep"],
                            "test_objective": test["objective"],
                            "all_accuracy": all_scores["accuracy"],
                            "all_hfpr": all_scores["hfpr"],
                            "all_mean_keep": all_scores["mean_keep"],
                            "all_objective": all_scores["objective"],
                        }
                    )
    if not candidates:
        raise RuntimeError("No policy candidates met the mean-keep target.")
    return sorted(
        candidates,
        key=lambda row: (
            -float(row["dev_objective"]),
            -float(row["dev_accuracy"]),
            float(row["dev_hfpr"]),
            float(row["dev_mean_keep"]),
            -float(row["test_objective"]),
            str(row["feature"]),
        ),
    )


def monotone_budget_tuples(budgets: tuple[float, ...], length: int) -> list[tuple[float, ...]]:
    return list(itertools.combinations_with_replacement(budgets, length))


def quantile(values: list[float], q: float) -> float:
    values = sorted(values)
    pos = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
    return values[pos]


def apply_policy(row: dict[str, Any], policy: dict[str, Any]) -> float:
    score = int(policy["direction"]) * float(row.get(policy["feature"], 0.0))
    thresholds = [float(x) for x in policy["thresholds"]]
    budgets = [float(x) for x in policy["budgets"]]
    for idx, threshold in enumerate(thresholds):
        if score <= threshold:
            return budgets[idx]
    return budgets[-1]


def eval_policy(
    rows: list[dict[str, Any]],
    policy: dict[str, Any],
    *,
    split: str,
    lambda_keep: float,
    beta_hfpr: float,
) -> dict[str, float]:
    selected = [row for row in rows if not split or row["split"] == split]
    if not selected:
        return {"accuracy": 0.0, "hfpr": 0.0, "mean_keep": 0.0, "objective": -1e9}
    correct = 0
    hallucinations = 0
    negatives = 0
    keep_sum = 0.0
    for row in selected:
        keep = apply_policy(row, policy)
        tag = RATIO_TAGS[keep]
        correct += 1 if row.get(f"correct_{tag}", False) else 0
        keep_sum += keep
        if row.get("target_is_negative", False):
            negatives += 1
            hallucinations += 1 if row.get(f"hallucination_{tag}", False) else 0
    accuracy = correct / len(selected)
    hfpr = hallucinations / negatives if negatives else 0.0
    mean_keep = keep_sum / len(selected)
    return {
        "accuracy": accuracy,
        "hfpr": hfpr,
        "mean_keep": mean_keep,
        "objective": accuracy - lambda_keep * mean_keep - beta_hfpr * hfpr,
    }


def write_report(path: Path, rows: list[dict[str, Any]], candidates: list[dict[str, Any]], policy: dict[str, Any]) -> None:
    split_counts = {split: sum(1 for row in rows if row["split"] == split) for split in ("dev", "test")}
    with path.open("w") as f:
        f.write("# Learned Pruning Budget Policy\n\n")
        f.write(f"Samples: {len(rows)} dev={split_counts['dev']} test={split_counts['test']}\n\n")
        f.write("## Selected Policy\n\n")
        f.write(json.dumps(policy, indent=2) + "\n\n")
        f.write("## Top Policies\n\n")
        f.write("| feature | dir | thresholds | budgets | dev acc | dev hFPR | dev keep | test acc | test hFPR | test keep |\n")
        f.write("|---|---:|---|---|---:|---:|---:|---:|---:|---:|\n")
        for row in candidates:
            f.write(
                f"| {row['feature']} | {row['direction']} | {row['thresholds']} | {row['budgets']} | "
                f"{row['dev_accuracy']:.4f} | {row['dev_hfpr']:.4f} | {row['dev_mean_keep']:.4f} | "
                f"{row['test_accuracy']:.4f} | {row['test_hfpr']:.4f} | {row['test_mean_keep']:.4f} |\n"
            )


if __name__ == "__main__":
    main()
