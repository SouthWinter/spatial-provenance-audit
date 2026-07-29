#!/usr/bin/env python3
"""Train split-safe calibrated risk predictors for TextOCR-Hard Qwen pruning.

This is an offline simulator over cached fixed-budget outputs. It asks whether
deployable low-budget signals can predict which samples should be escalated to
a larger visual-token budget. Unlike the oracle frontier, labels are used only
on the image-disjoint development split to fit a small logistic risk model and
to choose thresholds; the held-out split is untouched until reporting.

The script intentionally avoids scikit-learn so the artifact has no additional
runtime dependency beyond numpy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


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

BASE_SCORE_FEATURES = (
    "abs_text_yes_margin",
    "entropy_risk",
    "u_conf_rank",
    "rice_profile_mean",
    "rice_profile_max",
    "rice_recap_selector",
    "target_text_token_count",
    "full_visual_tokens",
    "predicted_yes",
)

SPATIAL_FEATURES = (
    "selected_row_span",
    "selected_col_span",
    "selected_area_span",
    "selected_row_coverage",
    "selected_col_coverage",
    "selected_compactness",
    "selected_keep_fraction",
)

FEATURE_NAMES = (*BASE_SCORE_FEATURES, *SPATIAL_FEATURES)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="runs/textocr_adaptive_policy/qwen_calibrated_risk_v1")
    parser.add_argument("--dev-buckets", type=int, default=5)
    parser.add_argument("--target-hfpr", type=float, default=0.224)
    parser.add_argument("--lambda-hfpr", type=float, default=2.0)
    parser.add_argument("--lambda-keep", type=float, default=0.04)
    parser.add_argument("--lambda-cascade", type=float, default=0.02)
    parser.add_argument("--lambda-ecr", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--top-k", type=int, default=80)
    args = parser.parse_args()

    scores, traces, ids = load_runs(DEFAULT_RUNS)
    split_by_id = {
        sid: stable_split(scores["1.00"][sid].get("image_id") or sid, args.dev_buckets)
        for sid in ids
    }
    fixed_rows = build_fixed_rows(scores, traces, ids, split_by_id)

    candidates: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for base in ("0.20", "0.25"):
        x, feature_names = build_feature_matrix(scores, traces, ids, base)
        for target_name in ("base_error", "recoverable_base_error"):
            y = build_target(scores, ids, base, target_name)
            model = fit_logistic(
                x[[split_by_id[sid] == "dev" for sid in ids]],
                y[[split_by_id[sid] == "dev" for sid in ids]],
                l2=args.l2,
                steps=args.steps,
                learning_rate=args.learning_rate,
            )
            risk = predict_logistic(model, x)
            candidates.extend(search_policies(scores, traces, ids, split_by_id, risk, base, target_name, args))
            predictions.extend(
                build_prediction_rows(ids, split_by_id, scores, base, target_name, risk, y)
            )

    candidates.sort(key=candidate_sort_key, reverse=True)
    best_rows = best_by_group(candidates, "target")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "fixed_budget_summary.csv", fixed_rows)
    write_csv(out_dir / "policy_candidates.csv", candidates)
    write_csv(out_dir / "top_policy_candidates.csv", candidates[: args.top_k])
    write_csv(out_dir / "best_by_target.csv", best_rows)
    write_csv(out_dir / "risk_predictions.csv", predictions)
    write_json(
        out_dir / "policy_config.json",
        {
            "features": feature_names,
            "dev_buckets": args.dev_buckets,
            "best_by_target": best_rows,
            "note": "Labels are used only for fitting on the dev split and for reporting held-out metrics.",
        },
    )
    (out_dir / "policy_report.md").write_text(
        build_report(fixed_rows, candidates[: args.top_k], best_rows, ids),
        encoding="utf-8",
    )

    print(f"Wrote calibrated risk predictor report to {out_dir / 'policy_report.md'}")
    for row in best_rows:
        print(
            f"{row['target']}: {row['policy']} test_acc={row['test_accuracy']:.3f} "
            f"test_hFPR={row['test_hFPR']:.3f} test_keep={row['test_mean_keep_ratio']:.3f} "
            f"test_ECR={row['test_mean_ecr']:.3f}"
        )


def search_policies(
    scores: dict[str, dict[str, dict[str, Any]]],
    traces: dict[str, dict[str, dict[str, Any]]],
    ids: list[str],
    split_by_id: dict[str, str],
    risk: np.ndarray,
    base: str,
    target_name: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    mids = ("0.30", "0.35", "0.40")
    highs = ("0.35", "0.40", "0.50", "1.00")
    quantiles = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
    dev_risk = [risk[idx] for idx, sid in enumerate(ids) if split_by_id[sid] == "dev"]
    thresholds = sorted(set(float(np.quantile(dev_risk, q)) for q in quantiles))
    rows = []
    risk_by_id = {sid: float(risk[idx]) for idx, sid in enumerate(ids)}
    for mid in mids:
        if float(mid) <= float(base):
            continue
        for high in highs:
            if float(high) <= float(mid):
                continue
            for low_threshold in thresholds:
                for high_threshold in thresholds:
                    if low_threshold > high_threshold:
                        continue
                    dev = evaluate_policy(
                        scores, traces, ids, split_by_id, risk_by_id, base, mid, high, low_threshold, high_threshold, "dev"
                    )
                    test = evaluate_policy(
                        scores, traces, ids, split_by_id, risk_by_id, base, mid, high, low_threshold, high_threshold, "test"
                    )
                    objective = (
                        dev["accuracy"]
                        - args.lambda_hfpr * max(0.0, dev["hFPR"] - args.target_hfpr)
                        - args.lambda_keep * dev["mean_keep_ratio"]
                        - args.lambda_cascade * dev["cascade_keep_ratio"]
                        + args.lambda_ecr * dev["mean_ecr"]
                    )
                    rows.append(
                        {
                            "target": target_name,
                            "policy": policy_name(base, mid, high, low_threshold, high_threshold),
                            "base_budget": base,
                            "mid_budget": mid,
                            "high_budget": high,
                            "low_threshold": low_threshold,
                            "high_threshold": high_threshold,
                            "objective": objective,
                            **prefixed("dev", dev),
                            **prefixed("test", test),
                        }
                    )
    return rows


def fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    l2: float,
    steps: int,
    learning_rate: float,
) -> dict[str, Any]:
    y = y.astype(float)
    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    sigma[sigma < 1e-6] = 1.0
    z = np.column_stack([np.ones(len(x)), (x - mu) / sigma])
    weights = np.zeros(z.shape[1])
    positives = max(1.0, float(y.sum()))
    negatives = max(1.0, float(len(y) - y.sum()))
    sample_weights = np.where(y > 0, len(y) / (2.0 * positives), len(y) / (2.0 * negatives))
    for _ in range(steps):
        logits = np.clip(z @ weights, -40.0, 40.0)
        probs = 1.0 / (1.0 + np.exp(-logits))
        grad = z.T @ ((probs - y) * sample_weights) / len(y)
        grad[1:] += l2 * weights[1:]
        weights -= learning_rate * grad
    return {"weights": weights, "mu": mu, "sigma": sigma}


def predict_logistic(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    z = np.column_stack([np.ones(len(x)), (x - model["mu"]) / model["sigma"]])
    logits = np.clip(z @ model["weights"], -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-logits))


def build_feature_matrix(
    scores: dict[str, dict[str, dict[str, Any]]],
    traces: dict[str, dict[str, dict[str, Any]]],
    ids: list[str],
    base: str,
) -> tuple[np.ndarray, list[str]]:
    rows = []
    for sid in ids:
        score = scores[base][sid]
        trace = traces[base][sid]
        rows.append(score_features(score, trace) + spatial_features(trace))
    return np.array(rows, dtype=float), list(FEATURE_NAMES)


def score_features(score: dict[str, Any], trace: dict[str, Any]) -> list[float]:
    return [
        abs(finite_float(score.get("text_yes_margin", score.get("support_orig", 0.0)))),
        finite_float(score.get("entropy_risk", 0.0)),
        finite_float(score.get("u_conf_rank", 0.0)),
        finite_float(score.get("rice_profile_mean", 0.0)),
        finite_float(score.get("rice_profile_max", 0.0)),
        finite_float(score.get("rice_recap_selector", 0.0)),
        finite_float(trace.get("target_text_token_count", 0.0)),
        finite_float(trace.get("full_visual_tokens", score.get("prune_full_visual_tokens", 0.0))),
        1.0 if str(score.get("direct_pred", score.get("pred_answer", ""))).strip().lower() == "yes" else 0.0,
    ]


def spatial_features(trace: dict[str, Any]) -> list[float]:
    kept = trace.get("kept_indices", [])
    if not isinstance(kept, list) or not kept:
        return [0.0 for _ in SPATIAL_FEATURES]
    grid_h = max(1, int(finite_float(trace.get("grid_h", 1))))
    grid_w = max(1, int(finite_float(trace.get("grid_w", 1))))
    rows = [max(0, min(grid_h - 1, int(idx) // grid_w)) for idx in kept]
    cols = [max(0, min(grid_w - 1, int(idx) % grid_w)) for idx in kept]
    row_span = (max(rows) - min(rows) + 1) / grid_h
    col_span = (max(cols) - min(cols) + 1) / grid_w
    selected_area_span = row_span * col_span
    selected_row_coverage = len(set(rows)) / grid_h
    selected_col_coverage = len(set(cols)) / grid_w
    selected_compactness = len(kept) / max(1.0, (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1))
    selected_keep_fraction = len(kept) / max(1.0, finite_float(trace.get("full_visual_tokens", len(kept))))
    return [
        row_span,
        col_span,
        selected_area_span,
        selected_row_coverage,
        selected_col_coverage,
        selected_compactness,
        selected_keep_fraction,
    ]


def build_target(
    scores: dict[str, dict[str, dict[str, Any]]],
    ids: list[str],
    base: str,
    target_name: str,
) -> np.ndarray:
    budgets = sorted(scores, key=float)
    values = []
    for sid in ids:
        base_correct = bool(scores[base][sid].get("direct_correct", False))
        if target_name == "base_error":
            values.append(0 if base_correct else 1)
        elif target_name == "recoverable_base_error":
            recoverable = (not base_correct) and any(
                bool(scores[budget][sid].get("direct_correct", False))
                for budget in budgets
                if float(budget) > float(base)
            )
            values.append(1 if recoverable else 0)
        else:
            raise ValueError(target_name)
    return np.array(values, dtype=int)


def evaluate_policy(
    scores: dict[str, dict[str, dict[str, Any]]],
    traces: dict[str, dict[str, dict[str, Any]]],
    ids: list[str],
    split_by_id: dict[str, str],
    risk_by_id: dict[str, float],
    base: str,
    mid: str,
    high: str,
    low_threshold: float,
    high_threshold: float,
    split: str,
) -> dict[str, float]:
    selected = [sid for sid in ids if split == "all" or split_by_id[sid] == split]
    correct = hallucinations = negatives = mid_count = high_count = 0
    final_keep = cascade_keep = ecr_total = 0.0
    for sid in selected:
        budget = choose_budget(risk_by_id[sid], base, mid, high, low_threshold, high_threshold)
        score = scores[budget][sid]
        trace = traces[budget][sid]
        correct += int(bool(score.get("direct_correct", False)))
        is_negative = bool(score.get("target_is_negative", False))
        negatives += int(is_negative)
        hallucinations += int(is_negative and bool(score.get("hallucination", False)))
        final_keep += float(budget)
        cascade_keep += float(base) + (float(budget) if float(budget) > float(base) else 0.0)
        ecr_total += finite_float(trace.get("ecr", 0.0))
        mid_count += int(budget == mid)
        high_count += int(budget == high)
    n = len(selected)
    return {
        "n": float(n),
        "accuracy": correct / n,
        "hFPR": hallucinations / negatives if negatives else 0.0,
        "mean_keep_ratio": final_keep / n,
        "cascade_keep_ratio": cascade_keep / n,
        "mean_ecr": ecr_total / n,
        "mid_rate": mid_count / n,
        "high_rate": high_count / n,
    }


def build_fixed_rows(
    scores: dict[str, dict[str, dict[str, Any]]],
    traces: dict[str, dict[str, dict[str, Any]]],
    ids: list[str],
    split_by_id: dict[str, str],
) -> list[dict[str, Any]]:
    rows = []
    dummy_risk = {sid: 0.0 for sid in ids}
    for budget in sorted(scores, key=float):
        row: dict[str, Any] = {"policy": f"Fixed {budget}", "budget": budget}
        for split in ("dev", "test", "all"):
            row.update(
                prefixed(
                    split,
                    evaluate_policy(scores, traces, ids, split_by_id, dummy_risk, budget, budget, budget, 1.0, 2.0, split),
                )
            )
        rows.append(row)
    return rows


def build_prediction_rows(
    ids: list[str],
    split_by_id: dict[str, str],
    scores: dict[str, dict[str, dict[str, Any]]],
    base: str,
    target_name: str,
    risk: np.ndarray,
    target: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for idx, sid in enumerate(ids):
        score = scores[base][sid]
        rows.append(
            {
                "sample_id": sid,
                "split": split_by_id[sid],
                "base_budget": base,
                "target": target_name,
                "risk": float(risk[idx]),
                "target_value": int(target[idx]),
                "base_pred": score.get("direct_pred", ""),
                "base_correct": bool(score.get("direct_correct", False)),
            }
        )
    return rows


def choose_budget(risk: float, base: str, mid: str, high: str, low_threshold: float, high_threshold: float) -> str:
    if risk >= high_threshold:
        return high
    if risk >= low_threshold:
        return mid
    return base


def policy_name(base: str, mid: str, high: str, low_threshold: float, high_threshold: float) -> str:
    return f"{base}->{mid}->{high} by calibrated risk >= {low_threshold:.3f}/{high_threshold:.3f}"


def best_by_group(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    best = {}
    for row in rows:
        group = row[key]
        if group not in best or candidate_sort_key(row) > candidate_sort_key(best[group]):
            best[group] = row
    return [best[group] for group in sorted(best)]


def candidate_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        finite_float(row.get("objective", 0.0)),
        finite_float(row.get("dev_accuracy", 0.0)),
        -finite_float(row.get("dev_hFPR", 0.0)),
        -finite_float(row.get("dev_mean_keep_ratio", 0.0)),
        finite_float(row.get("dev_mean_ecr", 0.0)),
    )


def load_runs(run_map: dict[str, str]) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, dict[str, Any]]], list[str]]:
    scores = {}
    traces = {}
    for budget, rel_path in run_map.items():
        path = ROOT / rel_path
        scores[budget] = load_jsonl_by_id(path / "sample_scores.jsonl")
        traces[budget] = load_jsonl_by_id(path / "prune_traces.jsonl")
    ids = sorted(set.intersection(*(set(rows) for rows in scores.values())))
    if not ids:
        raise ValueError("No common sample ids across score runs.")
    return scores, traces, ids


def load_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row.get("sample_id", row.get("id", "")))] = row
    return rows


def stable_split(sample_id_value: str, dev_buckets: int) -> str:
    image_id = sample_id_value.split(":", 1)[0]
    bucket = int(hashlib.sha1(image_id.encode("utf-8")).hexdigest()[:8], 16) % 10
    return "dev" if bucket < dev_buckets else "test"


def finite_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(out) or math.isinf(out):
        return 0.0
    return out


def prefixed(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_report(
    fixed_rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    best_rows: list[dict[str, Any]],
    ids: list[str],
) -> str:
    lines = [
        "# TextOCR-Hard Calibrated Risk Predictor",
        "",
        f"Samples: {len(ids)}. The split is image-level, so paired positive/negative probes from one image stay together.",
        "",
        "This experiment trains a small logistic risk predictor on the development split and uses it to decide whether a low-budget Qwen target-conditioned run should escalate to a larger visual-token budget. Features are deployable low-budget signals only: answer margin/entropy, existing RICE-style confidence scores, target-token length, predicted answer, and retained-token spatial spread. Ground-truth polarity is never used as a feature.",
        "",
        "## Best Learned Policies",
        "",
        "| Target | Policy | test Acc. | test hFPR | test Keep | test Cascade | test ECR | Interpretation |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    fixed_test_best = max(fixed_rows, key=lambda row: row["test_accuracy"])
    fixed_efficiency = min((row for row in fixed_rows if row["budget"] in {"0.20", "0.25", "0.30"}), key=lambda row: abs(row["test_mean_keep_ratio"] - 0.25))
    for row in best_rows:
        interpretation = interpret(row, fixed_test_best, fixed_efficiency)
        lines.append(
            f"| {row['target']} | {row['policy']} | {row['test_accuracy']:.3f} | {row['test_hFPR']:.3f} | "
            f"{row['test_mean_keep_ratio']:.3f} | {row['test_cascade_keep_ratio']:.3f} | {row['test_mean_ecr']:.3f} | {interpretation} |"
        )
    lines.extend(
        [
            "",
            "## Fixed-Budget Reference",
            "",
            "| Policy | dev Acc. | dev hFPR | test Acc. | test hFPR | test Keep | test ECR |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in fixed_rows:
        lines.append(
            f"| {row['policy']} | {row['dev_accuracy']:.3f} | {row['dev_hFPR']:.3f} | "
            f"{row['test_accuracy']:.3f} | {row['test_hFPR']:.3f} | "
            f"{row['test_mean_keep_ratio']:.3f} | {row['test_mean_ecr']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Top Candidate Policies",
            "",
            "| Target | Policy | dev Acc. | dev hFPR | dev Keep | test Acc. | test hFPR | test Keep | test Cascade | test ECR |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in candidates:
        lines.append(
            f"| {row['target']} | {row['policy']} | {row['dev_accuracy']:.3f} | {row['dev_hFPR']:.3f} | "
            f"{row['dev_mean_keep_ratio']:.3f} | {row['test_accuracy']:.3f} | {row['test_hFPR']:.3f} | "
            f"{row['test_mean_keep_ratio']:.3f} | {row['test_cascade_keep_ratio']:.3f} | {row['test_mean_ecr']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The learned risk scores are useful diagnostics but do not close the oracle frontier gap.",
            "- Held-out policies can reduce mean final keep relative to fixed 0.30, but the gain comes with lower accuracy and/or higher hFPR.",
            "- This supports the current paper stance: unified adaptive risk control remains open, and the safe method claim should stay fixed-budget or detector/source scoped.",
        ]
    )
    return "\n".join(lines) + "\n"


def interpret(row: dict[str, Any], fixed_test_best: dict[str, Any], fixed_efficiency: dict[str, Any]) -> str:
    if row["test_accuracy"] > fixed_test_best["test_accuracy"] and row["test_mean_keep_ratio"] <= fixed_test_best["test_mean_keep_ratio"]:
        return "beats the fixed test-accuracy frontier"
    if row["test_accuracy"] >= fixed_efficiency["test_accuracy"] and row["test_mean_keep_ratio"] < fixed_efficiency["test_mean_keep_ratio"]:
        return "improves the efficiency point but not the quality frontier"
    return "does not beat the fixed-budget frontier; treat as diagnostic"


if __name__ == "__main__":
    main()
