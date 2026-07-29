#!/usr/bin/env python3
"""Compare SCOPE saliency-plus-coverage with its pure-coverage ablation."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    positive = [row for row in rows if row["binary_polarity"] == "positive"]
    negative = [row for row in rows if row["binary_polarity"] == "negative"]
    correct_positive = [row for row in positive if row["correct"]]
    correct_ecr = [float(row["prune_ecr"]) for row in correct_positive]
    probabilities = [
        1.0 / (1.0 + math.exp(-(float(row["no_loss"]) - float(row["yes_loss"]))))
        for row in rows
    ]
    targets = [float(row["binary_polarity"] == "positive") for row in rows]
    confidence = [max(probability, 1.0 - probability) for probability in probabilities]
    correctness = [float(bool(row["correct"])) for row in rows]
    return {
        "n": len(rows),
        "accuracy": fmean(float(bool(row["correct"])) for row in rows),
        "positive_accuracy": fmean(float(bool(row["correct"])) for row in positive),
        "hfpr": fmean(float(row["pred_answer"] == "yes") for row in negative),
        "keep_ratio": fmean(float(row["prune_keep_ratio"]) for row in rows),
        "positive_ecr": fmean(float(row["prune_ecr"]) for row in positive),
        "negative_source_coverage": fmean(float(row["prune_ecr"]) for row in negative),
        "correct_positive_count": len(correct_positive),
        "correct_positive_mean_ecr": fmean(correct_ecr),
        "correct_positive_ecr_lt_0p50": fmean(float(value < 0.5) for value in correct_ecr),
        "correct_positive_ecr_eq_0": fmean(float(value <= 1e-12) for value in correct_ecr),
        "sequence_likelihood_mean_confidence": fmean(confidence),
        "sequence_likelihood_ece_15": expected_calibration_error(confidence, correctness, bins=15),
        "sequence_likelihood_brier": fmean(
            (probability - target) ** 2 for probability, target in zip(probabilities, targets)
        ),
        "sequence_likelihood_nll": -fmean(
            target * math.log(max(probability, 1e-12))
            + (1.0 - target) * math.log(max(1.0 - probability, 1e-12))
            for probability, target in zip(probabilities, targets)
        ),
    }


def expected_calibration_error(
    confidence: list[float], correctness: list[float], *, bins: int
) -> float:
    if len(confidence) != len(correctness) or not confidence:
        raise ValueError("ECE requires equally sized, non-empty confidence and correctness arrays")
    total = len(confidence)
    error = 0.0
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        members = [
            item
            for item, value in enumerate(confidence)
            if low <= value < high or (index == bins - 1 and value == high)
        ]
        if members:
            mean_confidence = fmean(confidence[item] for item in members)
            mean_accuracy = fmean(correctness[item] for item in members)
            error += len(members) / total * abs(mean_confidence - mean_accuracy)
    return error


def paired_by_image(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    value: Callable[[dict[str, Any], dict[str, Any]], float | None],
) -> dict[str, list[float]]:
    right_by_id = {str(row["sample_id"]): row for row in right}
    clusters: dict[str, list[float]] = defaultdict(list)
    for left_row in left:
        sample_id = str(left_row["sample_id"])
        if sample_id not in right_by_id:
            raise ValueError(f"Missing paired sample {sample_id}")
        right_row = right_by_id[sample_id]
        if str(left_row["image_id"]) != str(right_row["image_id"]):
            raise ValueError(f"Image mismatch for {sample_id}")
        difference = value(left_row, right_row)
        if difference is not None:
            clusters[str(left_row["image_id"])].append(float(difference))
    return clusters


def cluster_mean(clusters: dict[str, list[float]]) -> float:
    return fmean(value for cluster in clusters.values() for value in cluster)


def cluster_bootstrap_ci(
    clusters: dict[str, list[float]], *, draws: int, seed: int
) -> tuple[float, float]:
    keys = sorted(clusters)
    if not keys:
        raise ValueError("Cannot bootstrap an empty comparison")
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        selected = [rng.choice(keys) for _ in keys]
        samples.append(fmean(value for key in selected for value in clusters[key]))
    samples.sort()
    return samples[int(0.025 * (draws - 1))], samples[int(0.975 * (draws - 1))]


def comparison(
    left: list[dict[str, Any]], right: list[dict[str, Any]], draws: int, seed: int
) -> dict[str, Any]:
    definitions: dict[str, Callable[[dict[str, Any], dict[str, Any]], float | None]] = {
        "accuracy": lambda l, r: float(bool(l["correct"])) - float(bool(r["correct"])),
        "hfpr": lambda l, r: (
            float(l["pred_answer"] == "yes") - float(r["pred_answer"] == "yes")
            if l["binary_polarity"] == "negative" else None
        ),
        "positive_ecr": lambda l, r: (
            float(l["prune_ecr"]) - float(r["prune_ecr"])
            if l["binary_polarity"] == "positive" else None
        ),
        "negative_source_coverage": lambda l, r: (
            float(l["prune_ecr"]) - float(r["prune_ecr"])
            if l["binary_polarity"] == "negative" else None
        ),
    }
    result: dict[str, Any] = {
        "bootstrap_unit": "image_id", "bootstrap_draws": draws, "seed": seed
    }
    for offset, (name, value) in enumerate(definitions.items()):
        clusters = paired_by_image(left, right, value)
        result[name] = {
            "difference": cluster_mean(clusters),
            "ci_95": list(cluster_bootstrap_ci(clusters, draws=draws, seed=seed + offset)),
            "num_images": len(clusters),
        }
    return result


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# SCOPE Pure-Coverage Ablation", "",
        "Differences are pure coverage (alpha=0) minus saliency plus coverage (alpha=1).",
        "Confidence intervals use paired image-cluster bootstrap.", "",
        "| Split | Variant | Acc. | hFPR | PosECR | NegSRC | ECE15 | Brier | Correct-positive low/zero ECR |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("development", "confirmation"):
        for variant in ("saliency_plus_coverage", "pure_coverage"):
            row = report[split][variant]
            lines.append(
                f"| {split} | {variant.replace('_', ' ')} | {row['accuracy']:.3f} | "
                f"{row['hfpr']:.3f} | {row['positive_ecr']:.3f} | "
                f"{row['negative_source_coverage']:.3f} | "
                f"{row['sequence_likelihood_ece_15']:.3f} | "
                f"{row['sequence_likelihood_brier']:.3f} | "
                f"{100 * row['correct_positive_ecr_lt_0p50']:.1f}% / "
                f"{100 * row['correct_positive_ecr_eq_0']:.1f}% |"
            )
    lines.extend(["", "| Confirmation difference | Estimate | 95% CI |", "|---|---:|---:|"])
    for name in ("accuracy", "hfpr", "positive_ecr", "negative_source_coverage"):
        row = report["confirmation_comparison"][name]
        lines.append(
            f"| {name.replace('_', ' ')} | {row['difference']:+.3f} | "
            f"[{row['ci_95'][0]:+.3f}, {row['ci_95'][1]:+.3f}] |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve(path: str) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else ROOT / candidate


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-alpha1", required=True)
    parser.add_argument("--dev-alpha0", required=True)
    parser.add_argument("--conf-alpha1", required=True)
    parser.add_argument("--conf-alpha0", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()

    names = ("dev_alpha1", "dev_alpha0", "conf_alpha1", "conf_alpha0")
    paths = {name: resolve(getattr(args, name)) for name in names}
    rows = {name: load_jsonl(path) for name, path in paths.items()}
    report = {
        "inputs": {name: display_path(path) for name, path in paths.items()},
        "development": {
            "saliency_plus_coverage": summarize(rows["dev_alpha1"]),
            "pure_coverage": summarize(rows["dev_alpha0"]),
        },
        "confirmation": {
            "saliency_plus_coverage": summarize(rows["conf_alpha1"]),
            "pure_coverage": summarize(rows["conf_alpha0"]),
        },
        "confirmation_comparison": comparison(
            rows["conf_alpha0"], rows["conf_alpha1"], args.bootstrap, args.seed
        ),
    }
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "scope_coverage_ablation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(output_dir / "scope_coverage_ablation.md", report)
    print(f"Wrote SCOPE coverage ablation to {output_dir}")


if __name__ == "__main__":
    main()
