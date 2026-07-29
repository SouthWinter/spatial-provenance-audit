#!/usr/bin/env python
"""Summarize TextOCR-Hard blank-image and collision-free mismatch controls."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recap.metrics import roc_auc
from scripts.calibrate_yesno_thresholds import stable_split


MODEL_CONFIG = {
    "qwen": {
        "display": "Qwen3-VL-8B",
        "threshold": 0.0,
        "baseline": Path("runs/textocr_confirmation/qwen3_8b_full/probe_scores.jsonl"),
        "split": "all",
        "lexical_details": Path(
            "runs/textocr_confirmation/lexical_audit/hard_negative_lexical_details.csv"
        ),
    },
    "llava": {
        "display": "LLaVA-1.5-7B",
        "threshold": 0.0,
        "baseline": Path("runs/textocr_confirmation/llava15_7b_full/probe_scores.jsonl"),
        "split": "all",
        "lexical_details": Path(
            "runs/textocr_confirmation/lexical_audit/hard_negative_lexical_details.csv"
        ),
    },
    "internvl": {
        "display": "InternVL3.5-8B",
        "threshold": 2.0425992012023926,
        "baseline": Path(
            "runs/internvl_textocr_hard/"
            "internvl35_8b_textocr_hard_full1000_direct/probe_scores.jsonl"
        ),
        "split": "test",
        "lexical_details": Path(
            "runs/problem_optimization_audit/hard_negative_lexical_audit/"
            "hard_negative_lexical_details.csv"
        ),
    },
}

CONDITIONS = (
    "blank",
    "image_mismatch_seed101",
    "image_mismatch_seed202",
    "image_mismatch_seed303",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("runs/problem_optimization_audit/image_free_controls"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/problem_optimization_audit/image_free_controls/report"),
    )
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--auroc-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()

    summary_rows: list[dict[str, Any]] = []
    plausible_rows: list[dict[str, Any]] = []
    lexical_rows: list[dict[str, Any]] = []
    for model, config in MODEL_CONFIG.items():
        baseline_path = config["baseline"]
        model_root = args.run_root / model
        if not baseline_path.exists() or not model_root.exists():
            continue
        baseline = filter_split(load_scores(baseline_path), str(config["split"]))
        baseline_by_id = {row["sample_id"]: row for row in baseline}
        lexical_details = load_lexical_details(Path(config["lexical_details"]))
        plausible_image_ids = decoy_seen_image_ids(lexical_details)
        baseline_metrics = evaluate(baseline, float(config["threshold"]))
        baseline_auc_ci = bootstrap_auroc(
            baseline, n_bootstrap=args.auroc_bootstrap, seed=args.seed
        )
        baseline_metrics["auroc_ci_low"], baseline_metrics["auroc_ci_high"] = baseline_auc_ci
        summary_rows.append(
            result_row(
                model,
                str(config["display"]),
                "matched_image_full",
                float(config["threshold"]),
                baseline_metrics,
                None,
                None,
            )
        )
        plausible_baseline = [
            row for row in baseline if row["image_id"] in plausible_image_ids
        ]
        if plausible_baseline:
            plausible_metrics = evaluate(
                plausible_baseline, float(config["threshold"])
            )
            plausible_ci = bootstrap_auroc(
                plausible_baseline,
                n_bootstrap=args.auroc_bootstrap,
                seed=args.seed,
            )
            plausible_metrics["auroc_ci_low"], plausible_metrics["auroc_ci_high"] = plausible_ci
            plausible_rows.append(
                result_row(
                    model,
                    str(config["display"]),
                    "matched_image_full",
                    float(config["threshold"]),
                    plausible_metrics,
                    None,
                    None,
                )
            )

        for condition in CONDITIONS:
            score_path = model_root / condition / "probe_scores.jsonl"
            if not score_path.exists():
                continue
            rows = filter_split(load_scores(score_path), str(config["split"]))
            aligned = [
                row for row in rows if row["sample_id"] in baseline_by_id
            ]
            if len(aligned) != len(rows) or len(aligned) != len(baseline):
                raise ValueError(
                    f"{model}/{condition} does not align with baseline: "
                    f"{len(aligned)} control, {len(baseline)} baseline"
                )
            metrics = evaluate(aligned, float(config["threshold"]))
            auc_ci = bootstrap_auroc(
                aligned, n_bootstrap=args.auroc_bootstrap, seed=args.seed
            )
            metrics["auroc_ci_low"], metrics["auroc_ci_high"] = auc_ci
            delta, ci = paired_accuracy_difference(
                aligned,
                baseline_by_id,
                threshold=float(config["threshold"]),
                n_bootstrap=args.bootstrap,
                seed=args.seed,
            )
            summary_rows.append(
                result_row(
                    model,
                    str(config["display"]),
                    condition,
                    float(config["threshold"]),
                    metrics,
                    delta,
                    ci,
                )
            )
            plausible_control = [
                row for row in aligned if row["image_id"] in plausible_image_ids
            ]
            if plausible_control:
                plausible_metrics = evaluate(
                    plausible_control, float(config["threshold"])
                )
                plausible_auc_ci = bootstrap_auroc(
                    plausible_control,
                    n_bootstrap=args.auroc_bootstrap,
                    seed=args.seed,
                )
                (
                    plausible_metrics["auroc_ci_low"],
                    plausible_metrics["auroc_ci_high"],
                ) = plausible_auc_ci
                plausible_delta, plausible_delta_ci = paired_accuracy_difference(
                    plausible_control,
                    baseline_by_id,
                    threshold=float(config["threshold"]),
                    n_bootstrap=args.bootstrap,
                    seed=args.seed,
                )
                plausible_rows.append(
                    result_row(
                        model,
                        str(config["display"]),
                        condition,
                        float(config["threshold"]),
                        plausible_metrics,
                        plausible_delta,
                        plausible_delta_ci,
                    )
                )
            if condition == "blank":
                lexical_rows.extend(
                    blank_lexical_strata(
                        model,
                        aligned,
                        lexical_details,
                    )
                )

    if not summary_rows:
        raise RuntimeError(f"No completed image-free runs found under {args.run_root}")

    mismatch_seed_rows = summarize_mismatch_seeds(summary_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "image_free_summary.csv", summary_rows)
    write_csv(args.output_dir / "image_mismatch_seed_summary.csv", mismatch_seed_rows)
    write_csv(args.output_dir / "lexically_plausible_summary.csv", plausible_rows)
    write_csv(args.output_dir / "blank_lexical_strata.csv", lexical_rows)
    (args.output_dir / "image_free_report.md").write_text(
        markdown(summary_rows, mismatch_seed_rows, plausible_rows, lexical_rows),
        encoding="utf-8",
    )
    (args.output_dir / "image_free_report.json").write_text(
        json.dumps(
            {
                "summary": summary_rows,
                "image_mismatch_seed_summary": mismatch_seed_rows,
                "lexically_plausible_summary": plausible_rows,
                "blank_lexical_strata": lexical_rows,
                "threshold_contract": {
                    model: {
                        "threshold": config["threshold"],
                        "split": config["split"],
                        "baseline": str(config["baseline"]),
                    }
                    for model, config in MODEL_CONFIG.items()
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output_dir / 'image_free_report.md'}")


def load_scores(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(
                {
                    **row,
                    "sample_id": str(row["sample_id"]),
                    "image_id": str(row.get("image_id") or row["sample_id"].split(":", 1)[0]),
                    "target_yes": str(row.get("target_answer", "")).lower() == "yes",
                    "raw_margin": float(row["no_loss"]) - float(row["yes_loss"]),
                }
            )
    return rows


def filter_split(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    if split == "all":
        return rows
    return [
        row
        for row in rows
        if stable_split(str(row["image_id"]), dev_buckets=5) == split
    ]


def evaluate(rows: list[dict[str, Any]], threshold: float) -> dict[str, float]:
    positives = [row for row in rows if row["target_yes"]]
    negatives = [row for row in rows if not row["target_yes"]]
    if not positives or not negatives:
        raise ValueError("Evaluation requires positive and negative probes")

    predictions = [row["raw_margin"] >= threshold for row in rows]
    accuracy = statistics.fmean(
        prediction == row["target_yes"] for prediction, row in zip(predictions, rows)
    )
    positive_accuracy = statistics.fmean(
        row["raw_margin"] >= threshold for row in positives
    )
    hfpr = statistics.fmean(row["raw_margin"] >= threshold for row in negatives)
    auc = roc_auc(
        [1.0 if row["target_yes"] else 0.0 for row in rows],
        [float(row["raw_margin"]) for row in rows],
    )
    pair_gaps = paired_margin_gaps(rows)
    return {
        "n": len(rows),
        "images": len({row["image_id"] for row in rows}),
        "accuracy": accuracy,
        "positive_accuracy": positive_accuracy,
        "hfpr": hfpr,
        "negative_accuracy": 1.0 - hfpr,
        "balanced_accuracy": 0.5 * (positive_accuracy + 1.0 - hfpr),
        "auroc": auc,
        "yes_rate": statistics.fmean(predictions),
        "mean_positive_minus_negative_margin": statistics.fmean(pair_gaps),
        "pairwise_positive_margin_win_rate": statistics.fmean(gap > 0 for gap in pair_gaps),
    }


def paired_margin_gaps(rows: list[dict[str, Any]]) -> list[float]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["image_id"]].append(row)
    gaps = []
    for image_id, pair in groups.items():
        positives = [row for row in pair if row["target_yes"]]
        negatives = [row for row in pair if not row["target_yes"]]
        if len(positives) != 1 or len(negatives) != 1:
            raise ValueError(f"Expected one positive and one negative for {image_id}")
        gaps.append(positives[0]["raw_margin"] - negatives[0]["raw_margin"])
    return gaps


def paired_accuracy_difference(
    control: list[dict[str, Any]],
    baseline_by_id: dict[str, dict[str, Any]],
    *,
    threshold: float,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, tuple[float, float]]:
    by_image: dict[str, list[float]] = defaultdict(list)
    for row in control:
        baseline = baseline_by_id[row["sample_id"]]
        control_correct = (row["raw_margin"] >= threshold) == row["target_yes"]
        baseline_correct = (baseline["raw_margin"] >= threshold) == baseline["target_yes"]
        by_image[row["image_id"]].append(float(control_correct) - float(baseline_correct))
    image_deltas = [
        statistics.fmean(deltas) for _, deltas in sorted(by_image.items())
    ]
    observed = statistics.fmean(image_deltas)
    rng = random.Random(seed)
    draws = []
    for _ in range(n_bootstrap):
        draws.append(
            statistics.fmean(rng.choice(image_deltas) for _ in image_deltas)
        )
    draws.sort()
    return observed, (percentile(draws, 0.025), percentile(draws, 0.975))


def bootstrap_auroc(
    rows: list[dict[str, Any]],
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_image[row["image_id"]].append(row)
    image_groups = [pair for _, pair in sorted(by_image.items())]
    rng = random.Random(seed)
    draws = []
    for _ in range(n_bootstrap):
        sampled = [
            row
            for _ in image_groups
            for row in rng.choice(image_groups)
        ]
        draws.append(
            roc_auc(
                [1.0 if row["target_yes"] else 0.0 for row in sampled],
                [float(row["raw_margin"]) for row in sampled],
            )
        )
    draws.sort()
    return percentile(draws, 0.025), percentile(draws, 0.975)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    position = q * (len(values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def result_row(
    model: str,
    display: str,
    condition: str,
    threshold: float,
    metrics: dict[str, float],
    delta: float | None,
    ci: tuple[float, float] | None,
) -> dict[str, Any]:
    return {
        "model_key": model,
        "model": display,
        "condition": condition,
        "threshold": threshold,
        **metrics,
        "delta_accuracy_vs_matched": delta,
        "delta_accuracy_ci_low": None if ci is None else ci[0],
        "delta_accuracy_ci_high": None if ci is None else ci[1],
    }


def load_lexical_details(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return {row["sample_id"]: row for row in csv.DictReader(handle)}


def decoy_seen_image_ids(
    lexical_details: dict[str, dict[str, str]],
) -> set[str]:
    return {
        str(row["image_id"])
        for row in lexical_details.values()
        if (
            int(row.get("global_other_image_nfkc_casefold_matches", "0")) > 0
            or int(row.get("global_other_image_alnum_matches", "0")) > 0
        )
    }


def blank_lexical_strata(
    model: str,
    rows: list[dict[str, Any]],
    lexical_details: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_image[row["image_id"]].append(row)
    strata: dict[tuple[str, str], list[float]] = defaultdict(list)
    for pair in by_image.values():
        positive = next(row for row in pair if row["target_yes"])
        negative = next(row for row in pair if not row["target_yes"])
        source = str(positive.get("target_text", ""))
        decoy = str(negative.get("target_text", ""))
        gap = float(positive["raw_margin"]) - float(negative["raw_margin"])
        strata[("edit_type", edit_type(source, decoy))].append(gap)
        strata[("source_shape", text_shape(source))].append(gap)
        strata[("source_length", length_bucket(source))].append(gap)
        details = lexical_details.get(str(negative["sample_id"]))
        if details:
            plausible = (
                int(details.get("global_other_image_nfkc_casefold_matches", "0")) > 0
                or int(details.get("global_other_image_alnum_matches", "0")) > 0
            )
            strata[
                (
                    "decoy_seen_elsewhere_in_TextOCR",
                    "seen" if plausible else "unseen",
                )
            ].append(gap)
    output = []
    for (stratum, bucket), gaps in sorted(strata.items()):
        output.append(
            {
                "model_key": model,
                "stratum": stratum,
                "bucket": bucket,
                "image_pairs": len(gaps),
                "mean_positive_minus_negative_margin": statistics.fmean(gaps),
                "pairwise_positive_margin_win_rate": statistics.fmean(gap > 0 for gap in gaps),
            }
        )
    return output


def edit_type(source: str, decoy: str) -> str:
    if len(source) == len(decoy) and sum(a != b for a, b in zip(source, decoy)) == 1:
        return "single_substitution"
    if abs(len(source) - len(decoy)) == 1:
        longer, shorter = (source, decoy) if len(source) > len(decoy) else (decoy, source)
        if any(longer[:index] + longer[index + 1 :] == shorter for index in range(len(longer))):
            return "single_deletion_or_insertion"
    return "other"


def text_shape(text: str) -> str:
    classes = []
    if any(character.isalpha() for character in text):
        classes.append("alpha")
    if any(character.isdigit() for character in text):
        classes.append("digit")
    if any(not character.isalnum() for character in text):
        classes.append("punct")
    return "+".join(classes) or "other"


def length_bucket(text: str) -> str:
    length = len(text)
    if length <= 4:
        return "4_or_less"
    if length <= 7:
        return "5_to_7"
    return "8_or_more"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_mismatch_seeds(
    summary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for model_key, config in MODEL_CONFIG.items():
        rows = [
            row
            for row in summary
            if row["model_key"] == model_key
            and str(row["condition"]).startswith("image_mismatch_seed")
        ]
        if not rows:
            continue
        result: dict[str, Any] = {
            "model_key": model_key,
            "model": config["display"],
            "seeds_completed": len(rows),
            "n_per_seed": rows[0]["n"],
        }
        for metric in (
            "accuracy",
            "positive_accuracy",
            "hfpr",
            "auroc",
            "pairwise_positive_margin_win_rate",
            "delta_accuracy_vs_matched",
        ):
            values = [float(row[metric]) for row in rows]
            result[f"{metric}_mean"] = statistics.fmean(values)
            result[f"{metric}_min"] = min(values)
            result[f"{metric}_max"] = max(values)
        output.append(result)
    return output


def markdown(
    summary: list[dict[str, Any]],
    mismatch_seed_summary: list[dict[str, Any]],
    plausible: list[dict[str, Any]],
    lexical: list[dict[str, Any]],
) -> str:
    lines = [
        "# TextOCR-Hard Image-Free Control Audit",
        "",
        "Blank images preserve each source image's dimensions and aspect ratio. "
        "Each mismatch seed is a collision-free image-group derangement; paired "
        "positive and negative probes share the same wrong image. Thresholds are "
        "frozen from the matched-image protocol.",
        "",
        "| Model | Condition | n | Acc. | Pos. acc. | hFPR | AUROC (95% CI) | Pair margin win | $\\Delta$ Acc. vs. matched (95% CI) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        if row["delta_accuracy_vs_matched"] is None:
            delta = "--"
        else:
            delta = (
                f"{row['delta_accuracy_vs_matched']:+.3f} "
                f"[{row['delta_accuracy_ci_low']:+.3f},"
                f"{row['delta_accuracy_ci_high']:+.3f}]"
            )
        lines.append(
            f"| {row['model']} | {row['condition']} | {row['n']} | "
            f"{row['accuracy']:.3f} | {row['positive_accuracy']:.3f} | "
            f"{row['hfpr']:.3f} | {row['auroc']:.3f} "
            f"[{row['auroc_ci_low']:.3f},{row['auroc_ci_high']:.3f}] | "
            f"{row['pairwise_positive_margin_win_rate']:.3f} | {delta} |"
        )

    if mismatch_seed_summary:
        lines.extend(
            [
                "",
                "## Mismatch Seed Stability",
                "",
                "| Model | Seeds | Acc. mean [range] | hFPR mean [range] | AUROC mean [range] | $\\Delta$ Acc. mean [range] |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in mismatch_seed_summary:
            lines.append(
                f"| {row['model']} | {row['seeds_completed']} | "
                f"{row['accuracy_mean']:.3f} [{row['accuracy_min']:.3f},{row['accuracy_max']:.3f}] | "
                f"{row['hfpr_mean']:.3f} [{row['hfpr_min']:.3f},{row['hfpr_max']:.3f}] | "
                f"{row['auroc_mean']:.3f} [{row['auroc_min']:.3f},{row['auroc_max']:.3f}] | "
                f"{row['delta_accuracy_vs_matched_mean']:+.3f} "
                f"[{row['delta_accuracy_vs_matched_min']:+.3f},"
                f"{row['delta_accuracy_vs_matched_max']:+.3f}] |"
            )

    if plausible:
        lines.extend(
            [
                "",
                "## Lexically Plausible Decoy Subset",
                "",
                "This subset retains image pairs whose decoy occurs as a real OCR token "
                "in another TextOCR image, reducing the real-string versus synthetic-string confound.",
                "",
                "| Model | Condition | n | Acc. | Pos. acc. | hFPR | AUROC (95% CI) | $\\Delta$ Acc. vs. matched (95% CI) |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in plausible:
            if row["delta_accuracy_vs_matched"] is None:
                delta = "--"
            else:
                delta = (
                    f"{row['delta_accuracy_vs_matched']:+.3f} "
                    f"[{row['delta_accuracy_ci_low']:+.3f},"
                    f"{row['delta_accuracy_ci_high']:+.3f}]"
                )
            lines.append(
                f"| {row['model']} | {row['condition']} | {row['n']} | "
                f"{row['accuracy']:.3f} | {row['positive_accuracy']:.3f} | "
                f"{row['hfpr']:.3f} | {row['auroc']:.3f} "
                f"[{row['auroc_ci_low']:.3f},{row['auroc_ci_high']:.3f}] | "
                f"{delta} |"
            )

    if lexical:
        lines.extend(
            [
                "",
                "## Blank-Image Lexical Strata",
                "",
                "| Model | Stratum | Bucket | Pairs | Mean positive-negative margin | Positive margin win |",
                "|---|---|---|---:|---:|---:|",
            ]
        )
        display = {key: value["display"] for key, value in MODEL_CONFIG.items()}
        for row in lexical:
            lines.append(
                f"| {display[row['model_key']]} | {row['stratum']} | {row['bucket']} | "
                f"{row['image_pairs']} | "
                f"{row['mean_positive_minus_negative_margin']:.3f} | "
                f"{row['pairwise_positive_margin_win_rate']:.3f} |"
            )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
