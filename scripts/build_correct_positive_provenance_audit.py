#!/usr/bin/env python3
"""Audit correct positive answers whose retained tokens have low spatial provenance."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = (
    ("Qwen Target (30%)", "runs/textocr_confirmation/qwen3_8b_target_0p30/probe_scores.jsonl"),
    ("Qwen Random (30%)", "runs/textocr_confirmation/qwen3_8b_random_0p30/probe_scores.jsonl"),
    ("Qwen Grid (30%)", "runs/textocr_confirmation/qwen3_8b_grid_0p30/probe_scores.jsonl"),
    ("LLaVA Protected (40%)", "runs/textocr_confirmation/llava15_7b_protected_0p40/probe_scores.jsonl"),
    ("LLaVA Random (40%)", "runs/textocr_confirmation/llava15_7b_random_0p40/probe_scores.jsonl"),
    ("LLaVA Target (40%)", "runs/textocr_confirmation/llava15_7b_target_0p40/probe_scores.jsonl"),
    ("LLaVA SCOPE (40%)", "runs/textocr_confirmation/llava15_7b_scope_0p40/probe_scores.jsonl"),
    (
        "LLaVA SCOPE pure coverage (40%)",
        "runs/official_baseline_extension/scope_pure_coverage_alpha0/"
        "conf_llava15_7b_scope_alpha0_0p40/probe_scores.jsonl",
    ),
    ("LLaVA CoIn (40%)", "runs/textocr_confirmation/llava15_7b_coin_0p40/probe_scores.jsonl"),
    ("LLaVA VisionZip (40%)", "runs/textocr_confirmation/llava15_7b_visionzip_0p40/probe_scores.jsonl"),
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return center - radius, center + radius


def summarize(label: str, path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [row for row in rows if row.get("binary_polarity") == "positive"]
    correct = [row for row in positives if row.get("correct") is True]
    missing = [row.get("sample_id", "") for row in correct if row.get("prune_ecr") is None]
    if missing:
        raise ValueError(f"{label}: {len(missing)} correct positive rows lack prune_ecr")
    anchor_ecr = [float(row.get("prune_anchor_ecr", row["prune_ecr"])) for row in correct]
    is_visionzip = "visionzip" in label.casefold()
    ecr = [1.0 for _ in correct] if is_visionzip else anchor_ecr
    if any(value < 0.0 or value > 1.0 for value in ecr):
        raise ValueError(f"{label}: prune_ecr lies outside [0, 1]")
    low = sum(value < 0.5 for value in ecr)
    zero = sum(value <= 1e-12 for value in ecr)
    denominator = len(correct)
    low_ci = wilson_interval(low, denominator)
    zero_ci = wilson_interval(zero, denominator)
    return {
        "method": label,
        "score_file": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
        "positive_probes": len(positives),
        "correct_positive_probes": denominator,
        "correct_positive_accuracy": denominator / len(positives) if positives else 0.0,
        "mean_ecr_given_correct_positive": fmean(ecr) if ecr else 0.0,
        "mean_anchor_ecr_given_correct_positive": fmean(anchor_ecr) if anchor_ecr else 0.0,
        "correct_positive_ecr_lt_0p50_count": low,
        "correct_positive_ecr_lt_0p50_rate": low / denominator if denominator else 0.0,
        "correct_positive_ecr_lt_0p50_ci_low": low_ci[0],
        "correct_positive_ecr_lt_0p50_ci_high": low_ci[1],
        "correct_positive_ecr_eq_0_count": zero,
        "correct_positive_ecr_eq_0_rate": zero / denominator if denominator else 0.0,
        "correct_positive_ecr_eq_0_ci_low": zero_ci[0],
        "correct_positive_ecr_eq_0_ci_high": zero_ci[1],
    }


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("run must contain a non-empty label and path")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return label.strip(), path.resolve()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Correct-Positive Spatial-Provenance Audit",
        "",
        "This locked-confirmation audit conditions on positive probes answered correctly. "
        "ECR is geometric overlap between retained token cells and the annotated positive region; "
        "it is not an information-theoretic measure of whether contextualized tokens encode the region.",
        "",
        "| Method | Correct positive | Mean lineage ECR | Mean anchor ECR | ECR < 0.50 | ECR = 0 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        low = row["correct_positive_ecr_lt_0p50_count"]
        low_rate = row["correct_positive_ecr_lt_0p50_rate"]
        low_ci = (
            row["correct_positive_ecr_lt_0p50_ci_low"],
            row["correct_positive_ecr_lt_0p50_ci_high"],
        )
        zero = row["correct_positive_ecr_eq_0_count"]
        zero_rate = row["correct_positive_ecr_eq_0_rate"]
        zero_ci = (
            row["correct_positive_ecr_eq_0_ci_low"],
            row["correct_positive_ecr_eq_0_ci_high"],
        )
        lines.append(
            f"| {row['method']} | {row['correct_positive_probes']} | "
            f"{row['mean_ecr_given_correct_positive']:.3f} | "
            f"{row['mean_anchor_ecr_given_correct_positive']:.3f} | "
            f"{low} ({100 * low_rate:.1f}%; {100 * low_ci[0]:.1f}--{100 * low_ci[1]:.1f}) | "
            f"{zero} ({100 * zero_rate:.1f}%; {100 * zero_ci[0]:.1f}--{100 * zero_ci[1]:.1f}) |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run,
        help="Run specification LABEL=PATH; repeat for multiple runs. Defaults to locked confirmation runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs/problem_optimization_audit/correct_positive_provenance",
    )
    args = parser.parse_args()

    run_specs = args.run or [(label, (ROOT / relative).resolve()) for label, relative in DEFAULT_RUNS]
    summaries = []
    for label, path in run_specs:
        if not path.exists():
            raise FileNotFoundError(path)
        summaries.append(summarize(label, path, load_jsonl(path)))

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "correct_positive_provenance.json"
    csv_path = output_dir / "correct_positive_provenance.csv"
    markdown_path = output_dir / "correct_positive_provenance.md"
    json_path.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    write_csv(csv_path, summaries)
    write_markdown(markdown_path, summaries)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
