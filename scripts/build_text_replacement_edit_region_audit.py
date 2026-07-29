#!/usr/bin/env python3
"""Summarize exact edit-region coverage for human-validated replacements."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from statistics import fmean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recap.prune.metrics import evidence_coverage, make_token_grid


DEFAULT_ROOT = ROOT / "runs/problem_optimization_audit/text_replacement_edit_region"
METHODS = ("target", "random", "grid")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--probes",
        type=Path,
        default=ROOT / "data/textocr_counterfactual/text_replacement_edit_region_valid102.jsonl",
    )
    parser.add_argument(
        "--full-scores",
        type=Path,
        default=ROOT
        / "runs/problem_optimization_audit/text_replacement_control_pack_v3"
        / "cross_backbone_eval/qwen3_8b/probe_scores.jsonl",
    )
    args = parser.parse_args()

    probes = index_jsonl(args.probes.resolve())
    runs_root = args.runs_root.resolve()
    detail_by_method: dict[str, list[dict[str, Any]]] = {}
    summaries: list[dict[str, Any]] = []

    full_scores = [
        row
        for row in read_jsonl(args.full_scores.resolve())
        if row.get("sample_id") in probes
    ]
    summaries.append(
        {
            "method": "Full",
            "n": len(full_scores),
            "accuracy": fmean(float(row["correct"]) for row in full_scores),
            "word_ecr": 1.0,
            "edit_ecr": 1.0,
            "edit_miss_rate": 0.0,
            "word_kept_edit_missed_rate": 0.0,
        }
    )

    for method in METHODS:
        run_dir = runs_root / method
        scores = index_jsonl(run_dir / "probe_scores.jsonl")
        traces = index_jsonl(run_dir / "prune_traces.jsonl")
        detail: list[dict[str, Any]] = []
        for sample_id, probe in probes.items():
            score = scores[sample_id]
            trace = traces[sample_id]
            word_ecr = coverage_from_trace(trace, probe["word_evidence_regions"])
            edit_ecr = float(trace["ecr"])
            detail.append(
                {
                    "sample_id": sample_id,
                    "image_id": probe["image_id"],
                    "method": method,
                    "correct": int(bool(score["correct"])),
                    "margin": float(score["margin"]),
                    "word_ecr": word_ecr,
                    "edit_ecr": edit_ecr,
                    "edit_missed": int(edit_ecr == 0.0),
                    "word_kept_edit_missed": int(word_ecr >= 0.5 and edit_ecr < 0.5),
                }
            )
        detail_by_method[method] = detail
        summaries.append(summarize(method, detail))

    comparisons = [
        compare(detail_by_method["target"], detail_by_method[other], other)
        for other in ("random", "grid")
    ]
    runs_root.mkdir(parents=True, exist_ok=True)
    write_csv(runs_root / "edit_region_summary.csv", summaries)
    write_csv(
        runs_root / "edit_region_rows.csv",
        [row for method in METHODS for row in detail_by_method[method]],
    )
    write_csv(runs_root / "edit_region_paired_comparisons.csv", comparisons)
    write_markdown(runs_root / "edit_region_audit.md", summaries, comparisons)
    print(f"Wrote edit-region audit to {runs_root}")


def coverage_from_trace(trace: dict[str, Any], regions: list[list[float]]) -> float:
    grid_h = int(trace["grid_h"])
    grid_w = int(trace["grid_w"])
    full_tokens = int(trace["full_visual_tokens"])
    merge_area = grid_h * grid_w / full_tokens
    merge_size = int(round(math.sqrt(merge_area)))
    if merge_size <= 0 or (grid_h // merge_size) * (grid_w // merge_size) != full_tokens:
        raise ValueError(f"Cannot recover token grid from trace {trace['sample_id']}")
    token_boxes = make_token_grid(grid_h // merge_size, grid_w // merge_size)
    boxes = [tuple(float(value) for value in box) for box in regions]
    return evidence_coverage(trace["kept_indices"], token_boxes, boxes)


def summarize(method: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "method": method.capitalize(),
        "n": len(rows),
        "accuracy": fmean(row["correct"] for row in rows),
        "word_ecr": fmean(row["word_ecr"] for row in rows),
        "edit_ecr": fmean(row["edit_ecr"] for row in rows),
        "edit_miss_rate": fmean(row["edit_missed"] for row in rows),
        "word_kept_edit_missed_rate": fmean(
            row["word_kept_edit_missed"] for row in rows
        ),
    }


def compare(
    target: list[dict[str, Any]], other: list[dict[str, Any]], label: str
) -> dict[str, Any]:
    target_by_id = {row["sample_id"]: row for row in target}
    other_by_id = {row["sample_id"]: row for row in other}
    ids = sorted(target_by_id.keys() & other_by_id.keys())
    result: dict[str, Any] = {"comparison": f"Target - {label.capitalize()}", "n": len(ids)}
    for field in ("accuracy", "edit_ecr", "word_ecr"):
        source = "correct" if field == "accuracy" else field
        differences = [
            float(target_by_id[sample_id][source])
            - float(other_by_id[sample_id][source])
            for sample_id in ids
        ]
        low, high = bootstrap_ci(differences, seed=20260724 + len(result))
        result[f"{field}_diff"] = fmean(differences)
        result[f"{field}_ci_low"] = low
        result[f"{field}_ci_high"] = high
    return result


def bootstrap_ci(values: list[float], *, seed: int, draws: int = 20000) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    samples = sorted(
        fmean(values[rng.randrange(n)] for _ in range(n)) for _ in range(draws)
    )
    return samples[int(0.025 * draws)], samples[int(0.975 * draws)]


def index_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["sample_id"]): row for row in read_jsonl(path)}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path, summaries: list[dict[str, Any]], comparisons: list[dict[str, Any]]
) -> None:
    lines = [
        "# Human-Validated Edit-Region Audit",
        "",
        "The region is the renderer-derived glyph box at the unique substituted-character "
        "position. All 102 edits passed human semantic QC.",
        "",
        "| Method | n | Accuracy | Word ECR | Edit ECR | Edit ECR=0 | Word>=0.5, Edit<0.5 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['method']} | {row['n']} | {row['accuracy']:.3f} | "
            f"{row['word_ecr']:.3f} | {row['edit_ecr']:.3f} | "
            f"{row['edit_miss_rate']:.3f} | {row['word_kept_edit_missed_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "| Comparison | Accuracy diff [95% CI] | Edit-ECR diff [95% CI] | Word-ECR diff [95% CI] |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        lines.append(
            f"| {row['comparison']} | {format_ci(row, 'accuracy')} | "
            f"{format_ci(row, 'edit_ecr')} | {format_ci(row, 'word_ecr')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_ci(row: dict[str, Any], prefix: str) -> str:
    return (
        f"{row[f'{prefix}_diff']:+.3f} "
        f"[{row[f'{prefix}_ci_low']:+.3f}, {row[f'{prefix}_ci_high']:+.3f}]"
    )


if __name__ == "__main__":
    main()
