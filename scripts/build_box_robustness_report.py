#!/usr/bin/env python
"""Summarize TextOCR-Hard pruning robustness under noisy OCR boxes."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recap.prune.metrics import (
    evidence_center_recall,
    evidence_coverage,
    evidence_patch_recall,
    evidence_regions_from_sample,
    make_token_grid,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-probes", default="data/textocr_val_hard_probes_500img.jsonl")
    parser.add_argument("--output-dir", default="runs/box_robustness")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help=(
            "model,variant,selector,score_dir[,trace_dir]. Can be passed multiple times. "
            "Use trace_dir when score_dir contains calibrated probe scores but no pruning traces."
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_specs = list(args.run)
    existing_summary = out_dir / "box_robustness_summary.csv"
    if not run_specs and existing_summary.exists():
        with existing_summary.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                run_specs.append(
                    ",".join(
                        [row["model"], row["variant"], row["selector"], row["score_dir"], row["trace_dir"]]
                    )
                )
    if not run_specs:
        raise ValueError("Pass at least one --run, or retain an existing summary for deterministic rebuilding")

    oracle_by_id = load_oracle(args.oracle_probes)
    rows = [summarize_run(spec, oracle_by_id) for spec in run_specs]

    write_csv(out_dir / "box_robustness_summary.csv", rows)
    write_markdown(out_dir / "box_robustness_report.md", rows)
    print(f"Wrote {len(rows)} rows to {out_dir}")


def summarize_run(spec: str, oracle_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    model, variant, selector, score_dir_text, trace_dir_text = parse_run_spec(spec)
    score_dir = Path(score_dir_text)
    trace_dir = Path(trace_dir_text)
    metrics = json.loads((score_dir / "metrics.json").read_text(encoding="utf-8"))
    traces = read_jsonl(trace_dir / "prune_traces.jsonl")
    sample_scores = read_jsonl(score_dir / "sample_scores.jsonl")
    probe_scores_path = score_dir / "probe_scores.jsonl"
    if probe_scores_path.exists():
        eval_ids = {str(row.get("sample_id", "")) for row in read_jsonl(probe_scores_path)}
        eval_ids.discard("")
        if eval_ids:
            traces = [trace for trace in traces if str(trace.get("sample_id", "")) in eval_ids]

    true_rows: list[dict[str, float]] = []
    for trace in traces:
        sample_id = str(trace.get("sample_id", ""))
        oracle_info = oracle_by_id.get(sample_id, {})
        oracle = oracle_info.get("regions", [])
        polarity = str(oracle_info.get("binary_polarity", ""))
        token_boxes = token_boxes_from_trace(trace)
        kept = [int(idx) for idx in trace.get("kept_indices", [])]
        true_rows.append(
            {
                "true_ecr": evidence_coverage(kept, token_boxes, oracle),
                "true_center": evidence_center_recall(kept, token_boxes, oracle),
                "true_patch": evidence_patch_recall(kept, token_boxes, oracle),
                "reported_ecr": float(trace.get("ecr", 0.0) or 0.0),
                "keep_ratio": float(trace.get("effective_keep_ratio", 0.0) or 0.0),
                "overhead_ms": float(trace.get("prune_overhead_ms", 0.0) or 0.0),
                "binary_polarity": polarity,
            }
        )

    return {
        "model": model,
        "variant": variant,
        "selector": selector,
        "score_dir": str(score_dir),
        "trace_dir": str(trace_dir),
        "num_samples": int(metrics.get("num_samples", len(sample_scores))),
        "accuracy": float(metrics.get("direct_accuracy", 0.0)),
        "hFPR": float(metrics.get("direct_hallucination_fpr", 0.0)),
        "keep_ratio": mean(row["keep_ratio"] for row in true_rows),
        "true_ECR": mean(row["true_ecr"] for row in true_rows),
        "true_positive_ECR": mean(row["true_ecr"] for row in true_rows if row["binary_polarity"] == "positive"),
        "true_negative_source_coverage": mean(
            row["true_ecr"] for row in true_rows if row["binary_polarity"] == "negative"
        ),
        "true_CenterR": mean(row["true_center"] for row in true_rows),
        "true_PatchR": mean(row["true_patch"] for row in true_rows),
        "selector_box_ECR": mean(row["reported_ecr"] for row in true_rows),
        "mean_prune_overhead_ms": mean(row["overhead_ms"] for row in true_rows),
    }


def load_oracle(path: str | Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        sample_id = str(row.get("sample_id", row.get("id", "")))
        if not sample_id:
            continue
        out[sample_id] = {
            "regions": evidence_regions_from_sample(row),
            "binary_polarity": row.get("binary_polarity"),
        }
    return out


def token_boxes_from_trace(trace: dict[str, Any]) -> list[tuple[float, float, float, float]]:
    patch_h = int(trace.get("patch_grid_h") or 0)
    patch_w = int(trace.get("patch_grid_w") or 0)
    num_patches = int(trace.get("num_image_patches") or 0)
    tile_rows = int(trace.get("tile_rows") or 0)
    tile_cols = int(trace.get("tile_cols") or 0)
    has_thumbnail = bool(trace.get("has_thumbnail_patch"))
    if patch_h > 0 and patch_w > 0 and num_patches > 1 and tile_rows > 0 and tile_cols > 0:
        local_boxes = make_token_grid(patch_h, patch_w)
        tile_count = tile_rows * tile_cols
        if has_thumbnail and num_patches == tile_count + 1:
            image_tile_count = tile_count
        elif num_patches == tile_count:
            image_tile_count = tile_count
        else:
            image_tile_count = num_patches
            tile_rows, tile_cols = 1, num_patches
            has_thumbnail = False

        boxes: list[tuple[float, float, float, float]] = []
        for patch_idx in range(image_tile_count):
            row = patch_idx // tile_cols
            col = patch_idx % tile_cols
            outer = (col / tile_cols, row / tile_rows, (col + 1) / tile_cols, (row + 1) / tile_rows)
            boxes.extend(map_local_boxes(local_boxes, outer))
        if has_thumbnail:
            boxes.extend(local_boxes)
        return boxes

    grid_h = int(trace.get("grid_h") or 24)
    grid_w = int(trace.get("grid_w") or grid_h)
    return make_token_grid(grid_h, grid_w)


def map_local_boxes(
    local_boxes: list[tuple[float, float, float, float]],
    outer: tuple[float, float, float, float],
) -> list[tuple[float, float, float, float]]:
    ox1, oy1, ox2, oy2 = outer
    ow = ox2 - ox1
    oh = oy2 - oy1
    return [
        (ox1 + x1 * ow, oy1 + y1 * oh, ox1 + x2 * ow, oy1 + y2 * oh)
        for x1, y1, x2, y2 in local_boxes
    ]


def parse_run_spec(spec: str) -> tuple[str, str, str, str, str]:
    parts = [part.strip() for part in spec.split(",", 4)]
    if len(parts) == 4 and all(parts):
        return parts[0], parts[1], parts[2], parts[3], parts[3]
    if len(parts) == 5 and all(parts):
        return parts[0], parts[1], parts[2], parts[3], parts[4]
    raise ValueError("--run must be model,variant,selector,score_dir[,trace_dir]")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: str | Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Box-Source Robustness",
        "",
        "| Model | Box source | Selector | Acc. | hFPR | Keep | positive ECR | negative source coverage | aggregate RCR | true CenterR | selector-box RCR |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['variant']} | {row['selector']} | "
            f"{row['accuracy']:.3f} | {row['hFPR']:.3f} | {row['keep_ratio']:.3f} | "
            f"{row['true_positive_ECR']:.3f} | {row['true_negative_source_coverage']:.3f} | "
            f"{row['true_ECR']:.3f} | {row['true_CenterR']:.3f} | {row['selector_box_ECR']:.3f} |"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def mean(values: Any) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    main()
