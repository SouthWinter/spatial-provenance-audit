#!/usr/bin/env python
"""Backfill strict evidence-retention metrics for completed pruning runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recap.prune.metrics import (
    evidence_center_recall,
    evidence_coverage,
    evidence_patch_recall,
    evidence_regions_from_sample,
    make_token_grid,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", required=True, help="Directory containing per-run subdirectories.")
    parser.add_argument("--run-glob", default="*", help="Glob for run directories under --runs-dir.")
    parser.add_argument("--in-place", action="store_true", help="Rewrite prune_traces.jsonl with enriched fields.")
    parser.add_argument("--summary-csv", default="", help="Optional CSV summary output.")
    parser.add_argument("--summary-md", default="", help="Optional Markdown summary output.")
    parser.add_argument("--no-update-metrics", action="store_true", help="Do not patch metrics.json pruning summaries.")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    rows = []
    for run_dir in sorted(path for path in runs_dir.glob(args.run_glob) if path.is_dir()):
        trace_path = run_dir / "prune_traces.jsonl"
        probe_path = run_dir / "probes.jsonl"
        if not trace_path.exists() or not probe_path.exists():
            continue
        probes = load_probes(probe_path)
        traces = read_jsonl(trace_path)
        enriched, changed = enrich_traces(traces, probes)
        row = summarize_run(run_dir.name, enriched)
        rows.append(row)
        if args.in_place and changed:
            write_jsonl_atomic(trace_path, enriched)
        if not args.no_update_metrics:
            update_metrics(run_dir / "metrics.json", row)

    if args.summary_csv:
        write_csv(Path(args.summary_csv), rows)
    if args.summary_md:
        write_markdown(Path(args.summary_md), rows)
    print(f"Processed {len(rows)} runs.")


def load_probes(path: Path) -> dict[str, dict[str, Any]]:
    probes: dict[str, dict[str, Any]] = {}
    for probe in read_jsonl(path):
        for key in ("sample_id", "id"):
            value = probe.get(key)
            if value is not None:
                probes[str(value)] = probe
    return probes


def enrich_traces(
    traces: list[dict[str, Any]],
    probes: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    changed = False
    enriched = []
    for trace in traces:
        out = dict(trace)
        sample_id = str(out.get("sample_id", ""))
        probe = probes.get(sample_id)
        token_boxes = token_boxes_from_trace(out)
        evidence_regions = evidence_regions_from_sample(probe or {})
        if token_boxes and evidence_regions:
            kept = out.get("kept_indices", [])
            ecr = evidence_coverage(kept, token_boxes, evidence_regions)
            center = evidence_center_recall(kept, token_boxes, evidence_regions)
            patch = evidence_patch_recall(kept, token_boxes, evidence_regions)
            updates = {
                "ecr": ecr,
                "ecr_0_5": 1.0 if ecr >= 0.5 else 0.0,
                "evidence_center_recall": center,
                "evidence_patch_recall": patch,
                "has_evidence": True,
                "evidence_region_count": len(evidence_regions),
            }
            for key, value in updates.items():
                if out.get(key) != value:
                    out[key] = value
                    changed = True
        enriched.append(out)
    return enriched, changed


def token_boxes_from_trace(trace: dict[str, Any]):
    full = int(trace.get("full_visual_tokens") or 0)
    grid_h = int(trace.get("grid_h") or 0)
    grid_w = int(trace.get("grid_w") or 0)
    if full <= 0:
        return []
    if grid_h > 0 and grid_w > 0:
        if grid_h * grid_w == full:
            return make_token_grid(grid_h, grid_w)
        for merge in range(1, 9):
            if grid_h % merge == 0 and grid_w % merge == 0:
                rows = grid_h // merge
                cols = grid_w // merge
                if rows * cols == full:
                    return make_token_grid(rows, cols)
        aspect = grid_h / max(1, grid_w)
        rows, cols = closest_factor_grid(full, aspect)
        return make_token_grid(rows, cols)
    rows = int(round(math.sqrt(full)))
    while rows > 1 and full % rows != 0:
        rows -= 1
    return make_token_grid(rows, full // rows)


def closest_factor_grid(num_tokens: int, target_aspect: float) -> tuple[int, int]:
    best = (1, num_tokens)
    best_error = float("inf")
    for rows in range(1, int(math.sqrt(num_tokens)) + 1):
        if num_tokens % rows:
            continue
        for candidate in ((rows, num_tokens // rows), (num_tokens // rows, rows)):
            aspect = candidate[0] / max(1, candidate[1])
            error = abs(aspect - target_aspect)
            if error < best_error:
                best = candidate
                best_error = error
    return best


def summarize_run(run_name: str, traces: list[dict[str, Any]]) -> dict[str, Any]:
    evidence_traces = [trace for trace in traces if trace.get("has_evidence")]
    return {
        "run": run_name,
        "selector": str(traces[0].get("selector", "")) if traces else "",
        "num_traces": len(traces),
        "keep": mean_keep_ratio(traces),
        "ECR": mean_value(evidence_traces, "ecr"),
        "CenterR": mean_value(evidence_traces, "evidence_center_recall"),
        "PatchR": mean_value(evidence_traces, "evidence_patch_recall"),
    }


def update_metrics(path: Path, row: dict[str, Any]) -> None:
    if not path.exists():
        return
    metrics = read_json(path)
    pruning = dict(metrics.get("pruning") or {})
    pruning["mean_evidence_center_recall"] = row["CenterR"]
    pruning["mean_evidence_patch_recall"] = row["PatchR"]
    pruning["mean_ecr"] = row["ECR"]
    metrics["pruning"] = pruning
    write_json_atomic(path, metrics)


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run", "selector", "num_traces", "keep", "ECR", "CenterR", "PatchR"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Strict Evidence Metrics",
        "",
        "| run | selector | n | keep | ECR | CenterR | PatchR |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['run']} | {row['selector']} | {row['num_traces']} | "
            f"{row['keep']:.4f} | {row['ECR']:.4f} | {row['CenterR']:.4f} | {row['PatchR']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def mean_keep_ratio(rows: list[dict[str, Any]]) -> float:
    values = []
    for row in rows:
        kept = row.get("kept_visual_tokens")
        total = row.get("full_visual_tokens")
        if isinstance(kept, (int, float)) and isinstance(total, (int, float)) and total:
            values.append(float(kept) / float(total))
    return sum(values) / len(values) if values else 0.0


def mean_value(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    main()
