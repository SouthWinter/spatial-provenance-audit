#!/usr/bin/env python
"""Summarize spatial-aware GSR pruning probes."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "spatial_aware_gsr"


@dataclass(frozen=True)
class RunSpec:
    model: str
    label: str
    path: Path
    split: str
    role: str


RUNS = [
    RunSpec("LLaVA-1.5-7B", "direct_full", Path("runs/rice_v5/llava15_7b_profile_fast_gsrbench_coco_spatial_two"), "full880", "full visual"),
    RunSpec("LLaVA-1.5-7B", "embed_topk0p40_full", Path("runs/llava_prune/llava15_7b_gsr_coco_spatial_profile_fast_embed_topk_0p40"), "full880", "embedding top-k"),
    RunSpec("LLaVA-1.5-7B", "grid0p40_full", Path("runs/llava_prune/llava15_7b_gsr_coco_spatial_profile_fast_grid_0p40"), "full880", "coverage baseline"),
    RunSpec("LLaVA-1.5-7B", "spatial_c35_ctx25_0p40", Path("runs/llava_prune/llava15_7b_gsr_coco_spatial_profile_fast_spatial_aware_0p40_limit100"), "limit100", "spatial-aware"),
    RunSpec("LLaVA-1.5-7B", "spatial_c10_ctx10_0p40", Path("runs/llava_prune/llava15_7b_gsr_coco_spatial_profile_fast_spatial_aware_c10_ctx10_0p40_limit100"), "limit100", "spatial-aware conservative"),
    RunSpec("InternVL3.5-8B", "direct_full", Path("runs/rice_v5/internvl3_5_8b_profile_fast_gsrbench_coco_spatial_two"), "full880", "full visual"),
    RunSpec("InternVL3.5-8B", "target_embed_topk0p50_full", Path("runs/internvl_prune/internvl35_8b_gsr_coco_spatial_profile_fast_target_embed_topk_0p50"), "full880", "target embedding top-k"),
    RunSpec("InternVL3.5-8B", "grid0p50_full", Path("runs/internvl_prune/internvl35_8b_gsr_coco_spatial_profile_fast_grid_0p50"), "full880", "coverage baseline"),
    RunSpec("InternVL3.5-8B", "spatial_c35_ctx25_0p50", Path("runs/internvl_prune/internvl35_8b_gsr_coco_spatial_profile_fast_spatial_aware_0p50_limit100"), "limit100", "spatial-aware"),
    RunSpec("InternVL3.5-8B", "spatial_c10_ctx10_0p50", Path("runs/internvl_prune/internvl35_8b_gsr_coco_spatial_profile_fast_spatial_aware_c10_ctx10_0p50_limit100"), "limit100", "spatial-aware conservative"),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [row_for_spec(spec) for spec in RUNS]
    add_subset_rows(rows)
    write_csv(OUT_DIR / "spatial_aware_gsr_summary.csv", rows)
    (OUT_DIR / "spatial_aware_gsr_report.md").write_text(build_markdown(rows), encoding="utf-8")
    print(f"Wrote {OUT_DIR / 'spatial_aware_gsr_report.md'}")
    print(f"Wrote {OUT_DIR / 'spatial_aware_gsr_summary.csv'}")


def row_for_spec(spec: RunSpec) -> dict[str, Any]:
    run_dir = ROOT / spec.path
    row = {
        "model": spec.model,
        "label": spec.label,
        "split": spec.split,
        "role": spec.role,
        "path": str(spec.path),
        "status": "done" if (run_dir / "sample_scores.jsonl").exists() and (run_dir / "metrics.json").exists() else "missing",
        "n": "",
        "acc": "",
        "hFPR": "",
        "yes_rate": "",
        "pos_acc": "",
        "neg_acc": "",
        "keep": "",
        "kept_visual": "",
        "ECR": "",
        "CenterR": "",
        "PatchR": "",
        "forward_ms": "",
        "language_ms": "",
        "overhead_ms": "",
    }
    if row["status"] != "done":
        return row
    metrics = read_json(run_dir / "metrics.json")
    scores = read_jsonl(run_dir / "sample_scores.jsonl")
    traces = read_jsonl(run_dir / "prune_traces.jsonl") if (run_dir / "prune_traces.jsonl").exists() else []
    row.update(summarize_scores(scores, metrics))
    row.update(summarize_traces(traces))
    return row


def add_subset_rows(rows: list[dict[str, Any]]) -> None:
    subset_sources = {
        "LLaVA-1.5-7B": ROOT / "runs/llava_prune/llava15_7b_gsr_coco_spatial_profile_fast_spatial_aware_0p40_limit100/sample_scores.jsonl",
        "InternVL3.5-8B": ROOT / "runs/internvl_prune/internvl35_8b_gsr_coco_spatial_profile_fast_spatial_aware_0p50_limit100/sample_scores.jsonl",
    }
    for model, subset_path in subset_sources.items():
        if not subset_path.exists():
            continue
        subset_ids = {str(row["sample_id"]) for row in read_jsonl(subset_path)}
        for spec in RUNS:
            if spec.model != model:
                continue
            score_path = ROOT / spec.path / "sample_scores.jsonl"
            metrics_path = ROOT / spec.path / "metrics.json"
            trace_path = ROOT / spec.path / "prune_traces.jsonl"
            if not score_path.exists() or not metrics_path.exists():
                continue
            scores = [row for row in read_jsonl(score_path) if str(row.get("sample_id")) in subset_ids]
            if not scores:
                continue
            traces = read_jsonl(trace_path) if trace_path.exists() else []
            subset_row = {
                "model": model,
                "label": f"{spec.label}_same100",
                "split": "same100",
                "role": spec.role,
                "path": str(spec.path),
                "status": "done",
            }
            subset_row.update(summarize_scores(scores, {}))
            subset_row.update(summarize_traces(filter_traces(traces, scores)))
            rows.append(subset_row)


def summarize_scores(rows: list[dict[str, Any]], metrics: dict[str, Any]) -> dict[str, Any]:
    neg = [row for row in rows if row.get("target_is_negative")]
    pos = [row for row in rows if not row.get("target_is_negative")]
    return {
        "n": len(rows),
        "acc": float(metrics.get("direct_accuracy", mean(row.get("direct_correct") for row in rows))),
        "hFPR": float(metrics.get("direct_hallucination_fpr", mean(row.get("hallucination") for row in neg))),
        "yes_rate": mean(row.get("direct_pred") == "yes" for row in rows),
        "pos_acc": mean(row.get("direct_correct") for row in pos),
        "neg_acc": mean(row.get("direct_correct") for row in neg),
    }


def summarize_traces(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"keep": "", "kept_visual": "", "ECR": "", "CenterR": "", "PatchR": "", "forward_ms": "", "language_ms": "", "overhead_ms": ""}
    full = mean(row.get("full_visual_tokens") for row in rows)
    kept = mean(row.get("kept_visual_tokens") for row in rows)
    evidence_rows = [row for row in rows if row.get("has_evidence")]
    return {
        "keep": kept / full if is_number(kept) and is_number(full) and full > 0 else "",
        "kept_visual": kept,
        "ECR": mean(row.get("ecr") for row in evidence_rows),
        "CenterR": mean(row.get("evidence_center_recall") for row in evidence_rows),
        "PatchR": mean(row.get("evidence_patch_recall") for row in evidence_rows),
        "forward_ms": mean(row.get("mean_forward_ms", row.get("forward_ms")) for row in rows),
        "language_ms": mean(row.get("mean_language_ms", row.get("language_ms")) for row in rows),
        "overhead_ms": mean(row.get("mean_prune_overhead_ms", row.get("prune_overhead_ms")) for row in rows),
    }


def filter_traces(traces: list[dict[str, Any]], scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = {(str(row.get("sample_id")), str(row.get("probe", "orig"))) for row in scores}
    return [row for row in traces if (str(row.get("sample_id")), str(row.get("probe", "orig"))) in wanted]


def build_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Spatial-Aware GSR Report",
        "",
        "Purpose: test whether bbox/relation-aware pruning fixes the spatial-transfer failure observed on GSR-Bench.",
        "",
        "Interpretation rule: `same100` rows compare all methods on the same first-100 sample subset used by the new spatial-aware sanity runs. Full rows are included only as broader context.",
        "",
    ]
    for model in sorted({str(row["model"]) for row in rows}):
        lines.extend([f"## {model}", ""])
        for split in ("same100", "limit100", "full880"):
            split_rows = [row for row in rows if row["model"] == model and row["split"] == split]
            if not split_rows:
                continue
            lines.extend([
                f"### {split}",
                "",
                "| label | role | status | n | acc | hFPR | keep | ECR | CenterR | PatchR | fwd ms | lang ms | ovh ms |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ])
            for row in split_rows:
                lines.append(
                    "| {label} | {role} | {status} | {n} | {acc} | {hfpr} | {keep} | {ecr} | {center} | {patch} | {fwd} | {lang} | {ovh} |".format(
                        label=row["label"],
                        role=row["role"],
                        status=row["status"],
                        n=fmt(row["n"], 0),
                        acc=fmt(row["acc"], 3),
                        hfpr=fmt(row["hFPR"], 3),
                        keep=fmt(row["keep"], 3),
                        ecr=fmt(row["ECR"], 3),
                        center=fmt(row["CenterR"], 3),
                        patch=fmt(row["PatchR"], 3),
                        fwd=fmt(row["forward_ms"], 1),
                        lang=fmt(row["language_ms"], 1),
                        ovh=fmt(row["overhead_ms"], 1),
                    )
                )
            lines.append("")
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mean(values: Any) -> float:
    nums = [float(value) for value in values if is_number(value)]
    if not nums:
        return float("nan")
    return sum(nums) / len(nums)


def is_number(value: Any) -> bool:
    return isinstance(value, int | float) and value == value


def fmt(value: Any, digits: int) -> str:
    if value == "" or value is None:
        return ""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if value != value:
        return ""
    return f"{value:.{digits}f}"


if __name__ == "__main__":
    main()
