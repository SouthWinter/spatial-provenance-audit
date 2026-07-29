#!/usr/bin/env python3
"""Build paper-facing evidence/quality tables from final manual annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from build_open_ocr_qa_ecr_quality_association import (
    build_bucket_summary,
    build_correlation_summary,
    build_examples,
    ecr_bucket,
    fmt,
    read_csv,
    read_jsonl,
    safe_float,
    write_csv,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ECR_ROWS = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_final_annotations_ecr"
    / "bbox_ecr_rows.csv"
)
DEFAULT_OUTPUT = ROOT / "runs" / "problem_optimization_audit" / "open_ocr_qa_manual_evidence_audit"
GENERATION = {
    "TextVQA-lite": {
        0.30: ROOT / "runs/open_ocr_qa/qwen3_8b_textvqa_lite_target_grid0p30_full500/open_ocr_qa_generation.jsonl",
        0.50: ROOT / "runs/open_ocr_qa/qwen3_8b_textvqa_lite_target_grid0p50_full500/open_ocr_qa_generation.jsonl",
        0.70: ROOT / "runs/open_ocr_qa/qwen3_8b_textvqa_lite_target_grid0p70_full500/open_ocr_qa_generation.jsonl",
    },
    "DocVQA-lite": {
        0.30: ROOT / "runs/open_ocr_qa/qwen3_8b_docvqa_lite_target_grid0p30_full500/open_ocr_qa_generation.jsonl",
        0.50: ROOT / "runs/open_ocr_qa/qwen3_8b_docvqa_lite_target_grid0p50_full500/open_ocr_qa_generation.jsonl",
        0.70: ROOT / "runs/open_ocr_qa/qwen3_8b_docvqa_lite_target_grid0p70_full500/open_ocr_qa_generation.jsonl",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ecr-rows", default=str(DEFAULT_ECR_ROWS), help="bbox_ecr_rows.csv from final manual annotations")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    ecr_path = Path(args.ecr_rows)
    rows = build_joined_rows(ecr_path)
    bucket_rows = build_bucket_summary(rows)
    corr_rows = build_correlation_summary(rows)
    example_rows = build_examples(rows)
    key_rows = build_key_summary(rows)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "manual_evidence_quality_rows.csv", rows)
    write_csv(out_dir / "manual_evidence_bucket_summary.csv", bucket_rows)
    write_csv(out_dir / "manual_evidence_correlation_summary.csv", corr_rows)
    write_csv(out_dir / "manual_evidence_key_summary.csv", key_rows)
    write_csv(out_dir / "manual_evidence_counterexamples.csv", example_rows)
    (out_dir / "manual_evidence_audit_report.md").write_text(
        build_manual_report(ecr_path, key_rows, bucket_rows, corr_rows, example_rows),
        encoding="utf-8",
    )
    print(f"Wrote manual evidence audit for {len(rows)} joined rows to {out_dir}")


def build_joined_rows(ecr_path: Path) -> list[dict[str, Any]]:
    generation = {
        task: load_generation(paths)
        for task, paths in GENERATION.items()
    }
    rows: list[dict[str, Any]] = []
    for ecr in read_csv(ecr_path):
        if ecr.get("metric_status") != "scored":
            continue
        task = ecr.get("task", "")
        budget = round(float(ecr["budget_keep_ratio"]), 2)
        gen = generation.get(task, {}).get((ecr["sample_id"], budget))
        if not gen:
            continue
        full_score = float(gen["full_score"])
        pruned_score = float(gen["pruned_score"])
        delta = pruned_score - full_score
        rows.append(
            {
                "task": task,
                "sample_id": ecr["sample_id"],
                "question_id": ecr["question_id"],
                "budget_keep_ratio": f"{budget:.2f}",
                "box_count": ecr["box_count"],
                "ECR": ecr["ECR"],
                "CenterR": ecr["CenterR"],
                "PatchR": ecr["PatchR"],
                "mean_region_ECR": ecr["mean_region_ECR"],
                "worst_region_ECR": ecr["worst_region_ECR"],
                "all_regions_ECR_ge_0p50": ecr["all_regions_ECR_ge_0p50"],
                "full_score": f"{full_score:.3f}",
                "pruned_score": f"{pruned_score:.3f}",
                "score_delta": f"{delta:.3f}",
                "score_drop": f"{max(0.0, -delta):.3f}",
                "full_exact": f"{float(gen['full_exact']):.3f}",
                "pruned_exact": f"{float(gen['pruned_exact']):.3f}",
                "full_good": int(full_score >= 0.8),
                "pruned_good": int(pruned_score >= 0.8),
                "ecr_bucket": ecr_bucket(float(ecr["ECR"])),
                "raw_question": gen.get("raw_question", ""),
                "gold_answers": " | ".join(map(str, gen.get("gold_answers", []))),
                "full_answer": gen.get("full_answer", ""),
                "pruned_answer": gen.get("pruned_answer", ""),
            }
        )
    return rows


def load_generation(paths: dict[float, Path]) -> dict[tuple[str, float], dict[str, Any]]:
    out = {}
    for budget, path in paths.items():
        if not path.exists():
            continue
        for row in read_jsonl(path):
            out[(row["sample_id"], round(float(budget), 2))] = row
    return out


def build_key_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    scopes: list[tuple[str, list[dict[str, Any]]]] = [("all", rows)]
    for task in sorted({row["task"] for row in rows}):
        task_rows = [row for row in rows if row["task"] == task]
        scopes.append((task, task_rows))
        for budget in sorted({row["budget_keep_ratio"] for row in task_rows}):
            scopes.append((f"{task}@{budget}", [row for row in task_rows if row["budget_keep_ratio"] == budget]))
    for scope, subset in scopes:
        if not subset:
            continue
        out.append(
            {
                "scope": scope,
                "n": len(subset),
                "unique_samples": len({row["sample_id"] for row in subset}),
                "mean_box_count": fmt(mean(safe_float(row["box_count"]) for row in subset)),
                "mean_ECR": fmt(mean(safe_float(row["ECR"]) for row in subset)),
                "mean_worst_region_ECR": fmt(mean(safe_float(row["worst_region_ECR"]) for row in subset)),
                "all_regions_ECR_ge_0p50_rate": fmt(mean(safe_float(row["all_regions_ECR_ge_0p50"]) for row in subset)),
                "mean_full_score": fmt(mean(safe_float(row["full_score"]) for row in subset)),
                "mean_pruned_score": fmt(mean(safe_float(row["pruned_score"]) for row in subset)),
                "mean_score_delta": fmt(mean(safe_float(row["score_delta"]) for row in subset)),
                "pruned_good_rate": fmt(mean(safe_float(row["pruned_good"]) for row in subset)),
            }
        )
    return out


def build_manual_report(
    ecr_path: Path,
    key_rows: list[dict[str, Any]],
    bucket_rows: list[dict[str, Any]],
    corr_rows: list[dict[str, Any]],
    examples: list[dict[str, Any]],
) -> str:
    lines = [
        "# Manual Multi-Region Evidence Audit",
        "",
        f"ECR rows: `{ecr_path}`",
        "",
        "This report joins final manual evidence annotations with cached Qwen open-answer generation outputs. It is ready for manuscript use only when the final annotation validator reports no unresolved, invalid, unlabeled, or empty annotation rows.",
        "",
        "## Key Summary",
        "",
        "| Scope | n | Samples | Box count | ECR | Worst ECR | All-region pass | Pruned score | Delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in key_rows:
        lines.append(
            f"| {row['scope']} | {row['n']} | {row['unique_samples']} | {row['mean_box_count']} | "
            f"{row['mean_ECR']} | {row['mean_worst_region_ECR']} | {row['all_regions_ECR_ge_0p50_rate']} | "
            f"{row['mean_pruned_score']} | {row['mean_score_delta']} |"
        )
    lines.extend(
        [
            "",
            "## Association Checks",
            "",
            "| Scope | X | Y | n | Pearson | Spearman |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in corr_rows:
        if row["x"] in {"ECR", "worst_region_ECR", "all_regions_ECR_ge_0p50"} and row["y"] in {"score_delta", "score_drop"}:
            lines.append(
                f"| {row['scope']} | {row['x']} | {row['y']} | {row['n']} | {row['pearson']} | {row['spearman']} |"
            )
    lines.extend(
        [
            "",
            "## Bucket Readout",
            "",
            "| Scope | Group | n | ECR | Pruned score | Delta | Good rate |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in bucket_rows:
        lines.append(
            f"| {row['scope']} | {row['group']} | {row['n']} | {row['mean_ECR']} | "
            f"{row['mean_pruned_score']} | {row['mean_score_delta']} | {row['pruned_good_rate']} |"
        )
    lines.extend(
        [
            "",
            "## Counterexamples",
            "",
            "| Case | Task | Sample | Budget | ECR | Delta | Full | Pruned |",
            "| --- | --- | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in examples:
        lines.append(
            f"| {row['case_type']} | {row['task']} | {row['sample_id']} | {row['budget_keep_ratio']} | "
            f"{row['ECR']} | {row['score_delta']} | {escape(row['full_answer'])} | {escape(row['pruned_answer'])} |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "Manual evidence ECR is an evidence-availability audit. It can support claims about whether annotated evidence remains in the pruned visual prefix and whether low evidence availability correlates with answer degradation. It should not be phrased as proof that the model causally used every annotated region.",
        ]
    )
    return "\n".join(lines) + "\n"


def escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
