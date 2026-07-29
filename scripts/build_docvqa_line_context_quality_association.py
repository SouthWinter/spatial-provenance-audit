#!/usr/bin/env python3
"""Associate DocVQA line-context ECR with native generation quality."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from build_open_ocr_qa_ecr_quality_association import ecr_bucket, pearson, spearman


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "problem_optimization_audit" / "docvqa_line_context_quality_association"
ECR_ROWS = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_docvqa_hxlinh_line_context_bbox_ecr"
    / "bbox_ecr_rows.csv"
)
GENERATION = {
    0.30: ROOT / "runs/open_ocr_qa/qwen3_8b_docvqa_lite_target_grid0p30_full500/open_ocr_qa_generation.jsonl",
    0.50: ROOT / "runs/open_ocr_qa/qwen3_8b_docvqa_lite_target_grid0p50_full500/open_ocr_qa_generation.jsonl",
    0.70: ROOT / "runs/open_ocr_qa/qwen3_8b_docvqa_lite_target_grid0p70_full500/open_ocr_qa_generation.jsonl",
}


def main() -> None:
    rows = build_rows()
    bucket_rows = build_bucket_summary(rows)
    corr_rows = build_correlation_summary(rows)
    examples = build_examples(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "line_context_quality_rows.csv", rows)
    write_csv(OUT_DIR / "line_context_quality_bucket_summary.csv", bucket_rows)
    write_csv(OUT_DIR / "line_context_quality_correlation_summary.csv", corr_rows)
    write_csv(OUT_DIR / "line_context_quality_examples.csv", examples)
    (OUT_DIR / "line_context_quality_report.md").write_text(
        build_report(bucket_rows, corr_rows, examples),
        encoding="utf-8",
    )
    print(f"Wrote DocVQA line-context quality association to {OUT_DIR}")


def build_rows() -> list[dict[str, Any]]:
    generation = load_generation()
    out = []
    for ecr in read_csv(ECR_ROWS):
        if ecr.get("metric_status") != "scored":
            continue
        budget = round(float(ecr["budget_keep_ratio"]), 2)
        gen = generation.get((ecr["sample_id"], budget))
        if not gen:
            continue
        full_score = float(gen["full_score"])
        pruned_score = float(gen["pruned_score"])
        delta = pruned_score - full_score
        out.append(
            {
                "sample_id": ecr["sample_id"],
                "budget_keep_ratio": f"{budget:.2f}",
                "box_count": ecr["box_count"],
                "ECR": ecr["ECR"],
                "worst_region_ECR": ecr["worst_region_ECR"],
                "all_regions_ECR_ge_0p50": ecr["all_regions_ECR_ge_0p50"],
                "full_score": f"{full_score:.3f}",
                "pruned_score": f"{pruned_score:.3f}",
                "score_delta": f"{delta:.3f}",
                "score_drop": f"{max(0.0, -delta):.3f}",
                "pruned_good": int(pruned_score >= 0.8),
                "ecr_bucket": ecr_bucket(float(ecr["ECR"])),
                "raw_question": gen.get("raw_question", ""),
                "gold_answers": " | ".join(map(str, gen.get("gold_answers", []))),
                "full_answer": gen.get("full_answer", ""),
                "pruned_answer": gen.get("pruned_answer", ""),
            }
        )
    return out


def load_generation() -> dict[tuple[str, float], dict[str, Any]]:
    out = {}
    for budget, path in GENERATION.items():
        for row in read_jsonl(path):
            out[(row["sample_id"], round(float(budget), 2))] = row
    return out


def build_bucket_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    scopes = [("DocVQA-line-context", rows)]
    for budget in sorted({row["budget_keep_ratio"] for row in rows}):
        scopes.append((f"DocVQA-line-context@{budget}", [row for row in rows if row["budget_keep_ratio"] == budget]))
    for scope, subset in scopes:
        for bucket in ["[0,0.25)", "[0.25,0.50)", "[0.50,0.75)", "[0.75,1.00]"]:
            part = [row for row in subset if row["ecr_bucket"] == bucket]
            if part:
                out.append(summarize(scope, f"ECR {bucket}", part))
        pass_rows = [row for row in subset if safe_float(row["all_regions_ECR_ge_0p50"]) >= 0.5]
        fail_rows = [row for row in subset if safe_float(row["all_regions_ECR_ge_0p50"]) < 0.5]
        if pass_rows:
            out.append(summarize(scope, "all_regions_ECR_ge_0p50=1", pass_rows))
        if fail_rows:
            out.append(summarize(scope, "all_regions_ECR_ge_0p50=0", fail_rows))
    return out


def summarize(scope: str, group: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "scope": scope,
        "group": group,
        "n": len(rows),
        "mean_ECR": fmt(mean(safe_float(row["ECR"]) for row in rows)),
        "mean_worst_region_ECR": fmt(mean(safe_float(row["worst_region_ECR"]) for row in rows)),
        "mean_pruned_score": fmt(mean(safe_float(row["pruned_score"]) for row in rows)),
        "mean_score_delta": fmt(mean(safe_float(row["score_delta"]) for row in rows)),
        "mean_score_drop": fmt(mean(safe_float(row["score_drop"]) for row in rows)),
        "pruned_good_rate": fmt(mean(safe_float(row["pruned_good"]) for row in rows)),
    }


def build_correlation_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    scopes = [("DocVQA-line-context", rows)]
    for budget in sorted({row["budget_keep_ratio"] for row in rows}):
        scopes.append((f"DocVQA-line-context@{budget}", [row for row in rows if row["budget_keep_ratio"] == budget]))
    for scope, subset in scopes:
        for x_name in ("ECR", "worst_region_ECR", "all_regions_ECR_ge_0p50"):
            for y_name in ("pruned_score", "score_delta", "score_drop"):
                x = [safe_float(row[x_name]) for row in subset]
                y = [safe_float(row[y_name]) for row in subset]
                out.append(
                    {
                        "scope": scope,
                        "x": x_name,
                        "y": y_name,
                        "n": len(subset),
                        "pearson": fmt(pearson(x, y)),
                        "spearman": fmt(spearman(x, y)),
                    }
                )
    return out


def build_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if safe_float(row["all_regions_ECR_ge_0p50"]) >= 0.5 and safe_float(row["score_drop"]) >= 0.5
    ]
    candidates.sort(key=lambda row: (-safe_float(row["score_drop"]), row["sample_id"], row["budget_keep_ratio"]))
    out = []
    for row in candidates[:16]:
        out.append(
            {
                "case_type": "all_regions_pass_large_drop",
                "sample_id": row["sample_id"],
                "budget_keep_ratio": row["budget_keep_ratio"],
                "ECR": row["ECR"],
                "worst_region_ECR": row["worst_region_ECR"],
                "score_delta": row["score_delta"],
                "question": row["raw_question"],
                "gold_answers": row["gold_answers"],
                "full_answer": row["full_answer"],
                "pruned_answer": row["pruned_answer"],
            }
        )
    return out


def build_report(
    bucket_rows: list[dict[str, Any]],
    corr_rows: list[dict[str, Any]],
    examples: list[dict[str, Any]],
) -> str:
    lines = [
        "# DocVQA Line-Context ECR/Quality Association",
        "",
        "This audit joins deterministic OCR line-context ECR with cached DocVQA native generation scores. It is an association analysis, not a causal intervention.",
        "",
        "## Bucket Summary",
        "",
        "| Scope | Group | n | mean ECR | mean worst ECR | pruned score | delta | good rate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in bucket_rows:
        lines.append(
            f"| {row['scope']} | {row['group']} | {row['n']} | {row['mean_ECR']} | "
            f"{row['mean_worst_region_ECR']} | {row['mean_pruned_score']} | "
            f"{row['mean_score_delta']} | {row['pruned_good_rate']} |"
        )
    lines.extend(
        [
            "",
            "## Correlation Summary",
            "",
            "| Scope | X | Y | n | Pearson | Spearman |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in corr_rows:
        if row["x"] in {"ECR", "worst_region_ECR"} and row["y"] in {"score_delta", "score_drop"}:
            lines.append(
                f"| {row['scope']} | {row['x']} | {row['y']} | {row['n']} | {row['pearson']} | {row['spearman']} |"
            )
    lines.extend(
        [
            "",
            "## Counterexamples",
            "",
            "| Case | Sample | Budget | ECR | Worst ECR | Delta | Gold | Full | Pruned |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
        ]
    )
    for row in examples:
        lines.append(
            f"| {row['case_type']} | {row['sample_id']} | {row['budget_keep_ratio']} | "
            f"{row['ECR']} | {row['worst_region_ECR']} | {row['score_delta']} | "
            f"{escape(row['gold_answers'])} | {escape(row['full_answer'])} | {escape(row['pruned_answer'])} |"
        )
    return "\n".join(lines) + "\n"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def fmt(value: float) -> str:
    if math.isnan(value):
        return ""
    return f"{value:.3f}"


def escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
