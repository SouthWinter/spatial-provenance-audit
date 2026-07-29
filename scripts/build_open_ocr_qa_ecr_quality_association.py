#!/usr/bin/env python3
"""Associate open-QA bbox evidence coverage with native generation quality."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "problem_optimization_audit" / "open_ocr_qa_ecr_quality_association"
TASKS = {
    "TextVQA-lite": {
        "ecr": ROOT / "runs/problem_optimization_audit/open_ocr_qa_textvqa_gt_bbox_ecr/bbox_ecr_rows.csv",
        "generation": {
            0.30: ROOT / "runs/open_ocr_qa/qwen3_8b_textvqa_lite_target_grid0p30_full500/open_ocr_qa_generation.jsonl",
            0.50: ROOT / "runs/open_ocr_qa/qwen3_8b_textvqa_lite_target_grid0p50_full500/open_ocr_qa_generation.jsonl",
            0.70: ROOT / "runs/open_ocr_qa/qwen3_8b_textvqa_lite_target_grid0p70_full500/open_ocr_qa_generation.jsonl",
        },
    },
    "DocVQA-lite": {
        "ecr": ROOT / "runs/problem_optimization_audit/open_ocr_qa_docvqa_hxlinh_bbox_expanded_ecr/bbox_ecr_rows.csv",
        "generation": {
            0.30: ROOT / "runs/open_ocr_qa/qwen3_8b_docvqa_lite_target_grid0p30_full500/open_ocr_qa_generation.jsonl",
            0.50: ROOT / "runs/open_ocr_qa/qwen3_8b_docvqa_lite_target_grid0p50_full500/open_ocr_qa_generation.jsonl",
            0.70: ROOT / "runs/open_ocr_qa/qwen3_8b_docvqa_lite_target_grid0p70_full500/open_ocr_qa_generation.jsonl",
        },
    },
}


def main() -> None:
    rows = build_rows()
    bucket_rows = build_bucket_summary(rows)
    corr_rows = build_correlation_summary(rows)
    example_rows = build_examples(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "ecr_quality_rows.csv", rows)
    write_csv(OUT_DIR / "ecr_quality_bucket_summary.csv", bucket_rows)
    write_csv(OUT_DIR / "ecr_quality_correlation_summary.csv", corr_rows)
    write_csv(OUT_DIR / "ecr_quality_examples.csv", example_rows)
    (OUT_DIR / "ecr_quality_association_report.md").write_text(
        build_report(bucket_rows, corr_rows, example_rows),
        encoding="utf-8",
    )
    print(f"Wrote open-QA ECR/quality association audit to {OUT_DIR}")


def build_rows() -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    for task, spec in TASKS.items():
        generation = load_generation(spec["generation"])
        for ecr in read_csv(spec["ecr"]):
            if ecr.get("metric_status") != "scored":
                continue
            budget = round(float(ecr["budget_keep_ratio"]), 2)
            gen = generation.get((ecr["sample_id"], budget))
            if not gen:
                continue
            full_score = float(gen["full_score"])
            pruned_score = float(gen["pruned_score"])
            delta = pruned_score - full_score
            all_rows.append(
                {
                    "task": task,
                    "sample_id": ecr["sample_id"],
                    "question_id": ecr["question_id"],
                    "budget_keep_ratio": f"{budget:.2f}",
                    "box_count": ecr["box_count"],
                    "ECR": ecr["ECR"],
                    "CenterR": ecr["CenterR"],
                    "PatchR": ecr["PatchR"],
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
    return all_rows


def load_generation(paths: dict[float, Path]) -> dict[tuple[str, float], dict[str, Any]]:
    out = {}
    for budget, path in paths.items():
        for row in read_jsonl(path):
            out[(row["sample_id"], round(float(budget), 2))] = row
    return out


def build_bucket_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    scopes: list[tuple[str, list[dict[str, Any]]]] = [("all", rows)]
    for task in sorted({row["task"] for row in rows}):
        task_rows = [row for row in rows if row["task"] == task]
        scopes.append((task, task_rows))
        for budget in sorted({row["budget_keep_ratio"] for row in task_rows}):
            scopes.append((f"{task}@{budget}", [row for row in task_rows if row["budget_keep_ratio"] == budget]))
    for scope, subset in scopes:
        for bucket in ["[0,0.25)", "[0.25,0.50)", "[0.50,0.75)", "[0.75,1.00]"]:
            bucket_rows = [row for row in subset if row["ecr_bucket"] == bucket]
            if not bucket_rows:
                continue
            out.append(summarize_subset(scope, f"ECR {bucket}", bucket_rows))
        pass_rows = [row for row in subset if safe_float(row["all_regions_ECR_ge_0p50"]) >= 0.5]
        fail_rows = [row for row in subset if safe_float(row["all_regions_ECR_ge_0p50"]) < 0.5]
        if pass_rows:
            out.append(summarize_subset(scope, "all_regions_ECR_ge_0p50=1", pass_rows))
        if fail_rows:
            out.append(summarize_subset(scope, "all_regions_ECR_ge_0p50=0", fail_rows))
    return out


def summarize_subset(scope: str, group: str, subset: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "scope": scope,
        "group": group,
        "n": len(subset),
        "mean_ECR": fmt(mean(safe_float(row["ECR"]) for row in subset)),
        "mean_worst_region_ECR": fmt(mean(safe_float(row["worst_region_ECR"]) for row in subset)),
        "mean_full_score": fmt(mean(safe_float(row["full_score"]) for row in subset)),
        "mean_pruned_score": fmt(mean(safe_float(row["pruned_score"]) for row in subset)),
        "mean_score_delta": fmt(mean(safe_float(row["score_delta"]) for row in subset)),
        "mean_score_drop": fmt(mean(safe_float(row["score_drop"]) for row in subset)),
        "pruned_good_rate": fmt(mean(safe_float(row["pruned_good"]) for row in subset)),
        "full_good_rate": fmt(mean(safe_float(row["full_good"]) for row in subset)),
    }


def build_correlation_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    scopes: list[tuple[str, list[dict[str, Any]]]] = [("all", rows)]
    for task in sorted({row["task"] for row in rows}):
        task_rows = [row for row in rows if row["task"] == task]
        scopes.append((task, task_rows))
        for budget in sorted({row["budget_keep_ratio"] for row in task_rows}):
            scopes.append((f"{task}@{budget}", [row for row in task_rows if row["budget_keep_ratio"] == budget]))
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
    examples = []
    patterns = [
        ("high_ecr_large_drop", lambda r: safe_float(r["ECR"]) >= 0.70 and safe_float(r["score_drop"]) >= 0.50),
        ("low_ecr_good_answer", lambda r: safe_float(r["ECR"]) < 0.30 and safe_float(r["pruned_score"]) >= 0.80),
    ]
    for label, predicate in patterns:
        candidates = [row for row in rows if predicate(row)]
        candidates.sort(key=lambda row: (-safe_float(row["score_drop"]), row["task"], row["sample_id"]))
        for row in candidates[:12]:
            examples.append(
                {
                    "case_type": label,
                    "task": row["task"],
                    "sample_id": row["sample_id"],
                    "budget_keep_ratio": row["budget_keep_ratio"],
                    "ECR": row["ECR"],
                    "worst_region_ECR": row["worst_region_ECR"],
                    "full_score": row["full_score"],
                    "pruned_score": row["pruned_score"],
                    "score_delta": row["score_delta"],
                    "question": row["raw_question"],
                    "gold_answers": row["gold_answers"],
                    "full_answer": row["full_answer"],
                    "pruned_answer": row["pruned_answer"],
                }
            )
    return examples


def build_report(
    bucket_rows: list[dict[str, Any]],
    corr_rows: list[dict[str, Any]],
    examples: list[dict[str, Any]],
) -> str:
    lines = [
        "# Open QA ECR/Quality Association Audit",
        "",
        "This audit joins annotated open-QA bbox ECR rows with cached native generation outputs. It tests whether higher evidence availability is associated with smaller answer-quality drops. It is not a causal intervention.",
        "",
        "## Bucket Summary",
        "",
        "| Scope | Group | n | mean ECR | mean pruned score | mean delta | pruned-good rate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in bucket_rows:
        lines.append(
            f"| {row['scope']} | {row['group']} | {row['n']} | {row['mean_ECR']} | "
            f"{row['mean_pruned_score']} | {row['mean_score_delta']} | {row['pruned_good_rate']} |"
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
            "| Case | Task | Sample | Budget | ECR | Delta | Gold | Full | Pruned |",
            "| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |",
        ]
    )
    for row in examples:
        lines.append(
            f"| {row['case_type']} | {row['task']} | {row['sample_id']} | {row['budget_keep_ratio']} | "
            f"{row['ECR']} | {row['score_delta']} | {escape(row['gold_answers'])} | "
            f"{escape(row['full_answer'])} | {escape(row['pruned_answer'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A positive association would support evidence availability as a useful risk signal.",
            "- Counterexamples are expected: ECR measures whether annotated evidence remains visible, not whether the model uses it or whether all non-answer context is preserved.",
        ]
    )
    return "\n".join(lines) + "\n"


def ecr_bucket(value: float) -> str:
    if value < 0.25:
        return "[0,0.25)"
    if value < 0.50:
        return "[0.25,0.50)"
    if value < 0.75:
        return "[0.50,0.75)"
    return "[0.75,1.00]"


def pearson(x: list[float], y: list[float]) -> float:
    pairs = [(a, b) for a, b in zip(x, y) if not math.isnan(a) and not math.isnan(b)]
    if len(pairs) < 2:
        return math.nan
    xs, ys = zip(*pairs)
    mx, my = mean(xs), mean(ys)
    num = sum((a - mx) * (b - my) for a, b in pairs)
    den_x = math.sqrt(sum((a - mx) ** 2 for a in xs))
    den_y = math.sqrt(sum((b - my) ** 2 for b in ys))
    if den_x == 0 or den_y == 0:
        return math.nan
    return num / (den_x * den_y)


def spearman(x: list[float], y: list[float]) -> float:
    pairs = [(a, b) for a, b in zip(x, y) if not math.isnan(a) and not math.isnan(b)]
    if len(pairs) < 2:
        return math.nan
    xs, ys = zip(*pairs)
    return pearson(rank(list(xs)), rank(list(ys)))


def rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


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
