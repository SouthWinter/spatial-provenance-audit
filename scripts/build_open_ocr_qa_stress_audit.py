#!/usr/bin/env python3
"""Build subgroup stress audits from cached open OCR/document QA runs.

This script does not run a model. It summarizes existing Qwen TextVQA-lite and
DocVQA-lite native generation outputs by interpretable stress tags:
multi-token answers, long questions, multi-constraint questions, numeric
answers, and layout/field-style questions. These tags are proxies for harder
open-answer/document cases; they should not be described as ground-truth
multi-evidence annotations.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

RUNS = (
    ("TextVQA-lite", "0.30", "runs/open_ocr_qa/qwen3_8b_textvqa_lite_target_grid0p30_full500/open_ocr_qa_generation.jsonl"),
    ("TextVQA-lite", "0.50", "runs/open_ocr_qa/qwen3_8b_textvqa_lite_target_grid0p50_full500/open_ocr_qa_generation.jsonl"),
    ("TextVQA-lite", "0.70", "runs/open_ocr_qa/qwen3_8b_textvqa_lite_target_grid0p70_full500/open_ocr_qa_generation.jsonl"),
    ("DocVQA-lite", "0.30", "runs/open_ocr_qa/qwen3_8b_docvqa_lite_target_grid0p30_full500/open_ocr_qa_generation.jsonl"),
    ("DocVQA-lite", "0.50", "runs/open_ocr_qa/qwen3_8b_docvqa_lite_target_grid0p50_full500/open_ocr_qa_generation.jsonl"),
    ("DocVQA-lite", "0.70", "runs/open_ocr_qa/qwen3_8b_docvqa_lite_target_grid0p70_full500/open_ocr_qa_generation.jsonl"),
)

TAG_ORDER = (
    "all",
    "multi_token_answer_ge3",
    "long_question_ge10",
    "multi_constraint_question",
    "numeric_answer",
    "layout_field_question",
    "full_good_pruned_drop_ge50",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="runs/problem_optimization_audit/open_ocr_qa_stress")
    parser.add_argument("--drop-threshold", type=float, default=0.50)
    parser.add_argument("--example-limit", type=int, default=12)
    args = parser.parse_args()

    rows = []
    examples = []
    for task, ratio, rel_path in RUNS:
        run_rows = read_jsonl(ROOT / rel_path)
        tagged_rows = []
        for row in run_rows:
            tags = assign_tags(row, drop_threshold=args.drop_threshold)
            tagged = {**row, "stress_tags": ";".join(tags)}
            tagged_rows.append(tagged)
            if "full_good_pruned_drop_ge50" in tags:
                examples.append(example_row(task, ratio, tagged))
        rows.extend(summarize_run(task, ratio, tagged_rows))

    examples = balanced_examples(examples, args.example_limit)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "open_ocr_qa_stress_summary.csv", rows)
    write_csv(out_dir / "open_ocr_qa_stress_failure_examples.csv", examples)
    (out_dir / "open_ocr_qa_stress_report.md").write_text(
        build_report(rows, examples),
        encoding="utf-8",
    )
    print(f"Wrote open OCR QA stress audit to {out_dir}")


def summarize_run(task: str, ratio: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for tag in TAG_ORDER:
        if tag == "all":
            subset = rows
        else:
            subset = [row for row in rows if tag in row["stress_tags"].split(";")]
        if not subset:
            continue
        full = mean(float(row["full_score"]) for row in subset)
        pruned = mean(float(row["pruned_score"]) for row in subset)
        full_exact = mean(float(row["full_exact"]) for row in subset)
        pruned_exact = mean(float(row["pruned_exact"]) for row in subset)
        out.append(
            {
                "task": task,
                "ratio": ratio,
                "stress_tag": tag,
                "n": len(subset),
                "full_score": f"{full:.3f}",
                "pruned_score": f"{pruned:.3f}",
                "delta_score": f"{pruned - full:.3f}",
                "full_exact": f"{full_exact:.3f}",
                "pruned_exact": f"{pruned_exact:.3f}",
                "delta_exact": f"{pruned_exact - full_exact:.3f}",
                "note": tag_note(tag),
            }
        )
    return out


def assign_tags(row: dict[str, Any], *, drop_threshold: float) -> list[str]:
    tags = ["all"]
    question_tokens = normalize_tokens(str(row.get("raw_question", "")))
    answers = row.get("gold_answers") or []
    answer_tokens = [normalize_tokens(str(answer)) for answer in answers]
    max_answer_tokens = max((len(tokens) for tokens in answer_tokens), default=0)
    question_text = f" {str(row.get('raw_question', '')).lower()} "
    answer_text = " ".join(str(answer) for answer in answers)

    if max_answer_tokens >= 3:
        tags.append("multi_token_answer_ge3")
    if len(question_tokens) >= 10:
        tags.append("long_question_ge10")
    if re.search(r"\b(and|both|between|during|from|to|per|versus|vs|respectively)\b", question_text):
        tags.append("multi_constraint_question")
    if re.search(r"\d", answer_text):
        tags.append("numeric_answer")
    if re.search(
        r"\b(date|year|total|amount|number|value|price|address|organization|company|name|invoice|account|form|page|table)\b",
        question_text,
    ):
        tags.append("layout_field_question")

    full_score = float(row.get("full_score", 0.0) or 0.0)
    pruned_score = float(row.get("pruned_score", 0.0) or 0.0)
    if full_score >= 0.80 and full_score - pruned_score >= drop_threshold:
        tags.append("full_good_pruned_drop_ge50")
    return tags


def example_row(task: str, ratio: str, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": task,
        "ratio": ratio,
        "sample_id": row.get("sample_id", ""),
        "question": row.get("raw_question", ""),
        "gold_answers": " | ".join(str(answer) for answer in row.get("gold_answers", [])),
        "full_answer": row.get("full_answer", ""),
        "pruned_answer": row.get("pruned_answer", ""),
        "full_score": f"{float(row.get('full_score', 0.0) or 0.0):.3f}",
        "pruned_score": f"{float(row.get('pruned_score', 0.0) or 0.0):.3f}",
        "delta_score": f"{float(row.get('pruned_score', 0.0) or 0.0) - float(row.get('full_score', 0.0) or 0.0):.3f}",
        "stress_tags": row.get("stress_tags", ""),
    }


def balanced_examples(examples: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    by_group: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in examples:
        by_group.setdefault((row["task"], row["ratio"]), []).append(row)
    for rows in by_group.values():
        rows.sort(key=lambda row: float(row["delta_score"]))
    selected: list[dict[str, Any]] = []
    rounds = max(1, math_ceil(limit, max(1, len(by_group))))
    for group in sorted(by_group):
        selected.extend(by_group[group][:rounds])
    selected.sort(key=lambda row: (row["task"], row["ratio"], float(row["delta_score"])))
    return selected[:limit]


def math_ceil(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def build_report(rows: list[dict[str, Any]], examples: list[dict[str, Any]]) -> str:
    lines = [
        "# Open OCR/Document QA Stress Audit",
        "",
        "This audit summarizes cached Qwen3-VL-8B native generation runs on TextVQA-lite and DocVQA-lite. Stress tags are heuristic subgroups, not ground-truth evidence annotations. They are used to show where aggressive visual-token pruning becomes fragile.",
        "",
        "## Subgroup Summary",
        "",
        "| Task | Keep | Stress tag | n | Full | Pruned | Delta | Note |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['ratio']} | {row['stress_tag']} | {row['n']} | "
            f"{row['full_score']} | {row['pruned_score']} | {row['delta_score']} | {row['note']} |"
        )
    lines.extend(
        [
            "",
            "## Representative Failure Examples",
            "",
            "| Task | Keep | Question | Gold | Full answer | Pruned answer | Delta | Tags |",
            "| --- | ---: | --- | --- | --- | --- | ---: | --- |",
        ]
    )
    for row in examples:
        lines.append(
            f"| {row['task']} | {row['ratio']} | {escape(row['question'])} | {escape(row['gold_answers'])} | "
            f"{escape(row['full_answer'])} | {escape(row['pruned_answer'])} | {row['delta_score']} | {row['stress_tags']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The stress audit supports a boundary claim rather than a new positive result: broad open-document QA needs higher retention or a stronger risk-triggered fallback.",
            "- DocVQA-lite is especially sensitive at 30% keep across long-question, numeric, and multi-token-answer subgroups.",
            "- TextVQA-lite is more tolerant at 50-70% keep, but numeric and multi-token answers still expose larger drops than the overall average.",
        ]
    )
    return "\n".join(lines) + "\n"


def tag_note(tag: str) -> str:
    notes = {
        "all": "all cached samples",
        "multi_token_answer_ge3": "gold answer has at least three normalized tokens",
        "long_question_ge10": "question has at least ten normalized tokens",
        "multi_constraint_question": "question contains a multi-constraint cue such as and/during/per/from-to",
        "numeric_answer": "gold answer contains at least one digit",
        "layout_field_question": "question contains common document field/layout terms",
        "full_good_pruned_drop_ge50": "outcome-only failure slice: full score >= 0.80 and pruned drop >= 0.50",
    }
    return notes.get(tag, "")


def normalize_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def mean(values: Any) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def escape(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
