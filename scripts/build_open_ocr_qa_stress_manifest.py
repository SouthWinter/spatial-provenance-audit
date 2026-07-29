#!/usr/bin/env python3
"""Build a balanced annotation manifest for open OCR/document QA stress cases.

The manifest turns cached TextVQA-lite/DocVQA-lite generation results into a
stable queue for future evidence-box or multi-evidence annotation. It is not an
annotation result: the evidence fields are intentionally blank/TBD so the file
can be filled by a human or a later OCR/layout pipeline.
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

STRESS_TAGS = (
    "multi_token_answer_ge3",
    "long_question_ge10",
    "multi_constraint_question",
    "numeric_answer",
    "layout_field_question",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="runs/problem_optimization_audit/open_ocr_qa_stress_manifest")
    parser.add_argument("--per-task-limit", type=int, default=48)
    parser.add_argument("--drop-threshold", type=float, default=0.50)
    args = parser.parse_args()

    grouped = load_grouped_runs()
    candidates = [build_candidate(task, sample_id, by_ratio, args.drop_threshold) for (task, sample_id), by_ratio in grouped.items()]
    candidates = [row for row in candidates if row is not None]
    selected = select_balanced(candidates, per_task_limit=args.per_task_limit)
    summary = summarize(selected, candidates)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "open_ocr_qa_stress_manifest.csv", selected)
    write_jsonl(out_dir / "open_ocr_qa_stress_manifest.jsonl", selected)
    write_csv(out_dir / "open_ocr_qa_stress_manifest_summary.csv", summary)
    (out_dir / "open_ocr_qa_stress_manifest.md").write_text(build_markdown(selected, summary), encoding="utf-8")
    print(f"Wrote {len(selected)} stress manifest rows to {out_dir}")


def load_grouped_runs() -> dict[tuple[str, str], dict[str, dict[str, Any]]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for task, ratio, rel_path in RUNS:
        for row in read_jsonl(ROOT / rel_path):
            grouped.setdefault((task, str(row["sample_id"])), {})[ratio] = row
    return grouped


def build_candidate(
    task: str,
    sample_id: str,
    by_ratio: dict[str, dict[str, Any]],
    drop_threshold: float,
) -> dict[str, Any] | None:
    if not all(ratio in by_ratio for ratio in ("0.30", "0.50", "0.70")):
        return None
    base = by_ratio["0.30"]
    full_score = float(base.get("full_score", 0.0) or 0.0)
    if full_score < 0.80:
        return None
    tags = assign_tags(base)
    scores = {ratio: float(by_ratio[ratio].get("pruned_score", 0.0) or 0.0) for ratio in ("0.30", "0.50", "0.70")}
    drops = {ratio: scores[ratio] - full_score for ratio in scores}
    reasons = []
    if drops["0.30"] <= -drop_threshold:
        reasons.append("low_budget_failure")
    if drops["0.50"] <= -drop_threshold:
        reasons.append("mid_budget_failure")
    if drops["0.70"] <= -drop_threshold:
        reasons.append("persistent_failure")
    if drops["0.30"] <= -drop_threshold and drops["0.70"] >= -0.20:
        reasons.append("recovered_by_70")
    if tags and drops["0.30"] >= -0.20:
        reasons.append("stress_control_no_large_drop")
    if not reasons:
        return None
    question = str(base.get("raw_question", ""))
    answers = [str(answer) for answer in base.get("gold_answers", [])]
    return {
        "task": task,
        "sample_id": sample_id,
        "question_id": base.get("question_id", ""),
        "selection_reasons": ";".join(reasons),
        "stress_tags": ";".join(tags),
        "stress_tag_count": len(tags),
        "question": question,
        "gold_answers": " | ".join(answers),
        "evidence_text_candidates": " | ".join(short_unique_answers(answers)),
        "full_answer": base.get("full_answer", ""),
        "full_score": f"{full_score:.3f}",
        "pruned_0p30_answer": by_ratio["0.30"].get("pruned_answer", ""),
        "pruned_0p30_score": f"{scores['0.30']:.3f}",
        "delta_0p30": f"{drops['0.30']:.3f}",
        "pruned_0p50_answer": by_ratio["0.50"].get("pruned_answer", ""),
        "pruned_0p50_score": f"{scores['0.50']:.3f}",
        "delta_0p50": f"{drops['0.50']:.3f}",
        "pruned_0p70_answer": by_ratio["0.70"].get("pruned_answer", ""),
        "pruned_0p70_score": f"{scores['0.70']:.3f}",
        "delta_0p70": f"{drops['0.70']:.3f}",
        "annotation_status": "TBD",
        "manual_evidence_region_count": "",
        "manual_evidence_texts": "",
        "manual_bbox_or_region_notes": "",
        "audit_note": "Heuristic stress manifest; not yet manually annotated.",
    }


def assign_tags(row: dict[str, Any]) -> list[str]:
    tags = []
    question_tokens = normalize_tokens(str(row.get("raw_question", "")))
    answers = [str(answer) for answer in row.get("gold_answers", [])]
    answer_tokens = [normalize_tokens(answer) for answer in answers]
    max_answer_tokens = max((len(tokens) for tokens in answer_tokens), default=0)
    question_text = f" {str(row.get('raw_question', '')).lower()} "
    answer_text = " ".join(answers)
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
    return tags


def select_balanced(candidates: list[dict[str, Any]], *, per_task_limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for task in ("DocVQA-lite", "TextVQA-lite"):
        task_rows = [row for row in candidates if row["task"] == task]
        quotas = (
            ("persistent_failure", per_task_limit // 4),
            ("mid_budget_failure", per_task_limit // 4),
            ("recovered_by_70", per_task_limit // 4),
            ("stress_control_no_large_drop", per_task_limit - 3 * (per_task_limit // 4)),
        )
        task_selected: list[dict[str, Any]] = []
        for reason, quota in quotas:
            pool = [row for row in task_rows if reason in row["selection_reasons"].split(";") and row["sample_id"] not in seen]
            pool.sort(key=selection_key)
            for row in pool[:quota]:
                task_selected.append(row)
                seen.add(row["sample_id"])
        if len(task_selected) < per_task_limit:
            pool = [row for row in task_rows if row["sample_id"] not in seen]
            pool.sort(key=selection_key)
            for row in pool[: per_task_limit - len(task_selected)]:
                task_selected.append(row)
                seen.add(row["sample_id"])
        selected.extend(task_selected)
    selected.sort(key=lambda row: (row["task"], row["selection_reasons"], row["sample_id"]))
    return selected


def selection_key(row: dict[str, Any]) -> tuple[float, int, str]:
    worst_drop = min(float(row["delta_0p30"]), float(row["delta_0p50"]), float(row["delta_0p70"]))
    return (worst_drop, -int(row["stress_tag_count"]), str(row["sample_id"]))


def summarize(selected: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for task in ("DocVQA-lite", "TextVQA-lite"):
        task_candidates = [row for row in candidates if row["task"] == task]
        task_selected = [row for row in selected if row["task"] == task]
        rows.append({"scope": task, "group": "candidate_pool", "count": len(task_candidates)})
        rows.append({"scope": task, "group": "selected_manifest", "count": len(task_selected)})
        for reason in ("low_budget_failure", "mid_budget_failure", "persistent_failure", "recovered_by_70", "stress_control_no_large_drop"):
            rows.append(
                {
                    "scope": task,
                    "group": reason,
                    "count": sum(reason in row["selection_reasons"].split(";") for row in task_selected),
                }
            )
        for tag in STRESS_TAGS:
            rows.append(
                {
                    "scope": task,
                    "group": tag,
                    "count": sum(tag in row["stress_tags"].split(";") for row in task_selected),
                }
            )
    return rows


def build_markdown(selected: list[dict[str, Any]], summary: list[dict[str, Any]]) -> str:
    lines = [
        "# Open OCR/Document QA Stress Manifest",
        "",
        "This file is a reproducible queue for future manual evidence-box or multi-evidence annotation. Rows are selected from cached TextVQA-lite and DocVQA-lite native generation outputs. Evidence annotation fields are intentionally blank.",
        "",
        "## Summary",
        "",
        "| Scope | Group | Count |",
        "| --- | --- | ---: |",
    ]
    for row in summary:
        lines.append(f"| {row['scope']} | {row['group']} | {row['count']} |")
    lines.extend(
        [
            "",
            "## Preview",
            "",
            "| Task | Reasons | Tags | Question | Gold | Full | 30% | 50% | 70% |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in selected[:24]:
        lines.append(
            f"| {row['task']} | {row['selection_reasons']} | {row['stress_tags']} | "
            f"{escape(row['question'])} | {escape(row['gold_answers'])} | {escape(row['full_answer'])} | "
            f"{escape(row['pruned_0p30_answer'])} ({row['delta_0p30']}) | "
            f"{escape(row['pruned_0p50_answer'])} ({row['delta_0p50']}) | "
            f"{escape(row['pruned_0p70_answer'])} ({row['delta_0p70']}) |"
        )
    lines.extend(
        [
            "",
            "## Use",
            "",
            "- Fill `manual_evidence_region_count`, `manual_evidence_texts`, and `manual_bbox_or_region_notes` after inspecting the source image.",
            "- Treat this as a queue, not as evidence that multi-evidence boxes have already been annotated.",
            "- Prioritize persistent failures first, then cases recovered by 70% retention to study which evidence reappears at higher budgets.",
        ]
    )
    return "\n".join(lines) + "\n"


def short_unique_answers(answers: list[str]) -> list[str]:
    out = []
    seen = set()
    for answer in answers:
        text = re.sub(r"\s+", " ", answer.strip())
        key = text.lower()
        if text and key not in seen:
            out.append(text)
            seen.add(key)
    return out[:4]


def normalize_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def escape(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


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
