#!/usr/bin/env python3
"""Prefill evidence-target fields for the open OCR QA annotation pack.

This script does not infer boxes. It normalizes answer strings into text/number
targets and adds annotation hints so a human or a future OCR/layout detector can
fill region boxes consistently.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PACK_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_annotation_pack"
    / "annotation_pack.csv"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", default=str(PACK_CSV))
    parser.add_argument("--output-dir", default="runs/problem_optimization_audit/open_ocr_qa_evidence_prefill")
    args = parser.parse_args()

    rows = []
    for row in read_csv(Path(args.pack)):
        units = evidence_units(row)
        rows.append(
            {
                **row,
                "prefill_evidence_units": " | ".join(units),
                "prefill_unit_count": len(units),
                "prefill_has_numeric": int(any(re.search(r"\d", unit) for unit in units)),
                "prefill_complexity": complexity(row, units),
                "prefill_suggested_region_count": suggested_region_count(row, units),
                "prefill_annotation_hint": annotation_hint(row, units),
                "prefill_status": "text_targets_only_no_boxes",
            }
        )
    summary = summarize(rows)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "evidence_prefill_pack.csv", rows)
    write_jsonl(out_dir / "evidence_prefill_pack.jsonl", rows)
    write_csv(out_dir / "evidence_prefill_summary.csv", summary)
    (out_dir / "evidence_prefill_report.md").write_text(build_markdown(summary, rows), encoding="utf-8")
    print(f"Wrote evidence prefill for {len(rows)} rows to {out_dir}")


def evidence_units(row: dict[str, str]) -> list[str]:
    candidates = split_candidates(row.get("evidence_text_candidates", "")) or split_candidates(row.get("gold_answers", ""))
    out = []
    seen = set()
    for candidate in candidates:
        candidate = clean_text(candidate)
        if not candidate:
            continue
        chunks = split_phrase(candidate)
        if len(chunks) <= 1:
            chunks = [candidate]
        for chunk in chunks:
            chunk = clean_text(chunk)
            key = normalize_key(chunk)
            if key and key not in seen:
                out.append(chunk)
                seen.add(key)
    return out[:8]


def split_candidates(text: str) -> list[str]:
    parts = [part.strip() for part in str(text).split("|")]
    return [part for part in parts if part]


def split_phrase(text: str) -> list[str]:
    if re.search(r"\d", text):
        return [text]
    pieces = re.split(r"\b(?:and|or|,|;|/)\b", text, flags=re.IGNORECASE)
    pieces = [piece.strip(" ,;/.") for piece in pieces if piece.strip(" ,;/.")]
    if 1 < len(pieces) <= 4:
        return pieces
    return [text]


def complexity(row: dict[str, str], units: list[str]) -> str:
    tags = set(str(row.get("stress_tags", "")).split(";"))
    if "persistent_failure" in row.get("selection_reasons", ""):
        return "hard_persistent"
    if len(units) >= 3 or {"multi_constraint_question", "layout_field_question"} <= tags:
        return "likely_multi_region"
    if any(re.search(r"\d", unit) for unit in units) or "numeric_answer" in tags:
        return "numeric_or_field"
    return "single_region_likely"


def suggested_region_count(row: dict[str, str], units: list[str]) -> int:
    if not units:
        return 1
    if "likely_multi_region" == complexity(row, units):
        return min(4, max(2, len(units)))
    return min(3, max(1, len(units)))


def annotation_hint(row: dict[str, str], units: list[str]) -> str:
    if units:
        target = "; ".join(units)
    else:
        target = row.get("gold_answers", "")
    return (
        "Mark the minimal visible region(s) that support the answer. "
        f"Prioritize these text/value targets: {target}. "
        "If the answer requires a row, column, header, or nearby label, include that context as a separate region note."
    )


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for scope in ("all", "DocVQA-lite", "TextVQA-lite"):
        scoped = rows if scope == "all" else [row for row in rows if row["task"] == scope]
        if not scoped:
            continue
        out.append({"scope": scope, "metric": "rows", "value": len(scoped)})
        out.append({"scope": scope, "metric": "mean_prefill_unit_count", "value": f"{mean(int(row['prefill_unit_count']) for row in scoped):.2f}"})
        out.append({"scope": scope, "metric": "numeric_rows", "value": sum(int(row["prefill_has_numeric"]) for row in scoped)})
        for name in ("hard_persistent", "likely_multi_region", "numeric_or_field", "single_region_likely"):
            out.append({"scope": scope, "metric": f"complexity_{name}", "value": sum(row["prefill_complexity"] == name for row in scoped)})
    return out


def build_markdown(summary: list[dict[str, Any]], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Open OCR QA Evidence Prefill",
        "",
        "This artifact adds text/value evidence targets to the annotation pack. It does not provide bounding boxes.",
        "",
        "## Summary",
        "",
        "| Scope | Metric | Value |",
        "| --- | --- | ---: |",
    ]
    for row in summary:
        lines.append(f"| {row['scope']} | {row['metric']} | {row['value']} |")
    lines.extend(
        [
            "",
            "## Preview",
            "",
            "| Task | Reasons | Complexity | Units | Hint |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows[:24]:
        lines.append(
            f"| {row['task']} | {row['selection_reasons']} | {row['prefill_complexity']} | "
            f"{escape(row['prefill_evidence_units'])} | {escape(row['prefill_annotation_hint'])} |"
        )
    return "\n".join(lines) + "\n"


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip(" .")


def normalize_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def mean(values: Any) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def escape(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
