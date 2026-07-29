#!/usr/bin/env python3
"""Summarize open-QA detector-in-loop generation against cached controls."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotation-pack",
        default="runs/problem_optimization_audit/open_ocr_qa_annotation_pack/annotation_pack.csv",
    )
    parser.add_argument(
        "--detector-soft-run",
        default=(
            "runs/problem_optimization_audit/open_ocr_qa_detector_in_loop/"
            "qwen3_8b_easyocr_soft_evidence0p70_full96/open_ocr_qa_detector_generation.jsonl"
        ),
    )
    parser.add_argument(
        "--grid-metadata-run",
        default=(
            "runs/problem_optimization_audit/open_ocr_qa_detector_in_loop/"
            "qwen3_8b_easyocr_target_grid0p70_full96/open_ocr_qa_detector_generation.jsonl"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="runs/problem_optimization_audit/open_ocr_qa_detector_in_loop_readout",
    )
    args = parser.parse_args()

    annotation_rows = read_csv(Path(args.annotation_pack))
    soft_rows = index_by_sample(read_jsonl(Path(args.detector_soft_run)))
    grid_rows = index_by_sample(read_jsonl(Path(args.grid_metadata_run)))

    joined: list[dict[str, Any]] = []
    for row in annotation_rows:
        sample_id = row["sample_id"]
        soft = soft_rows.get(sample_id)
        grid = grid_rows.get(sample_id)
        if soft is None:
            continue
        cached_score = as_float(row.get("pruned_0p70_score"))
        soft_score = as_float(soft.get("pruned_score"))
        grid_score = as_float(grid.get("pruned_score")) if grid else None
        joined.append(
            {
                "sample_id": sample_id,
                "dataset": row["task"],
                "question_id": row["question_id"],
                "full_score": as_float(soft.get("full_score")),
                "cached_question_grid70_score": cached_score,
                "easyocr_soft_evidence70_score": soft_score,
                "grid_metadata_only70_score": grid_score,
                "soft_minus_cached_grid70": diff(soft_score, cached_score),
                "soft_minus_grid_metadata70": diff(soft_score, grid_score),
                "detector_box_count": int(soft.get("detector_box_count", 0) or 0),
                "has_detector_boxes": int(int(soft.get("detector_box_count", 0) or 0) > 0),
                "soft_answer": soft.get("pruned_answer", ""),
                "cached_grid70_answer": row.get("pruned_0p70_answer", ""),
                "grid_metadata_only_answer": grid.get("pruned_answer", "") if grid else "",
            }
        )

    summaries = build_summaries(joined)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "detector_in_loop_joined_rows.csv", joined)
    write_csv(out_dir / "detector_in_loop_summary.csv", summaries)
    write_markdown(out_dir / "detector_in_loop_readout.md", summaries, joined, args)
    print(f"Wrote detector-in-loop readout to {out_dir}")


def build_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {"all": rows}
    for task in sorted({str(row["dataset"]) for row in rows}):
        groups[task] = [row for row in rows if row["dataset"] == task]
    groups["rows_with_detector_boxes"] = [row for row in rows if row["has_detector_boxes"]]
    groups["rows_without_detector_boxes"] = [row for row in rows if not row["has_detector_boxes"]]

    summaries: list[dict[str, Any]] = []
    for name, group in groups.items():
        if not group:
            continue
        summaries.append(
            {
                "scope": name,
                "n": len(group),
                "full_score": fmt(mean(row["full_score"] for row in group)),
                "cached_question_grid70_score": fmt(mean(row["cached_question_grid70_score"] for row in group)),
                "easyocr_soft_evidence70_score": fmt(mean(row["easyocr_soft_evidence70_score"] for row in group)),
                "grid_metadata_only70_score": mean_present(row["grid_metadata_only70_score"] for row in group),
                "soft_minus_cached_grid70": fmt(mean(row["soft_minus_cached_grid70"] for row in group)),
                "soft_minus_grid_metadata70": mean_present(row["soft_minus_grid_metadata70"] for row in group),
                "mean_detector_box_count": fmt(mean(row["detector_box_count"] for row in group)),
                "rows_with_detector_boxes": sum(row["has_detector_boxes"] for row in group),
            }
        )
    return summaries


def write_markdown(path: Path, summaries: list[dict[str, Any]], rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        "# Open-QA Detector-in-Loop Readout",
        "",
        "This readout separates three 70% Qwen open-QA settings on the fixed 96-row stress pack:",
        "",
        "- cached question/grid 70%: original annotation-pack cached pruning result.",
        "- grid metadata-only 70%: rerun from detector probe files with `target_embed_grid_topk`; this selector ignores detector boxes and is not a detector-box-driven result.",
        "- EasyOCR soft-evidence 70%: rerun from detector probe files with `target_embed_soft_evidence_topk`; selector-visible EasyOCR boxes enter token selection.",
        "",
        f"Detector soft run: `{args.detector_soft_run}`",
        f"Grid metadata-only run: `{args.grid_metadata_run}`",
        "",
        "## Summary",
        "",
        "| Scope | n | Full | Cached grid70 | EasyOCR soft70 | Grid metadata70 | Soft - cached | Soft - metadata | Boxes/row | Rows w/ boxes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            "| {scope} | {n} | {full_score} | {cached_question_grid70_score} | "
            "{easyocr_soft_evidence70_score} | {grid_metadata_only70_score} | "
            "{soft_minus_cached_grid70} | {soft_minus_grid_metadata70} | "
            "{mean_detector_box_count} | {rows_with_detector_boxes} |".format(**row)
        )

    improved = sorted(rows, key=lambda row: row["soft_minus_cached_grid70"], reverse=True)[:8]
    worsened = sorted(rows, key=lambda row: row["soft_minus_cached_grid70"])[:8]
    lines.extend(["", "## Largest Soft-Evidence Improvements vs Cached Grid70", ""])
    lines.extend(example_table(improved))
    lines.extend(["", "## Largest Soft-Evidence Regressions vs Cached Grid70", ""])
    lines.extend(example_table(worsened))
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "The EasyOCR soft-evidence row is a true detector-box-driven selection run. "
            "The grid metadata-only row is retained only as a control because its selector does not read `evidence_regions`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def example_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Sample | Dataset | Cached | Soft | Delta | Boxes | Cached answer | Soft answer |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {sample_id} | {dataset} | {cached_question_grid70_score:.3f} | "
            "{easyocr_soft_evidence70_score:.3f} | {soft_minus_cached_grid70:+.3f} | "
            "{detector_box_count} | {cached_grid70_answer} | {soft_answer} |".format(
                **{**row, "cached_grid70_answer": clean_cell(row["cached_grid70_answer"]), "soft_answer": clean_cell(row["soft_answer"])}
            )
        )
    return lines


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def index_by_sample(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("sample_id", "")): row for row in rows}


def as_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def diff(left: float | None, right: float | None) -> float:
    if left is None or right is None:
        return 0.0
    return float(left) - float(right)


def mean_present(values: Any) -> str:
    present = [float(value) for value in values if value is not None]
    if not present:
        return ""
    return fmt(mean(present))


def fmt(value: float) -> str:
    return f"{value:.3f}"


def clean_cell(value: Any, max_len: int = 56) -> str:
    text = str(value).replace("|", "/").replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


if __name__ == "__main__":
    main()
