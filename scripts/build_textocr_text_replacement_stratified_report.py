#!/usr/bin/env python3
"""Stratify text-replacement model behavior by OCR edit readability."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, help="Joined before/after model pairs CSV.")
    parser.add_argument("--ocr-rows", required=True, help="OCR quality row CSV.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default="text_replacement_stratified")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_rows = read_csv(Path(args.pairs))
    ocr_rows = {row["source_sample_id"]: row for row in read_csv(Path(args.ocr_rows))}
    joined: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for row in pair_rows:
        source_id = row["source_sample_id"]
        ocr = ocr_rows.get(source_id)
        if not ocr:
            missing.append({"source_sample_id": source_id, "reason": "missing_ocr_row"})
            continue
        joined.append({**row, **{f"ocr_{key}": value for key, value in ocr.items() if key not in row}})

    groups = [
        ("all", lambda row: True),
        ("edited_crop_ocr_success", lambda row: boolish(row.get("ocr_edited_crop_ocr_success"))),
        ("edited_crop_ocr_failure", lambda row: not boolish(row.get("ocr_edited_crop_ocr_success"))),
        ("edited_crop_replacement_detected", lambda row: boolish(row.get("ocr_edited_crop_replacement_detected"))),
        ("edited_crop_replacement_not_detected", lambda row: not boolish(row.get("ocr_edited_crop_replacement_detected"))),
        ("source_original_detected_in_crop", lambda row: boolish(row.get("ocr_source_original_detected_in_crop"))),
    ]
    summary = [summarize_group(name, [row for row in joined if predicate(row)]) for name, predicate in groups]
    summary = [row for row in summary if int(row["n_pairs"]) > 0]

    write_csv(output_dir / f"{args.name}_joined_rows.csv", joined)
    write_csv(output_dir / f"{args.name}_summary.csv", summary)
    write_csv(output_dir / f"{args.name}_missing.csv", missing)
    (output_dir / f"{args.name}_report.md").write_text(render_markdown(summary, joined, missing), encoding="utf-8")
    print(f"Wrote stratified text-replacement report to {output_dir}")


def summarize_group(name: str, rows: list[dict[str, Any]]) -> dict[str, str]:
    return {
        "group": name,
        "n_pairs": str(len(rows)),
        "original_pair_correct_rate": rate(rows, "original_pair_correct"),
        "edited_pair_correct_rate": rate(rows, "edited_pair_correct"),
        "full_four_way_semantic_switch_rate": rate(rows, "full_semantic_switch"),
        "edited_pair_correct_given_original_correct": conditional_rate(rows, "original_pair_correct", "edited_pair_correct"),
        "source_absence_switch_rate": rate(rows, "source_absence_switch"),
        "replacement_presence_switch_rate": rate(rows, "replacement_presence_switch"),
        "mean_source_yes_support_drop": mean_field(rows, "source_yes_support_drop"),
        "mean_replacement_yes_support_gain": mean_field(rows, "replacement_yes_support_gain"),
        "ocr_replacement_detected_rate": rate(rows, "ocr_edited_crop_replacement_detected"),
        "ocr_source_absent_rate": rate(rows, "ocr_edited_crop_source_absent"),
        "ocr_success_rate": rate(rows, "ocr_edited_crop_ocr_success"),
    }


def conditional_rate(rows: list[dict[str, Any]], condition_key: str, value_key: str) -> str:
    selected = [row for row in rows if boolish(row.get(condition_key))]
    return rate(selected, value_key)


def rate(rows: list[dict[str, Any]], key: str) -> str:
    if not rows:
        return "0.000000"
    return f"{sum(boolish(row.get(key)) for row in rows) / len(rows):.6f}"


def mean_field(rows: list[dict[str, Any]], key: str) -> str:
    vals: list[float] = []
    for row in rows:
        try:
            vals.append(float(row.get(key, "")))
        except Exception:
            continue
    return f"{mean(vals):.6f}" if vals else "0.000000"


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def render_markdown(
    summary: list[dict[str, str]],
    rows: list[dict[str, Any]],
    missing: list[dict[str, str]],
) -> str:
    lines = [
        "# Text-Replacement OCR-Conditioned Model Behavior",
        "",
        "This report joins the before/after model counterfactual rows with the EasyOCR edit-quality audit. It asks whether model semantic switching improves when the inserted replacement is OCR-readable.",
        "",
        "## Summary",
        "",
        table_md(
            summary,
            [
                "group",
                "n_pairs",
                "full_four_way_semantic_switch_rate",
                "edited_pair_correct_given_original_correct",
                "source_absence_switch_rate",
                "replacement_presence_switch_rate",
                "mean_source_yes_support_drop",
                "mean_replacement_yes_support_gain",
                "ocr_success_rate",
            ],
        ),
        "",
        "## Interpretation",
        "",
        "Safe claim: conditioning on OCR-readable edits tests whether renderer quality explains the weak text-replacement result. Unsafe claim: the OCR-readable subset is a large human-verified benchmark.",
        "",
        "## Example OCR-Readable Rows",
        "",
        table_md(
            [row for row in rows if boolish(row.get("ocr_edited_crop_ocr_success"))][:20],
            [
                "image_id",
                "source_text",
                "replacement_text",
                "base_source_pred",
                "base_replacement_pred",
                "edited_source_pred",
                "edited_replacement_pred",
                "full_semantic_switch",
                "source_yes_support_drop",
                "replacement_yes_support_gain",
                "ocr_edited_crop_text",
            ],
        ),
    ]
    if missing:
        lines.extend(["", "## Missing", "", table_md(missing, ["source_sample_id", "reason"])])
    return "\n".join(lines) + "\n"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def table_md(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "(empty)"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")).replace("|", "\\|") for col in columns) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
