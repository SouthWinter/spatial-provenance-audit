#!/usr/bin/env python3
"""Validate and combine the 12+20 independent evidence-box annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from scripts.audit_open_ocr_qa_annotation_agreement import (
        build_report,
        build_summary,
        compare_sample,
        load_annotations,
        load_by_sample,
        write_csv,
    )
except ModuleNotFoundError:  # Direct execution adds scripts/, not the repository root.
    from audit_open_ocr_qa_annotation_agreement import (
        build_report,
        build_summary,
        compare_sample,
        load_annotations,
        load_by_sample,
        write_csv,
    )


ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = ROOT / "runs" / "problem_optimization_audit"
PRIMARY = (
    AUDIT_ROOT
    / "open_ocr_qa_manual_annotation_launch"
    / "primary_full_tool"
    / "primary_full_tool_annotations.json"
)
ORIGINAL_SECONDARY = (
    AUDIT_ROOT
    / "open_ocr_qa_manual_annotation_launch"
    / "secondary_calibration_tool"
    / "secondary_calibration_tool_annotations.json"
)
EXTENSION_ROOT = AUDIT_ROOT / "open_ocr_qa_secondary_extension"
EXTENSION_SECONDARY = (
    EXTENSION_ROOT
    / "secondary_extension_tool"
    / "secondary_extension_tool_annotations.json"
)
EXTENSION_SEED = EXTENSION_ROOT / "secondary_extension_seed.jsonl"
OUTPUT_DIR = EXTENSION_ROOT / "combined_32_agreement"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, default=PRIMARY)
    parser.add_argument("--original-secondary", type=Path, default=ORIGINAL_SECONDARY)
    parser.add_argument("--extension-secondary", type=Path, default=EXTENSION_SECONDARY)
    parser.add_argument("--extension-seed", type=Path, default=EXTENSION_SEED)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    args = parser.parse_args()

    required = (
        args.primary,
        args.original_secondary,
        args.extension_secondary,
        args.extension_seed,
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        paths = "\n".join(f"- {path}" for path in missing)
        raise SystemExit(f"Independent annotation package is not ready; missing:\n{paths}")

    primary = load_by_sample(args.primary)
    original_rows = load_annotations(args.original_secondary)
    extension_rows = load_annotations(args.extension_secondary)
    seed_ids = sample_ids(load_annotations(args.extension_seed), "extension seed")

    original = validate_secondary(original_rows, expected_count=12, name="original secondary")
    extension = validate_secondary(
        extension_rows,
        expected_count=20,
        name="secondary extension",
        expected_ids=seed_ids,
    )
    overlap = set(original) & set(extension)
    if overlap:
        raise SystemExit(f"Original and extension samples overlap: {sorted(overlap)}")

    secondary = {**original, **extension}
    missing_primary = sorted(set(secondary) - set(primary))
    if missing_primary:
        raise SystemExit(f"Secondary samples missing from primary export: {missing_primary}")

    agreement_rows = []
    for source_set, rows in (("original_12", original), ("extension_20", extension)):
        for sample_id in sorted(rows):
            comparison = compare_sample(
                sample_id,
                primary[sample_id],
                rows[sample_id],
                args.iou_threshold,
            )
            agreement_rows.append({"source_set": source_set, **comparison})

    summary = build_summary(agreement_rows)
    summary.extend(source_summary(agreement_rows, "original_12"))
    summary.extend(source_summary(agreement_rows, "extension_20"))
    label_type_rows = build_label_type_summary(agreement_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "annotation_agreement_rows.csv", agreement_rows)
    write_csv(args.output_dir / "annotation_agreement_summary.csv", summary)
    write_csv(args.output_dir / "annotation_label_type_summary.csv", label_type_rows)
    (args.output_dir / "combined_secondary_annotations.json").write_text(
        json.dumps(list(secondary.values()), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    generic_report = build_report(summary, agreement_rows, args.iou_threshold)
    header = (
        "# Combined 32-Row Independent Annotation Audit\n\n"
        "This report combines the original 12-row calibration set with the "
        "disjoint 20-row secondary extension. Agreement is measured before "
        "adjudication against the primary annotator's boxes.\n\n"
    )
    reconciliation = (
        "\n## Reconciliation Rule\n\n"
        "For the original 12 calibration rows, the secondary boxes were adopted after review, as recorded in the final 96-row package. The disjoint 20-row extension is retained as a pre-adjudication reliability audit and does not alter the final 96-row boxes. Any future replacement of those 20 primary rows requires an explicit human adjudication decision.\n"
    )
    (args.output_dir / "annotation_agreement_report.md").write_text(
        header
        + generic_report.removeprefix("# Annotation Agreement Audit\n\n")
        + build_label_type_report(label_type_rows)
        + reconciliation,
        encoding="utf-8",
    )
    print(
        "Audited 32 independent samples; "
        f"needs_adjudication={sum(int(row['needs_adjudication']) for row in agreement_rows)}"
    )


def sample_ids(rows: list[dict[str, Any]], name: str) -> set[str]:
    ids = [str(row.get("sample_id", "")).strip() for row in rows]
    if any(not sample_id for sample_id in ids):
        raise SystemExit(f"{name} contains an empty sample_id")
    if len(ids) != len(set(ids)):
        raise SystemExit(f"{name} contains duplicate sample_id values")
    return set(ids)


def validate_secondary(
    rows: list[dict[str, Any]],
    *,
    expected_count: int,
    name: str,
    expected_ids: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    ids = sample_ids(rows, name)
    if len(rows) != expected_count:
        raise SystemExit(f"{name} must contain {expected_count} rows, found {len(rows)}")
    if expected_ids is not None and ids != expected_ids:
        missing = sorted(expected_ids - ids)
        extra = sorted(ids - expected_ids)
        raise SystemExit(f"{name} sample mismatch; missing={missing}, extra={extra}")

    errors = []
    output = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        boxes = row.get("boxes", [])
        if row.get("status") != "annotated":
            errors.append(f"{sample_id}: status={row.get('status', '')!r}")
        if not isinstance(boxes, list) or not boxes:
            errors.append(f"{sample_id}: no evidence boxes")
        elif any(not str(box.get("label", "")).strip() for box in boxes if isinstance(box, dict)):
            errors.append(f"{sample_id}: unlabeled evidence box")
        output[sample_id] = row
    if errors:
        raise SystemExit(f"{name} is incomplete:\n- " + "\n- ".join(errors))
    return output


def source_summary(rows: list[dict[str, Any]], source_set: str) -> list[dict[str, Any]]:
    group = [row for row in rows if row["source_set"] == source_set]
    return [
        {"scope": source_set, "metric": "samples", "value": len(group)},
        {
            "scope": source_set,
            "metric": "box_count_match",
            "value": sum(row["box_count_a"] == row["box_count_b"] for row in group),
        },
        {
            "scope": source_set,
            "metric": "label_type_set_match",
            "value": sum(int(row["label_type_set_match"]) for row in group),
        },
        {
            "scope": source_set,
            "metric": "all_boxes_matched_iou",
            "value": sum(int(row["all_boxes_matched_iou"]) for row in group),
        },
        {
            "scope": source_set,
            "metric": "needs_adjudication",
            "value": sum(int(row["needs_adjudication"]) for row in group),
        },
        {
            "scope": source_set,
            "metric": "mean_best_iou",
            "value": round(
                sum(float(row["mean_matched_iou"]) for row in group) / len(group),
                4,
            ),
        },
        {
            "scope": source_set,
            "metric": "mean_union_region_iou",
            "value": round(
                sum(float(row["union_region_iou"]) for row in group) / len(group),
                4,
            ),
        },
        {
            "scope": source_set,
            "metric": "mean_primary_covered_by_secondary",
            "value": round(
                sum(float(row["primary_covered_by_secondary"]) for row in group) / len(group),
                4,
            ),
        },
        {
            "scope": source_set,
            "metric": "mean_secondary_covered_by_primary",
            "value": round(
                sum(float(row["secondary_covered_by_primary"]) for row in group) / len(group),
                4,
            ),
        },
    ]


def build_label_type_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scopes = [("all", rows)]
    scopes.extend(
        (task, [row for row in rows if row["task"] == task])
        for task in sorted({str(row["task"]) for row in rows})
    )
    output = []
    for scope, scoped_rows in scopes:
        label_types = sorted(
            {
                label
                for row in scoped_rows
                for field in ("label_types_a", "label_types_b")
                for label in str(row[field]).split(";")
                if label
            }
        )
        for label_type in label_types:
            a_present = [label_type in split_types(row["label_types_a"]) for row in scoped_rows]
            b_present = [label_type in split_types(row["label_types_b"]) for row in scoped_rows]
            both = sum(a and b for a, b in zip(a_present, b_present))
            either = sum(a or b for a, b in zip(a_present, b_present))
            exact = sum(a == b for a, b in zip(a_present, b_present))
            a_count = sum(a_present)
            b_count = sum(b_present)
            output.append(
                {
                    "scope": scope,
                    "label_type": label_type,
                    "samples": len(scoped_rows),
                    "primary_present": a_count,
                    "secondary_present": b_count,
                    "both_present": both,
                    "presence_agreement": round(exact / len(scoped_rows), 4),
                    "positive_jaccard": round(both / either, 4) if either else 1.0,
                    "positive_f1": round(2 * both / (a_count + b_count), 4) if a_count + b_count else 1.0,
                }
            )
    return output


def split_types(value: Any) -> set[str]:
    return {part for part in str(value).split(";") if part}


def build_label_type_report(rows: list[dict[str, Any]]) -> str:
    lines = [
        "\n## Region-Type Presence Agreement",
        "",
        "Presence is evaluated per sample before adjudication; Jaccard and F1 concern whether a region type appears, not geometric agreement.",
        "",
        "| Scope | Region type | n | Primary + | Secondary + | Both + | Agreement | Jaccard | F1 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['scope']} | {row['label_type']} | {row['samples']} | "
            f"{row['primary_present']} | {row['secondary_present']} | {row['both_present']} | "
            f"{row['presence_agreement']:.3f} | {row['positive_jaccard']:.3f} | "
            f"{row['positive_f1']:.3f} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
