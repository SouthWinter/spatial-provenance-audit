#!/usr/bin/env python3
"""Merge primary QC, independent overlap, and completed adjudications."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = (
    ROOT
    / "runs/problem_optimization_audit/hard_negative_full_qc_extension"
)
SECONDARY = PACKAGE / "secondary_100"
CORE_FIELDS = (
    "human_source_text_visible",
    "human_target_text_visible_same_image",
    "target_absent_after_case_punct_normalization",
    "source_bbox_matches_source_text",
    "qc_decision",
)
QC_FIELDS = CORE_FIELDS + (
    "invalid_reason",
    "annotator_id",
    "annotator_notes",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default=str(PACKAGE))
    parser.add_argument("--primary-batches", type=int, default=4)
    parser.add_argument(
        "--output-name",
        default="hard_negative_remaining_400_adjudicated.csv",
    )
    args = parser.parse_args()
    package = Path(args.package)
    secondary_dir = package / "secondary_100"

    primary = []
    for index in range(1, args.primary_batches + 1):
        primary.extend(
            read_csv(package / f"batch_{index:02d}/batch_{index:02d}_template.csv")
        )
    secondary = read_csv(
        secondary_dir / "secondary_100_pre_adjudication_frozen.csv"
    )
    queue = read_csv(secondary_dir / "disagreement_queue.csv")
    validate(primary, secondary, queue, args.primary_batches * 100)

    secondary_by_id = {row["sample_id"]: row for row in secondary}
    queue_by_id = {row["sample_id"]: row for row in queue}
    final_rows = []
    for primary_row in primary:
        row = dict(primary_row)
        sample_id = row["sample_id"]
        row["secondary_reviewed"] = "yes" if sample_id in secondary_by_id else "no"
        row["pre_adjudication_agreement"] = ""
        row["annotation_provenance"] = "primary_A_only"
        row["adjudicator_id"] = ""
        row["adjudication_notes"] = ""

        if sample_id in secondary_by_id:
            secondary_row = secondary_by_id[sample_id]
            matches = all(
                row[field].strip() == secondary_row[field].strip()
                for field in CORE_FIELDS
            )
            row["pre_adjudication_agreement"] = "yes" if matches else "no"
            row["annotation_provenance"] = "A_B_agreement"
            if not matches:
                decision = queue_by_id[sample_id]
                for field in QC_FIELDS:
                    row[field] = decision[f"final_{field}"]
                row["annotation_provenance"] = "adjudicated_by_B"
                row["adjudicator_id"] = decision["adjudicator_id"]
                row["adjudication_notes"] = decision["adjudication_notes"]
        final_rows.append(row)

    output = package / args.output_name
    write_csv(output, final_rows)
    summary = build_summary(final_rows, queue)
    write_csv(secondary_dir / "adjudication_summary.csv", summary)
    (secondary_dir / "adjudication_report.md").write_text(
        build_report(summary, output),
        encoding="utf-8",
    )
    print(f"Wrote final {len(final_rows)}-row adjudicated QC to {output}")


def validate(
    primary: list[dict[str, str]],
    secondary: list[dict[str, str]],
    queue: list[dict[str, str]],
    expected_primary: int,
) -> None:
    if (
        len(primary) != expected_primary
        or len({row["sample_id"] for row in primary}) != expected_primary
    ):
        raise SystemExit(
            f"Primary annotations must contain {expected_primary} unique rows"
        )
    if len(secondary) != 100 or len({row["sample_id"] for row in secondary}) != 100:
        raise SystemExit("Secondary annotations must contain 100 unique rows")
    secondary_ids = {row["sample_id"] for row in secondary}
    primary_by_id = {row["sample_id"]: row for row in primary}
    expected_queue = {
        sample_id
        for sample_id in secondary_ids
        if any(
            primary_by_id[sample_id][field].strip()
            != next(
                row[field].strip()
                for row in secondary
                if row["sample_id"] == sample_id
            )
            for field in CORE_FIELDS
        )
    }
    queue_ids = {row["sample_id"] for row in queue}
    if queue_ids != expected_queue:
        raise SystemExit(
            f"Adjudication queue IDs differ from disagreements: "
            f"queue={len(queue_ids)}, expected={len(expected_queue)}"
        )
    for row in queue:
        if row.get("adjudication_status", "").strip() != "adjudicated":
            raise SystemExit(f"Unresolved adjudication: {row['sample_id']}")
        if row.get("adjudicator_id", "").strip() != "B":
            raise SystemExit(
                f"Unexpected adjudicator for {row['sample_id']}: "
                f"{row.get('adjudicator_id')!r}"
            )
        missing = [
            f"final_{field}"
            for field in CORE_FIELDS
            if not row.get(f"final_{field}", "").strip()
        ]
        if missing:
            raise SystemExit(
                f"Adjudication {row['sample_id']} missing {','.join(missing)}"
            )


def build_summary(
    final_rows: list[dict[str, str]],
    queue: list[dict[str, str]],
) -> list[dict[str, Any]]:
    provenance = Counter(row["annotation_provenance"] for row in final_rows)
    decisions = Counter(row["qc_decision"] for row in final_rows)
    rows = [
        {"scope": "all", "metric": "final_rows", "value": len(final_rows)},
        {
            "scope": "overlap",
            "metric": "secondary_rows",
            "value": sum(row["secondary_reviewed"] == "yes" for row in final_rows),
        },
        {
            "scope": "overlap",
            "metric": "adjudicated_disagreement_rows",
            "value": len(queue),
        },
    ]
    rows.extend(
        {"scope": "provenance", "metric": key, "value": value}
        for key, value in sorted(provenance.items())
    )
    rows.extend(
        {"scope": "final_qc_decision", "metric": key, "value": value}
        for key, value in sorted(decisions.items())
    )
    return rows


def build_report(
    summary: list[dict[str, Any]],
    output: Path,
) -> str:
    return "\n".join(
        [
            "# Hard-Negative QC Adjudication",
            "",
            f"- Final artifact: `{output}`",
            f"- Primary scope: {sum(row['metric'] == 'final_rows' and int(row['value']) or 0 for row in summary)} rows labeled by A.",
            "- Reliability scope: a frozen 100-row subset independently labeled by B.",
            "- Disagreements on any of five core QC fields were reviewed by B after the independent export was frozen.",
            "",
            table_md(summary),
            "",
        ]
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table_md(rows: list[dict[str, Any]]) -> str:
    cols = list(rows[0])
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(row[col]).replace("|", "/") for col in cols) + " |"
        for row in rows
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
