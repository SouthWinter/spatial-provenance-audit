#!/usr/bin/env python3
"""Merge a primary annotation export with an approved calibration adjudication."""

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", required=True, help="Complete primary annotation JSON/JSONL export")
    parser.add_argument("--secondary", required=True, help="Approved secondary calibration JSON/JSONL export")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--resolution",
        choices=("prefer-primary", "prefer-secondary"),
        required=True,
        help="The human-approved source for overlapping calibration rows.",
    )
    args = parser.parse_args()

    primary = load_by_sample(Path(args.primary))
    secondary = load_by_sample(Path(args.secondary))
    unexpected = sorted(set(secondary) - set(primary))
    if unexpected:
        raise SystemExit(f"Secondary export contains {len(unexpected)} row(s) absent from primary: {unexpected}")
    if len(secondary) != 12:
        raise SystemExit(f"Expected exactly 12 calibration rows, found {len(secondary)}")

    resolved: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    for sample_id in sorted(primary):
        use_secondary = args.resolution == "prefer-secondary" and sample_id in secondary
        source_row = secondary[sample_id] if use_secondary else primary[sample_id]
        row = deepcopy(source_row)
        row["source"] = (
            "independent_secondary_adopted_after_adjudication"
            if use_secondary
            else "assisted_preannotation_human_confirmed"
        )
        row["annotation_provenance"] = (
            "independent secondary annotation adopted by human adjudication"
            if use_secondary
            else "assisted pre-annotation inspected, corrected where needed, and confirmed by a human annotator"
        )
        if use_secondary:
            row["notes"] = append_note(
                str(row.get("notes", "")),
                "Adjudication: independent secondary annotation adopted after primary-secondary review.",
            )
        resolved.append(row)
        ledger.append(
            {
                "sample_id": sample_id,
                "in_calibration_subset": int(sample_id in secondary),
                "selected_source": "secondary" if use_secondary else "primary",
                "primary_box_count": len(primary[sample_id].get("boxes", [])),
                "secondary_box_count": len(secondary[sample_id].get("boxes", [])) if sample_id in secondary else "",
            }
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "adjudicated_annotations.json", resolved)
    write_jsonl(out_dir / "adjudicated_annotations.jsonl", resolved)
    write_csv(out_dir / "adjudication_resolution_ledger.csv", ledger)
    summary = {
        "primary_rows": len(primary),
        "secondary_rows": len(secondary),
        "resolved_rows": len(resolved),
        "secondary_rows_adopted": sum(row["selected_source"] == "secondary" for row in ledger),
        "resolution": args.resolution,
    }
    (out_dir / "adjudication_resolution_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Resolved {len(resolved)} rows; adopted secondary for "
        f"{summary['secondary_rows_adopted']} calibration rows"
    )


def append_note(existing: str, note: str) -> str:
    existing = existing.strip()
    return f"{existing} | {note}" if existing else note


def load_by_sample(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_annotations(path)
    by_sample: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            raise SystemExit(f"Annotation row in {path} is missing sample_id")
        if sample_id in by_sample:
            raise SystemExit(f"Duplicate sample_id in {path}: {sample_id}")
        by_sample[sample_id] = row
    return by_sample


def load_annotations(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
