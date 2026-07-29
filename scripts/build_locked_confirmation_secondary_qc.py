#!/usr/bin/env python3
"""Freeze a 100-row independent QC subset from locked confirmation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from build_hard_negative_full_qc_extension import build_viewer  # noqa: E402


PACKAGE = ROOT / "runs/problem_optimization_audit/hard_negative_confirmation_full_qc"
SEED = PACKAGE / "locked_confirmation_500_seed.jsonl"
OUTPUT = PACKAGE / "secondary_100"
SAMPLE_SEED = "locked-confirmation-hard-negative-secondary-100-v1"
QC_FIELDS = (
    "human_source_text_visible",
    "human_target_text_visible_same_image",
    "target_absent_after_case_punct_normalization",
    "source_bbox_matches_source_text",
    "qc_decision",
    "invalid_reason",
    "annotator_id",
    "annotator_notes",
)


def main() -> None:
    rows = [
        json.loads(line)
        for line in SEED.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 500:
        raise RuntimeError(f"Expected 500 locked-confirmation seed rows, found {len(rows)}.")

    selected = []
    for batch in sorted({str(row["qc_batch"]) for row in rows}):
        batch_rows = [row for row in rows if row["qc_batch"] == batch]
        if len(batch_rows) != 100:
            raise RuntimeError(f"Expected 100 rows in {batch}, found {len(batch_rows)}.")
        selected.extend(stratified_sample(batch_rows, 20))
    selected.sort(key=lambda row: (row["qc_batch"], stable_key(row)))
    if len(selected) != 100 or len({row["sample_id"] for row in selected}) != 100:
        raise RuntimeError("Secondary sample must contain 100 unique rows.")

    template = [{**row, **{field: "" for field in QC_FIELDS}} for row in selected]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "secondary_100_manifest.csv", selected)
    write_csv(OUTPUT / "secondary_100_template.csv", template)
    (OUTPUT / "secondary_100_seed.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in template),
        encoding="ascii",
    )
    (OUTPUT / "secondary_100_viewer.html").write_text(
        build_viewer(template, "locked_confirmation_secondary_100"), encoding="utf-8"
    )
    (OUTPUT / "README.md").write_text(readme(selected), encoding="utf-8")
    print(f"Wrote locked-confirmation secondary QC package to {OUTPUT}")


def stratified_sample(rows: list[dict[str, Any]], sample_size: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("difficulty_bucket", ""))].append(row)
    exact = {key: sample_size * len(group) / len(rows) for key, group in groups.items()}
    allocation = {key: math.floor(value) for key, value in exact.items()}
    remainder = sample_size - sum(allocation.values())
    order = sorted(groups, key=lambda key: (-(exact[key] - allocation[key]), key))
    for key in order[:remainder]:
        allocation[key] += 1
    return [
        row
        for key in sorted(groups)
        for row in sorted(groups[key], key=stable_key)[: allocation[key]]
    ]


def stable_key(row: dict[str, Any]) -> str:
    return hashlib.sha256(f"{SAMPLE_SEED}|{row['sample_id']}".encode("utf-8")).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def readme(rows: list[dict[str, Any]]) -> str:
    batches = Counter(str(row["qc_batch"]) for row in rows)
    allocation = ", ".join(f"{key}={value}" for key, value in sorted(batches.items()))
    return f"""# Locked-Confirmation Independent Secondary QC

This package freezes 100 rows before primary annotations are revealed. It
samples 20 rows from each locked-confirmation batch and stratifies by
`difficulty_bucket` using SHA-256 ordering with seed `{SAMPLE_SEED}`.

- Batch allocation: {allocation}
- Source: `../locked_confirmation_500_seed.jsonl`
- All annotation fields were blank when this package was generated.

Annotator B must complete `secondary_100_viewer.html` independently and export
to `secondary_100_template.csv`. Do not inspect primary batch CSVs before the
independent export has been frozen.
"""


if __name__ == "__main__":
    main()
