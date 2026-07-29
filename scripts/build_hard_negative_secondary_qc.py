#!/usr/bin/env python3
"""Build a frozen 100-row independent reannotation package."""

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


PACKAGE = (
    ROOT
    / "runs/problem_optimization_audit/hard_negative_full_qc_extension"
)
SEED = PACKAGE / "hard_negative_remaining_400_seed.jsonl"
OUTPUT = PACKAGE / "secondary_100"
SAMPLE_SEED = "hard-negative-secondary-100-v1"
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
    rows = read_jsonl(SEED)
    if len(rows) != 400:
        raise SystemExit(f"Expected 400 seed rows, found {len(rows)}")

    selected: list[dict[str, Any]] = []
    for batch in sorted({str(row["qc_batch"]) for row in rows}):
        batch_rows = [row for row in rows if row["qc_batch"] == batch]
        if len(batch_rows) != 100:
            raise SystemExit(f"Expected 100 rows in {batch}, found {len(batch_rows)}")
        selected.extend(stratified_sample(batch_rows, 25))

    if len(selected) != 100:
        raise SystemExit(f"Expected 100 selected rows, found {len(selected)}")
    if len({row["sample_id"] for row in selected}) != 100:
        raise SystemExit("Secondary sample contains duplicate sample IDs")

    selected.sort(key=lambda row: (row["qc_batch"], stable_key(row)))
    template = [
        {
            **row,
            **{field: "" for field in QC_FIELDS},
        }
        for row in selected
    ]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "secondary_100_manifest.csv", selected)
    write_csv(OUTPUT / "secondary_100_template.csv", template)
    write_jsonl(OUTPUT / "secondary_100_seed.jsonl", template)
    (OUTPUT / "secondary_100_viewer.html").write_text(
        build_viewer(template, "secondary_100"),
        encoding="utf-8",
    )
    (OUTPUT / "README.md").write_text(build_readme(selected), encoding="utf-8")
    print(f"Wrote frozen 100-row secondary package to {OUTPUT}")


def stratified_sample(rows: list[dict[str, Any]], sample_size: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("difficulty_bucket", ""))].append(row)

    exact = {key: sample_size * len(group) / len(rows) for key, group in groups.items()}
    allocation = {key: math.floor(value) for key, value in exact.items()}
    remainder = sample_size - sum(allocation.values())
    order = sorted(
        groups,
        key=lambda key: (-(exact[key] - allocation[key]), key),
    )
    for key in order[:remainder]:
        allocation[key] += 1

    sampled: list[dict[str, Any]] = []
    for key in sorted(groups):
        ranked = sorted(groups[key], key=stable_key)
        sampled.extend(ranked[: allocation[key]])
    return sampled


def stable_key(row: dict[str, Any]) -> str:
    value = f"{SAMPLE_SEED}|{row['sample_id']}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_readme(rows: list[dict[str, Any]]) -> str:
    batch_counts = Counter(str(row["qc_batch"]) for row in rows)
    difficulty_counts = Counter(str(row["difficulty_bucket"]) for row in rows)
    batch_text = ", ".join(f"{key}={value}" for key, value in sorted(batch_counts.items()))
    difficulty_text = ", ".join(
        f"{key}={value}" for key, value in sorted(difficulty_counts.items())
    )
    return f"""# Independent Secondary Hard-Negative QC

This package freezes 100 rows before the primary annotations are revealed.
It samples 25 rows from each of the four primary batches and proportionally
stratifies within each batch by `difficulty_bucket`. Ties and within-stratum
selection use SHA-256 ordering with seed `{SAMPLE_SEED}`.

- Batch allocation: {batch_text}
- Difficulty allocation: {difficulty_text}
- Source: `../hard_negative_remaining_400_seed.jsonl`
- Primary annotation fields were blank at package construction.

Annotator B must complete `secondary_100_viewer.html` independently and export
the result to `secondary_100_template.csv`. Do not inspect any primary batch
CSV until this export has been finalized and validated.
"""


if __name__ == "__main__":
    main()
