#!/usr/bin/env python3
"""Build a stratified, annotation-blind extension for secondary OCR-QA review."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LAUNCH_DIR = ROOT / "runs" / "problem_optimization_audit" / "open_ocr_qa_manual_annotation_launch"
DEFAULT_INPUT = LAUNCH_DIR / "primary_full_prefill.jsonl"
DEFAULT_EXISTING = LAUNCH_DIR / "secondary_calibration_prefill.jsonl"
DEFAULT_OUTPUT = (
    ROOT / "runs" / "problem_optimization_audit" / "open_ocr_qa_secondary_extension"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--existing-secondary", type=Path, default=DEFAULT_EXISTING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-task", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()

    source = read_jsonl(args.input)
    existing_ids = {row["sample_id"] for row in read_jsonl(args.existing_secondary)}
    candidates = [row for row in source if row["sample_id"] not in existing_ids]
    assert_annotation_blind(candidates)

    selected: list[dict[str, Any]] = []
    for task in sorted({str(row["task"]) for row in candidates}):
        task_rows = [row for row in candidates if row["task"] == task]
        selected.extend(stratified_select(task_rows, args.per_task, args.seed))

    selected.sort(key=lambda row: (str(row["task"]), str(row["sample_id"])))
    for index, row in enumerate(selected, start=1):
        row["double_annotation"] = "yes_extension"
        row["annotation_priority"] = "secondary_extension"
        row["secondary_extension_order"] = index
        row["secondary_extension_seed"] = args.seed

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "secondary_extension_prefill.jsonl", selected)
    write_jsonl(args.output_dir / "secondary_extension_seed.jsonl", blank_annotations(selected))
    write_csv(args.output_dir / "secondary_extension_manifest.csv", manifest_rows(selected))
    (args.output_dir / "README.md").write_text(
        readme(args.output_dir, len(selected), args.per_task, args.seed), encoding="utf-8"
    )
    print(
        f"Wrote {len(selected)} independent secondary-annotation rows "
        f"to {args.output_dir.relative_to(ROOT)}"
    )


def stratified_select(
    rows: list[dict[str, Any]], count: int, seed: int
) -> list[dict[str, Any]]:
    if len(rows) < count:
        raise ValueError(f"Requested {count} rows but only {len(rows)} are available")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["prefill_complexity"])].append(dict(row))

    selected: list[dict[str, Any]] = []
    selected_batches: Counter[str] = Counter()
    complexities = sorted(groups)
    while len(selected) < count:
        made_progress = False
        for complexity in complexities:
            if len(selected) >= count or not groups[complexity]:
                continue
            groups[complexity].sort(
                key=lambda row: (
                    selected_batches[str(row["annotation_batch"])],
                    stable_rank(seed, str(row["sample_id"])),
                )
            )
            chosen = groups[complexity].pop(0)
            selected.append(chosen)
            selected_batches[str(chosen["annotation_batch"])] += 1
            made_progress = True
        if not made_progress:
            raise RuntimeError("Unable to complete the stratified selection")
    return selected


def stable_rank(seed: int, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()


def assert_annotation_blind(rows: list[dict[str, Any]]) -> None:
    protected = (
        "manual_evidence_region_count",
        "manual_evidence_texts",
        "manual_bbox_or_region_notes",
    )
    leaked = [
        str(row.get("sample_id", ""))
        for row in rows
        if any(str(row.get(field, "")).strip() for field in protected)
    ]
    if leaked:
        raise ValueError(f"Primary annotation content is present in candidate rows: {leaked[:5]}")


def blank_annotations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": row["sample_id"],
            "task": row["task"],
            "question_id": row["question_id"],
            "evidence_units": row["prefill_evidence_units"],
            "boxes": [],
            "notes": "",
            "status": "unannotated",
        }
        for row in rows
    ]


def manifest_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "order": row["secondary_extension_order"],
            "task": row["task"],
            "sample_id": row["sample_id"],
            "question_id": row["question_id"],
            "prefill_complexity": row["prefill_complexity"],
            "annotation_batch": row["annotation_batch"],
            "stress_tag_count": row["stress_tag_count"],
            "suggested_region_count": row["prefill_suggested_region_count"],
            "image_path": row["image_path"],
        }
        for row in rows
    ]


def readme(output_dir: Path, total: int, per_task: int, seed: int) -> str:
    tool_dir = output_dir / "secondary_extension_tool"
    export_path = tool_dir / "secondary_extension_tool_annotations.json"
    return f"""# Independent Secondary Annotation Extension

This package contains {total} previously unseen rows: {per_task} TextVQA-lite and
{per_task} DocVQA-lite examples. Rows were selected deterministically (seed
`{seed}`) after excluding the original 12-row calibration set. Sampling is
balanced across task, heuristic evidence complexity, and annotation batch.

The second annotator must work independently and must not inspect the primary
annotations or adjudicated boxes before exporting this set.

## Annotate

Open `{tool_dir / 'index.html'}` in a browser. Draw the minimal set of visible
regions needed to justify the gold answer, label every box, set each completed
row to `annotated`, and use **Export JSON**. Place the export at:

`{export_path}`

## Validate

```bash
python scripts/validate_open_ocr_qa_bbox_annotations.py \\
  --annotations {export_path} \\
  --output-dir {output_dir / 'secondary_extension_validation'}
```

Agreement and adjudication should be run only after the independent export is
complete. The final analysis will combine these 20 rows with the original 12
secondary rows, yielding 32 independently annotated examples.
"""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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
