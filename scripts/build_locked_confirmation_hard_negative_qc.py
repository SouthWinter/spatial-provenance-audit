#!/usr/bin/env python3
"""Build the exhaustive 500-row locked-confirmation hard-negative QC package."""

from __future__ import annotations

import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from build_hard_negative_full_qc_extension import build_viewer, sort_key  # noqa: E402
from build_hard_negative_human_qc_launch import (  # noqa: E402
    QC_FIELDS,
    build_manifest_row,
    read_csv,
    read_jsonl,
    write_csv,
    write_jsonl,
)


DETAILS = ROOT / "runs/textocr_confirmation/lexical_audit/hard_negative_lexical_details.csv"
SUSPICIOUS = (
    ROOT / "runs/textocr_confirmation/lexical_audit/hard_negative_suspicious_examples.csv"
)
PROBES = ROOT / "data/textocr_val_hard_confirmation_500img_seed20260720.jsonl"
DEVELOPMENT_QC = (
    ROOT
    / "runs/problem_optimization_audit/hard_negative_full_qc_extension/integrated_audit/"
    "hard_negative_all500_adjudicated.csv"
)
OUTPUT = ROOT / "runs/problem_optimization_audit/hard_negative_confirmation_full_qc"


def main() -> None:
    details = read_csv(DETAILS)
    suspicious = {row["sample_id"]: row for row in read_csv(SUSPICIOUS)}
    probes = {
        row["sample_id"]: row
        for row in read_jsonl(PROBES)
        if row.get("binary_polarity") == "negative"
    }
    if len(details) != 500 or len(probes) != 500:
        raise RuntimeError(f"Expected 500 details/probes, found {len(details)}/{len(probes)}.")

    details.sort(key=lambda row: (0 if row["sample_id"] in suspicious else 1, sort_key(row)))
    manifest = []
    for index, detail in enumerate(details):
        sample_id = detail["sample_id"]
        row = build_manifest_row(detail, suspicious.get(sample_id, {}), probes[sample_id])
        manifest.append({"qc_batch": f"batch_{index // 100 + 1:02d}", **row})
    template = [{**row, **{field: "" for field in QC_FIELDS}} for row in manifest]

    development_ids = development_qc_ids()
    overlap = development_ids & {row["sample_id"] for row in template}
    if overlap:
        raise RuntimeError(f"Locked confirmation overlaps development QC: {sorted(overlap)[:5]}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "locked_confirmation_500_manifest.csv", manifest)
    write_csv(OUTPUT / "locked_confirmation_500_template.csv", template)
    write_jsonl(OUTPUT / "locked_confirmation_500_seed.jsonl", template)
    for batch in sorted({row["qc_batch"] for row in template}):
        rows = [row for row in template if row["qc_batch"] == batch]
        batch_dir = OUTPUT / batch
        batch_dir.mkdir(parents=True, exist_ok=True)
        write_csv(batch_dir / f"{batch}_template.csv", rows)
        (batch_dir / f"{batch}_viewer.html").write_text(
            build_viewer(rows, f"locked_confirmation_{batch}"), encoding="utf-8"
        )

    summary = [
        {"metric": "locked_confirmation_rows", "value": len(template)},
        {"metric": "batches", "value": 5},
        {
            "metric": "image_paths_exist",
            "value": sum(Path(str(row["image"])).exists() for row in template),
        },
        {"metric": "overlap_with_development_qc", "value": len(overlap)},
        {"metric": "human_qc_completed_rows", "value": 0},
        {"metric": "paper_claim_status", "value": "locked_confirmation_qc_launch_ready"},
    ]
    write_csv(OUTPUT / "locked_confirmation_500_summary.csv", summary)
    (OUTPUT / "README.md").write_text(readme(), encoding="utf-8")
    print(f"Wrote locked-confirmation hard-negative QC package to {OUTPUT}")


def development_qc_ids() -> set[str]:
    if not DEVELOPMENT_QC.exists():
        return set()
    with DEVELOPMENT_QC.open("r", encoding="utf-8", newline="") as handle:
        return {row["sample_id"] for row in csv.DictReader(handle)}


def readme() -> str:
    return """# Locked-Confirmation Hard-Negative Human QC

This package contains all 500 hard negatives from the locked, image-disjoint
confirmation split. It is distinct from the completed 500-row development QC.

Open each batch viewer, visually inspect the entire image, and export the CSV
after reaching 100/100. Place each export at its corresponding
`batch_XX/batch_XX_template.csv` path. Browser-local progress is convenient but
is not the final audit record, so export periodically.

After all five batches are complete:

```bash
QC_DIR=runs/problem_optimization_audit/hard_negative_confirmation_full_qc
head -n 1 "$QC_DIR/batch_01/batch_01_template.csv" > "$QC_DIR/locked_confirmation_500_template.csv"
for csv in "$QC_DIR"/batch_*/batch_*_template.csv; do
  tail -n +2 "$csv" >> "$QC_DIR/locked_confirmation_500_template.csv"
done
python scripts/audit_hard_negative_human_qc_progress.py \
  --qc-export "$QC_DIR/locked_confirmation_500_template.csv" \
  --output-dir "$QC_DIR/progress"
```

Do not describe locked confirmation as exhaustively human-validated until the
progress audit reports 500/500 ready and disagreements from any secondary
review have been adjudicated.
"""


if __name__ == "__main__":
    main()
