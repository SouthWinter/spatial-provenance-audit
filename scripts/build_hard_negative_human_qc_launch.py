#!/usr/bin/env python3
"""Build a human-QC launch package for TextOCR-Hard near-miss negatives.

The lexical audit is automatic and should not be described as human
validation. This package creates a deterministic review set so the hard
negative quality gap can be closed later without changing the sample.
"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "problem_optimization_audit" / "hard_negative_human_qc_launch"
DETAILS_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "hard_negative_lexical_audit"
    / "hard_negative_lexical_details.csv"
)
SUSPICIOUS_CSV = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "hard_negative_lexical_audit"
    / "hard_negative_suspicious_examples.csv"
)
PROBES_JSONL = ROOT / "data" / "textocr_val_hard_probes_500img.jsonl"

TARGET_ROWS = 100
QC_FIELDS = [
    "human_source_text_visible",
    "human_target_text_visible_same_image",
    "target_absent_after_case_punct_normalization",
    "source_bbox_matches_source_text",
    "qc_decision",
    "invalid_reason",
    "annotator_id",
    "annotator_notes",
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    details = read_csv(DETAILS_CSV)
    suspicious = {row["sample_id"]: row for row in read_csv(SUSPICIOUS_CSV)}
    probes = {
        row["sample_id"]: row
        for row in read_jsonl(PROBES_JSONL)
        if row.get("binary_polarity") == "negative"
    }

    selected = select_rows(details, suspicious)
    manifest = [build_manifest_row(row, suspicious.get(row["sample_id"], {}), probes[row["sample_id"]]) for row in selected]
    template = [{**row, **{field: "" for field in QC_FIELDS}} for row in manifest]
    summary = build_summary(manifest, details, suspicious)

    write_csv(OUT_DIR / "hard_negative_human_qc_manifest.csv", manifest)
    write_csv(OUT_DIR / "hard_negative_human_qc_template.csv", template)
    write_jsonl(OUT_DIR / "hard_negative_human_qc_seed.jsonl", template)
    write_csv(OUT_DIR / "hard_negative_human_qc_launch_summary.csv", summary)
    (OUT_DIR / "annotator_handbook.md").write_text(build_handbook(), encoding="utf-8")
    (OUT_DIR / "hard_negative_human_qc_tool.html").write_text(build_html(template), encoding="utf-8")
    (OUT_DIR / "commands.sh").write_text(build_commands(), encoding="utf-8")
    print(f"Wrote hard-negative human QC launch package to {OUT_DIR}")


def select_rows(details: list[dict[str, str]], suspicious: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    by_id = {row["sample_id"]: row for row in details}
    selected_ids: list[str] = []
    for sample_id in sorted(suspicious):
        if sample_id in by_id and sample_id not in selected_ids:
            selected_ids.append(sample_id)
    buckets = [
        ("edit_type", "single_substitution"),
        ("edit_type", "single_deletion"),
        ("edit_type", "multi_edit_or_confusable_sequence"),
        ("changed_char_class", "digit"),
        ("changed_char_class", "digit+letter"),
        ("target_shape", "alpha+digit"),
        ("target_shape", "alpha+punct"),
        ("target_shape", "alpha+non_ascii"),
        ("token_area_rank_bucket", "rank_1"),
        ("token_area_rank_bucket", "rank_2_to_5"),
        ("token_area_rank_bucket", "rank_6_to_10"),
        ("difficulty_bucket", "norm_edit_gt_0p35"),
    ]
    for field, value in buckets:
        for row in sorted(details, key=lambda item: (item.get(field, ""), item.get("sample_id", ""))):
            if len(selected_ids) >= TARGET_ROWS:
                break
            if row.get(field) == value and row["sample_id"] not in selected_ids:
                selected_ids.append(row["sample_id"])
        if len(selected_ids) >= TARGET_ROWS:
            break
    for row in sorted(details, key=lambda item: item["sample_id"]):
        if len(selected_ids) >= TARGET_ROWS:
            break
        if row["sample_id"] not in selected_ids:
            selected_ids.append(row["sample_id"])
    return [by_id[sample_id] for sample_id in selected_ids[:TARGET_ROWS]]


def build_manifest_row(detail: dict[str, str], suspicious: dict[str, str], probe: dict[str, Any]) -> dict[str, Any]:
    priority = "suspicious" if suspicious else "stratified"
    return {
        "sample_id": detail["sample_id"],
        "image_id": detail["image_id"],
        "image": probe.get("image", ""),
        "question": probe.get("question", ""),
        "source_text": detail["source_text"],
        "target_text": detail["target_text"],
        "target_answer": probe.get("target_answer", ""),
        "evidence_regions_json": json.dumps(probe.get("evidence_regions", []), ensure_ascii=True),
        "selection_priority": priority,
        "auto_suspicion_reason": suspicious.get("reason", ""),
        "edit_distance": detail["edit_distance"],
        "normalized_edit_distance": detail["normalized_edit_distance"],
        "edit_type": detail["edit_type"],
        "changed_char_class": detail["changed_char_class"],
        "target_shape": detail["target_shape"],
        "difficulty_bucket": detail["difficulty_bucket"],
        "token_area_rank": detail["token_area_rank"],
        "same_image_collision_normalizers": detail["same_image_collision_normalizers"],
        "source_target_same_normalizers": detail["source_target_same_normalizers"],
        "global_other_image_nfkc_casefold_matches": detail["global_other_image_nfkc_casefold_matches"],
        "global_other_image_alnum_matches": detail["global_other_image_alnum_matches"],
    }


def build_summary(
    manifest: list[dict[str, Any]],
    details: list[dict[str, str]],
    suspicious: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    selected_suspicious = sum(row["selection_priority"] == "suspicious" for row in manifest)
    selected_stratified = len(manifest) - selected_suspicious
    return [
        {"metric": "total_negative_probes", "value": len(details)},
        {"metric": "total_auto_suspicious_rows", "value": len(suspicious)},
        {"metric": "qc_manifest_rows", "value": len(manifest)},
        {"metric": "qc_suspicious_priority_rows", "value": selected_suspicious},
        {"metric": "qc_stratified_rows", "value": selected_stratified},
        {"metric": "image_paths_exist", "value": sum(Path(str(row["image"])).exists() for row in manifest)},
        {"metric": "human_qc_completed_rows", "value": 0},
        {"metric": "paper_claim_status", "value": "launch_ready_not_human_validated"},
    ]


def build_handbook() -> str:
    return """# Hard-Negative Human QC Handbook

Goal: verify whether TextOCR-Hard near-miss negative probes are valid negatives.

For each row, inspect the image and answer four fields:

- `human_source_text_visible`: yes/no/unclear. Is the source OCR text visible in the image?
- `human_target_text_visible_same_image`: yes/no/unclear. Is the near-miss target text itself visible anywhere in the same image?
- `target_absent_after_case_punct_normalization`: yes/no/unclear. After ignoring case and trivial punctuation/spacing, is the target still absent?
- `source_bbox_matches_source_text`: yes/no/unclear. Does the supplied evidence box correspond to the source text?

Set `qc_decision` to:

- `valid_negative` when source is visible, target is absent, and the source box is plausible.
- `invalid_target_present` when the target text appears in the same image.
- `invalid_source_not_visible` when the source text is not visible.
- `invalid_bad_box` when the evidence box does not correspond to the source text.
- `unclear` when the image is too ambiguous.

This package is not paper evidence until all QC rows are completed and the progress audit reports ready.
"""


def build_html(rows: list[dict[str, Any]]) -> str:
    body = []
    for row in rows:
        image = html.escape(str(row["image"]))
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row['sample_id']))}</td>"
            f"<td><img src=\"file://{image}\" loading=\"lazy\"></td>"
            f"<td><b>source</b>: {html.escape(str(row['source_text']))}<br>"
            f"<b>target</b>: {html.escape(str(row['target_text']))}<br>"
            f"<b>reason</b>: {html.escape(str(row['auto_suspicion_reason']))}<br>"
            f"<b>box</b>: {html.escape(str(row['evidence_regions_json']))}</td>"
            "</tr>"
        )
    return """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Hard Negative Human QC</title>
<style>
body { font-family: Arial, sans-serif; margin: 20px; }
table { border-collapse: collapse; width: 100%; }
td, th { border: 1px solid #ccc; padding: 8px; vertical-align: top; }
img { max-width: 420px; max-height: 300px; }
</style>
</head>
<body>
<h1>Hard Negative Human QC</h1>
<p>Fill the CSV template after inspecting each image.</p>
<table>
<tr><th>Sample</th><th>Image</th><th>Probe</th></tr>
""" + "\n".join(body) + """
</table>
</body>
</html>
"""


def build_commands() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail

QC_EXPORT=${QC_EXPORT:-runs/problem_optimization_audit/hard_negative_human_qc_launch/hard_negative_human_qc_template.csv}

python scripts/audit_hard_negative_human_qc_progress.py \\
  --qc-export "$QC_EXPORT" \\
  --output-dir runs/problem_optimization_audit/hard_negative_human_qc_progress
"""


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
