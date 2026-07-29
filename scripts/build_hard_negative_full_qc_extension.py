#!/usr/bin/env python3
"""Build the remaining 400-row TextOCR-Hard negative human-QC package."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from build_hard_negative_human_qc_launch import (  # noqa: E402
    DETAILS_CSV,
    PROBES_JSONL,
    QC_FIELDS,
    SUSPICIOUS_CSV,
    build_manifest_row,
    read_csv,
    read_jsonl,
    write_csv,
    write_jsonl,
)


AUDIT_ROOT = ROOT / "runs/problem_optimization_audit"
COMPLETED = AUDIT_ROOT / "hard_negative_human_qc_launch/hard_negative_human_qc_template.csv"
OUTPUT = AUDIT_ROOT / "hard_negative_full_qc_extension"


def main() -> None:
    completed_rows = read_csv(COMPLETED)
    completed_ids = {row["sample_id"] for row in completed_rows}
    if len(completed_ids) != 100:
        raise SystemExit(f"Expected exactly 100 completed QC rows, found {len(completed_ids)}")

    details = read_csv(DETAILS_CSV)
    suspicious = {row["sample_id"]: row for row in read_csv(SUSPICIOUS_CSV)}
    probes = {
        row["sample_id"]: row
        for row in read_jsonl(PROBES_JSONL)
        if row.get("binary_polarity") == "negative"
    }
    remaining = [row for row in details if row["sample_id"] not in completed_ids]
    remaining.sort(key=sort_key)
    if len(remaining) != 400:
        raise SystemExit(f"Expected 400 remaining negatives, found {len(remaining)}")

    manifest = []
    for index, detail in enumerate(remaining):
        sample_id = detail["sample_id"]
        row = build_manifest_row(detail, suspicious.get(sample_id, {}), probes[sample_id])
        row = {"qc_batch": f"batch_{index // 100 + 1:02d}", **row}
        manifest.append(row)
    template = [{**row, **{field: "" for field in QC_FIELDS}} for row in manifest]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "hard_negative_remaining_400_manifest.csv", manifest)
    write_csv(OUTPUT / "hard_negative_remaining_400_template.csv", template)
    write_jsonl(OUTPUT / "hard_negative_remaining_400_seed.jsonl", template)
    for batch in sorted({row["qc_batch"] for row in template}):
        batch_rows = [row for row in template if row["qc_batch"] == batch]
        batch_dir = OUTPUT / batch
        batch_dir.mkdir(parents=True, exist_ok=True)
        write_csv(batch_dir / f"{batch}_template.csv", batch_rows)
        (batch_dir / f"{batch}_viewer.html").write_text(build_viewer(batch_rows, batch), encoding="utf-8")

    summary = [
        {"metric": "benchmark_negative_rows", "value": len(details)},
        {"metric": "previously_human_confirmed_rows", "value": len(completed_ids)},
        {"metric": "extension_rows", "value": len(template)},
        {"metric": "extension_batches", "value": 4},
        {"metric": "image_paths_exist", "value": sum(Path(str(row["image"])).exists() for row in template)},
        {"metric": "overlap_with_completed_rows", "value": sum(row["sample_id"] in completed_ids for row in template)},
        {"metric": "paper_claim_status", "value": "remaining_400_ready_for_human_qc"},
    ]
    write_csv(OUTPUT / "hard_negative_remaining_400_summary.csv", summary)
    (OUTPUT / "README.md").write_text(build_readme(), encoding="utf-8")
    print(f"Wrote 400-row hard-negative QC extension to {OUTPUT}")


def sort_key(row: dict[str, str]) -> tuple[Any, ...]:
    difficulty_order = {
        "norm_edit_le_0p10": 0,
        "norm_edit_gt_0p35": 1,
        "norm_edit_0p10_to_0p20": 2,
        "norm_edit_0p20_to_0p35": 3,
    }
    return (
        difficulty_order.get(row.get("difficulty_bucket", ""), 9),
        -int(float(row.get("token_area_rank", "0") or 0)),
        row["sample_id"],
    )


def build_viewer(rows: list[dict[str, Any]], batch: str) -> str:
    payload = json.dumps(rows, ensure_ascii=True).replace("</", "<\\/")
    document = """<!doctype html>
<html><head><meta charset="utf-8"><title>__BATCH__ hard-negative QC</title>
<style>
body { font-family: Arial, sans-serif; max-width: 1180px; margin: 0 auto; padding: 16px; color: #171717; }
header { position: sticky; top: 0; z-index: 10; background: #fff; border-bottom: 2px solid #222; padding: 8px 0 12px; }
.toolbar, .actions, .nav { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.toolbar { justify-content: space-between; }
button, select, input, textarea { font: inherit; }
button { padding: 7px 11px; cursor: pointer; border: 1px solid #777; background: #f5f5f5; border-radius: 4px; }
button:hover { background: #e8e8e8; }
button.primary { background: #e8f4e8; border-color: #397239; }
button.danger { background: #fff0f0; border-color: #a44; }
button:disabled { opacity: .45; cursor: default; }
main { padding: 18px 0 40px; }
.meta { display: grid; grid-template-columns: 1fr 1fr; gap: 8px 20px; margin: 12px 0; }
.meta p { margin: 0; }
.image-wrap { position: relative; display: inline-block; line-height: 0; border: 1px solid #999; }
.image-wrap img { display: block; max-width: min(100%, 1040px); max-height: 610px; }
.bbox { position: absolute; border: 3px solid #e1261c; box-sizing: border-box; pointer-events: none; }
.question { font-size: 18px; margin: 12px 0; }
.fields { display: grid; grid-template-columns: repeat(2, minmax(260px, 1fr)); gap: 12px 22px; margin: 14px 0; }
.field { display: grid; gap: 5px; }
.field select, .field input, .field textarea { padding: 7px; border: 1px solid #999; border-radius: 3px; }
.field textarea { min-height: 54px; }
.wide { grid-column: 1 / -1; }
.status { font-weight: bold; }
.complete { color: #176b24; }
.incomplete { color: #9b3d00; }
code { overflow-wrap: anywhere; }
@media (max-width: 720px) {
  .meta, .fields { grid-template-columns: 1fr; }
  .wide { grid-column: auto; }
}
</style></head>
<body>
<header>
  <div class="toolbar">
    <div>
      <strong>__BATCH__: hard-negative human QC</strong>
      <span id="progress" class="status incomplete"></span>
    </div>
    <div class="nav">
      <label>Annotator <input id="annotator" size="16" placeholder="name or ID"></label>
      <button id="prev">Previous</button>
      <span id="position"></span>
      <button id="next">Next</button>
      <button id="export" class="primary">Export CSV</button>
    </div>
  </div>
</header>
<main id="app"></main>
<script>
const rows = __ROWS__;
const batch = "__BATCH__";
const qcFields = [
  "human_source_text_visible",
  "human_target_text_visible_same_image",
  "target_absent_after_case_punct_normalization",
  "source_bbox_matches_source_text",
  "qc_decision",
  "invalid_reason",
  "annotator_id",
  "annotator_notes"
];
const storageKey = `spatial-provenance-hard-negative-qc-${batch}`;
let state = {};
let current = 0;

try { state = JSON.parse(localStorage.getItem(storageKey) || "{}"); } catch (_) { state = {}; }
for (const row of rows) {
  if (!state[row.sample_id]) {
    state[row.sample_id] = Object.fromEntries(qcFields.map(field => [field, row[field] || ""]));
  }
}

const esc = value => String(value ?? "")
  .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;").replaceAll('"', "&quot;");

function save() {
  try { localStorage.setItem(storageKey, JSON.stringify(state)); } catch (_) {}
  updateProgress();
}

function setFields(sampleId, values) {
  Object.assign(state[sampleId], values);
  if (document.getElementById("annotator").value.trim()) {
    state[sampleId].annotator_id = document.getElementById("annotator").value.trim();
  }
  save();
  render();
}

function quickDecision(kind) {
  const id = rows[current].sample_id;
  const presets = {
    valid_negative: {
      human_source_text_visible: "yes",
      human_target_text_visible_same_image: "no",
      target_absent_after_case_punct_normalization: "yes",
      source_bbox_matches_source_text: "yes",
      qc_decision: "valid_negative",
      invalid_reason: ""
    },
    invalid_target_present: {
      human_source_text_visible: "yes",
      human_target_text_visible_same_image: "yes",
      target_absent_after_case_punct_normalization: "no",
      qc_decision: "invalid_target_present",
      invalid_reason: "target text is present in the same image"
    },
    invalid_source_not_visible: {
      human_source_text_visible: "no",
      human_target_text_visible_same_image: "unclear",
      target_absent_after_case_punct_normalization: "unclear",
      source_bbox_matches_source_text: "no",
      qc_decision: "invalid_source_not_visible",
      invalid_reason: "source text is not visually identifiable"
    },
    invalid_bad_box: {
      human_source_text_visible: "yes",
      source_bbox_matches_source_text: "no",
      qc_decision: "invalid_bad_box",
      invalid_reason: "evidence box does not match source text"
    },
    unclear: {
      human_source_text_visible: "unclear",
      human_target_text_visible_same_image: "unclear",
      target_absent_after_case_punct_normalization: "unclear",
      source_bbox_matches_source_text: "unclear",
      qc_decision: "unclear",
      invalid_reason: "image evidence is ambiguous"
    }
  };
  setFields(id, presets[kind]);
}

function optionList(values, selected) {
  return values.map(value =>
    `<option value="${esc(value)}"${value === selected ? " selected" : ""}>${esc(value || "-- select --")}</option>`
  ).join("");
}

function fieldSelect(label, field, values, value) {
  return `<label class="field"><strong>${esc(label)}</strong>
    <select data-field="${field}">${optionList(["", ...values], value)}</select></label>`;
}

function render() {
  const row = rows[current];
  const answer = state[row.sample_id];
  let boxes = [];
  try { boxes = JSON.parse(row.evidence_regions_json || "[]"); } catch (_) {}
  const boxHtml = boxes.map(box => {
    const [x1, y1, x2, y2] = box.map(Number);
    return `<span class="bbox" style="left:${x1 * 100}%;top:${y1 * 100}%;width:${(x2 - x1) * 100}%;height:${(y2 - y1) * 100}%"></span>`;
  }).join("");
  document.getElementById("app").innerHTML = `
    <div class="meta">
      <p><strong>Sample:</strong> <code>${esc(row.sample_id)}</code></p>
      <p><strong>Priority:</strong> ${esc(row.selection_priority)} ${esc(row.auto_suspicion_reason)}</p>
      <p><strong>Source:</strong> ${esc(row.source_text)}</p>
      <p><strong>Near-miss target:</strong> ${esc(row.target_text)}</p>
    </div>
    <div class="image-wrap"><img src="file://${esc(row.image)}">${boxHtml}</div>
    <p class="question"><strong>Question:</strong> ${esc(row.question)}</p>
    <div class="actions">
      <button class="primary" data-quick="valid_negative">Valid negative</button>
      <button class="danger" data-quick="invalid_target_present">Target present</button>
      <button class="danger" data-quick="invalid_source_not_visible">Source invisible</button>
      <button class="danger" data-quick="invalid_bad_box">Bad box</button>
      <button data-quick="unclear">Unclear</button>
    </div>
    <div class="fields">
      ${fieldSelect("Source text visible", "human_source_text_visible", ["yes","no","unclear"], answer.human_source_text_visible)}
      ${fieldSelect("Target visible anywhere", "human_target_text_visible_same_image", ["yes","no","unclear"], answer.human_target_text_visible_same_image)}
      ${fieldSelect("Target absent after normalization", "target_absent_after_case_punct_normalization", ["yes","no","unclear"], answer.target_absent_after_case_punct_normalization)}
      ${fieldSelect("Red box matches source", "source_bbox_matches_source_text", ["yes","no","unclear"], answer.source_bbox_matches_source_text)}
      ${fieldSelect("QC decision", "qc_decision", ["valid_negative","invalid_target_present","invalid_source_not_visible","invalid_bad_box","unclear"], answer.qc_decision)}
      <label class="field"><strong>Invalid reason</strong><input data-field="invalid_reason" value="${esc(answer.invalid_reason)}"></label>
      <label class="field"><strong>Annotator ID</strong><input data-field="annotator_id" value="${esc(answer.annotator_id)}"></label>
      <label class="field wide"><strong>Notes</strong><textarea data-field="annotator_notes">${esc(answer.annotator_notes)}</textarea></label>
    </div>`;
  document.getElementById("position").textContent = `${current + 1} / ${rows.length}`;
  document.getElementById("prev").disabled = current === 0;
  document.getElementById("next").disabled = current === rows.length - 1;
  document.querySelectorAll("[data-quick]").forEach(button => {
    button.addEventListener("click", () => quickDecision(button.dataset.quick));
  });
  document.querySelectorAll("[data-field]").forEach(input => {
    input.addEventListener("change", event => {
      state[row.sample_id][event.target.dataset.field] = event.target.value;
      save();
    });
  });
  updateProgress();
}

function updateProgress() {
  const done = rows.filter(row => state[row.sample_id].qc_decision).length;
  const progress = document.getElementById("progress");
  progress.textContent = ` (${done}/${rows.length} decided)`;
  progress.className = `status ${done === rows.length ? "complete" : "incomplete"}`;
}

function csvCell(value) {
  const text = String(value ?? "");
  return /[",\\n\\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function exportCsv() {
  const fields = Object.keys(rows[0]);
  const completed = rows.map(row => ({...row, ...state[row.sample_id]}));
  const csv = [fields.join(","), ...completed.map(row => fields.map(field => csvCell(row[field])).join(","))].join("\\r\\n");
  const blob = new Blob([csv], {type: "text/csv;charset=utf-8"});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `${batch}_template.csv`;
  link.click();
  URL.revokeObjectURL(link.href);
}

document.getElementById("annotator").addEventListener("change", event => {
  localStorage.setItem(`${storageKey}-annotator`, event.target.value.trim());
});
document.getElementById("annotator").value = localStorage.getItem(`${storageKey}-annotator`) || "";
document.getElementById("prev").addEventListener("click", () => { current--; render(); window.scrollTo(0, 0); });
document.getElementById("next").addEventListener("click", () => { current++; render(); window.scrollTo(0, 0); });
document.getElementById("export").addEventListener("click", exportCsv);
document.addEventListener("keydown", event => {
  if (event.target.matches("input, textarea, select")) return;
  if (event.key === "ArrowLeft" && current > 0) { current--; render(); }
  if (event.key === "ArrowRight" && current < rows.length - 1) { current++; render(); }
  if (event.key.toLowerCase() === "v") quickDecision("valid_negative");
  if (event.key.toLowerCase() === "u") quickDecision("unclear");
});
render();
</script></body></html>"""
    return document.replace("__BATCH__", batch).replace("__ROWS__", payload)


def build_readme() -> str:
    return """# Remaining Development TextOCR-Hard Negative Human QC

This package contains the 400 development negative probes not included in the
completed 100-row stratified/suspicious audit. It is split into four
deterministic 100-row batches. There is no overlap with the completed sample
or with the distinct locked confirmation split.

For each batch, open its HTML viewer. The red rectangle shows the supplied
source-text box. Use the quick-decision buttons or edit the individual fields;
progress is saved in the browser's local storage. Press `Export CSV` when the
batch reaches 100/100 and place the exported file at the corresponding
`batch_XX/batch_XX_template.csv` path. The `V` key marks a valid negative,
`U` marks an unclear case, and the arrow keys change rows.

Do not decide from OCR metadata alone: visually search the full image for the
near-miss target. Use `unclear` when resolution is insufficient. Export
periodically as a backup because browser-local storage is not an audit record.

After all four batches are complete, concatenate them without duplicating the
header and run:

```bash
QC_DIR=runs/problem_optimization_audit/hard_negative_full_qc_extension
head -n 1 "$QC_DIR/batch_01/batch_01_template.csv" > "$QC_DIR/hard_negative_remaining_400_template.csv"
for csv in "$QC_DIR"/batch_*/batch_*_template.csv; do tail -n +2 "$csv" >> "$QC_DIR/hard_negative_remaining_400_template.csv"; done
python scripts/audit_hard_negative_human_qc_progress.py --qc-export "$QC_DIR/hard_negative_remaining_400_template.csv" --output-dir "$QC_DIR/progress"
```

Exhaustive human-QC claims are valid only for the development split after this
audit reports 400/400 ready and the decisions have been reviewed. Locked
confirmation requires its separate 500-row package.
"""


if __name__ == "__main__":
    main()
