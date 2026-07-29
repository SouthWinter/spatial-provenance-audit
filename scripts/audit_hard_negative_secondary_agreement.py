#!/usr/bin/env python3
"""Measure independent hard-negative QC agreement and build adjudication queue."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
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
    validate(primary, secondary, args.primary_batches * 100)

    primary_by_id = {row["sample_id"]: row for row in primary}
    agreement_rows = []
    queue_rows = []
    pairs_by_field: dict[str, list[tuple[str, str]]] = {
        field: [] for field in CORE_FIELDS
    }
    binary_pairs: list[tuple[str, str]] = []

    for b_row in secondary:
        a_row = primary_by_id[b_row["sample_id"]]
        row = {
            "sample_id": b_row["sample_id"],
            "qc_batch": b_row["qc_batch"],
            "difficulty_bucket": b_row["difficulty_bucket"],
        }
        for field in CORE_FIELDS:
            a_value = a_row[field].strip()
            b_value = b_row[field].strip()
            row[f"a_{field}"] = a_value
            row[f"b_{field}"] = b_value
            row[f"match_{field}"] = str(int(a_value == b_value))
            pairs_by_field[field].append((a_value, b_value))
        a_binary = binary_label(a_row["qc_decision"])
        b_binary = binary_label(b_row["qc_decision"])
        binary_pairs.append((a_binary, b_binary))
        row["a_validity_binary"] = a_binary
        row["b_validity_binary"] = b_binary
        row["match_validity_binary"] = str(int(a_binary == b_binary))
        row["all_core_fields_match"] = str(
            int(all(a_row[field].strip() == b_row[field].strip() for field in CORE_FIELDS))
        )
        agreement_rows.append(row)

        if row["all_core_fields_match"] == "0":
            queue = dict(b_row)
            for field in QC_FIELDS:
                queue[f"a_{field}"] = a_row.get(field, "")
                queue[f"b_{field}"] = b_row.get(field, "")
            for field in QC_FIELDS:
                queue[f"final_{field}"] = ""
            queue["adjudicator_id"] = ""
            queue["adjudication_notes"] = ""
            queue["adjudication_status"] = "needs_adjudication"
            queue_rows.append(queue)

    summary = []
    for field, pairs in pairs_by_field.items():
        summary.extend(metric_rows(field, pairs))
    summary.extend(metric_rows("validity_binary", binary_pairs))
    exact = sum(row["all_core_fields_match"] == "1" for row in agreement_rows)
    low, high = wilson_interval(exact, len(agreement_rows))
    summary.extend(
        [
            {"field": "all_core_fields", "metric": "n", "value": len(agreement_rows)},
            {
                "field": "all_core_fields",
                "metric": "exact_agreement",
                "value": fmt(exact / len(agreement_rows)),
            },
            {
                "field": "all_core_fields",
                "metric": "agreement_ci95_low",
                "value": fmt(low),
            },
            {
                "field": "all_core_fields",
                "metric": "agreement_ci95_high",
                "value": fmt(high),
            },
            {
                "field": "all_core_fields",
                "metric": "disagreement_rows",
                "value": len(queue_rows),
            },
        ]
    )

    write_csv(secondary_dir / "pre_adjudication_agreement_rows.csv", agreement_rows)
    write_csv(secondary_dir / "pre_adjudication_agreement_summary.csv", summary)
    write_csv(secondary_dir / "disagreement_queue.csv", queue_rows)
    (secondary_dir / "disagreement_viewer.html").write_text(
        build_adjudication_viewer(queue_rows),
        encoding="utf-8",
    )
    (secondary_dir / "pre_adjudication_agreement_report.md").write_text(
        build_report(summary, len(queue_rows)),
        encoding="utf-8",
    )
    print(
        f"Compared {len(agreement_rows)} rows; "
        f"core-field disagreements={len(queue_rows)}"
    )


def validate(
    primary: list[dict[str, str]],
    secondary: list[dict[str, str]],
    expected_primary: int,
) -> None:
    if len(primary) != expected_primary:
        raise SystemExit(
            f"Expected {expected_primary} primary rows, found {len(primary)}"
        )
    if len(secondary) != 100:
        raise SystemExit(f"Expected 100 secondary rows, found {len(secondary)}")
    primary_ids = [row["sample_id"] for row in primary]
    secondary_ids = [row["sample_id"] for row in secondary]
    if len(set(primary_ids)) != expected_primary:
        raise SystemExit("Primary sample IDs are not unique")
    if len(set(secondary_ids)) != 100:
        raise SystemExit("Secondary sample IDs are not unique")
    if not set(secondary_ids).issubset(primary_ids):
        raise SystemExit("Secondary sample is not a subset of primary rows")
    for name, rows, annotator in (
        ("primary", primary, "A"),
        ("secondary", secondary, "B"),
    ):
        for row in rows:
            missing = [field for field in CORE_FIELDS if not row.get(field, "").strip()]
            if missing:
                raise SystemExit(
                    f"{name} row {row.get('sample_id')} missing {','.join(missing)}"
                )
            if row.get("annotator_id", "").strip() != annotator:
                raise SystemExit(
                    f"{name} row {row.get('sample_id')} has annotator_id="
                    f"{row.get('annotator_id')!r}, expected {annotator!r}"
                )


def binary_label(decision: str) -> str:
    return "valid_negative" if decision.strip() == "valid_negative" else "non_valid"


def metric_rows(field: str, pairs: list[tuple[str, str]]) -> list[dict[str, Any]]:
    observed = sum(a == b for a, b in pairs) / len(pairs)
    low, high = wilson_interval(sum(a == b for a, b in pairs), len(pairs))
    kappa = cohen_kappa(pairs)
    kappa_low, kappa_high = bootstrap_kappa_ci(pairs)
    return [
        {"field": field, "metric": "n", "value": len(pairs)},
        {"field": field, "metric": "agreement", "value": fmt(observed)},
        {"field": field, "metric": "agreement_ci95_low", "value": fmt(low)},
        {"field": field, "metric": "agreement_ci95_high", "value": fmt(high)},
        {"field": field, "metric": "cohen_kappa", "value": fmt(kappa)},
        {"field": field, "metric": "kappa_ci95_low", "value": fmt(kappa_low)},
        {"field": field, "metric": "kappa_ci95_high", "value": fmt(kappa_high)},
    ]


def cohen_kappa(pairs: list[tuple[str, str]]) -> float:
    if not pairs:
        return float("nan")
    labels = sorted({value for pair in pairs for value in pair})
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    a_counts = Counter(a for a, _ in pairs)
    b_counts = Counter(b for _, b in pairs)
    expected = sum((a_counts[label] / n) * (b_counts[label] / n) for label in labels)
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else 0.0
    return (observed - expected) / (1.0 - expected)


def bootstrap_kappa_ci(
    pairs: list[tuple[str, str]], draws: int = 5000
) -> tuple[float, float]:
    rng = random.Random("hard-negative-agreement-v1")
    values = []
    for _ in range(draws):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        values.append(cohen_kappa(sample))
    values.sort()
    return percentile(values, 0.025), percentile(values, 0.975)


def percentile(values: list[float], quantile: float) -> float:
    position = (len(values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - position) + values[upper] * (position - lower)


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    estimate = successes / total
    denominator = 1 + z * z / total
    center = (estimate + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(estimate * (1 - estimate) / total + z * z / (4 * total * total))
        / denominator
    )
    return center - margin, center + margin


def build_adjudication_viewer(rows: list[dict[str, str]]) -> str:
    payload = json.dumps(rows, ensure_ascii=True).replace("</", "<\\/")
    fields = json.dumps(list(QC_FIELDS))
    return """<!doctype html>
<html><head><meta charset="utf-8"><title>Hard-negative disagreement adjudication</title>
<style>
body{font-family:Arial,sans-serif;max-width:1180px;margin:auto;padding:16px;color:#171717}
header{position:sticky;top:0;background:#fff;border-bottom:2px solid #222;padding:10px 0;z-index:5}
button,select,input,textarea{font:inherit;padding:7px}.nav,.actions{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.meta,.compare,.fields{display:grid;grid-template-columns:1fr 1fr;gap:8px 18px;margin:14px 0}
.image-wrap{position:relative;display:inline-block;line-height:0;border:1px solid #999}
.image-wrap img{display:block;max-width:min(100%,1040px);max-height:600px}
.bbox{position:absolute;border:3px solid #e1261c;box-sizing:border-box}
.compare div{border:1px solid #aaa;padding:8px}.fields label{display:grid;gap:4px}
.wide{grid-column:1/-1}code{overflow-wrap:anywhere}
</style></head><body>
<header><div class="nav"><strong>Disagreement adjudication</strong><span id="progress"></span>
<button id="prev">Previous</button><span id="position"></span><button id="next">Next</button>
<button id="export">Export CSV</button></div></header><main id="app"></main>
<script>
const rows=__ROWS__; const qcFields=__FIELDS__; const key="hard-negative-adjudication-v1";
let state={}; let current=0;
try{state=JSON.parse(localStorage.getItem(key)||"{}")}catch(_){state={}}
for(const row of rows){if(!state[row.sample_id]){state[row.sample_id]=Object.fromEntries(
  [...qcFields.map(f=>[`final_${f}`,row[`final_${f}`]||""]),["adjudicator_id",row.adjudicator_id||""],
   ["adjudication_notes",row.adjudication_notes||""],["adjudication_status",row.adjudication_status||"needs_adjudication"]])}}
const esc=v=>String(v??"").replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;").replaceAll('"',"&quot;");
function save(){localStorage.setItem(key,JSON.stringify(state));progress()}
function adopt(side){const row=rows[current],s=state[row.sample_id];for(const f of qcFields)s[`final_${f}`]=row[`${side}_${f}`]||"";
s.adjudicator_id="B";s.adjudication_status="adjudicated";s.adjudication_notes=`Adopted ${side.toUpperCase()} after image review`;save();render()}
function opts(values,selected){return values.map(v=>`<option value="${esc(v)}"${v===selected?" selected":""}>${esc(v||"-- select --")}</option>`).join("")}
function render(){const row=rows[current],s=state[row.sample_id];let boxes=[];try{boxes=JSON.parse(row.evidence_regions_json||"[]")}catch(_){}
const box=boxes.map(b=>`<span class="bbox" style="left:${b[0]*100}%;top:${b[1]*100}%;width:${(b[2]-b[0])*100}%;height:${(b[3]-b[1])*100}%"></span>`).join("");
const cmp=qcFields.map(f=>`<div><b>${esc(f)}</b><br>A: ${esc(row[`a_${f}`])}<br>B: ${esc(row[`b_${f}`])}</div>`).join("");
const ynu=["","yes","no","unclear"],dec=["","valid_negative","invalid_target_present","invalid_source_not_visible","invalid_bad_box","unclear"];
document.getElementById("app").innerHTML=`<div class="meta"><p><b>Sample:</b> <code>${esc(row.sample_id)}</code></p>
<p><b>Batch:</b> ${esc(row.qc_batch)}</p><p><b>Source:</b> ${esc(row.source_text)}</p><p><b>Target:</b> ${esc(row.target_text)}</p></div>
<div class="image-wrap"><img src="file://${esc(row.image)}">${box}</div><div class="compare">${cmp}</div>
<div class="actions"><button data-adopt="a">Adopt A</button><button data-adopt="b">Adopt B</button></div>
<div class="fields">${qcFields.slice(0,4).map(f=>`<label><b>${esc(f)}</b><select data-field="final_${f}">${opts(ynu,s[`final_${f}`])}</select></label>`).join("")}
<label><b>qc_decision</b><select data-field="final_qc_decision">${opts(dec,s.final_qc_decision)}</select></label>
<label><b>invalid_reason</b><input data-field="final_invalid_reason" value="${esc(s.final_invalid_reason)}"></label>
<label><b>annotator_id</b><input data-field="final_annotator_id" value="${esc(s.final_annotator_id)}"></label>
<label class="wide"><b>annotator_notes</b><textarea data-field="final_annotator_notes">${esc(s.final_annotator_notes)}</textarea></label>
<label><b>adjudicator_id</b><input data-field="adjudicator_id" value="${esc(s.adjudicator_id)}"></label>
<label><b>status</b><select data-field="adjudication_status">${opts(["needs_adjudication","adjudicated"],s.adjudication_status)}</select></label>
<label class="wide"><b>adjudication_notes</b><textarea data-field="adjudication_notes">${esc(s.adjudication_notes)}</textarea></label></div>`;
document.querySelectorAll("[data-adopt]").forEach(x=>x.onclick=()=>adopt(x.dataset.adopt));
document.querySelectorAll("[data-field]").forEach(x=>x.onchange=e=>{s[e.target.dataset.field]=e.target.value;save()});
document.getElementById("position").textContent=`${current+1}/${rows.length}`;document.getElementById("prev").disabled=current===0;
document.getElementById("next").disabled=current===rows.length-1;progress()}
function progress(){const n=rows.filter(r=>state[r.sample_id].adjudication_status==="adjudicated").length;
document.getElementById("progress").textContent=`${n}/${rows.length} adjudicated`}
function cell(v){const t=String(v??"");return /[",\\n\\r]/.test(t)?`"${t.replaceAll('"','""')}"`:t}
function exportCsv(){if(!rows.length)return;const fields=Object.keys(rows[0]);const out=rows.map(r=>({...r,...state[r.sample_id]}));
const csv=[fields.join(","),...out.map(r=>fields.map(f=>cell(r[f])).join(","))].join("\\r\\n");const blob=new Blob([csv],{type:"text/csv"});
const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="disagreement_queue.csv";a.click();URL.revokeObjectURL(a.href)}
document.getElementById("prev").onclick=()=>{current--;render()};document.getElementById("next").onclick=()=>{current++;render()};
document.getElementById("export").onclick=exportCsv;render();
</script></body></html>""".replace("__ROWS__", payload).replace("__FIELDS__", fields)


def build_report(summary: list[dict[str, Any]], disagreements: int) -> str:
    lookup = {(row["field"], row["metric"]): row["value"] for row in summary}
    lines = [
        "# Hard-Negative Independent Reannotation Agreement",
        "",
        "Agreement is measured on the frozen 100-row secondary sample before adjudication.",
        "",
        f"- QC-decision agreement: {lookup[('qc_decision', 'agreement')]}",
        f"- QC-decision Cohen's kappa: {lookup[('qc_decision', 'cohen_kappa')]}",
        f"- Valid-vs-non-valid agreement: {lookup[('validity_binary', 'agreement')]}",
        f"- Valid-vs-non-valid Cohen's kappa: {lookup[('validity_binary', 'cohen_kappa')]}",
        f"- Exact agreement on all five core fields: {lookup[('all_core_fields', 'exact_agreement')]}",
        f"- Rows requiring adjudication: {disagreements}",
        "",
        "## Full Metrics",
        "",
        table_md(summary),
    ]
    return "\n".join(lines) + "\n"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
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


def fmt(value: float) -> str:
    return f"{value:.3f}"


if __name__ == "__main__":
    main()
