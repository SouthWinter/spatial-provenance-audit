#!/usr/bin/env python3
"""Build a browser tool for the 20-row secondary-extension adjudication."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "runs/problem_optimization_audit/open_ocr_qa_secondary_extension/extension_20_adjudication"
QUEUE = PACKAGE / "adjudication_queue.jsonl"
OUTPUT = PACKAGE / "adjudication_tool"


def main() -> None:
    rows = [json.loads(line) for line in QUEUE.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 17:
        raise SystemExit(f"Expected 17 adjudication rows, found {len(rows)}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "index.html").write_text(build_html(rows), encoding="utf-8")
    (OUTPUT / "README.md").write_text(build_readme(), encoding="utf-8")
    print(f"Wrote extension adjudication tool to {OUTPUT / 'index.html'}")


def build_html(rows: list[dict]) -> str:
    data = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    return """<!doctype html>
<html><head><meta charset="utf-8"><title>20-row evidence adjudication</title>
<style>
body { font-family: Arial, sans-serif; margin: 0; color: #171717; background: #f5f5f5; }
header { position: sticky; top: 0; z-index: 3; background: #fff; border-bottom: 1px solid #bbb; padding: 10px 18px; display:flex; gap:16px; align-items:center; }
button { padding: 7px 12px; cursor: pointer; }
#app { max-width: 1240px; margin: 16px auto; }
.row { background:#fff; border:1px solid #bbb; margin:0 0 18px; padding:14px; }
.meta { display:grid; grid-template-columns: 1fr 1fr; gap:8px 20px; }
.stage { position:relative; display:inline-block; margin-top:10px; max-width:100%; }
.stage img { display:block; max-width:1160px; max-height:760px; width:auto; height:auto; }
.box { position:absolute; box-sizing:border-box; border:2px solid; pointer-events:none; }
.box span { position:absolute; left:0; top:0; transform:translateY(-100%); color:#fff; font-size:11px; padding:1px 3px; white-space:nowrap; max-width:300px; overflow:hidden; }
.a { border-color:#e24a33; } .a span { background:#e24a33; }
.b { border-color:#2878b5; border-style:dashed; } .b span { background:#2878b5; }
.controls { margin-top:10px; display:flex; flex-wrap:wrap; gap:12px; align-items:center; }
textarea { width:100%; min-height:42px; margin-top:8px; }
.legend { color:#555; } .status { font-weight:bold; }
</style></head><body>
<header><strong>Secondary-extension adjudication</strong><span id="progress"></span><button id="export">Export decisions JSON</button></header>
<main id="app"></main>
<script>
const rows = DATA_PLACEHOLDER;
const state = JSON.parse(localStorage.getItem('spatial-provenance-extension-adjudication') || '{}');
const app = document.getElementById('app');
function esc(s) { return String(s ?? '').replace(/[&<>\"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function render() {
  app.innerHTML = '';
  rows.forEach((row, idx) => {
    const saved = state[row.sample_id] || {decision:'', notes:''};
    const section = document.createElement('section'); section.className='row';
    section.innerHTML = `<h2>${idx+1}/17 ${esc(row.sample_id)}</h2>
      <div class="meta"><div><b>Question:</b> ${esc(row.question)}</div><div><b>Gold:</b> ${esc(row.gold_answers)}</div>
      <div><b>Primary:</b> ${row.agreement.box_count_a} boxes, ${esc(row.agreement.label_types_a)}</div>
      <div><b>Secondary:</b> ${row.agreement.box_count_b} boxes, ${esc(row.agreement.label_types_b)}</div></div>
      <div class="legend">solid red = primary; dashed blue = independent secondary. Overlapping rectangles may be visually coincident.</div>
      <div class="stage"><img src="file://${esc(row.image_path)}"></div>
      <div class="controls"><b>Final rule:</b>${['primary','secondary','union','needs_redraw'].map(v => `<label><input type="radio" name="d${idx}" value="${v}" ${saved.decision===v?'checked':''}> ${v}</label>`).join('')}</div>
      <textarea placeholder="Required adjudication rationale">${esc(saved.notes)}</textarea>`;
    app.appendChild(section);
    const img = section.querySelector('img');
    img.addEventListener('load', () => drawBoxes(section.querySelector('.stage'), img, row));
    section.querySelectorAll('input').forEach(input => input.addEventListener('change', () => saveRow(row.sample_id, section)));
    section.querySelector('textarea').addEventListener('input', () => saveRow(row.sample_id, section));
  });
  updateProgress();
}
function drawBoxes(stage, img, row) {
  stage.querySelectorAll('.box').forEach(node => node.remove());
  const sx=img.clientWidth/img.naturalWidth, sy=img.clientHeight/img.naturalHeight;
  [['a',row.annotator_a.boxes],['b',row.annotator_b.boxes]].forEach(([kind,boxes]) => boxes.forEach(box => {
    const div=document.createElement('div'); div.className=`box ${kind}`;
    Object.assign(div.style,{left:`${box.x*sx}px`,top:`${box.y*sy}px`,width:`${box.w*sx}px`,height:`${box.h*sy}px`});
    div.innerHTML=`<span>${kind.toUpperCase()}: ${esc(box.label)}</span>`; stage.appendChild(div);
  }));
}
function saveRow(id, section) {
  state[id]={decision:section.querySelector('input:checked')?.value || '',notes:section.querySelector('textarea').value.trim()};
  localStorage.setItem('spatial-provenance-extension-adjudication', JSON.stringify(state)); updateProgress();
}
function updateProgress() {
  const complete=rows.filter(row => state[row.sample_id]?.decision && state[row.sample_id]?.notes).length;
  document.getElementById('progress').textContent=`${complete}/17 decisions with rationale`;
}
document.getElementById('export').addEventListener('click', () => {
  const decisions=rows.map(row => ({sample_id:row.sample_id,decision:state[row.sample_id]?.decision||'',notes:state[row.sample_id]?.notes||''}));
  const incomplete=decisions.filter(row => !row.decision || !row.notes);
  if (incomplete.length && !confirm(`${incomplete.length} rows are incomplete. Export anyway?`)) return;
  const blob=new Blob([JSON.stringify(decisions,null,2)+'\\n'],{type:'application/json'});
  const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download='extension_20_adjudication_decisions.json'; a.click(); URL.revokeObjectURL(a.href);
});
render();
</script></body></html>""".replace("DATA_PLACEHOLDER", data)


def build_readme() -> str:
    return """# Extension-20 Adjudication Tool

Open `index.html`. Red solid boxes are the primary annotation and blue dashed
boxes are the independent secondary annotation. For each of the 17 conflict
rows, select one final rule and write a brief rationale:

- `primary`: retain the primary boxes;
- `secondary`: adopt the independent secondary boxes;
- `union`: retain both sets (later deduplicated by exact coordinates/labels);
- `needs_redraw`: neither proposal is acceptable; redraw in the original bbox tool.

Export the decisions to `extension_20_adjudication_decisions.json` in this
directory. A row without both a decision and rationale remains unresolved.
The three rows that passed the preregistered agreement rule are already in
`../consensus_draft.json` and do not appear in this tool.
"""


if __name__ == "__main__":
    main()
