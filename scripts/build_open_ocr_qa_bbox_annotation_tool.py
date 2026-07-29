#!/usr/bin/env python3
"""Build a static bbox annotation tool for the open OCR QA stress pack."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PREFILL_JSONL = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_evidence_prefill"
    / "evidence_prefill_pack.jsonl"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(PREFILL_JSONL))
    parser.add_argument("--output-dir", default="runs/problem_optimization_audit/open_ocr_qa_bbox_annotation_tool")
    args = parser.parse_args()

    rows = read_jsonl(Path(args.input))
    data = [tool_row(row) for row in rows]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tool_name = out_dir.name
    (out_dir / "index.html").write_text(
        build_html(
            data,
            storage_key=f"open_ocr_qa_bbox_annotations_v2_{tool_name}",
            export_filename=f"{tool_name}_annotations.json",
        ),
        encoding="utf-8",
    )
    write_jsonl(out_dir / "annotation_seed.jsonl", seed_rows(data))
    write_csv(out_dir / "annotation_tool_summary.csv", summary_rows(data))
    (out_dir / "annotation_schema.md").write_text(schema_markdown(), encoding="utf-8")
    print(f"Wrote bbox annotation tool for {len(data)} rows to {out_dir / 'index.html'}")


def tool_row(row: dict[str, Any]) -> dict[str, Any]:
    image_path = Path(str(row["image_path"]))
    return {
        "sample_id": row["sample_id"],
        "task": row["task"],
        "question_id": row["question_id"],
        "question": row["question"],
        "gold_answers": row["gold_answers"],
        "full_answer": row["full_answer"],
        "pruned_0p30_answer": row["pruned_0p30_answer"],
        "pruned_0p50_answer": row["pruned_0p50_answer"],
        "pruned_0p70_answer": row["pruned_0p70_answer"],
        "delta_0p30": row["delta_0p30"],
        "delta_0p50": row["delta_0p50"],
        "delta_0p70": row["delta_0p70"],
        "selection_reasons": row["selection_reasons"],
        "stress_tags": row["stress_tags"],
        "prefill_evidence_units": row["prefill_evidence_units"],
        "prefill_complexity": row["prefill_complexity"],
        "prefill_suggested_region_count": row["prefill_suggested_region_count"],
        "prefill_annotation_hint": row["prefill_annotation_hint"],
        "image_src": image_path.resolve().as_uri(),
        "absolute_image_path": str(image_path),
    }


def seed_rows(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in data:
        rows.append(
            {
                "sample_id": row["sample_id"],
                "task": row["task"],
                "question_id": row["question_id"],
                "evidence_units": row["prefill_evidence_units"],
                "boxes": [],
                "notes": "",
                "status": "unannotated",
            }
        )
    return rows


def summary_rows(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {"scope": "all", "metric": "rows", "value": len(data)},
        {"scope": "all", "metric": "images_linked", "value": sum(Path(row["absolute_image_path"]).exists() for row in data)},
    ]
    for task in sorted({row["task"] for row in data}):
        task_rows = [row for row in data if row["task"] == task]
        rows.append({"scope": task, "metric": "rows", "value": len(task_rows)})
        for complexity in sorted({row["prefill_complexity"] for row in task_rows}):
            rows.append(
                {
                    "scope": task,
                    "metric": f"complexity_{complexity}",
                    "value": sum(row["prefill_complexity"] == complexity for row in task_rows),
                }
            )
    return rows


def build_html(data: list[dict[str, Any]], *, storage_key: str, export_filename: str) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    storage_key_js = json.dumps(storage_key)
    export_filename_js = json.dumps(export_filename)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Open OCR QA Evidence Box Annotation</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f6f6f3; color: #1c1c1a; }}
    header {{ display: flex; align-items: center; gap: 12px; padding: 10px 14px; border-bottom: 1px solid #ccc; background: #fff; position: sticky; top: 0; z-index: 3; }}
    button {{ padding: 7px 10px; border: 1px solid #999; background: #fff; cursor: pointer; border-radius: 4px; }}
    button:hover {{ background: #eeeeea; }}
    main {{ display: grid; grid-template-columns: minmax(460px, 1fr) 420px; height: calc(100vh - 53px); }}
    #stageWrap {{ overflow: auto; padding: 16px; }}
    #canvas {{ background: #fff; border: 1px solid #aaa; box-shadow: 0 1px 4px rgba(0,0,0,.12); cursor: crosshair; }}
    aside {{ overflow: auto; border-left: 1px solid #ccc; padding: 14px; background: #fff; }}
    h2 {{ font-size: 16px; margin: 0 0 10px; }}
    .meta {{ font-size: 13px; line-height: 1.45; margin-bottom: 10px; }}
    .pill {{ display: inline-block; margin: 2px 4px 2px 0; padding: 2px 6px; border: 1px solid #bbb; border-radius: 999px; background: #fafafa; font-size: 12px; }}
    textarea, input, select {{ width: 100%; }}
    textarea {{ min-height: 72px; }}
    .field {{ margin: 8px 0; }}
    .field label {{ display: block; font-size: 12px; font-weight: bold; margin-bottom: 3px; }}
    .boxList {{ font-family: monospace; white-space: pre-wrap; font-size: 12px; background: #f3f3ef; padding: 8px; border: 1px solid #ddd; }}
    .hint {{ background: #fff8da; border: 1px solid #e0c85a; padding: 8px; margin: 8px 0; font-size: 13px; }}
  </style>
</head>
<body>
  <header>
    <button id="prevBtn">Prev</button>
    <button id="nextBtn">Next</button>
    <span id="counter"></span>
    <button id="undoBtn">Undo Box</button>
    <button id="clearBtn">Clear Boxes</button>
    <button id="exportBtn">Export JSON</button>
    <input id="jumpInput" type="number" min="1" style="width:70px" />
    <button id="jumpBtn">Jump</button>
  </header>
  <main>
    <section id="stageWrap"><canvas id="canvas"></canvas></section>
    <aside>
      <h2 id="title"></h2>
      <div class="meta" id="meta"></div>
      <div class="hint" id="hint"></div>
      <div class="field">
        <label for="boxType">New box type</label>
        <select id="boxType">
          <option value="answer_value">answer_value</option>
          <option value="field_label">field_label</option>
          <option value="row_header">row_header</option>
          <option value="column_header">column_header</option>
          <option value="comparison_anchor">comparison_anchor</option>
          <option value="context">context</option>
          <option value="other">other</option>
        </select>
      </div>
      <div class="field">
        <label for="boxText">New box text</label>
        <input id="boxText" type="text" />
      </div>
      <div class="field">
        <label for="statusSelect">Row status</label>
        <select id="statusSelect">
          <option value="in_progress">in_progress</option>
          <option value="annotated">annotated</option>
          <option value="needs_review">needs_review</option>
          <option value="not_visible">not_visible</option>
        </select>
      </div>
      <div class="field">
        <label for="notes">Notes</label>
        <textarea id="notes"></textarea>
      </div>
      <h2>Boxes</h2>
      <div class="boxList" id="boxList"></div>
    </aside>
  </main>
  <script id="data" type="application/json">{html.escape(payload)}</script>
  <script>
    const rows = JSON.parse(document.getElementById('data').textContent);
    const storeKey = {storage_key_js};
    let annotations = JSON.parse(localStorage.getItem(storeKey) || '{{}}');
    let idx = 0, img = new Image(), scale = 1, drawing = false, start = null, draft = null;
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const notes = document.getElementById('notes');
    const boxType = document.getElementById('boxType');
    const boxText = document.getElementById('boxText');
    const statusSelect = document.getElementById('statusSelect');

    function currentAnn() {{
      const id = rows[idx].sample_id;
      if (!annotations[id]) annotations[id] = {{ sample_id: id, task: rows[idx].task, question_id: rows[idx].question_id, boxes: [], notes: '', status: 'in_progress' }};
      return annotations[id];
    }}
    function save() {{ localStorage.setItem(storeKey, JSON.stringify(annotations)); }}
    function loadRow(i) {{
      idx = Math.max(0, Math.min(rows.length - 1, i));
      const row = rows[idx];
      img = new Image();
      img.onload = () => {{
        const maxW = Math.min(1100, window.innerWidth - 470);
        scale = Math.min(1, maxW / img.width);
        canvas.width = Math.round(img.width * scale);
        canvas.height = Math.round(img.height * scale);
        draw();
      }};
      img.src = row.image_src;
      notes.value = currentAnn().notes || '';
      statusSelect.value = currentAnn().status || 'in_progress';
      boxType.value = 'answer_value';
      boxText.value = row.prefill_evidence_units || row.gold_answers || '';
      renderSide();
    }}
    function draw() {{
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      for (const box of currentAnn().boxes) drawBox(box, '#ff6b00');
      if (draft) drawBox(draft, '#0066ff');
    }}
    function drawBox(box, color) {{
      ctx.strokeStyle = color; ctx.lineWidth = 2;
      ctx.strokeRect(box.x * scale, box.y * scale, box.w * scale, box.h * scale);
      if (box.label) {{
        ctx.fillStyle = color;
        ctx.font = '12px Arial';
        ctx.fillText(String(box.label).slice(0, 40), box.x * scale + 3, Math.max(12, box.y * scale - 4));
      }}
    }}
    function renderSide() {{
      const row = rows[idx], ann = currentAnn();
      document.getElementById('counter').textContent = `${{idx + 1}} / ${{rows.length}}`;
      document.getElementById('title').textContent = `${{row.task}} · ${{row.question_id}}`;
      document.getElementById('meta').innerHTML = `
        <b>Question:</b> ${{esc(row.question)}}<br>
        <b>Gold:</b> ${{esc(row.gold_answers)}}<br>
        <b>Full:</b> ${{esc(row.full_answer)}}<br>
        <b>30/50/70:</b> ${{esc(row.pruned_0p30_answer)}} (${{row.delta_0p30}}) / ${{esc(row.pruned_0p50_answer)}} (${{row.delta_0p50}}) / ${{esc(row.pruned_0p70_answer)}} (${{row.delta_0p70}})<br>
        <b>Reasons:</b> ${{pills(row.selection_reasons)}}<br>
        <b>Stress:</b> ${{pills(row.stress_tags)}}<br>
        <b>Targets:</b> ${{esc(row.prefill_evidence_units)}}<br>
        <b>Complexity:</b> ${{esc(row.prefill_complexity)}} · suggested regions ${{row.prefill_suggested_region_count}}
      `;
      document.getElementById('hint').textContent = row.prefill_annotation_hint;
      document.getElementById('boxList').textContent = JSON.stringify(ann.boxes, null, 2);
    }}
    function pills(text) {{ return String(text || '').split(';').filter(Boolean).map(x => `<span class="pill">${{esc(x)}}</span>`).join(' '); }}
    function esc(text) {{ return String(text || '').replace(/[&<>]/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;'}}[c])); }}
    function pointer(e) {{
      const r = canvas.getBoundingClientRect();
      return {{ x: (e.clientX - r.left) / scale, y: (e.clientY - r.top) / scale }};
    }}
    canvas.onmousedown = e => {{ drawing = true; start = pointer(e); draft = {{x:start.x,y:start.y,w:0,h:0}}; }};
    canvas.onmousemove = e => {{
      if (!drawing) return;
      const p = pointer(e);
      draft = {{ x: Math.min(start.x, p.x), y: Math.min(start.y, p.y), w: Math.abs(p.x - start.x), h: Math.abs(p.y - start.y) }};
      draw();
    }};
    canvas.onmouseup = () => {{
      if (drawing && draft && draft.w > 3 && draft.h > 3) {{
        currentAnn().boxes.push(roundBox(draft));
        currentAnn().status = 'annotated';
        statusSelect.value = 'annotated';
        save(); renderSide();
      }}
      drawing = false; draft = null; draw();
    }};
    function currentBoxLabel() {{
      const type = boxType.value || 'answer_value';
      const text = (boxText.value || rows[idx].prefill_evidence_units || rows[idx].gold_answers || '').trim();
      return text ? `${{type}}:${{text}}` : type;
    }}
    function roundBox(b) {{ return {{ x: Math.round(b.x), y: Math.round(b.y), w: Math.round(b.w), h: Math.round(b.h), label: currentBoxLabel() }}; }}
    notes.oninput = () => {{ currentAnn().notes = notes.value; save(); }};
    statusSelect.onchange = () => {{ currentAnn().status = statusSelect.value; save(); }};
    document.getElementById('prevBtn').onclick = () => loadRow(idx - 1);
    document.getElementById('nextBtn').onclick = () => loadRow(idx + 1);
    document.getElementById('undoBtn').onclick = () => {{ currentAnn().boxes.pop(); save(); draw(); renderSide(); }};
    document.getElementById('clearBtn').onclick = () => {{ currentAnn().boxes = []; save(); draw(); renderSide(); }};
    document.getElementById('jumpBtn').onclick = () => loadRow(Number(document.getElementById('jumpInput').value || 1) - 1);
    document.getElementById('exportBtn').onclick = () => {{
      const output = rows.map((row) => {{
        const ann = annotations[row.sample_id] || {{ sample_id: row.sample_id, task: row.task, question_id: row.question_id, boxes: [], notes: '', status: 'unannotated' }};
        if (ann.boxes.length && (!ann.status || ann.status === 'in_progress')) ann.status = 'annotated';
        return ann;
      }});
      const blob = new Blob([JSON.stringify(output, null, 2)], {{type:'application/json'}});
      const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = {export_filename_js}; a.click();
    }};
    window.onresize = () => loadRow(idx);
    loadRow(0);
  </script>
</body>
</html>
"""


def schema_markdown() -> str:
    return """# Annotation Schema

Exported JSON is a list of rows:

- `sample_id`: dataset-specific sample id.
- `task`: `DocVQA-lite` or `TextVQA-lite`.
- `question_id`: source question id.
- `boxes`: list of `{x, y, w, h, label}` in original exported image pixel coordinates.
- `notes`: optional annotator note.
- `status`: `annotated`, `needs_review`, `not_visible`, or `in_progress`.

Boxes should cover the minimal visible evidence needed to answer the question.
Use one box per contiguous region. Label each box as `type:text`, for example:

- `answer_value:3973`
- `field_label:Purchase Order Number`
- `row_header:26`
- `column_header:1969`
- `comparison_anchor:sugar`
- `context:Follow-up suggestions`

If a row, column, table header, field label, comparison anchor, or nearby text
is necessary to identify the answer, draw it as a separate box rather than only
mentioning it in notes. Use notes for uncertainty, unreadable regions, or cases
where the evidence cannot be localized without a large region.
"""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
