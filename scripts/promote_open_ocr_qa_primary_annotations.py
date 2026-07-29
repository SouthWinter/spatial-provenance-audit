#!/usr/bin/env python3
"""Promote a completed primary manual export to final annotations.

This is intentionally strict. It prevents incomplete, smoke-like, or
non-annotated exports from becoming paper-facing manual evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PREFILL_JSONL = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_evidence_prefill"
    / "evidence_prefill_pack.jsonl"
)
DEFAULT_OUT = ROOT / "runs" / "problem_optimization_audit" / "open_ocr_qa_final_annotations"
EXPECTED_ROWS = 96
SMOKE_MARKERS = {"foo", "bar", "smoke"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, help="Completed primary manual JSON/JSONL export")
    parser.add_argument("--agreement-summary", default="", help="Optional calibration agreement summary CSV")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--allow-nonannotated", action="store_true", help="Dry-run escape hatch; do not use for paper-facing final annotations")
    args = parser.parse_args()

    annotations = load_annotations(Path(args.annotations))
    prefill = {row["sample_id"]: row for row in read_jsonl(PREFILL_JSONL)}
    rows, problems = normalize_and_check(annotations, prefill, args.allow_nonannotated)
    agreement = read_agreement(Path(args.agreement_summary)) if args.agreement_summary else []
    problems.extend(agreement_problems(agreement))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "promotion_check_rows.csv", problems)
    write_csv(out_dir / "final_annotation_summary.csv", summary_rows(rows, problems, agreement))
    (out_dir / "final_annotation_report.md").write_text(build_report(rows, problems, agreement), encoding="utf-8")

    if problems:
        raise SystemExit(f"Promotion failed with {len(problems)} problem(s); see {out_dir / 'promotion_check_rows.csv'}")

    rows.sort(key=lambda row: (row["task"], row["sample_id"]))
    write_json(out_dir / "final_annotations.json", rows)
    write_jsonl(out_dir / "final_annotations.jsonl", rows)
    print(f"Promoted {len(rows)} final manual annotation rows to {out_dir}")


def normalize_and_check(
    annotations: list[dict[str, Any]],
    prefill: dict[str, dict[str, Any]],
    allow_nonannotated: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    problems: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    by_sample: dict[str, dict[str, Any]] = {}
    for row in annotations:
        sid = str(row.get("sample_id", ""))
        if not sid:
            problems.append(problem("", "missing_sample_id", "row has no sample_id"))
            continue
        if sid in by_sample:
            problems.append(problem(sid, "duplicate_sample_id", "duplicate annotation row"))
        by_sample[sid] = row

    expected_ids = set(prefill)
    observed_ids = set(by_sample)
    for sid in sorted(expected_ids - observed_ids):
        problems.append(problem(sid, "missing_expected_row", "expected row absent from annotation export"))
    for sid in sorted(observed_ids - expected_ids):
        problems.append(problem(sid, "unexpected_row", "annotation export contains a row outside the fixed 96-row pack"))
    if len(observed_ids & expected_ids) != EXPECTED_ROWS:
        problems.append(problem("all", "wrong_row_count", f"expected {EXPECTED_ROWS}, got {len(observed_ids & expected_ids)} fixed-pack rows"))

    for sid in sorted(observed_ids & expected_ids):
        row = by_sample[sid]
        meta = prefill[sid]
        boxes = row.get("boxes", [])
        if not isinstance(boxes, list):
            boxes = []
            problems.append(problem(sid, "boxes_not_list", "boxes field is not a list"))
        status = str(row.get("status", "")).strip()
        if status != "annotated" and not allow_nonannotated:
            problems.append(problem(sid, "status_not_annotated", f"status={status!r}"))
        if not boxes:
            problems.append(problem(sid, "empty_boxes", "annotated row has no evidence boxes"))

        width, height = image_size(meta)
        clean_boxes = []
        for box_idx, box in enumerate(boxes):
            box_problem = validate_box(box, width, height)
            if box_problem:
                problems.append(problem(sid, "invalid_box", f"box {box_idx}: {box_problem}"))
                continue
            label = str(box.get("label", "")).strip()
            if not label:
                problems.append(problem(sid, "unlabeled_box", f"box {box_idx} has empty label"))
            if is_smoke_label(label) or is_smoke_label(str(row.get("notes", ""))):
                problems.append(problem(sid, "smoke_like_label", f"box {box_idx} label/notes looks synthetic"))
            clean_boxes.append(
                {
                    "x": round(float(box["x"]), 3),
                    "y": round(float(box["y"]), 3),
                    "w": round(float(box["w"]), 3),
                    "h": round(float(box["h"]), 3),
                    "label": label,
                }
            )
        rows.append(
            {
                "sample_id": sid,
                "task": meta.get("task", row.get("task", "")),
                "question_id": meta.get("question_id", row.get("question_id", "")),
                "status": "annotated" if status == "annotated" else status,
                "boxes": clean_boxes,
                "notes": str(row.get("notes", "")),
                "source": str(row.get("source", "primary_manual_after_calibration")),
                "annotation_provenance": str(row.get("annotation_provenance", "")),
            }
        )
    return rows, problems


def validate_box(box: Any, width: int, height: int) -> str:
    if not isinstance(box, dict):
        return "box is not an object"
    try:
        x = float(box.get("x", 0))
        y = float(box.get("y", 0))
        w = float(box.get("w", 0))
        h = float(box.get("h", 0))
    except (TypeError, ValueError):
        return "non-numeric coordinates"
    if width <= 0 or height <= 0:
        return "missing image size"
    if w <= 0 or h <= 0:
        return "non-positive width/height"
    if x < 0 or y < 0 or x + w > width + 1 or y + h > height + 1:
        return f"outside image bounds {width}x{height}"
    return ""


def image_size(row: dict[str, Any]) -> tuple[int, int]:
    path = Path(str(row.get("image_path", "")))
    if not path.exists():
        return 0, 0
    with Image.open(path) as image:
        return image.size


def read_agreement(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return [{"scope": "all", "metric": "missing_agreement_summary", "value": "1"}]
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def agreement_problems(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    metrics = {(row.get("scope", ""), row.get("metric", "")): row.get("value", "") for row in rows}
    problems = []
    if metrics.get(("all", "missing_agreement_summary")) == "1":
        problems.append(problem("calibration", "missing_agreement_summary", "calibration agreement summary was not found"))
        return problems
    samples = int(float(metrics.get(("all", "samples"), "0") or 0))
    present = int(float(metrics.get(("all", "present_in_both"), "0") or 0))
    if samples < 12:
        problems.append(problem("calibration", "too_few_calibration_rows", f"samples={samples}, expected at least 12"))
    if present < 12:
        problems.append(problem("calibration", "missing_secondary_rows", f"present_in_both={present}, expected at least 12"))
    return problems


def summary_rows(
    rows: list[dict[str, Any]],
    problems: list[dict[str, Any]],
    agreement: list[dict[str, str]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [
        {"scope": "all", "metric": "final_rows", "value": len(rows)},
        {"scope": "all", "metric": "problem_rows", "value": len(problems)},
        {"scope": "all", "metric": "annotated_rows", "value": sum(row["status"] == "annotated" for row in rows)},
        {"scope": "all", "metric": "rows_with_boxes", "value": sum(bool(row["boxes"]) for row in rows)},
        {"scope": "calibration", "metric": "agreement_summary_rows", "value": len(agreement)},
    ]
    for task in sorted({row["task"] for row in rows}):
        task_rows = [row for row in rows if row["task"] == task]
        out.append({"scope": task, "metric": "final_rows", "value": len(task_rows)})
        out.append({"scope": task, "metric": "rows_with_boxes", "value": sum(bool(row["boxes"]) for row in task_rows)})
    return out


def build_report(rows: list[dict[str, Any]], problems: list[dict[str, Any]], agreement: list[dict[str, str]]) -> str:
    lines = [
        "# Final Manual Annotation Promotion",
        "",
        "This report records whether a primary human annotation export can become paper-facing final evidence.",
        "",
        "| Scope | Metric | Value |",
        "| --- | --- | ---: |",
    ]
    for row in summary_rows(rows, problems, agreement):
        lines.append(f"| {row['scope']} | {row['metric']} | {row['value']} |")
    if problems:
        lines.extend(["", "## Blocking Problems", "", "| Sample | Kind | Detail |", "| --- | --- | --- |"])
        for row in problems:
            lines.append(f"| {row['sample_id']} | {row['kind']} | {row['detail']} |")
    else:
        lines.extend(["", "Promotion status: pass. Final annotation JSON/JSONL files were written."])
    return "\n".join(lines) + "\n"


def problem(sample_id: str, kind: str, detail: str) -> dict[str, Any]:
    return {"sample_id": sample_id, "kind": kind, "detail": detail}


def is_smoke_label(text: str) -> bool:
    words = set(re.findall(r"[a-z0-9_]+", text.lower()))
    return bool(words & SMOKE_MARKERS)


def load_annotations(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
