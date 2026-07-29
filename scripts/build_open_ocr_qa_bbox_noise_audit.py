#!/usr/bin/env python3
"""Stress-test open-QA bbox evidence audits under box noise and missing boxes.

This script does not run model inference. It perturbs existing TextVQA GT boxes
and DocVQA OCR line-context boxes, reuses cached Qwen open-QA pruning traces,
and measures how evidence-coverage audit signals change.
"""

from __future__ import annotations

import csv
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "problem_optimization_audit" / "open_ocr_qa_bbox_noise_audit"
PREFILL_JSONL = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_evidence_prefill"
    / "evidence_prefill_pack.jsonl"
)
TEXTVQA_BOXES = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_textvqa_gt_bbox"
    / "external_bbox_annotations.jsonl"
)
DOCVQA_BOXES = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "open_ocr_qa_docvqa_hxlinh_line_context_bbox"
    / "external_bbox_annotations.jsonl"
)
EVALUATOR = ROOT / "scripts" / "evaluate_open_ocr_qa_bbox_ecr.py"
SEED = 20260717


VARIANTS = [
    {"variant": "clean", "jitter": 0.0, "drop": 0.0, "scale": 1.0, "note": "unmodified imported boxes"},
    {"variant": "jitter_10pct", "jitter": 0.10, "drop": 0.0, "scale": 1.0, "note": "box-center jitter up to 10% of box size"},
    {"variant": "jitter_25pct", "jitter": 0.25, "drop": 0.0, "scale": 1.0, "note": "box-center jitter up to 25% of box size"},
    {"variant": "shrink_20pct", "jitter": 0.0, "drop": 0.0, "scale": 0.80, "note": "boxes shrunk around center by 20%"},
    {"variant": "expand_20pct", "jitter": 0.0, "drop": 0.0, "scale": 1.20, "note": "boxes expanded around center by 20%"},
    {"variant": "drop_20pct", "jitter": 0.0, "drop": 0.20, "scale": 1.0, "note": "independent 20% evidence-box dropout"},
    {"variant": "drop_40pct", "jitter": 0.0, "drop": 0.40, "scale": 1.0, "note": "independent 40% evidence-box dropout"},
    {
        "variant": "mixed_light",
        "jitter": 0.10,
        "drop": 0.20,
        "scale": 1.0,
        "note": "10% jitter plus 20% evidence-box dropout",
    },
    {
        "variant": "mixed_heavy",
        "jitter": 0.25,
        "drop": 0.40,
        "scale": 1.0,
        "note": "25% jitter plus 40% evidence-box dropout",
    },
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prefill = {str(row["sample_id"]): row for row in read_jsonl(PREFILL_JSONL)}
    base_rows = load_annotated_rows([TEXTVQA_BOXES, DOCVQA_BOXES])
    base_rows = [row for row in base_rows if image_size(prefill, str(row["sample_id"])) != (0, 0)]

    all_variant_rows: list[dict[str, Any]] = []
    all_summary_rows: list[dict[str, Any]] = []
    for idx, spec in enumerate(VARIANTS):
        rng = random.Random(SEED + idx)
        variant_rows = [perturb_row(row, prefill, spec, rng) for row in base_rows]
        variant_meta = variant_metadata_rows(spec, base_rows, variant_rows)
        variant_dir = OUT_DIR / str(spec["variant"])
        variant_dir.mkdir(parents=True, exist_ok=True)
        annotations_path = variant_dir / "annotations.jsonl"
        write_jsonl(annotations_path, variant_rows)
        run_evaluator(annotations_path, variant_dir)

        all_variant_rows.extend(variant_meta)
        summary = read_csv(variant_dir / "bbox_ecr_summary.csv")
        all_summary_rows.extend(flatten_summary(spec, summary, variant_meta))

    write_csv(OUT_DIR / "bbox_noise_variant_rows.csv", all_variant_rows)
    write_csv(OUT_DIR / "bbox_noise_summary.csv", all_summary_rows)
    (OUT_DIR / "bbox_noise_report.md").write_text(build_report(all_summary_rows, all_variant_rows), encoding="utf-8")
    print(f"Wrote bbox noise audit to {OUT_DIR}")


def load_annotated_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            sample_id = str(row.get("sample_id", ""))
            boxes = row.get("boxes", [])
            if sample_id in seen or not isinstance(boxes, list) or not boxes:
                continue
            rows.append(row)
            seen.add(sample_id)
    return rows


def perturb_row(row: dict[str, Any], prefill: dict[str, dict[str, Any]], spec: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    sample_id = str(row["sample_id"])
    width, height = image_size(prefill, sample_id)
    boxes = [
        perturb_box(dict(box), width, height, float(spec["jitter"]), float(spec["scale"]), rng)
        for box in row.get("boxes", [])
        if rng.random() >= float(spec["drop"])
    ]
    out = dict(row)
    out["boxes"] = [box for box in boxes if box is not None]
    out["status"] = f"bbox_noise_{spec['variant']}"
    out["notes"] = f"{row.get('notes', '')}; bbox_noise_variant={spec['variant']}; {spec['note']}".strip("; ")
    return out


def perturb_box(
    box: dict[str, Any],
    width: int,
    height: int,
    jitter: float,
    scale: float,
    rng: random.Random,
) -> dict[str, Any] | None:
    try:
        x = float(box["x"])
        y = float(box["y"])
        w = float(box["w"])
        h = float(box["h"])
    except (KeyError, TypeError, ValueError):
        return None
    if width <= 0 or height <= 0 or w <= 0 or h <= 0:
        return None

    cx = x + 0.5 * w + rng.uniform(-jitter, jitter) * w
    cy = y + 0.5 * h + rng.uniform(-jitter, jitter) * h
    nw = max(1.0, w * scale)
    nh = max(1.0, h * scale)
    nx1 = clamp(cx - 0.5 * nw, 0.0, float(width))
    ny1 = clamp(cy - 0.5 * nh, 0.0, float(height))
    nx2 = clamp(cx + 0.5 * nw, 0.0, float(width))
    ny2 = clamp(cy + 0.5 * nh, 0.0, float(height))
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    out = dict(box)
    out.update({"x": round(nx1, 2), "y": round(ny1, 2), "w": round(nx2 - nx1, 2), "h": round(ny2 - ny1, 2)})
    return out


def variant_metadata_rows(
    spec: dict[str, Any],
    base_rows: list[dict[str, Any]],
    variant_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for base, row in zip(base_rows, variant_rows):
        base_count = len(base.get("boxes", []))
        kept_count = len(row.get("boxes", []))
        out.append(
            {
                "variant": spec["variant"],
                "task": row.get("task", ""),
                "sample_id": row.get("sample_id", ""),
                "base_box_count": base_count,
                "variant_box_count": kept_count,
                "box_retention_rate": fmt(kept_count / base_count if base_count else 0.0),
                "sample_has_box": int(kept_count > 0),
                "note": spec["note"],
            }
        )
    return out


def flatten_summary(
    spec: dict[str, Any],
    summary_rows: list[dict[str, str]],
    variant_meta: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_scope_metric = {(row["scope"], row["metric"]): row["value"] for row in summary_rows}
    meta_by_task = aggregate_variant_rows(variant_meta)
    retention_by_task = {
        str(row["task"]): (
            float(row["sample_has_box_rate"]),
            float(row["mean_box_retention_rate"]),
        )
        for row in meta_by_task
    }
    out: list[dict[str, Any]] = []
    for task in ("TextVQA-lite", "DocVQA-lite"):
        sample_has_box_rate, box_retention_rate = retention_by_task.get(task, (0.0, 0.0))
        for ratio in ("0.30", "0.50", "0.70"):
            scope = f"{task}@{ratio}"
            mean_ecr = safe_float(by_scope_metric.get((scope, "mean_ECR"), ""))
            mean_all_regions = safe_float(by_scope_metric.get((scope, "mean_all_regions_ECR_ge_0p50"), ""))
            out.append(
                {
                    "variant": spec["variant"],
                    "task": task,
                    "budget_keep_ratio": ratio,
                    "annotated_samples": by_scope_metric.get((scope, "annotated_samples"), ""),
                    "scored_rows": by_scope_metric.get((scope, "scored_rows"), ""),
                    "sample_has_box_rate": fmt(sample_has_box_rate),
                    "mean_box_retention_rate": fmt(box_retention_rate),
                    "mean_ECR": by_scope_metric.get((scope, "mean_ECR"), ""),
                    "retention_adjusted_ECR": fmt(mean_ecr * box_retention_rate) if mean_ecr is not None else "",
                    "mean_CenterR": by_scope_metric.get((scope, "mean_CenterR"), ""),
                    "mean_PatchR": by_scope_metric.get((scope, "mean_PatchR"), ""),
                    "mean_worst_region_ECR": by_scope_metric.get((scope, "mean_worst_region_ECR"), ""),
                    "mean_all_regions_ECR_ge_0p50": by_scope_metric.get((scope, "mean_all_regions_ECR_ge_0p50"), ""),
                    "sample_adjusted_all_regions_ECR_ge_0p50": fmt(mean_all_regions * sample_has_box_rate)
                    if mean_all_regions is not None
                    else "",
                    "note": spec["note"],
                }
            )
    return out


def build_report(summary_rows: list[dict[str, Any]], variant_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Open OCR QA BBox Noise Audit",
        "",
        "This audit perturbs imported evidence boxes and re-evaluates cached Qwen open-QA pruning masks. It tests the sensitivity of evidence-audit signals to detector-like localization noise and missing boxes; it is not an end-to-end noisy-detector pruning run.",
        "",
        "## Variant Coverage",
        "",
        table(
            aggregate_variant_rows(variant_rows),
            ["variant", "task", "samples", "mean_base_boxes", "mean_variant_boxes", "sample_has_box_rate", "mean_box_retention_rate"],
        ),
        "",
        "## Evidence Coverage Under Box Noise",
        "",
        table(
            summary_rows,
            [
                "variant",
                "task",
                "budget_keep_ratio",
                "scored_rows",
                "sample_has_box_rate",
                "mean_box_retention_rate",
                "mean_ECR",
                "retention_adjusted_ECR",
                "mean_worst_region_ECR",
                "mean_all_regions_ECR_ge_0p50",
                "sample_adjusted_all_regions_ECR_ge_0p50",
                "note",
            ],
        ),
        "",
        "## Reading",
        "",
        "- Coordinate jitter and box scaling mainly test whether the ECR audit is brittle to approximate localization.",
        "- Box dropout is a missing-detector stress test. Rows with no remaining boxes cannot be scored, so `sample_has_box_rate`, `mean_box_retention_rate`, and adjusted metrics should be read together with raw ECR.",
        "- If a variant keeps similar ECR but sharply reduces all-region coverage, it means average evidence availability is stable while multi-region document evidence remains fragile.",
    ]
    return "\n".join(lines) + "\n"


def aggregate_variant_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["variant"]), str(row["task"])), []).append(row)
    out: list[dict[str, Any]] = []
    for (variant, task), group in sorted(groups.items()):
        out.append(
            {
                "variant": variant,
                "task": task,
                "samples": len(group),
                "mean_base_boxes": fmt(mean(float(row["base_box_count"]) for row in group)),
                "mean_variant_boxes": fmt(mean(float(row["variant_box_count"]) for row in group)),
                "sample_has_box_rate": fmt(mean(float(row["sample_has_box"]) for row in group)),
                "mean_box_retention_rate": fmt(mean(float(row["box_retention_rate"]) for row in group)),
            }
        )
    return out


def run_evaluator(annotations_path: Path, out_dir: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(EVALUATOR),
            "--annotations",
            str(annotations_path),
            "--output-dir",
            str(out_dir),
        ],
        check=True,
        cwd=str(ROOT),
    )


def image_size(prefill: dict[str, dict[str, Any]], sample_id: str) -> tuple[int, int]:
    image_path = Path(str(prefill.get(sample_id, {}).get("image_path", "")))
    if not image_path.exists():
        return 0, 0
    with Image.open(image_path) as image:
        return image.size


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines)


def mean(values: Any) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def fmt(value: float) -> str:
    return f"{value:.3f}"


def safe_float(value: str) -> float | None:
    try:
        if value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def clamp(value: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, value))


if __name__ == "__main__":
    main()
