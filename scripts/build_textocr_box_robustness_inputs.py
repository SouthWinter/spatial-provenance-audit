#!/usr/bin/env python
"""Build TextOCR-Hard probe files with simulated OCR detector box noise.

The selector consumes ``evidence_regions`` from the emitted files, while the
original answer-supporting boxes are preserved in ``oracle_evidence_regions``
for post-hoc auditing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


Box = list[float]


VARIANTS: dict[str, dict[str, float]] = {
    "gt": {"recall": 1.0, "jitter": 0.0, "spurious": 0.0},
    "jitter_light": {"recall": 1.0, "jitter": 0.15, "spurious": 0.0},
    "jitter_heavy": {"recall": 1.0, "jitter": 0.35, "spurious": 0.0},
    "detected_like": {"recall": 0.85, "jitter": 0.20, "spurious": 1.0},
    "low_recall": {"recall": 0.60, "jitter": 0.20, "spurious": 1.0},
    "missing": {"recall": 0.0, "jitter": 0.0, "spurious": 0.0},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/textocr_val_hard_probes_500img.jsonl")
    parser.add_argument("--output-dir", default="data/textocr_box_robustness")
    parser.add_argument("--variants", default="gt,jitter_light,jitter_heavy,detected_like,low_recall,missing")
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    for variant_name in parse_variants(args.variants):
        spec = VARIANTS[variant_name]
        out_rows = [
            transform_row(row, variant_name=variant_name, spec=spec, seed=args.seed)
            for row in rows
        ]
        output = output_dir / f"textocr_hard_{variant_name}.jsonl"
        write_jsonl(output, out_rows)
        stats = summarize_variant(out_rows)
        stats.update({"variant": variant_name, "output": str(output)})
        summary_rows.append(stats)

    write_json(output_dir / "box_robustness_inputs_summary.json", summary_rows)
    write_markdown(output_dir / "box_robustness_inputs_summary.md", summary_rows)
    print(f"Wrote {len(summary_rows)} variants to {output_dir}")


def transform_row(row: dict[str, Any], *, variant_name: str, spec: dict[str, float], seed: int) -> dict[str, Any]:
    out = dict(row)
    oracle = normalize_regions(row.get("oracle_evidence_regions") or row.get("evidence_regions") or row.get("ocr_regions"))
    rng = random.Random(mixed_seed(seed, str(row.get("sample_id", row.get("id", ""))), variant_name))

    observed: list[Box] = []
    kept_oracle = 0
    for box in oracle:
        if rng.random() <= float(spec["recall"]):
            kept_oracle += 1
            observed.append(jitter_box(box, float(spec["jitter"]), rng))

    for _ in range(int(float(spec["spurious"]))):
        observed.append(random_spurious_box(oracle, rng))

    out["oracle_evidence_regions"] = oracle
    out["oracle_ocr_regions"] = oracle
    out["evidence_regions"] = observed
    out["ocr_regions"] = observed
    out["evidence_region_count"] = len(observed)
    out["has_bbox"] = bool(observed)
    out["base_has_bbox"] = bool(observed)
    out["bbox_source"] = f"simulated_ocr_{variant_name}"
    out["box_robustness_variant"] = variant_name
    out["box_detector_recall_target"] = float(spec["recall"])
    out["box_detector_jitter_frac"] = float(spec["jitter"])
    out["box_detector_spurious_count"] = int(float(spec["spurious"]))
    out["box_detector_observed_count"] = len(observed)
    out["box_detector_kept_oracle_count"] = kept_oracle
    out["box_detector_mean_best_iou"] = mean_best_iou(oracle, observed)
    return out


def jitter_box(box: Box, jitter: float, rng: random.Random) -> Box:
    if jitter <= 0:
        return list(box)
    x1, y1, x2, y2 = box
    width = max(1e-6, x2 - x1)
    height = max(1e-6, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    cx += rng.uniform(-jitter, jitter) * width
    cy += rng.uniform(-jitter, jitter) * height
    width *= max(0.25, 1.0 + rng.uniform(-jitter, jitter))
    height *= max(0.25, 1.0 + rng.uniform(-jitter, jitter))
    return clip_box([cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0])


def random_spurious_box(oracle: list[Box], rng: random.Random) -> Box:
    if oracle:
        ref = rng.choice(oracle)
        area_w = max(0.01, ref[2] - ref[0])
        area_h = max(0.01, ref[3] - ref[1])
    else:
        area_w = area_h = 0.04
    width = min(0.30, area_w * rng.uniform(0.7, 1.8))
    height = min(0.30, area_h * rng.uniform(0.7, 1.8))
    cx = rng.uniform(width / 2.0, 1.0 - width / 2.0)
    cy = rng.uniform(height / 2.0, 1.0 - height / 2.0)
    return clip_box([cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0])


def clip_box(box: Box) -> Box:
    x1, y1, x2, y2 = box
    x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
    if x2 <= x1:
        x2 = min(1.0, x1 + 1e-6)
    if y2 <= y1:
        y2 = min(1.0, y1 + 1e-6)
    return [x1, y1, x2, y2]


def normalize_regions(value: Any) -> list[Box]:
    if not isinstance(value, (list, tuple)):
        return []
    regions: list[Box] = []
    for item in value:
        raw = item.get("bbox") if isinstance(item, dict) else item
        if not isinstance(raw, (list, tuple)) or len(raw) < 4:
            continue
        try:
            box = [float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])]
        except Exception:
            continue
        box = clip_box(box)
        if (box[2] - box[0]) * (box[3] - box[1]) > 0:
            regions.append(box)
    return regions


def mean_best_iou(oracle: list[Box], observed: list[Box]) -> float:
    if not oracle:
        return 0.0
    return sum(max((iou(a, b) for b in observed), default=0.0) for a in oracle) / len(oracle)


def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def summarize_variant(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_probes": len(rows),
        "mean_observed_boxes": mean(float(row["box_detector_observed_count"]) for row in rows),
        "mean_kept_oracle_boxes": mean(float(row["box_detector_kept_oracle_count"]) for row in rows),
        "mean_best_iou": mean(float(row["box_detector_mean_best_iou"]) for row in rows),
        "missing_box_rate": mean(1.0 if not row.get("evidence_regions") else 0.0 for row in rows),
    }


def parse_variants(value: str) -> list[str]:
    variants = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants {unknown}. Choices: {sorted(VARIANTS)}")
    return variants


def mixed_seed(seed: int, sample_id: str, variant: str) -> int:
    digest = hashlib.sha256(f"{seed}:{sample_id}:{variant}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def mean(values: Any) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, obj: Any) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_markdown(path: str | Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# TextOCR Box Robustness Inputs",
        "",
        "| Variant | Probes | Obs. Boxes | Kept GT Boxes | Best IoU | Missing Rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {int(row['num_probes'])} | "
            f"{row['mean_observed_boxes']:.3f} | {row['mean_kept_oracle_boxes']:.3f} | "
            f"{row['mean_best_iou']:.3f} | {row['missing_box_rate']:.3f} |"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
