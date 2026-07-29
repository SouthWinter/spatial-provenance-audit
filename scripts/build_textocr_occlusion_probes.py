#!/usr/bin/env python
"""Build TextOCR-Hard input-space occlusion probes.

Each base yes/no probe is expanded into three matched views:

- ``orig``: the original image.
- ``evidence_masked``: gray-mask the annotated OCR evidence bbox.
- ``random_masked``: gray-mask a same-size random non-evidence region.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit-base-probes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--max-random-iou", type=float, default=0.05)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    base_rows = [row for row in rows if usable_base_probe(row)]
    if args.limit_base_probes is not None:
        base_rows = base_rows[: args.limit_base_probes]

    out: list[dict[str, Any]] = []
    for row in base_rows:
        evidence = normalize_regions(row.get("evidence_regions") or row.get("ocr_regions"))
        random_region = same_size_random_region(
            evidence[0],
            sample_id=str(row.get("sample_id", row.get("id", ""))),
            seed=args.seed,
            max_iou=args.max_random_iou,
        )
        out.extend(make_views(row, evidence=evidence, random_region=random_region))

    write_jsonl(args.output, out)
    print(f"Wrote {len(out)} occlusion probes from {len(base_rows)} base probes to {args.output}")


def usable_base_probe(row: dict[str, Any]) -> bool:
    if str(row.get("probe", "orig")) != "orig":
        return False
    if str(row.get("target_answer", row.get("answer", ""))).lower() not in {"yes", "no"}:
        return False
    return bool(normalize_regions(row.get("evidence_regions") or row.get("ocr_regions")))


def make_views(row: dict[str, Any], *, evidence: list[list[float]], random_region: list[float]) -> list[dict[str, Any]]:
    specs = (
        ("orig", "original", "original", []),
        ("evidence_masked", "evidence_masked", "evidence", evidence),
        ("random_masked", "mask_regions", "random_same_size", [random_region]),
    )
    views: list[dict[str, Any]] = []
    for probe_name, rice_view, occlusion_kind, mask_regions in specs:
        item = dict(row)
        item["probe"] = probe_name
        item["rice_view"] = rice_view
        item["probe_count"] = len(specs)
        item["occlusion_kind"] = occlusion_kind
        item["evidence_regions"] = evidence
        if mask_regions:
            item["mask_regions"] = mask_regions
        else:
            item.pop("mask_regions", None)
        if probe_name == "random_masked":
            item["random_regions"] = [random_region]
        else:
            item.pop("random_regions", None)
        views.append(item)
    return views


def same_size_random_region(
    evidence: list[float],
    *,
    sample_id: str,
    seed: int,
    max_iou: float,
    attempts: int = 200,
) -> list[float]:
    x1, y1, x2, y2 = evidence
    width = max(1e-6, min(1.0, x2 - x1))
    height = max(1e-6, min(1.0, y2 - y1))
    rng = random.Random(stable_seed(f"{seed}:{sample_id}"))
    best = [x1, y1, min(1.0, x1 + width), min(1.0, y1 + height)]
    best_iou = 1.0
    for _ in range(attempts):
        rx1 = rng.uniform(0.0, max(0.0, 1.0 - width))
        ry1 = rng.uniform(0.0, max(0.0, 1.0 - height))
        candidate = [rx1, ry1, rx1 + width, ry1 + height]
        score = iou(candidate, evidence)
        if score < best_iou:
            best = candidate
            best_iou = score
        if score <= max_iou:
            return candidate
    return best


def stable_seed(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def normalize_regions(value: Any) -> list[list[float]]:
    if not isinstance(value, (list, tuple)):
        return []
    out: list[list[float]] = []
    for item in value:
        box = normalize_box(item)
        if box is not None:
            out.append(box)
    return out


def normalize_box(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(coord) for coord in value[:4]]
    except Exception:
        return None
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        return None
    return [x1, y1, x2, y2]


def iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
