#!/usr/bin/env python
"""Build token-index files for TextOCR-Hard deletion/restoration experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recap.prune.metrics import evidence_regions_from_sample, intersection_area, make_token_grid


VARIANTS = (
    "selected",
    "remove_evidence",
    "restore_evidence_0p25",
    "restore_evidence_0p50",
    "restore_evidence_1p00",
    "restore_random_0p25",
    "restore_random_0p50",
    "restore_random_1p00",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probes", default="data/textocr_val_hard_probes_500img.jsonl")
    parser.add_argument(
        "--trace",
        default="runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_topk_0p30/prune_traces.jsonl",
    )
    parser.add_argument("--output-dir", default="runs/textocr_deletion_restoration/qwen_target30_indices")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    probes = {sample_id(row): row for row in read_jsonl(args.probes)}
    traces = read_jsonl(args.trace)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, list[dict[str, Any]]] = {variant: [] for variant in VARIANTS}
    summary_rows: list[dict[str, Any]] = []
    for trace in traces:
        sid = sample_id(trace)
        probe = probes.get(sid)
        if probe is None:
            continue
        num_tokens = int(trace.get("full_visual_tokens", 0) or 0)
        grid_h = int(trace.get("grid_h", 0) or 0)
        grid_w = int(trace.get("grid_w", 0) or 0)
        grid_h, grid_w = token_grid_shape(grid_h, grid_w, num_tokens)
        if num_tokens <= 0 or grid_h <= 0 or grid_w <= 0:
            continue
        token_boxes = token_boxes_from_trace(trace, grid_h, grid_w, num_tokens)
        if len(token_boxes) != num_tokens:
            continue
        evidence_regions = evidence_regions_from_sample(probe)
        evidence_indices = {
            idx
            for idx, box in enumerate(token_boxes)
            if any(intersection_area(box, region) > 0.0 for region in evidence_regions)
        }
        selected = sorted({int(idx) for idx in trace.get("kept_indices", []) if 0 <= int(idx) < num_tokens})
        selected_evidence = [idx for idx in selected if idx in evidence_indices]
        selected_non_evidence = [idx for idx in selected if idx not in evidence_indices]
        nonselected_non_evidence = [idx for idx in range(num_tokens) if idx not in set(selected) and idx not in evidence_indices]

        variant_indices = build_variants(
            sid=sid,
            selected=selected,
            selected_evidence=selected_evidence,
            selected_non_evidence=selected_non_evidence,
            nonselected_non_evidence=nonselected_non_evidence,
            seed=args.seed,
        )
        for variant, indices in variant_indices.items():
            outputs[variant].append(
                {
                    "sample_id": sid,
                    "kept_indices": indices,
                    "variant": variant,
                    "source_trace": str(args.trace),
                    "full_visual_tokens": num_tokens,
                    "kept_visual_tokens": len(indices),
                    "selected_evidence_tokens": len(selected_evidence),
                    "selected_non_evidence_tokens": len(selected_non_evidence),
                }
            )
        summary_rows.append(
            {
                "sample_id": sid,
                "full_visual_tokens": num_tokens,
                "selected_tokens": len(selected),
                "selected_evidence_tokens": len(selected_evidence),
                "selected_non_evidence_tokens": len(selected_non_evidence),
                "evidence_grid_tokens": len(evidence_indices),
                "has_selected_evidence": int(bool(selected_evidence)),
            }
        )

    for variant, rows in outputs.items():
        write_jsonl(out_dir / f"{variant}.jsonl", rows)
    write_csv(out_dir / "summary.csv", summary_rows)
    write_report(out_dir / "README.md", summary_rows, out_dir)
    print(f"Wrote {len(summary_rows)} sample masks for {len(outputs)} variants to {out_dir}")


def build_variants(
    *,
    sid: str,
    selected: list[int],
    selected_evidence: list[int],
    selected_non_evidence: list[int],
    nonselected_non_evidence: list[int],
    seed: int,
) -> dict[str, list[int]]:
    out = {
        "selected": selected,
        "remove_evidence": selected_non_evidence,
    }
    for ratio_tag, ratio in (("0p25", 0.25), ("0p50", 0.50), ("1p00", 1.00)):
        restore_count = min(len(selected_evidence), round(len(selected_evidence) * ratio))
        if ratio > 0.0 and selected_evidence and restore_count == 0:
            restore_count = 1
        evidence_restore = selected_evidence[:restore_count]
        out[f"restore_evidence_{ratio_tag}"] = sorted(selected_non_evidence + evidence_restore)

        rng = random.Random(mixed_seed(seed, sid, ratio_tag))
        random_pool = list(nonselected_non_evidence)
        rng.shuffle(random_pool)
        random_restore = random_pool[:restore_count]
        out[f"restore_random_{ratio_tag}"] = sorted(selected_non_evidence + random_restore)
    return out


def token_grid_shape(grid_h: int, grid_w: int, num_tokens: int) -> tuple[int, int]:
    if grid_h * grid_w == num_tokens:
        return grid_h, grid_w
    for merge in (2, 3, 4):
        if grid_h % merge == 0 and grid_w % merge == 0:
            merged_h = grid_h // merge
            merged_w = grid_w // merge
            if merged_h * merged_w == num_tokens:
                return merged_h, merged_w
    return grid_h, grid_w


def mixed_seed(seed: int, sid: str, tag: str) -> int:
    digest = hashlib.sha256(f"{seed}:{sid}:{tag}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def sample_id(row: dict[str, Any]) -> str:
    return str(row.get("sample_id", row.get("id", "")))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
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
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, Any]], out_dir: Path) -> None:
    n = len(rows)
    selected = mean(float(row["selected_tokens"]) for row in rows)
    selected_evidence = mean(float(row["selected_evidence_tokens"]) for row in rows)
    has_selected_evidence = mean(float(row["has_selected_evidence"]) for row in rows)
    evidence_grid = mean(float(row["evidence_grid_tokens"]) for row in rows)
    lines = [
        "# TextOCR-Hard Deletion/Restoration Index Files",
        "",
        f"Samples: {n}",
        "",
        "| Quantity | Mean |",
        "|---|---:|",
        f"| selected tokens | {selected:.2f} |",
        f"| selected evidence-overlapping tokens | {selected_evidence:.2f} |",
        f"| evidence-overlapping grid tokens | {evidence_grid:.2f} |",
        f"| samples with selected evidence tokens | {has_selected_evidence:.3f} |",
        "",
        "## Variants",
        "",
    ]
    for variant in VARIANTS:
        lines.append(f"- `{out_dir / f'{variant}.jsonl'}`")
    lines.extend(
        [
            "",
            "Use these files with a pruned backend that supports `--kept-indices <file>` to replay exact visual token masks. The intended causal curve compares `remove_evidence` against `restore_evidence_*` and `restore_random_*` at matched restored-token counts.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def token_boxes_from_trace(trace: dict[str, Any], grid_h: int, grid_w: int, num_tokens: int) -> list[tuple[float, float, float, float]]:
    """Reconstruct token boxes from a pruning trace.

    Qwen and LLaVA use a single regular grid, while InternVL dynamic tiling
    stores patch-grid and tile-canvas metadata. Reusing that metadata matters:
    otherwise evidence deletion can target the wrong token indices.
    """
    patch_h = int(trace.get("patch_grid_h", 0) or 0)
    patch_w = int(trace.get("patch_grid_w", 0) or 0)
    tile_rows = int(trace.get("tile_rows", 0) or 0)
    tile_cols = int(trace.get("tile_cols", 0) or 0)
    has_thumbnail = bool(trace.get("has_thumbnail_patch", False))
    if patch_h > 0 and patch_w > 0 and tile_rows > 0 and tile_cols > 0:
        local_boxes = make_token_grid(patch_h, patch_w)
        tokens_per_patch = patch_h * patch_w
        if tokens_per_patch > 0 and num_tokens % tokens_per_patch == 0:
            num_patches = num_tokens // tokens_per_patch
            tile_patches = tile_rows * tile_cols
            if num_patches in {tile_patches, tile_patches + 1}:
                boxes: list[tuple[float, float, float, float]] = []
                for patch_idx in range(tile_patches):
                    row = patch_idx // tile_cols
                    col = patch_idx % tile_cols
                    outer = (
                        col / tile_cols,
                        row / tile_rows,
                        (col + 1) / tile_cols,
                        (row + 1) / tile_rows,
                    )
                    boxes.extend(map_local_boxes(local_boxes, outer))
                if has_thumbnail and num_patches == tile_patches + 1:
                    boxes.extend(local_boxes)
                if len(boxes) == num_tokens:
                    return boxes
    return make_token_grid(grid_h, grid_w)


def map_local_boxes(
    local_boxes: list[tuple[float, float, float, float]],
    outer: tuple[float, float, float, float],
) -> list[tuple[float, float, float, float]]:
    ox1, oy1, ox2, oy2 = outer
    ow = ox2 - ox1
    oh = oy2 - oy1
    return [
        (
            ox1 + box[0] * ow,
            oy1 + box[1] * oh,
            ox1 + box[2] * ow,
            oy1 + box[3] * oh,
        )
        for box in local_boxes
    ]


def mean(values: Any) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    main()
