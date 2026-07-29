#!/usr/bin/env python3
"""Audit whether geometric ECR tracks cached evidence interventions.

The script never runs an MLLM. It joins existing token masks, TextOCR-Hard
regions, bbox-occlusion scores, and deletion/restoration runs. Geometry is
reconstructed from trace metadata so that all reported quantities remain
sample-auditable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recap.prune.metrics import (
    Box,
    box_area,
    evidence_coverage,
    evidence_regions_from_sample,
    intersection_area,
    make_token_grid,
    union_area,
)


OUTPUT = ROOT / "runs/problem_optimization_audit/ecr_construct_validity"

MODEL_SPECS = {
    "Qwen3-VL-8B": {
        "selector": ROOT
        / "runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_topk_0p30_targetfix_802816",
        "full": ROOT
        / "runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_topk_1p00/probe_scores.jsonl",
        "occlusion": ROOT / "runs/bbox_occlusion_qwen_textocr_hard_200/qwen3_8b_direct_802816/probe_scores.jsonl",
        "deletion": ROOT / "runs/textocr_deletion_restoration/qwen_target30_runs",
    },
    "LLaVA-1.5-7B": {
        "selector": ROOT
        / "runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_target_embed_topk_0p40_targetfix",
        "full": ROOT
        / "runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_direct/probe_scores.jsonl",
        "occlusion": ROOT
        / "runs/bbox_occlusion_cross_model_textocr_hard_100/llava15_7b_direct/probe_scores.jsonl",
        "deletion": None,
    },
    "InternVL3.5-8B": {
        "selector": ROOT
        / "runs/internvl_textocr_hard/internvl35_8b_textocr_hard_full1000_target_embed_soft_evidence_topk_0p50_b0p05",
        "full": ROOT
        / "runs/internvl_textocr_hard/internvl35_8b_textocr_hard_full1000_direct/probe_scores.jsonl",
        "occlusion": ROOT
        / "runs/bbox_occlusion_cross_model_textocr_hard_100/internvl35_8b_direct_calibrated/probe_scores.jsonl",
        "deletion": ROOT / "runs/textocr_deletion_restoration/internvl_soft50_runs",
    },
}

METRIC_LABELS = {
    "occlusion_drop": "Original minus evidence-box-masked yes margin",
    "occlusion_specific_drop": "Evidence-box drop minus same-area random-mask drop",
    "deletion_drop": "Selected-prefix minus evidence-deleted yes margin",
    "evidence_restoration_gain": "Full evidence restoration minus evidence-deleted yes margin",
    "restoration_specific_gain": "Evidence restoration minus matched-count random restoration",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument(
        "--sample-rows",
        type=Path,
        help="Recompute statistics from a packaged sample-row CSV instead of raw run directories.",
    )
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_sample_rows(path: Path) -> list[dict[str, Any]]:
    numeric_fields = {
        "full_visual_tokens",
        "kept_visual_tokens",
        "keep_ratio",
        "evidence_area",
        "median_cell_area",
        "box_cell_ratio",
        "ecr",
        "local_provenance_precision",
        "geo_f1",
        "selected_margin",
        "selected_correct",
        "full_margin",
        "full_abs_margin",
        *METRIC_LABELS,
    }
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field in numeric_fields:
            value = row.get(field, "")
            if value != "":
                row[field] = float(value)
            else:
                row.pop(field, None)
    return rows


def keyed(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["sample_id"]): row for row in rows}


def token_boxes_from_trace(model: str, trace: dict[str, Any]) -> tuple[list[Box], str]:
    n = int(trace["full_visual_tokens"])
    if model == "Qwen3-VL-8B":
        raw_h = int(trace["grid_h"])
        raw_w = int(trace["grid_w"])
        merge = int(round(math.sqrt(raw_h * raw_w / n)))
        if merge <= 0 or raw_h % merge or raw_w % merge or (raw_h // merge) * (raw_w // merge) != n:
            raise ValueError(f"Cannot recover Qwen merge grid for {trace['sample_id']}: {raw_h}x{raw_w}, n={n}")
        return make_token_grid(raw_h // merge, raw_w // merge), f"merged_grid_{raw_h // merge}x{raw_w // merge}"

    if model == "LLaVA-1.5-7B":
        h, w = int(trace["grid_h"]), int(trace["grid_w"])
        if h * w != n:
            raise ValueError(f"LLaVA grid mismatch for {trace['sample_id']}: {h}x{w}, n={n}")
        return make_token_grid(h, w), f"grid_{h}x{w}"

    patch_h = int(trace["patch_grid_h"])
    patch_w = int(trace["patch_grid_w"])
    tile_rows = int(trace["tile_rows"])
    tile_cols = int(trace["tile_cols"])
    num_patches = int(trace["num_image_patches"])
    has_thumbnail = bool(trace["has_thumbnail_patch"])
    local = make_token_grid(patch_h, patch_w)
    boxes: list[Box] = []
    tile_count = tile_rows * tile_cols
    for patch_idx in range(tile_count):
        row, col = divmod(patch_idx, tile_cols)
        tile = (col / tile_cols, row / tile_rows, (col + 1) / tile_cols, (row + 1) / tile_rows)
        boxes.extend(map_local_boxes(local, tile))
    if has_thumbnail:
        boxes.extend(local)
    if len(boxes) != n or num_patches != tile_count + int(has_thumbnail):
        raise ValueError(
            f"InternVL geometry mismatch for {trace['sample_id']}: boxes={len(boxes)}, n={n}, "
            f"patches={num_patches}, tiles={tile_count}, thumbnail={has_thumbnail}"
        )
    return boxes, f"tiles_{tile_rows}x{tile_cols}_thumb{int(has_thumbnail)}"


def map_local_boxes(boxes: list[Box], tile: Box) -> list[Box]:
    x1, y1, x2, y2 = tile
    width, height = x2 - x1, y2 - y1
    return [(x1 + b[0] * width, y1 + b[1] * height, x1 + b[2] * width, y1 + b[3] * height) for b in boxes]


def local_provenance_precision(kept: list[int], token_boxes: list[Box], regions: list[Box]) -> float:
    """Precision of retained cells that touch evidence, measured in image area."""
    touching = [
        token_boxes[idx]
        for idx in kept
        if 0 <= idx < len(token_boxes) and any(intersection_area(token_boxes[idx], region) > 0 for region in regions)
    ]
    if not touching:
        return 0.0
    local_cell_area = union_area(touching)
    if local_cell_area <= 0:
        return 0.0
    intersections: list[Box] = []
    for cell in touching:
        for region in regions:
            part = (
                max(cell[0], region[0]),
                max(cell[1], region[1]),
                min(cell[2], region[2]),
                min(cell[3], region[3]),
            )
            if box_area(part) > 0:
                intersections.append(part)
    return min(1.0, union_area(intersections) / local_cell_area)


def harmonic(a: float, b: float) -> float:
    return 0.0 if a + b <= 0 else 2 * a * b / (a + b)


def attach_occlusion(rows: list[dict[str, Any]], path: Path) -> None:
    by_id: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for score in read_jsonl(path):
        if score.get("binary_polarity") == "positive":
            by_id[str(score["sample_id"])][str(score["probe"])] = score
    index = {row["sample_id"]: row for row in rows}
    for sample_id, variants in by_id.items():
        if sample_id not in index or not {"orig", "evidence_masked", "random_masked"}.issubset(variants):
            continue
        orig = float(variants["orig"]["margin"])
        evidence = float(variants["evidence_masked"]["margin"])
        random = float(variants["random_masked"]["margin"])
        index[sample_id]["occlusion_drop"] = orig - evidence
        index[sample_id]["occlusion_specific_drop"] = (orig - evidence) - (orig - random)


def attach_deletion(rows: list[dict[str, Any]], directory: Path | None) -> None:
    if directory is None:
        return
    arms = {}
    for arm in ("selected", "remove_evidence", "restore_evidence_1p00", "restore_random_1p00"):
        arms[arm] = keyed(read_jsonl(directory / arm / "probe_scores.jsonl"))
    index = {row["sample_id"]: row for row in rows}
    common = set(index).intersection(*(set(values) for values in arms.values()))
    for sample_id in common:
        selected = float(arms["selected"][sample_id]["margin"])
        removed = float(arms["remove_evidence"][sample_id]["margin"])
        evidence = float(arms["restore_evidence_1p00"][sample_id]["margin"])
        random = float(arms["restore_random_1p00"][sample_id]["margin"])
        index[sample_id]["deletion_drop"] = selected - removed
        index[sample_id]["evidence_restoration_gain"] = evidence - removed
        index[sample_id]["restoration_specific_gain"] = evidence - random


def build_model_rows(model: str, spec: dict[str, Any]) -> list[dict[str, Any]]:
    selector = Path(spec["selector"])
    traces = keyed(read_jsonl(selector / "prune_traces.jsonl"))
    probes = keyed(read_jsonl(selector / "probes.jsonl"))
    score_rows = keyed(read_jsonl(selector / "probe_scores.jsonl"))
    full_rows = keyed(read_jsonl(Path(spec["full"])))
    rows: list[dict[str, Any]] = []
    for sample_id, probe in probes.items():
        if probe.get("binary_polarity") != "positive":
            continue
        trace = traces[sample_id]
        token_boxes, geometry = token_boxes_from_trace(model, trace)
        regions = evidence_regions_from_sample(probe)
        kept = [int(idx) for idx in trace["kept_indices"]]
        ecr = evidence_coverage(kept, token_boxes, regions)
        stored_ecr = float(trace["ecr"])
        if abs(ecr - stored_ecr) > 2e-6:
            raise ValueError(f"Recomputed ECR differs for {model}/{sample_id}: {ecr} vs {stored_ecr}")
        lpp = local_provenance_precision(kept, token_boxes, regions)
        region_area = union_area(regions)
        cell_areas = [box_area(box) for box in token_boxes]
        median_cell_area = float(np.median(cell_areas))
        rows.append(
            {
                "model": model,
                "sample_id": sample_id,
                "hard_type": probe.get("hard_type", ""),
                "geometry": geometry,
                "full_visual_tokens": int(trace["full_visual_tokens"]),
                "kept_visual_tokens": int(trace["kept_visual_tokens"]),
                "keep_ratio": float(trace["effective_keep_ratio"]),
                "evidence_area": region_area,
                "median_cell_area": median_cell_area,
                "box_cell_ratio": region_area / median_cell_area if median_cell_area > 0 else math.nan,
                "ecr": ecr,
                "local_provenance_precision": lpp,
                "geo_f1": harmonic(ecr, lpp),
                "selected_margin": float(score_rows[sample_id]["margin"]),
                "selected_correct": int(bool(score_rows[sample_id]["correct"])),
                "full_margin": float(full_rows[sample_id]["margin"]),
                "full_abs_margin": abs(float(full_rows[sample_id]["margin"])),
            }
        )
    attach_occlusion(rows, Path(spec["occlusion"]))
    attach_deletion(rows, spec["deletion"])
    return rows


def bootstrap_correlation(
    x: np.ndarray, y: np.ndarray, *, method: str, iterations: int, seed: int
) -> tuple[float, float, float, int]:
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    n = len(x)
    if n < 4 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return math.nan, math.nan, math.nan, 0
    fn = pearsonr if method == "pearson" else spearmanr
    estimate = float(fn(x, y).statistic)
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    batch_size = min(256, iterations)
    for start in range(0, iterations, batch_size):
        size = min(batch_size, iterations - start)
        idx = rng.integers(0, n, size=(size, n))
        bx, by = x[idx], y[idx]
        if method == "spearman":
            bx = rankdata(bx, axis=1, method="average")
            by = rankdata(by, axis=1, method="average")
        bx = bx - bx.mean(axis=1, keepdims=True)
        by = by - by.mean(axis=1, keepdims=True)
        denominator = np.sqrt(np.sum(bx * bx, axis=1) * np.sum(by * by, axis=1))
        valid = denominator > 0
        batch_estimates = np.sum(bx * by, axis=1)[valid] / denominator[valid]
        estimates.extend(float(value) for value in batch_estimates)
    if not estimates:
        return estimate, math.nan, math.nan, 0
    low, high = np.quantile(estimates, [0.025, 0.975])
    return estimate, float(low), float(high), len(estimates)


def correlation_rows(rows: list[dict[str, Any]], iterations: int, seed: int) -> list[dict[str, Any]]:
    output = []
    for model_index, model in enumerate(MODEL_SPECS):
        model_rows = [row for row in rows if row["model"] == model]
        for metric_index, metric in enumerate(METRIC_LABELS):
            paired = [row for row in model_rows if metric in row]
            if not paired:
                continue
            for predictor_index, predictor in enumerate(("ecr", "local_provenance_precision", "geo_f1")):
                x = np.asarray([row[predictor] for row in paired], dtype=float)
                y = np.asarray([row[metric] for row in paired], dtype=float)
                for method_index, method in enumerate(("spearman", "pearson")):
                    local_seed = seed + model_index * 1000 + metric_index * 100 + predictor_index * 10 + method_index
                    estimate, low, high, valid_boot = bootstrap_correlation(
                        x, y, method=method, iterations=iterations, seed=local_seed
                    )
                    output.append(
                        {
                            "model": model,
                            "intervention": metric,
                            "intervention_definition": METRIC_LABELS[metric],
                            "predictor": predictor,
                            "correlation": method,
                            "n": len(paired),
                            "estimate": estimate,
                            "ci_low": low,
                            "ci_high": high,
                            "valid_bootstrap_draws": valid_boot,
                            "estimable": int(math.isfinite(estimate)),
                        }
                    )
    return output


def residualize_ranks(values: np.ndarray, controls: np.ndarray) -> np.ndarray:
    ranked = rankdata(values, method="average")
    ranked_controls = np.column_stack(
        [rankdata(controls[:, idx], method="average") for idx in range(controls.shape[1])]
    )
    design = np.column_stack([np.ones(len(values)), ranked_controls])
    fitted = design @ np.linalg.lstsq(design, ranked, rcond=None)[0]
    return ranked - fitted


def partial_spearman(
    x: np.ndarray,
    y: np.ndarray,
    controls: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float, float, int, int]:
    finite = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(controls), axis=1)
    x, y, controls = x[finite], y[finite], controls[finite]
    varying = np.ptp(controls, axis=0) > 0
    controls = controls[:, varying]
    n = len(x)
    if n < 5 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return math.nan, math.nan, math.nan, 0, int(varying.sum())

    def estimate(sample: np.ndarray | None = None) -> float:
        if sample is None:
            bx, by, bc = x, y, controls
        else:
            bx, by, bc = x[sample], y[sample], controls[sample]
        rx = residualize_ranks(bx, bc) if bc.shape[1] else rankdata(bx, method="average")
        ry = residualize_ranks(by, bc) if bc.shape[1] else rankdata(by, method="average")
        denominator = math.sqrt(float(np.dot(rx, rx) * np.dot(ry, ry)))
        return float(np.dot(rx, ry) / denominator) if denominator > 0 else math.nan

    point = estimate()
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(iterations):
        value = estimate(rng.integers(0, n, size=n))
        if math.isfinite(value):
            estimates.append(value)
    if not estimates:
        return point, math.nan, math.nan, 0, int(varying.sum())
    low, high = np.quantile(estimates, [0.025, 0.975])
    return point, float(low), float(high), len(estimates), int(varying.sum())


def partial_correlation_rows(
    rows: list[dict[str, Any]], iterations: int, seed: int
) -> list[dict[str, Any]]:
    output = []
    outcomes = ("occlusion_specific_drop", "deletion_drop")
    predictors = ("ecr", "local_provenance_precision", "geo_f1")
    control_names = ("log_evidence_area", "log_median_cell_area", "full_margin")
    for model_index, model in enumerate(MODEL_SPECS):
        model_rows = [row for row in rows if row["model"] == model]
        for outcome_index, outcome in enumerate(outcomes):
            paired = [row for row in model_rows if outcome in row]
            if not paired:
                continue
            controls = np.asarray(
                [
                    [
                        math.log(max(float(row["evidence_area"]), 1e-12)),
                        math.log(max(float(row["median_cell_area"]), 1e-12)),
                        float(row["full_margin"]),
                    ]
                    for row in paired
                ],
                dtype=float,
            )
            for predictor_index, predictor in enumerate(predictors):
                point, low, high, valid_boot, varying_controls = partial_spearman(
                    np.asarray([row[predictor] for row in paired], dtype=float),
                    np.asarray([row[outcome] for row in paired], dtype=float),
                    controls,
                    iterations=iterations,
                    seed=seed + model_index * 100 + outcome_index * 10 + predictor_index,
                )
                output.append(
                    {
                        "model": model,
                        "intervention": outcome,
                        "intervention_definition": METRIC_LABELS[outcome],
                        "predictor": predictor,
                        "correlation": "partial_spearman",
                        "controls": ";".join(control_names),
                        "n": len(paired),
                        "estimate": point,
                        "ci_low": low,
                        "ci_high": high,
                        "varying_controls": varying_controls,
                        "valid_bootstrap_draws": valid_boot,
                        "estimable": int(math.isfinite(point)),
                    }
                )
    return output


def stratum_labels(values: list[float]) -> tuple[list[str], dict[str, tuple[float, float]]]:
    array = np.asarray(values, dtype=float)
    unique = np.unique(array[np.isfinite(array)])
    if len(unique) < 3:
        return ["constant" for _ in values], {"constant": (float(np.min(array)), float(np.max(array)))}
    q1, q2 = np.quantile(array, [1 / 3, 2 / 3])
    if q1 == q2:
        ranks = rankdata(array, method="average") / len(array)
        labels = ["small" if rank <= 1 / 3 else "medium" if rank <= 2 / 3 else "large" for rank in ranks]
    else:
        labels = ["small" if value <= q1 else "medium" if value <= q2 else "large" for value in array]
    bounds = {}
    for label in ("small", "medium", "large"):
        selected = array[np.asarray(labels) == label]
        if len(selected):
            bounds[label] = (float(np.min(selected)), float(np.max(selected)))
    return labels, bounds


def bootstrap_mean(values: list[float], iterations: int, seed: int) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    samples = array[rng.integers(0, len(array), size=(iterations, len(array)))].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(array.mean()), float(low), float(high)


def scale_rows(rows: list[dict[str, Any]], iterations: int, seed: int) -> list[dict[str, Any]]:
    output = []
    dimensions = ("evidence_area", "median_cell_area", "box_cell_ratio")
    summaries = ("ecr", "local_provenance_precision", "geo_f1", *METRIC_LABELS.keys())
    counter = 0
    for model in MODEL_SPECS:
        model_rows = [row for row in rows if row["model"] == model]
        for dimension in dimensions:
            labels, bounds = stratum_labels([float(row[dimension]) for row in model_rows])
            for row, label in zip(model_rows, labels):
                row[f"{dimension}_stratum"] = label
            for label in bounds:
                group = [row for row in model_rows if row[f"{dimension}_stratum"] == label]
                for metric in summaries:
                    values = [float(row[metric]) for row in group if metric in row]
                    if not values:
                        continue
                    mean, low, high = bootstrap_mean(values, iterations, seed + counter)
                    counter += 1
                    output.append(
                        {
                            "model": model,
                            "scale_dimension": dimension,
                            "stratum": label,
                            "stratum_min": bounds[label][0],
                            "stratum_max": bounds[label][1],
                            "metric": metric,
                            "n": len(values),
                            "mean": mean,
                            "ci_low": low,
                            "ci_high": high,
                        }
                    )
    return output


def geometry_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for model in MODEL_SPECS:
        model_rows = [row for row in rows if row["model"] == model]
        for geometry in sorted({str(row["geometry"]) for row in model_rows}):
            group = [row for row in model_rows if row["geometry"] == geometry]
            output.append(
                {
                    "model": model,
                    "geometry": geometry,
                    "n": len(group),
                    "visual_tokens_mean": float(np.mean([row["full_visual_tokens"] for row in group])),
                    "cell_area_mean": float(np.mean([row["median_cell_area"] for row in group])),
                    "evidence_area_mean": float(np.mean([row["evidence_area"] for row in group])),
                    "box_cell_ratio_mean": float(np.mean([row["box_cell_ratio"] for row in group])),
                    "ecr_mean": float(np.mean([row["ecr"] for row in group])),
                    "local_provenance_precision_mean": float(
                        np.mean([row["local_provenance_precision"] for row in group])
                    ),
                    "geo_f1_mean": float(np.mean([row["geo_f1"] for row in group])),
                }
            )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    extras = sorted({key for row in rows for key in row}.difference(fields))
    digits = 17 if path.name == "construct_validity_sample_rows.csv" else 12
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + extras)
        writer.writeheader()
        for row in rows:
            stable = {
                key: f"{value:.{digits}g}"
                if isinstance(value, float) and math.isfinite(value)
                else value
                for key, value in row.items()
            }
            writer.writerow(stable)


def fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "NA" if not math.isfinite(number) else f"{number:.{digits}f}"


def write_report(
    path: Path,
    rows: list[dict[str, Any]],
    correlations: list[dict[str, Any]],
    partial_correlations: list[dict[str, Any]],
) -> None:
    lines = [
        "# ECR Construct-Validity Audit",
        "",
        "This is a cached, positive-probe-only analysis. ECR measures retained spatial provenance; it is not treated as causal use. Local provenance precision (LPP) is the fraction of the union area of retained cells touching evidence that lies inside the evidence region. Geo-F1 is the harmonic mean of ECR and LPP.",
        "",
        "## Geometry Summary",
        "",
        "| Model | n | ECR | LPP | Geo-F1 | Occlusion n | Deletion n |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in MODEL_SPECS:
        group = [row for row in rows if row["model"] == model]
        lines.append(
            f"| {model} | {len(group)} | {np.mean([r['ecr'] for r in group]):.3f} | "
            f"{np.mean([r['local_provenance_precision'] for r in group]):.3f} | "
            f"{np.mean([r['geo_f1'] for r in group]):.3f} | "
            f"{sum('occlusion_drop' in r for r in group)} | {sum('deletion_drop' in r for r in group)} |"
        )
    lines.extend(
        [
            "",
            "## ECR Association With Interventions",
            "",
            "Spearman correlations are primary because ECR is bounded and often tied. Intervals are percentile bootstrap intervals over paired samples.",
            "",
            "| Model | Intervention | n | Spearman r (95% CI) |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    primary = [row for row in correlations if row["predictor"] == "ecr" and row["correlation"] == "spearman"]
    for row in primary:
        interval = f"{fmt(row['estimate'])} [{fmt(row['ci_low'])}, {fmt(row['ci_high'])}]"
        lines.append(f"| {row['model']} | {row['intervention']} | {row['n']} | {interval} |")
    lines.extend(
        [
            "",
            "## Conditional Association",
            "",
            "Partial Spearman correlations residualize ranked ECR and ranked intervention effects against log evidence-box area, log median token-cell area, and the Full-prefix yes margin. Constant controls within a model are dropped automatically. These are construct-validity diagnostics, not causal estimates.",
            "",
            "| Model | Intervention | n | Partial Spearman r (95% CI) |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    partial_primary = [row for row in partial_correlations if row["predictor"] == "ecr"]
    for row in partial_primary:
        interval = f"{fmt(row['estimate'])} [{fmt(row['ci_low'])}, {fmt(row['ci_high'])}]"
        lines.append(f"| {row['model']} | {row['intervention']} | {row['n']} | {interval} |")
    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- A positive association supports convergent validity: masks with more annotated-region provenance tend to lose more target support when that region is removed or occluded.",
            "- A null association does not prove ECR invalid. It can arise from restricted ECR range, contextualized receptive fields, weak model use of OCR evidence, or low intervention power.",
            "- LPP and Geo-F1 are geometry diagnostics, not additional selector objectives. They prevent a coarse token cell from receiving the same interpretation as a tightly localized cell solely because both cover the box.",
            "- InternVL thumbnail cells overlap tiled cells by design; union-area calculations avoid double counting. Qwen grids are reconstructed after spatial merging.",
            "- Full sample rows, scale strata, Pearson correlations, and valid bootstrap counts are in the companion CSV files.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.sample_rows:
        rows = read_sample_rows(args.sample_rows)
    else:
        rows = []
        for model, spec in MODEL_SPECS.items():
            rows.extend(build_model_rows(model, spec))
    correlations = correlation_rows(rows, args.bootstrap, args.seed)
    partial_correlations = partial_correlation_rows(rows, args.bootstrap, args.seed + 10_000)
    strata = scale_rows(rows, args.bootstrap, args.seed)
    geometry = geometry_rows(rows)
    write_csv(args.output_dir / "construct_validity_sample_rows.csv", rows)
    write_csv(args.output_dir / "construct_validity_correlations.csv", correlations)
    write_csv(args.output_dir / "construct_validity_partial_correlations.csv", partial_correlations)
    write_csv(args.output_dir / "construct_validity_scale_strata.csv", strata)
    write_csv(args.output_dir / "construct_validity_model_geometry.csv", geometry)
    write_report(
        args.output_dir / "construct_validity_report.md",
        rows,
        correlations,
        partial_correlations,
    )
    print(
        f"Wrote {len(rows)} positive-probe rows, {len(correlations)} marginal, and "
        f"{len(partial_correlations)} partial correlation rows to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
