#!/usr/bin/env python3
"""Test whether geometric provenance tracks behavior across confirmation masks."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recap.prune.metrics import box_area, evidence_coverage, evidence_regions_from_sample, union_area
from scripts.build_ecr_construct_validity_audit import (
    harmonic,
    local_provenance_precision,
    token_boxes_from_trace,
)


RUN_ROOT = ROOT / "runs/textocr_confirmation"
OUT_DIR = ROOT / "runs/problem_optimization_audit/confirmation_conditional_construct"
METHODS = {
    "Target": RUN_ROOT / "qwen3_8b_target_0p30",
    "Random": RUN_ROOT / "qwen3_8b_random_0p30",
    "Grid": RUN_ROOT / "qwen3_8b_grid_0p30",
    "VisionZip": RUN_ROOT / "qwen3_8b_visionzip_0p30",
}
FULL = RUN_ROOT / "qwen3_8b_full/probe_scores.jsonl"
BOOTSTRAP = 10_000
SEED = 20260725


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def keyed(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["sample_id"]): row for row in read_jsonl(path)}


def build_rows() -> list[dict[str, Any]]:
    full = keyed(FULL)
    rows: list[dict[str, Any]] = []
    for method, directory in METHODS.items():
        probes = keyed(directory / "probes.jsonl")
        traces = keyed(directory / "prune_traces.jsonl")
        scores = keyed(directory / "probe_scores.jsonl")
        if not (set(probes) == set(traces) == set(scores) == set(full)):
            raise ValueError(f"Sample mismatch for {method}")
        for sample_id, probe in probes.items():
            if probe.get("binary_polarity") != "positive":
                continue
            trace = traces[sample_id]
            token_boxes, geometry = token_boxes_from_trace("Qwen3-VL-8B", trace)
            regions = evidence_regions_from_sample(probe)
            kept = [int(index) for index in trace["kept_indices"]]
            coverage = evidence_coverage(kept, token_boxes, regions)
            stored = float(trace["ecr"])
            if abs(coverage - stored) > 2e-6:
                raise ValueError(f"ECR mismatch for {method}/{sample_id}: {coverage} vs {stored}")
            lpp = local_provenance_precision(kept, token_boxes, regions)
            cell_area = float(np.median([box_area(box) for box in token_boxes]))
            full_margin = float(full[sample_id]["margin"])
            selected_margin = float(scores[sample_id]["margin"])
            rows.append(
                {
                    "image_id": str(probe.get("image_id") or sample_id.split(":")[0]),
                    "sample_id": sample_id,
                    "method": method,
                    "geometry": geometry,
                    "keep_ratio": float(trace["effective_keep_ratio"]),
                    "evidence_area": union_area(regions),
                    "median_cell_area": cell_area,
                    "full_margin": full_margin,
                    "selected_margin": selected_margin,
                    "margin_retention": selected_margin - full_margin,
                    "full_correct": int(bool(full[sample_id]["correct"])),
                    "selected_correct": int(bool(scores[sample_id]["correct"])),
                    "ecr": coverage,
                    "lpp": lpp,
                    "geo_f1": harmonic(coverage, lpp),
                }
            )
    if len(rows) != 2000:
        raise ValueError(f"Expected 2,000 positive method-image rows, found {len(rows)}")
    return rows


def encode_method(rows: list[dict[str, Any]]) -> np.ndarray:
    methods = sorted(METHODS)
    return np.asarray([[float(row["method"] == method) for method in methods[1:]] for row in rows])


def residualize(values: np.ndarray, controls: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(values)), controls])
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def partial_rank_residuals(rows: list[dict[str, Any]], predictor: str) -> tuple[np.ndarray, np.ndarray]:
    x = rankdata(np.asarray([float(row[predictor]) for row in rows]), method="average")
    y = rankdata(np.asarray([float(row["margin_retention"]) for row in rows]), method="average")
    continuous = np.asarray(
        [
            [
                math.log(max(float(row["evidence_area"]), 1e-12)),
                math.log(max(float(row["median_cell_area"]), 1e-12)),
                float(row["full_margin"]),
            ]
            for row in rows
        ]
    )
    ranked_continuous = np.column_stack(
        [rankdata(continuous[:, index], method="average") for index in range(continuous.shape[1])]
    )
    controls = np.column_stack([ranked_continuous, encode_method(rows)])
    return residualize(x, controls), residualize(y, controls)


def two_way_rank_residuals(rows: list[dict[str, Any]], predictor: str) -> tuple[np.ndarray, np.ndarray]:
    x = rankdata(np.asarray([float(row[predictor]) for row in rows]), method="average")
    y = rankdata(np.asarray([float(row["margin_retention"]) for row in rows]), method="average")
    image_labels = np.asarray([row["image_id"] for row in rows])
    method_labels = np.asarray([row["method"] for row in rows])

    def double_demean(values: np.ndarray) -> np.ndarray:
        grand = float(values.mean())
        image_means = {label: float(values[image_labels == label].mean()) for label in set(image_labels)}
        method_means = {label: float(values[method_labels == label].mean()) for label in set(method_labels)}
        return np.asarray(
            [
                value - image_means[image] - method_means[method] + grand
                for value, image, method in zip(values, image_labels, method_labels)
            ]
        )

    return double_demean(x), double_demean(y)


def cluster_bootstrap(
    rows: list[dict[str, Any]], predictor: str, specification: str, seed: int
) -> tuple[float, float, float]:
    ordered = sorted(rows, key=lambda row: (row["image_id"], row["method"]))
    function = partial_rank_residuals if specification == "measured_controls_method_fe" else two_way_rank_residuals
    rx, ry = function(ordered, predictor)
    images = sorted({row["image_id"] for row in ordered})
    methods_per_image = len({row["method"] for row in ordered})
    if len(rx) != len(images) * methods_per_image:
        raise ValueError("The fixed-budget analysis requires one row per image and method")
    rx = rx.reshape(len(images), methods_per_image)
    ry = ry.reshape(len(images), methods_per_image)

    def correlations(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = x - x.mean(axis=1, keepdims=True)
        y = y - y.mean(axis=1, keepdims=True)
        denominator = np.sqrt(np.sum(x * x, axis=1) * np.sum(y * y, axis=1))
        return np.divide(np.sum(x * y, axis=1), denominator, out=np.full(len(x), np.nan), where=denominator > 0)

    point = float(correlations(rx.reshape(1, -1), ry.reshape(1, -1))[0])
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for start in range(0, BOOTSTRAP, 250):
        count = min(250, BOOTSTRAP - start)
        indices = rng.integers(0, len(images), size=(count, len(images)))
        values = correlations(rx[indices].reshape(count, -1), ry[indices].reshape(count, -1))
        draws.extend(float(value) for value in values if math.isfinite(float(value)))
    low, high = np.quantile(draws, [0.025, 0.975])
    return point, float(low), float(high)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = build_rows()
    results = []
    counter = 0
    scopes = (
        ("all_four", rows),
        ("deletion_only", [row for row in rows if row["method"] != "VisionZip"]),
    )
    for scope, scope_rows in scopes:
        for specification in ("measured_controls_method_fe", "image_and_method_fe"):
            for predictor in ("ecr", "lpp", "geo_f1"):
                point, low, high = cluster_bootstrap(scope_rows, predictor, specification, SEED + counter)
                counter += 1
                methods = sorted({row["method"] for row in scope_rows})
                results.append(
                    {
                        "model": "Qwen3-VL-8B",
                        "split": "locked confirmation",
                        "budget": "fixed 30%",
                        "scope": scope,
                        "methods": ";".join(methods),
                        "n_images": 500,
                        "n_method_image_rows": len(scope_rows),
                        "outcome": "selected minus Full yes-margin",
                        "predictor": predictor,
                        "specification": specification,
                        "estimate": point,
                        "ci_low": low,
                        "ci_high": high,
                        "bootstrap": "image-cluster percentile",
                        "draws": BOOTSTRAP,
                    }
                )
    write_csv(OUT_DIR / "conditional_construct_sample_rows.csv", rows)
    write_csv(OUT_DIR / "conditional_construct_correlations.csv", results)

    report = [
        "# Locked-Confirmation Conditional Construct Audit",
        "",
        "This cached analysis joins four Qwen masks at the same 30% visual-token budget on each of "
        "500 locked-confirmation images. The outcome is selected-prefix minus Full-prefix yes-margin "
        "on positive probes; larger values mean better preservation of target support.",
        "",
        "The measured-controls specification residualizes ranked variables against log evidence-box "
        "area, log median token-cell area, Full-prefix margin, and method indicators. The two-way "
        "fixed-effect specification controls every image-level attribute and every method-level shift. "
        "Keep ratio is fixed by design. Intervals resample image clusters and retain all four masks.",
        "",
        "| Scope | Specification | Predictor | Partial rank association (95% CI) |",
        "| --- | --- | --- | ---: |",
    ]
    for row in results:
        report.append(
            f"| {row['scope']} | {row['specification']} | {row['predictor']} | "
            f"{row['estimate']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}] |"
        )
    report.extend(
        [
            "",
            "These are convergent-validity diagnostics, not causal estimates: token-origin geometry "
            "cannot establish what information a contextualized token carries or what the decoder uses.",
            "",
        ]
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "conditional_construct_report.md").write_text("\n".join(report), encoding="utf-8")
    print(OUT_DIR / "conditional_construct_report.md")
    for row in results:
        print(
            f"{row['specification']} {row['predictor']}: {row['estimate']:.3f} "
            f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]"
        )


if __name__ == "__main__":
    main()
