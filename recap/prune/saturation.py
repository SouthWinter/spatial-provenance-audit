"""Pre-generation visual-budget selection from evidence-score saturation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from recap.prune.budgets import fixed_keep_count
from recap.prune.metrics import Box
from recap.prune.selectors import grid_topk_indices


@dataclass(frozen=True)
class SaturationConfig:
    candidate_ratios: tuple[float, ...] = (0.30, 0.50, 0.70)
    temperature: float = 0.12
    mass_target: float = 0.72
    cell_target: float = 0.75
    active_cell_mass: float = 0.02
    spatial_bins: int = 4
    grid_ratio: float = 0.50


def evidence_saturation_decision(
    relevance: Sequence[float],
    token_boxes: Sequence[Box],
    *,
    config: SaturationConfig | None = None,
) -> tuple[float, dict[str, Any]]:
    """Choose the smallest candidate budget satisfying evidence coverage.

    The decision uses only selector-side scores and token geometry, both
    available before the language-model prefill. It never reads gold answers,
    model outputs, or annotated evidence boxes.
    """
    cfg = config or SaturationConfig()
    if not relevance or len(relevance) != len(token_boxes):
        raise ValueError("relevance and token_boxes must be non-empty and aligned")
    ratios = tuple(sorted({_clip_ratio(value) for value in cfg.candidate_ratios}))
    if not ratios:
        raise ValueError("candidate_ratios must not be empty")
    weights = _softmax_weights(relevance, temperature=cfg.temperature)
    score_entropy = _normalized_entropy(weights)
    cell_masses = _cell_masses(weights, token_boxes, bins=cfg.spatial_bins)
    spatial_entropy = _normalized_entropy([mass for mass in cell_masses if mass > 0.0])
    dispersion = _weighted_spatial_dispersion(weights, token_boxes)
    active_cells = {idx for idx, mass in enumerate(cell_masses) if mass >= cfg.active_cell_mass}

    candidates: list[dict[str, Any]] = []
    selected_ratio: float | None = None
    for ratio in ratios:
        keep_count = fixed_keep_count(len(relevance), ratio)
        kept = grid_topk_indices(
            token_boxes=list(token_boxes),
            keep_count=keep_count,
            relevance=[float(value) for value in relevance],
            grid_ratio=cfg.grid_ratio,
        )
        mass_coverage = sum(weights[idx] for idx in kept)
        kept_cells = {_cell_index(token_boxes[idx], cfg.spatial_bins) for idx in kept}
        active_cell_coverage = (
            len(active_cells & kept_cells) / len(active_cells) if active_cells else 1.0
        )
        passed = mass_coverage >= cfg.mass_target and active_cell_coverage >= cfg.cell_target
        candidates.append(
            {
                "ratio": ratio,
                "keep_count": keep_count,
                "mass_coverage": mass_coverage,
                "active_cell_coverage": active_cell_coverage,
                "passed": passed,
            }
        )
        if passed and selected_ratio is None:
            selected_ratio = ratio
    if selected_ratio is None:
        selected_ratio = ratios[-1]

    return selected_ratio, {
        "selected_ratio": selected_ratio,
        "score_entropy": score_entropy,
        "spatial_entropy": spatial_entropy,
        "spatial_dispersion": dispersion,
        "active_cell_count": len(active_cells),
        "temperature": cfg.temperature,
        "mass_target": cfg.mass_target,
        "cell_target": cfg.cell_target,
        "candidate_diagnostics": candidates,
    }


def _softmax_weights(values: Sequence[float], *, temperature: float) -> list[float]:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    peak = max(float(value) for value in values)
    exps = [math.exp((float(value) - peak) / temperature) for value in values]
    total = sum(exps)
    if total <= 0.0:
        return [1.0 / len(exps) for _ in exps]
    return [value / total for value in exps]


def _normalized_entropy(probabilities: Sequence[float]) -> float:
    positive = [float(value) for value in probabilities if value > 0.0]
    if len(positive) <= 1:
        return 0.0
    entropy = -sum(value * math.log(value) for value in positive)
    return entropy / math.log(len(positive))


def _cell_masses(weights: Sequence[float], token_boxes: Sequence[Box], *, bins: int) -> list[float]:
    masses = [0.0 for _ in range(bins * bins)]
    for weight, box in zip(weights, token_boxes):
        masses[_cell_index(box, bins)] += float(weight)
    return masses


def _cell_index(box: Box, bins: int) -> int:
    cx = (float(box[0]) + float(box[2])) / 2.0
    cy = (float(box[1]) + float(box[3])) / 2.0
    col = min(bins - 1, max(0, int(cx * bins)))
    row = min(bins - 1, max(0, int(cy * bins)))
    return row * bins + col


def _weighted_spatial_dispersion(weights: Sequence[float], token_boxes: Sequence[Box]) -> float:
    centers = [
        ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)
        for box in token_boxes
    ]
    mean_x = sum(weight * center[0] for weight, center in zip(weights, centers))
    mean_y = sum(weight * center[1] for weight, center in zip(weights, centers))
    variance = sum(
        weight * ((center[0] - mean_x) ** 2 + (center[1] - mean_y) ** 2)
        for weight, center in zip(weights, centers)
    )
    return min(1.0, math.sqrt(max(0.0, variance)) / math.sqrt(0.5))


def _clip_ratio(value: float) -> float:
    ratio = float(value)
    if ratio <= 0.0 or ratio > 1.0:
        raise ValueError(f"Invalid candidate ratio: {value}")
    return ratio
