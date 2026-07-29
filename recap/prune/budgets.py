"""Budget helpers for visual token pruning experiments."""

from __future__ import annotations

import math
from typing import Iterable


DEFAULT_KEEP_RATIOS = (0.15, 0.25, 0.35, 0.50, 0.70, 1.00)


def parse_keep_ratios(value: str | Iterable[float] | None) -> list[float]:
    """Parse comma-separated keep ratios and validate they are in (0, 1]."""
    if value is None:
        ratios = list(DEFAULT_KEEP_RATIOS)
    elif isinstance(value, str):
        ratios = [float(item.strip()) for item in value.split(",") if item.strip()]
    else:
        ratios = [float(item) for item in value]
    if not ratios:
        raise ValueError("At least one keep ratio is required.")
    for ratio in ratios:
        if ratio <= 0.0 or ratio > 1.0:
            raise ValueError(f"keep_ratio must be in (0, 1], got {ratio}.")
    return ratios


def fixed_keep_count(num_tokens: int, keep_ratio: float) -> int:
    """Return a clipped ceil budget for a fixed keep ratio."""
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be non-negative, got {num_tokens}.")
    if num_tokens == 0:
        return 0
    if keep_ratio <= 0.0 or keep_ratio > 1.0:
        raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}.")
    return max(1, min(num_tokens, int(math.ceil(num_tokens * keep_ratio))))


def risk_adaptive_keep_ratio(risk: float, *, rho_min: float = 0.15, rho_max: float = 0.70) -> float:
    """Map a risk score in [0, 1] to a keep ratio in [rho_min, rho_max]."""
    if rho_min <= 0.0 or rho_min > 1.0:
        raise ValueError(f"rho_min must be in (0, 1], got {rho_min}.")
    if rho_max <= 0.0 or rho_max > 1.0:
        raise ValueError(f"rho_max must be in (0, 1], got {rho_max}.")
    if rho_min > rho_max:
        raise ValueError(f"rho_min cannot exceed rho_max: {rho_min} > {rho_max}.")
    clipped = min(1.0, max(0.0, float(risk)))
    return rho_min + clipped * (rho_max - rho_min)


def risk_bucket_keep_ratio(
    risk: float,
    *,
    rho_low: float = 0.25,
    rho_mid: float = 0.50,
    rho_high: float = 0.70,
    low_threshold: float = 1.0 / 3.0,
    high_threshold: float = 2.0 / 3.0,
) -> float:
    """Map a risk score to a conservative low/mid/high keep-ratio bucket."""
    for name, value in (("rho_low", rho_low), ("rho_mid", rho_mid), ("rho_high", rho_high)):
        if value <= 0.0 or value > 1.0:
            raise ValueError(f"{name} must be in (0, 1], got {value}.")
    if not (rho_low <= rho_mid <= rho_high):
        raise ValueError(f"Risk buckets must be monotonic, got {rho_low}, {rho_mid}, {rho_high}.")
    if low_threshold < 0.0 or high_threshold > 1.0 or low_threshold > high_threshold:
        raise ValueError(
            "Bucket thresholds must satisfy 0 <= low_threshold <= high_threshold <= 1, "
            f"got {low_threshold}, {high_threshold}."
        )
    clipped = min(1.0, max(0.0, float(risk)))
    if clipped < low_threshold:
        return rho_low
    if clipped < high_threshold:
        return rho_mid
    return rho_high


def risk_adaptive_keep_count(
    num_tokens: int,
    risk: float,
    *,
    rho_min: float = 0.15,
    rho_max: float = 0.70,
) -> int:
    """Return a token budget from a clipped risk-adaptive keep ratio."""
    return fixed_keep_count(num_tokens, risk_adaptive_keep_ratio(risk, rho_min=rho_min, rho_max=rho_max))


def risk_bucket_keep_count(
    num_tokens: int,
    risk: float,
    *,
    rho_low: float = 0.25,
    rho_mid: float = 0.50,
    rho_high: float = 0.70,
    low_threshold: float = 1.0 / 3.0,
    high_threshold: float = 2.0 / 3.0,
) -> int:
    """Return a token budget from low/mid/high risk buckets."""
    return fixed_keep_count(
        num_tokens,
        risk_bucket_keep_ratio(
            risk,
            rho_low=rho_low,
            rho_mid=rho_mid,
            rho_high=rho_high,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        ),
    )


def removal_fraction(num_tokens: int, keep_count: int) -> float:
    """Fraction of visual tokens removed by a budget."""
    if num_tokens <= 0:
        return 0.0
    keep_count = max(0, min(num_tokens, int(keep_count)))
    return 1.0 - keep_count / float(num_tokens)


def summarize_budgets(keep_counts: Iterable[int], full_counts: Iterable[int]) -> dict[str, float]:
    """Summarize per-sample budgets for audit reports."""
    pairs = [(int(k), int(n)) for k, n in zip(keep_counts, full_counts)]
    if not pairs:
        return {
            "num_samples": 0.0,
            "mean_keep_count": 0.0,
            "mean_keep_ratio": 0.0,
            "mean_removal_fraction": 0.0,
        }
    keep_total = sum(k for k, _ in pairs)
    full_total = sum(n for _, n in pairs)
    ratios = [k / n for k, n in pairs if n > 0]
    removals = [removal_fraction(n, k) for k, n in pairs]
    return {
        "num_samples": float(len(pairs)),
        "mean_keep_count": keep_total / float(len(pairs)),
        "mean_keep_ratio": sum(ratios) / float(len(ratios)) if ratios else 0.0,
        "mean_removal_fraction": sum(removals) / float(len(removals)) if removals else 0.0,
        "total_keep_tokens": float(keep_total),
        "total_full_tokens": float(full_total),
    }
