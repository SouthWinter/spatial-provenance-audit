"""JSONL record helpers for pruning traces."""

from __future__ import annotations

from typing import Any

from recap.prune.budgets import removal_fraction


def make_prune_trace_record(
    *,
    sample_id: str,
    method: str,
    full_visual_tokens: int,
    kept_indices: list[int],
    risk: float | None = None,
    scores: dict[str, list[float]] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a compact, auditable per-sample pruning trace record."""
    record: dict[str, Any] = {
        "sample_id": str(sample_id),
        "method": str(method),
        "full_visual_tokens": int(full_visual_tokens),
        "kept_visual_tokens": len(kept_indices),
        "removal_fraction": removal_fraction(full_visual_tokens, len(kept_indices)),
        "kept_indices": [int(index) for index in kept_indices],
    }
    if risk is not None:
        record["risk"] = float(risk)
    if scores:
        record["scores"] = scores
    if extra:
        record.update(extra)
    return record
