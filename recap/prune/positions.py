"""Position-ID policies for physical visual-token deletion."""

from __future__ import annotations


POSITION_MODES = ("compact", "preserve")


def validate_position_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in POSITION_MODES:
        choices = ", ".join(POSITION_MODES)
        raise ValueError(f"Unknown position mode {mode!r}; expected one of: {choices}.")
    return normalized


def pruned_position_ids(keep_sequence, *, mode: str):
    """Return logical position IDs after deleting tokens from a batch-one sequence."""
    import torch

    normalized = validate_position_mode(mode)
    if keep_sequence.ndim != 1 or keep_sequence.dtype != torch.bool:
        raise ValueError("keep_sequence must be a one-dimensional boolean tensor.")

    kept_count = int(keep_sequence.sum().item())
    if normalized == "compact":
        positions = torch.arange(kept_count, dtype=torch.long, device=keep_sequence.device)
    else:
        full_positions = torch.arange(keep_sequence.shape[0], dtype=torch.long, device=keep_sequence.device)
        positions = full_positions[keep_sequence]
    return positions.unsqueeze(0)
