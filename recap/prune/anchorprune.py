"""AnchorPrune selector port pinned to MULTI-cau/AnchorPrune@2e5d965.

The implementation mirrors ``anchorprune/selection.py`` from the official
repository. Model-specific signal extraction lives in the corresponding
backend so this module remains independently parity-testable.
"""

from __future__ import annotations

from dataclasses import dataclass


UPSTREAM_COMMIT = "2e5d965a0e7291e46eeda73d678529d641ef74d2"


@dataclass(frozen=True)
class AnchorPruneConfig:
    k_total: int
    k_min: int
    tau: float = 0.2
    patience: int = 3
    kmax_ratio: float = 0.5


def _as_1d(x, name: str):
    if x.ndim != 1:
        raise ValueError(f"AnchorPrune expects {name!r} to be one-dimensional, got {tuple(x.shape)}.")
    return x


def _as_2d(x, name: str):
    if x.ndim != 2:
        raise ValueError(f"AnchorPrune expects {name!r} to be two-dimensional, got {tuple(x.shape)}.")
    return x


def cosine_novelty_to_set(candidate_features, anchor_features):
    import torch
    import torch.nn.functional as functional

    candidate_features = _as_2d(candidate_features, "candidate_features")
    anchor_features = _as_2d(anchor_features, "anchor_features")
    if anchor_features.shape[0] == 0:
        return torch.ones(candidate_features.shape[0], device=candidate_features.device)
    candidates = functional.normalize(candidate_features.float(), dim=-1)
    anchors = functional.normalize(anchor_features.float(), dim=-1)
    return (1.0 - candidates @ anchors.t()).min(dim=1).values


def adaptive_relevance_anchor(
    relevance,
    features,
    k_total: int,
    k_min: int,
    tau: float = 0.2,
    patience: int = 3,
    kmax_ratio: float = 0.5,
):
    import torch

    relevance = _as_1d(relevance, "relevance")
    features = _as_2d(features, "features")
    if relevance.shape[0] != features.shape[0]:
        raise ValueError("AnchorPrune relevance and features must describe the same token sequence.")
    n_tokens = int(relevance.shape[0])
    if n_tokens == 0 or k_total <= 0:
        return torch.empty(0, dtype=torch.long, device=relevance.device)

    k_total = min(int(k_total), n_tokens)
    k_min_eff = min(max(1, int(k_min)), k_total)
    k_max_eff = min(max(k_min_eff, int(float(kmax_ratio) * k_total)), k_total, n_tokens)
    patience = max(1, int(patience))
    ranked = torch.argsort(relevance.float(), descending=True)
    chosen = k_max_eff

    if k_max_eff > k_min_eff:
        initial_anchor = ranked[:k_min_eff]
        anchor_features = features.index_select(0, initial_anchor)
        novelty_event_count = 0
        for position in range(k_min_eff + 1, k_max_eff + 1):
            candidate = ranked[position - 1]
            candidate_feature = features.index_select(0, candidate.view(1))
            novelty = float(cosine_novelty_to_set(candidate_feature, anchor_features)[0].item())
            if novelty > tau:
                novelty_event_count += 1
            if novelty_event_count >= patience:
                chosen = position
                break
    return ranked[:chosen]


def importance_weighted_expansion(features, importance, k_total: int, preselected=None):
    import torch
    import torch.nn.functional as functional

    features = _as_2d(features, "features")
    importance = _as_1d(importance, "importance")
    if features.shape[0] != importance.shape[0]:
        raise ValueError("AnchorPrune features and importance must describe the same token sequence.")
    device = features.device
    n_tokens = int(features.shape[0])
    if n_tokens == 0 or k_total <= 0:
        return torch.empty(0, dtype=torch.long, device=device)

    k_total = min(int(k_total), n_tokens)
    normalized = functional.normalize(features.float(), dim=-1)
    weights = torch.clamp(importance.float().to(device), min=0.0)
    selected_mask = torch.zeros(n_tokens, dtype=torch.bool, device=device)
    selected: list[int] = []

    if preselected is not None and preselected.numel() > 0:
        preselected = preselected.to(device=device, dtype=torch.long).reshape(-1)
        preselected = preselected[(preselected >= 0) & (preselected < n_tokens)][:k_total]
        selected.extend(int(index) for index in preselected.tolist())
        selected_mask.scatter_(0, preselected, True)
    if not selected:
        first = int(torch.argmax(weights).item())
        selected.append(first)
        selected_mask[first] = True

    selected_tensor = torch.tensor(selected, dtype=torch.long, device=device)
    min_distance = torch.ones(n_tokens, dtype=torch.float32, device=device)
    distance = 1.0 - normalized @ normalized.index_select(0, selected_tensor).t()
    min_distance = torch.minimum(min_distance, distance.min(dim=1).values)
    min_distance[selected_mask] = -1e9

    while len(selected) < k_total:
        score = min_distance * weights
        score[selected_mask] = -1e9
        next_index = int(torch.argmax(score).item())
        selected.append(next_index)
        selected_mask[next_index] = True
        min_distance = torch.minimum(min_distance, 1.0 - normalized @ normalized[next_index])
        min_distance[next_index] = -1e9

    return torch.sort(torch.tensor(selected, dtype=torch.long, device=device)).values


def anchorprune_select(relevance, features, importance, config: AnchorPruneConfig, expansion_features=None):
    anchor = adaptive_relevance_anchor(
        relevance=relevance,
        features=features,
        k_total=config.k_total,
        k_min=config.k_min,
        tau=config.tau,
        patience=config.patience,
        kmax_ratio=config.kmax_ratio,
    )
    selected = importance_weighted_expansion(
        features=features if expansion_features is None else expansion_features,
        importance=importance,
        k_total=config.k_total,
        preselected=anchor,
    )
    return selected, anchor
