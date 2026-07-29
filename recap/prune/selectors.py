"""Deterministic visual token selection baselines."""

from __future__ import annotations

import hashlib
import math
import random
from typing import Iterable

from recap.prune.metrics import Box, box_area, intersection_area


SELECTOR_NAMES = (
    "random",
    "grid",
    "grid_topk",
    "center",
    "topk",
    "bottomk",
    "shuffled_topk",
    "rise",
    "hybrid",
    "rel_hybrid",
    "soft_relboost",
    "protected_rel_hybrid",
    "protected_topk",
    "protected_center_topk",
    "soft_evidence_topk",
    "spatial_aware",
    "coverage_greedy",
)


def parse_selectors(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        names = ["random", "grid", "center"]
    elif isinstance(value, str):
        names = [item.strip() for item in value.split(",") if item.strip()]
    else:
        names = [str(item).strip() for item in value if str(item).strip()]
    if not names:
        raise ValueError("At least one selector is required.")
    unknown = sorted(set(names) - set(SELECTOR_NAMES))
    if unknown:
        raise ValueError(f"Unknown selectors {unknown}. Use one of: {list(SELECTOR_NAMES)}")
    return names


def select_indices(
    selector: str,
    *,
    num_tokens: int,
    keep_count: int,
    token_boxes: list[Box] | None = None,
    scores: list[float] | None = None,
    relevance: list[float] | None = None,
    uniqueness: list[float] | None = None,
    evidence_regions: list[Box] | None = None,
    relation: str | None = None,
    seed: int = 13,
    salt: str = "",
    hybrid_core_ratio: float = 0.50,
    hybrid_context_ratio: float = 0.25,
    evidence_boost: float = 0.10,
) -> list[int]:
    """Dispatch a selector by name."""
    selector = selector.strip().lower()
    if selector == "random":
        return random_indices(num_tokens, keep_count, seed=seed, salt=salt)
    if selector == "grid":
        return grid_indices(_boxes_or_default(num_tokens, token_boxes), keep_count)
    if selector == "grid_topk":
        return grid_topk_indices(
            token_boxes=_boxes_or_default(num_tokens, token_boxes),
            keep_count=keep_count,
            relevance=_scores_or_default(num_tokens, relevance if relevance is not None else scores),
            grid_ratio=hybrid_core_ratio,
        )
    if selector == "center":
        return center_indices(_boxes_or_default(num_tokens, token_boxes), keep_count)
    if selector == "topk":
        return topk_indices(_scores_or_default(num_tokens, scores), keep_count)
    if selector == "bottomk":
        return bottomk_indices(_scores_or_default(num_tokens, scores), keep_count)
    if selector == "shuffled_topk":
        return shuffled_topk_indices(_scores_or_default(num_tokens, scores), keep_count, seed=seed, salt=salt)
    if selector == "rise":
        return rise_indices(
            token_boxes=_boxes_or_default(num_tokens, token_boxes),
            keep_count=keep_count,
            relevance=_scores_or_default(num_tokens, relevance if relevance is not None else scores),
            uniqueness=_scores_or_default(num_tokens, uniqueness, default=0.0),
        )
    if selector == "hybrid":
        return hybrid_indices(
            token_boxes=_boxes_or_default(num_tokens, token_boxes),
            keep_count=keep_count,
            relevance=_scores_or_default(num_tokens, relevance if relevance is not None else scores),
            core_ratio=hybrid_core_ratio,
            context_ratio=hybrid_context_ratio,
        )
    if selector == "rel_hybrid":
        return relation_hybrid_indices(
            token_boxes=_boxes_or_default(num_tokens, token_boxes),
            keep_count=keep_count,
            relevance=_scores_or_default(num_tokens, relevance if relevance is not None else scores),
            evidence_regions=evidence_regions or [],
            relation=relation or "",
            core_ratio=hybrid_core_ratio,
            context_ratio=hybrid_context_ratio,
        )
    if selector == "soft_relboost":
        return soft_relation_hybrid_indices(
            token_boxes=_boxes_or_default(num_tokens, token_boxes),
            keep_count=keep_count,
            relevance=_scores_or_default(num_tokens, relevance if relevance is not None else scores),
            evidence_regions=evidence_regions or [],
            relation=relation or "",
            core_ratio=hybrid_core_ratio,
            context_ratio=hybrid_context_ratio,
        )
    if selector == "protected_rel_hybrid":
        return protected_relation_hybrid_indices(
            token_boxes=_boxes_or_default(num_tokens, token_boxes),
            keep_count=keep_count,
            relevance=_scores_or_default(num_tokens, relevance if relevance is not None else scores),
            evidence_regions=evidence_regions or [],
            relation=relation or "",
            core_ratio=hybrid_core_ratio,
            context_ratio=hybrid_context_ratio,
        )
    if selector == "protected_topk":
        return protected_topk_indices(
            token_boxes=_boxes_or_default(num_tokens, token_boxes),
            keep_count=keep_count,
            relevance=_scores_or_default(num_tokens, relevance if relevance is not None else scores),
            evidence_regions=evidence_regions or [],
            core_ratio=hybrid_core_ratio,
        )
    if selector == "protected_center_topk":
        return protected_center_topk_indices(
            token_boxes=_boxes_or_default(num_tokens, token_boxes),
            keep_count=keep_count,
            relevance=_scores_or_default(num_tokens, relevance if relevance is not None else scores),
            evidence_regions=evidence_regions or [],
        )
    if selector == "soft_evidence_topk":
        return soft_evidence_topk_indices(
            token_boxes=_boxes_or_default(num_tokens, token_boxes),
            keep_count=keep_count,
            relevance=_scores_or_default(num_tokens, relevance if relevance is not None else scores),
            evidence_regions=evidence_regions or [],
            evidence_boost=evidence_boost,
        )
    if selector == "spatial_aware":
        return spatial_aware_indices(
            token_boxes=_boxes_or_default(num_tokens, token_boxes),
            keep_count=keep_count,
            relevance=_scores_or_default(num_tokens, relevance if relevance is not None else scores),
            evidence_regions=evidence_regions or [],
            relation=relation or "",
            core_ratio=hybrid_core_ratio,
            context_ratio=hybrid_context_ratio,
        )
    if selector == "coverage_greedy":
        return coverage_greedy_indices(
            token_boxes=_boxes_or_default(num_tokens, token_boxes),
            keep_count=keep_count,
            relevance=_scores_or_default(num_tokens, relevance if relevance is not None else scores),
            uniqueness=_scores_or_default(num_tokens, uniqueness, default=0.0),
            evidence_regions=evidence_regions or [],
            evidence_weight=max(0.50, float(evidence_boost)),
        )
    raise ValueError(f"Unknown selector {selector!r}.")


def random_indices(num_tokens: int, keep_count: int, *, seed: int = 13, salt: str = "") -> list[int]:
    keep_count = _clip_keep_count(num_tokens, keep_count)
    rng = random.Random(_mixed_seed(seed, salt))
    return sorted(rng.sample(range(num_tokens), keep_count))


def grid_indices(token_boxes: list[Box], keep_count: int) -> list[int]:
    """Select tokens near an even grid of image targets."""
    num_tokens = len(token_boxes)
    keep_count = _clip_keep_count(num_tokens, keep_count)
    if keep_count == num_tokens:
        return list(range(num_tokens))
    centers = [_center(box) for box in token_boxes]
    side = max(1, int(math.ceil(math.sqrt(keep_count))))
    target_cells = [(row, col) for row in range(side) for col in range(side)][:keep_count]
    target_cell_set = set(target_cells)
    best_by_cell: dict[tuple[int, int], tuple[float, int]] = {}
    for idx, center in enumerate(centers):
        col = min(side - 1, max(0, int(center[0] * side)))
        row = min(side - 1, max(0, int(center[1] * side)))
        cell = (row, col)
        if cell not in target_cell_set:
            continue
        target = ((col + 0.5) / side, (row + 0.5) / side)
        candidate = (_distance2(center, target), idx)
        if cell not in best_by_cell or candidate < best_by_cell[cell]:
            best_by_cell[cell] = candidate

    selected = [best_by_cell[cell][1] for cell in target_cells if cell in best_by_cell]
    used = set(selected)
    if len(selected) < keep_count:
        remaining = [idx for idx in range(num_tokens) if idx not in used]
        remaining.sort(key=lambda idx: (centers[idx][1], centers[idx][0], idx))
        selected.extend(remaining[: keep_count - len(selected)])
    return sorted(selected)


def center_indices(token_boxes: list[Box], keep_count: int) -> list[int]:
    keep_count = _clip_keep_count(len(token_boxes), keep_count)
    ordered = sorted(range(len(token_boxes)), key=lambda idx: (_distance2(_center(token_boxes[idx]), (0.5, 0.5)), idx))
    return sorted(ordered[:keep_count])


def grid_topk_indices(
    *,
    token_boxes: list[Box],
    keep_count: int,
    relevance: list[float],
    grid_ratio: float = 0.50,
) -> list[int]:
    """Keep an even spatial floor, then fill the remaining budget by relevance."""
    num_tokens = len(token_boxes)
    keep_count = _clip_keep_count(num_tokens, keep_count)
    if keep_count == num_tokens:
        return list(range(num_tokens))
    if len(relevance) != num_tokens:
        raise ValueError("relevance must match token_boxes length.")
    if grid_ratio < 0.0 or grid_ratio > 1.0:
        raise ValueError(f"Invalid grid floor ratio: {grid_ratio}.")

    selected: list[int] = []
    selected_set: set[int] = set()
    grid_count = max(1, min(keep_count, int(round(float(grid_ratio) * keep_count))))
    _extend_unique(selected, selected_set, grid_indices(token_boxes, grid_count), keep_count)
    if len(selected) < keep_count:
        relevance_order = sorted(range(num_tokens), key=lambda idx: (-float(relevance[idx]), idx))
        _extend_unique(selected, selected_set, relevance_order, keep_count)
    return sorted(selected[:keep_count])


def topk_indices(scores: list[float], keep_count: int) -> list[int]:
    keep_count = _clip_keep_count(len(scores), keep_count)
    ordered = sorted(range(len(scores)), key=lambda idx: (-float(scores[idx]), idx))
    return sorted(ordered[:keep_count])


def bottomk_indices(scores: list[float], keep_count: int) -> list[int]:
    keep_count = _clip_keep_count(len(scores), keep_count)
    ordered = sorted(range(len(scores)), key=lambda idx: (float(scores[idx]), idx))
    return sorted(ordered[:keep_count])


def shuffled_topk_indices(scores: list[float], keep_count: int, *, seed: int = 13, salt: str = "") -> list[int]:
    keep_count = _clip_keep_count(len(scores), keep_count)
    shuffled = list(scores)
    random.Random(_mixed_seed(seed, salt)).shuffle(shuffled)
    return topk_indices(shuffled, keep_count)


def rise_indices(
    *,
    token_boxes: list[Box],
    keep_count: int,
    relevance: list[float],
    uniqueness: list[float],
    alpha_relevance: float = 1.0,
    beta_uniqueness: float = 0.6,
    gamma_coverage: float = 0.4,
    shortlist_mul: int = 3,
    dominant_ratio: float = 0.7,
) -> list[int]:
    """Lightweight RISE-style relevance/uniqueness/coverage selector."""
    num_tokens = len(token_boxes)
    keep_count = _clip_keep_count(num_tokens, keep_count)
    if keep_count == num_tokens:
        return list(range(num_tokens))
    if len(relevance) != num_tokens or len(uniqueness) != num_tokens:
        raise ValueError("relevance and uniqueness must match token_boxes length.")
    base_scores = [
        alpha_relevance * float(rel) + beta_uniqueness * float(uniq)
        for rel, uniq in zip(relevance, uniqueness)
    ]
    shortlist_count = min(num_tokens, max(keep_count, int(shortlist_mul) * keep_count))
    shortlist = sorted(range(num_tokens), key=lambda idx: (-base_scores[idx], idx))[:shortlist_count]
    dominant_count = max(0, min(keep_count, int(round(dominant_ratio * keep_count))))
    selected = sorted(shortlist, key=lambda idx: (-base_scores[idx], idx))[:dominant_count]
    selected_set = set(selected)
    centers = [_center(box) for box in token_boxes]
    min_distances = _init_min_distances(shortlist, selected, centers)

    while len(selected) < keep_count:
        best = None
        best_gain = -float("inf")
        for idx in shortlist:
            if idx in selected_set:
                continue
            coverage = math.sqrt(min_distances.get(idx, 0.0)) if selected else 0.0
            gain = base_scores[idx] + gamma_coverage * coverage
            if best is None or gain > best_gain or (gain == best_gain and idx < best):
                best = idx
                best_gain = gain
        if best is None:
            break
        selected.append(best)
        selected_set.add(best)
        _update_min_distances(min_distances, shortlist, selected_set, centers, best)
    return sorted(selected)


def hybrid_indices(
    *,
    token_boxes: list[Box],
    keep_count: int,
    relevance: list[float],
    core_ratio: float = 0.50,
    context_ratio: float = 0.25,
) -> list[int]:
    """Select evidence tokens, nearby context, and global coverage tokens."""
    num_tokens = len(token_boxes)
    keep_count = _clip_keep_count(num_tokens, keep_count)
    if keep_count == num_tokens:
        return list(range(num_tokens))
    if len(relevance) != num_tokens:
        raise ValueError("relevance must match token_boxes length.")
    if core_ratio < 0.0 or context_ratio < 0.0 or core_ratio + context_ratio > 1.0:
        raise ValueError(f"Invalid hybrid ratios: core={core_ratio}, context={context_ratio}.")

    selected: list[int] = []
    selected_set: set[int] = set()

    core_count = max(1, min(keep_count, int(round(core_ratio * keep_count))))
    context_count = max(0, min(keep_count - core_count, int(round(context_ratio * keep_count))))
    coverage_count = keep_count - core_count - context_count

    _extend_unique(selected, selected_set, topk_indices(relevance, core_count), keep_count)
    if context_count:
        _extend_unique(
            selected,
            selected_set,
            _context_ring_indices(token_boxes, relevance, selected_set, context_count),
            keep_count,
        )
    if coverage_count:
        _extend_unique(
            selected,
            selected_set,
            _coverage_indices(token_boxes, selected_set, coverage_count),
            keep_count,
        )
    if len(selected) < keep_count:
        _extend_unique(
            selected,
            selected_set,
            _coverage_fill_indices(token_boxes, relevance, selected, keep_count - len(selected)),
            keep_count,
        )
    return sorted(selected[:keep_count])


def relation_hybrid_indices(
    *,
    token_boxes: list[Box],
    keep_count: int,
    relevance: list[float],
    evidence_regions: list[Box],
    relation: str,
    core_ratio: float = 0.50,
    context_ratio: float = 0.25,
) -> list[int]:
    """Hybrid selector with relation-axis context between evidence boxes."""
    num_tokens = len(token_boxes)
    keep_count = _clip_keep_count(num_tokens, keep_count)
    if keep_count == num_tokens:
        return list(range(num_tokens))
    if len(relevance) != num_tokens:
        raise ValueError("relevance must match token_boxes length.")
    if core_ratio < 0.0 or context_ratio < 0.0 or core_ratio + context_ratio > 1.0:
        raise ValueError(f"Invalid hybrid ratios: core={core_ratio}, context={context_ratio}.")

    selected: list[int] = []
    selected_set: set[int] = set()
    core_count = max(1, min(keep_count, int(round(core_ratio * keep_count))))
    context_count = max(0, min(keep_count - core_count, int(round(context_ratio * keep_count))))
    coverage_count = keep_count - core_count - context_count

    _extend_unique(selected, selected_set, topk_indices(relevance, core_count), keep_count)
    if context_count:
        context = _relation_context_indices(token_boxes, relevance, selected_set, context_count, evidence_regions, relation)
        if len(context) < context_count:
            fallback = _context_ring_indices(token_boxes, relevance, selected_set | set(context), context_count - len(context))
            context = [*context, *fallback]
        _extend_unique(selected, selected_set, context, keep_count)
    if coverage_count:
        _extend_unique(selected, selected_set, _coverage_indices(token_boxes, selected_set, coverage_count), keep_count)
    if len(selected) < keep_count:
        _extend_unique(
            selected,
            selected_set,
            _coverage_fill_indices(token_boxes, relevance, selected, keep_count - len(selected)),
            keep_count,
        )
    return sorted(selected[:keep_count])


def soft_relation_hybrid_indices(
    *,
    token_boxes: list[Box],
    keep_count: int,
    relevance: list[float],
    evidence_regions: list[Box],
    relation: str,
    core_ratio: float = 0.50,
    context_ratio: float = 0.25,
    relation_tail_ratio: float = 0.15,
) -> list[int]:
    """Hybrid selector with a small relation-aware boost in the context tail."""
    num_tokens = len(token_boxes)
    keep_count = _clip_keep_count(num_tokens, keep_count)
    if keep_count == num_tokens:
        return list(range(num_tokens))
    if len(relevance) != num_tokens:
        raise ValueError("relevance must match token_boxes length.")
    if core_ratio < 0.0 or context_ratio < 0.0 or core_ratio + context_ratio > 1.0:
        raise ValueError(f"Invalid hybrid ratios: core={core_ratio}, context={context_ratio}.")
    if relation_tail_ratio < 0.0 or relation_tail_ratio > 1.0:
        raise ValueError(f"Invalid relation tail ratio: {relation_tail_ratio}.")

    selected: list[int] = []
    selected_set: set[int] = set()
    core_count = max(1, min(keep_count, int(round(core_ratio * keep_count))))
    context_count = max(0, min(keep_count - core_count, int(round(context_ratio * keep_count))))
    coverage_count = keep_count - core_count - context_count

    _extend_unique(selected, selected_set, topk_indices(relevance, core_count), keep_count)
    if context_count:
        context = _soft_relation_context_indices(
            token_boxes,
            relevance,
            selected_set,
            context_count,
            evidence_regions,
            relation,
            relation_tail_ratio=relation_tail_ratio,
        )
        _extend_unique(selected, selected_set, context, keep_count)
    if coverage_count:
        _extend_unique(selected, selected_set, _coverage_indices(token_boxes, selected_set, coverage_count), keep_count)
    if len(selected) < keep_count:
        _extend_unique(
            selected,
            selected_set,
            _coverage_fill_indices(token_boxes, relevance, selected, keep_count - len(selected)),
            keep_count,
        )
    return sorted(selected[:keep_count])


def protected_relation_hybrid_indices(
    *,
    token_boxes: list[Box],
    keep_count: int,
    relevance: list[float],
    evidence_regions: list[Box],
    relation: str,
    core_ratio: float = 0.50,
    context_ratio: float = 0.25,
) -> list[int]:
    """Protect capped bbox evidence tokens, then add relation/context/coverage."""
    num_tokens = len(token_boxes)
    keep_count = _clip_keep_count(num_tokens, keep_count)
    if keep_count == num_tokens:
        return list(range(num_tokens))
    if len(relevance) != num_tokens:
        raise ValueError("relevance must match token_boxes length.")
    if core_ratio < 0.0 or context_ratio < 0.0 or core_ratio + context_ratio > 1.0:
        raise ValueError(f"Invalid hybrid ratios: core={core_ratio}, context={context_ratio}.")
    if not evidence_regions:
        return relation_hybrid_indices(
            token_boxes=token_boxes,
            keep_count=keep_count,
            relevance=relevance,
            evidence_regions=evidence_regions,
            relation=relation,
            core_ratio=core_ratio,
            context_ratio=context_ratio,
        )

    selected: list[int] = []
    selected_set: set[int] = set()
    protected_budget = max(1, min(keep_count, int(round(core_ratio * keep_count))))
    protected = _balanced_evidence_protected_indices(token_boxes, evidence_regions, relevance, protected_budget)
    _extend_unique(selected, selected_set, protected, keep_count)

    remaining = keep_count - len(selected)
    relation_count = max(0, min(remaining, int(round(context_ratio * keep_count))))
    if relation_count:
        context = _relation_context_indices(token_boxes, relevance, selected_set, relation_count, evidence_regions, relation)
        if len(context) < relation_count:
            fallback = _context_ring_indices(
                token_boxes,
                relevance,
                selected_set | set(context),
                relation_count - len(context),
            )
            context = [*context, *fallback]
        _extend_unique(selected, selected_set, context, keep_count)

    remaining = keep_count - len(selected)
    if remaining:
        _extend_unique(selected, selected_set, _coverage_indices(token_boxes, selected_set, remaining), keep_count)
    if len(selected) < keep_count:
        _extend_unique(
            selected,
            selected_set,
            _coverage_fill_indices(token_boxes, relevance, selected, keep_count - len(selected)),
            keep_count,
        )
    return sorted(selected[:keep_count])


def protected_topk_indices(
    *,
    token_boxes: list[Box],
    keep_count: int,
    relevance: list[float],
    evidence_regions: list[Box],
    core_ratio: float = 0.50,
) -> list[int]:
    """Protect capped bbox evidence tokens, then fill remaining slots by relevance."""
    num_tokens = len(token_boxes)
    keep_count = _clip_keep_count(num_tokens, keep_count)
    if keep_count == num_tokens:
        return list(range(num_tokens))
    if len(relevance) != num_tokens:
        raise ValueError("relevance must match token_boxes length.")
    if core_ratio < 0.0 or core_ratio > 1.0:
        raise ValueError(f"Invalid protected core ratio: {core_ratio}.")
    if not evidence_regions:
        return topk_indices(relevance, keep_count)

    selected: list[int] = []
    selected_set: set[int] = set()
    protected_budget = max(1, min(keep_count, int(round(core_ratio * keep_count))))
    protected = _balanced_evidence_protected_indices(token_boxes, evidence_regions, relevance, protected_budget)
    _extend_unique(selected, selected_set, protected, keep_count)
    if len(selected) < keep_count:
        relevance_order = sorted(range(num_tokens), key=lambda idx: (-float(relevance[idx]), idx))
        _extend_unique(selected, selected_set, relevance_order, keep_count)
    return sorted(selected[:keep_count])


def protected_center_topk_indices(
    *,
    token_boxes: list[Box],
    keep_count: int,
    relevance: list[float],
    evidence_regions: list[Box],
) -> list[int]:
    """Protect one center token per bbox evidence region, then fill by relevance."""
    num_tokens = len(token_boxes)
    keep_count = _clip_keep_count(num_tokens, keep_count)
    if keep_count == num_tokens:
        return list(range(num_tokens))
    if len(relevance) != num_tokens:
        raise ValueError("relevance must match token_boxes length.")
    if not evidence_regions:
        return topk_indices(relevance, keep_count)

    selected: list[int] = []
    selected_set: set[int] = set()
    protected = _evidence_center_indices(token_boxes, evidence_regions, relevance)
    _extend_unique(selected, selected_set, protected, keep_count)
    if len(selected) < keep_count:
        relevance_order = sorted(range(num_tokens), key=lambda idx: (-float(relevance[idx]), idx))
        _extend_unique(selected, selected_set, relevance_order, keep_count)
    return sorted(selected[:keep_count])


def soft_evidence_topk_indices(
    *,
    token_boxes: list[Box],
    keep_count: int,
    relevance: list[float],
    evidence_regions: list[Box],
    evidence_boost: float = 0.10,
) -> list[int]:
    """Softly boost bbox evidence tokens, then select top-k by the boosted score."""
    num_tokens = len(token_boxes)
    keep_count = _clip_keep_count(num_tokens, keep_count)
    if keep_count == num_tokens:
        return list(range(num_tokens))
    if len(relevance) != num_tokens:
        raise ValueError("relevance must match token_boxes length.")
    if evidence_boost < 0.0:
        raise ValueError(f"Invalid evidence boost: {evidence_boost}.")
    if not evidence_regions or evidence_boost == 0.0:
        return topk_indices(relevance, keep_count)

    normalized = _normalize_scores(relevance)
    evidence_scores = _evidence_soft_scores(token_boxes, evidence_regions)
    boosted = [
        float(rel) + float(evidence_boost) * float(evidence)
        for rel, evidence in zip(normalized, evidence_scores)
    ]
    return topk_indices(boosted, keep_count)


def spatial_aware_indices(
    *,
    token_boxes: list[Box],
    keep_count: int,
    relevance: list[float],
    evidence_regions: list[Box],
    relation: str,
    core_ratio: float = 0.35,
    context_ratio: float = 0.25,
) -> list[int]:
    """Protect spatial anchors, relation context, then fill with global coverage."""
    num_tokens = len(token_boxes)
    keep_count = _clip_keep_count(num_tokens, keep_count)
    if keep_count == num_tokens:
        return list(range(num_tokens))
    if len(relevance) != num_tokens:
        raise ValueError("relevance must match token_boxes length.")
    if core_ratio < 0.0 or context_ratio < 0.0 or core_ratio + context_ratio > 1.0:
        raise ValueError(f"Invalid hybrid ratios: core={core_ratio}, context={context_ratio}.")
    if not evidence_regions:
        return grid_indices(token_boxes, keep_count)

    selected: list[int] = []
    selected_set: set[int] = set()

    anchor_floor = min(keep_count, len(evidence_regions))
    anchor_budget = max(anchor_floor, min(keep_count, int(round(core_ratio * keep_count))))
    centers = _evidence_center_indices(token_boxes, evidence_regions, relevance)
    _extend_unique(selected, selected_set, centers, anchor_budget)
    if len(selected) < anchor_budget:
        protected = _balanced_evidence_protected_indices(token_boxes, evidence_regions, relevance, anchor_budget)
        _extend_unique(selected, selected_set, protected, anchor_budget)

    remaining = keep_count - len(selected)
    relation_count = max(0, min(remaining, int(round(context_ratio * keep_count))))
    if relation_count:
        context = _relation_context_indices(token_boxes, relevance, selected_set, relation_count, evidence_regions, relation)
        if len(context) < relation_count:
            fallback = _context_ring_indices(
                token_boxes,
                relevance,
                selected_set | set(context),
                relation_count - len(context),
            )
            context = [*context, *fallback]
        _extend_unique(selected, selected_set, context, keep_count)

    remaining = keep_count - len(selected)
    if remaining:
        _extend_unique(selected, selected_set, _coverage_indices(token_boxes, selected_set, remaining), keep_count)
    if len(selected) < keep_count:
        _extend_unique(
            selected,
            selected_set,
            _coverage_fill_indices(token_boxes, relevance, selected, keep_count - len(selected)),
            keep_count,
        )
    return sorted(selected[:keep_count])


def coverage_greedy_indices(
    *,
    token_boxes: list[Box],
    keep_count: int,
    relevance: list[float],
    uniqueness: list[float],
    evidence_regions: list[Box],
    target_weight: float = 1.0,
    grid_weight: float = 0.30,
    evidence_weight: float = 0.50,
    uniqueness_weight: float = 0.20,
) -> list[int]:
    """Greedily maximize target, spatial, evidence, and diversity coverage.

    The objective is a training-free budgeted coverage surrogate:
    modular target relevance plus modular uniqueness, coarse grid-cell coverage,
    and clipped per-region evidence coverage. The latter two terms have
    diminishing marginal gains, so repeated tokens in the same cell or evidence
    box become less attractive as the selected set grows.
    """
    num_tokens = len(token_boxes)
    keep_count = _clip_keep_count(num_tokens, keep_count)
    if keep_count == num_tokens:
        return list(range(num_tokens))
    if len(relevance) != num_tokens or len(uniqueness) != num_tokens:
        raise ValueError("relevance and uniqueness must match token_boxes length.")
    if min(target_weight, grid_weight, evidence_weight, uniqueness_weight) < 0.0:
        raise ValueError("coverage_greedy weights must be non-negative.")

    rel = _normalize_scores(relevance)
    uniq = _normalize_scores(uniqueness)
    side = max(1, int(math.ceil(math.sqrt(keep_count))))
    token_cells = [_grid_cell(_center(box), side) for box in token_boxes]
    evidence_progress = [0.0 for _ in evidence_regions]
    evidence_denoms = [max(1e-12, box_area(region)) for region in evidence_regions]
    token_evidence: list[list[tuple[int, float]]] = []
    for token in token_boxes:
        overlaps: list[tuple[int, float]] = []
        for region_idx, (region, denom) in enumerate(zip(evidence_regions, evidence_denoms)):
            overlap = intersection_area(token, region)
            if overlap > 0.0:
                overlaps.append((region_idx, overlap / denom))
        token_evidence.append(overlaps)

    selected: list[int] = []
    selected_set: set[int] = set()
    covered_cells: set[tuple[int, int]] = set()
    num_cells = float(max(1, side * side))

    while len(selected) < keep_count:
        best_idx = None
        best_gain = -float("inf")
        for idx in range(num_tokens):
            if idx in selected_set:
                continue
            cell_gain = 0.0 if token_cells[idx] in covered_cells else 1.0 / num_cells
            evidence_gain = 0.0
            if evidence_regions:
                for region_idx, overlap in token_evidence[idx]:
                    before = min(1.0, evidence_progress[region_idx])
                    after = min(1.0, evidence_progress[region_idx] + overlap)
                    evidence_gain += after - before
                evidence_gain /= float(len(evidence_regions))
            gain = (
                target_weight * rel[idx]
                + uniqueness_weight * uniq[idx]
                + grid_weight * cell_gain
                + evidence_weight * evidence_gain
            )
            if best_idx is None or gain > best_gain or (gain == best_gain and idx < best_idx):
                best_idx = idx
                best_gain = gain
        if best_idx is None:
            break
        selected.append(best_idx)
        selected_set.add(best_idx)
        covered_cells.add(token_cells[best_idx])
        for region_idx, overlap in token_evidence[best_idx]:
            evidence_progress[region_idx] = min(1.0, evidence_progress[region_idx] + overlap)
    return sorted(selected)


def _boxes_or_default(num_tokens: int, token_boxes: list[Box] | None) -> list[Box]:
    if token_boxes is not None:
        if len(token_boxes) != num_tokens:
            raise ValueError(f"token_boxes length {len(token_boxes)} != num_tokens {num_tokens}.")
        return token_boxes
    cols = max(1, int(math.ceil(math.sqrt(num_tokens))))
    rows = max(1, int(math.ceil(num_tokens / cols)))
    boxes: list[Box] = []
    for idx in range(num_tokens):
        row, col = divmod(idx, cols)
        boxes.append((col / cols, row / rows, (col + 1) / cols, min(1.0, (row + 1) / rows)))
    return boxes


def _scores_or_default(num_tokens: int, scores: list[float] | None, *, default: float | None = None) -> list[float]:
    if scores is None:
        if default is None:
            return [1.0 - idx / max(1, num_tokens - 1) for idx in range(num_tokens)]
        return [float(default) for _ in range(num_tokens)]
    if len(scores) != num_tokens:
        raise ValueError(f"scores length {len(scores)} != num_tokens {num_tokens}.")
    return [float(score) for score in scores]


def _normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo = min(float(score) for score in scores)
    hi = max(float(score) for score in scores)
    span = hi - lo
    if span <= 1e-12:
        return [0.0 for _ in scores]
    return [(float(score) - lo) / span for score in scores]


def _clip_keep_count(num_tokens: int, keep_count: int) -> int:
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be non-negative, got {num_tokens}.")
    if num_tokens == 0:
        return 0
    return max(1, min(num_tokens, int(keep_count)))


def _center(box: Box) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _distance2(a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _grid_cell(point: tuple[float, float], side: int) -> tuple[int, int]:
    side = max(1, int(side))
    col = min(side - 1, max(0, int(point[0] * side)))
    row = min(side - 1, max(0, int(point[1] * side)))
    return row, col


def _min_distance2(point: tuple[float, float], others: list[tuple[float, float]]) -> float:
    if not others:
        return 0.0
    return min(_distance2(point, other) for other in others)


def _init_min_distances(
    candidates: list[int],
    selected: list[int],
    centers: list[tuple[float, float]],
) -> dict[int, float]:
    if not selected:
        return {idx: 0.0 for idx in candidates}
    return {idx: min(_distance2(centers[idx], centers[j]) for j in selected) for idx in candidates}


def _update_min_distances(
    min_distances: dict[int, float],
    candidates: list[int],
    selected_set: set[int],
    centers: list[tuple[float, float]],
    new_idx: int,
) -> None:
    new_center = centers[new_idx]
    for idx in candidates:
        if idx in selected_set:
            continue
        distance = _distance2(centers[idx], new_center)
        if distance < min_distances.get(idx, float("inf")):
            min_distances[idx] = distance


def _context_ring_indices(
    token_boxes: list[Box],
    relevance: list[float],
    selected_set: set[int],
    count: int,
) -> list[int]:
    if count <= 0:
        return []
    centers = [_center(box) for box in token_boxes]
    max_rel = max(relevance, default=0.0)
    positive = [idx for idx, rel in enumerate(relevance) if float(rel) > 0.0]
    if positive and len(positive) < len(relevance):
        anchors = positive
    else:
        anchor_count = max(1, min(len(token_boxes), len(selected_set) or count))
        anchors = sorted(range(len(token_boxes)), key=lambda idx: (-float(relevance[idx]), idx))[:anchor_count]
    anchor_centers = [centers[idx] for idx in anchors]
    if max_rel <= 0.0 or not anchor_centers:
        return []
    candidates = [idx for idx in range(len(token_boxes)) if idx not in selected_set]
    ordered = sorted(
        candidates,
        key=lambda idx: (
            _min_distance2(centers[idx], anchor_centers),
            -float(relevance[idx]),
            idx,
        ),
    )
    return ordered[:count]


def _coverage_indices(token_boxes: list[Box], selected_set: set[int], count: int) -> list[int]:
    if count <= 0:
        return []
    selected = [idx for idx in selected_set if 0 <= idx < len(token_boxes)]
    if selected:
        return _farthest_indices(token_boxes, set(selected_set), selected, count)
    return grid_indices(token_boxes, count)


def _coverage_fill_indices(token_boxes: list[Box], relevance: list[float], selected: list[int], count: int) -> list[int]:
    if count <= 0:
        return []
    selected_set = set(selected)
    centers = [_center(box) for box in token_boxes]
    selected_centers = [centers[idx] for idx in selected if 0 <= idx < len(token_boxes)]
    candidates = [idx for idx in range(len(token_boxes)) if idx not in selected_set]
    ordered = sorted(
        candidates,
        key=lambda idx: (
            -float(relevance[idx]),
            -_min_distance2(centers[idx], selected_centers),
            idx,
        ),
    )
    return ordered[:count]


def _relation_context_indices(
    token_boxes: list[Box],
    relevance: list[float],
    selected_set: set[int],
    count: int,
    evidence_regions: list[Box],
    relation: str,
) -> list[int]:
    if count <= 0:
        return []
    corridor = _relation_corridor(evidence_regions, relation)
    if corridor is None:
        return []
    corridor_center = _center(corridor)
    candidates = [idx for idx in range(len(token_boxes)) if idx not in selected_set]
    ordered = sorted(
        candidates,
        key=lambda idx: (
            0 if _boxes_intersect(token_boxes[idx], corridor) else 1,
            _distance_to_box(_center(token_boxes[idx]), corridor),
            _distance2(_center(token_boxes[idx]), corridor_center),
            -float(relevance[idx]),
            idx,
        ),
    )
    return ordered[:count]


def _soft_relation_context_indices(
    token_boxes: list[Box],
    relevance: list[float],
    selected_set: set[int],
    count: int,
    evidence_regions: list[Box],
    relation: str,
    *,
    relation_tail_ratio: float,
) -> list[int]:
    if count <= 0:
        return []
    base_context = _context_ring_indices(token_boxes, relevance, selected_set, count)
    if not base_context:
        return []
    boost_count = int(round(relation_tail_ratio * count))
    if relation_tail_ratio > 0.0:
        boost_count = max(1, boost_count)
    boost_count = min(count, boost_count)
    protected_count = count - boost_count
    protected = base_context[:protected_count]
    used = set(selected_set) | set(protected)
    relation_context = _relation_context_indices(
        token_boxes,
        relevance,
        used,
        boost_count,
        evidence_regions,
        relation,
    )

    context = [*protected]
    context.extend(idx for idx in relation_context if idx not in set(context))
    if len(context) < count:
        context.extend(idx for idx in base_context if idx not in set(context))
    return context[:count]


def _evidence_protected_indices(
    token_boxes: list[Box],
    evidence_regions: list[Box],
    relevance: list[float],
    keep_count: int,
) -> list[int]:
    if keep_count <= 0 or not evidence_regions:
        return []
    ranked: list[tuple[float, float, int]] = []
    for idx, token in enumerate(token_boxes):
        overlap = sum(intersection_area(token, evidence) for evidence in evidence_regions)
        if overlap <= 0.0:
            continue
        token_area = max(1e-12, box_area(token))
        ranked.append((overlap / token_area, float(relevance[idx]), idx))
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [idx for _, _, idx in ranked[:keep_count]]


def _balanced_evidence_protected_indices(
    token_boxes: list[Box],
    evidence_regions: list[Box],
    relevance: list[float],
    keep_count: int,
) -> list[int]:
    if keep_count <= 0 or not evidence_regions:
        return []
    per_region: list[list[int]] = []
    for evidence in evidence_regions:
        ranked: list[tuple[float, float, int]] = []
        for idx, token in enumerate(token_boxes):
            overlap = intersection_area(token, evidence)
            if overlap <= 0.0:
                continue
            token_area = max(1e-12, box_area(token))
            ranked.append((overlap / token_area, float(relevance[idx]), idx))
        ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
        if ranked:
            per_region.append([idx for _, _, idx in ranked])
    selected: list[int] = []
    selected_set: set[int] = set()
    cursors = [0 for _ in per_region]
    while len(selected) < keep_count and per_region:
        changed = False
        for region_idx, candidates in enumerate(per_region):
            while cursors[region_idx] < len(candidates) and candidates[cursors[region_idx]] in selected_set:
                cursors[region_idx] += 1
            if cursors[region_idx] >= len(candidates):
                continue
            idx = candidates[cursors[region_idx]]
            selected.append(idx)
            selected_set.add(idx)
            cursors[region_idx] += 1
            changed = True
            if len(selected) >= keep_count:
                break
        if not changed:
            break
    if len(selected) < keep_count:
        for idx in _evidence_protected_indices(token_boxes, evidence_regions, relevance, keep_count):
            if idx in selected_set:
                continue
            selected.append(idx)
            selected_set.add(idx)
            if len(selected) >= keep_count:
                break
    return selected[:keep_count]


def _evidence_center_indices(
    token_boxes: list[Box],
    evidence_regions: list[Box],
    relevance: list[float],
) -> list[int]:
    selected: list[int] = []
    selected_set: set[int] = set()
    token_centers = [_center(token) for token in token_boxes]
    for evidence in evidence_regions:
        evidence_center = _center(evidence)
        ordered = sorted(
            range(len(token_boxes)),
            key=lambda idx: (
                0 if _point_in_box(token_centers[idx], evidence) else 1,
                _distance2(token_centers[idx], evidence_center),
                -intersection_area(token_boxes[idx], evidence) / max(1e-12, box_area(token_boxes[idx])),
                -float(relevance[idx]),
                idx,
            ),
        )
        for idx in ordered:
            if idx in selected_set:
                continue
            selected.append(idx)
            selected_set.add(idx)
            break
    return selected


def _evidence_soft_scores(
    token_boxes: list[Box],
    evidence_regions: list[Box],
) -> list[float]:
    if not evidence_regions:
        return [0.0 for _ in token_boxes]
    scores: list[float] = []
    for token in token_boxes:
        token_area = max(1e-12, box_area(token))
        token_center = _center(token)
        best = 0.0
        for evidence in evidence_regions:
            overlap = intersection_area(token, evidence) / token_area
            center_bonus = 1.0 if _point_in_box(token_center, evidence) else 0.0
            best = max(best, overlap, center_bonus)
        scores.append(best)
    return scores


def _relation_corridor(evidence_regions: list[Box], relation: str) -> Box | None:
    if len(evidence_regions) < 2:
        return None
    subject, obj = evidence_regions[0], evidence_regions[1]
    relation = relation.lower()
    margin = 0.04
    if relation in {"left_of", "right_of", "left", "right"}:
        left_box, right_box = (subject, obj) if _center(subject)[0] <= _center(obj)[0] else (obj, subject)
        gap_x1, gap_x2 = left_box[2], right_box[0]
        if gap_x2 <= gap_x1:
            gap_x1, gap_x2 = min(subject[0], obj[0]), max(subject[2], obj[2])
        return _clip_box(
            (
                min(subject[0], obj[0], gap_x1) - margin,
                min(subject[1], obj[1]) - margin,
                max(subject[2], obj[2], gap_x2) + margin,
                max(subject[3], obj[3]) + margin,
            )
        )
    if relation in {"above", "below", "over", "under"}:
        top_box, bottom_box = (subject, obj) if _center(subject)[1] <= _center(obj)[1] else (obj, subject)
        gap_y1, gap_y2 = top_box[3], bottom_box[1]
        if gap_y2 <= gap_y1:
            gap_y1, gap_y2 = min(subject[1], obj[1]), max(subject[3], obj[3])
        return _clip_box(
            (
                min(subject[0], obj[0]) - margin,
                min(subject[1], obj[1], gap_y1) - margin,
                max(subject[2], obj[2]) + margin,
                max(subject[3], obj[3], gap_y2) + margin,
            )
        )
    return _clip_box(
        (
            min(subject[0], obj[0]) - margin,
            min(subject[1], obj[1]) - margin,
            max(subject[2], obj[2]) + margin,
            max(subject[3], obj[3]) + margin,
        )
    )


def _clip_box(box: Box) -> Box:
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def _boxes_intersect(a: Box, b: Box) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _point_in_box(point: tuple[float, float], box: Box) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def _distance_to_box(point: tuple[float, float], box: Box) -> float:
    x, y = point
    dx = max(box[0] - x, 0.0, x - box[2])
    dy = max(box[1] - y, 0.0, y - box[3])
    return dx * dx + dy * dy


def _farthest_indices(token_boxes: list[Box], used: set[int], selected: list[int], count: int) -> list[int]:
    centers = [_center(box) for box in token_boxes]
    selected_centers = [centers[idx] for idx in selected if 0 <= idx < len(token_boxes)]
    candidates = [idx for idx in range(len(token_boxes)) if idx not in used]
    min_distances = {
        idx: _min_distance2(centers[idx], selected_centers)
        for idx in candidates
    }
    out: list[int] = []
    while len(out) < count and candidates:
        best = max(candidates, key=lambda idx: (min_distances[idx], -idx))
        used.add(best)
        out.append(best)
        candidates.remove(best)
        best_center = centers[best]
        for idx in candidates:
            distance = _distance2(centers[idx], best_center)
            if distance < min_distances[idx]:
                min_distances[idx] = distance
    return out


def _extend_unique(selected: list[int], selected_set: set[int], candidates: list[int], keep_count: int) -> None:
    for idx in candidates:
        if len(selected) >= keep_count:
            break
        if idx in selected_set:
            continue
        selected_set.add(idx)
        selected.append(idx)


def _mixed_seed(seed: int, salt: str) -> int:
    digest = hashlib.sha1(f"{seed}:{salt}".encode("utf-8")).hexdigest()[:16]
    return int(digest, 16)
