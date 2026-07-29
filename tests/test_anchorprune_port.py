import importlib.util
from pathlib import Path
import sys

import pytest

from recap.prune.anchorprune import (
    AnchorPruneConfig,
    UPSTREAM_COMMIT,
    anchorprune_select,
)


def _load_upstream_module():
    source = Path(__file__).parents[1] / "third_party" / "AnchorPrune" / "anchorprune" / "selection.py"
    if not source.exists():
        pytest.skip("Pinned AnchorPrune checkout is unavailable.")
    spec = importlib.util.spec_from_file_location("anchorprune_upstream_selection", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("seed", [3, 17, 101, 2026])
@pytest.mark.parametrize("budget,minimum", [(16, 3), (32, 5), (64, 10)])
def test_selector_indices_match_pinned_official_implementation(seed: int, budget: int, minimum: int) -> None:
    import torch

    upstream = _load_upstream_module()
    generator = torch.Generator().manual_seed(seed)
    relevance = torch.rand(96, generator=generator)
    anchor_features = torch.randn(96, 32, generator=generator)
    expansion_features = torch.randn(96, 48, generator=generator)
    importance = torch.rand(96, generator=generator)

    ours, ours_anchor = anchorprune_select(
        relevance,
        anchor_features,
        importance,
        AnchorPruneConfig(k_total=budget, k_min=minimum),
        expansion_features=expansion_features,
    )
    theirs, theirs_anchor = upstream.anchorprune_select(
        relevance,
        anchor_features,
        importance,
        upstream.AnchorPruneConfig(k_total=budget, k_min=minimum),
        expansion_features=expansion_features,
    )

    assert UPSTREAM_COMMIT == "2e5d965a0e7291e46eeda73d678529d641ef74d2"
    assert torch.equal(ours_anchor, theirs_anchor)
    assert torch.equal(ours, theirs)
