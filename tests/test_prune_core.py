import json

import pytest

from recap.qwen_pruned_backend import PruneConfig
from recap.prune.budgets import fixed_keep_count, risk_adaptive_keep_count, risk_bucket_keep_count, risk_bucket_keep_ratio
from recap.prune.metrics import (
    evidence_center_recall,
    evidence_coverage,
    evidence_patch_recall,
    evidence_regions_from_sample,
    exhaustive_merge_source_indices,
    make_token_grid,
    saa_lite,
    union_area,
)
from recap.prune.offline import build_offline_prune_baselines, summarize_offline_prune_records
from recap.prune.positions import pruned_position_ids, validate_position_mode
from recap.prune.selectors import (
    center_indices,
    coverage_greedy_indices,
    grid_indices,
    hybrid_indices,
    bottomk_indices,
    protected_center_topk_indices,
    protected_topk_indices,
    protected_relation_hybrid_indices,
    random_indices,
    rise_indices,
    soft_evidence_topk_indices,
    soft_relation_hybrid_indices,
    spatial_aware_indices,
)
from recap.prune.saturation import SaturationConfig, evidence_saturation_decision
from recap.llava_pruned_backend import LlavaPruneConfig, _coin_select_indices, _scope_select_indices, _visionzip_budget_split
from recap.qwen_pruned_backend import _target_text_from_probe, _target_text_positions, _target_texts_from_probe
from scripts.analyze_open_qa_selector_comparison import compare_runs
from scripts.analyze_locked_confirmation_human_qc import binary_auroc
from scripts.audit_open_ocr_qa_annotation_agreement import region_set_overlap
from scripts.promote_open_ocr_qa_primary_annotations import is_smoke_label
from scripts.audit_problem_md_manual_evidence_readiness import is_smoke_like_annotation
from scripts.run_qwen_open_ocr_qa_generation import (
    extract_focus_terms,
    load_reused_full_rows,
    order_completed_rows,
    selector_probe,
)
from scripts.run_llava_open_ocr_qa_generation import (
    SUPPORTED_SELECTORS as LLAVA_OPEN_QA_SELECTORS,
    contextual_newline_target_positions,
    subsequence_starts,
)
from scripts.build_ecr_construct_validity_audit import partial_spearman
from scripts.build_secondary_extension_agreement import build_label_type_summary


def test_fixed_and_risk_budgets_are_clipped_and_ceil():
    assert fixed_keep_count(10, 0.25) == 3
    assert fixed_keep_count(10, 1.0) == 10
    assert fixed_keep_count(1, 0.15) == 1
    assert risk_adaptive_keep_count(100, 0.0, rho_min=0.2, rho_max=0.8) == 20
    assert risk_adaptive_keep_count(100, 1.0, rho_min=0.2, rho_max=0.8) == 80
    assert risk_bucket_keep_ratio(0.10, rho_low=0.25, rho_mid=0.50, rho_high=0.70) == 0.25
    assert risk_bucket_keep_ratio(0.50, rho_low=0.25, rho_mid=0.50, rho_high=0.70) == 0.50
    assert risk_bucket_keep_ratio(0.90, rho_low=0.25, rho_mid=0.50, rho_high=0.70) == 0.70
    assert risk_bucket_keep_count(100, 0.90, rho_low=0.25, rho_mid=0.50, rho_high=0.70) == 70


def test_pruned_position_ids_support_compact_and_preserved_logical_positions():
    import torch

    keep_sequence = torch.tensor([True, False, True, False, True], dtype=torch.bool)
    assert pruned_position_ids(keep_sequence, mode="compact").tolist() == [[0, 1, 2]]
    assert pruned_position_ids(keep_sequence, mode="preserve").tolist() == [[0, 2, 4]]


def test_pruned_position_ids_validate_policy_and_mask_shape():
    import torch

    assert validate_position_mode(" Preserve ") == "preserve"
    with pytest.raises(ValueError, match="Unknown position mode"):
        validate_position_mode("shift")
    with pytest.raises(ValueError, match="one-dimensional boolean"):
        pruned_position_ids(torch.tensor([[True, False]]), mode="compact")


def test_llava_prune_config_accepts_replayed_indices():
    config = LlavaPruneConfig(
        selector="embed_protected_topk",
        keep_ratio=0.4,
        kept_indices_by_sample={"sample": [1, 3, 5]},
        position_mode="preserve",
    )
    assert config.kept_indices_by_sample == {"sample": [1, 3, 5]}
    assert config.position_mode == "preserve"


def test_llava_prune_config_rejects_negative_scope_alpha():
    with pytest.raises(ValueError, match="SCOPE alpha must be non-negative"):
        LlavaPruneConfig(selector="scope", keep_ratio=0.4, scope_alpha=-0.1)


def test_open_qa_checkpoint_salvages_only_a_trailing_incomplete_pair():
    samples = [{"sample_id": str(index)} for index in range(4)]
    rows = [{"sample_id": str(index)} for index in range(2)]
    traces = [{"sample_id": str(index)} for index in range(3)]
    kept_rows, kept_traces = order_completed_rows(
        rows,
        traces,
        samples=samples,
        allow_partial=True,
    )
    assert [row["sample_id"] for row in kept_rows] == ["0", "1"]
    assert [trace["sample_id"] for trace in kept_traces] == ["0", "1"]


def test_open_qa_checkpoint_rejects_a_nonprefix_mismatch():
    samples = [{"sample_id": str(index)} for index in range(4)]
    rows = [{"sample_id": "0"}, {"sample_id": "2"}]
    traces = [{"sample_id": "0"}, {"sample_id": "2"}]
    with pytest.raises(ValueError, match="contiguous dataset prefix"):
        order_completed_rows(rows, traces, samples=samples, allow_partial=True)


def test_scope_pure_coverage_ignores_saliency_values():
    import torch

    features = torch.tensor([[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]])
    first = _scope_select_indices(features, torch.tensor([[1.0, 0.1, 0.2, 0.3]]), keep_count=2, alpha=0.0)
    second = _scope_select_indices(features, torch.tensor([[0.1, 1.0, 0.9, 0.8]]), keep_count=2, alpha=0.0)
    assert first.tolist() == second.tolist()


def test_evidence_saturation_uses_low_budget_for_concentrated_scores():
    boxes = make_token_grid(10)
    relevance = [0.0] * 100
    relevance[44] = 1.0
    relevance[45] = 0.95
    ratio, diagnostics = evidence_saturation_decision(
        relevance,
        boxes,
        config=SaturationConfig(mass_target=0.70, cell_target=0.70),
    )
    assert ratio == 0.30
    assert diagnostics["candidate_diagnostics"][0]["passed"]


def test_evidence_saturation_escalates_flat_scores_to_max_budget():
    boxes = make_token_grid(10)
    ratio, diagnostics = evidence_saturation_decision(
        [0.5] * 100,
        boxes,
        config=SaturationConfig(mass_target=0.80, cell_target=0.80),
    )
    assert ratio == 0.70
    assert diagnostics["score_entropy"] == pytest.approx(1.0)


def test_manual_readiness_smoke_detection_uses_complete_words():
    assert is_smoke_like_annotation({"boxes": [{"label": "answer_value:foo"}], "notes": ""})
    assert not is_smoke_like_annotation(
        {"boxes": [{"label": "answer_value:foods in moderate amounts"}], "notes": ""}
    )


def test_random_selector_is_same_budget_and_deterministic():
    first = random_indices(16, 4, seed=13, salt="sample")
    second = random_indices(16, 4, seed=13, salt="sample")
    other = random_indices(16, 4, seed=13, salt="other")
    assert len(first) == 4
    assert first == second
    assert first != other


def test_llava_open_qa_exposes_matched_random_and_grid_controls():
    assert {"random", "grid"}.issubset(LLAVA_OPEN_QA_SELECTORS)


def test_partial_spearman_removes_monotone_control_confound():
    import numpy as np

    rng = np.random.default_rng(4)
    control = np.arange(40, dtype=float)
    x = control + rng.normal(0.0, 8.0, size=len(control))
    y = control + rng.normal(0.0, 8.0, size=len(control))
    estimate, low, high, valid, varying = partial_spearman(
        x,
        y,
        control[:, None],
        iterations=500,
        seed=7,
    )
    assert abs(estimate) < 0.25
    assert low < estimate < high
    assert valid == 500
    assert varying == 1


def test_grid_and_center_select_different_masks():
    boxes = make_token_grid(4)
    grid = grid_indices(boxes, 4)
    center = center_indices(boxes, 4)
    assert len(grid) == 4
    assert len(center) == 4
    assert grid != center
    assert center == [5, 6, 9, 10]


def test_bottomk_selector_keeps_least_relevant_tokens_for_causal_ablation():
    kept = bottomk_indices([0.9, 0.1, 0.4, 0.0, 0.8], 2)
    assert kept == [1, 3]


def test_evidence_coverage_and_saa_lite_on_token_grid():
    boxes = make_token_grid(2)
    evidence = [(0.0, 0.0, 0.5, 0.5)]
    assert evidence_coverage([0], boxes, evidence) == 1.0
    assert evidence_coverage([1], boxes, evidence) == 0.0
    assert saa_lite(True, 1.0, threshold=0.5) == 1.0
    assert saa_lite(False, 1.0, threshold=0.5) == 0.0


def test_evidence_center_and_patch_recall_on_token_grid():
    boxes = make_token_grid(2)
    tiny_evidence = [(0.10, 0.10, 0.20, 0.20)]
    assert evidence_center_recall([0], boxes, tiny_evidence) == 1.0
    assert evidence_center_recall([1], boxes, tiny_evidence) == 0.0
    assert evidence_patch_recall([0], boxes, tiny_evidence) == 1.0
    assert evidence_patch_recall([1], boxes, tiny_evidence) == 0.0

    spanning_evidence = [(0.25, 0.10, 0.75, 0.20)]
    assert evidence_patch_recall([0], boxes, spanning_evidence) == 0.5
    assert evidence_patch_recall([0, 1], boxes, spanning_evidence) == 1.0


def test_union_area_merges_overlapping_rectangles_exactly():
    boxes = [
        (0.0, 0.0, 0.75, 1.0),
        (0.25, 0.0, 1.0, 1.0),
        (0.0, 0.0, 0.25, 0.25),
    ]
    assert union_area(boxes) == 1.0


def test_evidence_regions_from_subject_object_boxes_are_deduped():
    sample = {
        "subject_bbox": [0.0, 0.0, 0.5, 0.5],
        "object_bbox": [0.0, 0.0, 0.5, 0.5],
    }
    assert evidence_regions_from_sample(sample) == [(0.0, 0.0, 0.5, 0.5)]


def test_rise_selector_uses_coverage_after_dominant_tokens():
    boxes = make_token_grid(4)
    relevance = [1.0] + [0.1] * 15
    uniqueness = [0.0] * 16
    kept = rise_indices(
        token_boxes=boxes,
        keep_count=4,
        relevance=relevance,
        uniqueness=uniqueness,
        dominant_ratio=0.25,
        gamma_coverage=1.0,
    )
    assert 0 in kept
    assert len(kept) == 4
    assert max(kept) >= 10


def test_hybrid_selector_keeps_evidence_context_and_coverage():
    boxes = make_token_grid(4)
    relevance = [1.0, 0.8] + [0.0] * 14
    kept = hybrid_indices(
        token_boxes=boxes,
        keep_count=4,
        relevance=relevance,
        core_ratio=0.50,
        context_ratio=0.25,
    )
    assert 0 in kept
    assert 1 in kept
    assert len(kept) == 4
    assert kept != [0, 1, 2, 3]
    assert max(kept) >= 10


def test_soft_relation_boost_preserves_evidence_core_budget():
    boxes = make_token_grid(4)
    relevance = [1.0, 0.9, 0.8, 0.7] + [0.0] * 12
    kept = soft_relation_hybrid_indices(
        token_boxes=boxes,
        keep_count=8,
        relevance=relevance,
        evidence_regions=[(0.0, 0.0, 0.25, 0.25), (0.75, 0.0, 1.0, 0.25)],
        relation="left_of",
        core_ratio=0.50,
        context_ratio=0.25,
    )
    assert len(kept) == 8
    assert set(range(4)).issubset(kept)


def test_protected_relation_hybrid_keeps_bbox_tokens_before_filling_budget():
    boxes = make_token_grid(4)
    relevance = [0.0] * 16
    relevance[5] = 1.0
    kept = protected_relation_hybrid_indices(
        token_boxes=boxes,
        keep_count=2,
        relevance=relevance,
        evidence_regions=[(0.0, 0.0, 0.25, 0.25), (0.75, 0.75, 1.0, 1.0)],
        relation="left_of",
        core_ratio=1.0,
        context_ratio=0.0,
    )
    assert 0 in kept
    assert 15 in kept
    assert 5 not in kept
    assert len(kept) == 2


def test_protected_relation_hybrid_caps_evidence_budget_by_core_ratio():
    boxes = make_token_grid(4)
    kept = protected_relation_hybrid_indices(
        token_boxes=boxes,
        keep_count=4,
        relevance=[0.0] * 16,
        evidence_regions=[(0.0, 0.0, 0.25, 0.25), (0.75, 0.75, 1.0, 1.0)],
        relation="left_of",
        core_ratio=0.50,
        context_ratio=0.25,
    )
    assert 0 in kept
    assert 15 in kept
    assert len(kept) == 4


def test_protected_relation_hybrid_balances_subject_and_object_boxes():
    boxes = make_token_grid(4)
    kept = protected_relation_hybrid_indices(
        token_boxes=boxes,
        keep_count=2,
        relevance=[0.0] * 16,
        evidence_regions=[(0.0, 0.0, 0.75, 0.75), (0.75, 0.75, 1.0, 1.0)],
        relation="left_of",
        core_ratio=1.0,
        context_ratio=0.0,
    )
    assert 15 in kept
    assert len(kept) == 2


def test_protected_relation_hybrid_truncates_large_evidence_to_budget():
    boxes = make_token_grid(4)
    kept = protected_relation_hybrid_indices(
        token_boxes=boxes,
        keep_count=2,
        relevance=[0.0] * 16,
        evidence_regions=[(0.0, 0.0, 1.0, 1.0)],
        relation="above",
        core_ratio=1.0,
        context_ratio=0.0,
    )
    assert kept == [0, 1]


def test_protected_topk_keeps_bbox_tokens_then_fills_by_relevance():
    boxes = make_token_grid(4)
    relevance = [0.0] * 16
    relevance[5] = 1.0
    relevance[6] = 0.9
    kept = protected_topk_indices(
        token_boxes=boxes,
        keep_count=3,
        relevance=relevance,
        evidence_regions=[(0.0, 0.0, 0.25, 0.25)],
        core_ratio=0.50,
    )
    assert kept == [0, 5, 6]


def test_protected_center_topk_keeps_one_bbox_center_then_fills_by_relevance():
    boxes = make_token_grid(4)
    relevance = [0.0] * 16
    relevance[5] = 1.0
    relevance[6] = 0.9
    kept = protected_center_topk_indices(
        token_boxes=boxes,
        keep_count=3,
        relevance=relevance,
        evidence_regions=[(0.60, 0.60, 0.65, 0.65)],
    )
    assert kept == [5, 6, 10]


def test_soft_evidence_topk_boosts_without_forcing_evidence_tokens():
    boxes = make_token_grid(4)
    relevance = [0.0] * 16
    relevance[5] = 1.0
    relevance[6] = 0.9
    relevance[10] = 0.1
    kept = soft_evidence_topk_indices(
        token_boxes=boxes,
        keep_count=2,
        relevance=relevance,
        evidence_regions=[(0.60, 0.60, 0.65, 0.65)],
        evidence_boost=0.10,
    )
    assert kept == [5, 6]

    kept_with_room = soft_evidence_topk_indices(
        token_boxes=boxes,
        keep_count=3,
        relevance=relevance,
        evidence_regions=[(0.60, 0.60, 0.65, 0.65)],
        evidence_boost=0.10,
    )
    assert kept_with_room == [5, 6, 10]


def test_spatial_aware_keeps_bbox_centers_relation_context_and_coverage():
    boxes = make_token_grid(4)
    kept = spatial_aware_indices(
        token_boxes=boxes,
        keep_count=6,
        relevance=[0.0] * 16,
        evidence_regions=[(0.0, 0.0, 0.25, 0.25), (0.75, 0.0, 1.0, 0.25)],
        relation="left_of",
        core_ratio=0.35,
        context_ratio=0.35,
    )
    assert 0 in kept
    assert 3 in kept
    assert any(idx in kept for idx in {1, 2})
    assert len(kept) == 6
    assert max(kept) >= 12


def test_spatial_aware_falls_back_to_grid_without_evidence_regions():
    boxes = make_token_grid(4)
    kept = spatial_aware_indices(
        token_boxes=boxes,
        keep_count=4,
        relevance=[0.0] * 16,
        evidence_regions=[],
        relation="left_of",
    )
    assert kept == grid_indices(boxes, 4)


def test_coverage_greedy_balances_relevance_and_spatial_coverage():
    boxes = make_token_grid(4)
    relevance = [0.0] * 16
    relevance[0] = 1.0
    relevance[1] = 0.9
    relevance[2] = 0.8
    relevance[3] = 0.7
    kept = coverage_greedy_indices(
        token_boxes=boxes,
        keep_count=4,
        relevance=relevance,
        uniqueness=[0.0] * 16,
        evidence_regions=[],
        grid_weight=6.0,
        uniqueness_weight=0.0,
    )
    assert 0 in kept
    assert len(kept) == 4
    assert kept != [0, 1, 2, 3]
    assert max(kept) >= 8


def test_coverage_greedy_uses_evidence_marginal_gain():
    boxes = make_token_grid(4)
    relevance = [0.0] * 16
    relevance[5] = 1.0
    relevance[10] = 0.9
    kept = coverage_greedy_indices(
        token_boxes=boxes,
        keep_count=1,
        relevance=relevance,
        uniqueness=[0.0] * 16,
        evidence_regions=[(0.60, 0.60, 0.65, 0.65)],
        evidence_weight=2.0,
        grid_weight=0.0,
        uniqueness_weight=0.0,
    )
    assert kept == [10]


def test_visionzip_budget_split_matches_official_64_token_ratio():
    dominant_patch_count, contextual_count = _visionzip_budget_split(576, 64)
    assert dominant_patch_count == 53
    assert contextual_count == 10
    assert dominant_patch_count + contextual_count + 1 == 64

    dominant_patch_count, contextual_count = _visionzip_budget_split(576, 231)
    assert dominant_patch_count + contextual_count + 1 == 231
    assert contextual_count == 36


def test_exhaustive_merge_provenance_distinguishes_lineage_from_anchors():
    anchors = [1, 4, 7]
    assert exhaustive_merge_source_indices(9, anchors, contextual_tokens=1) == list(range(9))
    assert exhaustive_merge_source_indices(9, anchors, contextual_tokens=0) == anchors


def test_scope_selector_matches_official_multiplicative_rule():
    torch = pytest.importorskip("torch")
    features = torch.tensor(
        [[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]]],
        dtype=torch.float32,
    )
    attention = torch.tensor([[0.1, 0.2, 0.9, 0.8]], dtype=torch.float32)
    selected = _scope_select_indices(features, attention, keep_count=3)

    assert selected.shape == (1, 3)
    assert selected[0].tolist() == [2, 3, 1]
    assert len(set(selected[0].tolist())) == 3


def test_scope_selector_clips_budget_and_supports_batches():
    torch = pytest.importorskip("torch")
    features = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)
    attention = torch.tensor([[0.9, 0.2, 0.1], [0.1, 0.2, 0.9]], dtype=torch.float32)
    selected = _scope_select_indices(features, attention, keep_count=9)

    assert selected.shape == (2, 3)
    assert selected[:, 0].tolist() == [0, 2]
    assert all(len(set(row.tolist())) == 3 for row in selected)


def test_coin_selector_starts_with_informativeness_then_adds_coverage():
    torch = pytest.importorskip("torch")
    features = torch.tensor(
        [[[2.0, 0.0], [1.8, 0.1], [0.0, 1.0], [-1.0, 0.0]]],
        dtype=torch.float32,
    )
    text = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]], dtype=torch.float32)
    selected = _coin_select_indices(features, text, keep_count=3, alpha=0.9, beta=0.6)

    assert selected.shape == (1, 3)
    assert selected[0, 0].item() == 0
    assert selected[0, 1].item() in {2, 3}
    assert len(set(selected[0].tolist())) == 3


def test_coin_selector_clips_budget_and_supports_batches():
    torch = pytest.importorskip("torch")
    features = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)
    text = torch.tensor([[[1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]]], dtype=torch.float32)
    selected = _coin_select_indices(features, text, keep_count=9)

    assert selected.shape == (2, 3)
    assert selected[:, 0].tolist() == [0, 2]
    assert all(len(set(row.tolist())) == 3 for row in selected)


def test_offline_prune_baselines_emit_ecr_summary():
    samples = [
        {
            "id": "one",
            "dataset": "toy",
            "relation": "left_of",
            "subject_bbox": [0.0, 0.0, 0.5, 0.5],
            "object_bbox": [0.5, 0.5, 1.0, 1.0],
        }
    ]
    records = build_offline_prune_baselines(
        samples,
        keep_ratios=[0.25],
        selectors=["random", "grid"],
        grid_rows=2,
        seed=13,
    )
    assert len(records) == 2
    assert all(record["kept_visual_tokens"] == 1 for record in records)
    summary = summarize_offline_prune_records(records)
    assert summary["num_records"] == 2
    assert len(summary["groups"]) == 2


def test_target_text_positions_falls_back_to_decoded_span_for_context_bpe():
    import torch

    class FakeTokenizer:
        pieces = {
            1: "<s>",
            15: '"',
            16: "RO",
            17: "BER",
            18: "T",
            19: '"?',
        }

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [999]}

        def decode(self, token_ids, **kwargs):
            return "".join(self.pieces.get(int(token_id), "") for token_id in token_ids)

    input_ids = torch.tensor([[1, 15, 16, 17, 18, 19]])
    labels = torch.full_like(input_ids, -100)
    positions = _target_text_positions(
        input_ids=input_ids,
        labels=labels,
        attention_mask=torch.ones_like(input_ids),
        tokenizer=FakeTokenizer(),
        probe={"target_text": "ROBERT", "target_answer": "yes"},
    )
    assert positions.tolist() == [2, 3, 4]


def test_target_text_positions_unions_multiple_focus_terms():
    import torch

    class FakeTokenizer:
        ids = {"invoice": [2], "date": [4]}

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": self.ids.get(text.strip(), [999])}

        def decode(self, token_ids, **kwargs):
            pieces = {1: "what ", 2: "invoice", 3: " ", 4: "date", 5: "?"}
            return "".join(pieces.get(int(token_id), "") for token_id in token_ids)

    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    positions = _target_text_positions(
        input_ids=input_ids,
        labels=torch.full_like(input_ids, -100),
        attention_mask=torch.ones_like(input_ids),
        tokenizer=FakeTokenizer(),
        probe={"selector_target_texts": ["invoice", "date"]},
    )
    assert positions.tolist() == [1, 3]


def test_open_qa_focus_terms_remove_prompt_boilerplate_without_using_answer():
    assert extract_focus_terms("What is the invoice date shown in this document?") == ["invoice", "date"]
    assert extract_focus_terms("What does the street sign say?") == ["street", "sign"]
    sample = {
        "sample_id": "x",
        "dataset": "textvqa_val_lite",
        "image": None,
        "question": "What does the street sign say?\nAnswer the question using a single word or phrase.",
        "raw_question": "What does the street sign say?",
    }
    probe = selector_probe(sample, target_source="focus")
    assert probe["selector_target_texts"] == ["street", "sign"]
    assert "gold_answers" not in probe


def test_target_text_from_probe_does_not_use_binary_yes_no_as_target_text():
    assert _target_text_from_probe({"target_answer": "yes"}) == ""
    assert _target_text_from_probe({"target_answer": "castle"}) == "castle"
    assert _target_texts_from_probe({"selector_target_texts": ["invoice", "date", "invoice"]}) == [
        "invoice",
        "date",
    ]


def test_load_reused_full_rows_requires_matching_protocol(tmp_path):
    rows_path = tmp_path / "open_ocr_qa_generation.jsonl"
    rows_path.write_text(json.dumps({"sample_id": "sample-1", "full_answer": "42"}) + "\n")
    (tmp_path / "metrics.json").write_text(
        json.dumps({"task": "docvqa_val_lite", "max_new_tokens": 32}) + "\n"
    )
    samples = [{"sample_id": "sample-1"}]
    assert load_reused_full_rows(
        str(rows_path), samples=samples, task="docvqa_val_lite", max_new_tokens=32
    )["sample-1"]["full_answer"] == "42"
    with pytest.raises(ValueError, match="max_new_tokens"):
        load_reused_full_rows(
            str(rows_path), samples=samples, task="docvqa_val_lite", max_new_tokens=16
        )


def test_load_reused_full_rows_rejects_missing_samples(tmp_path):
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text(json.dumps({"sample_id": "sample-1", "full_answer": "42"}) + "\n")
    with pytest.raises(ValueError, match="missing 1 requested samples"):
        load_reused_full_rows(
            str(rows_path),
            samples=[{"sample_id": "sample-1"}, {"sample_id": "sample-2"}],
            task="docvqa_val_lite",
            max_new_tokens=32,
        )


def test_compare_open_qa_runs_is_paired_by_sample_id():
    baseline = [
        {"sample_id": "a", "dataset": "toy", "question_id": "1", "keep_ratio": 0.3, "pruned_score": 0.0},
        {"sample_id": "b", "dataset": "toy", "question_id": "2", "keep_ratio": 0.3, "pruned_score": 0.5},
    ]
    candidate = [
        {"sample_id": "b", "dataset": "toy", "question_id": "2", "keep_ratio": 0.3, "pruned_score": 0.5},
        {"sample_id": "a", "dataset": "toy", "question_id": "1", "keep_ratio": 0.3, "pruned_score": 1.0},
    ]
    result = compare_runs(baseline, candidate, bootstrap_samples=100, permutation_samples=100, seed=13)
    assert result["mean_paired_difference"] == 0.5
    assert (result["wins"], result["losses"], result["ties"]) == (1, 0, 1)


def test_locked_confirmation_auroc_uses_average_ranks_for_ties():
    rows = [
        {"margin": 0.0, "binary_polarity": "negative"},
        {"margin": 1.0, "binary_polarity": "negative"},
        {"margin": 1.0, "binary_polarity": "positive"},
        {"margin": 2.0, "binary_polarity": "positive"},
    ]
    assert binary_auroc(rows) == pytest.approx(0.875)


def test_annotation_union_overlap_is_invariant_to_box_partition():
    whole = [{"x": 0.0, "y": 0.0, "w": 10.0, "h": 4.0, "label": "answer_value"}]
    split = [
        {"x": 0.0, "y": 0.0, "w": 5.0, "h": 4.0, "label": "answer_value"},
        {"x": 5.0, "y": 0.0, "w": 5.0, "h": 4.0, "label": "answer_value"},
    ]
    overlap = region_set_overlap(whole, split)
    assert overlap["iou"] == pytest.approx(1.0)
    assert overlap["a_covered_by_b"] == pytest.approx(1.0)
    assert overlap["b_covered_by_a"] == pytest.approx(1.0)


def test_annotation_label_type_summary_uses_sample_level_presence():
    rows = [
        {"task": "A", "label_types_a": "answer_value;context", "label_types_b": "answer_value"},
        {"task": "A", "label_types_a": "answer_value", "label_types_b": "context"},
    ]
    summary = build_label_type_summary(rows)
    answer = next(row for row in summary if row["scope"] == "all" and row["label_type"] == "answer_value")
    assert answer["primary_present"] == 2
    assert answer["secondary_present"] == 1
    assert answer["both_present"] == 1
    assert answer["positive_jaccard"] == 0.5


def test_llava_contextual_question_span_ignores_newline_token():
    torch = pytest.importorskip("torch")

    class FakeTokenizer:
        def __call__(self, text, add_special_tokens=False):
            assert text == "\nwhat?"
            assert not add_special_tokens
            return {"input_ids": [13, 20, 21]}

        def decode(self, token_ids, **_kwargs):
            return {13: "\n", 20: "what", 21: "?"}[token_ids[0]]

    input_ids = torch.tensor([[1, 13, 20, 21, 2]])
    positions = contextual_newline_target_positions(
        input_ids=input_ids,
        labels=torch.full_like(input_ids, -100),
        attention_mask=torch.ones_like(input_ids),
        tokenizer=FakeTokenizer(),
        target_texts=["what?"],
    )
    assert positions.tolist() == [2, 3]
    assert subsequence_starts([1, 2, 1, 2], [1, 2]) == [0, 2]


def test_smoke_label_detection_uses_complete_words():
    assert is_smoke_label("answer_value:foo")
    assert is_smoke_label("smoke placeholder")
    assert not is_smoke_label("answer_value:foods in moderate amounts")


def test_qwen_embedding_score_configuration_validates_bounds():
    PruneConfig(selector="target_embed_topk", keep_ratio=0.3)
    with pytest.raises(ValueError, match="embedding_relevance_weight"):
        PruneConfig(selector="target_embed_topk", keep_ratio=0.3, embedding_relevance_weight=1.1)
    with pytest.raises(ValueError, match="embedding_query_topk"):
        PruneConfig(selector="target_embed_topk", keep_ratio=0.3, embedding_query_topk=0)
