import pytest

from scripts.build_correct_positive_provenance_audit import wilson_interval
from scripts.build_scope_coverage_ablation import (
    expected_calibration_error,
    paired_by_image,
    summarize,
)


def row(sample_id, image_id, polarity, *, correct, prediction, ecr):
    return {
        "sample_id": sample_id,
        "image_id": image_id,
        "binary_polarity": polarity,
        "correct": correct,
        "pred_answer": prediction,
        "prune_ecr": ecr,
        "prune_keep_ratio": 0.4,
        "yes_loss": 1.0 if prediction == "yes" else 2.0,
        "no_loss": 2.0 if prediction == "yes" else 1.0,
    }


def test_summarize_keeps_positive_and_negative_coverage_semantics_separate() -> None:
    rows = [
        row("a-pos", "a", "positive", correct=True, prediction="yes", ecr=0.8),
        row("a-neg", "a", "negative", correct=False, prediction="yes", ecr=0.2),
        row("b-pos", "b", "positive", correct=False, prediction="no", ecr=0.4),
        row("b-neg", "b", "negative", correct=True, prediction="no", ecr=0.6),
    ]

    summary = summarize(rows)

    assert summary["accuracy"] == pytest.approx(0.5)
    assert summary["hfpr"] == pytest.approx(0.5)
    assert summary["positive_ecr"] == pytest.approx(0.6)
    assert summary["negative_source_coverage"] == pytest.approx(0.4)
    assert summary["correct_positive_count"] == 1


def test_paired_values_remain_clustered_by_image() -> None:
    left = [
        row("a-pos", "a", "positive", correct=True, prediction="yes", ecr=0.8),
        row("a-neg", "a", "negative", correct=True, prediction="no", ecr=0.2),
        row("b-pos", "b", "positive", correct=False, prediction="no", ecr=0.4),
    ]
    right = [
        row("a-pos", "a", "positive", correct=False, prediction="no", ecr=0.6),
        row("a-neg", "a", "negative", correct=True, prediction="no", ecr=0.5),
        row("b-pos", "b", "positive", correct=False, prediction="no", ecr=0.4),
    ]

    clusters = paired_by_image(
        left,
        right,
        lambda left_row, right_row: float(bool(left_row["correct"]))
        - float(bool(right_row["correct"])),
    )

    assert clusters == {"a": [1.0, 0.0], "b": [0.0]}


def test_wilson_interval_is_bounded_and_contains_observed_rate() -> None:
    low, high = wilson_interval(5, 10)

    assert 0.0 < low < 0.5 < high < 1.0


def test_expected_calibration_error_is_zero_for_matched_bin_accuracy() -> None:
    assert expected_calibration_error([0.5, 0.5], [1.0, 0.0], bins=2) == pytest.approx(0.0)
