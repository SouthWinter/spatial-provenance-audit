from scripts.build_textocr_image_free_report import evaluate


def row(image_id: str, target_yes: bool, margin: float) -> dict:
    return {
        "image_id": image_id,
        "target_yes": target_yes,
        "raw_margin": margin,
    }


def test_evaluate_balanced_pairs() -> None:
    rows = [
        row("a", True, 2.0),
        row("a", False, -1.0),
        row("b", True, -0.5),
        row("b", False, 0.5),
    ]
    metrics = evaluate(rows, threshold=0.0)

    assert metrics["accuracy"] == 0.5
    assert metrics["positive_accuracy"] == 0.5
    assert metrics["hfpr"] == 0.5
    assert metrics["pairwise_positive_margin_win_rate"] == 0.5
    assert metrics["mean_positive_minus_negative_margin"] == 1.0
