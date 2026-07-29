from scripts.build_textocr_image_free_controls import (
    assignment_allowed,
    build_collision_free_derangement,
    group_probe_pairs,
    validate_derangement,
)


def make_pair(image_id: str, source: str, decoy: str) -> list[dict]:
    base = {
        "image": f"/tmp/{image_id}.jpg",
        "image_id": image_id,
        "source_text": source,
    }
    return [
        {
            **base,
            "sample_id": f"{image_id}:pos",
            "target_answer": "yes",
            "target_text": source,
        },
        {
            **base,
            "sample_id": f"{image_id}:neg",
            "target_answer": "no",
            "target_text": decoy,
        },
    ]


def token_index(*tokens: str) -> dict[str, set[str]]:
    from scripts.build_textocr_image_free_controls import NORMALIZERS

    return {
        name: {normalizer(token) for token in tokens}
        for name, normalizer in NORMALIZERS.items()
    }


def test_derangement_is_pair_preserving_and_collision_free() -> None:
    rows = (
        make_pair("a", "CONTROL", "CONTROl")
        + make_pair("b", "PRICE", "PR1CE")
        + make_pair("c", "TOTAL", "T0TAL")
        + make_pair("d", "DATE", "D4TE")
    )
    groups = group_probe_pairs(rows)
    regions = {
        "a": token_index("CONTROL", "OTHER"),
        "b": token_index("PRICE", "VALUE"),
        "c": token_index("TOTAL", "AMOUNT"),
        "d": token_index("DATE", "NUMBER"),
    }

    mapping = build_collision_free_derangement(groups, regions, seed=17)
    validate_derangement(mapping, groups, regions)

    assert set(mapping) == set(groups)
    assert set(mapping.values()) == set(groups)
    assert all(source != target for source, target in mapping.items())
    assert all(assignment_allowed(source, target, groups, regions) for source, target in mapping.items())


def test_normalized_collision_is_rejected() -> None:
    rows = make_pair("a", "Café", "Cafe") + make_pair("b", "TOTAL", "T0TAL")
    groups = group_probe_pairs(rows)
    regions = {
        "a": token_index("Café"),
        "b": token_index("CAFE"),
    }

    assert not assignment_allowed("a", "b", groups, regions)
