import pytest

from scripts.build_textocr_hard_probes import (
    DEVELOPMENT_V1,
    image_ids_from_probes,
    replacement_chars,
)


def test_replacement_chars_ignores_non_ascii_digits() -> None:
    assert replacement_chars("⁷") == []


def test_development_v1_preserves_original_unicode_candidate_behavior() -> None:
    assert replacement_chars("ō", construction_version=DEVELOPMENT_V1) == ["Ŏ"]


def test_unknown_construction_version_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown construction version"):
        replacement_chars("a", construction_version="unknown")


def test_image_ids_from_probes_deduplicates_and_ignores_empty_ids() -> None:
    rows = [{"image_id": "a"}, {"image_id": "a"}, {"image_id": "b"}, {}]
    assert image_ids_from_probes(rows) == {"a", "b"}
