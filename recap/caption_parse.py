"""Lightweight spatial caption parsing for external RECAP datasets."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from recap.relations import normalize_relation


@dataclass(frozen=True)
class ParsedSpatialCaption:
    relation: str
    subject: str = ""
    object: str = ""


RELATION_PHRASES: tuple[tuple[str, str], ...] = (
    ("at the left side of", "left_of"),
    ("on the left side of", "left_of"),
    ("to the left of", "left_of"),
    ("left side of", "left_of"),
    ("left of", "left_of"),
    ("at the right side of", "right_of"),
    ("on the right side of", "right_of"),
    ("to the right of", "right_of"),
    ("right side of", "right_of"),
    ("right of", "right_of"),
    ("to the behind of", "behind"),
    ("to the front of", "in_front_of"),
    ("to the bottom of", "below"),
    ("to the top of", "above"),
    ("in front of", "in_front_of"),
    ("far away from", "far"),
    ("on top of", "above"),
    ("next to", "near"),
    ("close to", "near"),
    ("inside of", "inside"),
    ("within", "inside"),
    ("inside", "inside"),
    ("underneath", "below"),
    ("contains", "contains"),
    ("holding", "holding"),
    ("facing", "facing"),
    ("occluding", "occluding"),
    ("behind", "behind"),
    ("above", "above"),
    ("below", "below"),
    ("under", "below"),
    ("over", "above"),
    ("near", "near"),
    ("on", "on"),
    ("in", "inside"),
)


def parse_spatial_caption(caption: str, *, relation_hint: str = "", left_right_only: bool = False) -> ParsedSpatialCaption | None:
    relation = normalize_relation(relation_hint)
    phrases = _phrases_for_relation(relation) if relation else RELATION_PHRASES
    parsed = _parse_with_phrases(caption, phrases)
    if parsed is None and relation:
        parsed = ParsedSpatialCaption(relation=relation)
    if parsed is None:
        return None
    if left_right_only and parsed.relation not in {"left_of", "right_of"}:
        return None
    return parsed


def infer_relation_from_caption(caption: str, *, left_right_only: bool = False) -> str:
    parsed = parse_spatial_caption(caption, left_right_only=left_right_only)
    return parsed.relation if parsed is not None else ""


def _phrases_for_relation(relation: str) -> tuple[tuple[str, str], ...]:
    relation = normalize_relation(relation)
    matches = tuple((phrase, rel) for phrase, rel in RELATION_PHRASES if rel == relation)
    if matches:
        return matches
    phrase = relation.replace("_", " ")
    return ((phrase, relation),) if phrase else tuple()


def _parse_with_phrases(caption: str, phrases: Iterable[tuple[str, str]]) -> ParsedSpatialCaption | None:
    text = _clean_caption(caption)
    lowered = text.lower()
    for phrase, relation in sorted(phrases, key=lambda item: len(item[0]), reverse=True):
        pattern = re.compile(rf"\b{re.escape(phrase)}\b", flags=re.IGNORECASE)
        match = pattern.search(lowered)
        if match is None:
            continue
        prefix = text[: match.start()].strip()
        suffix = text[match.end() :].strip()
        subject = _clean_entity(prefix, side="left")
        obj = _clean_entity(suffix, side="right")
        return ParsedSpatialCaption(relation=normalize_relation(relation), subject=subject, object=obj)
    return None


def _clean_caption(caption: str) -> str:
    text = str(caption or "").strip()
    text = text.strip("\"'")
    text = re.sub(r"\s+", " ", text)
    return text.rstrip(" .!?")


def _clean_entity(text: str, *, side: str) -> str:
    text = str(text or "").strip(" ,;:")
    if side == "left":
        text = re.sub(r"\b(is|are|was|were|be|being)$", "", text, flags=re.IGNORECASE).strip()
    else:
        text = re.sub(r"^(is|are|was|were|be|being)\b", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^(the|a|an)\s+", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s+", " ", text)
    return text
