"""Relation and answer normalization for RECAP."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

YES_VALUES = {"yes", "y", "true", "1"}
NO_VALUES = {"no", "n", "false", "0"}
LEFT_RIGHT_RELATIONS = {"left_of", "right_of", "left", "right", "to_the_left_of", "to_the_right_of"}
RELATION_FAMILIES: dict[str, set[str]] = {
    "left_right": {"left_of", "right_of"},
    "vertical": {"above", "below", "on"},
    "topology": {"inside", "contains"},
    "proximity": {"near", "far"},
    "depth": {"in_front_of", "behind"},
    "interaction": {"holding", "facing", "occluding"},
}

RELATION_CANDIDATES: dict[str, tuple[str, ...]] = {
    "left_right": ("left_of", "right_of"),
    "vertical": ("above", "below", "on"),
    "topology": ("inside", "contains", "on"),
    "proximity": ("near", "far"),
    "depth": ("in_front_of", "behind"),
    "interaction": ("holding", "facing", "occluding"),
}


@dataclass(frozen=True)
class CanonicalRelation:
    relation: str
    swap_roles: bool = False


@dataclass(frozen=True)
class RelationGraph:
    anti: tuple[str, ...] = ()
    support: tuple[str, ...] = ()
    support_beta: float = 0.5


CANONICAL_RELATIONS: dict[str, CanonicalRelation] = {
    "right_of": CanonicalRelation("left_of", swap_roles=True),
    "below": CanonicalRelation("above", swap_roles=True),
    "contains": CanonicalRelation("inside", swap_roles=True),
    "behind": CanonicalRelation("in_front_of", swap_roles=True),
}

RECAP_RELATION_GRAPH: dict[str, RelationGraph] = {
    "left_of": RelationGraph(anti=("right_of",)),
    "right_of": RelationGraph(anti=("left_of",)),
    "above": RelationGraph(anti=("below",)),
    "below": RelationGraph(anti=("above",)),
    "on": RelationGraph(anti=("below",), support=("above",), support_beta=0.5),
    "inside": RelationGraph(anti=("contains",)),
    "contains": RelationGraph(anti=("inside",)),
    "near": RelationGraph(anti=("far",), support_beta=0.25),
    "far": RelationGraph(anti=("near",), support_beta=0.25),
    "in_front_of": RelationGraph(anti=("behind",)),
    "behind": RelationGraph(anti=("in_front_of",)),
}


def answer_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    text = str(value).strip().lower()
    if text in YES_VALUES:
        return True
    if text in NO_VALUES:
        return False
    return text.startswith("yes")


def bool_to_answer(value: bool) -> str:
    return "yes" if value else "no"


def normalize_relation(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.replace("to the ", "")
    text = text.replace("at the ", "")
    text = text.replace("on the ", "")
    text = text.replace("side of", "of")
    text = text.replace(" ", "_")
    if text in {"left", "leftof", "left_of", "left_of_of"}:
        return "left_of"
    if text in {"right", "rightof", "right_of", "right_of_of"}:
        return "right_of"
    if text in {"above", "over", "top", "on_top_of"}:
        return "above"
    if text in {"below", "under", "beneath", "bottom", "underneath"}:
        return "below"
    if text in {"on"}:
        return "on"
    if text in {"in_front_of", "front_of"}:
        return "in_front_of"
    if text in {"near", "next_to", "beside", "close_to"}:
        return "near"
    if text in {"far_away_from", "away_from"}:
        return "far"
    if text in {"in", "inside_of", "within"}:
        return "inside"
    return text


def opposite_relation(relation: str) -> str:
    relation = normalize_relation(relation)
    if relation == "left_of":
        return "right_of"
    if relation == "right_of":
        return "left_of"
    if relation == "above":
        return "below"
    if relation == "below":
        return "above"
    if relation == "on":
        return "below"
    if relation == "inside":
        return "contains"
    if relation == "contains":
        return "inside"
    if relation == "in_front_of":
        return "behind"
    if relation == "behind":
        return "in_front_of"
    if relation == "near":
        return "far"
    if relation == "far":
        return "near"
    return relation


def horizontal_flip_relation(relation: str) -> str:
    relation = normalize_relation(relation)
    if relation == "left_of":
        return "right_of"
    if relation == "right_of":
        return "left_of"
    return relation


def vertical_flip_relation(relation: str) -> str:
    relation = normalize_relation(relation)
    if relation == "above":
        return "below"
    if relation == "below":
        return "above"
    return relation


def horizontal_flip_wrong_relation(relation: str) -> str:
    """A deliberately wrong relation mapping for the horizontal-flip control.

    For left/right, the wrong mapping is "do not swap". For relations that are
    invariant under horizontal flip, the wrong mapping is their mutual-exclusion
    opposite when such an opposite is available.
    """

    relation = normalize_relation(relation)
    mapped = horizontal_flip_relation(relation)
    if mapped != relation:
        return relation
    opposite = opposite_relation(relation)
    if opposite != relation:
        return opposite
    return relation


def vertical_flip_wrong_relation(relation: str) -> str:
    relation = normalize_relation(relation)
    mapped = vertical_flip_relation(relation)
    if mapped != relation:
        return relation
    opposite = opposite_relation(relation)
    if opposite != relation:
        return opposite
    return relation


def inverse_role_relation(relation: str) -> str:
    relation = normalize_relation(relation)
    if relation == "inside":
        return "contains"
    if relation == "contains":
        return "inside"
    if relation == "in_front_of":
        return "behind"
    if relation == "behind":
        return "in_front_of"
    if relation == "above":
        return "below"
    if relation == "below":
        return "above"
    if relation == "left_of":
        return "right_of"
    if relation == "right_of":
        return "left_of"
    return ""


def supports_vertical_flip_probe(relation: str) -> bool:
    return normalize_relation(relation) in {"above", "below"}


def supports_inverse_role_probe(relation: str) -> bool:
    return bool(inverse_role_relation(relation))


def relation_family(relation: str) -> str:
    relation = normalize_relation(relation)
    for family, relations in RELATION_FAMILIES.items():
        if relation in relations:
            return family
    return "other"


def relation_candidates(relation: str) -> tuple[str, ...]:
    relation = normalize_relation(relation)
    family = relation_family(relation)
    candidates = list(RELATION_CANDIDATES.get(family, ()))
    if relation and relation not in candidates:
        candidates.insert(0, relation)
    opposite = opposite_relation(relation)
    if opposite and opposite != relation and opposite not in candidates:
        candidates.append(opposite)
    return tuple(candidates)


def canonicalize_relation(relation: str, *, allow_role_swap: bool = True) -> CanonicalRelation:
    relation = normalize_relation(relation)
    canonical = CANONICAL_RELATIONS.get(relation)
    if canonical is None or (canonical.swap_roles and not allow_role_swap):
        return CanonicalRelation(relation, swap_roles=False)
    return canonical


def recap_relation_graph(relation: str) -> RelationGraph:
    return RECAP_RELATION_GRAPH.get(normalize_relation(relation), RelationGraph())


def recap_probe_relations(relation: str, *, allow_role_swap: bool = True) -> tuple[CanonicalRelation, RelationGraph]:
    canonical = canonicalize_relation(relation, allow_role_swap=allow_role_swap)
    return canonical, recap_relation_graph(canonical.relation)


def infer_relation_from_question(question: str) -> str:
    lowered = str(question or "").lower()
    if "left" in lowered:
        return "left_of"
    if "right" in lowered:
        return "right_of"
    return ""


def is_left_right_relation(relation: str, question: str = "") -> bool:
    relation = normalize_relation(relation) or infer_relation_from_question(question)
    return relation in {"left_of", "right_of"}


def build_relation_question(sample: dict, relation: str) -> str:
    relation = normalize_relation(relation)
    relation_questions = sample.get("relation_questions", {})
    if isinstance(relation_questions, dict):
        question = str(relation_questions.get(relation, "")).strip()
        if question:
            return question

    source_question = str(sample.get("source_question", "")).strip()
    if source_question:
        phrase = relation_phrase(relation)
        options = sample.get("source_options", sample.get("options", ""))
        options_text = ""
        if isinstance(options, (list, tuple)) and options:
            options_text = "Available answers: " + ", ".join(str(option) for option in options) + "\n"
        elif isinstance(options, str) and options:
            options_text = f"Available answers: {options}\n"
        return f"For this spatial question:\n{source_question}\n{options_text}Is the correct answer \"{phrase}\"?"

    subject = _clean_name(sample.get("subject", sample.get("subj")))
    obj = _clean_name(sample.get("object", sample.get("obj")))

    if subject and obj:
        relation_question = build_subject_object_question(subject, obj, relation)
        if relation_question:
            return relation_question

    question = str(sample.get("question", "")).strip()
    if relation == "left_of":
        return re.sub(r"\bright\b", "left", question, flags=re.IGNORECASE)
    if relation == "right_of":
        return re.sub(r"\bleft\b", "right", question, flags=re.IGNORECASE)
    return question


def relation_phrase(relation: str) -> str:
    relation = normalize_relation(relation)
    if relation == "left_of":
        return "left of"
    if relation == "right_of":
        return "right of"
    if relation == "in_front_of":
        return "in front of"
    return relation.replace("_", " ")


def build_subject_object_question(subject: str, obj: str, relation: str) -> str:
    relation = normalize_relation(relation)
    if relation == "left_of":
        return f"Is {subject} to the left of {obj}?"
    if relation == "right_of":
        return f"Is {subject} to the right of {obj}?"
    if relation == "above":
        return f"Is {subject} above {obj}?"
    if relation == "below":
        return f"Is {subject} below {obj}?"
    if relation == "on":
        return f"Is {subject} on {obj}?"
    if relation == "inside":
        return f"Is {subject} inside {obj}?"
    if relation == "contains":
        return f"Does {subject} contain {obj}?"
    if relation == "near":
        return f"Is {subject} near {obj}?"
    if relation == "far":
        return f"Is {subject} far from {obj}?"
    if relation == "in_front_of":
        return f"Is {subject} in front of {obj}?"
    if relation == "behind":
        return f"Is {subject} behind {obj}?"
    if relation == "holding":
        return f"Is {subject} holding {obj}?"
    if relation == "facing":
        return f"Is {subject} facing {obj}?"
    if relation == "occluding":
        return f"Is {subject} occluding {obj}?"
    return ""


def _clean_name(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "none", "null", "nan"} else text
