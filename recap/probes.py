"""Build RECAP probes from canonical relation samples."""

from __future__ import annotations

from typing import Any, Iterable

from recap.bbox import get_bbox
from recap.relations import (
    answer_to_bool,
    bool_to_answer,
    build_relation_question,
    horizontal_flip_relation,
    horizontal_flip_wrong_relation,
    infer_relation_from_question,
    inverse_role_relation,
    is_left_right_relation,
    normalize_relation,
    opposite_relation,
    recap_probe_relations,
    relation_family,
    relation_phrase,
    relation_candidates,
    supports_inverse_role_probe,
    supports_vertical_flip_probe,
    vertical_flip_relation,
    vertical_flip_wrong_relation,
)

BASE_PROBES = ("orig", "flip_mapped", "flip_unmapped", "contra", "text_only")
BBOX_PROBES = ("subject_masked", "object_masked", "crop")
PROBE_MODES = (
    "full",
    "orig_text",
    "orig_text_boxed",
    "orig_text_crop",
    "orig_text_crop_boxed",
    "profile_fast",
    "profile_fast_text",
    "rec",
    "rec_profile",
    "recap",
    "recap_profile",
    "prompt_sc",
)


def build_probes(sample: dict[str, Any], *, require_left_right: bool = True, probe_mode: str = "full") -> list[dict[str, Any]]:
    relation = normalize_relation(sample.get("relation") or infer_relation_from_question(sample.get("question", "")))
    if require_left_right and not is_left_right_relation(relation, sample.get("question", "")):
        return []
    if probe_mode not in PROBE_MODES:
        raise ValueError(f"Unknown probe_mode={probe_mode!r}; expected one of {PROBE_MODES}")

    sample_id = str(sample.get("id", sample.get("idx", sample.get("sample_id", ""))))
    if not sample_id:
        raise ValueError(f"RECAP sample is missing id/sample_id: {sample}")

    answer = answer_to_bool(sample.get("answer", sample.get("label", sample.get("target", True))))
    subject_bbox = get_bbox(sample, "subject")
    object_bbox = get_bbox(sample, "object")
    has_bbox = subject_bbox is not None and object_bbox is not None
    opposite = opposite_relation(relation)
    flip_relation = horizontal_flip_relation(relation)
    wrong_flip_relation = horizontal_flip_wrong_relation(relation)
    vertical_relation = vertical_flip_relation(relation)
    wrong_vertical_relation = vertical_flip_wrong_relation(relation)
    inverse_relation = inverse_role_relation(relation)

    specs: list[dict[str, Any]] = [
        {"probe": "orig", "view": "original", "relation": relation, "answer": answer},
        {"probe": "flip_mapped", "view": "flip_mapped", "relation": flip_relation, "answer": answer},
        {"probe": "flip_unmapped", "view": "flip_unmapped", "relation": wrong_flip_relation, "answer": not answer},
        {"probe": "contra", "view": "original", "relation": opposite, "answer": not answer},
        {"probe": "text_only", "view": "text_only", "relation": relation, "answer": answer},
    ]
    if supports_vertical_flip_probe(relation):
        specs.extend(
            [
                {"probe": "vertical_flip_mapped", "view": "vertical_flip_mapped", "relation": vertical_relation, "answer": answer},
                {"probe": "vertical_flip_unmapped", "view": "vertical_flip_unmapped", "relation": wrong_vertical_relation, "answer": not answer},
            ]
        )
    if supports_inverse_role_probe(relation) and _has_subject_object(sample):
        specs.extend(
            [
                {"probe": "inverse_role", "view": "original", "relation": inverse_relation, "answer": answer, "swap_roles": True},
                {"probe": "role_reversal", "view": "original", "relation": relation, "answer": not answer, "swap_roles": True},
            ]
        )
    if has_bbox:
        specs.extend(
            [
                {"probe": "subject_masked", "view": "subject_masked", "relation": relation, "answer": answer},
                {"probe": "object_masked", "view": "object_masked", "relation": relation, "answer": answer},
                {"probe": "crop", "view": "crop", "relation": relation, "answer": answer},
            ]
        )
        if probe_mode in {"orig_text_boxed", "orig_text_crop_boxed"}:
            specs.append(
                {
                    "probe": "boxed",
                    "view": "boxed",
                    "relation": relation,
                    "answer": answer,
                    "question_override": _boxed_relation_question(relation),
                }
            )
    if probe_mode in {"rec", "rec_profile"}:
        specs.extend(_relation_candidate_specs(relation))
    if probe_mode in {"recap", "recap_profile"}:
        specs.extend(_recap_relation_specs(sample, relation))
    if probe_mode == "prompt_sc":
        specs.extend(_self_consistency_specs(sample, relation, answer))

    specs = _filter_specs_for_probe_mode(specs, relation=relation, probe_mode=probe_mode)
    probes = []
    for spec in specs:
        probe = _make_probe(
            sample,
            sample_id,
            relation,
            str(spec["probe"]),
            str(spec["view"]),
            str(spec["relation"]),
            bool(spec["answer"]),
            has_bbox,
            swap_roles=bool(spec.get("swap_roles", False)),
        )
        if "rec_candidate_relation" in spec:
            probe["rec_candidate_relation"] = str(spec["rec_candidate_relation"])
            probe["rec_candidate_kind"] = str(spec.get("rec_candidate_kind", ""))
            probe["rec_candidate_family"] = relation_family(str(spec["rec_candidate_relation"]))
        if "recap_candidate_relation" in spec:
            probe["recap_candidate_relation"] = str(spec["recap_candidate_relation"])
            probe["recap_candidate_kind"] = str(spec.get("recap_candidate_kind", ""))
            probe["recap_candidate_role"] = str(spec.get("recap_candidate_role", ""))
            probe["recap_canonical_relation"] = str(spec.get("recap_canonical_relation", ""))
            probe["recap_support_beta"] = float(spec.get("recap_support_beta", 0.0))
            probe["recap_candidate_family"] = relation_family(str(spec["recap_candidate_relation"]))
        if "question_override" in spec:
            probe["question"] = str(spec["question_override"])
            probe["prompt_sc_variant"] = str(spec.get("prompt_sc_variant", spec["probe"]))
        probes.append(probe)
    for probe in probes:
        probe["probe_count"] = len(probes)
    return probes


def build_probe_dataset(samples: Iterable[dict[str, Any]], *, require_left_right: bool = True, probe_mode: str = "full") -> list[dict[str, Any]]:
    probes: list[dict[str, Any]] = []
    for sample in samples:
        probes.extend(build_probes(sample, require_left_right=require_left_right, probe_mode=probe_mode))
    return probes


def _filter_specs_for_probe_mode(specs: list[dict[str, Any]], *, relation: str, probe_mode: str) -> list[dict[str, Any]]:
    if probe_mode == "full":
        return specs
    if probe_mode == "orig_text":
        return [spec for spec in specs if str(spec["probe"]) in {"orig", "text_only"}]
    if probe_mode == "orig_text_boxed":
        return [spec for spec in specs if str(spec["probe"]) in {"orig", "text_only", "boxed"}]
    if probe_mode == "orig_text_crop":
        return [spec for spec in specs if str(spec["probe"]) in {"orig", "text_only", "crop"}]
    if probe_mode == "orig_text_crop_boxed":
        return [spec for spec in specs if str(spec["probe"]) in {"orig", "text_only", "crop", "boxed"}]
    if probe_mode == "rec":
        return [spec for spec in specs if str(spec["probe"]) in {"orig", "text_only"} or str(spec["probe"]).startswith("rec_")]
    if probe_mode == "recap":
        return [spec for spec in specs if str(spec["probe"]) in {"orig", "text_only"} or str(spec["probe"]).startswith("recap_")]
    if probe_mode == "prompt_sc":
        return [spec for spec in specs if str(spec["probe"]) == "orig" or str(spec["probe"]).startswith("prompt_sc_")]

    family = relation_family(relation)
    keep = {"orig", "contra"}
    if probe_mode in {"profile_fast_text", "rec_profile"}:
        keep.add("text_only")
    if family == "left_right":
        keep.update({"flip_mapped", "flip_unmapped"})
    elif family == "vertical":
        keep.update({"vertical_flip_mapped", "vertical_flip_unmapped", "inverse_role", "role_reversal"})
    elif family in {"topology", "depth"}:
        keep.update({"inverse_role", "role_reversal"})
    elif family == "interaction":
        keep.add("role_reversal")
    if probe_mode == "rec_profile":
        return [spec for spec in specs if str(spec["probe"]) in keep or str(spec["probe"]).startswith("rec_")]
    if probe_mode == "recap_profile":
        return [spec for spec in specs if str(spec["probe"]) in keep or str(spec["probe"]).startswith("recap_")]

    return [spec for spec in specs if str(spec["probe"]) in keep]


def _self_consistency_specs(sample: dict[str, Any], relation: str, answer: bool) -> list[dict[str, Any]]:
    base_question = build_relation_question(sample, relation).strip()
    if not base_question:
        base_question = str(sample.get("question", "")).strip()
    core = base_question.rstrip(" ?")
    lower_core = core[:1].lower() + core[1:] if core else core
    questions = (
        f"According to the image, {lower_core}?",
        f"Look at the image carefully. {core}?",
        f"Answer yes or no using only visual evidence: {lower_core}?",
        f"Based on what is visible, is it true that {lower_core}?",
    )
    return [
        {
            "probe": f"prompt_sc_{index}",
            "view": "original",
            "relation": relation,
            "answer": answer,
            "question_override": question,
            "prompt_sc_variant": f"paraphrase_{index}",
        }
        for index, question in enumerate(questions, start=1)
    ]


def _boxed_relation_question(relation: str) -> str:
    phrase = relation_phrase(relation)
    return (
        "In the image, a red box labeled A marks one object and a blue box labeled B marks another object. "
        f"Is the red-boxed A object {phrase} the blue-boxed B object? Answer yes or no."
    )


def _relation_candidate_specs(relation: str) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for candidate in relation_candidates(relation):
        safe_candidate = candidate.replace("/", "_")
        for kind, view in (("img", "original"), ("text", "text_only")):
            specs.append(
                {
                    "probe": f"rec_{kind}__{safe_candidate}",
                    "view": view,
                    "relation": candidate,
                    "answer": True,
                    "rec_candidate_relation": candidate,
                    "rec_candidate_kind": kind,
                }
            )
    return specs


def _recap_relation_specs(sample: dict[str, Any], relation: str) -> list[dict[str, Any]]:
    allow_swap = _has_subject_object(sample)
    canonical, graph = recap_probe_relations(relation, allow_role_swap=allow_swap)
    canonical_relation = canonical.relation
    specs: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add(role: str, candidate: str, *, support_beta: float = 0.0) -> None:
        candidate = normalize_relation(candidate)
        if not candidate:
            return
        key = (role, candidate)
        if key in seen:
            return
        seen.add(key)
        safe_candidate = candidate.replace("/", "_")
        for kind, view in (("img", "original"), ("text", "text_only")):
            specs.append(
                {
                    "probe": f"recap_{kind}__{role}__{safe_candidate}",
                    "view": view,
                    "relation": candidate,
                    "answer": True,
                    "swap_roles": canonical.swap_roles,
                    "recap_candidate_relation": candidate,
                    "recap_candidate_kind": kind,
                    "recap_candidate_role": role,
                    "recap_canonical_relation": canonical_relation,
                    "recap_support_beta": support_beta,
                }
            )

    add("claim", canonical_relation)
    for anti_relation in graph.anti:
        add("anti", anti_relation)
    for support_relation in graph.support:
        add("support", support_relation, support_beta=graph.support_beta)
    return specs


def _make_probe(
    sample: dict[str, Any],
    sample_id: str,
    base_relation: str,
    probe: str,
    view: str,
    relation: str,
    answer: bool,
    has_bbox: bool,
    *,
    swap_roles: bool = False,
) -> dict[str, Any]:
    out = dict(sample)
    if swap_roles:
        out = _swap_subject_object(out)
    out["sample_id"] = sample_id
    out["probe"] = probe
    out["rice_view"] = view
    out["base_relation"] = base_relation
    out["relation"] = relation
    out["question"] = build_relation_question(out, relation)
    out["target_answer"] = bool_to_answer(answer)
    out["has_bbox"] = has_bbox
    out["base_has_bbox"] = has_bbox
    out["bbox_source"] = str(sample.get("bbox_source", "")).lower()
    return out


def _has_subject_object(sample: dict[str, Any]) -> bool:
    return bool(sample.get("subject") or sample.get("subj")) and bool(sample.get("object") or sample.get("obj"))


def _swap_subject_object(sample: dict[str, Any]) -> dict[str, Any]:
    out = dict(sample)
    out.pop("relation_questions", None)
    out.pop("source_question", None)
    out.pop("source_options", None)
    out.pop("options", None)
    subject = out.get("subject", out.get("subj", ""))
    obj = out.get("object", out.get("obj", ""))
    out["subject"] = obj
    out["object"] = subject
    if "subj" in out:
        out["subj"] = obj
    if "obj" in out:
        out["obj"] = subject

    for left_key, right_key in (
        ("subject_bbox", "object_bbox"),
        ("bbox_subject", "bbox_object"),
        ("subject_box", "object_box"),
        ("subj_bbox", "obj_bbox"),
        ("subj_box", "obj_box"),
        ("subjectBox", "objectBox"),
    ):
        if left_key in out or right_key in out:
            out[left_key], out[right_key] = out.get(right_key), out.get(left_key)

    boxes = out.get("bboxes") or out.get("boxes")
    if isinstance(boxes, dict):
        swapped = dict(boxes)
        for left_key, right_key in (("subject", "object"), ("subj", "obj")):
            if left_key in swapped or right_key in swapped:
                swapped[left_key], swapped[right_key] = swapped.get(right_key), swapped.get(left_key)
        if "bboxes" in out:
            out["bboxes"] = swapped
        if "boxes" in out:
            out["boxes"] = swapped
    return out
