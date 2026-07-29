"""RECAP risk scoring from yes/no losses."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from recap.metrics import binary_entropy, rank_norm
from recap.relations import answer_to_bool, bool_to_answer, relation_family

CHOICES = ("yes", "no")
REMOVAL_PROBES = ("text_only", "subject_masked", "object_masked")
RISK_COMPONENTS = ("v_eq", "v_rem", "v_pres", "v_contra")
RICE_V6_COMPONENTS = ("u_conf", "c_contra", "g_prior", "e_spec")
RICE_PROFILE_COMPONENTS = ("u_conf", "c_contra", "g_prior", "r_struct")
RICE_LIFT_COMPONENTS = ("u_conf", "c_contra", "visual_lift_risk", "r_struct")
RICE_REC_COMPONENTS = ("u_conf", "rec_relation_risk", "rec_prior_dominance_risk", "c_contra", "r_struct")
RICE_RECAP_COMPONENTS = ("u_conf", "recap_evidence_risk")


def score_probe(probe: dict[str, Any], yes_loss: float, no_loss: float) -> dict[str, Any]:
    margin = float(no_loss) - float(yes_loss)
    pred_yes = margin >= 0.0
    target_yes = answer_to_bool(probe.get("target_answer", probe.get("answer", True)))
    out = {
        "sample_id": str(probe["sample_id"]),
        "probe": probe["probe"],
        "rice_view": probe.get("rice_view", "original"),
        "dataset": probe.get("dataset", ""),
        "relation": probe.get("relation", ""),
        "base_relation": probe.get("base_relation", probe.get("relation", "")),
        "question": probe.get("question", ""),
        "target_answer": bool_to_answer(target_yes),
        "pred_answer": bool_to_answer(pred_yes),
        "correct": pred_yes == target_yes,
        "margin": margin,
        "support_pred": abs(margin),
        "yes_loss": float(yes_loss),
        "no_loss": float(no_loss),
        "has_bbox": bool(probe.get("has_bbox", False)),
        "base_has_bbox": bool(probe.get("base_has_bbox", probe.get("has_bbox", False))),
        "bbox_source": str(probe.get("bbox_source", "")),
        "probe_count": int(probe.get("probe_count", 1)),
    }
    for key in (
        "source_dataset",
        "image",
        "image_id",
        "source_caption",
        "source_caption_options",
        "option_index",
        "choice_group_id",
        "choice_index",
        "choice_is_correct",
        "task_form",
        "binary_polarity",
        "reference_frame",
        "rec_candidate_relation",
        "rec_candidate_kind",
        "rec_candidate_family",
        "recap_candidate_relation",
        "recap_candidate_kind",
        "recap_candidate_role",
        "recap_candidate_family",
        "recap_canonical_relation",
        "recap_support_beta",
        "prompt_sc_variant",
        "target_text",
        "source_text",
        "answer_options",
        "candidate_answer",
        "task_family",
        "hard_type",
        "ocrbench_question",
        "ocrbench_question_type",
        "ocrbench_row_index",
        "ocrbench_local_index",
    ):
        if key in probe:
            out[key] = probe[key]
    return out


def prepare_sample_scores(probe_scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in probe_scores:
        grouped[str(record["sample_id"])][record["probe"]] = record

    samples: list[dict[str, Any]] = []
    for sample_id, probes in grouped.items():
        orig = probes.get("orig")
        if orig is None:
            continue

        yhat = orig["pred_answer"]
        support_orig = support_for_answer(float(orig["margin"]), yhat)
        support_flip = support_for_answer(float(probes.get("flip_mapped", orig)["margin"]), yhat)
        support_wrong = support_for_answer(float(probes.get("flip_unmapped", orig)["margin"]), yhat)
        support_vflip = support_for_answer(float(probes.get("vertical_flip_mapped", orig)["margin"]), yhat)
        support_wrong_vflip = support_for_answer(float(probes.get("vertical_flip_unmapped", orig)["margin"]), yhat)
        support_inverse = support_for_answer(float(probes.get("inverse_role", orig)["margin"]), yhat)
        support_role_reversal = support_for_answer(float(probes.get("role_reversal", orig)["margin"]), yhat)
        support_removed = [support_for_answer(float(probes[probe]["margin"]), yhat) for probe in REMOVAL_PROBES if probe in probes]
        support_crop = support_for_answer(float(probes["crop"]["margin"]), yhat) if "crop" in probes else support_orig
        prompt_sc_records = [probes[name] for name in sorted(probes) if name.startswith("prompt_sc_")]
        prompt_sc_supports = [support_orig] + [
            support_for_answer(float(record["margin"]), yhat) for record in prompt_sc_records
        ]
        prompt_sc_worst_support = min(prompt_sc_supports)
        prompt_sc_mean_support = sum(prompt_sc_supports) / len(prompt_sc_supports)
        prompt_sc_disagreement = sum(record["pred_answer"] != yhat for record in prompt_sc_records) / max(
            1, len(prompt_sc_records)
        )

        v_eq = abs(support_orig - support_flip)
        wrong_mapping = abs(support_orig - support_wrong)
        v_vflip = abs(support_orig - support_vflip) if "vertical_flip_mapped" in probes else 0.0
        wrong_vflip = abs(support_orig - support_wrong_vflip) if "vertical_flip_unmapped" in probes else 0.0
        v_inverse = abs(support_orig - support_inverse) if "inverse_role" in probes else 0.0
        wrong_inverse = abs(support_orig - support_role_reversal) if "role_reversal" in probes else 0.0
        v_rem = max(support_removed) if support_removed else 0.0
        v_pres = abs(support_orig - support_crop)
        if "crop" in probes and probes["crop"]["pred_answer"] != yhat:
            v_pres += 1.0

        orig_yes_margin = float(orig["margin"])
        has_text_only = "text_only" in probes
        text_yes_margin = float(probes["text_only"]["margin"]) if has_text_only else orig_yes_margin
        visual_lift_yes_margin = orig_yes_margin - text_yes_margin if has_text_only else 0.0
        text_prior_risk = support_for_answer(text_yes_margin, yhat) if has_text_only else 0.0
        visual_lift_support = support_for_answer(visual_lift_yes_margin, yhat) if has_text_only else 0.0
        visual_lift_risk = -visual_lift_support
        visual_lift_abs_risk = -abs(visual_lift_yes_margin) if has_text_only else 0.0
        contra_yes_margin = float(probes["contra"]["margin"]) if "contra" in probes else 0.0
        role_reversal_yes_margin = float(probes["role_reversal"]["margin"]) if "role_reversal" in probes else 0.0
        v_contra = min(max(orig_yes_margin, 0.0), max(contra_yes_margin, 0.0))
        role_contra = min(max(orig_yes_margin, 0.0), max(role_reversal_yes_margin, 0.0))
        e_spec_signed = v_eq - wrong_mapping
        e_spec = max(0.0, e_spec_signed)
        e_vflip_signed = v_vflip - wrong_vflip
        e_vflip = max(0.0, e_vflip_signed)
        e_inverse_signed = v_inverse - wrong_inverse
        e_inverse = max(0.0, e_inverse_signed)
        family = relation_family(orig.get("base_relation", orig.get("relation", "")))
        r_struct = relation_structural_risk(family=family, e_spec=e_spec, e_vflip=e_vflip, e_inverse=e_inverse, role_contra=role_contra)
        rec = relation_evidence_profile(probes, base_relation=str(orig.get("base_relation", orig.get("relation", ""))), predicted_answer=yhat)
        recap = recap_evidence_profile(probes, predicted_answer=yhat)
        target_is_negative = orig["target_answer"] == "no"
        hallucination = target_is_negative and orig["pred_answer"] == "yes"

        sample = {
            "sample_id": sample_id,
            "dataset": orig.get("dataset", ""),
            "base_relation": orig.get("base_relation", orig.get("relation", "")),
            "relation_family": family,
            "error": 0.0 if orig["correct"] else 1.0,
            "hallucination": 1.0 if hallucination else 0.0,
            "target_is_negative": target_is_negative,
            "direct_correct": bool(orig["correct"]),
            "direct_pred": orig["pred_answer"],
            "direct_target": orig["target_answer"],
            "support_orig": support_orig,
            "confidence_risk": -support_orig,
            "prompt_sc_risk": -prompt_sc_worst_support,
            "prompt_sc_mean_risk": -prompt_sc_mean_support,
            "prompt_sc_disagreement_risk": prompt_sc_disagreement,
            "prompt_sc_probe_count": len(prompt_sc_supports),
            "u_conf": -support_orig,
            "entropy_risk": binary_entropy(orig_yes_margin),
            "text_yes_margin": text_yes_margin,
            "text_prior_risk": text_prior_risk,
            "visual_lift_yes_margin": visual_lift_yes_margin,
            "visual_lift_support": visual_lift_support,
            "visual_lift_risk": visual_lift_risk,
            "visual_lift_abs_risk": visual_lift_abs_risk,
            "visual_lift_pred": bool_to_answer(visual_lift_yes_margin >= 0.0) if has_text_only else "",
            "visual_lift_correct": (
                bool(visual_lift_yes_margin >= 0.0) == answer_to_bool(orig["target_answer"])
                if has_text_only
                else None
            ),
            "rec_candidate_count": rec["candidate_count"],
            "rec_claim_relation": rec["claim_relation"],
            "rec_best_negative_relation": rec["best_negative_relation"],
            "rec_claim_img_margin": rec["claim_img_margin"],
            "rec_claim_text_margin": rec["claim_text_margin"],
            "rec_claim_cal_margin": rec["claim_cal_margin"],
            "rec_best_negative_cal_margin": rec["best_negative_cal_margin"],
            "rec_relation_margin": rec["relation_margin"],
            "rec_relation_rank": rec["relation_rank"],
            "rec_relation_risk": rec["relation_risk"],
            "rec_relation_confidence_risk": rec["relation_confidence_risk"],
            "rec_prior_risk": rec["prior_risk"],
            "rec_prior_dominance_risk": rec["prior_dominance_risk"],
            "rec_candidate_margins": rec["candidate_margins"],
            "recap_candidate_count": recap["candidate_count"],
            "recap_canonical_relation": recap["canonical_relation"],
            "recap_claim_delta": recap["claim_delta"],
            "recap_best_anti_relation": recap["best_anti_relation"],
            "recap_best_anti_delta": recap["best_anti_delta"],
            "recap_pair_margin": recap["pair_margin"],
            "recap_support_penalty": recap["support_penalty"],
            "recap_evidence_margin": recap["evidence_margin"],
            "recap_evidence_risk": recap["evidence_risk"],
            "recap_confidence_risk": recap["confidence_risk"],
            "recap_has_contrast": recap["has_contrast"],
            "recap_candidate_margins": recap["candidate_margins"],
            "g_prior": v_rem,
            "cap_risk": v_contra,
            "c_contra": v_contra,
            "v_eq": v_eq,
            "v_rem": v_rem,
            "v_pres": v_pres,
            "v_contra": v_contra,
            "role_contra": role_contra,
            "wrong_mapping": wrong_mapping,
            "e_spec": e_spec,
            "e_spec_signed": e_spec_signed,
            "v_vflip": v_vflip,
            "wrong_vflip": wrong_vflip,
            "e_vflip": e_vflip,
            "e_vflip_signed": e_vflip_signed,
            "v_inverse": v_inverse,
            "wrong_inverse": wrong_inverse,
            "e_inverse": e_inverse,
            "e_inverse_signed": e_inverse_signed,
            "r_struct": r_struct,
            "probe_count": int(orig.get("probe_count", len(probes))),
            "bbox_source": str(orig.get("bbox_source", "")),
            "base_has_bbox": bool(orig.get("base_has_bbox", False)),
        }
        for key in (
            "source_dataset",
            "image",
            "image_id",
            "target_text",
            "source_text",
            "answer_options",
            "candidate_answer",
            "task_family",
            "hard_type",
            "binary_polarity",
            "ocrbench_question",
            "ocrbench_question_type",
            "ocrbench_row_index",
            "ocrbench_local_index",
        ):
            if key in orig:
                sample[key] = orig[key]
        samples.append(sample)

    add_ranked_risk(samples)
    return samples


def add_ranked_risk(samples: list[dict[str, Any]]) -> None:
    ranked = {component: rank_norm([sample[component] for sample in samples]) for component in RISK_COMPONENTS}
    ranked_v6 = {component: active_rank_norm([sample[component] for sample in samples]) for component in RICE_V6_COMPONENTS}
    ranked_profile = {component: active_rank_norm([sample[component] for sample in samples]) for component in RICE_PROFILE_COMPONENTS}
    ranked_lift = {component: active_rank_norm([sample[component] for sample in samples]) for component in RICE_LIFT_COMPONENTS}
    ranked_rec = {component: active_rank_norm([sample[component] for sample in samples]) for component in RICE_REC_COMPONENTS}
    ranked_recap = {component: active_rank_norm([sample[component] for sample in samples]) for component in RICE_RECAP_COMPONENTS}
    if not has_signal_variation([sample["wrong_mapping"] for sample in samples]):
        ranked_v6["e_spec"] = [0.0 for _ in samples]
    if not has_signal_variation([sample["r_struct"] for sample in samples]):
        ranked_profile["r_struct"] = [0.0 for _ in samples]
        ranked_lift["r_struct"] = [0.0 for _ in samples]
        ranked_rec["r_struct"] = [0.0 for _ in samples]
    if not has_signal_variation([sample["visual_lift_risk"] for sample in samples]):
        ranked_lift["visual_lift_risk"] = [0.0 for _ in samples]
    if not has_signal_variation([sample["rec_relation_risk"] for sample in samples]):
        ranked_rec["rec_relation_risk"] = [0.0 for _ in samples]
    if not has_signal_variation([sample["rec_prior_dominance_risk"] for sample in samples]):
        ranked_rec["rec_prior_dominance_risk"] = [0.0 for _ in samples]
    if not has_signal_variation([sample["recap_evidence_risk"] for sample in samples]):
        ranked_recap["recap_evidence_risk"] = [0.0 for _ in samples]
    for i, sample in enumerate(samples):
        sample["rice_risk"] = sum(ranked[component][i] for component in RISK_COMPONENTS)
        sample["rice_wo_v_eq"] = sum(ranked[component][i] for component in RISK_COMPONENTS if component != "v_eq")
        sample["rice_wo_v_rem"] = sum(ranked[component][i] for component in RISK_COMPONENTS if component != "v_rem")
        sample["rice_wo_v_pres"] = sum(ranked[component][i] for component in RISK_COMPONENTS if component != "v_pres")
        sample["rice_wo_v_contra"] = sum(ranked[component][i] for component in RISK_COMPONENTS if component != "v_contra")

        v6_values = {component: ranked_v6[component][i] for component in RICE_V6_COMPONENTS}
        for component, value in v6_values.items():
            sample[f"{component}_rank"] = value
        component_scores = list(v6_values.values())
        sample["rice_v6_mean"] = sum(component_scores) / len(component_scores)
        sample["rice_v6_max"] = max(component_scores)
        sample["rice_v6_top2"] = mean_top_k(component_scores, 2)
        sample["rice_v6_wo_u_conf"] = mean_top_k([value for component, value in v6_values.items() if component != "u_conf"], 2)
        sample["rice_v6_wo_c_contra"] = mean_top_k([value for component, value in v6_values.items() if component != "c_contra"], 2)
        sample["rice_v6_wo_g_prior"] = mean_top_k([value for component, value in v6_values.items() if component != "g_prior"], 2)
        sample["rice_v6_wo_e_spec"] = mean_top_k([value for component, value in v6_values.items() if component != "e_spec"], 2)

        profile_values = {component: ranked_profile[component][i] for component in RICE_PROFILE_COMPONENTS}
        for component, value in profile_values.items():
            sample[f"{component}_profile_rank"] = value
        profile_scores = list(profile_values.values())
        sample["rice_profile_mean"] = sum(profile_scores) / len(profile_scores)
        sample["rice_profile_max"] = max(profile_scores)
        sample["rice_profile_top2"] = mean_top_k(profile_scores, 2)
        sample["rice_profile_wo_g_prior"] = mean_top_k([value for component, value in profile_values.items() if component != "g_prior"], 2)
        sample["rice_profile_wo_struct"] = mean_top_k([value for component, value in profile_values.items() if component != "r_struct"], 2)

        lift_values = {component: ranked_lift[component][i] for component in RICE_LIFT_COMPONENTS}
        for component, value in lift_values.items():
            sample[f"{component}_lift_rank"] = value
        lift_scores = list(lift_values.values())
        sample["rice_profile_lift"] = mean_top_k(lift_scores, 2)
        sample["rice_profile_lift_mean"] = sum(lift_scores) / len(lift_scores)
        sample["rice_profile_lift_max"] = max(lift_scores)
        sample["rice_profile_lift_wo_visual"] = mean_top_k(
            [value for component, value in lift_values.items() if component != "visual_lift_risk"], 2
        )

        rec_values = {component: ranked_rec[component][i] for component in RICE_REC_COMPONENTS}
        for component, value in rec_values.items():
            sample[f"{component}_rec_rank"] = value
        rec_scores = list(rec_values.values())
        sample["rice_rec"] = mean_top_k(rec_scores, 2)
        sample["rice_rec_mean"] = sum(rec_scores) / len(rec_scores)
        sample["rice_rec_max"] = max(rec_scores)
        sample["rice_rec_wo_prior"] = mean_top_k(
            [value for component, value in rec_values.items() if component != "rec_prior_dominance_risk"], 2
        )
        sample["rice_rec_wo_struct"] = mean_top_k(
            [value for component, value in rec_values.items() if component != "r_struct"], 2
        )
        sample["rice_rec_wo_conf"] = mean_top_k(
            [value for component, value in rec_values.items() if component != "u_conf"], 2
        )

        recap_values = {component: ranked_recap[component][i] for component in RICE_RECAP_COMPONENTS}
        for component, value in recap_values.items():
            sample[f"{component}_recap_rank"] = value
        recap_scores = list(recap_values.values())
        sample["rice_recap_selector"] = max(recap_scores)
        sample["rice_recap_mean"] = sum(recap_scores) / len(recap_scores)
        sample["rice_recap_wo_conf"] = recap_values["recap_evidence_risk"]


def relation_evidence_profile(probes: dict[str, dict[str, Any]], *, base_relation: str, predicted_answer: str) -> dict[str, Any]:
    img_margins = candidate_margins(probes, "rec_img__")
    text_margins = candidate_margins(probes, "rec_text__")
    candidates = sorted(set(img_margins) | set(text_margins))
    base_relation = str(base_relation or "").strip()
    if base_relation not in candidates and "orig" in probes:
        img_margins[base_relation] = float(probes["orig"]["margin"])
        candidates = sorted(set(img_margins) | set(text_margins))

    calibrated: dict[str, float] = {}
    for relation in candidates:
        if relation not in img_margins:
            continue
        calibrated[relation] = img_margins[relation] - text_margins.get(relation, 0.0)

    claim_cal = calibrated.get(base_relation, 0.0)
    claim_img = img_margins.get(base_relation, 0.0)
    claim_text = text_margins.get(base_relation, 0.0)
    negative_items = [(relation, margin) for relation, margin in calibrated.items() if relation != base_relation]
    if negative_items:
        best_negative_relation, best_negative_margin = max(negative_items, key=lambda item: item[1])
        relation_margin = claim_cal - best_negative_margin
        relation_rank = 1 + sum(1 for _, margin in negative_items if margin > claim_cal)
    else:
        best_negative_relation = ""
        best_negative_margin = 0.0
        relation_margin = 0.0
        relation_rank = 0

    has_contrast = bool(negative_items) and base_relation in calibrated
    relation_support = support_for_answer(relation_margin, predicted_answer) if has_contrast else 0.0
    relation_risk = -relation_support if has_contrast else 0.0
    relation_confidence_risk = -abs(relation_margin) if has_contrast else 0.0
    prior_support = support_for_answer(claim_text, predicted_answer) if base_relation in text_margins else 0.0
    prior_risk = prior_support if base_relation in text_margins else 0.0
    prior_dominance_risk = prior_support - relation_support if has_contrast and base_relation in text_margins else 0.0

    return {
        "candidate_count": len(calibrated),
        "claim_relation": base_relation,
        "best_negative_relation": best_negative_relation,
        "claim_img_margin": claim_img,
        "claim_text_margin": claim_text,
        "claim_cal_margin": claim_cal,
        "best_negative_cal_margin": best_negative_margin,
        "relation_margin": relation_margin,
        "relation_rank": relation_rank,
        "relation_risk": relation_risk,
        "relation_confidence_risk": relation_confidence_risk,
        "prior_risk": prior_risk,
        "prior_dominance_risk": prior_dominance_risk,
        "candidate_margins": {
            relation: {
                "img": img_margins.get(relation, 0.0),
                "text": text_margins.get(relation, 0.0),
                "calibrated": calibrated.get(relation, 0.0),
            }
            for relation in candidates
        },
    }


def recap_evidence_profile(probes: dict[str, dict[str, Any]], *, predicted_answer: str) -> dict[str, Any]:
    candidates = recap_candidate_records(probes)
    canonical_relation = ""
    claim_delta = 0.0
    anti_deltas: list[tuple[str, float]] = []
    support_deltas: list[tuple[str, float, float]] = []
    candidate_margins: dict[str, dict[str, float | str]] = {}

    for key, records in candidates.items():
        relation, role = key
        canonical_relation = canonical_relation or str(records.get("canonical_relation", ""))
        img_margin = records.get("img")
        text_margin = records.get("text")
        if img_margin is None:
            continue
        delta = float(img_margin) - float(text_margin or 0.0)
        candidate_margins[f"{role}:{relation}"] = {
            "relation": relation,
            "role": role,
            "img": float(img_margin),
            "text": float(text_margin or 0.0),
            "delta": delta,
        }
        if role == "claim":
            claim_delta = delta
        elif role == "anti":
            anti_deltas.append((relation, delta))
        elif role == "support":
            support_deltas.append((relation, delta, float(records.get("support_beta", 0.0))))

    if not candidates:
        return {
            "candidate_count": 0,
            "canonical_relation": "",
            "claim_delta": 0.0,
            "best_anti_relation": "",
            "best_anti_delta": 0.0,
            "pair_margin": 0.0,
            "support_penalty": 0.0,
            "evidence_margin": 0.0,
            "evidence_risk": 0.0,
            "confidence_risk": 0.0,
            "has_contrast": False,
            "candidate_margins": {},
        }

    if anti_deltas:
        best_anti_relation, best_anti_delta = max(anti_deltas, key=lambda item: item[1])
        pair_margin = claim_delta - best_anti_delta
    else:
        best_anti_relation = ""
        best_anti_delta = 0.0
        pair_margin = claim_delta

    support_penalty = 0.0
    for _, support_delta, support_beta in support_deltas:
        support_penalty += support_beta * max(0.0, -support_delta)

    has_contrast = bool(anti_deltas or support_deltas)
    evidence_margin = pair_margin - support_penalty if has_contrast else 0.0
    evidence_support = support_for_answer(evidence_margin, predicted_answer) if has_contrast else 0.0
    evidence_risk = -evidence_support if has_contrast else 0.0
    confidence_risk = -abs(evidence_margin) if has_contrast else 0.0

    return {
        "candidate_count": len(candidate_margins),
        "canonical_relation": canonical_relation,
        "claim_delta": claim_delta,
        "best_anti_relation": best_anti_relation,
        "best_anti_delta": best_anti_delta,
        "pair_margin": pair_margin,
        "support_penalty": support_penalty,
        "evidence_margin": evidence_margin,
        "evidence_risk": evidence_risk,
        "confidence_risk": confidence_risk,
        "has_contrast": has_contrast,
        "candidate_margins": candidate_margins,
    }


def recap_candidate_records(probes: dict[str, dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for probe_name, record in probes.items():
        if not probe_name.startswith("recap_"):
            continue
        relation = str(record.get("recap_candidate_relation", "")).strip()
        kind = str(record.get("recap_candidate_kind", "")).strip()
        role = str(record.get("recap_candidate_role", "")).strip()
        if not relation or kind not in {"img", "text"} or not role:
            continue
        key = (relation, role)
        bucket = records.setdefault(
            key,
            {
                "canonical_relation": str(record.get("recap_canonical_relation", "")),
                "support_beta": float(record.get("recap_support_beta", 0.0)),
            },
        )
        bucket[kind] = float(record["margin"])
    return records


def candidate_margins(probes: dict[str, dict[str, Any]], prefix: str) -> dict[str, float]:
    margins: dict[str, float] = {}
    for probe_name, record in probes.items():
        if not probe_name.startswith(prefix):
            continue
        relation = str(record.get("rec_candidate_relation", "")).strip()
        if not relation:
            relation = probe_name[len(prefix) :]
        margins[relation] = float(record["margin"])
    return margins


def relation_structural_risk(*, family: str, e_spec: float, e_vflip: float, e_inverse: float, role_contra: float) -> float:
    if family == "left_right":
        return e_spec
    if family == "vertical":
        return max(e_vflip, role_contra)
    if family in {"topology", "depth"}:
        return max(e_inverse, role_contra)
    if family == "interaction":
        return role_contra
    return 0.0


def support_for_answer(margin: float, answer: str) -> float:
    return margin if str(answer).lower() == "yes" else -margin


def active_rank_norm(values: list[float]) -> list[float]:
    if not values:
        return []
    if not has_signal_variation(values):
        return [0.0 for _ in values]
    return rank_norm(values)


def has_signal_variation(values: list[float], *, eps: float = 1e-8) -> bool:
    return bool(values) and max(values) - min(values) > eps


def mean_top_k(values: list[float], k: int) -> float:
    if not values:
        return 0.0
    k = max(1, min(k, len(values)))
    return sum(sorted(values, reverse=True)[:k]) / k
