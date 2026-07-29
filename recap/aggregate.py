"""Aggregate probe-level RECAP scores into sample-level metrics."""

from __future__ import annotations

import math
from typing import Any

from recap.metrics import aurc, average_precision, mean, roc_auc
from recap.scoring import prepare_sample_scores, score_probe

RISK_KEYS = {
    "rice": "rice_risk",
    "rice_v6_top2": "rice_v6_top2",
    "rice_v6_max": "rice_v6_max",
    "rice_v6_mean": "rice_v6_mean",
    "confidence": "confidence_risk",
    "prompt_sc": "prompt_sc_risk",
    "prompt_sc_mean": "prompt_sc_mean_risk",
    "prompt_sc_disagreement": "prompt_sc_disagreement_risk",
    "entropy": "entropy_risk",
    "text_prior": "text_prior_risk",
    "visual_lift": "visual_lift_risk",
    "visual_lift_abs": "visual_lift_abs_risk",
    "rec_relation": "rec_relation_risk",
    "rec_relation_confidence": "rec_relation_confidence_risk",
    "rec_prior": "rec_prior_risk",
    "rec_prior_dominance": "rec_prior_dominance_risk",
    "recap_evidence": "recap_evidence_risk",
    "recap_confidence": "recap_confidence_risk",
    "recap_support_penalty": "recap_support_penalty",
    "cap": "cap_risk",
    "u_conf": "u_conf",
    "c_contra": "c_contra",
    "g_prior": "g_prior",
    "e_spec": "e_spec",
    "e_spec_signed": "e_spec_signed",
    "v_vflip": "v_vflip",
    "e_vflip": "e_vflip",
    "e_vflip_signed": "e_vflip_signed",
    "v_inverse": "v_inverse",
    "e_inverse": "e_inverse",
    "e_inverse_signed": "e_inverse_signed",
    "r_struct": "r_struct",
    "v_eq": "v_eq",
    "v_rem": "v_rem",
    "v_pres": "v_pres",
    "v_contra": "v_contra",
    "role_contra": "role_contra",
    "wrong_mapping": "wrong_mapping",
    "rice_wo_v_eq": "rice_wo_v_eq",
    "rice_wo_v_rem": "rice_wo_v_rem",
    "rice_wo_v_pres": "rice_wo_v_pres",
    "rice_wo_v_contra": "rice_wo_v_contra",
    "rice_v6_wo_u_conf": "rice_v6_wo_u_conf",
    "rice_v6_wo_c_contra": "rice_v6_wo_c_contra",
    "rice_v6_wo_g_prior": "rice_v6_wo_g_prior",
    "rice_v6_wo_e_spec": "rice_v6_wo_e_spec",
    "rice_profile_mean": "rice_profile_mean",
    "rice_profile_max": "rice_profile_max",
    "rice_profile_top2": "rice_profile_top2",
    "rice_profile_wo_g_prior": "rice_profile_wo_g_prior",
    "rice_profile_wo_struct": "rice_profile_wo_struct",
    "rice_profile_lift": "rice_profile_lift",
    "rice_profile_lift_mean": "rice_profile_lift_mean",
    "rice_profile_lift_max": "rice_profile_lift_max",
    "rice_profile_lift_wo_visual": "rice_profile_lift_wo_visual",
    "rice_rec": "rice_rec",
    "rice_rec_mean": "rice_rec_mean",
    "rice_rec_max": "rice_rec_max",
    "rice_rec_wo_prior": "rice_rec_wo_prior",
    "rice_rec_wo_struct": "rice_rec_wo_struct",
    "rice_rec_wo_conf": "rice_rec_wo_conf",
    "rice_recap_selector": "rice_recap_selector",
    "rice_recap_mean": "rice_recap_mean",
    "rice_recap_wo_conf": "rice_recap_wo_conf",
}

GROUP_REPORT_RISKS = (
    "confidence",
    "prompt_sc",
    "prompt_sc_mean",
    "prompt_sc_disagreement",
    "text_prior",
    "visual_lift",
    "rec_relation",
    "rec_prior_dominance",
    "recap_evidence",
    "cap",
    "v_eq",
    "e_spec",
    "e_spec_signed",
    "e_vflip_signed",
    "e_inverse_signed",
    "role_contra",
    "r_struct",
    "rice",
    "rice_v6_top2",
    "rice_v6_max",
    "rice_v6_mean",
    "rice_v6_wo_g_prior",
    "rice_profile_mean",
    "rice_profile_wo_g_prior",
    "rice_profile_lift",
    "rice_rec",
    "rice_rec_wo_prior",
    "rice_recap_selector",
    "rice_recap_wo_conf",
)


def aggregate_scores(records: list[dict[str, Any]]) -> dict[str, Any]:
    probe_scores = [ensure_probe_score(record) for record in records]
    samples = prepare_sample_scores(probe_scores)
    labels = [sample["error"] for sample in samples]
    hallucination_labels = [sample["hallucination"] for sample in samples]

    metrics: dict[str, float] = {
        "num_samples": float(len(samples)),
        "direct_accuracy": mean([1.0 - sample["error"] for sample in samples]),
        "num_negative_samples": float(sum(1 for sample in samples if sample.get("target_is_negative"))),
        "direct_hallucination_fpr": hallucination_fpr(samples),
        "avg_inference_calls": mean([sample["probe_count"] * 2.0 for sample in samples]),
        "detector_failure_rate": detector_failure_rate(samples),
        "rice_aurc": aurc(labels, [sample["rice_risk"] for sample in samples]),
    }

    for name, key in RISK_KEYS.items():
        scores = [sample[key] for sample in samples]
        metrics[f"{name}_aurc"] = aurc(labels, scores)
        metrics[f"{name}_error_auroc"] = roc_auc(labels, scores)
        metrics[f"{name}_error_auprc"] = average_precision(labels, scores)
        metrics[f"{name}_inverted_error_auroc"] = roc_auc(labels, [-score for score in scores])
        metrics[f"{name}_inverted_error_auprc"] = average_precision(labels, [-score for score in scores])
        metrics[f"{name}_hallucination_auroc"] = roc_auc(hallucination_labels, scores)
        metrics[f"{name}_hallucination_auprc"] = average_precision(hallucination_labels, scores)

    for coverage in (0.7, 0.8, 0.9):
        suffix = int(coverage * 100)
        for name, key in RISK_KEYS.items():
            err_for_key = coverage_error(samples, coverage, risk_key=key)
            hallucination_for_key = coverage_hallucination_fpr(samples, coverage, risk_key=key)
            metrics[f"{name}_error_rate_{suffix}cov"] = err_for_key
            metrics[f"{name}_selective_accuracy_{suffix}cov"] = 1.0 - err_for_key
            metrics[f"{name}_hallucination_fpr_{suffix}cov"] = hallucination_for_key

        err = coverage_error(samples, coverage)
        inverted_err = coverage_error(samples, coverage, risk_key="rice_risk", reverse=True)
        hallucination = coverage_hallucination_fpr(samples, coverage)
        metrics[f"rice_hallucination_rate_{suffix}cov"] = err
        metrics[f"rice_selective_accuracy_{suffix}cov"] = 1.0 - err
        metrics[f"rice_hallucination_fpr_{suffix}cov"] = hallucination
        metrics[f"rice_inverted_hallucination_rate_{suffix}cov"] = inverted_err
        metrics[f"rice_inverted_selective_accuracy_{suffix}cov"] = 1.0 - inverted_err

    metrics["by_relation"] = grouped_metrics(samples, key="base_relation")
    metrics["by_relation_family"] = grouped_metrics(samples, key="relation_family")
    return {"metrics": metrics, "samples": samples, "probe_scores": probe_scores}


def ensure_probe_score(record: dict[str, Any]) -> dict[str, Any]:
    if {"margin", "pred_answer", "correct"}.issubset(record):
        return record
    if "yes_loss" not in record or "no_loss" not in record:
        raise ValueError(f"Probe result needs yes_loss/no_loss or scored fields: {record}")
    return score_probe(record, float(record["yes_loss"]), float(record["no_loss"]))


def coverage_error(samples: list[dict[str, Any]], coverage: float, *, risk_key: str = "rice_risk", reverse: bool = False) -> float:
    if not samples:
        return 0.0
    accepted_count = max(1, int(math.ceil(len(samples) * coverage)))
    accepted = sorted(samples, key=lambda sample: sample[risk_key], reverse=reverse)[:accepted_count]
    return mean([sample["error"] for sample in accepted])


def detector_failure_rate(samples: list[dict[str, Any]]) -> float:
    detector = [sample for sample in samples if sample.get("bbox_source") == "detector"]
    if not detector:
        return 0.0
    return mean([0.0 if sample.get("base_has_bbox") else 1.0 for sample in detector])


def hallucination_fpr(samples: list[dict[str, Any]]) -> float:
    negatives = [sample for sample in samples if sample.get("target_is_negative")]
    if not negatives:
        return 0.0
    return mean([sample["hallucination"] for sample in negatives])


def coverage_hallucination_fpr(samples: list[dict[str, Any]], coverage: float, *, risk_key: str = "rice_risk") -> float:
    if not samples:
        return 0.0
    accepted_count = max(1, int(math.ceil(len(samples) * coverage)))
    accepted = sorted(samples, key=lambda sample: sample[risk_key])[:accepted_count]
    return hallucination_fpr(accepted)


def grouped_metrics(samples: list[dict[str, Any]], *, key: str) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        name = str(sample.get(key, "") or "<empty>")
        groups.setdefault(name, []).append(sample)

    out: dict[str, dict[str, float]] = {}
    for name, group in sorted(groups.items()):
        labels = [sample["error"] for sample in group]
        hallucination_labels = [sample["hallucination"] for sample in group]
        row: dict[str, float] = {
            "num_samples": float(len(group)),
            "direct_accuracy": mean([1.0 - sample["error"] for sample in group]),
            "num_negative_samples": float(sum(1 for sample in group if sample.get("target_is_negative"))),
            "direct_hallucination_fpr": hallucination_fpr(group),
        }
        for risk_name in GROUP_REPORT_RISKS:
            risk_key = RISK_KEYS.get(risk_name)
            if not risk_key:
                continue
            scores = [sample[risk_key] for sample in group]
            row[f"{risk_name}_error_auroc"] = roc_auc(labels, scores)
            row[f"{risk_name}_hallucination_auroc"] = roc_auc(hallucination_labels, scores)
            row[f"{risk_name}_error_rate_80cov"] = coverage_error(group, 0.8, risk_key=risk_key)
            row[f"{risk_name}_selective_accuracy_80cov"] = 1.0 - row[f"{risk_name}_error_rate_80cov"]
            row[f"{risk_name}_hallucination_fpr_80cov"] = coverage_hallucination_fpr(group, 0.8, risk_key=risk_key)
        out[name] = row
    return out
