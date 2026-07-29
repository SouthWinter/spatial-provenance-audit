"""Compact reporting helpers for RECAP metrics."""

from __future__ import annotations

from typing import Any

RECAP_MAIN_RISKS = (
    "confidence",
    "recap_evidence",
    "rice_recap_selector",
    "rice_recap_wo_conf",
)

REC_MAIN_RISKS = (
    "confidence",
    "rec_relation",
    "rice_rec",
    "rice_rec_wo_prior",
)

PROFILE_MAIN_RISKS = (
    "confidence",
    "visual_lift",
    "rice_profile_lift",
    "rice_profile_wo_g_prior",
)

DEFAULT_MAIN_RISKS = RECAP_MAIN_RISKS

PROMPT_SC_RISKS = (
    "confidence",
    "prompt_sc",
)

DEFAULT_CHOICE_RISKS = (
    "confidence",
    "e_spec",
    "rice_choice_top2",
    "rice_choice_wo_g_prior",
)

MAIN_RELATION_FAMILIES = (
    "left_right",
    "vertical",
    "topology",
    "depth",
)


def compact_metrics(
    metrics: dict[str, Any],
    *,
    risks: list[str] | None = None,
    coverage: int = 80,
    include_by_family: bool = True,
    include_by_relation: bool = False,
    family_groups: list[str] | None = None,
) -> dict[str, Any]:
    if risks is None:
        risks = _default_risks(metrics)
    overview_keys = [
        "num_samples",
        "direct_accuracy",
        "direct_lr_accuracy",
        "direct_hallucination_fpr",
        "avg_inference_calls",
        "detector_failure_rate",
    ]
    compact: dict[str, Any] = {
        "overview": {key: metrics[key] for key in overview_keys if key in metrics},
        "coverage": coverage,
        "risks": {},
    }
    for risk in risks:
        row = {}
        for suffix in (
            f"selective_accuracy_{coverage}cov",
            f"hallucination_fpr_{coverage}cov",
            "error_auroc",
            "hallucination_auroc",
        ):
            key = f"{risk}_{suffix}"
            if key in metrics:
                row[suffix] = metrics[key]
        if row:
            compact["risks"][risk] = row
    if include_by_family and isinstance(metrics.get("by_relation_family"), dict):
        compact["by_relation_family"] = _compact_groups(
            metrics["by_relation_family"],
            risks=risks,
            coverage=coverage,
            group_names=family_groups or list(MAIN_RELATION_FAMILIES),
        )
    if include_by_relation and isinstance(metrics.get("by_relation"), dict):
        compact["by_relation"] = _compact_groups(metrics["by_relation"], risks=risks, coverage=coverage)
    return compact


def _default_risks(metrics: dict[str, Any]) -> list[str]:
    if any(key.startswith("prompt_sc_") for key in metrics) and not any(
        key.startswith("recap_evidence_") for key in metrics
    ):
        return [risk for risk in PROMPT_SC_RISKS if any(key.startswith(f"{risk}_") for key in metrics)]
    if any(key.startswith("rice_choice_") for key in metrics):
        return [risk for risk in DEFAULT_CHOICE_RISKS if any(key.startswith(f"{risk}_") for key in metrics)]
    if any(key.startswith("rice_recap_") for key in metrics) or any(key.startswith("recap_evidence_") for key in metrics):
        return [risk for risk in RECAP_MAIN_RISKS if any(key.startswith(f"{risk}_") for key in metrics)]
    if any(key.startswith("rice_rec_") for key in metrics) or any(key.startswith("rec_relation_") for key in metrics):
        return [risk for risk in REC_MAIN_RISKS if any(key.startswith(f"{risk}_") for key in metrics)]
    return [risk for risk in PROFILE_MAIN_RISKS if any(key.startswith(f"{risk}_") for key in metrics)]


def _compact_groups(
    groups: dict[str, Any],
    *,
    risks: list[str],
    coverage: int,
    group_names: list[str] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    overview_keys = ("num_samples", "direct_accuracy", "direct_hallucination_fpr")
    items = ((name, groups.get(name)) for name in group_names) if group_names else groups.items()
    for group_name, metrics in items:
        if not isinstance(metrics, dict):
            continue
        group_row: dict[str, Any] = {
            "overview": {key: metrics[key] for key in overview_keys if key in metrics},
            "risks": {},
        }
        for risk in risks:
            risk_row = {}
            for suffix in (
                f"selective_accuracy_{coverage}cov",
                f"hallucination_fpr_{coverage}cov",
                "error_auroc",
                "hallucination_auroc",
            ):
                key = f"{risk}_{suffix}"
                if key in metrics:
                    risk_row[suffix] = metrics[key]
            if risk_row:
                group_row["risks"][risk] = risk_row
        out[group_name] = group_row
    return out
