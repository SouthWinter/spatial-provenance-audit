"""Offline pruning baselines that do not require model hooks."""

from __future__ import annotations

from typing import Any

from recap.prune.audit import infer_task_family, infer_risk_tags
from recap.prune.budgets import fixed_keep_count, removal_fraction
from recap.prune.metrics import (
    evidence_center_recall,
    evidence_coverage,
    evidence_patch_recall,
    evidence_regions_from_sample,
    make_token_grid,
    saa_lite,
)
from recap.prune.selectors import select_indices


def build_offline_prune_baselines(
    samples: list[dict[str, Any]],
    *,
    keep_ratios: list[float],
    selectors: list[str],
    grid_rows: int = 24,
    grid_cols: int | None = None,
    seed: int = 13,
) -> list[dict[str, Any]]:
    """Build deterministic mask/evidence records for fixed-budget baselines."""
    token_boxes = make_token_grid(grid_rows, grid_cols)
    num_tokens = len(token_boxes)
    records: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(samples):
        sample_id = str(sample.get("id", sample.get("sample_id", sample_index)))
        evidence_regions = evidence_regions_from_sample(sample)
        task_family = infer_task_family(sample)
        risk_tags = infer_risk_tags(sample)
        for keep_ratio in keep_ratios:
            keep_count = fixed_keep_count(num_tokens, keep_ratio)
            for selector in selectors:
                kept = select_indices(
                    selector,
                    num_tokens=num_tokens,
                    keep_count=keep_count,
                    token_boxes=token_boxes,
                    seed=seed,
                    salt=f"{sample_id}:{keep_ratio}:{selector}",
                )
                coverage = evidence_coverage(kept, token_boxes, evidence_regions)
                center_recall = evidence_center_recall(kept, token_boxes, evidence_regions)
                patch_recall = evidence_patch_recall(kept, token_boxes, evidence_regions)
                record: dict[str, Any] = {
                    "sample_id": sample_id,
                    "dataset": sample.get("dataset", sample.get("source_dataset", "")),
                    "task_family": task_family,
                    "risk_tags": risk_tags,
                    "selector": selector,
                    "budget_type": "fixed_ratio",
                    "keep_ratio": keep_ratio,
                    "full_visual_tokens": num_tokens,
                    "kept_visual_tokens": len(kept),
                    "removal_fraction": removal_fraction(num_tokens, len(kept)),
                    "kept_indices": kept,
                    "evidence_region_count": len(evidence_regions),
                    "ecr": coverage,
                    "ecr_0_5": 1.0 if coverage >= 0.5 else 0.0,
                    "evidence_center_recall": center_recall,
                    "evidence_patch_recall": patch_recall,
                    "has_evidence": bool(evidence_regions),
                }
                if "answer_correct" in sample:
                    record["saa_lite_0_5"] = saa_lite(bool(sample["answer_correct"]), coverage, threshold=0.5)
                records.append(record)
    return records


def summarize_offline_prune_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate lightweight offline baseline records."""
    groups: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for record in records:
        key = (str(record["selector"]), float(record["keep_ratio"]))
        groups.setdefault(key, []).append(record)

    rows: list[dict[str, Any]] = []
    for (selector, keep_ratio), group in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        evidence_group = [record for record in group if record.get("has_evidence")]
        rows.append(
            {
                "selector": selector,
                "keep_ratio": keep_ratio,
                "num_samples": len(group),
                "num_evidence_samples": len(evidence_group),
                "mean_kept_visual_tokens": _mean([float(record["kept_visual_tokens"]) for record in group]),
                "mean_removal_fraction": _mean([float(record["removal_fraction"]) for record in group]),
                "mean_ecr": _mean([float(record["ecr"]) for record in evidence_group]),
                "mean_ecr_0_5": _mean([float(record["ecr_0_5"]) for record in evidence_group]),
                "mean_evidence_center_recall": _mean(
                    [float(record["evidence_center_recall"]) for record in evidence_group]
                ),
                "mean_evidence_patch_recall": _mean(
                    [float(record["evidence_patch_recall"]) for record in evidence_group]
                ),
            }
        )
    return {"num_records": len(records), "groups": rows}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
