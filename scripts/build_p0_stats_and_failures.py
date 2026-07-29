#!/usr/bin/env python
"""Build P0 statistical evidence and qualitative case exports.

The script consumes cached sample-level JSONL outputs from completed pruning
runs. It does not run any model inference.
"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "p0_stats"
BOOTSTRAP = 5000
SEED = 20260704


@dataclass(frozen=True)
class RunSpec:
    key: str
    model: str
    method: str
    path: Path
    family: str


@dataclass(frozen=True)
class PairSpec:
    key: str
    left: str
    right: str
    family: str
    claim: str
    note: str = ""


RUNS: dict[str, RunSpec] = {
    "qwen_full": RunSpec(
        "qwen_full",
        "Qwen3-VL-8B",
        "full",
        Path("runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_topk_1p00"),
        "TextOCR-Hard",
    ),
    "qwen_target0p20": RunSpec(
        "qwen_target0p20",
        "Qwen3-VL-8B",
        "ours_target0p20",
        Path("runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_topk_0p20"),
        "TextOCR-Hard",
    ),
    "qwen_target0p30": RunSpec(
        "qwen_target0p30",
        "Qwen3-VL-8B",
        "ours_target0p30",
        Path("runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_topk_0p30_targetfix_802816"),
        "TextOCR-Hard",
    ),
    "qwen_fastv_proxy0p30": RunSpec(
        "qwen_fastv_proxy0p30",
        "Qwen3-VL-8B",
        "FastV/TopV-style proxy0p30",
        Path("runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_embed_topk_0p30"),
        "TextOCR-Hard",
    ),
    "qwen_random0p20": RunSpec(
        "qwen_random0p20",
        "Qwen3-VL-8B",
        "random0p20",
        Path("runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_random_0p20"),
        "TextOCR-Hard",
    ),
    "qwen_grid0p20": RunSpec(
        "qwen_grid0p20",
        "Qwen3-VL-8B",
        "grid0p20",
        Path("runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_grid_0p20"),
        "TextOCR-Hard",
    ),
    "qwen_shuffled0p20": RunSpec(
        "qwen_shuffled0p20",
        "Qwen3-VL-8B",
        "shuffled_score0p20",
        Path("runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_shuffled_topk_0p20"),
        "TextOCR-Hard",
    ),
    "qwen_random0p30": RunSpec(
        "qwen_random0p30",
        "Qwen3-VL-8B",
        "random0p30",
        Path("runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_random_0p30"),
        "TextOCR-Hard",
    ),
    "qwen_visionzip_proxy0p30": RunSpec(
        "qwen_visionzip_proxy0p30",
        "Qwen3-VL-8B",
        "VisionZip-style proxy0p30",
        Path("runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_embed_rise_0p30"),
        "TextOCR-Hard",
    ),
    "qwen_shuffled0p30": RunSpec(
        "qwen_shuffled0p30",
        "Qwen3-VL-8B",
        "shuffled_score0p30",
        Path("runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_shuffled_topk_0p30"),
        "TextOCR-Hard",
    ),
    "qwen_grid0p30": RunSpec(
        "qwen_grid0p30",
        "Qwen3-VL-8B",
        "grid0p30",
        Path("runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_grid_0p30"),
        "TextOCR-Hard",
    ),
    "llava_full": RunSpec(
        "llava_full",
        "LLaVA-1.5-7B",
        "full",
        Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_direct"),
        "TextOCR-Hard",
    ),
    "llava_embed0p40": RunSpec(
        "llava_embed0p40",
        "LLaVA-1.5-7B",
        "FastV/TopV-style proxy0p40",
        Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_embed_topk_0p40"),
        "TextOCR-Hard",
    ),
    "llava_protected0p40": RunSpec(
        "llava_protected0p40",
        "LLaVA-1.5-7B",
        "ours_protected_embed0p40",
        Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_embed_protected_topk_0p40"),
        "TextOCR-Hard",
    ),
    "llava_target0p40": RunSpec(
        "llava_target0p40",
        "LLaVA-1.5-7B",
        "ours_target0p40",
        Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_target_embed_topk_0p40_targetfix"),
        "TextOCR-Hard",
    ),
    "llava_grid0p40": RunSpec(
        "llava_grid0p40",
        "LLaVA-1.5-7B",
        "grid0p40",
        Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_grid_0p40"),
        "TextOCR-Hard",
    ),
    "llava_random0p40": RunSpec(
        "llava_random0p40",
        "LLaVA-1.5-7B",
        "random0p40",
        Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_random_0p40"),
        "TextOCR-Hard",
    ),
    "llava_shuffled0p40": RunSpec(
        "llava_shuffled0p40",
        "LLaVA-1.5-7B",
        "shuffled_score0p40",
        Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_embed_shuffled_topk_0p40"),
        "TextOCR-Hard",
    ),
    "llava_visionzip0p40": RunSpec(
        "llava_visionzip0p40",
        "LLaVA-1.5-7B",
        "VisionZip official-port0p40",
        Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_visionzip_0p40"),
        "TextOCR-Hard",
    ),
    "llava_fastv0p40": RunSpec(
        "llava_fastv0p40",
        "LLaVA-1.5-7B",
        "FastV official-port0p40",
        Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_fastv_0p40"),
        "TextOCR-Hard",
    ),
    "llava_scope0p40": RunSpec(
        "llava_scope0p40",
        "LLaVA-1.5-7B",
        "SCOPE official-port0p40",
        Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_scope_0p40"),
        "TextOCR-Hard",
    ),
    "llava_coin0p40": RunSpec(
        "llava_coin0p40",
        "LLaVA-1.5-7B",
        "CoIn paper-port0p40",
        Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_coin_0p40"),
        "TextOCR-Hard",
    ),
    "internvl_full_cal": RunSpec(
        "internvl_full_cal",
        "InternVL3.5-8B",
        "full_cal",
        Path("runs/internvl_textocr_hard/calibrated_test_full_devthr"),
        "TextOCR-Hard calibrated-test",
    ),
    "internvl_soft_hfpr0p50": RunSpec(
        "internvl_soft_hfpr0p50",
        "InternVL3.5-8B",
        "ours_soft_evidence0p50_hfpr_cal",
        Path("runs/internvl_textocr_hard/calibrated_test_target_soft_evidence0p50_b0p05_hfprconstr_devthr"),
        "TextOCR-Hard calibrated-test",
    ),
    "internvl_target0p50": RunSpec(
        "internvl_target0p50",
        "InternVL3.5-8B",
        "target0p50_cal",
        Path("runs/internvl_textocr_hard/calibrated_test_target0p50_devthr"),
        "TextOCR-Hard calibrated-test",
    ),
    "internvl_grid0p50": RunSpec(
        "internvl_grid0p50",
        "InternVL3.5-8B",
        "grid0p50_cal",
        Path("runs/internvl_textocr_hard/calibrated_test_grid0p50_devthr"),
        "TextOCR-Hard calibrated-test",
    ),
    "internvl_random0p50": RunSpec(
        "internvl_random0p50",
        "InternVL3.5-8B",
        "random0p50_cal",
        Path("runs/internvl_textocr_hard/calibrated_test_random0p50_devthr"),
        "TextOCR-Hard calibrated-test",
    ),
    "internvl_shuffled0p50": RunSpec(
        "internvl_shuffled0p50",
        "InternVL3.5-8B",
        "shuffled_score0p50_cal",
        Path("runs/internvl_textocr_hard/calibrated_test_target_shuffled0p50_devthr"),
        "TextOCR-Hard calibrated-test",
    ),
    "internvl_embed0p50": RunSpec(
        "internvl_embed0p50",
        "InternVL3.5-8B",
        "FastV/TopV-style proxy0p50_cal",
        Path("runs/internvl_textocr_hard/calibrated_test_embed0p50_devthr"),
        "TextOCR-Hard calibrated-test",
    ),
    "llava_gsr_direct": RunSpec(
        "llava_gsr_direct",
        "LLaVA-1.5-7B",
        "direct_full",
        Path("runs/rice_v5/llava15_7b_profile_fast_gsrbench_coco_spatial_two"),
        "GSR-Bench spatial",
    ),
    "llava_gsr_grid0p40": RunSpec(
        "llava_gsr_grid0p40",
        "LLaVA-1.5-7B",
        "grid0p40",
        Path("runs/llava_prune/llava15_7b_gsr_coco_spatial_profile_fast_grid_0p40"),
        "GSR-Bench spatial",
    ),
    "llava_gsr_spatial0p40": RunSpec(
        "llava_gsr_spatial0p40",
        "LLaVA-1.5-7B",
        "spatial_c35_ctx25_0p40",
        Path("runs/llava_prune/llava15_7b_gsr_coco_spatial_profile_fast_spatial_aware_0p40_limit100"),
        "GSR-Bench spatial",
    ),
    "llava_gsr_spatial_c10_0p40": RunSpec(
        "llava_gsr_spatial_c10_0p40",
        "LLaVA-1.5-7B",
        "spatial_c10_ctx10_0p40",
        Path("runs/llava_prune/llava15_7b_gsr_coco_spatial_profile_fast_spatial_aware_c10_ctx10_0p40_limit100"),
        "GSR-Bench spatial",
    ),
    "internvl_gsr_direct": RunSpec(
        "internvl_gsr_direct",
        "InternVL3.5-8B",
        "direct_full",
        Path("runs/rice_v5/internvl3_5_8b_profile_fast_gsrbench_coco_spatial_two"),
        "GSR-Bench spatial",
    ),
    "internvl_gsr_grid0p50": RunSpec(
        "internvl_gsr_grid0p50",
        "InternVL3.5-8B",
        "grid0p50",
        Path("runs/internvl_prune/internvl35_8b_gsr_coco_spatial_profile_fast_grid_0p50"),
        "GSR-Bench spatial",
    ),
    "internvl_gsr_target0p50": RunSpec(
        "internvl_gsr_target0p50",
        "InternVL3.5-8B",
        "target_embed_topk0p50",
        Path("runs/internvl_prune/internvl35_8b_gsr_coco_spatial_profile_fast_target_embed_topk_0p50"),
        "GSR-Bench spatial",
    ),
    "internvl_gsr_spatial0p50": RunSpec(
        "internvl_gsr_spatial0p50",
        "InternVL3.5-8B",
        "spatial_c35_ctx25_0p50",
        Path("runs/internvl_prune/internvl35_8b_gsr_coco_spatial_profile_fast_spatial_aware_0p50_limit100"),
        "GSR-Bench spatial",
    ),
    "internvl_gsr_spatial_c10_0p50": RunSpec(
        "internvl_gsr_spatial_c10_0p50",
        "InternVL3.5-8B",
        "spatial_c10_ctx10_0p50",
        Path("runs/internvl_prune/internvl35_8b_gsr_coco_spatial_profile_fast_spatial_aware_c10_ctx10_0p50_limit100"),
        "GSR-Bench spatial",
    ),
}


PAIRS = [
    PairSpec("qwen_target0p20_vs_full", "qwen_target0p20", "qwen_full", "TextOCR-Hard", "extreme-efficiency vs full"),
    PairSpec("qwen_target0p20_vs_random", "qwen_target0p20", "qwen_random0p20", "TextOCR-Hard", "same-budget random"),
    PairSpec("qwen_target0p20_vs_grid", "qwen_target0p20", "qwen_grid0p20", "TextOCR-Hard", "same-budget spatial baseline"),
    PairSpec("qwen_target0p20_vs_shuffled", "qwen_target0p20", "qwen_shuffled0p20", "TextOCR-Hard", "same-score shuffled baseline"),
    PairSpec("qwen_target0p30_vs_full", "qwen_target0p30", "qwen_full", "TextOCR-Hard", "quality point vs full"),
    PairSpec("qwen_target0p30_vs_fastv_proxy", "qwen_target0p30", "qwen_fastv_proxy0p30", "TextOCR-Hard", "proxy baseline"),
    PairSpec("qwen_target0p30_vs_visionzip_proxy", "qwen_target0p30", "qwen_visionzip_proxy0p30", "TextOCR-Hard", "proxy baseline"),
    PairSpec("qwen_target0p30_vs_grid", "qwen_target0p30", "qwen_grid0p30", "TextOCR-Hard", "same-budget spatial baseline"),
    PairSpec("qwen_target0p30_vs_random", "qwen_target0p30", "qwen_random0p30", "TextOCR-Hard", "same-budget random"),
    PairSpec("qwen_target0p30_vs_shuffled", "qwen_target0p30", "qwen_shuffled0p30", "TextOCR-Hard", "same-score shuffled baseline"),
    PairSpec("llava_protected0p40_vs_full", "llava_protected0p40", "llava_full", "TextOCR-Hard", "evidence-protected vs full"),
    PairSpec("llava_protected0p40_vs_embed_proxy", "llava_protected0p40", "llava_embed0p40", "TextOCR-Hard", "proxy baseline"),
    PairSpec("llava_protected0p40_vs_grid", "llava_protected0p40", "llava_grid0p40", "TextOCR-Hard", "same-budget spatial baseline"),
    PairSpec("llava_protected0p40_vs_random", "llava_protected0p40", "llava_random0p40", "TextOCR-Hard", "same-budget random"),
    PairSpec("llava_protected0p40_vs_shuffled", "llava_protected0p40", "llava_shuffled0p40", "TextOCR-Hard", "same-score shuffled baseline"),
    PairSpec("llava_protected0p40_vs_visionzip_official", "llava_protected0p40", "llava_visionzip0p40", "TextOCR-Hard", "official-algorithm port"),
    PairSpec("llava_protected0p40_vs_fastv_official", "llava_protected0p40", "llava_fastv0p40", "TextOCR-Hard", "official-algorithm port"),
    PairSpec("llava_protected0p40_vs_scope_official", "llava_protected0p40", "llava_scope0p40", "TextOCR-Hard", "box-assisted vs official-algorithm port"),
    PairSpec("llava_target0p40_vs_scope_official", "llava_target0p40", "llava_scope0p40", "TextOCR-Hard", "box-free vs official-algorithm port"),
    PairSpec("llava_scope0p40_vs_visionzip_official", "llava_scope0p40", "llava_visionzip0p40", "TextOCR-Hard", "recent vs legacy official-algorithm port"),
    PairSpec("llava_protected0p40_vs_coin_paper", "llava_protected0p40", "llava_coin0p40", "TextOCR-Hard", "box-assisted vs paper-algorithm port"),
    PairSpec("llava_target0p40_vs_coin_paper", "llava_target0p40", "llava_coin0p40", "TextOCR-Hard", "box-free vs paper-algorithm port"),
    PairSpec("llava_coin0p40_vs_scope_official", "llava_coin0p40", "llava_scope0p40", "TextOCR-Hard", "paper-algorithm port vs official-algorithm port"),
    PairSpec("internvl_soft0p50_vs_full_cal", "internvl_soft_hfpr0p50", "internvl_full_cal", "TextOCR-Hard calibrated-test", "risk-constrained vs full"),
    PairSpec("internvl_soft0p50_vs_target0p50", "internvl_soft_hfpr0p50", "internvl_target0p50", "TextOCR-Hard calibrated-test", "soft evidence vs hard target"),
    PairSpec("internvl_soft0p50_vs_grid", "internvl_soft_hfpr0p50", "internvl_grid0p50", "TextOCR-Hard calibrated-test", "same-budget spatial baseline"),
    PairSpec("internvl_soft0p50_vs_random", "internvl_soft_hfpr0p50", "internvl_random0p50", "TextOCR-Hard calibrated-test", "same-budget random"),
    PairSpec("internvl_soft0p50_vs_shuffled", "internvl_soft_hfpr0p50", "internvl_shuffled0p50", "TextOCR-Hard calibrated-test", "same-score shuffled baseline"),
    PairSpec("internvl_soft0p50_vs_embed_proxy", "internvl_soft_hfpr0p50", "internvl_embed0p50", "TextOCR-Hard calibrated-test", "proxy baseline"),
    PairSpec("llava_gsr_spatial_c35_vs_grid_same100", "llava_gsr_spatial0p40", "llava_gsr_grid0p40", "GSR-Bench spatial", "spatial-aware stress"),
    PairSpec("llava_gsr_spatial_c10_vs_grid_same100", "llava_gsr_spatial_c10_0p40", "llava_gsr_grid0p40", "GSR-Bench spatial", "spatial-aware stress"),
    PairSpec("llava_gsr_spatial_c35_vs_direct_same100", "llava_gsr_spatial0p40", "llava_gsr_direct", "GSR-Bench spatial", "spatial-aware stress"),
    PairSpec("internvl_gsr_spatial_c35_vs_grid_same100", "internvl_gsr_spatial0p50", "internvl_gsr_grid0p50", "GSR-Bench spatial", "spatial-aware stress"),
    PairSpec("internvl_gsr_spatial_c10_vs_grid_same100", "internvl_gsr_spatial_c10_0p50", "internvl_gsr_grid0p50", "GSR-Bench spatial", "spatial-aware stress"),
    PairSpec("internvl_gsr_spatial_c35_vs_direct_same100", "internvl_gsr_spatial0p50", "internvl_gsr_direct", "GSR-Bench spatial", "spatial-aware stress"),
    PairSpec("internvl_gsr_spatial_c35_vs_target_same100", "internvl_gsr_spatial0p50", "internvl_gsr_target0p50", "GSR-Bench spatial", "spatial-aware stress"),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    loaded = {key: load_run(spec) for key, spec in RUNS.items()}

    method_rows = []
    for index, (key, run) in enumerate(loaded.items()):
        method_rows.append(method_summary(RUNS[key], run, seed=SEED + index))

    pair_rows = []
    for index, pair in enumerate(PAIRS):
        pair_rows.append(compare_pair(pair, loaded[pair.left], loaded[pair.right], seed=SEED + 100 + index))

    write_csv(OUT_DIR / "method_ci.csv", method_rows)
    write_csv(OUT_DIR / "pairwise_stats.csv", pair_rows)
    write_case_exports(loaded)
    write_report(method_rows, pair_rows)
    print(f"Wrote P0 report to {OUT_DIR / 'p0_stats_report.md'}")
    print(f"Wrote paired statistics to {OUT_DIR / 'pairwise_stats.csv'}")


def load_run(spec: RunSpec) -> dict[str, Any]:
    run_dir = ROOT / spec.path
    score_path = run_dir / "sample_scores.jsonl"
    if not score_path.exists():
        raise FileNotFoundError(score_path)
    return {
        "scores": load_jsonl_by_id(score_path, "sample_id"),
        "probe_scores": load_optional_jsonl_by_id(run_dir / "probe_scores.jsonl", "sample_id"),
        "probes": load_optional_jsonl_by_id(run_dir / "probes.jsonl", "sample_id"),
        "traces": load_optional_jsonl_by_id(run_dir / "prune_traces.jsonl", "sample_id"),
    }


def load_jsonl_by_id(path: Path, key: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row[key])] = row
    return rows


def load_optional_jsonl_by_id(path: Path, key: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return load_jsonl_by_id(path, key)


def method_summary(spec: RunSpec, run: dict[str, Any], *, seed: int) -> dict[str, Any]:
    rows = list(run["scores"].values())
    trace_rows = list(run["traces"].values()) or list(run["probe_scores"].values())
    neg = [row for row in rows if is_negative(row)]
    pos = [row for row in rows if not is_negative(row)]
    acc_values = [as_float(row.get("direct_correct")) for row in rows]
    hfpr_values = [as_float(row.get("hallucination")) for row in neg]
    acc_ci = bootstrap_ci(acc_values, seed=seed)
    hfpr_ci = bootstrap_ci(hfpr_values, seed=seed + 1)
    return {
        "key": spec.key,
        "model": spec.model,
        "family": spec.family,
        "method": spec.method,
        "n": len(rows),
        "n_negative": len(neg),
        "acc": mean(acc_values),
        "acc_ci_low": acc_ci[0],
        "acc_ci_high": acc_ci[1],
        "hFPR": mean(hfpr_values),
        "hFPR_ci_low": hfpr_ci[0],
        "hFPR_ci_high": hfpr_ci[1],
        "yes_rate": mean([1.0 if str(row.get("direct_pred", "")).lower() == "yes" else 0.0 for row in rows]),
        "pos_acc": mean([as_float(row.get("direct_correct")) for row in pos]),
        "neg_acc": mean([as_float(row.get("direct_correct")) for row in neg]),
        **trace_summary(trace_rows),
        "path": str(spec.path),
    }


def trace_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "keep": "",
            "ECR": "",
            "CenterR": "",
            "PatchR": "",
            "kept_visual": "",
            "full_visual": "",
        }
    evidence_rows = [row for row in rows if row.get("has_evidence") or has_metric(row, "ecr")]
    full_visual = mean_optional([metric(row, "full_visual_tokens") for row in rows])
    kept_visual = mean_optional([metric(row, "kept_visual_tokens") for row in rows])
    keep = kept_visual / full_visual if is_number(full_visual) and is_number(kept_visual) and full_visual else ""
    return {
        "keep": keep,
        "ECR": mean_optional([metric(row, "ecr") for row in evidence_rows]),
        "CenterR": mean_optional([metric(row, "evidence_center_recall") for row in evidence_rows]),
        "PatchR": mean_optional([metric(row, "evidence_patch_recall") for row in evidence_rows]),
        "kept_visual": kept_visual,
        "full_visual": full_visual,
    }


def compare_pair(pair: PairSpec, left: dict[str, Any], right: dict[str, Any], *, seed: int) -> dict[str, Any]:
    left_ids = set(left["scores"])
    right_ids = set(right["scores"])
    ids = sorted(left_ids & right_ids)
    if not ids:
        raise ValueError(f"No overlap for {pair.key}")
    rows = [(left["scores"][sid], right["scores"][sid]) for sid in ids]
    neg_rows = [(lrow, rrow) for lrow, rrow in rows if is_negative(lrow) or is_negative(rrow)]

    acc_left_values = [as_float(lrow.get("direct_correct")) for lrow, _ in rows]
    acc_right_values = [as_float(rrow.get("direct_correct")) for _, rrow in rows]
    acc_diffs = [l - r for l, r in zip(acc_left_values, acc_right_values)]
    hfpr_left_values = [as_float(lrow.get("hallucination")) for lrow, _ in neg_rows]
    hfpr_right_values = [as_float(rrow.get("hallucination")) for _, rrow in neg_rows]
    hfpr_diffs = [l - r for l, r in zip(hfpr_left_values, hfpr_right_values)]

    acc_ci = bootstrap_ci(acc_diffs, seed=seed)
    hfpr_ci = bootstrap_ci(hfpr_diffs, seed=seed + 1)
    acc_b, acc_c = discordance(rows, key="direct_correct")
    hfpr_b, hfpr_c = discordance(neg_rows, key="hallucination")
    left_spec = RUNS[pair.left]
    right_spec = RUNS[pair.right]
    return {
        "comparison": pair.key,
        "family": pair.family,
        "claim": pair.claim,
        "left_model": left_spec.model,
        "left_method": left_spec.method,
        "right_model": right_spec.model,
        "right_method": right_spec.method,
        "n_overlap": len(rows),
        "n_left_only": len(left_ids - right_ids),
        "n_right_only": len(right_ids - left_ids),
        "n_negative": len(neg_rows),
        "left_acc": mean(acc_left_values),
        "right_acc": mean(acc_right_values),
        "acc_diff": mean(acc_diffs),
        "acc_diff_ci_low": acc_ci[0],
        "acc_diff_ci_high": acc_ci[1],
        "acc_sign_p": exact_sign_p(acc_b, acc_c),
        "acc_left_better": acc_b,
        "acc_left_worse": acc_c,
        "left_hFPR": mean(hfpr_left_values),
        "right_hFPR": mean(hfpr_right_values),
        "hFPR_diff": mean(hfpr_diffs),
        "hFPR_diff_ci_low": hfpr_ci[0],
        "hFPR_diff_ci_high": hfpr_ci[1],
        "hFPR_sign_p": exact_sign_p(hfpr_b, hfpr_c),
        "hFPR_left_worse": hfpr_b,
        "hFPR_left_better": hfpr_c,
        "note": pair.note,
    }


def discordance(rows: list[tuple[dict[str, Any], dict[str, Any]]], *, key: str) -> tuple[int, int]:
    left_true_right_false = 0
    right_true_left_false = 0
    for left, right in rows:
        lval = bool(left.get(key, False))
        rval = bool(right.get(key, False))
        if lval and not rval:
            left_true_right_false += 1
        elif rval and not lval:
            right_true_left_false += 1
    return left_true_right_false, right_true_left_false


def write_case_exports(loaded: dict[str, dict[str, Any]]) -> None:
    textocr_pairs = [
        ("qwen_target0p30_vs_full", "qwen_target0p30", "qwen_full"),
        ("qwen_target0p30_vs_fastv_proxy", "qwen_target0p30", "qwen_fastv_proxy0p30"),
        ("qwen_target0p30_vs_visionzip_proxy", "qwen_target0p30", "qwen_visionzip_proxy0p30"),
        ("llava_protected0p40_vs_full", "llava_protected0p40", "llava_full"),
        ("llava_protected0p40_vs_visionzip_official", "llava_protected0p40", "llava_visionzip0p40"),
        ("llava_protected0p40_vs_fastv_official", "llava_protected0p40", "llava_fastv0p40"),
        ("internvl_soft0p50_vs_full_cal", "internvl_soft_hfpr0p50", "internvl_full_cal"),
        ("internvl_soft0p50_vs_embed_proxy", "internvl_soft_hfpr0p50", "internvl_embed0p50"),
    ]
    gsr_pairs = [
        ("llava_gsr_spatial_c35_vs_grid_same100", "llava_gsr_spatial0p40", "llava_gsr_grid0p40"),
        ("llava_gsr_spatial_c10_vs_grid_same100", "llava_gsr_spatial_c10_0p40", "llava_gsr_grid0p40"),
        ("internvl_gsr_spatial_c35_vs_grid_same100", "internvl_gsr_spatial0p50", "internvl_gsr_grid0p50"),
        ("internvl_gsr_spatial_c10_vs_grid_same100", "internvl_gsr_spatial_c10_0p50", "internvl_gsr_grid0p50"),
        ("internvl_gsr_spatial_c35_vs_direct_same100", "internvl_gsr_spatial0p50", "internvl_gsr_direct"),
    ]

    success_rows: list[dict[str, Any]] = []
    textocr_failure_rows: list[dict[str, Any]] = []
    gsr_failure_rows: list[dict[str, Any]] = []

    for label, left_key, right_key in textocr_pairs:
        success_rows.extend(case_rows(label, left_key, right_key, loaded, outcome="win", limit=25))
        textocr_failure_rows.extend(case_rows(label, left_key, right_key, loaded, outcome="loss", limit=25))
    for label, left_key, right_key in gsr_pairs:
        gsr_failure_rows.extend(case_rows(label, left_key, right_key, loaded, outcome="loss_or_hfpr_loss", limit=30))

    write_jsonl(OUT_DIR / "textocr_success_cases.jsonl", success_rows)
    write_jsonl(OUT_DIR / "textocr_failure_cases.jsonl", textocr_failure_rows)
    write_jsonl(OUT_DIR / "gsr_failure_cases.jsonl", gsr_failure_rows)
    write_failure_index(success_rows, textocr_failure_rows, gsr_failure_rows)


def case_rows(
    label: str,
    left_key: str,
    right_key: str,
    loaded: dict[str, dict[str, Any]],
    *,
    outcome: str,
    limit: int,
) -> list[dict[str, Any]]:
    left = loaded[left_key]
    right = loaded[right_key]
    ids = sorted(set(left["scores"]) & set(right["scores"]))
    rows = []
    for sid in ids:
        lscore = left["scores"][sid]
        rscore = right["scores"][sid]
        left_correct = bool(lscore.get("direct_correct"))
        right_correct = bool(rscore.get("direct_correct"))
        left_hall = bool(lscore.get("hallucination"))
        right_hall = bool(rscore.get("hallucination"))
        if outcome == "win":
            keep = left_correct and not right_correct
        elif outcome == "loss":
            keep = right_correct and not left_correct
        elif outcome == "loss_or_hfpr_loss":
            keep = (right_correct and not left_correct) or (is_negative(lscore) and left_hall and not right_hall)
        else:
            raise ValueError(outcome)
        if not keep:
            continue
        rows.append(build_case_row(label, sid, left_key, right_key, left, right))
        if len(rows) >= limit:
            break
    return rows


def build_case_row(
    label: str,
    sid: str,
    left_key: str,
    right_key: str,
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    lscore = left["scores"][sid]
    rscore = right["scores"][sid]
    probe = left["probes"].get(sid) or right["probes"].get(sid) or {}
    ltrace = left["traces"].get(sid) or left["probe_scores"].get(sid, {})
    rtrace = right["traces"].get(sid) or right["probe_scores"].get(sid, {})
    return {
        "comparison": label,
        "sample_id": sid,
        "model": RUNS[left_key].model,
        "left_method": RUNS[left_key].method,
        "right_method": RUNS[right_key].method,
        "target": lscore.get("direct_target", ""),
        "target_is_negative": bool(lscore.get("target_is_negative", False)),
        "left_pred": lscore.get("direct_pred", ""),
        "right_pred": rscore.get("direct_pred", ""),
        "left_correct": bool(lscore.get("direct_correct")),
        "right_correct": bool(rscore.get("direct_correct")),
        "left_hallucination": bool(lscore.get("hallucination", False)),
        "right_hallucination": bool(rscore.get("hallucination", False)),
        "base_relation": probe.get("base_relation", lscore.get("base_relation", "")),
        "relation_family": lscore.get("relation_family", probe.get("relation", "")),
        "binary_polarity": probe.get("binary_polarity", ""),
        "question": probe.get("question", ""),
        "image": probe.get("image", ""),
        "subject": probe.get("subject", ""),
        "object": probe.get("object", ""),
        "bbox_source": probe.get("bbox_source", lscore.get("bbox_source", "")),
        "evidence_region_count": probe.get("evidence_region_count", ""),
        "left_ecr": metric(ltrace, "ecr"),
        "left_center_recall": metric(ltrace, "evidence_center_recall"),
        "left_patch_recall": metric(ltrace, "evidence_patch_recall"),
        "right_ecr": metric(rtrace, "ecr"),
        "right_center_recall": metric(rtrace, "evidence_center_recall"),
        "right_patch_recall": metric(rtrace, "evidence_patch_recall"),
        "left_keep": trace_keep(ltrace),
        "right_keep": trace_keep(rtrace),
    }


def trace_keep(trace: dict[str, Any]) -> Any:
    full = metric(trace, "full_visual_tokens")
    kept = metric(trace, "kept_visual_tokens")
    if is_number(full) and is_number(kept) and full:
        return kept / full
    return ""


def write_failure_index(
    success_rows: list[dict[str, Any]],
    textocr_failure_rows: list[dict[str, Any]],
    gsr_failure_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# P0 Qualitative Case Index",
        "",
        "These JSONL exports are deterministic first-match slices from paired comparisons. They are for manual inspection and figure/case selection, not new metrics.",
        "",
        "| export | rows | purpose |",
        "|---|---:|---|",
        f"| textocr_success_cases.jsonl | {len(success_rows)} | proposed method correct while baseline is wrong |",
        f"| textocr_failure_cases.jsonl | {len(textocr_failure_rows)} | baseline correct while proposed method is wrong |",
        f"| gsr_failure_cases.jsonl | {len(gsr_failure_rows)} | spatial-aware pruning fails or raises hFPR relative to baseline |",
        "",
    ]
    for title, rows in [
        ("TextOCR Success Examples", success_rows),
        ("TextOCR Failure Examples", textocr_failure_rows),
        ("GSR Spatial Failure Examples", gsr_failure_rows),
    ]:
        lines.extend([f"## {title}", "", "| comparison | sample_id | relation | target | left | right |", "|---|---|---|---|---|---|"])
        for row in rows[:15]:
            lines.append(
                "| {comparison} | {sample_id} | {relation} | {target} | {left} | {right} |".format(
                    comparison=row["comparison"],
                    sample_id=row["sample_id"],
                    relation=row.get("base_relation", ""),
                    target=row.get("target", ""),
                    left=row.get("left_pred", ""),
                    right=row.get("right_pred", ""),
                )
            )
        lines.append("")
    (OUT_DIR / "failure_case_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(method_rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# P0 Statistical Evidence and Failure Cases",
        "",
        f"Bootstrap samples: {BOOTSTRAP}",
        f"Seed: {SEED}",
        "",
        "Notes: accuracy differences are `left - right`; hFPR differences are also `left - right`, so lower is better for hFPR. Sign-test p-values use paired discordant samples.",
        "",
        "## Main Paired Comparisons",
        "",
        "| comparison | n | left | right | acc diff | acc 95% CI | acc p | hFPR diff | hFPR 95% CI | hFPR p | reading |",
        "|---|---:|---|---|---:|---|---:|---:|---|---:|---|",
    ]
    for row in pair_rows:
        if row["family"].startswith("GSR"):
            continue
        lines.append(pair_table_line(row))
    lines.extend(
        [
            "",
            "## Spatial Stress Comparisons",
            "",
            "| comparison | n | left | right | acc diff | acc 95% CI | acc p | hFPR diff | hFPR 95% CI | hFPR p | reading |",
            "|---|---:|---|---|---:|---|---:|---:|---|---:|---|",
        ]
    )
    for row in pair_rows:
        if row["family"].startswith("GSR"):
            lines.append(pair_table_line(row))

    lines.extend(
        [
            "",
            "## Method CIs",
            "",
            "| model | method | family | n | acc | acc 95% CI | hFPR | hFPR 95% CI | keep | ECR | CenterR | PatchR |",
            "|---|---|---|---:|---:|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in method_rows:
        if not include_method_in_report(row):
            continue
        lines.append(
            "| {model} | {method} | {family} | {n} | {acc} | [{acc_l}, {acc_h}] | {hfpr} | [{hfpr_l}, {hfpr_h}] | {keep} | {ecr} | {center} | {patch} |".format(
                model=row["model"],
                method=row["method"],
                family=row["family"],
                n=row["n"],
                acc=fmt(row["acc"]),
                acc_l=fmt(row["acc_ci_low"]),
                acc_h=fmt(row["acc_ci_high"]),
                hfpr=fmt(row["hFPR"]),
                hfpr_l=fmt(row["hFPR_ci_low"]),
                hfpr_h=fmt(row["hFPR_ci_high"]),
                keep=fmt(row["keep"]),
                ecr=fmt(row["ECR"]),
                center=fmt(row["CenterR"]),
                patch=fmt(row["PatchR"]),
            )
        )
    lines.extend(
        [
            "",
            "## Qualitative Exports",
            "",
            "- `textocr_success_cases.jsonl`: proposed method correct while compared baseline is wrong.",
            "- `textocr_failure_cases.jsonl`: compared baseline correct while proposed method is wrong.",
            "- `gsr_failure_cases.jsonl`: spatial-aware pruning failure/hFPR-loss cases.",
            "- `failure_case_index.md`: short deterministic index for manual inspection.",
            "",
            "## Evidence Reading",
            "",
            "- Strongest positive statistical evidence: Qwen `target0p30` is accuracy-tied with full while improving accuracy over proxy/same-budget baselines; LLaVA protected pruning clearly beats the official VisionZip/FastV ports in accuracy; InternVL soft evidence mainly improves hFPR rather than accuracy.",
            "- Qwen same-budget random/grid/shuffled baselines often lower hFPR by becoming conservative/no-biased, but lose substantial accuracy and evidence recall. Treat Qwen as an accuracy/efficiency/evidence-retention result, not as the universal hFPR winner.",
            "- Caveat: LLaVA protected pruning trades hFPR against grid/random-style spatial baselines, so the safest claim is evidence preservation plus competitive accuracy, not universal hallucination reduction.",
            "- Spatial-aware GSR is a negative result: bbox/region retention raises evidence recall, but it does not reliably fix spatial hallucination on LLaVA/InternVL.",
        ]
    )
    (OUT_DIR / "p0_stats_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def include_method_in_report(row: dict[str, Any]) -> bool:
    keys = {
        "qwen_full",
        "qwen_target0p20",
        "qwen_target0p30",
        "qwen_random0p20",
        "qwen_grid0p20",
        "qwen_shuffled0p20",
        "qwen_random0p30",
        "qwen_fastv_proxy0p30",
        "qwen_visionzip_proxy0p30",
        "qwen_grid0p30",
        "qwen_shuffled0p30",
        "llava_full",
        "llava_protected0p40",
        "llava_grid0p40",
        "llava_random0p40",
        "llava_shuffled0p40",
        "llava_visionzip0p40",
        "llava_fastv0p40",
        "internvl_full_cal",
        "internvl_soft_hfpr0p50",
        "internvl_target0p50",
        "internvl_grid0p50",
        "internvl_random0p50",
        "internvl_shuffled0p50",
        "llava_gsr_spatial0p40",
        "llava_gsr_spatial_c10_0p40",
        "internvl_gsr_spatial0p50",
        "internvl_gsr_spatial_c10_0p50",
    }
    return str(row["key"]) in keys


def pair_table_line(row: dict[str, Any]) -> str:
    return (
        "| {comparison} | {n} | {left} | {right} | {acc_diff} | [{acc_l}, {acc_h}] | {acc_p} | "
        "{hfpr_diff} | [{hfpr_l}, {hfpr_h}] | {hfpr_p} | {reading} |"
    ).format(
        comparison=row["comparison"],
        n=row["n_overlap"],
        left=row["left_method"],
        right=row["right_method"],
        acc_diff=fmt_signed(row["acc_diff"]),
        acc_l=fmt_signed(row["acc_diff_ci_low"]),
        acc_h=fmt_signed(row["acc_diff_ci_high"]),
        acc_p=fmt_p(row["acc_sign_p"]),
        hfpr_diff=fmt_signed(row["hFPR_diff"]),
        hfpr_l=fmt_signed(row["hFPR_diff_ci_low"]),
        hfpr_h=fmt_signed(row["hFPR_diff_ci_high"]),
        hfpr_p=fmt_p(row["hFPR_sign_p"]),
        reading=reading(row),
    )


def reading(row: dict[str, Any]) -> str:
    acc_sig = row["acc_diff_ci_low"] > 0 or row["acc_diff_ci_high"] < 0
    hfpr_sig = row["hFPR_diff_ci_low"] > 0 or row["hFPR_diff_ci_high"] < 0
    parts = []
    if acc_sig:
        parts.append("acc higher" if row["acc_diff"] > 0 else "acc lower")
    else:
        parts.append("acc tied")
    if hfpr_sig:
        parts.append("hFPR lower" if row["hFPR_diff"] < 0 else "hFPR higher")
    else:
        parts.append("hFPR tied")
    return "; ".join(parts)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def bootstrap_ci(values: list[float], *, seed: int) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (values[0], values[0])
    rng = random.Random(seed)
    draws = []
    n = len(values)
    for _ in range(BOOTSTRAP):
        draws.append(sum(rng.choices(values, k=n)) / n)
    return percentile(draws, 0.025), percentile(draws, 0.975)


def exact_sign_p(left_true_right_false: int, right_true_left_false: int) -> float:
    n = left_true_right_false + right_true_left_false
    if n == 0:
        return 1.0
    k = min(left_true_right_false, right_true_left_false)
    log_terms = [
        math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2.0)
        for i in range(k + 1)
    ]
    max_log = max(log_terms)
    tail = math.exp(max_log) * sum(math.exp(value - max_log) for value in log_terms)
    return min(1.0, 2.0 * tail)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def is_negative(row: dict[str, Any]) -> bool:
    return bool(row.get("target_is_negative", False))


def as_float(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


def to_float(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return 0.0


def metric(row: dict[str, Any], key: str) -> Any:
    prefixed = {
        "ecr": "prune_ecr",
        "evidence_center_recall": "prune_evidence_center_recall",
        "evidence_patch_recall": "prune_evidence_patch_recall",
        "full_visual_tokens": "prune_full_visual_tokens",
        "kept_visual_tokens": "prune_kept_visual_tokens",
    }.get(key, f"prune_{key}")
    if key in row and is_number(row[key]):
        return float(row[key])
    if prefixed in row and is_number(row[prefixed]):
        return float(row[prefixed])
    return ""


def has_metric(row: dict[str, Any], key: str) -> bool:
    return metric(row, key) != ""


def mean_optional(values: list[Any]) -> Any:
    vals = [float(value) for value in values if is_number(value)]
    return sum(vals) / len(vals) if vals else ""


def mean(values: list[float]) -> float:
    vals = [float(value) for value in values if math.isfinite(float(value))]
    return sum(vals) / len(vals) if vals else 0.0


def is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return False


def fmt(value: Any) -> str:
    if value == "" or value is None:
        return ""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_signed(value: Any) -> str:
    try:
        return f"{float(value):+.3f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_p(value: Any) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if val < 0.0001:
        return "<1e-4"
    return f"{val:.4f}"


if __name__ == "__main__":
    main()
