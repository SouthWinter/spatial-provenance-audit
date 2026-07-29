#!/usr/bin/env python
"""Split TextOCR-Hard region coverage by probe semantics.

Positive probes use the annotated source-word box as answer-supporting evidence.
Near-miss negatives use the same box only as the confusable source region; its
retention cannot prove that the queried decoy is absent elsewhere in the image.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "problem_optimization_audit" / "evidence_semantics"


@dataclass(frozen=True)
class RunSpec:
    block: str
    model: str
    method: str
    path: str


RUNS = [
    RunSpec("development", "Qwen3-VL-8B", "Full", "runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_topk_1p00"),
    RunSpec("development", "Qwen3-VL-8B", "Target (20%)", "runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_topk_0p20"),
    RunSpec("development", "Qwen3-VL-8B", "Target (30%)", "runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_topk_0p30_targetfix_802816"),
    RunSpec("development", "Qwen3-VL-8B", "Random (30%)", "runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_random_0p30"),
    RunSpec("development", "Qwen3-VL-8B", "Grid (30%)", "runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_grid_0p30"),
    RunSpec("development", "Qwen3-VL-8B", "Shuffled (30%)", "runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_shuffled_topk_0p30"),
    RunSpec("ablation", "Qwen3-VL-8B", "Center protected (30%)", "runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_protected_center_topk_0p30_targetfix_802816"),
    RunSpec("development", "LLaVA-1.5-7B", "Full", "runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_direct"),
    RunSpec("development", "LLaVA-1.5-7B", "Protected (40%)", "runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_embed_protected_topk_0p40"),
    RunSpec("development", "LLaVA-1.5-7B", "Target (40%)", "runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_target_embed_topk_0p40_targetfix"),
    RunSpec("development", "LLaVA-1.5-7B", "Random (40%)", "runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_random_0p40"),
    RunSpec("development", "LLaVA-1.5-7B", "Grid (40%)", "runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_grid_0p40"),
    RunSpec("development", "LLaVA-1.5-7B", "SCOPE (40%)", "runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_scope_0p40"),
    RunSpec("development", "LLaVA-1.5-7B", "CoIn (40%)", "runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_coin_0p40"),
    RunSpec("development", "LLaVA-1.5-7B", "VisionZip (40%)", "runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_visionzip_0p40"),
    RunSpec("development", "LLaVA-1.5-7B", "FastV (40%)", "runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_fastv_0p40"),
    RunSpec("held-out", "InternVL3.5-8B", "Full calibrated", "runs/internvl_textocr_hard/calibrated_test_full_devthr"),
    RunSpec("held-out", "InternVL3.5-8B", "Soft evidence (50%, cal.)", "runs/internvl_textocr_hard/calibrated_test_target_soft_evidence0p50_b0p05_devthr"),
    RunSpec("held-out", "InternVL3.5-8B", "Random (50%)", "runs/internvl_textocr_hard/calibrated_test_random0p50_devthr"),
    RunSpec("held-out", "InternVL3.5-8B", "Grid (50%)", "runs/internvl_textocr_hard/calibrated_test_grid0p50_devthr"),
    RunSpec("held-out", "InternVL3.5-8B", "Shuffled (50%)", "runs/internvl_textocr_hard/calibrated_test_target_shuffled0p50_devthr"),
    RunSpec("ablation", "InternVL3.5-8B", "Hard protected (50%)", "runs/internvl_textocr_hard/calibrated_test_target_protected0p50_devthr"),
    RunSpec("confirmation", "Qwen3-VL-8B", "Full", "runs/textocr_confirmation/qwen3_8b_full"),
    RunSpec("confirmation", "Qwen3-VL-8B", "Target (30%)", "runs/textocr_confirmation/qwen3_8b_target_0p30"),
    RunSpec("confirmation", "Qwen3-VL-8B", "Random (30%)", "runs/textocr_confirmation/qwen3_8b_random_0p30"),
    RunSpec("confirmation", "Qwen3-VL-8B", "Grid (30%)", "runs/textocr_confirmation/qwen3_8b_grid_0p30"),
    RunSpec("confirmation", "LLaVA-1.5-7B", "Full", "runs/textocr_confirmation/llava15_7b_full"),
    RunSpec("confirmation", "LLaVA-1.5-7B", "Protected (40%)", "runs/textocr_confirmation/llava15_7b_protected_0p40"),
    RunSpec("confirmation", "LLaVA-1.5-7B", "Random (40%)", "runs/textocr_confirmation/llava15_7b_random_0p40"),
    RunSpec("confirmation", "LLaVA-1.5-7B", "Target (40%)", "runs/textocr_confirmation/llava15_7b_target_0p40"),
    RunSpec("confirmation", "LLaVA-1.5-7B", "SCOPE (40%)", "runs/textocr_confirmation/llava15_7b_scope_0p40"),
    RunSpec("confirmation", "LLaVA-1.5-7B", "CoIn (40%)", "runs/textocr_confirmation/llava15_7b_coin_0p40"),
    RunSpec("box-source", "LLaVA-1.5-7B", "Protected, EasyOCR", "runs/box_robustness/llava15_7b_easyocr_all_embed_protected_topk0p40"),
    RunSpec("box-source", "LLaVA-1.5-7B", "Protected, missing", "runs/box_robustness/llava15_7b_missing_embed_protected_topk0p40"),
    RunSpec("box-source", "InternVL3.5-8B", "Soft evidence, GT, risk threshold", "runs/internvl_textocr_hard/calibrated_test_target_soft_evidence0p50_b0p05_hfprconstr_devthr"),
    RunSpec("box-source", "InternVL3.5-8B", "Soft evidence, EasyOCR, risk threshold", "runs/box_robustness/calibrated_test_internvl35_8b_easyocr_all_soft_evidence0p50_b0p05_mainthr"),
    RunSpec("box-source", "InternVL3.5-8B", "Soft evidence, missing, risk threshold", "runs/box_robustness/calibrated_test_internvl35_8b_missing_soft_evidence0p50_b0p05_mainthr"),
]


def main() -> None:
    rows = [summarize(spec) for spec in RUNS]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "coverage_by_probe_semantics.csv", rows)
    write_markdown(OUT_DIR / "coverage_by_probe_semantics.md", rows)
    print(f"Wrote {len(rows)} rows to {OUT_DIR}")


def summarize(spec: RunSpec) -> dict[str, object]:
    if spec.block == "box-source":
        return summarize_box_source(spec)

    path = ROOT / spec.path / "probe_scores.jsonl"
    positive: list[float] = []
    negative: list[float] = []
    all_values: list[float] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            polarity = row.get("binary_polarity")
            value = row.get("prune_ecr")
            if value is None:
                value = 1.0 if row.get("has_bbox") else 0.0
            value = float(value)
            all_values.append(value)
            if polarity == "positive":
                positive.append(value)
            elif polarity == "negative":
                negative.append(value)
            else:
                raise ValueError(f"Missing polarity in {path}: {row.get('sample_id')}")
    if not positive or not negative:
        raise ValueError(f"Expected both probe polarities in {path}")
    aggregate = mean(all_values)
    reconstructed = (sum(positive) + sum(negative)) / (len(positive) + len(negative))
    if abs(aggregate - reconstructed) > 1e-12:
        raise AssertionError(f"Aggregate mismatch for {path}")
    return {
        "block": spec.block,
        "model": spec.model,
        "method": spec.method,
        "n_positive": len(positive),
        "n_negative": len(negative),
        "positive_ecr": f"{mean(positive):.6f}",
        "negative_source_coverage": f"{mean(negative):.6f}",
        "legacy_aggregate_coverage": f"{aggregate:.6f}",
        "source": spec.path,
    }


def summarize_box_source(spec: RunSpec) -> dict[str, object]:
    """Use coverage recomputed against audit-only TextOCR boxes.

    Detector-driven runs store ``prune_ecr`` against selector-visible detector
    boxes.  The box-robustness audit separately reconstructs token geometry and
    evaluates the retained indices against the held-out TextOCR boxes; those
    are the values needed for PosECR and NegSRC.
    """
    summary_path = ROOT / "runs/box_robustness/box_robustness_summary.csv"
    with summary_path.open(encoding="utf-8", newline="") as handle:
        matches = [row for row in csv.DictReader(handle) if row["score_dir"] == spec.path]
    if len(matches) != 1:
        raise ValueError(f"Expected one box-source audit row for {spec.path}, got {len(matches)}")
    row = matches[0]
    n = int(row["num_samples"])
    if n % 2:
        raise ValueError(f"Expected balanced positive/negative probes for {spec.path}, got {n}")
    return {
        "block": spec.block,
        "model": spec.model,
        "method": spec.method,
        "n_positive": n // 2,
        "n_negative": n // 2,
        "positive_ecr": f"{float(row['true_positive_ECR']):.6f}",
        "negative_source_coverage": f"{float(row['true_negative_source_coverage']):.6f}",
        "legacy_aggregate_coverage": f"{float(row['true_ECR']):.6f}",
        "source": spec.path,
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Region coverage split by probe semantics",
        "",
        "Positive ECR measures coverage of an answer-supporting word box on positive probes. "
        "Negative source coverage measures retention of the confusable source-word box used to "
        "construct each near-miss negative; it does not certify global target absence.",
        "",
        "| Split | Model | Method | n+ | n- | Positive ECR | Negative source coverage | Legacy aggregate |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['block']} | {row['model']} | {row['method']} | {row['n_positive']} | "
            f"{row['n_negative']} | {float(row['positive_ecr']):.3f} | "
            f"{float(row['negative_source_coverage']):.3f} | "
            f"{float(row['legacy_aggregate_coverage']):.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
