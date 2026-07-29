#!/usr/bin/env python
"""Build a cross-model report for actual pruning overhead timing."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "efficiency"


RUNS: list[dict[str, Any]] = [
    {
        "model": "Qwen3-VL-8B",
        "group": "target_embed_topk",
        "point": "full1p00",
        "efficiency_path": Path("runs/efficiency/qwen3_8b_textocr_hard_target_embed_topk_1p00_limit100_overhead"),
        "quality_path": Path("runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_topk_1p00"),
        "quality_note": "full 1000-probe split",
    },
    {
        "model": "Qwen3-VL-8B",
        "group": "target_embed_topk",
        "point": "target0p20",
        "efficiency_path": Path("runs/efficiency/qwen3_8b_textocr_hard_target_embed_topk_0p20_limit100_overhead"),
        "quality_path": Path("runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_topk_0p20"),
        "quality_note": "full 1000-probe split",
    },
    {
        "model": "Qwen3-VL-8B",
        "group": "target_embed_topk",
        "point": "target0p30",
        "efficiency_path": Path("runs/efficiency/qwen3_8b_textocr_hard_target_embed_topk_0p30_limit100_overhead"),
        "quality_path": Path(
            "runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_topk_0p30_targetfix_802816"
        ),
        "quality_note": "full 1000-probe split",
    },
    {
        "model": "LLaVA-1.5-7B",
        "group": "target_embed_topk",
        "point": "full1p00",
        "efficiency_path": Path("runs/efficiency/llava15_7b_textocr_hard_target_embed_topk_1p00_limit100_overhead"),
        "quality_path": Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_direct"),
        "quality_note": "full 1000-probe split",
    },
    {
        "model": "LLaVA-1.5-7B",
        "group": "target_embed_topk",
        "point": "target0p40",
        "efficiency_path": Path("runs/efficiency/llava15_7b_textocr_hard_target_embed_topk_0p40_limit100_overhead"),
        "quality_path": Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_target_embed_topk_0p40_targetfix"),
        "quality_note": "full 1000-probe split; current target-conditioned selector, quality-risk point",
    },
    {
        "model": "LLaVA-1.5-7B",
        "group": "target_embed_topk",
        "point": "target0p50",
        "efficiency_path": Path("runs/efficiency/llava15_7b_textocr_hard_target_embed_topk_0p50_limit100_overhead"),
        "quality_path": Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_target_embed_topk_0p50_targetfix"),
        "quality_note": "full 1000-probe split; current target-conditioned selector, quality-risk point",
    },
    {
        "model": "LLaVA-1.5-7B",
        "group": "embed_topk",
        "point": "full1p00",
        "efficiency_path": Path("runs/efficiency/llava15_7b_textocr_hard_embed_topk_1p00_limit100_overhead"),
        "quality_path": Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_direct"),
        "quality_note": "full 1000-probe split",
    },
    {
        "model": "LLaVA-1.5-7B",
        "group": "embed_topk",
        "point": "embed0p40",
        "efficiency_path": Path("runs/efficiency/llava15_7b_textocr_hard_embed_topk_0p40_limit100_overhead"),
        "quality_path": Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_embed_topk_0p40"),
        "quality_note": "full 1000-probe split; accuracy-oriented LLaVA point",
    },
    {
        "model": "LLaVA-1.5-7B",
        "group": "embed_protected_topk",
        "point": "full1p00",
        "efficiency_path": Path("runs/efficiency/llava15_7b_textocr_hard_embed_topk_1p00_limit100_overhead"),
        "quality_path": Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_direct"),
        "quality_note": "full 1000-probe split; full-token reference for protected embedding selector",
    },
    {
        "model": "LLaVA-1.5-7B",
        "group": "embed_protected_topk",
        "point": "protected_embed0p40",
        "efficiency_path": Path("runs/efficiency/llava15_7b_textocr_hard_embed_protected_topk_0p40_limit100_overhead"),
        "quality_path": Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_embed_protected_topk_0p40"),
        "quality_note": "full 1000-probe split; bbox/OCR evidence-protected LLaVA point",
    },
    {
        "model": "LLaVA-1.5-7B",
        "group": "grid",
        "point": "full1p00",
        "efficiency_path": Path("runs/efficiency/llava15_7b_textocr_hard_grid_1p00_limit100_overhead"),
        "quality_path": Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_direct"),
        "quality_note": "full 1000-probe split",
    },
    {
        "model": "LLaVA-1.5-7B",
        "group": "grid",
        "point": "grid0p40",
        "efficiency_path": Path("runs/efficiency/llava15_7b_textocr_hard_grid_0p40_limit100_overhead"),
        "quality_path": Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_grid_0p40"),
        "quality_note": "full 1000-probe split; low-hallucination spatial baseline",
    },
    {
        "model": "InternVL3.5-8B",
        "group": "target_embed_topk",
        "point": "full1p00",
        "efficiency_path": Path("runs/efficiency/internvl35_8b_textocr_hard_target_embed_topk_1p00_limit100_overhead"),
        "quality_path": Path("runs/internvl_textocr_hard/calibrated_test_full_devthr"),
        "quality_note": "held-out calibrated test, dev threshold",
    },
    {
        "model": "InternVL3.5-8B",
        "group": "target_embed_topk",
        "point": "target0p40",
        "efficiency_path": Path("runs/efficiency/internvl35_8b_textocr_hard_target_embed_topk_0p40_limit100_overhead"),
        "quality_path": Path("runs/internvl_textocr_hard/calibrated_test_target0p40_devthr"),
        "quality_note": "held-out calibrated test, dev threshold; stress point",
    },
    {
        "model": "InternVL3.5-8B",
        "group": "target_embed_topk",
        "point": "target0p50",
        "efficiency_path": Path("runs/efficiency/internvl35_8b_textocr_hard_target_embed_topk_0p50_limit100_overhead"),
        "quality_path": Path("runs/internvl_textocr_hard/calibrated_test_target0p50_devthr"),
        "quality_note": "held-out calibrated test, dev threshold",
    },
    {
        "model": "InternVL3.5-8B",
        "group": "target_embed_soft_evidence_topk",
        "point": "full1p00",
        "efficiency_path": Path("runs/efficiency/internvl35_8b_textocr_hard_target_embed_topk_1p00_limit100_overhead"),
        "quality_path": Path("runs/internvl_textocr_hard/calibrated_test_full_devthr"),
        "quality_note": "held-out calibrated test; full-token reference for soft-evidence selector",
    },
    {
        "model": "InternVL3.5-8B",
        "group": "target_embed_soft_evidence_topk",
        "point": "soft_evidence0p50_hfpr",
        "efficiency_path": Path(
            "runs/efficiency/internvl35_8b_textocr_hard_target_embed_soft_evidence_topk_0p50_b0p05_limit100_overhead"
        ),
        "quality_path": Path("runs/internvl_textocr_hard/calibrated_test_target_soft_evidence0p50_b0p05_hfprconstr_devthr"),
        "quality_note": "held-out calibrated test, main risk-constrained soft-evidence point",
    },
    {
        "model": "InternVL3.5-8B",
        "group": "grid",
        "point": "full1p00",
        "efficiency_path": Path("runs/efficiency/internvl35_8b_textocr_hard_grid_1p00_limit100_overhead"),
        "quality_path": Path("runs/internvl_textocr_hard/calibrated_test_full_devthr"),
        "quality_note": "held-out calibrated test, dev threshold",
    },
    {
        "model": "InternVL3.5-8B",
        "group": "grid",
        "point": "grid0p50",
        "efficiency_path": Path("runs/efficiency/internvl35_8b_textocr_hard_grid_0p50_limit100_overhead"),
        "quality_path": Path("runs/internvl_textocr_hard/calibrated_test_grid0p50_devthr"),
        "quality_note": "held-out calibrated test, dev threshold; calibrated accuracy baseline",
    },
    {
        "model": "InternVL3.5-8B",
        "group": "embed_topk",
        "point": "full1p00",
        "efficiency_path": Path("runs/efficiency/internvl35_8b_textocr_hard_embed_topk_1p00_limit100_overhead"),
        "quality_path": Path("runs/internvl_textocr_hard/calibrated_test_full_devthr"),
        "quality_note": "held-out calibrated test, dev threshold",
    },
    {
        "model": "InternVL3.5-8B",
        "group": "embed_topk",
        "point": "embed0p50",
        "efficiency_path": Path("runs/efficiency/internvl35_8b_textocr_hard_embed_topk_0p50_limit100_overhead"),
        "quality_path": Path("runs/internvl_textocr_hard/calibrated_test_embed0p50_devthr"),
        "quality_note": "held-out calibrated test, dev threshold; embedding baseline",
    },
]


def main() -> None:
    rows = build_rows()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "cross_model_actual_overhead_limit100.csv"
    md_path = OUT_DIR / "cross_model_actual_overhead_limit100.md"
    write_csv(csv_path, rows)
    md_path.write_text(build_report(rows))
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in RUNS:
        efficiency = load_efficiency(spec["efficiency_path"])
        quality = load_quality(spec["quality_path"])
        rows.append(
            {
                "model": spec["model"],
                "group": spec["group"],
                "point": spec["point"],
                "quality_note": spec["quality_note"],
                "quality_path": str(spec["quality_path"]),
                "efficiency_path": str(spec["efficiency_path"]),
                **quality,
                **efficiency,
            }
        )

    baselines = {(row["model"], row["group"]): row for row in rows if row["point"] == "full1p00"}
    for row in rows:
        baseline = baselines[(row["model"], row["group"])]
        language_saved = baseline["language_ms"] - row["language_ms"]
        forward_saved = baseline["forward_ms"] - row["forward_ms"]
        total_prune_path_ms = row["target_text_ms"] + row["prune_overhead_ms"]
        row["language_saved_ms"] = language_saved
        row["forward_saved_ms"] = forward_saved
        row["language_speedup_vs_full"] = safe_div(baseline["language_ms"], row["language_ms"])
        row["forward_speedup_vs_full"] = safe_div(baseline["forward_ms"], row["forward_ms"])
        row["prune_overhead_pct_saved_language"] = safe_div(row["prune_overhead_ms"] * 100.0, language_saved)
        row["total_prune_path_ms"] = total_prune_path_ms
        row["total_prune_path_pct_saved_language"] = safe_div(total_prune_path_ms * 100.0, language_saved)
        row["selector_pct_saved_language"] = safe_div(row["selector_ms"] * 100.0, language_saved)
        row["score_compute_pct_saved_language"] = safe_div(row["score_compute_ms"] * 100.0, language_saved)
    return rows


def load_efficiency(run_dir: Path) -> dict[str, Any]:
    metrics = load_json(ROOT / run_dir / "metrics.json")
    pruning = metrics["pruning"]
    return {
        "efficiency_n": pruning.get("num_pruned_probes"),
        "full_sequence_tokens": pruning.get("mean_full_sequence_tokens"),
        "pruned_sequence_tokens": pruning.get("mean_pruned_sequence_tokens"),
        "full_visual_tokens": pruning.get("mean_full_visual_tokens"),
        "kept_visual_tokens": pruning.get("mean_kept_visual_tokens"),
        "keep_ratio": pruning.get("mean_keep_ratio"),
        "removal_fraction": pruning.get("mean_removal_fraction"),
        "vision_ms": pruning.get("mean_vision_ms"),
        "target_text_ms": pruning.get("mean_target_text_ms"),
        "score_compute_ms": pruning.get("mean_score_compute_ms"),
        "selector_ms": pruning.get("mean_selector_ms"),
        "materialize_ms": pruning.get("mean_prune_materialize_ms"),
        "prune_overhead_ms": pruning.get("mean_prune_overhead_ms"),
        "language_ms": pruning.get("mean_language_ms"),
        "forward_ms": pruning.get("mean_forward_ms"),
    }


def load_quality(run_dir: Path) -> dict[str, Any]:
    metrics_path = ROOT / run_dir / "metrics.json"
    if metrics_path.exists():
        metrics = load_json(metrics_path)
        return {
            "quality_n": metrics.get("num_samples"),
            "quality_acc": metrics.get("direct_accuracy"),
            "quality_hFPR": metrics.get("direct_hallucination_fpr"),
            "quality_yes_rate": metrics.get("direct_yes_rate"),
        }
    score_path = ROOT / run_dir / "probe_scores.jsonl"
    rows = read_jsonl(score_path)
    negative = [row for row in rows if row.get("binary_polarity") == "negative"]
    return {
        "quality_n": len(rows),
        "quality_acc": mean(float(row.get("correct", False)) for row in rows),
        "quality_hFPR": mean(float(row.get("pred_answer") == "yes") for row in negative),
        "quality_yes_rate": mean(float(row.get("pred_answer") == "yes") for row in rows),
    }


def build_report(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Cross-Model Actual Pruning Overhead (limit=100)",
        "",
        "Timing rows use each selector group's own backend path with `--orig-only --limit 100`. Every non-full row is compared against the matching `@1.00` row for the same model and selector group, so speedups include the selector's actual score/materialization overhead.",
        "",
        "Quality columns are taken from the larger cached evaluation for each model. Qwen and LLaVA use the 1000-probe TextOCR-Hard split; InternVL uses the held-out calibrated test split because its raw default threshold collapses toward almost-all-yes on this benchmark.",
        "",
        "## Main Timing Table",
        "",
        "| model | selector | point | quality acc | hFPR | keep | visual kept/full | seq pruned/full | language ms | lang x | forward ms | fwd x | prune overhead ms | target text ms | overhead/saved lang | total path/saved lang |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['model']} | {row['group']} | {row['point']} | {fmt(row['quality_acc'])} | {fmt(row['quality_hFPR'])} | "
            f"{fmt(row['keep_ratio'])} | {fmt(row['kept_visual_tokens'])}/{fmt(row['full_visual_tokens'])} | "
            f"{fmt(row['pruned_sequence_tokens'])}/{fmt(row['full_sequence_tokens'])} | {fmt(row['language_ms'])} | "
            f"{fmt(row['language_speedup_vs_full'])} | {fmt(row['forward_ms'])} | {fmt(row['forward_speedup_vs_full'])} | "
            f"{fmt(row['prune_overhead_ms'])} | {fmt(row['target_text_ms'])} | "
            f"{fmt_pct(row['prune_overhead_pct_saved_language'])} | {fmt_pct(row['total_prune_path_pct_saved_language'])} |"
        )

    lines.extend(
        [
            "",
            "## Component Readout",
            "",
            "| model | selector | point | score ms | selector ms | materialize ms | selector/saved lang | score/saved lang | quality note |",
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"{row['model']} | {row['group']} | {row['point']} | {fmt(row['score_compute_ms'])} | {fmt(row['selector_ms'])} | "
            f"{fmt(row['materialize_ms'])} | {fmt_pct(row['selector_pct_saved_language'])} | "
            f"{fmt_pct(row['score_compute_pct_saved_language'])} | {row['quality_note']} |"
        )

    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- Qwen is the cleanest current claim: `target0p20` keeps 20.0% of visual tokens, has full-split accuracy 0.793 versus full 0.787, and gives 3.10x language-stage / 1.45x single-sample forward speedup. The measured prune overhead is 4.7% of saved language time, or 5.6% if target-text embedding time is counted.",
            "- Qwen `target0p30` is the quality point: full-split accuracy 0.798 and hFPR 0.222, with 2.47x language-stage / 1.37x forward speedup.",
            "- LLaVA target-conditioned pruning is a negative result: it gives speed but hFPR rises from 0.304 to 0.496/0.522. LLaVA's positive results are `embed0p40` and `grid0p40`: `embed0p40` reaches acc 0.662 with 1.47x language-stage / 1.36x forward speedup, while `grid0p40` lowers hFPR to 0.258 with 1.46x language-stage / 1.34x forward speedup.",
            "- InternVL has the largest absolute sequence reduction. `target0p50` is the safer calibrated hallucination point, with hFPR 0.179 and 1.82x language-stage / 1.57x forward speedup. `grid0p50` has the best calibrated accuracy in the current table, acc 0.647 with 1.84x language-stage / 1.59x forward speedup. `embed0p50` is weaker on quality, acc 0.618.",
            "- Selector kernels are not the bottleneck. For Qwen and InternVL they are usually below about 3% of saved language time. LLaVA's target selector is expensive mainly because target-text scoring eats a large share of the small saved language time; non-target `embed/grid` reduce that overhead substantially.",
            "",
            "## Output Files",
            "",
            "- `runs/efficiency/cross_model_actual_overhead_limit100.csv`",
            "- `runs/efficiency/cross_model_actual_overhead_limit100.md`",
        ]
    )
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def mean(values: Any) -> float:
    vals = [float(value) for value in values]
    return sum(vals) / len(vals) if vals else 0.0


def safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(denominator) < 1e-12:
        return None
    return numerator / denominator


def fmt(value: Any) -> str:
    if value is None or value == "":
        return "NA"
    return f"{float(value):.3f}"


def fmt_pct(value: Any) -> str:
    if value is None or value == "":
        return "NA"
    return f"{float(value):.1f}%"


if __name__ == "__main__":
    main()
