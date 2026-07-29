#!/usr/bin/env python
"""Build TextOCR-Hard cross-model pruning tables and a small Qwen budget search."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "cross_model_textocr_hard"


SUMMARY_RUNS: dict[str, list[tuple[str, Path, str]]] = {
    "Qwen3-VL-8B": [
        ("full", Path("runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_topk_1p00"), "default full tokens"),
        ("target0p20", Path("runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_topk_0p20"), "efficiency point"),
        ("target0p25", Path("runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_topk_0p25"), "efficiency/accuracy tradeoff"),
        ("target0p30", Path("runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_topk_0p30_targetfix_802816"), "quality point"),
        ("target_grid_topk0p30", Path("runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_grid_topk_0p30_targetfix_802816"), "OCR-safe grid floor plus target salience"),
        ("target_grid_topk0p50", Path("runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_grid_topk_0p50_targetfix_802816"), "OCR-safe higher-budget point"),
        ("soft_evidence0p30", Path("runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_soft_evidence_topk_0p30_b0p05_targetfix_802816"), "evidence-balanced ablation"),
        ("protected_center0p30", Path("runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_protected_center_topk_0p30_targetfix_802816"), "max center recall ablation"),
        ("grid0p50", Path("runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_grid_0p50"), "spatial baseline"),
        ("random0p50", Path("runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_random_0p50"), "random baseline"),
    ],
    "LLaVA-1.5-7B": [
        ("full", Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_direct"), "default full tokens"),
        ("embed0p40", Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_embed_topk_0p40"), "accuracy-efficiency point"),
        ("protected_embed0p40", Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_embed_protected_topk_0p40"), "bbox/OCR evidence-protected point"),
        ("grid0p40", Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_grid_0p40"), "low-hallucination spatial point"),
        ("embed0p50", Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_embed_topk_0p50"), "embedding baseline"),
        ("grid0p50", Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_grid_0p50"), "spatial baseline"),
        ("target0p50", Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_target_embed_topk_0p50_targetfix"), "negative target-conditioned ablation"),
    ],
    "InternVL3.5-8B calibrated-test": [
        ("full_cal", Path("runs/internvl_textocr_hard/calibrated_test_full_devthr"), "dev-threshold calibrated full"),
        ("target0p50_cal", Path("runs/internvl_textocr_hard/calibrated_test_target0p50_devthr"), "low-hallucination point"),
        (
            "target_grid_topk0p50_cal",
            Path("runs/internvl_textocr_hard/calibrated_test_target_grid_topk0p50_devthr"),
            "OCR-safe grid floor plus target salience",
        ),
        (
            "target_grid_topk0p60_cal",
            Path("runs/internvl_textocr_hard/calibrated_test_target_grid_topk0p60_devthr"),
            "OCR-safe higher-budget point",
        ),
        (
            "target_soft_evidence0p50_hfpr_cal",
            Path("runs/internvl_textocr_hard/calibrated_test_target_soft_evidence0p50_b0p05_hfprconstr_devthr"),
            "risk-constrained soft evidence point",
        ),
        (
            "target_protected0p50_cal",
            Path("runs/internvl_textocr_hard/calibrated_test_target_protected0p50_devthr"),
            "bbox/OCR evidence-protected trade-off",
        ),
        (
            "target_soft_evidence0p50_cal",
            Path("runs/internvl_textocr_hard/calibrated_test_target_soft_evidence0p50_b0p05_devthr"),
            "soft evidence-boost negative ablation",
        ),
        ("grid0p50_cal", Path("runs/internvl_textocr_hard/calibrated_test_grid0p50_devthr"), "accuracy/AUROC spatial point"),
        ("embed0p50_cal", Path("runs/internvl_textocr_hard/calibrated_test_embed0p50_devthr"), "embedding baseline"),
        ("target0p40_cal", Path("runs/internvl_textocr_hard/calibrated_test_target0p40_devthr"), "over-pruning stress"),
        ("grid0p40_cal", Path("runs/internvl_textocr_hard/calibrated_test_grid0p40_devthr"), "over-pruning spatial stress"),
    ],
}


QWEN_BUDGET_RUNS: dict[float, Path] = {
    0.20: Path("runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_topk_0p20"),
    0.25: Path("runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_topk_0p25"),
    0.30: Path("runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_topk_0p30_targetfix_802816"),
    0.35: Path("runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_topk_0p35"),
    0.40: Path("runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_topk_0p40"),
    0.50: Path("runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_topk_0p50"),
    1.00: Path("runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_topk_1p00"),
}

STRUCTURAL_FEATURES = (
    "prune_ecr",
    "prune_evidence_center_recall",
    "prune_evidence_patch_recall",
    "prune_full_visual_tokens",
    "prune_kept_visual_tokens",
    "prune_removal_fraction",
)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = build_summary_rows()
    write_csv(OUT_DIR / "cross_model_summary.csv", summary_rows)

    policy_rows, candidates = search_qwen_structural_policy()
    write_csv(OUT_DIR / "qwen_structural_policy_candidates.csv", candidates)
    write_jsonl(OUT_DIR / "qwen_structural_policy_rows.jsonl", policy_rows)

    report = build_report(summary_rows, candidates)
    (OUT_DIR / "cross_model_report.md").write_text(report)
    print(f"Wrote report to {OUT_DIR / 'cross_model_report.md'}")
    print(f"Wrote summary to {OUT_DIR / 'cross_model_summary.csv'}")
    print(f"Wrote Qwen policy candidates to {OUT_DIR / 'qwen_structural_policy_candidates.csv'}")


def build_summary_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, runs in SUMMARY_RUNS.items():
        for label, run_dir, note in runs:
            score_path = ROOT / run_dir / "probe_scores.jsonl"
            if not score_path.exists():
                continue
            metrics = summarize_scores(read_jsonl(score_path))
            rows.append(
                {
                    "model": model,
                    "run": label,
                    "note": note,
                    "path": str(run_dir),
                    **metrics,
                }
            )
    return rows


def search_qwen_structural_policy() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    score_grid = {budget: load_scores(ROOT / run_dir / "probe_scores.jsonl") for budget, run_dir in QWEN_BUDGET_RUNS.items()}
    ids = sorted(set.intersection(*(set(rows) for rows in score_grid.values())))
    reference_budget = 0.30
    feature_budget = 0.20
    reference = make_fixed_policy(reference_budget)
    ref_dev = evaluate_policy(ids, score_grid, reference, "dev")
    ref_test = evaluate_policy(ids, score_grid, reference, "test")
    ref_all = evaluate_policy(ids, score_grid, reference, "all")

    policy_rows: list[dict[str, Any]] = []
    for sid in ids:
        base = score_grid[feature_budget][sid]
        row = {
            "sample_id": sid,
            "image_id": group_id(base),
            "split": split_for_row(base),
            "binary_polarity": base.get("binary_polarity", ""),
        }
        for feature in STRUCTURAL_FEATURES:
            row[feature] = to_float(base.get(feature))
        for budget in sorted(score_grid):
            score = score_grid[budget][sid]
            tag = budget_tag(budget)
            row[f"correct_{tag}"] = bool(score.get("correct", False))
            row[f"pred_yes_{tag}"] = bool(score.get("pred_answer") == "yes")
        policy_rows.append(row)

    candidates: list[dict[str, Any]] = []
    budgets = sorted(QWEN_BUDGET_RUNS)
    for budget in budgets:
        policy = make_fixed_policy(budget)
        add_candidate(
            candidates,
            ids=ids,
            score_grid=score_grid,
            policy=policy,
            policy_type="fixed",
            feature="",
            threshold=None,
            direction=0,
            low_budget=budget,
            high_budget=budget,
            ref_dev=ref_dev,
        )

    dev_ids = [sid for sid in ids if split_for_row(score_grid[reference_budget][sid]) == "dev"]
    for low_budget, high_budget in itertools.combinations(budgets, 2):
        for feature in STRUCTURAL_FEATURES:
            values = sorted({to_float(score_grid[feature_budget][sid].get(feature)) for sid in dev_ids})
            if not values:
                continue
            thresholds = quantile_thresholds(values, bins=20)
            for threshold in thresholds:
                for direction in (1, -1):
                    policy = make_threshold_policy(
                        score_grid[feature_budget],
                        feature=feature,
                        threshold=threshold,
                        direction=direction,
                        low_budget=low_budget,
                        high_budget=high_budget,
                    )
                    add_candidate(
                        candidates,
                        ids=ids,
                        score_grid=score_grid,
                        policy=policy,
                        policy_type="threshold",
                        feature=feature,
                        threshold=threshold,
                        direction=direction,
                        low_budget=low_budget,
                        high_budget=high_budget,
                        ref_dev=ref_dev,
                    )

    candidates.sort(
        key=lambda row: (
            int(row["dev_safe"]),
            -row["dev_keep"],
            row["dev_acc"],
            -row["dev_hFPR"],
            -row["all_keep"],
        ),
        reverse=True,
    )

    for row in candidates:
        row["ref_dev_acc"] = ref_dev["acc"]
        row["ref_dev_hFPR"] = ref_dev["hFPR"]
        row["ref_dev_keep"] = ref_dev["keep"]
        row["ref_test_acc"] = ref_test["acc"]
        row["ref_test_hFPR"] = ref_test["hFPR"]
        row["ref_test_keep"] = ref_test["keep"]
        row["ref_all_acc"] = ref_all["acc"]
        row["ref_all_hFPR"] = ref_all["hFPR"]
        row["ref_all_keep"] = ref_all["keep"]
    return policy_rows, candidates


def add_candidate(
    candidates: list[dict[str, Any]],
    *,
    ids: list[str],
    score_grid: dict[float, dict[str, dict[str, Any]]],
    policy: Callable[[str], float],
    policy_type: str,
    feature: str,
    threshold: float | None,
    direction: int,
    low_budget: float,
    high_budget: float,
    ref_dev: dict[str, float],
) -> None:
    dev = evaluate_policy(ids, score_grid, policy, "dev")
    test = evaluate_policy(ids, score_grid, policy, "test")
    all_metrics = evaluate_policy(ids, score_grid, policy, "all")
    dev_safe = dev["acc"] >= ref_dev["acc"] - 0.005 and dev["hFPR"] <= ref_dev["hFPR"] + 0.015
    candidates.append(
        {
            "policy_type": policy_type,
            "feature": feature,
            "threshold": "" if threshold is None else threshold,
            "direction": direction,
            "low_budget": low_budget,
            "high_budget": high_budget,
            "dev_safe": dev_safe,
            **prefixed("dev", dev),
            **prefixed("test", test),
            **prefixed("all", all_metrics),
        }
    )


def summarize_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    negative = [row for row in rows if row.get("binary_polarity") == "negative"]
    positive = [row for row in rows if row.get("binary_polarity") == "positive"]
    return {
        "n": len(rows),
        "acc": mean(float(row.get("correct", False)) for row in rows),
        "hFPR": mean(float(row.get("pred_answer") == "yes") for row in negative),
        "yes_rate": mean(float(row.get("pred_answer") == "yes") for row in rows),
        "pos_acc": mean(float(row.get("correct", False)) for row in positive),
        "neg_acc": mean(float(row.get("correct", False)) for row in negative),
        "kept_visual": mean_optional(row.get("prune_kept_visual_tokens") for row in rows),
        "full_visual": mean_optional(row.get("prune_full_visual_tokens") for row in rows),
        "keep_ratio": mean_optional(
            (row.get("prune_kept_visual_tokens") / row.get("prune_full_visual_tokens"))
            if row.get("prune_full_visual_tokens")
            else None
            for row in rows
        ),
        "ECR": mean_optional(row.get("prune_ecr") for row in rows),
        "CenterR": mean_optional(row.get("prune_evidence_center_recall") for row in rows),
        "PatchR": mean_optional(row.get("prune_evidence_patch_recall") for row in rows),
    }


def evaluate_policy(
    ids: list[str],
    score_grid: dict[float, dict[str, dict[str, Any]]],
    policy: Callable[[str], float],
    split_name: str,
) -> dict[str, float]:
    selected: list[tuple[float, dict[str, Any]]] = []
    reference_budget = 0.30 if 0.30 in score_grid else sorted(score_grid)[0]
    for sid in ids:
        ref_row = score_grid[reference_budget][sid]
        if split_name != "all" and split_for_row(ref_row) != split_name:
            continue
        budget = policy(sid)
        selected.append((budget, score_grid[budget][sid]))
    negative = [row for _, row in selected if row.get("binary_polarity") == "negative"]
    return {
        "n": float(len(selected)),
        "acc": mean(float(row.get("correct", False)) for _, row in selected),
        "hFPR": mean(float(row.get("pred_answer") == "yes") for row in negative),
        "yes_rate": mean(float(row.get("pred_answer") == "yes") for _, row in selected),
        "keep": mean(budget for budget, _ in selected),
    }


def make_fixed_policy(budget: float) -> Callable[[str], float]:
    def policy(_: str) -> float:
        return budget

    return policy


def make_threshold_policy(
    feature_rows: dict[str, dict[str, Any]],
    *,
    feature: str,
    threshold: float,
    direction: int,
    low_budget: float,
    high_budget: float,
) -> Callable[[str], float]:
    def policy(sid: str) -> float:
        value = to_float(feature_rows[sid].get(feature))
        use_low = value >= threshold if direction > 0 else value <= threshold
        return low_budget if use_low else high_budget

    return policy


def build_report(summary_rows: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> str:
    best = candidates[0] if candidates else {}
    lines = [
        "# TextOCR-Hard Cross-Model Checkpoint",
        "",
        "This report is generated from cached `probe_scores.jsonl` files. Qwen and LLaVA use the full 1000-probe split; InternVL uses the held-out calibrated test split because its default yes/no threshold collapses to almost-all-yes.",
        "",
        "## Main Tables",
        "",
    ]
    for model in SUMMARY_RUNS:
        rows = [row for row in summary_rows if row["model"] == model]
        lines.extend(render_summary_table(model, rows))
        lines.append("")
    lines.extend(
        [
            "## Qwen Structural Budget Search",
            "",
            "Reference policy: Qwen `target@0.30` on the dev split. A candidate is marked dev-safe when dev accuracy is within 0.5 percentage points of the reference and dev hFPR is no more than 1.5 percentage points higher.",
            "",
        ]
    )
    if best:
        lines.extend(
            [
                "| policy | feature | threshold | direction | low | high | dev acc | dev hFPR | dev keep | test acc | test hFPR | test keep | all acc | all hFPR | all keep |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in candidates[:15]:
            lines.append(
                "| "
                f"{row['policy_type']} | {row['feature'] or '-'} | {fmt(row['threshold'])} | {row['direction']} | "
                f"{row['low_budget']:.2f} | {row['high_budget']:.2f} | "
                f"{row['dev_acc']:.3f} | {row['dev_hFPR']:.3f} | {row['dev_keep']:.3f} | "
                f"{row['test_acc']:.3f} | {row['test_hFPR']:.3f} | {row['test_keep']:.3f} | "
                f"{row['all_acc']:.3f} | {row['all_hFPR']:.3f} | {row['all_keep']:.3f} |"
            )
        lines.extend(
            [
                "",
                "Interpretation: the dev-safe optimum degenerates to fixed `target@0.20`. In other words, structural thresholding does not currently beat the simpler result that Qwen can keep only 20% of visual tokens with small accuracy loss relative to `target@0.30` and no clear hFPR penalty.",
                "",
                "## Current Decisions",
                "",
                "- Qwen: report `target@0.30` as the quality point and `target@0.20` as the extreme-efficiency point.",
                "- LLaVA: report `embed@0.40` as the accuracy-efficiency point, `protected_embed@0.40` as the bbox/OCR evidence-preserving point, and `grid@0.40` as the low-hallucination spatial baseline.",
                "- InternVL: report only calibrated results; use risk-constrained `soft_evidence@0.50` as the cleanest balanced point, `target@0.50` as the low-hallucination reference, and `grid@0.50` as the spatial baseline. Pure accuracy-calibrated hard/soft evidence protection raises hFPR, so keep those as negative ablations.",
                "- Adaptive budget: do not claim a complex adaptive policy yet. The current evidence says fixed low-budget Qwen is stronger and cleaner.",
                "",
                "## Key Files",
                "",
                "- `runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_target0p20_efficiency_paired_stats.md`",
                "- `runs/llava_textocr_hard/llava15_7b_textocr_hard_checkpoint.md`",
                "- `runs/internvl_textocr_hard/internvl35_8b_textocr_hard_checkpoint.md`",
                "- `runs/cross_model_textocr_hard/cross_model_summary.csv`",
                "- `runs/cross_model_textocr_hard/qwen_structural_policy_candidates.csv`",
            ]
        )
    return "\n".join(lines) + "\n"


def render_summary_table(title: str, rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| run | n | acc | hFPR | yes | pos | neg | visual | keep | ECR | CenterR | PatchR | note |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['run']} | {row['n']} | {fmt(row['acc'])} | {fmt(row['hFPR'])} | {fmt(row['yes_rate'])} | "
            f"{fmt(row['pos_acc'])} | {fmt(row['neg_acc'])} | {fmt(row['kept_visual'])} | {fmt(row['keep_ratio'])} | "
            f"{fmt(row['ECR'])} | {fmt(row['CenterR'])} | {fmt(row['PatchR'])} | {row['note']} |"
        )
    return lines


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_scores(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["sample_id"]): row for row in read_jsonl(path)}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def split_for_row(row: dict[str, Any]) -> str:
    return "dev" if int(hashlib.sha1(group_id(row).encode("utf-8")).hexdigest()[:8], 16) % 10 < 5 else "test"


def group_id(row: dict[str, Any]) -> str:
    return str(row.get("image_id") or str(row.get("sample_id", "")).split(":")[0])


def quantile_thresholds(values: list[float], *, bins: int) -> list[float]:
    if not values:
        return []
    thresholds = {values[int((len(values) - 1) * idx / bins)] for idx in range(1, bins)}
    return sorted(thresholds)


def prefixed(prefix: str, row: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in row.items()}


def budget_tag(budget: float) -> str:
    return f"{budget:.2f}".replace(".", "p")


def mean(values: Any) -> float:
    vals = [float(value) for value in values]
    return sum(vals) / len(vals) if vals else 0.0


def mean_optional(values: Any) -> float | None:
    vals = [float(value) for value in values if value is not None]
    return sum(vals) / len(vals) if vals else None


def to_float(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def fmt(value: Any) -> str:
    if value == "":
        return "-"
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.3f}"


if __name__ == "__main__":
    main()
