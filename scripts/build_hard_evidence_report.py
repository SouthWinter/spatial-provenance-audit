#!/usr/bin/env python
"""Build the hard-evidence package for visual-token pruning claims.

The report is intentionally conservative: missing runs are printed as missing
instead of silently dropped, and external-method rows are labelled as
protocol-compatible proxies unless an official implementation path is supplied.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "hard_evidence"


@dataclass(frozen=True)
class RunSpec:
    group: str
    family: str
    model: str
    label: str
    path: Path
    role: str
    baseline: str = ""
    note: str = ""


RUN_SPECS = [
    # Same-budget baselines for Qwen extreme-efficiency and quality points.
    RunSpec("same_budget_qwen_0p20", "same_budget", "Qwen3-VL-8B", "ours_target0p20", Path("runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_topk_0p20"), "ours", note="main extreme-efficiency point"),
    RunSpec("same_budget_qwen_0p20", "same_budget", "Qwen3-VL-8B", "random0p20", Path("runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_random_0p20"), "same-budget random", baseline="ours_target0p20"),
    RunSpec("same_budget_qwen_0p20", "same_budget", "Qwen3-VL-8B", "grid0p20", Path("runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_grid_0p20"), "same-budget grid", baseline="ours_target0p20"),
    RunSpec("same_budget_qwen_0p20", "same_budget", "Qwen3-VL-8B", "shuffled_score0p20", Path("runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_shuffled_topk_0p20"), "same-score shuffled", baseline="ours_target0p20"),
    RunSpec("same_budget_qwen_0p30", "same_budget", "Qwen3-VL-8B", "ours_target0p30", Path("runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_topk_0p30_targetfix_802816"), "ours", note="main quality point"),
    RunSpec("same_budget_qwen_0p30", "same_budget", "Qwen3-VL-8B", "random0p30", Path("runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_random_0p30"), "same-budget random", baseline="ours_target0p30"),
    RunSpec("same_budget_qwen_0p30", "same_budget", "Qwen3-VL-8B", "grid0p30", Path("runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_grid_0p30"), "same-budget grid", baseline="ours_target0p30"),
    RunSpec("same_budget_qwen_0p30", "same_budget", "Qwen3-VL-8B", "shuffled_score0p30", Path("runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_shuffled_topk_0p30"), "same-score shuffled", baseline="ours_target0p30"),
    RunSpec("same_budget_llava_0p40", "same_budget", "LLaVA-1.5-7B", "ours_protected_embed0p40", Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_embed_protected_topk_0p40"), "ours", note="bbox/OCR evidence-protected point"),
    RunSpec("same_budget_llava_0p40", "same_budget", "LLaVA-1.5-7B", "random0p40", Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_random_0p40"), "same-budget random", baseline="ours_protected_embed0p40"),
    RunSpec("same_budget_llava_0p40", "same_budget", "LLaVA-1.5-7B", "grid0p40", Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_grid_0p40"), "same-budget grid", baseline="ours_protected_embed0p40"),
    RunSpec("same_budget_llava_0p40", "same_budget", "LLaVA-1.5-7B", "shuffled_score0p40", Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_embed_shuffled_topk_0p40"), "same-score shuffled", baseline="ours_protected_embed0p40"),
    RunSpec("same_budget_internvl_0p50", "same_budget", "InternVL3.5-8B", "ours_soft_evidence0p50_hfpr_cal", Path("runs/internvl_textocr_hard/calibrated_test_target_soft_evidence0p50_b0p05_hfprconstr_devthr"), "ours", note="risk-constrained calibrated point"),
    RunSpec("same_budget_internvl_0p50", "same_budget", "InternVL3.5-8B", "random0p50_cal", Path("runs/internvl_textocr_hard/calibrated_test_random0p50_devthr"), "same-budget random", baseline="ours_soft_evidence0p50_hfpr_cal"),
    RunSpec("same_budget_internvl_0p50", "same_budget", "InternVL3.5-8B", "grid0p50_cal", Path("runs/internvl_textocr_hard/calibrated_test_grid0p50_devthr"), "same-budget grid", baseline="ours_soft_evidence0p50_hfpr_cal"),
    RunSpec("same_budget_internvl_0p50", "same_budget", "InternVL3.5-8B", "shuffled_score0p50_cal", Path("runs/internvl_textocr_hard/calibrated_test_target_shuffled0p50_devthr"), "same-score shuffled", baseline="ours_soft_evidence0p50_hfpr_cal"),
    # Causal evidence ablations. `topk` uses OCR/bbox evidence overlap as an oracle relevance score;
    # `bottomk` keeps the least evidence-overlapping tokens under the same budget.
    RunSpec("causal_qwen_0p20", "causal", "Qwen3-VL-8B", "full", Path("runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_topk_1p00"), "full reference"),
    RunSpec("causal_qwen_0p20", "causal", "Qwen3-VL-8B", "ours_target0p20", Path("runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_topk_0p20"), "ours"),
    RunSpec("causal_qwen_0p20", "causal", "Qwen3-VL-8B", "evidence_topk0p20", Path("runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_topk_0p20"), "evidence sufficiency", baseline="anti_evidence_bottomk0p20"),
    RunSpec("causal_qwen_0p20", "causal", "Qwen3-VL-8B", "anti_evidence_bottomk0p20", Path("runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_bottomk_0p20"), "evidence necessity"),
    RunSpec("causal_qwen_0p30", "causal", "Qwen3-VL-8B", "full", Path("runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_topk_1p00"), "full reference"),
    RunSpec("causal_qwen_0p30", "causal", "Qwen3-VL-8B", "ours_target0p30", Path("runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_topk_0p30_targetfix_802816"), "ours"),
    RunSpec("causal_qwen_0p30", "causal", "Qwen3-VL-8B", "evidence_topk0p30", Path("runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_topk_0p30"), "evidence sufficiency", baseline="anti_evidence_bottomk0p30"),
    RunSpec("causal_qwen_0p30", "causal", "Qwen3-VL-8B", "anti_evidence_bottomk0p30", Path("runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_bottomk_0p30"), "evidence necessity"),
    RunSpec("causal_llava_0p40", "causal", "LLaVA-1.5-7B", "full", Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_direct"), "full reference"),
    RunSpec("causal_llava_0p40", "causal", "LLaVA-1.5-7B", "ours_protected_embed0p40", Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_embed_protected_topk_0p40"), "ours"),
    RunSpec("causal_llava_0p40", "causal", "LLaVA-1.5-7B", "evidence_topk0p40", Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_topk_0p40"), "evidence sufficiency", baseline="anti_evidence_bottomk0p40"),
    RunSpec("causal_llava_0p40", "causal", "LLaVA-1.5-7B", "anti_evidence_bottomk0p40", Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_bottomk_0p40"), "evidence necessity"),
    RunSpec("causal_internvl_0p50", "causal", "InternVL3.5-8B", "full_cal", Path("runs/internvl_textocr_hard/calibrated_test_full_devthr"), "full reference"),
    RunSpec("causal_internvl_0p50", "causal", "InternVL3.5-8B", "ours_soft_evidence0p50_hfpr_cal", Path("runs/internvl_textocr_hard/calibrated_test_target_soft_evidence0p50_b0p05_hfprconstr_devthr"), "ours"),
    RunSpec("causal_internvl_0p50", "causal", "InternVL3.5-8B", "evidence_topk0p50_cal", Path("runs/internvl_textocr_hard/calibrated_test_topk0p50_devthr"), "evidence sufficiency", baseline="anti_evidence_bottomk0p50_cal"),
    RunSpec("causal_internvl_0p50", "causal", "InternVL3.5-8B", "anti_evidence_bottomk0p50_cal", Path("runs/internvl_textocr_hard/calibrated_test_bottomk0p50_devthr"), "evidence necessity"),
    # External method-level comparisons. These rows are local ports of public
    # reference algorithms; keep them separate from looser protocol proxies.
    RunSpec("external_method_llava_0p40", "external_method", "LLaVA-1.5-7B", "ours_protected_embed0p40", Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_embed_protected_topk_0p40"), "ours", note="same budget as VisionZip"),
    RunSpec("external_method_llava_0p40", "external_method", "LLaVA-1.5-7B", "VisionZip_official_alg0p40", Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_visionzip_0p40"), "VisionZip official-algorithm port", baseline="ours_protected_embed0p40", note="ported from JIA-Lab-research/VisionZip@8f86b55; CLS-attention dominant tokens plus contextual merge"),
    RunSpec("external_method_llava_0p40", "external_method", "LLaVA-1.5-7B", "FastV_official_alg0p40", Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_fastv_0p40"), "FastV official-algorithm port", baseline="ours_protected_embed0p40", note="ported from pkunlp-icler/fastv@d165972; prune at K=3 from previous-layer last-token attention"),
    # Protocol-compatible external-method proxies.
    RunSpec("external_proxy_qwen", "external_proxy", "Qwen3-VL-8B", "FastV_like_embed_topk0p30", Path("runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_embed_topk_0p30"), "FastV/TopV-style proxy", note="single-pass visual/text embedding salience top-k, not official code"),
    RunSpec("external_proxy_qwen", "external_proxy", "Qwen3-VL-8B", "VisionZip_like_embed_rise0p30", Path("runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_embed_rise_0p30"), "VisionZip-style proxy", note="dominant relevance plus diversity/coverage, not official code"),
    RunSpec("external_proxy_qwen", "external_proxy", "Qwen3-VL-8B", "ours_target0p30", Path("runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_topk_0p30_targetfix_802816"), "ours"),
    RunSpec("external_proxy_llava", "external_proxy", "LLaVA-1.5-7B", "FastV_like_embed_topk0p40", Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_embed_topk_0p40"), "FastV/TopV-style proxy", note="single-pass embedding salience top-k, not official code"),
    RunSpec("external_proxy_llava", "external_proxy", "LLaVA-1.5-7B", "ours_protected_embed0p40", Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_embed_protected_topk_0p40"), "ours"),
    RunSpec("external_proxy_internvl", "external_proxy", "InternVL3.5-8B", "FastV_like_embed0p50_cal", Path("runs/internvl_textocr_hard/calibrated_test_embed0p50_devthr"), "FastV/TopV-style proxy", note="single-pass embedding salience top-k, not official code"),
    RunSpec("external_proxy_internvl", "external_proxy", "InternVL3.5-8B", "ours_soft_evidence0p50_hfpr_cal", Path("runs/internvl_textocr_hard/calibrated_test_target_soft_evidence0p50_b0p05_hfprconstr_devthr"), "ours"),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [row_for_spec(spec) for spec in RUN_SPECS]
    add_pairwise_deltas(rows)
    write_csv(OUT_DIR / "hard_evidence_summary.csv", rows)
    (OUT_DIR / "hard_evidence_report.md").write_text(build_markdown(rows))
    print(f"Wrote {OUT_DIR / 'hard_evidence_report.md'}")
    print(f"Wrote {OUT_DIR / 'hard_evidence_summary.csv'}")


def row_for_spec(spec: RunSpec) -> dict[str, Any]:
    run_dir = ROOT / spec.path
    metrics_path = run_dir / "metrics.json"
    score_path = run_dir / "probe_scores.jsonl"
    trace_path = trace_path_for(run_dir, metrics_path)
    row: dict[str, Any] = {
        "family": spec.family,
        "group": spec.group,
        "model": spec.model,
        "label": spec.label,
        "role": spec.role,
        "baseline": spec.baseline,
        "path": str(spec.path),
        "note": spec.note,
        "status": "done" if metrics_path.exists() and score_path.exists() else "missing",
        "n": "",
        "acc": "",
        "hFPR": "",
        "yes_rate": "",
        "pos_acc": "",
        "neg_acc": "",
        "full_visual": "",
        "kept_visual": "",
        "keep_ratio": "",
        "ECR": "",
        "CenterR": "",
        "PatchR": "",
        "acc_delta_vs_baseline": "",
        "hFPR_delta_vs_baseline": "",
        "ECR_delta_vs_baseline": "",
    }
    if row["status"] != "done":
        return row
    metrics = read_json(metrics_path)
    scores = read_jsonl(score_path)
    traces = read_jsonl(trace_path) if trace_path.exists() else []
    quality = summarize_scores(scores, metrics)
    pruning = summarize_traces(traces, scores)
    row.update(quality)
    row.update(pruning)
    return row


def trace_path_for(run_dir: Path, metrics_path: Path) -> Path:
    direct = run_dir / "prune_traces.jsonl"
    if direct.exists() or not metrics_path.exists():
        return direct
    metrics = read_json(metrics_path)
    input_score = metrics.get("yesno_input_score")
    if input_score:
        candidate = (ROOT / str(input_score)).parent / "prune_traces.jsonl"
        if candidate.exists():
            return candidate
    return direct


def summarize_scores(rows: list[dict[str, Any]], metrics: dict[str, Any]) -> dict[str, Any]:
    neg = [row for row in rows if row.get("binary_polarity") == "negative"]
    pos = [row for row in rows if row.get("binary_polarity") == "positive"]
    return {
        "n": len(rows),
        "acc": float(metrics.get("direct_accuracy", mean_bool(row.get("correct") for row in rows))),
        "hFPR": float(metrics.get("direct_hallucination_fpr", h_fpr(rows))),
        "yes_rate": mean_bool(row.get("pred_answer") == "yes" for row in rows),
        "pos_acc": mean_bool(row.get("correct") for row in pos),
        "neg_acc": mean_bool(row.get("correct") for row in neg),
    }


def summarize_traces(rows: list[dict[str, Any]], scores: list[dict[str, Any]]) -> dict[str, Any]:
    wanted = {(str(row.get("sample_id", "")), str(row.get("probe", ""))) for row in scores}
    if wanted:
        rows = [
            row
            for row in rows
            if (str(row.get("sample_id", "")), str(row.get("probe", ""))) in wanted
        ]
    if not rows:
        return {"full_visual": "", "kept_visual": "", "keep_ratio": "", "ECR": "", "CenterR": "", "PatchR": ""}
    full = mean_numeric(row.get("full_visual_tokens") for row in rows)
    kept = mean_numeric(row.get("kept_visual_tokens") for row in rows)
    evidence_rows = [row for row in rows if row.get("has_evidence")]
    return {
        "full_visual": full,
        "kept_visual": kept,
        "keep_ratio": kept / full if isinstance(full, float) and full > 0.0 else "",
        "ECR": mean_numeric(row.get("ecr") for row in evidence_rows),
        "CenterR": mean_numeric(row.get("evidence_center_recall") for row in evidence_rows),
        "PatchR": mean_numeric(row.get("evidence_patch_recall") for row in evidence_rows),
    }


def add_pairwise_deltas(rows: list[dict[str, Any]]) -> None:
    by_group_label = {(str(row["group"]), str(row["label"])): row for row in rows}
    for row in rows:
        baseline = str(row.get("baseline") or "")
        if not baseline:
            continue
        base = by_group_label.get((str(row["group"]), baseline))
        if not base:
            continue
        for out_key, key in (
            ("acc_delta_vs_baseline", "acc"),
            ("hFPR_delta_vs_baseline", "hFPR"),
            ("ECR_delta_vs_baseline", "ECR"),
        ):
            if is_number(row.get(key)) and is_number(base.get(key)):
                row[out_key] = float(row[key]) - float(base[key])


def build_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Hard Evidence Report",
        "",
        "This report tracks the three reviewer-facing evidence gaps: same-budget baselines, causal evidence ablations, and external-method comparisons. Missing rows are intentional placeholders for runs that have not completed yet.",
        "",
        "External-method rows marked as official-algorithm ports are local ports of public reference algorithms. External-method rows marked as proxies are protocol-compatible reimplementations inside this codebase, not official numbers.",
        "",
    ]
    for family in ("same_budget", "causal", "external_method", "external_proxy"):
        family_rows = [row for row in rows if row["family"] == family]
        lines.extend([f"## {family}", ""])
        for group in sorted({str(row["group"]) for row in family_rows}):
            group_rows = [row for row in family_rows if row["group"] == group]
            lines.extend(
                [
                    f"### {group}",
                    "",
                    "| label | role | status | n | acc | hFPR | keep | ECR | CenterR | PatchR | Δacc | ΔhFPR | ΔECR | note |",
                    "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
                ]
            )
            for row in group_rows:
                lines.append(
                    "| {label} | {role} | {status} | {n} | {acc} | {hfpr} | {keep} | {ecr} | {center} | {patch} | {dacc} | {dhfpr} | {decr} | {note} |".format(
                        label=row["label"],
                        role=row["role"],
                        status=row["status"],
                        n=fmt(row["n"], 0),
                        acc=fmt(row["acc"], 3),
                        hfpr=fmt(row["hFPR"], 3),
                        keep=fmt(row["keep_ratio"], 3),
                        ecr=fmt(row["ECR"], 3),
                        center=fmt(row["CenterR"], 3),
                        patch=fmt(row["PatchR"], 3),
                        dacc=fmt(row["acc_delta_vs_baseline"], 3, signed=True),
                        dhfpr=fmt(row["hFPR_delta_vs_baseline"], 3, signed=True),
                        decr=fmt(row["ECR_delta_vs_baseline"], 3, signed=True),
                        note=row["note"],
                    )
                )
            lines.append("")
    lines.extend(["## Missing Runs", ""])
    missing = [row for row in rows if row["status"] != "done"]
    if not missing:
        lines.append("None.")
    else:
        for row in missing:
            lines.append(f"- `{row['path']}` ({row['group']} / {row['label']})")
    lines.append("")
    return "\n".join(lines)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "family",
        "group",
        "model",
        "label",
        "role",
        "baseline",
        "status",
        "n",
        "acc",
        "hFPR",
        "yes_rate",
        "pos_acc",
        "neg_acc",
        "full_visual",
        "kept_visual",
        "keep_ratio",
        "ECR",
        "CenterR",
        "PatchR",
        "acc_delta_vs_baseline",
        "hFPR_delta_vs_baseline",
        "ECR_delta_vs_baseline",
        "path",
        "note",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_bool(values: Any) -> float:
    items = [bool(value) for value in values]
    return sum(1 for value in items if value) / len(items) if items else 0.0


def h_fpr(rows: list[dict[str, Any]]) -> float:
    neg = [row for row in rows if row.get("binary_polarity") == "negative"]
    if not neg:
        return 0.0
    return sum(1 for row in neg if row.get("pred_answer") == "yes") / len(neg)


def mean_numeric(values: Any) -> float | str:
    nums = [float(value) for value in values if isinstance(value, (int, float))]
    return sum(nums) / len(nums) if nums else ""


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float))


def fmt(value: Any, digits: int, *, signed: bool = False) -> str:
    if value == "" or value is None:
        return ""
    if isinstance(value, int) and digits == 0:
        return str(value)
    if isinstance(value, (int, float)):
        prefix = "+" if signed and float(value) > 0 else ""
        return f"{prefix}{float(value):.{digits}f}"
    return str(value)


if __name__ == "__main__":
    sys.exit(main())
