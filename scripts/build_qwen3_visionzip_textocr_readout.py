#!/usr/bin/env python3
"""Summarize the matched-budget Qwen3 VisionZip TextOCR-Hard run."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "qwen3_visionzip_textocr_hard"
PAPER_EVIDENCE = ROOT / "runs" / "paper_evidence"

RUNS = [
    {
        "method": "Qwen Target 0.30 (ours)",
        "comparison_type": "ours",
        "path": ROOT
        / "runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_topk_0p30_targetfix_802816",
        "note": "main Qwen quality operating point",
    },
    {
        "method": "Qwen VisionZip 0.30 (native port)",
        "comparison_type": "official_algorithm_port",
        "path": ROOT
        / "runs/qwen3_visionzip_textocr_hard/qwen3_8b_textocr_hard_full1000_visionzip_0p30_minmax802816",
        "note": "Qwen3-specific native port; min=max=802816 matched to main Qwen runs",
    },
    {
        "method": "Qwen Random 0.30",
        "comparison_type": "sanity_baseline",
        "path": ROOT / "runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_random_0p30",
        "note": "same-budget random mask",
    },
]

EXCLUDED_RUNS = [
    {
        "method": "Qwen VisionZip 0.30 (unmatched pixel policy)",
        "path": ROOT / "runs/qwen3_visionzip_textocr_hard/qwen3_8b_textocr_hard_full1000_visionzip_0p30_802816",
        "reason": "uses min=50176,max=802816 and therefore has fewer visual tokens than the main Qwen 802816 runs",
    }
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_EVIDENCE.mkdir(parents=True, exist_ok=True)
    rows = [summarize_run(spec) for spec in RUNS]
    excluded = [summarize_excluded(spec) for spec in EXCLUDED_RUNS if spec["path"].exists()]

    target = rows[0]
    for row in rows:
        row["matched_budget_to_target"] = str(
            abs(float(row["mean_full_visual_tokens"]) - float(target["mean_full_visual_tokens"])) < 1e-9
            and abs(float(row["mean_kept_visual_tokens"]) - float(target["mean_kept_visual_tokens"])) < 1e-9
        )
        row["accuracy_delta_vs_ours"] = fmt(float(row["accuracy"]) - float(target["accuracy"]))
        row["hfpr_delta_vs_ours"] = fmt(float(row["hFPR"]) - float(target["hFPR"]))
        row["ecr_delta_vs_ours"] = fmt(float(row["ECR"]) - float(target["ECR"]))

    write_csv(OUT_DIR / "metrics.csv", rows)
    write_csv(OUT_DIR / "excluded_runs.csv", excluded)
    write_csv(PAPER_EVIDENCE / "table_qwen3_visionzip_native_port.csv", rows)
    (OUT_DIR / "qwen3_visionzip_textocr_readout.md").write_text(build_report(rows, excluded), encoding="utf-8")
    print(f"Wrote {OUT_DIR / 'qwen3_visionzip_textocr_readout.md'}")


def summarize_run(spec: dict[str, Any]) -> dict[str, str]:
    path = Path(spec["path"])
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    traces = read_jsonl(path / "prune_traces.jsonl")
    if not traces:
        raise ValueError(f"No traces found in {path}")
    full_tokens = [float(row["full_visual_tokens"]) for row in traces]
    kept_tokens = [float(row["kept_visual_tokens"]) for row in traces]
    keep_ratios = [kept / full if full else 0.0 for kept, full in zip(kept_tokens, full_tokens)]
    anchor_ecr = [float(row.get("anchor_ecr", row.get("ecr", 0.0))) for row in traces]
    is_exhaustive_merge = all(
        int(row.get("visionzip_contextual_tokens", 0)) > 0 for row in traces
    ) and str(metrics.get("prune_selector", "")).strip().lower() in {
        "visionzip",
        "official_visionzip",
        "qwen3_visionzip",
    }
    lineage_ecr = [1.0 for _ in traces] if is_exhaustive_merge else anchor_ecr
    center = [1.0 for _ in traces] if is_exhaustive_merge else [
        float(row.get("evidence_center_recall", 0.0)) for row in traces
    ]
    patch = [1.0 for _ in traces] if is_exhaustive_merge else [
        float(row.get("evidence_patch_recall", 0.0)) for row in traces
    ]
    return {
        "method": spec["method"],
        "comparison_type": spec["comparison_type"],
        "n": fmt(metrics.get("num_samples", len(traces))),
        "accuracy": fmt(metrics["direct_accuracy"]),
        "hFPR": fmt(metrics["direct_hallucination_fpr"]),
        "mean_full_visual_tokens": fmt(mean(full_tokens)),
        "mean_kept_visual_tokens": fmt(mean(kept_tokens)),
        "mean_actual_keep_ratio": fmt(mean(keep_ratios)),
        "ECR": fmt(mean(lineage_ecr)),
        "AnchorECR": fmt(mean(anchor_ecr)),
        "center_recall": fmt(mean(center)),
        "patch_recall": fmt(mean(patch)),
        "selector": str(metrics.get("prune_selector", "")),
        "score_source": str(metrics.get("prune_score_source", "")),
        "run_dir": str(path.relative_to(ROOT)),
        "note": spec["note"],
    }


def summarize_excluded(spec: dict[str, Any]) -> dict[str, str]:
    path = Path(spec["path"])
    row = summarize_run(
        {
            "method": spec["method"],
            "comparison_type": "excluded_control",
            "path": path,
            "note": spec["reason"],
        }
    )
    row["exclusion_reason"] = spec["reason"]
    return row


def build_report(rows: list[dict[str, str]], excluded: list[dict[str, str]]) -> str:
    lines = [
        "# Qwen3 VisionZip TextOCR-Hard Native-Port Readout",
        "",
        "This report summarizes the matched-budget Qwen3 VisionZip native-port run. The VisionZip row uses the same TextOCR-Hard 1000-probe input and the same fixed 802816-pixel Qwen image policy as the main Qwen rows.",
        "",
        "## Verdict",
        "",
        "The native Qwen3 VisionZip port is runnable and matched-budget on TextOCR-Hard, but its merge-aware provenance needs two readings. LineageECR propagates every source cell into its contextual merge and is therefore 1.0; AnchorECR measures only the representative output locations. VisionZip lowers hFPR relative to Target 0.30 but does not match its accuracy. This is a fair external-baseline result, not a state-of-the-art claim.",
        "",
        "## Matched-Budget Rows",
        "",
        table_md(
            rows,
            [
                "method",
                "comparison_type",
                "n",
                "accuracy",
                "hFPR",
                "mean_actual_keep_ratio",
                "ECR",
                "AnchorECR",
                "center_recall",
                "patch_recall",
                "matched_budget_to_target",
            ],
        ),
        "",
        "## Deltas vs Ours",
        "",
        table_md(rows, ["method", "accuracy_delta_vs_ours", "hfpr_delta_vs_ours", "ecr_delta_vs_ours"]),
    ]
    if excluded:
        lines.extend(
            [
                "",
                "## Excluded Run",
                "",
                "The following completed run is excluded from matched-budget claims because its image pixel policy differs from the main Qwen 802816 setting.",
                "",
                table_md(excluded, ["method", "mean_full_visual_tokens", "mean_kept_visual_tokens", "accuracy", "ECR", "exclusion_reason"]),
            ]
        )
    return "\n".join(lines) + "\n"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    return f"{float(value):.6f}"


def table_md(rows: list[dict[str, str]], columns: list[str]) -> str:
    if not rows:
        return "(empty)"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")).replace("|", "\\|") for col in columns) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
