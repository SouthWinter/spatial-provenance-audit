#!/usr/bin/env python
"""Build the matched-budget SCOPE quality, parity, confirmation, and timing audit."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "official_baseline_extension" / "scope_textocr_hard_audit"
DEV_SCOPE = ROOT / "runs" / "llava_textocr_hard" / "llava15_7b_textocr_hard_full1000_scope_0p40" / "metrics.json"
PARITY = ROOT / "runs" / "official_baseline_extension" / "scope_port_parity" / "scope_port_parity.json"
DEV_STATS = ROOT / "runs" / "p0_stats" / "pairwise_stats.csv"
CONFIRMATION_STATS = ROOT / "runs" / "official_baseline_extension" / "scope_confirmation_pairwise.csv"
TIMING_SUMMARY = ROOT / "runs" / "efficiency" / "scope_repeated_exclusive" / "scope_repeated_timing_summary.csv"
CONFIRMATION_DIR = ROOT / "runs" / "textocr_confirmation"
CONFIRMATION_RUNS = {
    "Full prefix": CONFIRMATION_DIR / "llava15_7b_full" / "metrics.json",
    "Target (40%)": CONFIRMATION_DIR / "llava15_7b_target_0p40" / "metrics.json",
    "Protected (40%)": CONFIRMATION_DIR / "llava15_7b_protected_0p40" / "metrics.json",
    "Random (40%)": CONFIRMATION_DIR / "llava15_7b_random_0p40" / "metrics.json",
    "SCOPE (40%)": CONFIRMATION_DIR / "llava15_7b_scope_0p40" / "metrics.json",
    "VisionZip (40%)": CONFIRMATION_DIR / "llava15_7b_visionzip_0p40" / "metrics.json",
}
DEV_COMPARISONS = {
    "llava_protected0p40_vs_scope_official",
    "llava_target0p40_vs_scope_official",
    "llava_scope0p40_vs_visionzip_official",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def metric_row(label: str, path: Path) -> dict[str, Any]:
    data = load_json(path)
    pruning = data.get("pruning") or {}
    return {
        "method": label,
        "n": data["num_samples"],
        "accuracy": data["direct_accuracy"],
        "hFPR": data["direct_hallucination_fpr"],
        "keep_ratio": pruning.get("mean_keep_ratio", 1.0),
        "ECR": pruning.get("mean_ecr", 1.0),
        "center_recall": pruning.get("mean_evidence_center_recall", 1.0),
        "patch_recall": pruning.get("mean_evidence_patch_recall", 1.0),
    }


def scope_quality(path: Path) -> dict[str, Any]:
    return metric_row("SCOPE (40%)", path)


def dev_stats() -> list[dict[str, str]]:
    return [row for row in load_csv(DEV_STATS) if row["comparison"] in DEV_COMPARISONS]


def confirmation_stats() -> list[dict[str, str]]:
    return load_csv(CONFIRMATION_STATS)


def timing_rows() -> list[dict[str, Any]]:
    rows = []
    for row in load_csv(TIMING_SUMMARY):
        rows.append(
            {
                "method": row["method"],
                "repetitions": int(row["repetitions"]),
                "timed_probes_per_rep": int(row["timed_probes_per_rep"]),
                "keep_ratio": float(row["keep_ratio"]),
                "vision_ms_mean": float(row["mean_vision_ms_mean"]),
                "vision_ms_std": float(row["mean_vision_ms_std"]),
                "selector_ms_mean": float(row["mean_prune_overhead_ms_mean"]),
                "selector_ms_std": float(row["mean_prune_overhead_ms_std"]),
                "language_ms_mean": float(row["mean_language_ms_mean"]),
                "language_ms_std": float(row["mean_language_ms_std"]),
                "total_ms_mean": float(row["mean_forward_ms_mean"]),
                "total_ms_std": float(row["mean_forward_ms_std"]),
                "speedup_vs_full": float(row["speedup_vs_full"]),
            }
        )
    return rows


def main() -> None:
    parity = load_json(PARITY)
    payload = {
        "implementation": {
            "method": "SCOPE",
            "upstream": "kinredon/SCOPE",
            "commit": "6bf73069e0d61307051cfda8e25925bc7b7afdd9",
            "alpha": 1.0,
            "combination": "multiplicative",
            "parity_status": parity.get("status"),
            "official_source_sha256": parity.get("official_source_sha256"),
        },
        "development": scope_quality(DEV_SCOPE),
        "confirmation": [metric_row(label, path) for label, path in CONFIRMATION_RUNS.items()],
        "paired_development": dev_stats(),
        "paired_confirmation": confirmation_stats(),
        "exclusive_timing_repeated": timing_rows(),
        "claim_boundary": [
            "Protected uses evidence boxes; its comparison with box-free SCOPE is not resource-symmetric.",
            "On development data, box-free Target and SCOPE are tied on answer metrics; on the locked confirmation set, Target is 3.8 accuracy points higher while hFPR remains tied.",
            "SCOPE retains more annotated evidence than box-free Target on both splits, but evidence coverage alone does not guarantee lower answer risk.",
            "Across three fresh-process timing repetitions, SCOPE lowers LLM prefill time but its greedy coverage selector makes total single-sample forward slower than the full-prefix reference.",
        ],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "scope_textocr_hard_audit.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    write_timing_csv(payload["exclusive_timing_repeated"])
    (OUT_DIR / "scope_textocr_hard_audit.md").write_text(markdown(payload), encoding="utf-8")
    print(f"Wrote audit to {OUT_DIR.relative_to(ROOT)}")


def write_timing_csv(rows: list[dict[str, Any]]) -> None:
    path = OUT_DIR / "scope_exclusive_timing.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_table(lines: list[str], rows: list[dict[str, str]], label_key: str) -> None:
    lines.extend(
        [
            "| comparison | delta accuracy [95% CI] | p | delta hFPR [95% CI] | p |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        label = row[label_key]
        acc_p = row.get("acc_sign_p", row.get("acc_mcnemar_p", "nan"))
        hfpr_p = row.get("hFPR_sign_p", row.get("hfpr_mcnemar_p", "nan"))
        hfpr_diff = row.get("hFPR_diff", row.get("hfpr_diff", "nan"))
        hfpr_low = row.get("hFPR_diff_ci_low", row.get("hfpr_diff_ci_low", "nan"))
        hfpr_high = row.get("hFPR_diff_ci_high", row.get("hfpr_diff_ci_high", "nan"))
        lines.append(
            f"| {label} | {float(row['acc_diff']):+.3f} "
            f"[{float(row['acc_diff_ci_low']):+.3f}, {float(row['acc_diff_ci_high']):+.3f}] | "
            f"{float(acc_p):.4g} | {float(hfpr_diff):+.3f} "
            f"[{float(hfpr_low):+.3f}, {float(hfpr_high):+.3f}] | "
            f"{float(hfpr_p):.4g} |"
        )


def markdown(payload: dict[str, Any]) -> str:
    impl = payload["implementation"]
    development = payload["development"]
    lines = [
        "# SCOPE TextOCR-Hard Audit",
        "",
        "## Implementation Fidelity",
        "",
        f"- Upstream: `{impl['upstream']}@{impl['commit']}`.",
        f"- Official default: `alpha={impl['alpha']:.0f}`, {impl['combination']} saliency--coverage combination.",
        f"- Exact-index parity audit: **{impl['parity_status']}**.",
        f"- Extracted official source SHA-256: `{impl['official_source_sha256']}`.",
        "",
        "## Development Quality",
        "",
        "| method | probes | keep | accuracy | hFPR | ECR | CenterR | PatchR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| {development['method']} | {int(development['n'])} | {development['keep_ratio']:.3f} | "
        f"{development['accuracy']:.3f} | {development['hFPR']:.3f} | {development['ECR']:.3f} | "
        f"{development['center_recall']:.3f} | {development['patch_recall']:.3f} |",
        "",
        "## Locked Image-Disjoint Confirmation",
        "",
        "| method | probes | keep | accuracy | hFPR | ECR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["confirmation"]:
        lines.append(
            f"| {row['method']} | {int(row['n'])} | {row['keep_ratio']:.3f} | "
            f"{row['accuracy']:.3f} | {row['hFPR']:.3f} | {row['ECR']:.3f} |"
        )
    lines.extend(["", "## Paired Development Comparisons", ""])
    paired_table(lines, payload["paired_development"], "comparison")
    lines.extend(["", "## Paired Confirmation Comparisons", ""])
    paired_table(lines, payload["paired_confirmation"], "label")
    lines.extend(
        [
            "",
            "## Repeated Exclusive Single-Sample Timing",
            "",
            "Each method was measured in three fresh processes on the same 100 confirmation probes. The first five probes per process were discarded, leaving 95 timed probes per repetition. Method order was rotated across repetitions; all rows use one A800 GPU, eager attention, float16, and the same pruning backend.",
            "",
            "| method | keep | vision ms | selector/materialize ms | LLM ms | total ms | speedup |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["exclusive_timing_repeated"]:
        lines.append(
            f"| {row['method']} | {row['keep_ratio']:.3f} | "
            f"{row['vision_ms_mean']:.1f} +/- {row['vision_ms_std']:.1f} | "
            f"{row['selector_ms_mean']:.1f} +/- {row['selector_ms_std']:.1f} | "
            f"{row['language_ms_mean']:.1f} +/- {row['language_ms_std']:.1f} | "
            f"{row['total_ms_mean']:.1f} +/- {row['total_ms_std']:.1f} | "
            f"{row['speedup_vs_full']:.2f}x |"
        )
    lines.extend(["", "## Claim Boundary", ""])
    lines.extend(f"- {item}" for item in payload["claim_boundary"])
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
