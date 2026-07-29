#!/usr/bin/env python
"""Summarize TextOCR-Hard deletion/restoration runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


VARIANTS = (
    "selected",
    "remove_evidence",
    "restore_evidence_0p25",
    "restore_evidence_0p50",
    "restore_evidence_1p00",
    "restore_random_0p25",
    "restore_random_0p50",
    "restore_random_1p00",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="runs/textocr_deletion_restoration/qwen_target30_runs")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    output_dir = Path(args.output_dir) if args.output_dir else runs_root
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for variant in VARIANTS:
        metrics_path = runs_root / variant / "metrics.json"
        if not metrics_path.exists():
            rows.append({"variant": variant, "status": "missing"})
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        pruning = metrics.get("pruning", {})
        rows.append(
            {
                "variant": variant,
                "status": "done",
                "n": int(metrics.get("num_samples", 0) or metrics.get("n", 0) or 0),
                "acc": float(metrics.get("direct_accuracy", 0.0) or 0.0),
                "hFPR": float(metrics.get("direct_hallucination_fpr", 0.0) or 0.0),
                "keep_ratio": float(pruning.get("mean_keep_ratio", 0.0) or 0.0),
                "ECR": float(pruning.get("mean_ecr", 0.0) or 0.0),
                "CenterR": float(pruning.get("mean_evidence_center_recall", 0.0) or 0.0),
                "PatchR": float(pruning.get("mean_evidence_patch_recall", 0.0) or 0.0),
            }
        )

    selected = next((row for row in rows if row.get("variant") == "selected" and row.get("status") == "done"), None)
    removed = next((row for row in rows if row.get("variant") == "remove_evidence" and row.get("status") == "done"), None)
    if selected and removed:
        selected_removed_gap = float(selected["acc"]) - float(removed["acc"])
        denom = selected_removed_gap if selected_removed_gap > 0.0 else None
        for row in rows:
            if row.get("status") != "done":
                continue
            row["delta_acc_vs_removed"] = float(row["acc"]) - float(removed["acc"])
            row["gap_acc_to_selected"] = float(row["acc"]) - float(selected["acc"])
            row["acc_recovery_frac"] = row["delta_acc_vs_removed"] / denom if denom else None

    write_csv(output_dir / "deletion_restoration_summary.csv", rows)
    write_markdown(output_dir / "deletion_restoration_report.md", rows)
    print(f"Wrote deletion/restoration report to {output_dir}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# TextOCR-Hard Deletion/Restoration Report",
        "",
        "| Variant | Status | Acc. | hFPR | Keep | ECR | Delta Acc. vs Removed | Recovery |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row.get("status") != "done":
            lines.append(f"| {row['variant']} | {row.get('status', 'missing')} | -- | -- | -- | -- | -- | -- |")
            continue
        delta = row.get("delta_acc_vs_removed")
        recovery = row.get("acc_recovery_frac")
        delta_text = "--" if delta is None else f"{delta:+.3f}"
        recovery_text = "--" if recovery is None else f"{100.0 * recovery:.1f}%"
        lines.append(
            f"| {row['variant']} | done | {row['acc']:.3f} | {row['hFPR']:.3f} | "
            f"{row['keep_ratio']:.3f} | {row['ECR']:.3f} | {delta_text} | {recovery_text} |"
        )

    paired = []
    for suffix in ("0p25", "0p50", "1p00"):
        evidence = next((row for row in rows if row.get("variant") == f"restore_evidence_{suffix}"), None)
        random = next((row for row in rows if row.get("variant") == f"restore_random_{suffix}"), None)
        if evidence and random and evidence.get("status") == "done" and random.get("status") == "done":
            paired.append((suffix, evidence, random))
    if paired:
        lines.extend(
            [
                "",
                "## Evidence-vs-Random Restoration",
                "",
                "| Restored budget | Evidence Acc. | Random Acc. | Evidence Advantage | Evidence ECR | Random ECR |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for suffix, evidence, random in paired:
            budget = suffix.replace("p", ".")
            lines.append(
                f"| {budget} | {evidence['acc']:.3f} | {random['acc']:.3f} | "
                f"{evidence['acc'] - random['acc']:+.3f} | {evidence['ECR']:.3f} | {random['ECR']:.3f} |"
            )
    lines.extend(
        [
            "",
            "Expected causal readout: removing selected evidence should reduce answer support or accuracy; restoring evidence tokens should recover more than restoring the same number of random non-evidence tokens.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
