#!/usr/bin/env python
"""Audit InternVL soft-evidence beta without conflating selector and calibration."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recap.metrics import roc_auc
from scripts.calibrate_yesno_thresholds import best_threshold, evaluate, load_rows


DEFAULT_RUNS = {
    "0.00": Path(
        "runs/internvl_textocr_hard/"
        "internvl35_8b_textocr_hard_full1000_target_embed_topk_0p50/probe_scores.jsonl"
    ),
    "0.02": Path(
        "runs/internvl_textocr_hard/beta_sensitivity/soft_evidence_beta_0p02/probe_scores.jsonl"
    ),
    "0.05": Path(
        "runs/internvl_textocr_hard/"
        "internvl35_8b_textocr_hard_full1000_target_embed_soft_evidence_topk_0p50_b0p05/probe_scores.jsonl"
    ),
    "0.10": Path(
        "runs/internvl_textocr_hard/beta_sensitivity/soft_evidence_beta_0p10/probe_scores.jsonl"
    ),
    "0.20": Path(
        "runs/internvl_textocr_hard/beta_sensitivity/soft_evidence_beta_0p20/probe_scores.jsonl"
    ),
}
FULL_RUN = Path(
    "runs/internvl_textocr_hard/internvl35_8b_textocr_hard_full1000_direct/probe_scores.jsonl"
)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/internvl_textocr_hard/beta_sensitivity"),
    )
    args = parser.parse_args()

    missing = [str(path) for path in [FULL_RUN, *DEFAULT_RUNS.values()] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing score files:\n" + "\n".join(missing))

    full_rows = load_rows(FULL_RUN, dev_buckets=5)
    full_threshold = best_threshold([row for row in full_rows if row["split"] == "dev"])
    output_rows: list[dict[str, object]] = []

    for beta, path in DEFAULT_RUNS.items():
        rows = load_rows(path, dev_buckets=5)
        dev = [row for row in rows if row["split"] == "dev"]
        test = [row for row in rows if row["split"] == "test"]
        own_threshold = best_threshold(dev)
        own_metrics = evaluate(test, threshold=own_threshold)
        shared_metrics = evaluate(test, threshold=full_threshold)
        test_ids = {str(row["sample_id"]) for row in test}
        ecr_values = []
        keep_values = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            import json

            row = json.loads(raw)
            if str(row.get("sample_id", "")) not in test_ids:
                continue
            if row.get("prune_ecr") is not None:
                ecr_values.append(float(row["prune_ecr"]))
            if row.get("prune_keep_ratio") is not None:
                keep_values.append(float(row["prune_keep_ratio"]))
        output_rows.append(
            {
                "beta": beta,
                "test_auroc": roc_auc(
                    [1.0 if row["target_yes"] else 0.0 for row in test],
                    [row["margin"] for row in test],
                ),
                "own_threshold": own_threshold,
                "own_acc": own_metrics["acc"],
                "own_hfpr": own_metrics["hFPR"],
                "full_shared_acc": shared_metrics["acc"],
                "full_shared_hfpr": shared_metrics["hFPR"],
                "mean_ecr": mean(ecr_values),
                "mean_keep": mean(keep_values),
                "dev_n": len(dev),
                "test_n": len(test),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "beta_sensitivity.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    md_path = args.out_dir / "beta_sensitivity.md"
    lines = [
        "# InternVL soft-evidence beta sensitivity",
        "",
        "All thresholds are selected on the fixed 464-probe development split and evaluated on the 536-probe test split. The shared columns use the Full-prefix development threshold, so they do not give each selector a separate operating point.",
        "",
        "| Beta | AUROC | Own-threshold Acc. | Own-threshold hFPR | Full-shared Acc. | Full-shared hFPR | ECR | Keep |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in output_rows:
        lines.append(
            "| {beta} | {test_auroc:.3f} | {own_acc:.3f} | {own_hfpr:.3f} | "
            "{full_shared_acc:.3f} | {full_shared_hfpr:.3f} | {mean_ecr:.3f} | "
            "{mean_keep:.3f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "The beta conclusion should be based primarily on AUROC and shared-threshold behavior; per-method thresholds are reported to expose, rather than hide, calibration effects.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
