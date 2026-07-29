#!/usr/bin/env python
"""Separate InternVL selector effects from yes/no threshold calibration."""

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
    "Full": Path("runs/internvl_textocr_hard/internvl35_8b_textocr_hard_full1000_direct/probe_scores.jsonl"),
    "Target": Path(
        "runs/internvl_textocr_hard/internvl35_8b_textocr_hard_full1000_target_embed_topk_0p50/probe_scores.jsonl"
    ),
    "Grid": Path("runs/internvl_textocr_hard/internvl35_8b_textocr_hard_full1000_grid_0p50/probe_scores.jsonl"),
    "Random": Path(
        "runs/internvl_textocr_hard/internvl35_8b_textocr_hard_full1000_random_0p50/probe_scores.jsonl"
    ),
    "Soft evidence": Path(
        "runs/internvl_textocr_hard/"
        "internvl35_8b_textocr_hard_full1000_target_embed_soft_evidence_topk_0p50_b0p05/probe_scores.jsonl"
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/internvl_textocr_hard/shared_threshold_audit"),
    )
    args = parser.parse_args()

    loaded = {name: load_rows(path, dev_buckets=5) for name, path in DEFAULT_RUNS.items()}
    full_threshold = best_threshold([row for row in loaded["Full"] if row["split"] == "dev"])
    pooled_threshold = best_threshold(
        [row for rows in loaded.values() for row in rows if row["split"] == "dev"]
    )

    output_rows = []
    for name, rows in loaded.items():
        dev = [row for row in rows if row["split"] == "dev"]
        test = [row for row in rows if row["split"] == "test"]
        method_threshold = best_threshold(dev)
        method_metrics = evaluate(test, threshold=method_threshold)
        shared_metrics = evaluate(test, threshold=full_threshold)
        pooled_metrics = evaluate(test, threshold=pooled_threshold)
        output_rows.append(
            {
                "method": name,
                "dev_n": len(dev),
                "test_n": len(test),
                "test_auroc": roc_auc(
                    [1.0 if row["target_yes"] else 0.0 for row in test],
                    [row["margin"] for row in test],
                ),
                "method_threshold": method_threshold,
                "method_acc": method_metrics["acc"],
                "method_hfpr": method_metrics["hFPR"],
                "full_shared_threshold": full_threshold,
                "full_shared_acc": shared_metrics["acc"],
                "full_shared_hfpr": shared_metrics["hFPR"],
                "pooled_shared_threshold": pooled_threshold,
                "pooled_shared_acc": pooled_metrics["acc"],
                "pooled_shared_hfpr": pooled_metrics["hFPR"],
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "internvl_threshold_decomposition.csv", output_rows)
    (args.out_dir / "internvl_threshold_decomposition.md").write_text(
        markdown(output_rows, full_threshold, pooled_threshold), encoding="utf-8"
    )
    print(f"Wrote {args.out_dir / 'internvl_threshold_decomposition.md'}")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown(rows: list[dict[str, object]], full_threshold: float, pooled_threshold: float) -> str:
    lines = [
        "# InternVL Threshold Decomposition",
        "",
        "All thresholds are selected on the fixed 464-probe development split and evaluated on the same 536-probe test split.",
        f"The Full-only shared threshold is {full_threshold:.3f}; the pooled shared threshold is {pooled_threshold:.3f}.",
        "",
        "| Method | Test AUROC | Per-method Acc. | Per-method hFPR | Full-shared Acc. | Full-shared hFPR | Pooled Acc. | Pooled hFPR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {test_auroc:.3f} | {method_acc:.3f} | {method_hfpr:.3f} | "
            "{full_shared_acc:.3f} | {full_shared_hfpr:.3f} | {pooled_shared_acc:.3f} | "
            "{pooled_shared_hfpr:.3f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Per-method thresholds conflate selector and operating-point calibration. Full-shared and pooled-shared columns isolate behavior under one common threshold, while AUROC is threshold-free.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
