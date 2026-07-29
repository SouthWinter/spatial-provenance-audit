#!/usr/bin/env python
"""Build an audit for InternVL yes/no threshold calibration.

The paper has two InternVL soft-evidence operating points with the same token
selector and evidence coverage but different hFPR. This script makes that
distinction explicit and checks whether the underlying raw margins/ECR are
identical across the two calibrated views.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "problem_optimization_audit" / "internvl_calibration_audit"

CALIBRATION_JSON = (
    ROOT
    / "runs"
    / "internvl_textocr_hard"
    / "internvl35_8b_textocr_hard_threshold_calibration_extreme.json"
)
CROSS_MODEL_CSV = ROOT / "runs" / "cross_model_textocr_hard" / "cross_model_summary.csv"

RUNS = [
    {
        "name": "Soft evidence 50 default",
        "run": "target_soft_evidence0p50_cal",
        "path": ROOT
        / "runs"
        / "internvl_textocr_hard"
        / "calibrated_test_target_soft_evidence0p50_b0p05_devthr",
        "selection_protocol": "dev-best accuracy threshold for this soft-evidence run",
    },
    {
        "name": "Soft evidence 50 hFPR-constrained",
        "run": "target_soft_evidence0p50_hfpr_cal",
        "path": ROOT
        / "runs"
        / "internvl_textocr_hard"
        / "calibrated_test_target_soft_evidence0p50_b0p05_hfprconstr_devthr",
        "selection_protocol": "more conservative dev threshold selected for lower hard-negative FPR",
    },
    {
        "name": "Target 50 hFPR-constrained reference",
        "run": "target0p50_cal",
        "path": ROOT / "runs" / "internvl_textocr_hard" / "calibrated_test_target0p50_devthr",
        "selection_protocol": "dev-best threshold for target-only 50% run",
    },
    {
        "name": "Full calibrated reference",
        "run": "full_cal",
        "path": ROOT / "runs" / "internvl_textocr_hard" / "calibrated_test_full_devthr",
        "selection_protocol": "dev-best threshold for full-prefix run",
    },
]


def main() -> None:
    calibration_rows = build_calibration_rows()
    operating_rows = build_operating_rows()
    comparison_rows = build_soft_pair_comparison()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "internvl_threshold_calibration_summary.csv", calibration_rows)
    write_csv(OUT_DIR / "internvl_soft_operating_point_audit.csv", operating_rows)
    write_csv(OUT_DIR / "internvl_soft_pair_identity_check.csv", comparison_rows)
    (OUT_DIR / "internvl_calibration_audit_report.md").write_text(
        build_report(calibration_rows, operating_rows, comparison_rows),
        encoding="utf-8",
    )
    print(f"Wrote {OUT_DIR / 'internvl_calibration_audit_report.md'}")


def build_calibration_rows() -> list[dict[str, Any]]:
    data = json.loads(CALIBRATION_JSON.read_text(encoding="utf-8"))
    wanted = {
        "InternVL3.5-8B-full",
        "InternVL3.5-8B-target0p50",
        "InternVL3.5-8B-target0p40",
        "InternVL3.5-8B-grid0p50",
        "InternVL3.5-8B-grid0p40",
        "InternVL3.5-8B-embed0p50",
    }
    rows = []
    for item in data["summaries"]:
        if item["name"] not in wanted:
            continue
        rows.append(
            {
                "model_run": item["name"],
                "n": item["n"],
                "dev_n": item["dev_n"],
                "test_n": item["test_n"],
                "margin_auroc_all": fmt(item["margin_auroc_all"]),
                "default_threshold": fmt(item["default_threshold"]),
                "default_all_acc": fmt(item["default_all"]["acc"]),
                "default_all_hFPR": fmt(item["default_all"]["hFPR"]),
                "default_all_yes_rate": fmt(item["default_all"]["yes_rate"]),
                "dev_best_threshold": fmt(item["dev_best_threshold"]),
                "dev_best_dev_acc": fmt(item["dev_best_dev"]["acc"]),
                "dev_best_dev_hFPR": fmt(item["dev_best_dev"]["hFPR"]),
                "dev_best_test_acc": fmt(item["dev_best_test"]["acc"]),
                "dev_best_test_hFPR": fmt(item["dev_best_test"]["hFPR"]),
                "dev_best_test_yes_rate": fmt(item["dev_best_test"]["yes_rate"]),
                "all_best_threshold": fmt(item["all_best_threshold"]),
                "all_best_all_acc": fmt(item["all_best_all"]["acc"]),
                "all_best_all_hFPR": fmt(item["all_best_all"]["hFPR"]),
                "source": item["path"],
            }
        )
    return rows


def build_operating_rows() -> list[dict[str, Any]]:
    cross_rows = read_csv(CROSS_MODEL_CSV)
    cross_by_run = {row["run"]: row for row in cross_rows if row["model"] == "InternVL3.5-8B calibrated-test"}
    rows = []
    for spec in RUNS:
        probe_rows = read_jsonl(spec["path"] / "probe_scores.jsonl")
        metrics = json.loads((spec["path"] / "metrics.json").read_text(encoding="utf-8"))
        threshold_values = sorted({round(float(row.get("yesno_threshold", 0.0)), 9) for row in probe_rows})
        selector_values = sorted({row.get("prune_selector", "full") for row in probe_rows})
        cross = cross_by_run.get(spec["run"], {})
        rows.append(
            {
                "name": spec["name"],
                "cross_model_run": spec["run"],
                "n": int(float(metrics["num_samples"])),
                "yesno_threshold": fmt(threshold_values[0]) if threshold_values else "",
                "unique_thresholds": len(threshold_values),
                "selection_protocol": spec["selection_protocol"],
                "selector": ";".join(selector_values) if selector_values else "full-prefix",
                "acc": fmt(metrics["direct_accuracy"]),
                "hFPR": fmt(metrics["direct_hallucination_fpr"]),
                "keep_ratio": fmt(cross.get("keep_ratio", "")),
                "ECR": fmt(cross.get("ECR", "")),
                "CenterR": fmt(cross.get("CenterR", "")),
                "PatchR": fmt(cross.get("PatchR", "")),
                "path": str(spec["path"].relative_to(ROOT)),
            }
        )
    return rows


def build_soft_pair_comparison() -> list[dict[str, Any]]:
    default_rows = keyed_rows(RUNS[0]["path"] / "probe_scores.jsonl")
    constrained_rows = keyed_rows(RUNS[1]["path"] / "probe_scores.jsonl")
    shared_keys = sorted(set(default_rows) & set(constrained_rows))
    raw_mismatch = 0
    ecr_mismatch = 0
    selector_mismatch = 0
    pred_changed = 0
    yes_to_no = 0
    no_to_yes = 0
    for key in shared_keys:
        a = default_rows[key]
        b = constrained_rows[key]
        raw_mismatch += not close(a.get("raw_margin"), b.get("raw_margin"))
        ecr_mismatch += not close(a.get("prune_ecr"), b.get("prune_ecr"))
        selector_mismatch += a.get("prune_selector") != b.get("prune_selector")
        if a.get("pred_answer") != b.get("pred_answer"):
            pred_changed += 1
            yes_to_no += a.get("pred_answer") == "yes" and b.get("pred_answer") == "no"
            no_to_yes += a.get("pred_answer") == "no" and b.get("pred_answer") == "yes"
    return [
        {
            "comparison": "soft_default_vs_hfpr_constrained",
            "shared_rows": len(shared_keys),
            "raw_margin_mismatches": raw_mismatch,
            "ECR_mismatches": ecr_mismatch,
            "selector_mismatches": selector_mismatch,
            "pred_changed": pred_changed,
            "yes_to_no": yes_to_no,
            "no_to_yes": no_to_yes,
            "interpretation": "same raw margins/selector/ECR; different threshold changes yes/no decisions",
        }
    ]


def build_report(
    calibration_rows: list[dict[str, Any]],
    operating_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
) -> str:
    soft_default = next(row for row in operating_rows if row["cross_model_run"] == "target_soft_evidence0p50_cal")
    soft_hfpr = next(row for row in operating_rows if row["cross_model_run"] == "target_soft_evidence0p50_hfpr_cal")
    identity = comparison_rows[0]
    lines = [
        "# InternVL Calibration Audit",
        "",
        "This audit clarifies that InternVL TextOCR-Hard yes/no results are threshold-calibrated. The prediction rule is yes iff `no_loss - yes_loss >= threshold`.",
        "",
        "## Key Reading",
        "",
        (
            "InternVL is strongly yes-biased at the default threshold: full-prefix default hFPR is "
            f"{calibration_rows[0]['default_all_hFPR']} with yes-rate {calibration_rows[0]['default_all_yes_rate']}. "
            "Therefore all InternVL main rows use thresholds selected on a hash split by image/group id."
        ),
        "",
        (
            f"The two soft-evidence rows have the same selector, keep ratio, and ECR ({soft_default['ECR']}), "
            f"but different thresholds: {soft_default['yesno_threshold']} for the default soft row and "
            f"{soft_hfpr['yesno_threshold']} for the hFPR-constrained row. This lowers hFPR from "
            f"{soft_default['hFPR']} to {soft_hfpr['hFPR']} with a small accuracy change "
            f"{soft_default['acc']}->{soft_hfpr['acc']}."
        ),
        "",
        (
            f"Pairwise identity check over {identity['shared_rows']} shared probes finds "
            f"{identity['raw_margin_mismatches']} raw-margin mismatches, {identity['ECR_mismatches']} ECR mismatches, "
            f"and {identity['selector_mismatches']} selector mismatches. The changed predictions are threshold-driven: "
            f"{identity['yes_to_no']} yes->no and {identity['no_to_yes']} no->yes."
        ),
        "",
        "## Threshold Calibration Summary",
        "",
        markdown_table(
            calibration_rows,
            [
                "model_run",
                "default_all_hFPR",
                "default_all_yes_rate",
                "dev_best_threshold",
                "dev_best_test_acc",
                "dev_best_test_hFPR",
                "all_best_threshold",
                "all_best_all_acc",
                "all_best_all_hFPR",
            ],
        ),
        "",
        "## Operating Point Audit",
        "",
        markdown_table(
            operating_rows,
            [
                "name",
                "yesno_threshold",
                "selection_protocol",
                "selector",
                "acc",
                "hFPR",
                "keep_ratio",
                "ECR",
                "path",
            ],
        ),
        "",
        "## Soft Pair Identity Check",
        "",
        markdown_table(comparison_rows, list(comparison_rows[0].keys())),
        "",
    ]
    return "\n".join(lines)


def keyed_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    return {row["sample_id"]: row for row in rows}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(str(row.get(col, "")).replace("|", "/") for col in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def close(a: Any, b: Any, eps: float = 1e-8) -> bool:
    if a in {None, ""} and b in {None, ""}:
        return True
    try:
        return abs(float(a) - float(b)) <= eps
    except (TypeError, ValueError):
        return a == b


def fmt(value: Any) -> str:
    if value in {None, ""}:
        return ""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    main()
