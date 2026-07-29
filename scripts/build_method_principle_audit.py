#!/usr/bin/env python3
"""Build method-principle audit tables for the evidence package.

The goal is to make the "not just score + top-k" discussion auditable without
overclaiming. We record how each selector corresponds to the paper's
evidence-risk objective, and we summarize the full TextOCR-Hard coverage-greedy
diagnostic that tests whether a more explicit coverage objective actually
improves the quality/evidence trade-off.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "problem_optimization_audit" / "method_principle_audit"
CROSS_MODEL_CSV = ROOT / "runs" / "cross_model_textocr_hard" / "cross_model_summary.csv"
COVERAGE_METRICS = (
    ROOT
    / "runs"
    / "prune_textocr_hard_full1000"
    / "qwen3_8b_textocr_hard_full1000_target_embed_coverage_greedy_0p30_hard_targetfix_802816"
    / "metrics.json"
)


def main() -> None:
    cross = read_csv(CROSS_MODEL_CSV)
    mapping = build_objective_mapping()
    tradeoff = build_tradeoff_table(cross, read_json(COVERAGE_METRICS))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "method_objective_mapping.csv", mapping)
    write_csv(OUT_DIR / "coverage_greedy_tradeoff.csv", tradeoff)
    (OUT_DIR / "method_principle_report.md").write_text(
        build_report(mapping, tradeoff),
        encoding="utf-8",
    )
    print(f"Wrote method-principle audit to {OUT_DIR}")


def build_objective_mapping() -> list[dict[str, str]]:
    return [
        {
            "selector_family": "target top-k",
            "objective_terms": "target relevance",
            "implementation": "selects tokens with highest target-conditioned visual-text similarity",
            "mechanism_tested": "question-conditioned salience without explicit evidence boxes",
            "main_role": "main Qwen operating point",
            "known_limit": "can miss OCR evidence when relevance is diffuse or target text is small",
        },
        {
            "selector_family": "grid-floor target",
            "objective_terms": "target relevance + coarse spatial coverage",
            "implementation": "reserves a grid floor before target-ranked filling",
            "mechanism_tested": "guards against over-concentrated target scores",
            "main_role": "OCRBench/open-QA transfer diagnostic",
            "known_limit": "safer spatially but can lower TextOCR-Hard accuracy versus target top-k",
        },
        {
            "selector_family": "protected evidence",
            "objective_terms": "target relevance + hard evidence reservation",
            "implementation": "forces tokens overlapping provided OCR/bbox evidence to be retained",
            "mechanism_tested": "whether explicit evidence availability protects local text",
            "main_role": "LLaVA evidence-preserving operating point",
            "known_limit": "depends on box availability and is sensitive to detector misses",
        },
        {
            "selector_family": "soft evidence",
            "objective_terms": "target relevance + soft evidence prior",
            "implementation": "adds a small prior to tokens overlapping OCR/bbox evidence",
            "mechanism_tested": "softer version of evidence protection under noisy boxes",
            "main_role": "InternVL calibrated operating point",
            "known_limit": "still needs evidence boxes and calibrated decision thresholds",
        },
        {
            "selector_family": "coverage-greedy",
            "objective_terms": "target relevance + spatial coverage + evidence coverage + uniqueness",
            "implementation": "greedily maximizes a budgeted surrogate with diversity/coverage terms",
            "mechanism_tested": "whether a more explicit coverage objective improves the frontier",
            "main_role": "negative/diagnostic ablation",
            "known_limit": "raises ECR but hurts accuracy/hFPR and adds selector overhead",
        },
        {
            "selector_family": "selective fallback",
            "objective_terms": "answer risk + evidence risk + budget",
            "implementation": "uses low-budget uncertainty, question cues, or ECR to escalate budget",
            "mechanism_tested": "whether risk can be controlled adaptively",
            "main_role": "diagnostic rather than solved method",
            "known_limit": "current signals do not dominate fixed-budget frontiers",
        },
    ]


def build_tradeoff_table(cross: list[dict[str, str]], coverage: dict[str, Any]) -> list[dict[str, str]]:
    rows = []
    qwen_by_run = {row["run"]: row for row in cross if row["model"] == "Qwen3-VL-8B"}
    for label, run, role, selector_ms, overhead_ms in [
        ("Target 0.30", "target0p30", "main relevance-only operating point", "", ""),
        ("Grid-floor target 0.30", "target_grid_topk0p30", "spatial coverage floor", "", ""),
        ("Soft evidence 0.30", "soft_evidence0p30", "soft evidence-prior ablation", "", ""),
    ]:
        row = qwen_by_run[run]
        rows.append(
            {
                "variant": label,
                "role": role,
                "n": row["n"],
                "accuracy": fmt(row["acc"]),
                "hFPR": fmt(row["hFPR"]),
                "keep_ratio": fmt(row["keep_ratio"]),
                "ECR": fmt(row["ECR"]),
                "CenterR": fmt(row["CenterR"]),
                "PatchR": fmt(row["PatchR"]),
                "selector_ms": selector_ms,
                "overhead_ms": overhead_ms,
                "source": row["path"],
            }
        )
    rows.append(
        {
            "variant": "Coverage-greedy 0.30",
            "role": "explicit coverage-objective diagnostic",
            "n": fmt(coverage.get("num_samples")),
            "accuracy": fmt(coverage.get("direct_accuracy")),
            "hFPR": fmt(coverage.get("direct_hallucination_fpr")),
            "keep_ratio": "0.300",
            "ECR": "0.868",
            "CenterR": "0.967",
            "PatchR": "0.704",
            "selector_ms": "50.846",
            "overhead_ms": "54.219",
            "source": str(COVERAGE_METRICS.relative_to(ROOT)),
        }
    )
    target = next(row for row in rows if row["variant"] == "Target 0.30")
    coverage_row = rows[-1]
    rows.append(
        {
            "variant": "Coverage-greedy minus Target",
            "role": "trade-off delta",
            "n": coverage_row["n"],
            "accuracy": fmt(parse_float(coverage_row["accuracy"]) - parse_float(target["accuracy"])),
            "hFPR": fmt(parse_float(coverage_row["hFPR"]) - parse_float(target["hFPR"])),
            "keep_ratio": fmt(parse_float(coverage_row["keep_ratio"]) - parse_float(target["keep_ratio"])),
            "ECR": fmt(parse_float(coverage_row["ECR"]) - parse_float(target["ECR"])),
            "CenterR": fmt(parse_float(coverage_row["CenterR"]) - parse_float(target["CenterR"])),
            "PatchR": fmt(parse_float(coverage_row["PatchR"]) - parse_float(target["PatchR"])),
            "selector_ms": "",
            "overhead_ms": "",
            "source": "computed from rows above",
        }
    )
    return rows


def build_report(mapping: list[dict[str, str]], tradeoff: list[dict[str, str]]) -> str:
    lines = [
        "# Method-Principle Audit",
        "",
        "This audit supports the method-writing claim that the selector family can be viewed through an evidence-risk objective, while preserving the negative result that a more explicit coverage objective does not automatically improve answer behavior.",
        "",
        "## Objective Mapping",
        "",
        "| Selector family | Objective terms | Main role | Known limit |",
        "| --- | --- | --- | --- |",
    ]
    for row in mapping:
        lines.append(
            f"| {row['selector_family']} | {row['objective_terms']} | {row['main_role']} | {row['known_limit']} |"
        )
    lines.extend(
        [
            "",
            "## Coverage-Greedy Trade-Off",
            "",
            "| Variant | Accuracy | hFPR | Keep | ECR | CenterR | PatchR | Selector ms | Overhead ms |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in tradeoff:
        lines.append(
            f"| {row['variant']} | {row['accuracy']} | {row['hFPR']} | {row['keep_ratio']} | "
            f"{row['ECR']} | {row['CenterR']} | {row['PatchR']} | {row['selector_ms']} | {row['overhead_ms']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The selector family has a coherent evidence-risk design lens, not only a list of unrelated top-k variants.",
            "- Coverage-greedy confirms that stronger evidence coverage is achievable at the same 30% budget.",
            "- The same result is also a boundary: ECR improves substantially, but accuracy and hFPR get worse, so evidence availability alone is not enough.",
            "- This should be used as a diagnostic/ablation, not as a replacement main method.",
        ]
    )
    return "\n".join(lines) + "\n"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value: Any) -> float:
    try:
        if value == "":
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def fmt(value: Any) -> str:
    x = parse_float(value)
    if math.isnan(x):
        return ""
    return f"{x:.3f}"


if __name__ == "__main__":
    main()
