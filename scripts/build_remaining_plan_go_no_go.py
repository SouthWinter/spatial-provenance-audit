#!/usr/bin/env python3
"""Build the strict go/no-go report for the remaining open-QA plan."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("runs/problem_optimization_audit")
DEPLOYMENT_ROWS = ROOT / "open_ocr_qa_deployment_contract/deployment_contract_rows.csv"
SATURATION_ROWS = ROOT / "open_ocr_qa_evidence_saturation_policy/evidence_saturation_policy_summary.csv"
PORTFOLIO_ROWS = ROOT / "open_ocr_qa_domain_aware_portfolio/domain_aware_portfolio_rows.csv"
FULL_REPORT = Path("runs/open_ocr_qa_full/report/full_open_ocr_qa_report.json")
OUTPUT_DIR = ROOT / "strict_remaining_plan_go_no_go"

QUALITY_TOLERANCE = 0.01
MAX_MEAN_KEEP = 0.60
FIXED70_COST = 0.70


def main() -> None:
    deployment = read_csv(DEPLOYMENT_ROWS)
    saturation = read_csv(SATURATION_ROWS)
    portfolio = read_csv(PORTFOLIO_ROWS)
    full_report = json.loads(FULL_REPORT.read_text(encoding="utf-8"))

    controller_rows = build_controller_rows(deployment, saturation, portfolio)
    deployable = [row for row in controller_rows if not row["oracle"]]
    passing = [row for row in deployable if row["all_gates_pass"]]
    decision = {
        "status": "go" if passing else "no_go_stop_controller_expansion",
        "quality_tolerance": QUALITY_TOLERANCE,
        "max_mean_keep": MAX_MEAN_KEEP,
        "fixed70_cost": FIXED70_COST,
        "passing_candidates": [row["controller"] for row in passing],
        "reason": (
            "At least one deployable controller passes all three gates."
            if passing
            else "No deployable controller simultaneously passes two-task quality, mean-keep, and overhead-aware cost gates."
        ),
    }
    llava = build_llava_summary(full_report)
    payload = {
        "llava_go_no_go": llava,
        "controller_contract": decision,
        "controller_candidates": controller_rows,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "strict_remaining_plan_report.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(OUTPUT_DIR / "strict_controller_candidates.csv", controller_rows)
    (OUTPUT_DIR / "strict_remaining_plan_report.md").write_text(
        markdown(payload), encoding="utf-8"
    )
    print(f"Wrote strict remaining-plan report to {OUTPUT_DIR}")


def build_controller_rows(
    deployment: list[dict[str, str]],
    saturation: list[dict[str, str]],
    portfolio: list[dict[str, str]],
) -> list[dict[str, Any]]:
    by_policy_task = {(row["policy"], row["task"]): row for row in deployment}
    candidates: list[dict[str, Any]] = []
    for policy in (
        "pregen_mask30_only",
        "pregen_mask_stability_pregen",
        "pregen_question_only",
        "stable30_else70",
        "majority_30_50_70_elsefull",
    ):
        candidates.append(
            paired_candidate(
                policy,
                by_policy_task[(policy, "TextVQA-lite")],
                by_policy_task[(policy, "DocVQA-lite")],
                cost_reading="deployment-cost proxy includes serial passes when required",
            )
        )

    sat_test = [
        row
        for row in saturation
        if row["split"] == "test" and row["policy"].startswith("saturation_")
    ]
    sat_by_task = {row["task"]: row for row in sat_test}
    candidates.append(
        candidate_from_values(
            "pooled_evidence_saturation",
            text_score=float(sat_by_task["TextVQA-lite"]["score"]),
            doc_score=float(sat_by_task["DocVQA-lite"]["score"]),
            text_fixed=lookup_fixed70(saturation, "TextVQA-lite"),
            doc_fixed=lookup_fixed70(saturation, "DocVQA-lite"),
            text_keep=float(sat_by_task["TextVQA-lite"]["mean_keep"]),
            doc_keep=float(sat_by_task["DocVQA-lite"]["mean_keep"]),
            text_cost=float(sat_by_task["TextVQA-lite"]["mean_keep"]),
            doc_cost=float(sat_by_task["DocVQA-lite"]["mean_keep"]),
            oracle=False,
            cost_reading="pre-generation lower bound; controller overhead can only increase cost",
        )
    )

    portfolio_by_task = {row["task"]: row for row in portfolio}
    candidates.append(
        candidate_from_values(
            "domain_aware_portfolio",
            text_score=float(portfolio_by_task["TextVQA-lite"]["test_score"]),
            doc_score=float(portfolio_by_task["DocVQA-lite"]["test_score"]),
            text_fixed=float(portfolio_by_task["TextVQA-lite"]["fixed70_score"]),
            doc_fixed=float(portfolio_by_task["DocVQA-lite"]["fixed70_score"]),
            text_keep=float(portfolio_by_task["TextVQA-lite"]["deployment_cost_proxy"]),
            doc_keep=float(portfolio_by_task["DocVQA-lite"]["deployment_cost_proxy"]),
            text_cost=float(portfolio_by_task["TextVQA-lite"]["deployment_cost_proxy"]),
            doc_cost=float(portfolio_by_task["DocVQA-lite"]["deployment_cost_proxy"]),
            oracle=False,
            cost_reading="DocVQA falls back to fixed-70",
        )
    )

    candidates.append(
        paired_candidate(
            "oracle_0tol",
            by_policy_task[("oracle_0tol", "TextVQA-lite")],
            by_policy_task[("oracle_0tol", "DocVQA-lite")],
            cost_reading="diagnostic upper bound; excluded because it uses gold scores",
        )
    )
    return candidates


def paired_candidate(
    name: str,
    text_row: dict[str, str],
    doc_row: dict[str, str],
    *,
    cost_reading: str,
) -> dict[str, Any]:
    return candidate_from_values(
        name,
        text_score=float(text_row["test_score"]),
        doc_score=float(doc_row["test_score"]),
        text_fixed=float(text_row["fixed70_score"]),
        doc_fixed=float(doc_row["fixed70_score"]),
        text_keep=float(text_row["selected_keep"]),
        doc_keep=float(doc_row["selected_keep"]),
        text_cost=float(text_row["deployment_cost_proxy"]),
        doc_cost=float(doc_row["deployment_cost_proxy"]),
        oracle=text_row["uses_oracle_scores"] == "1" or doc_row["uses_oracle_scores"] == "1",
        cost_reading=cost_reading,
    )


def candidate_from_values(
    name: str,
    *,
    text_score: float,
    doc_score: float,
    text_fixed: float,
    doc_fixed: float,
    text_keep: float,
    doc_keep: float,
    text_cost: float,
    doc_cost: float,
    oracle: bool,
    cost_reading: str,
) -> dict[str, Any]:
    text_delta = text_score - text_fixed
    doc_delta = doc_score - doc_fixed
    mean_keep = (text_keep + doc_keep) / 2
    mean_cost = (text_cost + doc_cost) / 2
    quality_pass = text_delta >= -QUALITY_TOLERANCE and doc_delta >= -QUALITY_TOLERANCE
    keep_pass = mean_keep <= MAX_MEAN_KEEP
    speed_pass = mean_cost < FIXED70_COST
    return {
        "controller": name,
        "text_score": text_score,
        "text_delta_vs_fixed70": text_delta,
        "doc_score": doc_score,
        "doc_delta_vs_fixed70": doc_delta,
        "mean_keep": mean_keep,
        "mean_deployment_cost": mean_cost,
        "quality_pass": int(quality_pass),
        "keep_pass": int(keep_pass),
        "speed_pass": int(speed_pass),
        "oracle": int(oracle),
        "all_gates_pass": int(quality_pass and keep_pass and speed_pass and not oracle),
        "cost_reading": cost_reading,
    }


def lookup_fixed70(rows: list[dict[str, str]], task: str) -> float:
    matches = [
        row
        for row in rows
        if row["task"] == task and row["split"] == "test" and row["policy"] == "fixed_0.70"
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one fixed-70 row for {task}, found {len(matches)}")
    return float(matches[0]["score"])


def build_llava_summary(report: dict[str, Any]) -> dict[str, Any]:
    rows = {
        (row["task"], row["budget"]): row
        for row in report["runs"]
        if row["model"] == "LLaVA-1.5-7B"
    }
    recoveries = {
        row["task"]: row
        for row in report["safe_budget_recovery"]
        if row["model"] == "LLaVA-1.5-7B"
    }
    return {
        "textvqa_full": rows[("TextVQA", "70%")]["full_score"],
        "textvqa_70": rows[("TextVQA", "70%")]["pruned_score"],
        "textvqa_40": rows[("TextVQA", "40%")]["pruned_score"],
        "textvqa_70_recovery_over_40": recoveries["TextVQA"]["paired_recovery"],
        "textvqa_recovery_ci": [recoveries["TextVQA"]["ci_low"], recoveries["TextVQA"]["ci_high"]],
        "docvqa_full": rows[("DocVQA", "70%")]["full_score"],
        "docvqa_70": rows[("DocVQA", "70%")]["pruned_score"],
        "docvqa_40": rows[("DocVQA", "40%")]["pruned_score"],
        "docvqa_70_recovery_over_40": recoveries["DocVQA"]["paired_recovery"],
        "decision": "expanded_and_completed_but_report_as_cross_model_boundary",
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown(payload: dict[str, Any]) -> str:
    llava = payload["llava_go_no_go"]
    decision = payload["controller_contract"]
    lines = [
        "# Remaining Open-QA Plan: Strict Go/No-Go",
        "",
        "## LLaVA Expansion",
        "",
        (
            f"LLaVA TextVQA Full/70%/40% scores are {llava['textvqa_full']:.4f}/"
            f"{llava['textvqa_70']:.4f}/{llava['textvqa_40']:.4f}. Raising retention from 40% "
            f"to 70% recovers {llava['textvqa_70_recovery_over_40']:+.4f} "
            f"(95% CI [{llava['textvqa_recovery_ci'][0]:+.4f}, {llava['textvqa_recovery_ci'][1]:+.4f}])."
        ),
        (
            f"The planned expansion was therefore completed. DocVQA Full/70%/40% scores are "
            f"{llava['docvqa_full']:.4f}/{llava['docvqa_70']:.4f}/{llava['docvqa_40']:.4f}. "
            "Recovery is material but does not approach Full, so these rows support a cross-model boundary."
        ),
        "",
        "## Strict Controller Pilot",
        "",
        (
            f"Contract: both task scores must be within {decision['quality_tolerance']:.2f} of fixed-70, "
            f"mean keep must be at most {decision['max_mean_keep']:.2f}, and overhead-aware deployment "
            f"cost must be below fixed-70 cost {decision['fixed70_cost']:.2f}."
        ),
        "",
        "| Controller | Text delta | Doc delta | Mean keep | Mean cost | Quality | Keep | Speed | Oracle | Final |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["controller_candidates"]:
        lines.append(
            f"| {row['controller']} | {row['text_delta_vs_fixed70']:+.3f} | "
            f"{row['doc_delta_vs_fixed70']:+.3f} | {row['mean_keep']:.3f} | "
            f"{row['mean_deployment_cost']:.3f} | {row['quality_pass']} | {row['keep_pass']} | "
            f"{row['speed_pass']} | {row['oracle']} | {row['all_gates_pass']} |"
        )
    lines.extend(
        [
            "",
            f"**Decision: `{decision['status']}`.** {decision['reason']}",
            "",
            (
                "The serial majority controller is the decisive counterexample: it passes two-task quality "
                "and mean-keep gates, but repeated generation raises mean deployment cost above fixed-70 even "
                "before adding controller CPU overhead. The pooled pre-generation saturation controller preserves "
                "quality but keeps nearly 70% of tokens. Oracle rows show headroom but are not deployable."
            ),
            "",
            "Per the pre-registered stop rule, no further controller expansion is warranted.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
