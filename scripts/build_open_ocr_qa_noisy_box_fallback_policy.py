#!/usr/bin/env python3
"""Evaluate selective fallback policies under noisy/missing evidence boxes.

This is a cached-score diagnostic. It combines open-QA generation outputs at
30/50/70% retention with bbox-noise ECR rows, then asks whether a policy that
falls back when boxes are missing or low-ECR can recover answer quality without
always using the full visual prefix.
"""

from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
NOISE_DIR = ROOT / "runs" / "problem_optimization_audit" / "open_ocr_qa_bbox_noise_audit"
OUT_DIR = ROOT / "runs" / "problem_optimization_audit" / "open_ocr_qa_noisy_box_fallback"
BUDGET_RUNS = {
    ("TextVQA-lite", 0.30): ROOT / "runs/open_ocr_qa/qwen3_8b_textvqa_lite_target_grid0p30_full500/open_ocr_qa_generation.jsonl",
    ("TextVQA-lite", 0.50): ROOT / "runs/open_ocr_qa/qwen3_8b_textvqa_lite_target_grid0p50_full500/open_ocr_qa_generation.jsonl",
    ("TextVQA-lite", 0.70): ROOT / "runs/open_ocr_qa/qwen3_8b_textvqa_lite_target_grid0p70_full500/open_ocr_qa_generation.jsonl",
    ("DocVQA-lite", 0.30): ROOT / "runs/open_ocr_qa/qwen3_8b_docvqa_lite_target_grid0p30_full500/open_ocr_qa_generation.jsonl",
    ("DocVQA-lite", 0.50): ROOT / "runs/open_ocr_qa/qwen3_8b_docvqa_lite_target_grid0p50_full500/open_ocr_qa_generation.jsonl",
    ("DocVQA-lite", 0.70): ROOT / "runs/open_ocr_qa/qwen3_8b_docvqa_lite_target_grid0p70_full500/open_ocr_qa_generation.jsonl",
}
VARIANTS = (
    "clean",
    "jitter_25pct",
    "drop_20pct",
    "drop_40pct",
    "mixed_light",
    "mixed_heavy",
)
THRESHOLDS = (0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.85, 0.90)
BUDGETS = (0.30, 0.50, 0.70, 1.00)


@dataclass(frozen=True)
class Policy:
    name: str
    family: str
    choose_budget: Callable[[dict[str, Any]], float]
    note: str


def main() -> None:
    scores = load_generation_scores()
    rows = load_variant_rows(scores)
    policies = build_policies()
    summary_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []

    for scope, scope_rows in iter_scopes(rows):
        for policy in [p for p in policies if p.family in {"fixed", "oracle"}]:
            summary_rows.extend(evaluate_policy(scope_rows, policy, scope=scope, selected_by="preset"))
        candidates = [p for p in policies if p.family not in {"fixed", "oracle"}]
        for family in sorted({p.family for p in candidates}):
            fam = [p for p in candidates if p.family == family]
            scored = []
            for policy in fam:
                dev = summarize(scope_rows, policy, split="dev")
                obj = objective(dev)
                if not math.isnan(obj):
                    scored.append((obj, dev, policy))
            if not scored:
                continue
            scored.sort(key=lambda item: item[0], reverse=True)
            best_obj, best_dev, best_policy = scored[0]
            selection_rows.append(
                {
                    "scope": scope,
                    "family": family,
                    "selected_policy": best_policy.name,
                    "dev_objective": fmt(best_obj),
                    "dev_score": fmt(best_dev["score"]),
                    "dev_mean_keep": fmt(best_dev["mean_keep"]),
                    "dev_delta_vs_full": fmt(best_dev["delta_vs_full"]),
                    "dev_missing_box_rate": fmt(best_dev["missing_box_rate"]),
                    "dev_full_fallback_rate": fmt(best_dev["full_fallback_rate"]),
                    "note": best_policy.note,
                }
            )
            summary_rows.extend(evaluate_policy(scope_rows, best_policy, scope=scope, selected_by="dev_best"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "noisy_box_fallback_summary.csv", summary_rows)
    write_csv(OUT_DIR / "noisy_box_fallback_selection.csv", selection_rows)
    write_csv(OUT_DIR / "noisy_box_fallback_key_summary.csv", key_summary_rows(summary_rows))
    (OUT_DIR / "noisy_box_fallback_report.md").write_text(build_report(summary_rows, selection_rows), encoding="utf-8")
    print(f"Wrote noisy-box fallback policy audit to {OUT_DIR}")


def load_generation_scores() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for (task, budget), path in BUDGET_RUNS.items():
        for row in read_jsonl(path):
            sample_id = str(row["sample_id"])
            rec = records.setdefault(
                sample_id,
                {
                    "sample_id": sample_id,
                    "task": task,
                    "raw_question": row.get("raw_question", ""),
                    "full_score": parse_float(row.get("full_score")),
                    "budgets": {},
                },
            )
            rec["budgets"][budget] = parse_float(row.get("pruned_score"))
            if math.isnan(rec["full_score"]):
                rec["full_score"] = parse_float(row.get("full_score"))
    for rec in records.values():
        rec["budgets"][1.00] = rec["full_score"]
    return records


def load_variant_rows(scores: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for variant in VARIANTS:
        path = NOISE_DIR / variant / "bbox_ecr_rows.csv"
        feature_by_sample: dict[str, dict[str, Any]] = {}
        for row in read_csv(path):
            if row.get("budget_keep_ratio") != "0.30":
                continue
            sample_id = str(row.get("sample_id", ""))
            if sample_id not in scores:
                continue
            box_count = parse_float(row.get("box_count"))
            has_box = bool(box_count > 0 and row.get("metric_status") == "scored")
            feature_by_sample[sample_id] = {
                "variant": variant,
                "sample_id": sample_id,
                "task": row.get("task", scores[sample_id]["task"]),
                "detector_has_box": has_box,
                "box_count": box_count if not math.isnan(box_count) else 0.0,
                "ECR_0p30": parse_float(row.get("ECR")) if has_box else math.nan,
                "worst_region_ECR_0p30": parse_float(row.get("worst_region_ECR")) if has_box else math.nan,
                "all_regions_ECR_ge_0p50_0p30": parse_float(row.get("all_regions_ECR_ge_0p50")) if has_box else 0.0,
            }
        for sample_id, feat in feature_by_sample.items():
            rec = scores[sample_id]
            if not all(b in rec["budgets"] and not math.isnan(float(rec["budgets"][b])) for b in BUDGETS):
                continue
            row = dict(feat)
            row.update(
                {
                    "raw_question": rec.get("raw_question", ""),
                    "full_score": rec["full_score"],
                    "budgets": dict(rec["budgets"]),
                    "split": split_for_id(sample_id),
                }
            )
            out.append(row)
    return sorted(out, key=lambda r: (r["variant"], r["task"], r["sample_id"]))


def iter_scopes(rows: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    scopes: list[tuple[str, list[dict[str, Any]]]] = []
    by_variant: dict[str, list[dict[str, Any]]] = {}
    by_variant_task: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_variant.setdefault(row["variant"], []).append(row)
        by_variant_task.setdefault((row["variant"], row["task"]), []).append(row)
    for variant in sorted(by_variant):
        scopes.append((variant, by_variant[variant]))
    for variant, task in sorted(by_variant_task):
        scopes.append((f"{variant}:{task}", by_variant_task[(variant, task)]))
    return scopes


def build_policies() -> list[Policy]:
    policies: list[Policy] = []
    for budget in BUDGETS:
        policies.append(
            Policy(
                name=f"fixed_{budget:.2f}",
                family="fixed",
                choose_budget=lambda _row, b=budget: b,
                note="fixed budget reference",
            )
        )
    policies.append(
        Policy(
            name="oracle_best_budget",
            family="oracle",
            choose_budget=oracle_budget,
            note="uses gold scores; diagnostic upper bound only",
        )
    )
    for fallback_budget in (0.70, 1.00):
        policies.append(
            Policy(
                name=f"missing_box_to_{tag(fallback_budget)}",
                family=f"missing_box_to_{tag(fallback_budget)}",
                choose_budget=lambda row, b=fallback_budget: b if not row["detector_has_box"] else 0.30,
                note="fallback only when detector/evidence boxes are missing",
            )
        )
    for feature in ("ECR_0p30", "worst_region_ECR_0p30", "all_regions_ECR_ge_0p50_0p30"):
        for threshold in THRESHOLDS:
            for fallback_budget in (0.70, 1.00):
                policies.append(
                    Policy(
                        name=f"{feature}_lt_{tag(threshold)}_or_missing_to_{tag(fallback_budget)}",
                        family=f"noisy_box_{feature}_to_{tag(fallback_budget)}",
                        choose_budget=lambda row, f=feature, t=threshold, b=fallback_budget: b
                        if (not row["detector_has_box"] or low_feature(row, f) < t)
                        else 0.30,
                        note="fallback when boxes are missing or noisy-box low-budget evidence coverage is low",
                    )
                )
    return policies


def oracle_budget(row: dict[str, Any]) -> float:
    best_budget = 0.30
    best_score = -1.0
    for budget in BUDGETS:
        score = float(row["budgets"][budget])
        if score > best_score or (score == best_score and budget < best_budget):
            best_score = score
            best_budget = budget
    return best_budget


def low_feature(row: dict[str, Any], feature: str) -> float:
    value = parse_float(row.get(feature))
    if math.isnan(value):
        return -1.0
    return value


def evaluate_policy(rows: list[dict[str, Any]], policy: Policy, *, scope: str, selected_by: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for split in ("dev", "test", "all"):
        summary = summarize(rows, policy, split=split)
        out.append(
            {
                "scope": scope,
                "split": split,
                "policy": policy.name,
                "family": policy.family,
                "selected_by": selected_by,
                "n": summary["n"],
                "score": fmt(summary["score"]),
                "mean_keep": fmt(summary["mean_keep"]),
                "delta_vs_full": fmt(summary["delta_vs_full"]),
                "missing_box_rate": fmt(summary["missing_box_rate"]),
                "fallback_rate_ge_0p70": fmt(summary["fallback_rate_ge_0p70"]),
                "full_fallback_rate": fmt(summary["full_fallback_rate"]),
                "objective": fmt(objective(summary)),
                "note": policy.note,
            }
        )
    return out


def summarize(rows: list[dict[str, Any]], policy: Policy, *, split: str) -> dict[str, Any]:
    subset = [row for row in rows if split == "all" or row["split"] == split]
    scores: list[float] = []
    full_scores: list[float] = []
    keeps: list[float] = []
    missing: list[bool] = []
    for row in subset:
        budget = policy.choose_budget(row)
        scores.append(float(row["budgets"][budget]))
        full_scores.append(float(row["full_score"]))
        keeps.append(budget)
        missing.append(not bool(row["detector_has_box"]))
    return {
        "n": len(subset),
        "score": mean(scores) if scores else math.nan,
        "mean_keep": mean(keeps) if keeps else math.nan,
        "delta_vs_full": mean([s - f for s, f in zip(scores, full_scores)]) if scores else math.nan,
        "missing_box_rate": mean(missing) if missing else math.nan,
        "fallback_rate_ge_0p70": mean([k >= 0.70 for k in keeps]) if keeps else math.nan,
        "full_fallback_rate": mean([k >= 1.0 for k in keeps]) if keeps else math.nan,
    }


def objective(summary: dict[str, Any]) -> float:
    if summary["n"] == 0 or math.isnan(float(summary["score"])):
        return math.nan
    return float(summary["score"]) - 0.10 * float(summary["mean_keep"])


def build_report(summary_rows: list[dict[str, Any]], selection_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Open OCR QA Noisy-Box Fallback Policy Audit",
        "",
        "This diagnostic simulates detector-assisted fallback with cached open-QA generation scores. A noisy-box variant provides the evidence boxes and the 30% ECR feature; the answer score is taken from cached 30/50/70/full-prefix generations. No model inference is run here.",
        "",
        "## Dev-Selected Policies",
        "",
        "| Scope | Family | Selected policy | Dev score | Dev keep | Dev missing boxes | Dev full fallback | Note |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in selection_rows:
        lines.append(
            f"| {row['scope']} | {row['family']} | {row['selected_policy']} | {row['dev_score']} | "
            f"{row['dev_mean_keep']} | {row['dev_missing_box_rate']} | {row['dev_full_fallback_rate']} | {row['note']} |"
        )
    lines.extend(
        [
            "",
            "## Test Summary",
            "",
            "| Scope | Policy | Family | Score | Keep | Delta vs full | Missing boxes | Fallback >=0.70 | Full fallback | Note |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in summary_rows:
        if row["split"] != "test":
            continue
        keep_row = row["family"] in {"fixed", "oracle"} or row["selected_by"] == "dev_best"
        if not keep_row:
            continue
        if row["family"] == "fixed" and row["policy"] not in {"fixed_0.30", "fixed_0.70", "fixed_1.00"}:
            continue
        lines.append(
            f"| {row['scope']} | {row['policy']} | {row['family']} | {row['score']} | {row['mean_keep']} | "
            f"{row['delta_vs_full']} | {row['missing_box_rate']} | {row['fallback_rate_ge_0p70']} | "
            f"{row['full_fallback_rate']} | {row['note']} |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- Missing-box-only fallback tests whether detector recall alone explains low-budget risk.",
            "- Noisy-ECR fallback tests whether evidence coverage remains useful after detector-like box perturbations.",
            "- These rows are still a policy simulation over cached budgets, not an end-to-end detector-in-the-loop MLLM run.",
        ]
    )
    return "\n".join(lines) + "\n"


def key_summary_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted_scopes = {"clean", "jitter_25pct", "drop_40pct", "mixed_light"}
    wanted_fixed = {"fixed_0.30", "fixed_0.70", "fixed_1.00"}
    wanted_families = {
        "oracle",
        "missing_box_to_0p70",
        "noisy_box_ECR_0p30_to_0p70",
        "noisy_box_all_regions_ECR_ge_0p50_0p30_to_0p70",
    }
    out: list[dict[str, Any]] = []
    for row in summary_rows:
        if row["split"] != "test" or row["scope"] not in wanted_scopes:
            continue
        if row["family"] == "fixed" and row["policy"] not in wanted_fixed:
            continue
        if row["family"] != "fixed":
            if row["family"] not in wanted_families:
                continue
            if row["family"] != "oracle" and row["selected_by"] != "dev_best":
                continue
        out.append(row)
    return out


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    import json

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def split_for_id(sample_id: str) -> str:
    digest = hashlib.md5(sample_id.encode("utf-8")).hexdigest()
    return "dev" if int(digest[:8], 16) % 2 == 0 else "test"


def parse_float(value: Any) -> float:
    try:
        if value == "":
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def fmt(value: Any) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(x):
        return ""
    return f"{x:.3f}"


def tag(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


if __name__ == "__main__":
    main()
