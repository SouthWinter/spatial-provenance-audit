"""Build paper-ready RECAP tables and figures from runs/rice_v5.

This script is intentionally dependency-light. It uses existing run artifacts
when available and falls back to the offline RECAP analysis helpers when a run
only has probe_scores.jsonl.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recap.io import read_jsonl, write_json
from recap.recap_analysis import (
    bootstrap_ci_to_csv_rows,
    coverage_curves_to_csv_rows,
    extract_recap_case_studies,
    low_cost_equivalence_to_csv_rows,
    recap_bootstrap_ci,
    recap_cost_utility,
    recap_coverage_curves,
    recap_low_cost_equivalence,
)


RUNS = (
    ("Qwen3-8B", "VSR", "qwen3_8b_recap_vsr_other_relations"),
    ("Qwen3-8B", "What'sUp", "qwen3_8b_recap_whatsup_controlled_other_relations"),
    ("InternVL3.5-8B", "VSR", "internvl3_5_8b_hf_vsr_other_relations"),
    ("InternVL3.5-8B", "What'sUp", "internvl3_5_8b_hf_whatsup_controlled_other_relations"),
    ("LLaVA-1.5-7B", "VSR", "llava15_7b_hf_vsr_other_relations"),
    ("LLaVA-1.5-7B", "What'sUp", "llava15_7b_hf_whatsup_controlled_other_relations"),
)

GSR_RUNS = (
    ("Qwen3-8B", "qwen3_8b_recap_gsrbench_external_two_object"),
    ("InternVL3.5-8B", "internvl3_5_8b_hf_gsrbench_external_two_object"),
    ("LLaVA-1.5-7B", "llava15_7b_hf_gsrbench_external_two_object"),
)

PROMPT_SC_RUNS = (
    ("Qwen3-8B", "VSR", "qwen3_8b_prompt_sc_vsr_other_relations"),
    ("Qwen3-8B", "What'sUp", "qwen3_8b_prompt_sc_whatsup_controlled_other_relations"),
    ("InternVL3.5-8B", "VSR", "internvl3_5_8b_hf_prompt_sc_vsr_other_relations"),
    ("InternVL3.5-8B", "What'sUp", "internvl3_5_8b_hf_prompt_sc_whatsup_controlled_other_relations"),
    ("LLaVA-1.5-7B", "VSR", "llava15_7b_hf_prompt_sc_vsr_other_relations"),
    ("LLaVA-1.5-7B", "What'sUp", "llava15_7b_hf_prompt_sc_whatsup_controlled_other_relations"),
)

MODELS = ("Qwen3-8B", "InternVL3.5-8B", "LLaVA-1.5-7B")
DATASETS = ("VSR", "What'sUp")

MAIN_RISKS = (
    ("confidence", "Confidence"),
    ("recap_evidence", "RECAP-Full"),
    ("rice_recap_selector", "RECAP-Selector"),
)

ABLATION_RISKS = (
    ("confidence", "Confidence"),
    ("recap_claim_delta", "Claim-Delta"),
    ("recap_text_pair", "TextPair"),
    ("recap_img_pair", "ImgPair"),
    ("recap_anti_delta", "Anti-Delta"),
    ("recap_pair_delta", "CalPair"),
    ("recap_evidence", "RECAP-Full"),
    ("rice_recap_selector", "Selector"),
)

COST_RISKS = (
    ("confidence", "Confidence"),
    ("recap_anti_delta", "Anti-Delta"),
    ("recap_img_pair", "ImgPair"),
    ("recap_pair_delta", "CalPair"),
    ("recap_evidence", "RECAP-Full"),
    ("rice_recap_selector", "Selector"),
)

CURVE_RISKS = (
    ("confidence", "Confidence", "#444444"),
    ("recap_evidence", "Full", "#0072B2"),
    ("rice_recap_selector", "Selector", "#D55E00"),
    ("recap_img_pair", "ImgPair", "#009E73"),
    ("recap_anti_delta", "AntiDelta", "#CC79A7"),
)

METRICS = (
    ("selective_accuracy_80cov", "Acc@80", "max"),
    ("hallucination_fpr_80cov", "H-FPR@80", "min"),
    ("error_auroc", "Err-AUC", "max"),
    ("hallucination_auroc", "Hall-AUC", "max"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="runs/rice_v5")
    parser.add_argument("--output-dir", default="runs/rice_v5/paper_artifacts")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=13)
    parser.add_argument("--skip-bootstrap", action="store_true")
    args = parser.parse_args()

    runs_root = normalize_cli_path(args.runs_root)
    output_dir = normalize_cli_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_run_bundle(
        runs_root,
        output_dir,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        skip_bootstrap=args.skip_bootstrap,
    )
    require_complete_bundle(bundle, runs_root)
    prompt_sc_bundle = load_prompt_sc_bundle(runs_root)
    write_main_results(bundle, output_dir)
    write_ablation_results(bundle, output_dir, prompt_sc_bundle=prompt_sc_bundle)
    write_cost_utility(bundle, output_dir, prompt_sc_bundle=prompt_sc_bundle)
    write_bootstrap_ci(bundle, output_dir)
    write_low_cost_equivalence(bundle, output_dir)
    write_case_index(bundle, output_dir)
    write_curve_artifacts(bundle, output_dir)
    gsr_bundle = load_gsrbench_bundle(runs_root)
    if len(gsr_bundle) == len(GSR_RUNS):
        write_gsrbench_results(gsr_bundle, output_dir)
    elif gsr_bundle:
        missing = [model for model, _ in GSR_RUNS if model not in gsr_bundle]
        print(f"warning: incomplete GSR-Bench runs; table not written. Missing: {missing}")
    write_summary(bundle, output_dir)

    print(f"Wrote RECAP paper artifacts to {output_dir}")


def normalize_cli_path(path_text: str) -> Path:
    """Normalize paths passed from both Windows and POSIX shells."""
    text = str(path_text).replace("\\", "/")
    # Recover the common Linux shell typo caused by using unquoted Windows
    # separators: runs\rice_v5 is received by Python as runsrice_v5.
    if text.startswith("runsrice_v5"):
        suffix = text[len("runsrice_v5") :]
        if suffix and not suffix.startswith("/"):
            suffix = "/" + suffix
        text = "runs/rice_v5" + suffix
    return Path(text)


def require_complete_bundle(bundle: dict[tuple[str, str], dict[str, Any]], runs_root: Path) -> None:
    missing = [(model, dataset, run_name) for model, dataset, run_name in RUNS if (model, dataset) not in bundle]
    if not missing:
        return
    lines = [
        f"Missing {len(missing)} required run directories under {runs_root}:",
        *[f"  - {model} / {dataset}: {runs_root / run_name}" for model, dataset, run_name in missing],
        "",
        "On Linux, use forward slashes, for example:",
        "  python scripts/build_recap_paper_artifacts.py --runs-root runs/rice_v5 --output-dir runs/rice_v5/paper_artifacts",
    ]
    raise SystemExit("\n".join(lines))


def load_run_bundle(
    runs_root: Path,
    output_dir: Path,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    skip_bootstrap: bool,
) -> dict[tuple[str, str], dict[str, Any]]:
    bundle: dict[tuple[str, str], dict[str, Any]] = {}
    for model, dataset, run_name in RUNS:
        run_dir = runs_root / run_name
        if not run_dir.exists():
            print(f"warning: missing run directory: {run_dir}")
            continue
        probe_scores_path = run_dir / "probe_scores.jsonl"
        metrics = load_json(run_dir / "metrics_compact.json")
        ablation = load_or_build_json(
            run_dir / "recap_ablation.json",
            lambda: build_ablation_from_scores(probe_scores_path),
        )
        curves = load_or_build_json(
            run_dir / "recap_curves.json",
            lambda: recap_coverage_curves(read_jsonl(probe_scores_path), risks=[name for name, _, _ in CURVE_RISKS]),
        )
        cost = load_or_build_json(
            run_dir / "recap_cost_utility_safe_reuse.json",
            lambda: recap_cost_utility(read_jsonl(probe_scores_path), risks=[name for name, _ in COST_RISKS]),
        )
        bootstrap = None
        if not skip_bootstrap:
            bootstrap = load_or_build_json(
                run_dir / f"recap_bootstrap_ci_b{bootstrap_samples}_s{bootstrap_seed}.json",
                lambda: recap_bootstrap_ci(
                    read_jsonl(probe_scores_path),
                    risks=[name for name, _ in ABLATION_RISKS],
                    n_bootstrap=bootstrap_samples,
                    seed=bootstrap_seed,
                ),
            )
        low_cost_check = load_or_build_json(
            run_dir / "recap_low_cost_equivalence_safe_reuse.json",
            lambda: recap_low_cost_equivalence(read_jsonl(probe_scores_path), variants=[name for name, _ in COST_RISKS]),
        )
        cases = load_or_build_json(
            run_dir / "recap_cases.json",
            lambda: extract_recap_case_studies(read_jsonl(probe_scores_path), examples=8),
        )
        bundle[(model, dataset)] = {
            "run_dir": str(run_dir),
            "metrics": metrics,
            "ablation": ablation,
            "curves": curves,
            "cost": cost,
            "bootstrap": bootstrap,
            "low_cost_check": low_cost_check,
            "cases": cases,
        }
    return bundle


def load_gsrbench_bundle(runs_root: Path) -> dict[str, dict[str, Any]]:
    """Load optional GSR-Bench runs without making them a main-table requirement."""

    bundle: dict[str, dict[str, Any]] = {}
    prefixes = {
        "Qwen3-8B": "qwen",
        "InternVL3.5-8B": "internvl",
        "LLaVA-1.5-7B": "llava",
    }
    suffix = "_gsrbench_external_two_object"
    for model, expected_name in GSR_RUNS:
        run_dir = runs_root / expected_name
        if not run_dir.exists():
            candidates = sorted(
                path
                for path in runs_root.glob(f"{prefixes[model]}*{suffix}")
                if path.is_dir()
            )
            if len(candidates) == 1:
                run_dir = candidates[0]
            else:
                print(f"warning: missing optional GSR-Bench run: {run_dir}")
                continue
        metrics_path = run_dir / "metrics_compact.json"
        if not metrics_path.exists():
            print(f"warning: missing optional GSR-Bench metrics: {metrics_path}")
            continue
        bundle[model] = load_json(metrics_path)
    return bundle


def load_prompt_sc_bundle(runs_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Load Prompt-SC metrics only when the complete primary matrix exists."""

    bundle: dict[tuple[str, str], dict[str, Any]] = {}
    for model, dataset, run_name in PROMPT_SC_RUNS:
        metrics_path = runs_root / run_name / "metrics_compact.json"
        if metrics_path.exists():
            bundle[(model, dataset)] = load_json(metrics_path)
    if bundle and len(bundle) != len(PROMPT_SC_RUNS):
        missing = [
            f"{model}/{dataset}"
            for model, dataset, _ in PROMPT_SC_RUNS
            if (model, dataset) not in bundle
        ]
        print(f"warning: incomplete Prompt-SC matrix; omitting it from the table. Missing: {missing}")
        return {}
    return bundle


def build_ablation_from_scores(path: Path) -> dict[str, Any]:
    from recap.recap_ablation import ablate_recap_probe_scores

    return ablate_recap_probe_scores(read_jsonl(path), include_by_family=True)


def write_main_results(bundle: dict[tuple[str, str], dict[str, Any]], output_dir: Path) -> None:
    rows = []
    for model in MODELS:
        for risk_name, risk_label in MAIN_RISKS:
            row = {"model": model, "risk": risk_label}
            for dataset in DATASETS:
                run = bundle[(model, dataset)]
                risk_row = run["metrics"]["risks"][risk_name]
                add_metric_values(row, dataset, risk_row)
            rows.append(row)
    write_csv(output_dir / "main_results.csv", rows)
    (output_dir / "main_results_table.tex").write_text(
        side_by_side_table(
            rows,
            caption="Main selective prediction results on VSR and What'sUp.",
            label="tab:main-results",
            first_cols=("model", "risk"),
            first_header="Model & Risk",
            group_key="model",
        ),
        encoding="utf-8",
    )


def write_gsrbench_results(bundle: dict[str, dict[str, Any]], output_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        for risk_name, risk_label in MAIN_RISKS:
            risk_row = bundle[model]["risks"][risk_name]
            row = {"model": model, "risk": risk_label}
            for metric_key, _, _ in METRICS:
                row[metric_key] = risk_row[metric_key]
            rows.append(row)
    write_csv(output_dir / "gsrbench_results.csv", rows)
    (output_dir / "gsrbench_results_table.tex").write_text(
        gsrbench_table(rows),
        encoding="utf-8",
    )


def write_ablation_results(
    bundle: dict[tuple[str, str], dict[str, Any]],
    output_dir: Path,
    *,
    prompt_sc_bundle: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> None:
    rows = []
    for model in MODELS:
        for risk_name, risk_label in ABLATION_RISKS:
            row = {"model": model, "variant": risk_label}
            for dataset in DATASETS:
                run = bundle[(model, dataset)]
                risk_row = run["ablation"]["risks"][risk_name]
                add_metric_values(row, dataset, risk_row)
            rows.append(row)
            if risk_name == "confidence" and prompt_sc_bundle:
                sc_row = {"model": model, "variant": "Prompt-SC"}
                for dataset in DATASETS:
                    risk_row = prompt_sc_bundle[(model, dataset)]["risks"]["prompt_sc"]
                    add_metric_values(sc_row, dataset, risk_row)
                rows.append(sc_row)
    write_csv(output_dir / "ablation_results.csv", rows)
    (output_dir / "ablation_table.tex").write_text(
        side_by_side_table(
            rows,
            caption="Baselines and diagnostic ablations for RECAP. Prompt-SC is the fixed ten-call compute-matched self-consistency baseline; Claim-Delta, TextPair, and ImgPair are probe-derived comparison methods; CalPair and RECAP-Full add calibrated contradiction-aware evidence.",
            label="tab:ablation",
            first_cols=("model", "variant"),
            first_header="Model & Variant",
            group_key="model",
        ),
        encoding="utf-8",
    )


def write_cost_utility(
    bundle: dict[tuple[str, str], dict[str, Any]],
    output_dir: Path,
    *,
    prompt_sc_bundle: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> None:
    rows = []
    for model in MODELS:
        for dataset in DATASETS:
            run = bundle[(model, dataset)]
            for risk_name, risk_label in COST_RISKS:
                risk_row = run["cost"]["risks"][risk_name]
                rows.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "variant": risk_label,
                        "avg_inference_calls": risk_row["avg_inference_calls"],
                        "selective_accuracy": risk_row["selective_accuracy"],
                        "hallucination_fpr": risk_row["hallucination_fpr"],
                        "error_auroc": risk_row["error_auroc"],
                        "hallucination_auroc": risk_row["hallucination_auroc"],
                    }
                )
            if prompt_sc_bundle:
                risk_row = prompt_sc_bundle[(model, dataset)]["risks"]["prompt_sc"]
                rows.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "variant": "Prompt-SC",
                        "avg_inference_calls": 10.0,
                        "selective_accuracy": risk_row["selective_accuracy_80cov"],
                        "hallucination_fpr": risk_row["hallucination_fpr_80cov"],
                        "error_auroc": risk_row["error_auroc"],
                        "hallucination_auroc": risk_row["hallucination_auroc"],
                    }
                )
    write_csv(output_dir / "cost_utility.csv", rows)
    (output_dir / "cost_utility_table.tex").write_text(cost_table(rows), encoding="utf-8")


def write_bootstrap_ci(bundle: dict[tuple[str, str], dict[str, Any]], output_dir: Path) -> None:
    rows = []
    for model in MODELS:
        for dataset in DATASETS:
            run = bundle[(model, dataset)]
            report = run.get("bootstrap")
            if not report:
                continue
            for row in bootstrap_ci_to_csv_rows(report):
                out = {"model": model, "dataset": dataset}
                out.update(row)
                rows.append(out)
    if not rows:
        table_path = output_dir / "bootstrap_ci_table.tex"
        csv_path = output_dir / "bootstrap_ci.csv"
        if table_path.exists() and table_path.stat().st_size > 0:
            print(f"Skipping bootstrap CI rewrite because no bootstrap reports were loaded; preserving {table_path}")
            return
        if csv_path.exists() and csv_path.stat().st_size > 0:
            print(f"Skipping bootstrap CI rewrite because no bootstrap reports were loaded; preserving {csv_path}")
            return
    write_csv(output_dir / "bootstrap_ci.csv", rows)
    (output_dir / "bootstrap_ci_table.tex").write_text(bootstrap_ci_table(rows), encoding="utf-8")


def write_low_cost_equivalence(bundle: dict[tuple[str, str], dict[str, Any]], output_dir: Path) -> None:
    rows = []
    for model in MODELS:
        for dataset in DATASETS:
            run = bundle[(model, dataset)]
            for row in low_cost_equivalence_to_csv_rows(run["low_cost_check"]):
                out = {"model": model, "dataset": dataset}
                out.update(row)
                rows.append(out)
    write_csv(output_dir / "low_cost_equivalence.csv", rows)
    (output_dir / "low_cost_equivalence_table.tex").write_text(low_cost_equivalence_table(rows), encoding="utf-8")


def write_case_index(bundle: dict[tuple[str, str], dict[str, Any]], output_dir: Path) -> None:
    rows = []
    for model in MODELS:
        for dataset in DATASETS:
            run = bundle[(model, dataset)]
            for category, cases in run["cases"].get("categories", {}).items():
                for rank, case in enumerate(cases, start=1):
                    row = {"model": model, "dataset": dataset, "category": category, "rank": rank}
                    for key in (
                        "sample_id",
                        "image",
                        "question",
                        "source_caption",
                        "base_relation",
                        "relation_family",
                        "target",
                        "prediction",
                        "baseline_risk",
                        "method_risk",
                        "recap_best_anti_relation",
                        "recap_pair_margin",
                    ):
                        row[key] = case.get(key, "")
                    rows.append(row)
    write_csv(output_dir / "case_studies_index.csv", rows)


def write_curve_artifacts(bundle: dict[tuple[str, str], dict[str, Any]], output_dir: Path) -> None:
    rows = []
    for model in MODELS:
        for dataset in DATASETS:
            run = bundle[(model, dataset)]
            for row in coverage_curves_to_csv_rows(run["curves"]):
                out = {"model": model, "dataset": dataset}
                out.update(row)
                rows.append(out)
    write_csv(output_dir / "coverage_curves_all.csv", rows)
    draw_curve_grid(bundle, output_dir / "coverage_accuracy.png", metric="selective_accuracy", title="Selective Accuracy vs Coverage")
    draw_curve_grid(bundle, output_dir / "coverage_hallucination_fpr.png", metric="hallucination_fpr", title="Hallucination FPR vs Coverage")


def write_summary(bundle: dict[tuple[str, str], dict[str, Any]], output_dir: Path) -> None:
    summary = {
        "runs": {f"{model}/{dataset}": run["run_dir"] for (model, dataset), run in bundle.items()},
        "artifacts": [
            "main_results.csv",
            "main_results_table.tex",
            "ablation_results.csv",
            "ablation_table.tex",
            "cost_utility.csv",
            "cost_utility_table.tex",
            "bootstrap_ci.csv",
            "bootstrap_ci_table.tex",
            "low_cost_equivalence.csv",
            "low_cost_equivalence_table.tex",
            "coverage_curves_all.csv",
            "coverage_accuracy.png",
            "coverage_hallucination_fpr.png",
            "case_studies_index.csv",
        ],
    }
    write_json(output_dir / "summary.json", summary)


def add_metric_values(row: dict[str, Any], dataset: str, metrics: dict[str, Any]) -> None:
    prefix = dataset_key(dataset)
    for key, _, _ in METRICS:
        row[f"{prefix}_{key}"] = metrics[key]


def side_by_side_table(
    rows: list[dict[str, Any]],
    *,
    caption: str,
    label: str,
    first_cols: tuple[str, str],
    first_header: str,
    group_key: str,
) -> str:
    best = table_best_values(rows)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        rf"\caption{{{caption} Arrows indicate the preferred direction; all entries are percentages.}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        r" & & \multicolumn{4}{c}{VSR} & \multicolumn{4}{c}{What'sUp} \\",
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
        first_header + r" & Acc@80$\uparrow$ & H-FPR@80$\downarrow$ & Err-AUC$\uparrow$ & Hall-AUC$\uparrow$ & Acc@80$\uparrow$ & H-FPR@80$\downarrow$ & Err-AUC$\uparrow$ & Hall-AUC$\uparrow$ \\",
        r"\midrule",
    ]
    previous_group = None
    for row in rows:
        group = row[group_key]
        if previous_group is not None and group != previous_group:
            lines.append(r"\midrule")
        previous_group = group
        cells = [tex_escape(str(row[first_cols[0]])), tex_escape(str(row[first_cols[1]]))]
        for dataset in DATASETS:
            prefix = dataset_key(dataset)
            for metric_key, _, direction in METRICS:
                key = f"{prefix}_{metric_key}"
                is_best = is_best_value(row, best, dataset, group, metric_key, direction)
                cells.append(format_pct(row[key], bold=is_best))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}", ""])
    return "\n".join(lines)


def gsrbench_table(rows: list[dict[str, Any]]) -> str:
    best: dict[tuple[str, str], float] = {}
    for model in sorted({str(row["model"]) for row in rows}):
        model_rows = [row for row in rows if str(row["model"]) == model]
        for metric_key, _, direction in METRICS:
            values = [float(row[metric_key]) for row in model_rows]
            best[(model, metric_key)] = min(values) if direction == "min" else max(values)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{External validation on the COCO-Spatial-Two and GQA-Spatial-Two splits of GSR-Bench, which are outside our controlled What'sUp run. All entries are percentages.}",
        r"\label{tab:gsrbench-results}",
        r"\resizebox{\columnwidth}{!}{",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Model & Risk & Acc@80$\uparrow$ & H-FPR@80$\downarrow$ & Err-AUC$\uparrow$ & Hall-AUC$\uparrow$ \\",
        r"\midrule",
    ]
    previous_model = None
    for row in rows:
        model = str(row["model"])
        if previous_model is not None and model != previous_model:
            lines.append(r"\midrule")
        previous_model = model
        cells = [tex_escape(model), tex_escape(str(row["risk"]))]
        for metric_key, _, _ in METRICS:
            is_best = abs(float(row[metric_key]) - best[(model, metric_key)]) <= 5e-4
            cells.append(format_pct(row[metric_key], bold=is_best))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", ""])
    return "\n".join(lines)


def table_best_values(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], float]:
    out: dict[tuple[str, str, str], float] = {}
    for dataset in DATASETS:
        prefix = dataset_key(dataset)
        for group in sorted({row["model"] for row in rows}):
            group_rows = [row for row in rows if row["model"] == group]
            for metric_key, _, direction in METRICS:
                values = [float(row[f"{prefix}_{metric_key}"]) for row in group_rows]
                out[(dataset, group, metric_key)] = min(values) if direction == "min" else max(values)
    return out


def is_best_value(row: dict[str, Any], best: dict[tuple[str, str, str], float], dataset: str, group: str, metric_key: str, direction: str) -> bool:
    value = float(row[f"{dataset_key(dataset)}_{metric_key}"])
    best_value = best[(dataset, group, metric_key)]
    return abs(value - best_value) <= 5e-4


def cost_table(rows: list[dict[str, Any]]) -> str:
    by_key = {(row["model"], row["dataset"], row["variant"]): row for row in rows}
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Compute--utility frontier for low-cost RECAP variants on VSR and What'sUp. Calls are average yes/no likelihood calls per sample.}",
        r"\label{tab:cost-utility}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{llrrrrrrrrrr}",
        r"\toprule",
        r" & & \multicolumn{5}{c}{VSR} & \multicolumn{5}{c}{What'sUp} \\",
        r"\cmidrule(lr){3-7}\cmidrule(lr){8-12}",
        r"Model & Variant & Calls$\downarrow$ & Acc@80$\uparrow$ & H-FPR@80$\downarrow$ & Err-AUC$\uparrow$ & Hall-AUC$\uparrow$ & Calls$\downarrow$ & Acc@80$\uparrow$ & H-FPR@80$\downarrow$ & Err-AUC$\uparrow$ & Hall-AUC$\uparrow$ \\",
        r"\midrule",
    ]
    previous_model = None
    variants = [label for _, label in COST_RISKS]
    if any(row["variant"] == "Prompt-SC" for row in rows):
        variants.insert(1, "Prompt-SC")
    for model in MODELS:
        if previous_model is not None:
            lines.append(r"\midrule")
        previous_model = model
        for variant_label in variants:
            cells = [tex_escape(model), tex_escape(variant_label)]
            for dataset in DATASETS:
                row = by_key[(model, dataset, variant_label)]
                cells.extend(
                    [
                        f"{float(row['avg_inference_calls']):.1f}",
                        pct(row["selective_accuracy"]),
                        pct(row["hallucination_fpr"]),
                        pct(row["error_auroc"]),
                        pct(row["hallucination_auroc"]),
                    ]
                )
            lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}", ""])
    return "\n".join(lines)


def bootstrap_ci_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    by_key = {(row["model"], row["dataset"], row["risk"], row["metric"]): row for row in rows}
    risk_labels = dict(ABLATION_RISKS)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Main selective prediction results with bootstrap 95\% confidence intervals. Intervals are computed by sample-level resampling of cached probe scores. Acc@80 is selective accuracy at 80\% coverage; H-FPR@80 is hallucination false positive rate at 80\% coverage.}",
        r"\label{tab:main-results}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        r"Model & Dataset & Variant & Acc@80$\uparrow$ & H-FPR@80$\downarrow$ & Err-AUC$\uparrow$ & Hall-AUC$\uparrow$ \\",
        r"\midrule",
    ]
    previous = None
    for model in MODELS:
        for dataset in DATASETS:
            for risk_name, _ in MAIN_RISKS:
                group = (model, dataset)
                if previous is not None and group != previous:
                    lines.append(r"\midrule")
                previous = group
                cells = [tex_escape(model), tex_escape(dataset), tex_escape(risk_labels.get(risk_name, risk_name))]
                for metric_key, _, _ in METRICS:
                    row = by_key.get((model, dataset, risk_name, metric_key), {})
                    cells.append(ci_pct_cell(row))
                lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}", ""])
    return "\n".join(lines)


def low_cost_equivalence_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    summary_rows: list[tuple[str, list[dict[str, Any]]]] = []
    for dataset in DATASETS:
        dataset_rows = [row for row in rows if row.get("dataset") == dataset]
        if dataset_rows:
            summary_rows.append((dataset, dataset_rows))
    summary_rows.append(("Overall", rows))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Low-cost equivalence summary. Filtered probe subsets are recomputed offline and compared against full-cache metrics and sample-level risks.}",
        r"\label{tab:low-cost-equivalence}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Dataset & Checks & Max metric diff & Max risk diff & Equivalent \\",
        r"\midrule",
    ]
    for dataset, group_rows in summary_rows:
        max_metric_diff = max(float(row.get("max_metric_abs_diff", 0.0)) for row in group_rows)
        max_risk_diff = max(float(row.get("max_sample_risk_abs_diff", 0.0)) for row in group_rows)
        equivalent = all(str(row.get("equivalent", "")).lower() == "true" for row in group_rows)
        cells = [
            tex_escape(dataset),
            str(len(group_rows)),
            sci(max_metric_diff),
            sci(max_risk_diff),
            "yes" if equivalent else "no",
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def ci_pct_cell(row: dict[str, Any]) -> str:
    if not row:
        return "--"
    return f"{float(row['estimate']) * 100.0:.1f} [{float(row['ci_low']) * 100.0:.1f}, {float(row['ci_high']) * 100.0:.1f}]"


def sci(value: Any) -> str:
    return f"{float(value):.1e}"


def draw_curve_grid(bundle: dict[tuple[str, str], dict[str, Any]], output_path: Path, *, metric: str, title: str) -> None:
    width, height = 1800, 1120
    margin_left, margin_top = 90, 110
    panel_w, panel_h = 500, 360
    gap_x, gap_y = 55, 95
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    title_font = ImageFont.load_default(size=28)
    draw.text((width // 2, 30), title, fill="#111111", font=title_font, anchor="ma")

    y_min, y_max = metric_range(bundle, metric)
    for r, dataset in enumerate(DATASETS):
        for c, model in enumerate(MODELS):
            x0 = margin_left + c * (panel_w + gap_x)
            y0 = margin_top + r * (panel_h + gap_y)
            draw_panel(
                draw,
                font,
                bundle[(model, dataset)]["curves"],
                x0=x0,
                y0=y0,
                w=panel_w,
                h=panel_h,
                metric=metric,
                y_min=y_min,
                y_max=y_max,
                title=f"{model} / {dataset}",
            )
    draw_legend(draw, font, x=margin_left, y=height - 105)
    image.save(output_path)


def draw_panel(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    curves: dict[str, Any],
    *,
    x0: int,
    y0: int,
    w: int,
    h: int,
    metric: str,
    y_min: float,
    y_max: float,
    title: str,
) -> None:
    plot_left, plot_top = x0 + 55, y0 + 35
    plot_w, plot_h = w - 75, h - 80
    draw.rectangle([plot_left, plot_top, plot_left + plot_w, plot_top + plot_h], outline="#333333", width=1)
    draw.text((x0 + w / 2, y0), title, fill="#111111", font=font, anchor="ma")

    for tick in (0, 25, 50, 75, 100):
        x = plot_left + plot_w * (tick / 100.0)
        draw.line([x, plot_top + plot_h, x, plot_top + plot_h + 5], fill="#333333")
        draw.text((x, plot_top + plot_h + 10), str(tick), fill="#333333", font=font, anchor="ma")
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        val = y_min + frac * (y_max - y_min)
        y = plot_top + plot_h - plot_h * frac
        draw.line([plot_left - 5, y, plot_left, y], fill="#333333")
        draw.text((plot_left - 10, y), f"{val:.1f}", fill="#333333", font=font, anchor="ra")
        if frac not in (0.0, 1.0):
            draw.line([plot_left, y, plot_left + plot_w, y], fill="#eeeeee")

    for risk_name, _, color in CURVE_RISKS:
        points = curves.get("curves", {}).get(risk_name, [])
        xy = []
        for point in points:
            cov = float(point["coverage_pct"])
            val = float(point[metric])
            x = plot_left + plot_w * (cov / 100.0)
            y = plot_top + plot_h - plot_h * ((val - y_min) / (y_max - y_min))
            xy.append((x, y))
        if len(xy) >= 2:
            draw.line(xy, fill=color, width=3)
        for x, y in xy[::4]:
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=color)
    draw.text((plot_left + plot_w / 2, plot_top + plot_h + 35), "Coverage (%)", fill="#333333", font=font, anchor="ma")


def draw_legend(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, *, x: int, y: int) -> None:
    cursor = x
    for _, label, color in CURVE_RISKS:
        draw.line([cursor, y, cursor + 35, y], fill=color, width=4)
        draw.text((cursor + 45, y - 7), label, fill="#111111", font=font)
        cursor += 210


def metric_range(bundle: dict[tuple[str, str], dict[str, Any]], metric: str) -> tuple[float, float]:
    values = []
    for run in bundle.values():
        for risk_name, _, _ in CURVE_RISKS:
            for point in run["curves"].get("curves", {}).get(risk_name, []):
                values.append(float(point[metric]))
    if not values:
        return 0.0, 1.0
    lo = max(0.0, math.floor((min(values) - 0.03) * 10.0) / 10.0)
    hi = min(1.0, math.ceil((max(values) + 0.03) * 10.0) / 10.0)
    if hi - lo < 0.2:
        hi = min(1.0, lo + 0.2)
    return lo, hi


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_or_build_json(path: Path, build: Any) -> dict[str, Any]:
    if path.exists():
        return load_json(path)
    payload = build()
    write_json(path, payload)
    return payload


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def dataset_key(dataset: str) -> str:
    return dataset.lower().replace("'", "").replace(" ", "_")


def pct(value: Any) -> str:
    return f"{float(value) * 100.0:.1f}"


def format_pct(value: Any, *, bold: bool = False) -> str:
    text = pct(value)
    return rf"\textbf{{{text}}}" if bold else text


def tex_escape(text: str) -> str:
    return text.replace("&", r"\&").replace("_", r"\_")


if __name__ == "__main__":
    main()
