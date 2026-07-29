#!/usr/bin/env python3
"""Build matched-budget CoIn quality, coverage, timing, and paired audits."""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "official_baseline_extension" / "coin_textocr_hard_audit"
BOOTSTRAP = 10000
SEED = 20260721


@dataclass(frozen=True)
class RunSpec:
    label: str
    path: str


DEVELOPMENT_RUNS = [
    RunSpec("Full", "runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_direct"),
    RunSpec("Protected (40%)", "runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_embed_protected_topk_0p40"),
    RunSpec("Target (40%)", "runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_target_embed_topk_0p40_targetfix"),
    RunSpec("CoIn (40%)", "runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_coin_0p40"),
    RunSpec("SCOPE (40%)", "runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_scope_0p40"),
    RunSpec("Random (40%)", "runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_random_0p40"),
    RunSpec("Grid (40%)", "runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_grid_0p40"),
    RunSpec("VisionZip (40%)", "runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_visionzip_0p40"),
]
CONFIRMATION_RUNS = [
    RunSpec("Full", "runs/textocr_confirmation/llava15_7b_full"),
    RunSpec("Protected (40%)", "runs/textocr_confirmation/llava15_7b_protected_0p40"),
    RunSpec("Target (40%)", "runs/textocr_confirmation/llava15_7b_target_0p40"),
    RunSpec("CoIn (40%)", "runs/textocr_confirmation/llava15_7b_coin_0p40"),
    RunSpec("SCOPE (40%)", "runs/textocr_confirmation/llava15_7b_scope_0p40"),
    RunSpec("Random (40%)", "runs/textocr_confirmation/llava15_7b_random_0p40"),
    RunSpec("VisionZip (40%)", "runs/textocr_confirmation/llava15_7b_visionzip_0p40"),
]
COMPARATORS = ["Full", "Protected (40%)", "Target (40%)", "SCOPE (40%)", "Random (40%)", "Grid (40%)", "VisionZip (40%)"]


def main() -> None:
    summary, pairs = analyze(DEVELOPMENT_RUNS, COMPARATORS, SEED)
    report = markdown(summary, pairs, "Development")

    confirmation_coin = ROOT / "runs/textocr_confirmation/llava15_7b_coin_0p40/probe_scores.jsonl"
    confirmation_summary: list[dict[str, Any]] = []
    confirmation_pairs: list[dict[str, Any]] = []
    if confirmation_coin.exists():
        confirmation_comparators = [label for label in COMPARATORS if label != "Grid (40%)"]
        confirmation_summary, confirmation_pairs = analyze(
            CONFIRMATION_RUNS, confirmation_comparators, SEED + 100
        )
        report += "\n\n" + markdown(confirmation_summary, confirmation_pairs, "Locked confirmation")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "coin_method_summary.csv", summary)
    write_csv(OUT_DIR / "coin_paired_comparisons.csv", pairs)
    if confirmation_summary:
        write_csv(OUT_DIR / "coin_confirmation_summary.csv", confirmation_summary)
        write_csv(OUT_DIR / "coin_confirmation_paired_comparisons.csv", confirmation_pairs)
    (OUT_DIR / "coin_textocr_hard_audit.md").write_text(report, encoding="utf-8")
    print(f"Wrote CoIn audit to {OUT_DIR.relative_to(ROOT)}")


def analyze(
    specs: list[RunSpec], comparators: list[str], seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    loaded = {spec.label: load_run(ROOT / spec.path) for spec in specs}
    summary = [summarize(label, run) for label, run in loaded.items()]
    pairs = [compare(loaded["CoIn (40%)"], loaded[label], label, seed + i) for i, label in enumerate(comparators)]
    return summary, pairs


def load_run(path: Path) -> dict[str, Any]:
    scores = read_jsonl(path / "probe_scores.jsonl")
    traces = read_jsonl(path / "prune_traces.jsonl") if (path / "prune_traces.jsonl").exists() else []
    keyed = {key(row): row for row in scores}
    trace_keyed = {key(row): row for row in traces}
    if len(keyed) != len(scores):
        raise ValueError(f"Duplicate probe keys in {path}")
    return {"path": str(path.relative_to(ROOT)), "scores": keyed, "traces": trace_keyed}


def summarize(label: str, run: dict[str, Any]) -> dict[str, Any]:
    rows = list(run["scores"].values())
    positive = [row for row in rows if row.get("binary_polarity") == "positive"]
    negative = [row for row in rows if row.get("binary_polarity") == "negative"]
    traces = run["traces"]
    coverage = lambda row: float(traces.get(key(row), {}).get("ecr", row.get("prune_ecr", 1.0)))
    trace_rows = list(traces.values())
    full_tokens = mean([float(row.get("full_visual_tokens", 0.0)) for row in trace_rows]) if trace_rows else 0.0
    kept_tokens = mean([float(row.get("kept_visual_tokens", 0.0)) for row in trace_rows]) if trace_rows else 0.0
    return {
        "method": label,
        "n": len(rows),
        "accuracy": mean([float(bool(row.get("correct"))) for row in rows]),
        "hFPR": mean([float(row.get("pred_answer") == "yes") for row in negative]),
        "positive_accuracy": mean([float(bool(row.get("correct"))) for row in positive]),
        "negative_accuracy": mean([float(bool(row.get("correct"))) for row in negative]),
        "positive_ECR": mean([coverage(row) for row in positive]),
        "negative_source_coverage": mean([coverage(row) for row in negative]),
        "keep_ratio": kept_tokens / full_tokens if full_tokens else 1.0,
        "selector_ms": mean([float(row.get("selector_ms", 0.0)) for row in trace_rows]) if trace_rows else 0.0,
        "prune_overhead_ms": mean([float(row.get("prune_overhead_ms", 0.0)) for row in trace_rows]) if trace_rows else 0.0,
        "forward_ms": mean([float(row.get("forward_ms", 0.0)) for row in trace_rows]) if trace_rows else 0.0,
        "path": run["path"],
    }


def compare(left: dict[str, Any], right: dict[str, Any], right_label: str, seed: int) -> dict[str, Any]:
    keys = sorted(set(left["scores"]) & set(right["scores"]))
    if len(keys) != 1000:
        raise ValueError(f"Expected 1000 shared probes for {right_label}, got {len(keys)}")
    rows = [(left["scores"][item], right["scores"][item]) for item in keys]
    negative = [pair for pair in rows if pair[0].get("binary_polarity") == "negative"]
    acc_diff = [float(bool(a.get("correct"))) - float(bool(b.get("correct"))) for a, b in rows]
    hfpr_diff = [float(a.get("pred_answer") == "yes") - float(b.get("pred_answer") == "yes") for a, b in negative]
    acc_ci = cluster_bootstrap(rows, lambda pair: float(bool(pair[0].get("correct"))) - float(bool(pair[1].get("correct"))), seed)
    hfpr_ci = cluster_bootstrap(negative, lambda pair: float(pair[0].get("pred_answer") == "yes") - float(pair[1].get("pred_answer") == "yes"), seed + 1)
    return {
        "comparison": f"CoIn (40%) - {right_label}",
        "n": len(rows),
        "accuracy_diff": mean(acc_diff),
        "accuracy_ci_low": acc_ci[0],
        "accuracy_ci_high": acc_ci[1],
        "accuracy_mcnemar_p": mcnemar(rows, lambda row: bool(row.get("correct"))),
        "hFPR_diff": mean(hfpr_diff),
        "hFPR_ci_low": hfpr_ci[0],
        "hFPR_ci_high": hfpr_ci[1],
        "hFPR_mcnemar_p": mcnemar(negative, lambda row: row.get("pred_answer") == "yes"),
    }


def cluster_bootstrap(rows, value_fn, seed: int) -> tuple[float, float]:
    by_image: dict[str, list[Any]] = {}
    for pair in rows:
        by_image.setdefault(image_id(pair[0]), []).append(pair)
    groups = list(by_image.values())
    rng = random.Random(seed)
    values = []
    for _ in range(BOOTSTRAP):
        sampled = [groups[rng.randrange(len(groups))] for _ in groups]
        flat = [value_fn(pair) for group in sampled for pair in group]
        values.append(mean(flat))
    values.sort()
    return quantile(values, 0.025), quantile(values, 0.975)


def mcnemar(rows, predicate) -> float:
    left_only = sum(predicate(a) and not predicate(b) for a, b in rows)
    right_only = sum(predicate(b) and not predicate(a) for a, b in rows)
    total = left_only + right_only
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, k) for k in range(min(left_only, right_only) + 1)) / (2**total)
    return min(1.0, 2.0 * tail)


def key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("sample_id", "")), str(row.get("probe", ""))


def image_id(row: dict[str, Any]) -> str:
    return str(row.get("sample_id", "")).split(":hard-", 1)[0]


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def quantile(values: list[float], q: float) -> float:
    index = (len(values) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - index) + values[upper] * (index - lower)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown(summary: list[dict[str, Any]], pairs: list[dict[str, Any]], split: str) -> str:
    lines = [
        f"# CoIn TextOCR-Hard Audit: {split}",
        "",
        "CoIn is a paper-algorithm port of CVPR 2026 Algorithm 1, not an official-code reproduction. The port uses projected LLaVA tokens, mean non-image prompt alignment, and the reported LLaVA-1.5 128-token setting alpha=0.9, beta=0.6 without TextOCR tuning.",
        "",
        "| method | n | keep | accuracy | hFPR | PosECR | NegSRC | selector ms | overhead ms | forward ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['method']} | {row['n']} | {row['keep_ratio']:.3f} | {row['accuracy']:.3f} | "
            f"{row['hFPR']:.3f} | {row['positive_ECR']:.3f} | {row['negative_source_coverage']:.3f} | "
            f"{row['selector_ms']:.1f} | {row['prune_overhead_ms']:.1f} | {row['forward_ms']:.1f} |"
        )
    lines.extend(["", "| comparison | delta accuracy [image-cluster 95% CI] | p | delta hFPR [image-cluster 95% CI] | p |", "|---|---:|---:|---:|---:|"])
    for row in pairs:
        lines.append(
            f"| {row['comparison']} | {row['accuracy_diff']:+.3f} [{row['accuracy_ci_low']:+.3f}, {row['accuracy_ci_high']:+.3f}] | "
            f"{row['accuracy_mcnemar_p']:.4g} | {row['hFPR_diff']:+.3f} [{row['hFPR_ci_low']:+.3f}, {row['hFPR_ci_high']:+.3f}] | "
            f"{row['hFPR_mcnemar_p']:.4g} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
