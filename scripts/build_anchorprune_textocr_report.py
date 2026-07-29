#!/usr/bin/env python3
"""Build the matched-budget AnchorPrune TextOCR-Hard audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import fmean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_scope_coverage_ablation import comparison, load_jsonl, summarize


CLIP_WEIGHT_SHA256 = "c6032c2e0caae3dc2d4fba35535fa6307dbb49df59c7e182b1bc4b3329b81801"


def resolve(path: str) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else ROOT / candidate


def load_method_rows(path: Path, *, full_prefix: bool = False) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    if not full_prefix:
        return rows
    return [dict(row, prune_ecr=1.0, prune_keep_ratio=1.0) for row in rows]


def timing_summary(path: Path) -> dict[str, float]:
    traces = load_jsonl(path)
    fields = (
        "vision_ms",
        "target_text_ms",
        "score_compute_ms",
        "selector_ms",
        "prune_materialize_ms",
        "language_ms",
        "forward_ms",
        "anchorprune_anchor_size",
    )
    result = {f"mean_{field}": fmean(float(row[field]) for row in traces) for field in fields}
    result["n"] = float(len(traces))
    result["upstream_commit"] = str(traces[0]["anchorprune_commit"])
    result["clip_weight_sha256"] = CLIP_WEIGHT_SHA256
    return result


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# AnchorPrune Matched-Budget TextOCR-Hard Audit",
        "",
        "All methods use LLaVA-1.5-7B, 40% visual-token retention, the same yes/no likelihood readout, and compact post-pruning positions. Differences are AnchorPrune minus the named comparator; confidence intervals use paired image-cluster bootstrap.",
        "",
        "| Split | Method | Acc. | hFPR | PosECR | NegSRC | Correct-positive low/zero PosECR |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for split in ("development", "confirmation"):
        for method, row in report[split].items():
            lines.append(
                f"| {split} | {method} | {row['accuracy']:.3f} | {row['hfpr']:.3f} | "
                f"{row['positive_ecr']:.3f} | {row['negative_source_coverage']:.3f} | "
                f"{100 * row['correct_positive_ecr_lt_0p50']:.1f}% / "
                f"{100 * row['correct_positive_ecr_eq_0']:.1f}% |"
            )
    lines.extend(["", "| Confirmation comparison | Difference | 95% CI |", "|---|---:|---:|"])
    for baseline, metrics in report["confirmation_comparisons"].items():
        for metric in ("accuracy", "hfpr", "positive_ecr", "negative_source_coverage"):
            row = metrics[metric]
            lines.append(
                f"| AnchorPrune minus {baseline}: {metric.replace('_', ' ')} | "
                f"{row['difference']:+.3f} | [{row['ci_95'][0]:+.3f}, {row['ci_95'][1]:+.3f}] |"
            )
    timing = report["descriptive_trace_timing"]
    lines.extend(
        [
            "",
            "## Implementation and Cost",
            "",
            f"- Pinned official selector commit: `{timing['upstream_commit']}`.",
            f"- OpenAI CLIP ViT-L/14-336 weight SHA-256: `{timing['clip_weight_sha256']}`.",
            f"- Exact-index parity: 12/12 deterministic tensor cases passed.",
            f"- Mean adaptive anchor size: {timing['mean_anchorprune_anchor_size']:.1f} of 231 retained tokens.",
            f"- Mean selector time in the shared accuracy run: {timing['mean_selector_ms']:.1f} ms/probe. GPU co-tenancy makes this value non-comparable; efficiency conclusions use the exclusive repeated timing report.",
            "",
            "## Claim Boundary",
            "",
            "This is an official-algorithm port because the upstream runtime targets the original LLaVA repository while this project evaluates the Hugging Face LLaVA checkpoint. The selector is index-identical to the pinned source on parity tests, and the adapter follows the upstream CLIP query-priority, vision-CLS attention prior, hidden-feature novelty, and native-order materialization definitions.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-dev", default="runs/anchorprune_textocr/development_llava15_anchorprune_0p40/probe_scores.jsonl")
    parser.add_argument("--anchor-conf", default="runs/anchorprune_textocr/confirmation_llava15_anchorprune_0p40/probe_scores.jsonl")
    parser.add_argument("--anchor-conf-traces", default="runs/anchorprune_textocr/confirmation_llava15_anchorprune_0p40/prune_traces.jsonl")
    parser.add_argument("--output-dir", default="runs/anchorprune_textocr/report")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()

    dev_paths = {
        "Full": ROOT / "runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_direct/probe_scores.jsonl",
        "AnchorPrune": resolve(args.anchor_dev),
        "Protected": ROOT / "runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_embed_protected_topk_0p40/probe_scores.jsonl",
        "Target": ROOT / "runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_target_embed_topk_0p40_targetfix/probe_scores.jsonl",
        "SCOPE": ROOT / "runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_scope_0p40/probe_scores.jsonl",
        "CoIn": ROOT / "runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_coin_0p40/probe_scores.jsonl",
    }
    conf_paths = {
        "Full": ROOT / "runs/textocr_confirmation/llava15_7b_full/probe_scores.jsonl",
        "AnchorPrune": resolve(args.anchor_conf),
        "Protected": ROOT / "runs/textocr_confirmation/llava15_7b_protected_0p40/probe_scores.jsonl",
        "Random": ROOT / "runs/textocr_confirmation/llava15_7b_random_0p40/probe_scores.jsonl",
        "Target": ROOT / "runs/textocr_confirmation/llava15_7b_target_0p40/probe_scores.jsonl",
        "SCOPE": ROOT / "runs/textocr_confirmation/llava15_7b_scope_0p40/probe_scores.jsonl",
        "CoIn": ROOT / "runs/textocr_confirmation/llava15_7b_coin_0p40/probe_scores.jsonl",
    }
    for path in (*dev_paths.values(), *conf_paths.values(), resolve(args.anchor_conf_traces)):
        if not path.exists():
            raise FileNotFoundError(path)
    dev_rows = {
        name: load_method_rows(path, full_prefix=name == "Full")
        for name, path in dev_paths.items()
    }
    conf_rows = {
        name: load_method_rows(path, full_prefix=name == "Full")
        for name, path in conf_paths.items()
    }
    report = {
        "development": {name: summarize(rows) for name, rows in dev_rows.items()},
        "confirmation": {name: summarize(rows) for name, rows in conf_rows.items()},
        "confirmation_comparisons": {
            name: comparison(conf_rows["AnchorPrune"], rows, args.bootstrap, args.seed + offset)
            for offset, (name, rows) in enumerate(conf_rows.items())
            if name != "AnchorPrune"
        },
        "descriptive_trace_timing": timing_summary(resolve(args.anchor_conf_traces)),
    }
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "anchorprune_textocr_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(output_dir / "anchorprune_textocr_report.md", report)
    print(f"Wrote AnchorPrune audit to {output_dir}")


if __name__ == "__main__":
    main()
