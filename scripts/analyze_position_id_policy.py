#!/usr/bin/env python
"""Compare compact and preserved logical position IDs under identical prune masks."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recap.metrics import roc_auc
from scripts.calibrate_yesno_thresholds import best_threshold, evaluate, load_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-score", type=Path, required=True)
    parser.add_argument("--preserve-score", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--name", default="Position-ID policy audit")
    parser.add_argument("--dev-buckets", type=int, default=5)
    args = parser.parse_args()

    compact = load_rows(args.compact_score, dev_buckets=args.dev_buckets)
    preserve = load_rows(args.preserve_score, dev_buckets=args.dev_buckets)
    ensure_aligned(compact, preserve)
    mask_audit = compare_traces(args.compact_score.parent, args.preserve_score.parent)

    compact_dev = [row for row in compact if row["split"] == "dev"]
    compact_test = [row for row in compact if row["split"] == "test"]
    preserve_dev = [row for row in preserve if row["split"] == "dev"]
    preserve_test = [row for row in preserve if row["split"] == "test"]
    compact_threshold = best_threshold(compact_dev)
    preserve_threshold = best_threshold(preserve_dev)

    summary = {
        "name": args.name,
        "n": len(compact),
        "dev_n": len(compact_dev),
        "test_n": len(compact_test),
        "compact_score": str(args.compact_score),
        "preserve_score": str(args.preserve_score),
        "mask_audit": mask_audit,
        "raw_all": {
            "compact": evaluate(compact, threshold=0.0),
            "preserve": evaluate(preserve, threshold=0.0),
        },
        "own_threshold_test": {
            "compact_threshold": compact_threshold,
            "preserve_threshold": preserve_threshold,
            "compact": evaluate(compact_test, threshold=compact_threshold),
            "preserve": evaluate(preserve_test, threshold=preserve_threshold),
        },
        "compact_shared_threshold_test": {
            "threshold": compact_threshold,
            "compact": evaluate(compact_test, threshold=compact_threshold),
            "preserve": evaluate(preserve_test, threshold=compact_threshold),
        },
        "test_auroc": {
            "compact": auroc(compact_test),
            "preserve": auroc(preserve_test),
        },
        "paired_all": paired_summary(compact, preserve, threshold_a=0.0, threshold_b=0.0),
        "paired_test_own_threshold": paired_summary(
            compact_test,
            preserve_test,
            threshold_a=compact_threshold,
            threshold_b=preserve_threshold,
        ),
        "paired_test_compact_shared_threshold": paired_summary(
            compact_test,
            preserve_test,
            threshold_a=compact_threshold,
            threshold_b=compact_threshold,
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "position_id_policy_audit.json"
    md_path = args.out_dir / "position_id_policy_audit.md"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(summary), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


def ensure_aligned(first: list[dict[str, Any]], second: list[dict[str, Any]]) -> None:
    first_ids = [str(row["sample_id"]) for row in first]
    second_ids = [str(row["sample_id"]) for row in second]
    if first_ids != second_ids:
        raise ValueError("Compact and preserve score files do not have identical ordered sample IDs.")


def compare_traces(compact_dir: Path, preserve_dir: Path) -> dict[str, Any]:
    compact_path = compact_dir / "prune_traces.jsonl"
    preserve_path = preserve_dir / "prune_traces.jsonl"
    if not compact_path.exists() or not preserve_path.exists():
        return {"available": False}
    compact = read_jsonl(compact_path)
    preserve = read_jsonl(preserve_path)
    if len(compact) != len(preserve):
        raise ValueError("Compact and preserve trace files have different row counts.")

    id_mismatches = 0
    mask_mismatches = 0
    ecr_mismatches = 0
    for first, second in zip(compact, preserve):
        if str(first.get("sample_id")) != str(second.get("sample_id")):
            id_mismatches += 1
        if first.get("kept_indices") != second.get("kept_indices"):
            mask_mismatches += 1
        if not math.isclose(float(first.get("ecr", 0.0)), float(second.get("ecr", 0.0)), abs_tol=1e-12):
            ecr_mismatches += 1
    return {
        "available": True,
        "rows": len(compact),
        "sample_id_mismatches": id_mismatches,
        "kept_indices_mismatches": mask_mismatches,
        "ecr_mismatches": ecr_mismatches,
    }


def paired_summary(
    first: list[dict[str, Any]],
    second: list[dict[str, Any]],
    *,
    threshold_a: float,
    threshold_b: float,
) -> dict[str, float | int]:
    ensure_aligned(first, second)
    deltas = [float(b["margin"]) - float(a["margin"]) for a, b in zip(first, second)]
    flips = sum(
        (float(a["margin"]) >= threshold_a) != (float(b["margin"]) >= threshold_b)
        for a, b in zip(first, second)
    )
    return {
        "prediction_flips": flips,
        "mean_margin_delta_preserve_minus_compact": mean(deltas),
        "mean_abs_margin_delta": mean([abs(value) for value in deltas]),
        "max_abs_margin_delta": max((abs(value) for value in deltas), default=0.0),
    }


def auroc(rows: list[dict[str, Any]]) -> float:
    return roc_auc(
        [1.0 if row["target_yes"] else 0.0 for row in rows],
        [float(row["margin"]) for row in rows],
    )


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def render_markdown(summary: dict[str, Any]) -> str:
    raw = summary["raw_all"]
    own = summary["own_threshold_test"]
    shared = summary["compact_shared_threshold_test"]
    auc = summary["test_auroc"]
    paired = summary["paired_test_compact_shared_threshold"]
    mask = summary["mask_audit"]
    return "\n".join(
        [
            f"# {summary['name']}",
            "",
            "Compact uses the model's ordinary physically shortened-prefix path, which assigns consecutive logical positions; preserve explicitly keeps the pre-pruning logical position IDs. Both policies use contiguous physical cache slots.",
            "",
            "| Policy | Raw all Acc. | Raw all hFPR | Own-dev test Acc. | Own-dev test hFPR | Compact-shared test Acc. | Compact-shared test hFPR | Test AUROC |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            row("Compact", raw["compact"], own["compact"], shared["compact"], auc["compact"]),
            row("Preserve", raw["preserve"], own["preserve"], shared["preserve"], auc["preserve"]),
            "",
            f"Development thresholds: compact={own['compact_threshold']:.6f}, preserve={own['preserve_threshold']:.6f}. Development/test sizes: {summary['dev_n']}/{summary['test_n']}.",
            "",
            f"At the compact-derived shared threshold, {paired['prediction_flips']} of {summary['test_n']} held-out predictions flip; mean absolute margin change is {paired['mean_abs_margin_delta']:.4f} and the maximum is {paired['max_abs_margin_delta']:.4f}.",
            "",
            f"Mask invariance: kept-index mismatches={mask.get('kept_indices_mismatches', 'NA')}, ECR mismatches={mask.get('ecr_mismatches', 'NA')}, sample-ID mismatches={mask.get('sample_id_mismatches', 'NA')}.",
            "",
            "Interpretation: position handling is an inference-policy variable, not a selector or coverage change. Answer metrics must therefore be reported with the position policy fixed and named.",
            "",
        ]
    )


def row(name: str, raw: dict[str, float], own: dict[str, float], shared: dict[str, float], auc: float) -> str:
    return (
        f"| {name} | {raw['acc']:.3f} | {raw['hFPR']:.3f} | {own['acc']:.3f} | {own['hFPR']:.3f} | "
        f"{shared['acc']:.3f} | {shared['hFPR']:.3f} | {auc:.3f} |"
    )


if __name__ == "__main__":
    main()
