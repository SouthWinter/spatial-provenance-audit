#!/usr/bin/env python3
"""Build validation readouts for the Qwen3 VisionZip native-port smoke tests."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SMOKE_DIR = ROOT / "runs" / "qwen3_visionzip_native_smoke"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-dir", type=Path, default=DEFAULT_SMOKE_DIR)
    parser.add_argument("--direct-dir", type=Path, default=DEFAULT_SMOKE_DIR / "direct_eager_reference_limit2")
    parser.add_argument("--keep1-dir", type=Path, default=DEFAULT_SMOKE_DIR / "visionzip_keep1_limit2")
    args = parser.parse_args()

    args.smoke_dir.mkdir(parents=True, exist_ok=True)
    equivalence_rows = build_equivalence_rows(args.direct_dir, args.keep1_dir)
    keep_rows = build_keep_ratio_rows(args.smoke_dir)
    write_csv(args.smoke_dir / "full_equivalence_summary.csv", equivalence_rows)
    write_csv(args.smoke_dir / "keep_ratio_summary.csv", keep_rows)
    (args.smoke_dir / "qwen3_visionzip_smoke_readout.md").write_text(
        build_report(equivalence_rows, keep_rows),
        encoding="utf-8",
    )
    print(f"Wrote {args.smoke_dir / 'qwen3_visionzip_smoke_readout.md'}")


def build_equivalence_rows(direct_dir: Path, keep1_dir: Path) -> list[dict[str, Any]]:
    direct = read_jsonl_by_id(direct_dir / "probe_scores.jsonl")
    keep1 = read_jsonl_by_id(keep1_dir / "probe_scores.jsonl")
    traces = read_jsonl_by_id(keep1_dir / "prune_traces.jsonl")

    rows: list[dict[str, Any]] = []
    for sample_id in sorted(set(direct) & set(keep1)):
        drow = direct[sample_id]
        vrow = keep1[sample_id]
        trace = traces.get(sample_id, {})
        margin_diff = abs_float_diff(drow.get("margin"), vrow.get("margin"))
        yes_loss_diff = abs_float_diff(drow.get("yes_loss"), vrow.get("yes_loss"))
        no_loss_diff = abs_float_diff(drow.get("no_loss"), vrow.get("no_loss"))
        token_passthrough = (
            bool(trace.get("visionzip_passthrough"))
            and int(trace.get("full_visual_tokens", -1)) == int(trace.get("kept_visual_tokens", -2))
            and float(trace.get("effective_keep_ratio", -1.0)) == 1.0
        )
        row = {
            "sample_id": sample_id,
            "pred_match": str(drow.get("pred_answer") == vrow.get("pred_answer")),
            "correct_match": str(drow.get("correct") == vrow.get("correct")),
            "margin_abs_diff": f"{margin_diff:.9g}",
            "yes_loss_abs_diff": f"{yes_loss_diff:.9g}",
            "no_loss_abs_diff": f"{no_loss_diff:.9g}",
            "full_visual_tokens": trace.get("full_visual_tokens", ""),
            "kept_visual_tokens": trace.get("kept_visual_tokens", ""),
            "effective_keep_ratio": trace.get("effective_keep_ratio", ""),
            "visionzip_passthrough": str(bool(trace.get("visionzip_passthrough"))),
            "selector_impl": trace.get("selector_impl", ""),
            "score_source": trace.get("score_source", ""),
            "pass": str(
                drow.get("pred_answer") == vrow.get("pred_answer")
                and drow.get("correct") == vrow.get("correct")
                and margin_diff <= 1e-6
                and yes_loss_diff <= 1e-6
                and no_loss_diff <= 1e-6
                and token_passthrough
            ),
        }
        rows.append(row)

    rows.append(
        {
            "sample_id": "__overall__",
            "pred_match": str(all(row["pred_match"] == "True" for row in rows)),
            "correct_match": str(all(row["correct_match"] == "True" for row in rows)),
            "margin_abs_diff": f"{max_float(rows, 'margin_abs_diff'):.9g}",
            "yes_loss_abs_diff": f"{max_float(rows, 'yes_loss_abs_diff'):.9g}",
            "no_loss_abs_diff": f"{max_float(rows, 'no_loss_abs_diff'):.9g}",
            "full_visual_tokens": "",
            "kept_visual_tokens": "",
            "effective_keep_ratio": "",
            "visionzip_passthrough": str(all(row["visionzip_passthrough"] == "True" for row in rows)),
            "selector_impl": "visionzip",
            "score_source": "qwen3_vision_attention",
            "pass": str(bool(rows) and all(row["pass"] == "True" for row in rows)),
        }
    )
    return rows


def build_keep_ratio_rows(smoke_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace_path in sorted(smoke_dir.glob("visionzip_*/prune_traces.jsonl")):
        run_dir = trace_path.parent
        for trace in read_jsonl(trace_path):
            full_tokens = int(trace.get("full_visual_tokens", 0))
            kept_tokens = int(trace.get("kept_visual_tokens", 0))
            target_keep_ratio = float(trace.get("target_keep_ratio", 0.0))
            expected_tokens = fixed_keep_count(full_tokens, target_keep_ratio)
            dominant = int(trace.get("visionzip_dominant_tokens", 0))
            contextual = int(trace.get("visionzip_contextual_tokens", 0))
            row = {
                "run": run_dir.name,
                "sample_id": trace.get("sample_id", ""),
                "target_keep_ratio": f"{target_keep_ratio:.6g}",
                "full_visual_tokens": full_tokens,
                "expected_kept_tokens": expected_tokens,
                "kept_visual_tokens": kept_tokens,
                "effective_keep_ratio": f"{float(trace.get('effective_keep_ratio', 0.0)):.9g}",
                "dominant_tokens": dominant,
                "contextual_tokens": contextual,
                "passthrough": str(bool(trace.get("visionzip_passthrough"))),
                "selector_impl": trace.get("selector_impl", ""),
                "score_source": trace.get("score_source", ""),
                "pass": str(
                    kept_tokens == expected_tokens
                    and dominant + contextual == kept_tokens
                    and trace.get("selector_impl") == "visionzip"
                    and trace.get("score_source") == "qwen3_vision_attention"
                ),
            }
            rows.append(row)
    rows.append(
        {
            "run": "__overall__",
            "sample_id": "",
            "target_keep_ratio": "",
            "full_visual_tokens": "",
            "expected_kept_tokens": "",
            "kept_visual_tokens": "",
            "effective_keep_ratio": "",
            "dominant_tokens": "",
            "contextual_tokens": "",
            "passthrough": "",
            "selector_impl": "visionzip",
            "score_source": "qwen3_vision_attention",
            "pass": str(bool(rows) and all(row["pass"] == "True" for row in rows)),
        }
    )
    return rows


def build_report(equivalence_rows: list[dict[str, Any]], keep_rows: list[dict[str, Any]]) -> str:
    equivalence_pass = get_overall_pass(equivalence_rows)
    keep_pass = get_overall_pass(keep_rows)
    lines = [
        "# Qwen3 VisionZip Smoke Readout",
        "",
        "This readout validates smoke-test artifacts for the native Qwen3 VisionZip port. It is not a matched-budget benchmark result.",
        "",
        "## Verdict",
        "",
        f"- keep=1.0 full-prefix equivalence smoke: {'pass' if equivalence_pass else 'fail'}",
        f"- actual-token budget accounting smoke: {'pass' if keep_pass else 'fail'}",
        "",
        "## Keep=1.0 Equivalence",
        "",
        table_md(equivalence_rows, ["sample_id", "pred_match", "margin_abs_diff", "yes_loss_abs_diff", "no_loss_abs_diff", "visionzip_passthrough", "pass"]),
        "",
        "## Budget Accounting",
        "",
        table_md(keep_rows, ["run", "sample_id", "target_keep_ratio", "full_visual_tokens", "expected_kept_tokens", "kept_visual_tokens", "dominant_tokens", "contextual_tokens", "pass"]),
    ]
    return "\n".join(lines) + "\n"


def fixed_keep_count(num_tokens: int, keep_ratio: float) -> int:
    if num_tokens == 0:
        return 0
    return max(1, min(num_tokens, int(math.ceil(num_tokens * keep_ratio))))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["sample_id"]): row for row in read_jsonl(path) if "sample_id" in row}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def abs_float_diff(a: Any, b: Any) -> float:
    try:
        return abs(float(a) - float(b))
    except (TypeError, ValueError):
        return float("inf")


def max_float(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return float("nan")
    return max(float(row[key]) for row in rows)


def get_overall_pass(rows: list[dict[str, Any]]) -> bool:
    return bool(rows) and str(rows[-1].get("pass")) == "True"


def table_md(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "(empty)"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")).replace("|", "\\|") for col in columns) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
