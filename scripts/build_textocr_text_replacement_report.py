#!/usr/bin/env python3
"""Summarize TextOCR text-replacement counterfactual yes/no results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True, help="probe_scores.jsonl from a counterfactual run.")
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    rows = read_jsonl(Path(args.scores))
    by_source: dict[str, dict[str, Any]] = {}
    for row in rows:
        source_id = str(row.get("source_sample_id") or str(row.get("sample_id", "")).split(":replace-", 1)[0])
        bucket = by_source.setdefault(source_id, {"rows": []})
        bucket["rows"].append(row)

    pairs = []
    for source_id, bucket in sorted(by_source.items()):
        source_absent = None
        replacement_present = None
        for row in bucket["rows"]:
            probe = str(row.get("probe", ""))
            if probe == "counterfactual_source_absent":
                source_absent = row
            elif probe == "counterfactual_replacement_present":
                replacement_present = row
        if source_absent and replacement_present:
            pairs.append((source_absent, replacement_present))

    source_no = sum(pred(row) == "no" for row, _ in pairs)
    replacement_yes = sum(pred(row) == "yes" for _, row in pairs)
    both_switched = sum(pred(src) == "no" and pred(rep) == "yes" for src, rep in pairs)
    source_margin = [margin(row) for row, _ in pairs]
    replacement_margin = [margin(row) for _, row in pairs]
    summary = {
        "num_pairs": len(pairs),
        "num_scores": len(rows),
        "source_absent_no_rate": source_no / len(pairs) if pairs else 0.0,
        "replacement_present_yes_rate": replacement_yes / len(pairs) if pairs else 0.0,
        "semantic_switch_rate": both_switched / len(pairs) if pairs else 0.0,
        "mean_source_margin": sum(source_margin) / len(source_margin) if source_margin else 0.0,
        "mean_replacement_margin": sum(replacement_margin) / len(replacement_margin) if replacement_margin else 0.0,
    }
    text = render_markdown(summary, pairs[:10])
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text(text, encoding="utf-8")
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(text)


def pred(row: dict[str, Any]) -> str:
    return str(row.get("pred_answer") or row.get("direct_pred") or "").strip().lower()


def margin(row: dict[str, Any]) -> float:
    try:
        return float(row.get("margin", 0.0))
    except Exception:
        return 0.0


def render_markdown(summary: dict[str, Any], examples: list[tuple[dict[str, Any], dict[str, Any]]]) -> str:
    lines = [
        "# TextOCR Text-Replacement Counterfactual Report",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Pairs | {summary['num_pairs']} |",
        f"| Source-absent no rate | {summary['source_absent_no_rate']:.3f} |",
        f"| Replacement-present yes rate | {summary['replacement_present_yes_rate']:.3f} |",
        f"| Full semantic switch rate | {summary['semantic_switch_rate']:.3f} |",
        f"| Mean source margin | {summary['mean_source_margin']:.3f} |",
        f"| Mean replacement margin | {summary['mean_replacement_margin']:.3f} |",
        "",
        "## Example Predictions",
        "",
        "| Image | Source | Replacement | Source pred | Replacement pred |",
        "| --- | --- | --- | --- | --- |",
    ]
    for src, rep in examples:
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                src.get("image_id", ""),
                src.get("source_text", src.get("target_text", "")),
                src.get("replacement_text", rep.get("target_text", "")),
                pred(src),
                pred(rep),
            )
        )
    return "\n".join(lines) + "\n"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    main()
