#!/usr/bin/env python3
"""Join original and edited TextOCR text-replacement counterfactual scores."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-scores", required=True, help="Original TextOCR-Hard probe_scores.jsonl.")
    parser.add_argument("--edited-scores", required=True, help="Text-replacement probe_scores.jsonl.")
    parser.add_argument("--manifest", default="", help="Optional counterfactual edit manifest JSONL.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default="text_replacement_joined")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_scores = {str(row.get("sample_id", "")): row for row in read_jsonl(Path(args.base_scores))}
    edited_pairs = pair_edited_scores(read_jsonl(Path(args.edited_scores)))
    manifest = {
        str(row.get("source_sample_id", "")): row
        for row in read_jsonl(Path(args.manifest))
    } if args.manifest else {}

    rows: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for source_id, pair in sorted(edited_pairs.items()):
        edited_source = pair.get("source_absent")
        edited_replacement = pair.get("replacement_present")
        if not edited_source or not edited_replacement:
            missing.append({"source_sample_id": source_id, "reason": "missing_edited_pair"})
            continue
        edit_manifest = manifest.get(source_id, {})
        negative_id = str(
            edited_source.get("negative_sample_id")
            or edited_replacement.get("negative_sample_id")
            or edit_manifest.get("negative_sample_id")
            or ""
        )
        base_source = base_scores.get(source_id)
        base_replacement = base_scores.get(negative_id)
        if not base_source or not base_replacement:
            missing.append(
                {
                    "source_sample_id": source_id,
                    "negative_sample_id": negative_id,
                    "reason": "missing_base_score",
                }
            )
            continue
        rows.append(build_row(base_source, base_replacement, edited_source, edited_replacement, edit_manifest))

    summary_rows = build_summary(rows)
    write_csv(output_dir / f"{args.name}_pairs.csv", rows)
    write_csv(output_dir / f"{args.name}_summary.csv", summary_rows)
    write_csv(output_dir / f"{args.name}_missing.csv", missing)
    (output_dir / f"{args.name}_summary.json").write_text(
        json.dumps(summary_rows[0] if summary_rows else {}, indent=2),
        encoding="utf-8",
    )
    (output_dir / f"{args.name}_report.md").write_text(build_markdown(summary_rows, rows, missing), encoding="utf-8")
    print(f"Wrote joined text-replacement report to {output_dir}")


def pair_edited_scores(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    pairs: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        source_id = str(row.get("source_sample_id") or str(row.get("sample_id", "")).split(":replace-", 1)[0])
        probe = str(row.get("probe", ""))
        bucket = pairs.setdefault(source_id, {})
        if probe == "counterfactual_source_absent":
            bucket["source_absent"] = row
        elif probe == "counterfactual_replacement_present":
            bucket["replacement_present"] = row
    return pairs


def build_row(
    base_source: dict[str, Any],
    base_replacement: dict[str, Any],
    edited_source: dict[str, Any],
    edited_replacement: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    base_source_yes = pred(base_source) == "yes"
    base_replacement_no = pred(base_replacement) == "no"
    edited_source_no = pred(edited_source) == "no"
    edited_replacement_yes = pred(edited_replacement) == "yes"
    original_pair_correct = base_source_yes and base_replacement_no
    edited_pair_correct = edited_source_no and edited_replacement_yes
    full_switch = original_pair_correct and edited_pair_correct
    source_yes_drop = margin(base_source) - margin(edited_source)
    replacement_yes_gain = margin(edited_replacement) - margin(base_replacement)
    return {
        "image_id": edited_source.get("image_id", ""),
        "source_sample_id": base_source.get("sample_id", ""),
        "negative_sample_id": base_replacement.get("sample_id", ""),
        "source_text": edited_source.get("source_text", edited_source.get("target_text", "")),
        "replacement_text": edited_source.get("replacement_text", edited_replacement.get("target_text", "")),
        "box_width_px": manifest.get("box_width_px", ""),
        "box_height_px": manifest.get("box_height_px", ""),
        "font_size": manifest.get("font_size", ""),
        "base_source_pred": pred(base_source),
        "base_replacement_pred": pred(base_replacement),
        "edited_source_pred": pred(edited_source),
        "edited_replacement_pred": pred(edited_replacement),
        "base_source_margin": f"{margin(base_source):.6f}",
        "base_replacement_margin": f"{margin(base_replacement):.6f}",
        "edited_source_margin": f"{margin(edited_source):.6f}",
        "edited_replacement_margin": f"{margin(edited_replacement):.6f}",
        "source_yes_support_drop": f"{source_yes_drop:.6f}",
        "replacement_yes_support_gain": f"{replacement_yes_gain:.6f}",
        "original_pair_correct": original_pair_correct,
        "edited_pair_correct": edited_pair_correct,
        "full_semantic_switch": full_switch,
        "source_absence_switch": base_source_yes and edited_source_no,
        "replacement_presence_switch": base_replacement_no and edited_replacement_yes,
        "original_image": base_source.get("image", ""),
        "edited_image": edited_source.get("image", ""),
    }


def build_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    original_correct = [bool(row["original_pair_correct"]) for row in rows]
    edited_correct = [bool(row["edited_pair_correct"]) for row in rows]
    full_switch = [bool(row["full_semantic_switch"]) for row in rows]
    source_switch = [bool(row["source_absence_switch"]) for row in rows]
    replacement_switch = [bool(row["replacement_presence_switch"]) for row in rows]
    conditioned = [
        bool(row["edited_pair_correct"])
        for row in rows
        if bool(row["original_pair_correct"])
    ]
    return [
        {
            "n_pairs": len(rows),
            "original_pair_correct_rate": rate(original_correct),
            "edited_pair_correct_rate": rate(edited_correct),
            "full_four_way_semantic_switch_rate": rate(full_switch),
            "edited_pair_correct_given_original_correct": rate(conditioned),
            "source_absence_switch_rate": rate(source_switch),
            "replacement_presence_switch_rate": rate(replacement_switch),
            "mean_source_yes_support_drop": mean_float(rows, "source_yes_support_drop"),
            "mean_replacement_yes_support_gain": mean_float(rows, "replacement_yes_support_gain"),
            "mean_box_width_px": mean_float(rows, "box_width_px"),
            "mean_box_height_px": mean_float(rows, "box_height_px"),
            "mean_font_size": mean_float(rows, "font_size"),
        }
    ]


def build_markdown(summary_rows: list[dict[str, Any]], rows: list[dict[str, Any]], missing: list[dict[str, str]]) -> str:
    summary = summary_rows[0] if summary_rows else {}
    lines = [
        "# TextOCR Text-Replacement Before/After Counterfactual",
        "",
        "This report joins original TextOCR-Hard scores with edited-image scores. A full semantic switch requires four predictions to hold: original source=yes, original near-miss=no, edited source=no, and edited near-miss=yes.",
        "",
        "## Summary",
        "",
        table_md(summary_rows, list(summary.keys())) if summary else "(empty)",
        "",
        "## Interpretation Boundary",
        "",
        "Safe claim: this is a semantic text-replacement diagnostic that tests whether Qwen's yes/no decision follows a local word edit rather than only a masked region. Unsafe claim: it is not a photorealistic renderer and the current switch rate is not a universal causal proof.",
        "",
        "## Example Rows",
        "",
        table_md(
            rows[:12],
            [
                "image_id",
                "source_text",
                "replacement_text",
                "base_source_pred",
                "base_replacement_pred",
                "edited_source_pred",
                "edited_replacement_pred",
                "source_yes_support_drop",
                "replacement_yes_support_gain",
                "full_semantic_switch",
            ],
        ),
    ]
    if missing:
        lines.extend(
            [
                "",
                "## Missing Joins",
                "",
                table_md(missing[:20], ["source_sample_id", "negative_sample_id", "reason"]),
            ]
        )
    return "\n".join(lines) + "\n"


def pred(row: dict[str, Any]) -> str:
    return str(row.get("pred_answer") or row.get("direct_pred") or "").strip().lower()


def margin(row: dict[str, Any]) -> float:
    try:
        return float(row.get("margin", 0.0))
    except Exception:
        return 0.0


def rate(values: list[bool]) -> str:
    if not values:
        return "0.000000"
    return f"{sum(values) / len(values):.6f}"


def mean_float(rows: list[dict[str, Any]], key: str) -> str:
    vals: list[float] = []
    for row in rows:
        value = row.get(key, "")
        if value == "":
            continue
        try:
            vals.append(float(value))
        except Exception:
            continue
    return f"{mean(vals):.6f}" if vals else "0.000000"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


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
