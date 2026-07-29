#!/usr/bin/env python3
"""Audit original/replacement/sham/erase behavior for one model backbone."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


PROBE_KEYS = {
    "counterfactual_source_absent": "replacement_source",
    "counterfactual_replacement_present": "replacement_target",
    "counterfactual_sham_source": "sham_source",
    "counterfactual_sham_replacement": "sham_target",
    "counterfactual_erase_source": "erase_source",
    "counterfactual_erase_replacement": "erase_target",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-scores", required=True)
    parser.add_argument("--control-scores", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    base = {str(row.get("sample_id", "")): row for row in read_jsonl(Path(args.base_scores))}
    manifest = {str(row["image_id"]): row for row in read_csv(Path(args.manifest))}
    controls = group_controls(read_jsonl(Path(args.control_scores)))
    rows = []
    missing = []
    for image_id, item in manifest.items():
        source_id = str(item["source_sample_id"])
        target_id = str(item["negative_sample_id"])
        control = controls.get(image_id, {})
        if source_id not in base or target_id not in base or set(control) != set(PROBE_KEYS.values()):
            missing.append({"image_id": image_id, "reason": "missing base or six-way control score"})
            continue
        rows.append(build_row(args.model, base[source_id], base[target_id], control, args.threshold))

    if missing:
        raise ValueError(f"Incomplete model audit: {len(missing)} missing rows; first={missing[0]}")
    summary = summarize(args.model, rows, args.threshold)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "text_replacement_control_rows.csv", rows)
    write_csv(output_dir / "text_replacement_control_summary.csv", [summary])
    (output_dir / "text_replacement_control_report.md").write_text(
        render_report(summary), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def group_controls(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = PROBE_KEYS.get(str(row.get("probe", "")))
        if key:
            grouped.setdefault(str(row.get("image_id", "")), {})[key] = row
    return grouped


def build_row(
    model: str,
    base_source: dict[str, Any],
    base_target: dict[str, Any],
    control: dict[str, dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    predictions = {
        "original_source": predict(base_source, threshold),
        "original_target": predict(base_target, threshold),
        **{key: predict(row, threshold) for key, row in control.items()},
    }
    original_ok = predictions["original_source"] == "yes" and predictions["original_target"] == "no"
    replacement_ok = predictions["replacement_source"] == "no" and predictions["replacement_target"] == "yes"
    sham_ok = predictions["sham_source"] == "yes" and predictions["sham_target"] == "no"
    erase_ok = predictions["erase_source"] == "no" and predictions["erase_target"] == "no"
    return {
        "model": model,
        "image_id": control["replacement_source"].get("image_id", ""),
        "source_text": control["replacement_source"].get("source_text", ""),
        "replacement_text": control["replacement_source"].get("replacement_text", ""),
        **{f"{key}_pred": value for key, value in predictions.items()},
        "original_pair_correct": int(original_ok),
        "replacement_pair_correct": int(replacement_ok),
        "sham_pair_correct": int(sham_ok),
        "erase_pair_correct": int(erase_ok),
        "full_semantic_switch": int(original_ok and replacement_ok),
        "all_four_controls_correct": int(original_ok and replacement_ok and sham_ok and erase_ok),
        "source_yes_support_drop": f"{margin(base_source) - margin(control['replacement_source']):.6f}",
        "replacement_yes_support_gain": f"{margin(control['replacement_target']) - margin(base_target):.6f}",
        "sham_source_margin_drift": f"{margin(control['sham_source']) - margin(base_source):.6f}",
        "sham_target_margin_drift": f"{margin(control['sham_target']) - margin(base_target):.6f}",
    }


def summarize(model: str, rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    metric_keys = [
        "original_pair_correct",
        "replacement_pair_correct",
        "sham_pair_correct",
        "erase_pair_correct",
        "full_semantic_switch",
        "all_four_controls_correct",
    ]
    summary: dict[str, Any] = {
        "model": model,
        "threshold": f"{threshold:.6f}",
        "n": len(rows),
    }
    for key in metric_keys:
        summary[f"{key}_rate"] = f"{mean(float(row[key]) for row in rows):.6f}"
    for key in (
        "source_yes_support_drop",
        "replacement_yes_support_gain",
        "sham_source_margin_drift",
        "sham_target_margin_drift",
    ):
        summary[f"mean_{key}"] = f"{mean(float(row[key]) for row in rows):.6f}"
    return summary


def predict(row: dict[str, Any], threshold: float) -> str:
    return "yes" if margin(row) >= threshold else "no"


def margin(row: dict[str, Any]) -> float:
    return float(row.get("margin", 0.0) or 0.0)


def render_report(summary: dict[str, Any]) -> str:
    columns = list(summary)
    return "\n".join(
        [
            "# Cross-Backbone Text-Replacement Control Audit",
            "",
            "The audit requires the original pair, semantic replacement pair, sham re-render pair, and erase pair to behave consistently under a threshold fixed before this counterfactual evaluation.",
            "",
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
            "| " + " | ".join(str(summary[column]) for column in columns) + " |",
            "",
            "These rows are automatically screened candidates. Claim promotion additionally requires the preregistered human edit-validity QC.",
            "",
        ]
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
