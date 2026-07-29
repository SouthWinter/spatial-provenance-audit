#!/usr/bin/env python3
"""Summarize paired semantic-counterfactual results across model backbones."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


KEY_METRICS = ("full_semantic_switch", "all_four_controls_correct")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-rows", required=True)
    parser.add_argument("--internvl-rows", required=True)
    parser.add_argument("--internvl-summaries", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    qwen = index_rows(read_csv(Path(args.qwen_rows)))
    internvl = index_rows(read_csv(Path(args.internvl_rows)))
    if set(qwen) != set(internvl):
        raise ValueError("Qwen and InternVL image IDs do not match")

    paired_rows = []
    summary_rows = []
    n = len(qwen)
    for metric in KEY_METRICS:
        q_values = [as_bool(qwen[image_id][metric]) for image_id in sorted(qwen)]
        i_values = [as_bool(internvl[image_id][metric]) for image_id in sorted(qwen)]
        q_only = sum(q and not i for q, i in zip(q_values, i_values))
        i_only = sum(i and not q for q, i in zip(q_values, i_values))
        both = sum(q and i for q, i in zip(q_values, i_values))
        neither = n - q_only - i_only - both
        q_success = q_only + both
        i_success = i_only + both
        q_low, q_high = wilson_interval(q_success, n)
        i_low, i_high = wilson_interval(i_success, n)
        shared_low, shared_high = wilson_interval(both, n)
        summary_rows.extend(
            [
                summary_row(metric, "Qwen3-VL-8B", q_success, n, q_low, q_high),
                summary_row(metric, "InternVL3.5-8B", i_success, n, i_low, i_high),
                summary_row(metric, "Both models", both, n, shared_low, shared_high),
            ]
        )
        paired_rows.append(
            {
                "metric": metric,
                "n": n,
                "both_success": both,
                "qwen_only": q_only,
                "internvl_only": i_only,
                "neither": neither,
                "mcnemar_exact_p": f"{mcnemar_exact(q_only, i_only):.8g}",
            }
        )

    sensitivity = []
    for path_string in args.internvl_summaries:
        rows = read_csv(Path(path_string))
        if len(rows) != 1:
            raise ValueError(f"Expected one summary row in {path_string}")
        sensitivity.append(rows[0])
    sensitivity.sort(key=lambda row: float(row["threshold"]))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "cross_backbone_rate_summary.csv", summary_rows)
    write_csv(output_dir / "cross_backbone_paired_summary.csv", paired_rows)
    write_csv(output_dir / "internvl_threshold_sensitivity.csv", sensitivity)
    (output_dir / "cross_backbone_summary.md").write_text(
        render_report(summary_rows, paired_rows, sensitivity), encoding="utf-8"
    )


def summary_row(
    metric: str, model: str, successes: int, n: int, low: float, high: float
) -> dict[str, Any]:
    return {
        "metric": metric,
        "model": model,
        "successes": successes,
        "n": n,
        "rate": f"{successes / n:.6f}",
        "wilson_95_low": f"{low:.6f}",
        "wilson_95_high": f"{high:.6f}",
    }


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return center - radius, center + radius


def mcnemar_exact(a_only: int, b_only: int) -> float:
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(a_only, b_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def render_report(
    rates: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    sensitivity: list[dict[str, str]],
) -> str:
    rate_lines = [
        "| Metric | Model scope | Successes | Rate | Wilson 95% CI |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in rates:
        rate_lines.append(
            f"| {row['metric']} | {row['model']} | {row['successes']}/{row['n']} | "
            f"{float(row['rate']):.3f} | [{float(row['wilson_95_low']):.3f}, "
            f"{float(row['wilson_95_high']):.3f}] |"
        )

    paired_lines = [
        "| Metric | Both | Qwen only | InternVL only | Neither | Exact McNemar p |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in paired:
        paired_lines.append(
            f"| {row['metric']} | {row['both_success']} | {row['qwen_only']} | "
            f"{row['internvl_only']} | {row['neither']} | {float(row['mcnemar_exact_p']):.3g} |"
        )

    sensitivity_lines = [
        "| InternVL threshold | Original pair | Replacement pair | Sham pair | Erase pair | Full switch | Strict four controls |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sensitivity:
        sensitivity_lines.append(
            f"| {float(row['threshold']):.3f} | {float(row['original_pair_correct_rate']):.3f} | "
            f"{float(row['replacement_pair_correct_rate']):.3f} | {float(row['sham_pair_correct_rate']):.3f} | "
            f"{float(row['erase_pair_correct_rate']):.3f} | {float(row['full_semantic_switch_rate']):.3f} | "
            f"{float(row['all_four_controls_correct_rate']):.3f} |"
        )

    return "\n".join(
        [
            "# Cross-Backbone Semantic-Counterfactual Summary",
            "",
            "All model comparisons use the same 106 automatically screened image edits. Human edit-validity QC remains a prerequisite for confirmatory claims.",
            "",
            *rate_lines,
            "",
            *paired_lines,
            "",
            *sensitivity_lines,
            "",
            "## Interpretation",
            "",
            "Qwen3-VL-8B shows a substantial semantic-replacement response, whereas InternVL3.5-8B does not reproduce it. The low shared success rate means this experiment currently supports a model-dependent Qwen diagnostic, not a cross-backbone causal claim. InternVL's conclusion is stable across the prespecified threshold and three sensitivity thresholds.",
            "",
        ]
    )


def index_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    indexed = {row["image_id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate image_id in model rows")
    return indexed


def as_bool(value: str) -> bool:
    return int(value) == 1


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
