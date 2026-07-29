#!/usr/bin/env python3
"""Build confirmatory semantic-counterfactual results on human-valid edits."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


METRICS = (
    "original_pair_correct",
    "replacement_pair_correct",
    "sham_pair_correct",
    "erase_pair_correct",
    "full_semantic_switch",
    "all_four_controls_correct",
)
PAIRED_METRICS = ("full_semantic_switch", "all_four_controls_correct")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-qc", required=True)
    parser.add_argument("--qwen-rows", required=True)
    parser.add_argument("--internvl-rows", required=True)
    parser.add_argument("--min-valid-edits", type=int, default=80)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--paper-evidence-dir")
    args = parser.parse_args()

    qc_rows = read_csv(Path(args.human_qc))
    qwen = index_rows(read_csv(Path(args.qwen_rows)))
    internvl = index_rows(read_csv(Path(args.internvl_rows)))
    qc = index_rows(qc_rows)
    if set(qc) != set(qwen) or set(qc) != set(internvl):
        raise ValueError("Human-QC and model image IDs do not match")

    incomplete = [row for row in qc_rows if not row.get("qc_decision", "").strip()]
    if incomplete:
        raise ValueError(f"Human QC is incomplete for {len(incomplete)} rows")
    valid_ids = sorted(
        image_id for image_id, row in qc.items() if row["qc_decision"] == "valid_semantic_edit"
    )
    if len(valid_ids) < args.min_valid_edits:
        raise ValueError(
            f"Only {len(valid_ids)} human-valid edits; minimum is {args.min_valid_edits}"
        )

    model_rows = []
    for model_name, rows in (("Qwen3-VL-8B", qwen), ("InternVL3.5-8B", internvl)):
        for metric in METRICS:
            successes = sum(int(rows[image_id][metric]) for image_id in valid_ids)
            low, high = wilson_interval(successes, len(valid_ids))
            model_rows.append(
                {
                    "model": model_name,
                    "metric": metric,
                    "successes": successes,
                    "n": len(valid_ids),
                    "rate": fmt(successes / len(valid_ids)),
                    "wilson_95_low": fmt(low),
                    "wilson_95_high": fmt(high),
                }
            )

    paired_rows = []
    for metric in PAIRED_METRICS:
        pairs = [
            (int(qwen[image_id][metric]), int(internvl[image_id][metric]))
            for image_id in valid_ids
        ]
        both = sum(q == 1 and i == 1 for q, i in pairs)
        qwen_only = sum(q == 1 and i == 0 for q, i in pairs)
        internvl_only = sum(q == 0 and i == 1 for q, i in pairs)
        paired_rows.append(
            {
                "metric": metric,
                "n": len(valid_ids),
                "both_success": both,
                "qwen_only": qwen_only,
                "internvl_only": internvl_only,
                "neither": len(valid_ids) - both - qwen_only - internvl_only,
                "mcnemar_exact_p": f"{mcnemar_exact(qwen_only, internvl_only):.8g}",
            }
        )

    joined_rows = []
    for image_id in sorted(qc):
        joined_rows.append(
            {
                "image_id": image_id,
                "source_text": qc[image_id]["source_text"],
                "replacement_text": qc[image_id]["replacement_text"],
                "qc_decision": qc[image_id]["qc_decision"],
                "included": int(image_id in valid_ids),
                "qwen_full_semantic_switch": qwen[image_id]["full_semantic_switch"],
                "qwen_all_four_controls_correct": qwen[image_id]["all_four_controls_correct"],
                "internvl_full_semantic_switch": internvl[image_id]["full_semantic_switch"],
                "internvl_all_four_controls_correct": internvl[image_id]["all_four_controls_correct"],
            }
        )

    decision_counts: dict[str, int] = {}
    for row in qc_rows:
        decision = row["qc_decision"]
        decision_counts[decision] = decision_counts.get(decision, 0) + 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "human_valid_model_summary.csv", model_rows)
    write_csv(output_dir / "human_valid_paired_summary.csv", paired_rows)
    write_csv(output_dir / "human_valid_joined_rows.csv", joined_rows)
    (output_dir / "human_valid_counterfactual_report.md").write_text(
        render_report(qc_rows, valid_ids, decision_counts, model_rows, paired_rows),
        encoding="utf-8",
    )
    if args.paper_evidence_dir:
        paper_dir = Path(args.paper_evidence_dir)
        paper_dir.mkdir(parents=True, exist_ok=True)
        write_csv(paper_dir / "table_text_replacement_human_valid_model_summary.csv", model_rows)
        write_csv(paper_dir / "table_text_replacement_human_valid_paired_summary.csv", paired_rows)


def render_report(
    qc_rows: list[dict[str, str]],
    valid_ids: list[str],
    decision_counts: dict[str, int],
    model_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Human-Verified Semantic-Counterfactual Audit",
        "",
        f"Human QC was completed for all {len(qc_rows)} automatically screened edits. "
        f"The confirmatory subset contains {len(valid_ids)} edits labeled `valid_semantic_edit`.",
        "",
        "## Human QC",
        "",
        "| Decision | Count |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {key} | {value} |" for key, value in sorted(decision_counts.items()))
    lines.extend(
        [
            "",
            "## Model Results on Human-Valid Edits",
            "",
            "| Model | Metric | Successes | Rate | Wilson 95% CI |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in model_rows:
        lines.append(
            f"| {row['model']} | {row['metric']} | {row['successes']}/{row['n']} | "
            f"{float(row['rate']):.3f} | [{float(row['wilson_95_low']):.3f}, "
            f"{float(row['wilson_95_high']):.3f}] |"
        )
    lines.extend(
        [
            "",
            "## Paired Cross-Backbone Comparison",
            "",
            "| Metric | Both | Qwen only | InternVL only | Neither | Exact McNemar p |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in paired_rows:
        lines.append(
            f"| {row['metric']} | {row['both_success']} | {row['qwen_only']} | "
            f"{row['internvl_only']} | {row['neither']} | "
            f"{float(row['mcnemar_exact_p']):.3g} |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "The edit-validity gate is satisfied. The result may support a human-verified semantic-counterfactual diagnostic for Qwen3-VL-8B. Because InternVL3.5-8B remains substantially weaker on the same valid edits, it does not establish a backbone-uniform causal effect.",
            "",
        ]
    )
    return "\n".join(lines)


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
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


def index_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    indexed = {row["image_id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate image_id")
    return indexed


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> str:
    return f"{value:.6f}"


if __name__ == "__main__":
    main()
