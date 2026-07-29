#!/usr/bin/env python3
"""Build hard-robustness and conflict-slice audits from cached results.

The audit reorganizes existing experiment outputs around reviewer-facing
failure modes: near-miss negatives, low evidence coverage, open-QA stress
slices, noisy evidence boxes, and detector-in-loop settings. It does not rerun
models or invent any values.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "problem_optimization_audit" / "hard_robustness_conflict"
PAPER_DIR = ROOT / "runs" / "paper_evidence"

PROBLEM_DIR = ROOT / "runs" / "problem_optimization_audit"

TEXT_OCR_METHODS = [
    (
        "Target 30%",
        "target relevance",
        ROOT
        / "runs"
        / "prune_textocr_hard_full1000"
        / "qwen3_8b_textocr_hard_full1000_target_embed_topk_0p30_targetfix_802816"
        / "probe_scores.jsonl",
    ),
    (
        "Soft evidence 30%",
        "target plus soft evidence prior",
        ROOT
        / "runs"
        / "prune_textocr_hard_full1000"
        / "qwen3_8b_textocr_hard_full1000_target_embed_soft_evidence_topk_0p30_b0p05_targetfix_802816"
        / "probe_scores.jsonl",
    ),
    (
        "Protected evidence 30%",
        "hard evidence protection",
        ROOT
        / "runs"
        / "prune_textocr_hard_full1000"
        / "qwen3_8b_textocr_hard_full1000_target_embed_protected_topk_0p30"
        / "probe_scores.jsonl",
    ),
    (
        "Coverage-greedy 30%",
        "coverage-greedy evidence objective",
        ROOT
        / "runs"
        / "prune_textocr_hard_full1000"
        / "qwen3_8b_textocr_hard_full1000_target_embed_coverage_greedy_0p30_hard_targetfix_802816"
        / "probe_scores.jsonl",
    ),
    (
        "Random 30%",
        "matched-budget random",
        ROOT
        / "runs"
        / "prune_textocr_hard_full1000"
        / "qwen3_8b_textocr_hard_full1000_random_0p30"
        / "probe_scores.jsonl",
    ),
]

HARD_NEGATIVE_LEXICAL_SUMMARY = (
    PROBLEM_DIR / "hard_negative_lexical_audit" / "hard_negative_lexical_summary.csv"
)
HARD_NEGATIVE_EDIT_CLASS = (
    PROBLEM_DIR / "hard_negative_lexical_audit" / "hard_negative_edit_class_summary.csv"
)
OPEN_QA_STRESS = PROBLEM_DIR / "open_ocr_qa_stress" / "open_ocr_qa_stress_summary.csv"
BBOX_NOISE = PROBLEM_DIR / "open_ocr_qa_bbox_noise_audit" / "bbox_noise_summary.csv"
TEXTOCR_DETECTOR = PROBLEM_DIR / "textocr_detector_in_loop_audit" / "detector_in_loop_key_summary.csv"
OPENQA_DETECTOR = (
    PROBLEM_DIR / "open_ocr_qa_detector_in_loop_readout" / "detector_in_loop_summary.csv"
)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    textocr_rows = build_textocr_method_slice()
    openqa_rows = build_openqa_stress_slice()
    detector_rows = build_detector_noise_slice()
    summary_rows = build_summary(textocr_rows, openqa_rows, detector_rows)

    outputs = {
        "hard_robustness_textocr_method_slice.csv": textocr_rows,
        "hard_robustness_openqa_slice.csv": openqa_rows,
        "hard_robustness_detector_slice.csv": detector_rows,
        "hard_robustness_conflict_summary.csv": summary_rows,
    }
    for name, rows in outputs.items():
        write_csv(OUT_DIR / name, rows)
        write_csv(PAPER_DIR / f"table_{name}", rows)

    (OUT_DIR / "hard_robustness_conflict_report.md").write_text(
        build_markdown(summary_rows, textocr_rows, openqa_rows, detector_rows),
        encoding="utf-8",
    )
    print(f"Wrote hard robustness audit to {OUT_DIR}")


def build_textocr_method_slice() -> list[dict[str, Any]]:
    slices: list[tuple[str, Callable[[dict[str, Any]], bool], str]] = [
        ("all probes", lambda r: True, "overall TextOCR-Hard behavior"),
        ("positive probes", lambda r: r.get("target_answer") == "yes", "evidence-present queries"),
        (
            "near-miss negative probes",
            lambda r: r.get("target_answer") == "no",
            "hard false-positive risk under visually plausible decoys",
        ),
        (
            "low-ECR probes",
            lambda r: as_float(r.get("prune_ecr")) < 0.50,
            "samples where the selector mostly misses annotated evidence",
        ),
        (
            "near-miss negatives with low ECR",
            lambda r: r.get("target_answer") == "no" and as_float(r.get("prune_ecr")) < 0.50,
            "worst conjunction of decoy query and missing evidence",
        ),
        (
            "high-ECR probes",
            lambda r: as_float(r.get("prune_ecr")) >= 0.75,
            "samples where evidence is mostly available after pruning",
        ),
    ]

    rows: list[dict[str, Any]] = []
    for method, family, path in TEXT_OCR_METHODS:
        records = read_jsonl(path)
        for slice_name, predicate, note in slices:
            subset = [row for row in records if predicate(row)]
            rows.append(summarize_probe_rows(method, family, slice_name, subset, note))
    return rows


def build_openqa_stress_slice() -> list[dict[str, Any]]:
    wanted_tags = {
        "all",
        "numeric_answer",
        "long_question_ge10",
        "multi_token_answer_ge3",
        "layout_field_question",
        "full_good_pruned_drop_ge50",
    }
    rows = []
    for row in read_csv(OPEN_QA_STRESS):
        if row["stress_tag"] not in wanted_tags:
            continue
        if row["ratio"] not in {"0.30", "0.70"}:
            continue
        rows.append(
            {
                "source": "open_ocr_qa_stress",
                "task": row["task"],
                "ratio": row["ratio"],
                "stress_tag": row["stress_tag"],
                "n": row["n"],
                "full_score": row["full_score"],
                "pruned_score": row["pruned_score"],
                "delta_score": row["delta_score"],
                "full_exact": row["full_exact"],
                "pruned_exact": row["pruned_exact"],
                "delta_exact": row["delta_exact"],
                "reading": stress_reading(row),
            }
        )
    return rows


def build_detector_noise_slice() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for row in read_csv(BBOX_NOISE):
        if row["budget_keep_ratio"] != "0.70":
            continue
        if row["variant"] not in {"clean", "jitter_25pct", "drop_40pct", "mixed_heavy"}:
            continue
        rows.append(
            {
                "source": "bbox_noise",
                "scope": row["task"],
                "condition": row["variant"],
                "n_or_rows": row["scored_rows"],
                "primary_metric": "retention_adjusted_ECR",
                "primary_value": row["retention_adjusted_ECR"],
                "secondary_metric": "mean_ECR",
                "secondary_value": row["mean_ECR"],
                "cost_or_latency": "",
                "reading": bbox_noise_reading(row),
            }
        )

    for row in read_csv(TEXTOCR_DETECTOR):
        if row["metric"] not in {
            "mean_detector_ms_per_image",
            "easyocr_detector_in_loop",
            "easyocr_minus_gt",
        }:
            continue
        rows.append(
            {
                "source": "textocr_detector_in_loop",
                "scope": row["scope"],
                "condition": row["metric"],
                "n_or_rows": "",
                "primary_metric": "reported_metric",
                "primary_value": row["value"],
                "secondary_metric": "",
                "secondary_value": "",
                "cost_or_latency": detector_latency(row),
                "reading": row["interpretation"],
            }
        )

    for row in read_csv(OPENQA_DETECTOR):
        if row["scope"] not in {"all", "DocVQA-lite", "TextVQA-lite", "rows_without_detector_boxes"}:
            continue
        rows.append(
            {
                "source": "open_ocr_qa_easyocr_detector",
                "scope": row["scope"],
                "condition": "EasyOCR soft evidence 70%",
                "n_or_rows": row["n"],
                "primary_metric": "easyocr_soft_evidence70_score",
                "primary_value": row["easyocr_soft_evidence70_score"],
                "secondary_metric": "soft_minus_cached_grid70",
                "secondary_value": row["soft_minus_cached_grid70"],
                "cost_or_latency": f"mean_detector_box_count={row['mean_detector_box_count']}; rows_with_detector_boxes={row['rows_with_detector_boxes']}",
                "reading": "Detector boxes improve over cached grid on TextVQA and overall, but DocVQA gains are smaller and missing detector boxes give no benefit.",
            }
        )
    return rows


def build_summary(
    textocr_rows: list[dict[str, Any]],
    openqa_rows: list[dict[str, Any]],
    detector_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    lex = key_value_csv(HARD_NEGATIVE_LEXICAL_SUMMARY)
    edit = read_csv(HARD_NEGATIVE_EDIT_CLASS)

    target_neg = find_row(textocr_rows, method="Target 30%", slice="near-miss negative probes")
    soft_neg = find_row(textocr_rows, method="Soft evidence 30%", slice="near-miss negative probes")
    protected_neg = find_row(textocr_rows, method="Protected evidence 30%", slice="near-miss negative probes")
    random_neg = find_row(textocr_rows, method="Random 30%", slice="near-miss negative probes")
    textvqa70_numeric = find_row(
        openqa_rows, task="TextVQA-lite", ratio="0.70", stress_tag="numeric_answer"
    )
    docvqa70_long = find_row(
        openqa_rows, task="DocVQA-lite", ratio="0.70", stress_tag="long_question_ge10"
    )
    jitter_text = find_row(detector_rows, source="bbox_noise", scope="TextVQA-lite", condition="jitter_25pct")
    drop_doc = find_row(detector_rows, source="bbox_noise", scope="DocVQA-lite", condition="drop_40pct")
    internvl_detector = find_detector_metric(detector_rows, "InternVL3.5-8B", "easyocr_detector_in_loop")
    llava_detector = find_detector_metric(detector_rows, "LLaVA-1.5-7B", "easyocr_detector_in_loop")
    openqa_detector_all = find_row(
        detector_rows,
        source="open_ocr_qa_easyocr_detector",
        scope="all",
        condition="EasyOCR soft evidence 70%",
    )

    substitutions = find_edit_count(edit, "edit_type", "single_substitution")
    deletions = find_edit_count(edit, "edit_type", "single_deletion")

    return [
        {
            "category": "near-miss benchmark validity",
            "source": "hard_negative_lexical_audit",
            "key_result": (
                f"same-image collision rate {lex['same_image_collision_any_normalizer_rate']}; "
                f"mean edit distance {lex['mean_edit_distance_nfkc_casefold']}; "
                f"{substitutions} substitutions and {deletions} deletions."
            ),
            "reviewer_reading": "The negative probes are close lexical decoys without automatic same-image OCR collisions.",
            "claim_boundary": "This is automatic lexical validation, not a substitute for full human label validation.",
            "status": "strong_support_for_data_quality",
        },
        {
            "category": "near-miss answer risk",
            "source": "TextOCR-Hard probe slices",
            "key_result": (
                f"Target hFPR {target_neg['hFPR']}; Soft evidence {soft_neg['hFPR']}; "
                f"Protected evidence {protected_neg['hFPR']}; Random {random_neg['hFPR']}."
            ),
            "reviewer_reading": "Evidence-aware variants must be judged with hFPR, because higher coverage can raise false positives on decoys.",
            "claim_boundary": "Do not claim evidence coverage alone solves hard negatives; report answer-risk trade-offs.",
            "status": "mixed_but_measured",
        },
        {
            "category": "open-QA stress",
            "source": "TextVQA/DocVQA native generation slices",
            "key_result": (
                f"TextVQA numeric at 70% changes score by {textvqa70_numeric['delta_score']}; "
                f"DocVQA long questions at 70% change score by {docvqa70_long['delta_score']}."
            ),
            "reviewer_reading": "Higher retention largely recovers TextVQA, while DocVQA long/document questions remain a boundary.",
            "claim_boundary": "Use as generalization plus failure taxonomy, not leaderboard-scale proof.",
            "status": "boundary_support",
        },
        {
            "category": "noisy and missing evidence boxes",
            "source": "bbox_noise_audit",
            "key_result": (
                f"TextVQA 25% jitter adjusted ECR {jitter_text['primary_value']}; "
                f"DocVQA 40% dropout adjusted ECR {drop_doc['primary_value']}."
            ),
            "reviewer_reading": "Coordinate jitter is mild; detector dropout is the damaging failure mode.",
            "claim_boundary": "A deployment system should expose detector-missing fallback rather than only box-aware pruning.",
            "status": "boundary_support",
        },
        {
            "category": "detector-in-loop backbone robustness",
            "source": "TextOCR-Hard EasyOCR detector-in-loop",
            "key_result": (
                f"InternVL EasyOCR row: {internvl_detector['primary_value']}; "
                f"LLaVA EasyOCR row: {llava_detector['primary_value']}."
            ),
            "reviewer_reading": "Soft evidence is less brittle than hard protection when selector-visible boxes are noisy.",
            "claim_boundary": "Online detector latency must be reported separately from detector-free pruning speed.",
            "status": "model_dependent_support",
        },
        {
            "category": "open-QA detector boxes",
            "source": "open-QA EasyOCR detector readout",
            "key_result": (
                f"Overall EasyOCR soft-evidence score {openqa_detector_all['primary_value']}; "
                f"gain over cached grid70 {openqa_detector_all['secondary_value']}."
            ),
            "reviewer_reading": "Automatic OCR boxes can help in a scoped open-QA setting, but the gain is task dependent.",
            "claim_boundary": "Do not count this as detector-inclusive speedup without adding OCR runtime.",
            "status": "scoped_positive",
        },
    ]


def summarize_probe_rows(
    method: str,
    family: str,
    slice_name: str,
    rows: list[dict[str, Any]],
    note: str,
) -> dict[str, Any]:
    positives = [row for row in rows if row.get("target_answer") == "yes"]
    negatives = [row for row in rows if row.get("target_answer") == "no"]
    false_positive_rows = [
        row
        for row in negatives
        if str(row.get("pred_answer", "")).strip().lower().startswith("yes")
    ]
    yes_rows = [
        row for row in rows if str(row.get("pred_answer", "")).strip().lower().startswith("yes")
    ]
    return {
        "method": method,
        "selector_family": family,
        "slice": slice_name,
        "n": len(rows),
        "accuracy": fmt(rate([bool(row.get("correct")) for row in rows])),
        "positive_accuracy": fmt(rate([bool(row.get("correct")) for row in positives])),
        "negative_accuracy": fmt(rate([bool(row.get("correct")) for row in negatives])),
        "hFPR": fmt(len(false_positive_rows) / len(negatives)) if negatives else "",
        "yes_rate": fmt(len(yes_rows) / len(rows)) if rows else "",
        "mean_ECR": fmt(avg(row.get("prune_ecr") for row in rows)),
        "mean_center_recall": fmt(avg(row.get("prune_evidence_center_recall") for row in rows)),
        "mean_patch_recall": fmt(avg(row.get("prune_evidence_patch_recall") for row in rows)),
        "mean_margin": fmt(avg(row.get("margin") for row in rows)),
        "note": note,
    }


def stress_reading(row: dict[str, str]) -> str:
    delta = as_float(row["delta_score"])
    if row["stress_tag"] == "full_good_pruned_drop_ge50":
        return "Outcome-only failure slice: pruning can still catastrophically fail on a small set of full-prefix-correct examples."
    if row["ratio"] == "0.70" and delta > -0.05:
        return "High retention largely recovers this stress slice."
    if row["ratio"] == "0.70":
        return "Even high retention leaves a measurable stress-slice drop."
    return "Low retention is fragile on this stress slice."


def bbox_noise_reading(row: dict[str, str]) -> str:
    if row["variant"].startswith("jitter"):
        return "Localization jitter has limited effect compared with dropout at the same budget."
    if row["variant"].startswith("drop"):
        return "Evidence-box dropout sharply reduces retention-adjusted coverage."
    if row["variant"].startswith("mixed"):
        return "Combined jitter and dropout exposes the detector-failure boundary."
    return "Clean-box reference condition."


def detector_latency(row: dict[str, str]) -> str:
    if row["metric"] == "mean_detector_ms_per_image":
        return f"{row['value']} ms/image"
    return ""


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def key_value_csv(path: Path) -> dict[str, str]:
    return {row["metric"]: row["value"] for row in read_csv(path)}


def find_row(rows: list[dict[str, Any]], **criteria: Any) -> dict[str, Any]:
    for row in rows:
        if all(str(row.get(key)) == str(value) for key, value in criteria.items()):
            return row
    raise KeyError(f"Missing row for {criteria}")


def find_detector_metric(rows: list[dict[str, Any]], scope: str, condition: str) -> dict[str, Any]:
    return find_row(rows, source="textocr_detector_in_loop", scope=scope, condition=condition)


def find_edit_count(rows: list[dict[str, str]], group: str, bucket: str) -> str:
    for row in rows:
        if row["group"] == group and row["bucket"] == bucket:
            return row["count"]
    return "0"


def rate(values: list[bool]) -> float | None:
    if not values:
        return None
    return sum(1 for value in values if value) / len(values)


def avg(values: Any) -> float | None:
    vals = [as_float(value) for value in values if value not in (None, "")]
    if not vals:
        return None
    return mean(vals)


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.3f}"


def build_markdown(
    summary_rows: list[dict[str, Any]],
    textocr_rows: list[dict[str, Any]],
    openqa_rows: list[dict[str, Any]],
    detector_rows: list[dict[str, Any]],
) -> str:
    return "\n".join(
        [
            "# Hard Robustness and Conflict-Slice Audit",
            "",
            "This audit reorganizes existing cached results around the hard-case concerns in `problem.md`. It is not a new benchmark run.",
            "",
            "## Summary",
            "",
            table_md(
                summary_rows,
                ["category", "source", "key_result", "reviewer_reading", "claim_boundary", "status"],
            ),
            "",
            "## TextOCR-Hard Method Slices",
            "",
            table_md(
                textocr_rows,
                [
                    "method",
                    "selector_family",
                    "slice",
                    "n",
                    "accuracy",
                    "hFPR",
                    "yes_rate",
                    "mean_ECR",
                    "mean_margin",
                    "note",
                ],
            ),
            "",
            "## Open-QA Stress Slices",
            "",
            table_md(
                openqa_rows,
                [
                    "task",
                    "ratio",
                    "stress_tag",
                    "n",
                    "full_score",
                    "pruned_score",
                    "delta_score",
                    "reading",
                ],
            ),
            "",
            "## Detector and Noisy-Box Slices",
            "",
            table_md(
                detector_rows,
                [
                    "source",
                    "scope",
                    "condition",
                    "n_or_rows",
                    "primary_metric",
                    "primary_value",
                    "secondary_metric",
                    "secondary_value",
                    "cost_or_latency",
                    "reading",
                ],
            ),
            "",
            "## Safe Reading",
            "",
            "- The near-miss benchmark is lexically clean under automatic normalization checks, but still lacks complete human label validation.",
            "- Evidence preservation improves the audit surface, yet hard protection can raise false positives; answer risk must remain paired with ECR.",
            "- TextVQA-style open OCR is more recoverable at 70% retention than DocVQA-style document QA.",
            "- Coordinate noise is milder than detector dropout; detector-assisted claims must include a fallback policy and OCR latency when boxes are not already available.",
            "",
        ]
    )


def table_md(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "(no rows)"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
