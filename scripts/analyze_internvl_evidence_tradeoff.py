#!/usr/bin/env python
"""Analyze InternVL evidence-protection trade-offs on paired TextOCR probes."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_BASELINE = "runs/internvl_textocr_hard/calibrated_test_target0p50_devthr"
DEFAULT_CANDIDATES = (
    "soft_b0p05_hfprconstr:runs/internvl_textocr_hard/calibrated_test_target_soft_evidence0p50_b0p05_hfprconstr_devthr",
    "soft_b0p05:runs/internvl_textocr_hard/calibrated_test_target_soft_evidence0p50_b0p05_devthr",
    "hard_protected:runs/internvl_textocr_hard/calibrated_test_target_protected0p50_devthr",
    "grid0p50:runs/internvl_textocr_hard/calibrated_test_grid0p50_devthr",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probes", default="data/textocr_val_hard_probes_500img.jsonl")
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument("--candidate", action="append", default=list(DEFAULT_CANDIDATES), help="label:run_dir")
    parser.add_argument("--out-dir", default="runs/internvl_textocr_hard/evidence_tradeoff_analysis")
    parser.add_argument("--top-examples", type=int, default=24)
    args = parser.parse_args()

    probes = load_jsonl_by_id(Path(args.probes), "sample_id")
    baseline = load_run(Path(args.baseline))
    candidates = [parse_candidate(spec) for spec in args.candidate]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    pattern_rows: list[dict[str, Any]] = []
    flip_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []
    candidate_reports: list[str] = []

    for label, run_dir in candidates:
        candidate = load_run(run_dir)
        ids = sorted(set(baseline.rows) & set(candidate.rows) & set(probes))
        summaries.append(summarize_run("baseline_target0p50", baseline, probes, ids))
        cand_summary = summarize_run(label, candidate, probes, ids)
        summaries.append(cand_summary)

        threshold_delta = candidate.threshold - baseline.threshold
        threshold_drop = baseline.threshold - candidate.threshold
        patterns = image_pair_patterns(ids, probes, baseline, candidate)
        neg_flips = flip_decomposition(ids, probes, baseline, candidate, polarity="negative")
        pos_flips = flip_decomposition(ids, probes, baseline, candidate, polarity="positive")
        set_stat_context(probes, baseline, candidate)

        for row in pattern_summary_rows(label, patterns):
            pattern_rows.append(row)
        for row in flip_summary_rows(label, "negative", neg_flips, threshold_delta, threshold_drop):
            flip_rows.append(row)
        for row in flip_summary_rows(label, "positive", pos_flips, threshold_delta, threshold_drop):
            flip_rows.append(row)

        examples = negative_flip_examples(ids, probes, baseline, candidate, label, limit=args.top_examples)
        example_rows.extend(examples)

        candidate_reports.append(
            render_candidate_section(
                label=label,
                baseline=baseline,
                candidate=candidate,
                cand_summary=cand_summary,
                patterns=patterns,
                neg_flips=neg_flips,
                pos_flips=pos_flips,
                threshold_delta=threshold_delta,
                threshold_drop=threshold_drop,
                examples=examples[:8],
            )
        )

    # The baseline summary is repeated per candidate above; keep one copy in the CSV/report.
    unique_summaries = dedupe_summaries(summaries)
    write_csv(out_dir / "run_summaries.csv", unique_summaries)
    write_csv(out_dir / "pair_patterns.csv", pattern_rows)
    write_csv(out_dir / "flip_decomposition.csv", flip_rows)
    write_csv(out_dir / "negative_flip_examples.csv", example_rows)
    report = render_report(unique_summaries, candidate_reports)
    (out_dir / "evidence_tradeoff_report.md").write_text(report, encoding="utf-8")
    print(f"Wrote evidence trade-off report to {out_dir / 'evidence_tradeoff_report.md'}")


class Run:
    def __init__(self, run_dir: Path, rows: dict[str, dict[str, Any]]) -> None:
        self.run_dir = run_dir
        self.rows = rows
        self.threshold = first_threshold(rows)


def parse_candidate(spec: str) -> tuple[str, Path]:
    label, sep, path = spec.partition(":")
    if not sep or not label or not path:
        raise ValueError(f"Invalid candidate {spec!r}; expected label:run_dir")
    return label, Path(path)


def load_run(run_dir: Path) -> Run:
    return Run(run_dir=run_dir, rows=load_jsonl_by_id(run_dir / "probe_scores.jsonl", "sample_id"))


def load_jsonl_by_id(path: Path, key: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row[key])] = row
    return rows


def first_threshold(rows: dict[str, dict[str, Any]]) -> float:
    for row in rows.values():
        return float(row.get("yesno_threshold", 0.0))
    return 0.0


def summarize_run(label: str, run: Run, probes: dict[str, dict[str, Any]], ids: list[str]) -> dict[str, Any]:
    selected = [run.rows[sid] for sid in ids]
    negatives = [row for row in selected if probes[str(row["sample_id"])].get("binary_polarity") == "negative"]
    positives = [row for row in selected if probes[str(row["sample_id"])].get("binary_polarity") == "positive"]
    return {
        "run": label,
        "path": str(run.run_dir),
        "n": len(selected),
        "threshold": run.threshold,
        "acc": mean(float(row.get("correct", False)) for row in selected),
        "hFPR": mean(float(row.get("pred_answer") == "yes") for row in negatives),
        "yes_rate": mean(float(row.get("pred_answer") == "yes") for row in selected),
        "pos_acc": mean(float(row.get("correct", False)) for row in positives),
        "neg_acc": mean(float(row.get("correct", False)) for row in negatives),
        "ECR": mean_optional(row.get("prune_ecr") for row in selected),
        "CenterR": mean_optional(row.get("prune_evidence_center_recall") for row in selected),
        "PatchR": mean_optional(row.get("prune_evidence_patch_recall") for row in selected),
        "keep_ratio": mean_optional(row.get("prune_keep_ratio") for row in selected),
    }


def dedupe_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row["run"]), str(row["path"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def image_pair_patterns(
    ids: list[str],
    probes: dict[str, dict[str, Any]],
    baseline: Run,
    candidate: Run,
) -> Counter[tuple[str, str]]:
    by_image: dict[str, dict[str, tuple[str, str]]] = defaultdict(dict)
    for sid in ids:
        probe = probes[sid]
        image_id = str(probe.get("image_id", ""))
        polarity = str(probe.get("binary_polarity", ""))
        by_image[image_id][polarity] = (
            str(baseline.rows[sid].get("pred_answer", "")),
            str(candidate.rows[sid].get("pred_answer", "")),
        )

    patterns: Counter[tuple[str, str]] = Counter()
    for pair in by_image.values():
        if "positive" not in pair or "negative" not in pair:
            continue
        baseline_pattern = f"{pair['positive'][0]}/{pair['negative'][0]}"
        candidate_pattern = f"{pair['positive'][1]}/{pair['negative'][1]}"
        patterns[(baseline_pattern, candidate_pattern)] += 1
    return patterns


def pattern_summary_rows(label: str, patterns: Counter[tuple[str, str]]) -> list[dict[str, Any]]:
    total = sum(patterns.values())
    rows = []
    for (baseline_pattern, candidate_pattern), count in patterns.most_common():
        rows.append(
            {
                "candidate": label,
                "baseline_pattern": baseline_pattern,
                "candidate_pattern": candidate_pattern,
                "count": count,
                "fraction": count / total if total else 0.0,
            }
        )
    return rows


def flip_decomposition(
    ids: list[str],
    probes: dict[str, dict[str, Any]],
    baseline: Run,
    candidate: Run,
    *,
    polarity: str,
) -> dict[str, list[str]]:
    buckets = {"new_yes": [], "new_no": [], "both_yes": [], "both_no": []}
    for sid in ids:
        if probes[sid].get("binary_polarity") != polarity:
            continue
        base_yes = baseline.rows[sid].get("pred_answer") == "yes"
        cand_yes = candidate.rows[sid].get("pred_answer") == "yes"
        if cand_yes and not base_yes:
            buckets["new_yes"].append(sid)
        elif base_yes and not cand_yes:
            buckets["new_no"].append(sid)
        elif cand_yes and base_yes:
            buckets["both_yes"].append(sid)
        else:
            buckets["both_no"].append(sid)
    return buckets


def flip_summary_rows(
    label: str,
    polarity: str,
    buckets: dict[str, list[str]],
    threshold_delta: float,
    threshold_drop: float,
) -> list[dict[str, Any]]:
    rows = []
    for bucket, ids in buckets.items():
        rows.append(
            {
                "candidate": label,
                "polarity": polarity,
                "bucket": bucket,
                "count": len(ids),
                "threshold_delta_candidate_minus_baseline": threshold_delta,
                "threshold_drop_baseline_minus_candidate": threshold_drop,
                "mean_raw_margin_delta": mean_id_stat(ids, "raw_delta"),
                "median_raw_margin_delta": median_id_stat(ids, "raw_delta"),
                "mean_calibrated_margin_delta": mean_id_stat(ids, "cal_delta"),
                "mean_ecr_delta": mean_id_stat(ids, "ecr_delta"),
                "mean_token_area": mean_id_stat(ids, "token_area"),
                "mean_token_area_rank": mean_id_stat(ids, "token_area_rank"),
            }
        )
    return rows


_STAT_CONTEXT: dict[str, Any] = {}


def set_stat_context(probes: dict[str, dict[str, Any]], baseline: Run, candidate: Run) -> None:
    _STAT_CONTEXT.clear()
    _STAT_CONTEXT.update({"probes": probes, "baseline": baseline, "candidate": candidate})


def stat_value(sid: str, key: str) -> float:
    probes: dict[str, dict[str, Any]] = _STAT_CONTEXT["probes"]
    baseline: Run = _STAT_CONTEXT["baseline"]
    candidate: Run = _STAT_CONTEXT["candidate"]
    base = baseline.rows[sid]
    cand = candidate.rows[sid]
    probe = probes[sid]
    if key == "raw_delta":
        return float(cand.get("raw_margin", cand.get("margin", 0.0))) - float(base.get("raw_margin", base.get("margin", 0.0)))
    if key == "cal_delta":
        return float(cand.get("margin", 0.0)) - float(base.get("margin", 0.0))
    if key == "ecr_delta":
        return float(cand.get("prune_ecr", 0.0)) - float(base.get("prune_ecr", 0.0))
    if key == "token_area":
        return float(probe.get("token_area", 0.0))
    if key == "token_area_rank":
        return float(probe.get("token_area_rank", 0.0))
    raise KeyError(key)


def mean_id_stat(ids: list[str], key: str) -> float:
    return mean(stat_value(sid, key) for sid in ids) if ids else 0.0


def median_id_stat(ids: list[str], key: str) -> float:
    return statistics.median(stat_value(sid, key) for sid in ids) if ids else 0.0


def negative_flip_examples(
    ids: list[str],
    probes: dict[str, dict[str, Any]],
    baseline: Run,
    candidate: Run,
    label: str,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sid in ids:
        probe = probes[sid]
        if probe.get("binary_polarity") != "negative":
            continue
        base = baseline.rows[sid]
        cand = candidate.rows[sid]
        if base.get("pred_answer") != "no" or cand.get("pred_answer") != "yes":
            continue
        rows.append(
            {
                "candidate": label,
                "sample_id": sid,
                "image_id": probe.get("image_id", ""),
                "source_text": probe.get("source_text", ""),
                "target_text": probe.get("target_text", ""),
                "question": probe.get("question", ""),
                "baseline_raw_margin": float(base.get("raw_margin", 0.0)),
                "candidate_raw_margin": float(cand.get("raw_margin", 0.0)),
                "raw_margin_delta": float(cand.get("raw_margin", 0.0)) - float(base.get("raw_margin", 0.0)),
                "baseline_margin": float(base.get("margin", 0.0)),
                "candidate_margin": float(cand.get("margin", 0.0)),
                "baseline_ecr": float(base.get("prune_ecr", 0.0)),
                "candidate_ecr": float(cand.get("prune_ecr", 0.0)),
                "token_area": probe.get("token_area", ""),
                "token_area_rank": probe.get("token_area_rank", ""),
                "image": probe.get("image", ""),
            }
        )
    rows.sort(key=lambda row: (-row["candidate_margin"], row["sample_id"]))
    return rows[:limit]


def render_candidate_section(
    *,
    label: str,
    baseline: Run,
    candidate: Run,
    cand_summary: dict[str, Any],
    patterns: Counter[tuple[str, str]],
    neg_flips: dict[str, list[str]],
    pos_flips: dict[str, list[str]],
    threshold_delta: float,
    threshold_drop: float,
    examples: list[dict[str, Any]],
) -> str:
    lines = [
        f"## {label}",
        "",
        f"- Threshold: baseline {baseline.threshold:.6f}, candidate {candidate.threshold:.6f}, candidate-baseline {threshold_delta:+.6f}.",
        f"- Candidate Acc {cand_summary['acc']:.4f}, hFPR {cand_summary['hFPR']:.4f}, ECR {fmt_optional(cand_summary['ECR'])}.",
        f"- Positive `no -> yes` recoveries: {len(pos_flips['new_yes'])}; positive `yes -> no` regressions: {len(pos_flips['new_no'])}.",
        f"- Negative `no -> yes` new false positives: {len(neg_flips['new_yes'])}; negative `yes -> no` fixes: {len(neg_flips['new_no'])}.",
        "",
        "### Image-Pair Pattern Shifts",
        "",
        "| baseline pos/neg | candidate pos/neg | images |",
        "|---|---|---:|",
    ]
    for (base_pat, cand_pat), count in patterns.most_common(12):
        lines.append(f"| {base_pat} | {cand_pat} | {count} |")

    lines.extend(
        [
            "",
            "### Margin Decomposition",
            "",
            "| polarity | bucket | n | raw delta | calibrated delta | threshold drop | ECR delta |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for polarity, buckets in [("positive", pos_flips), ("negative", neg_flips)]:
        for bucket in ["new_yes", "new_no", "both_yes", "both_no"]:
            ids = buckets[bucket]
            lines.append(
                f"| {polarity} | {bucket} | {len(ids)} | "
                f"{mean_id_stat(ids, 'raw_delta'):.4f} | {mean_id_stat(ids, 'cal_delta'):.4f} | "
                f"{threshold_drop:.4f} | {mean_id_stat(ids, 'ecr_delta'):.4f} |"
            )

    lines.extend(
        [
            "",
            "### New Negative False-Yes Examples",
            "",
            "| sample | source | near miss | raw delta | candidate margin | ECR base->cand | question |",
            "|---|---|---|---:|---:|---|---|",
        ]
    )
    for row in examples:
        question = str(row["question"]).replace("|", "/")
        lines.append(
            f"| {row['sample_id']} | {row['source_text']} | {row['target_text']} | "
            f"{row['raw_margin_delta']:.4f} | {row['candidate_margin']:.4f} | "
            f"{row['baseline_ecr']:.3f}->{row['candidate_ecr']:.3f} | {question} |"
        )
    return "\n".join(lines)


def render_report(summaries: list[dict[str, Any]], candidate_sections: list[str]) -> str:
    lines = [
        "# InternVL Evidence-Protection Trade-Off Analysis",
        "",
        "This report compares calibrated test-split InternVL runs on TextOCR-Hard paired positive/near-miss negative probes.",
        "",
        "## Run Summary",
        "",
        "| run | acc | hFPR | yes | pos acc | neg acc | keep | ECR | CenterR | PatchR | threshold |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['run']} | {row['acc']:.3f} | {row['hFPR']:.3f} | {row['yes_rate']:.3f} | "
            f"{row['pos_acc']:.3f} | {row['neg_acc']:.3f} | {fmt_optional(row['keep_ratio'])} | "
            f"{fmt_optional(row['ECR'])} | {fmt_optional(row['CenterR'])} | {fmt_optional(row['PatchR'])} | "
            f"{row['threshold']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Takeaway",
            "",
            "Evidence protection mainly moves InternVL toward a more permissive yes/no operating point: it recovers missed positive OCR detections, but it also turns many near-miss negatives into false yes predictions. For the soft boost, most new false positives have little raw-margin increase; the lower calibrated threshold explains most of the shift. For hard protection, raw margins also rise, suggesting forced evidence tokens amplify image-level text presence rather than target-text discrimination.",
            "",
        ]
    )
    lines.extend(candidate_sections)
    return "\n\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: Any) -> float:
    vals = [float(value) for value in values]
    return sum(vals) / len(vals) if vals else 0.0


def mean_optional(values: Any) -> float | None:
    vals = [float(value) for value in values if value is not None]
    return sum(vals) / len(vals) if vals else None


def fmt_optional(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.3f}"


if __name__ == "__main__":
    main()
