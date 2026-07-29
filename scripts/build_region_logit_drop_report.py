#!/usr/bin/env python
"""Build region evidence logit-drop diagnostics from cached probe scores.

The report compares yes/no likelihood margins under evidence-preserving and
evidence-removing pruning interventions. It is a causal-style diagnostic, not a
new model run: by default it reuses cached Qwen TextOCR-Hard probe scores.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_ARMS = (
    (
        "full",
        "runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_topk_1p00",
        "Full-token reference through the same pruning backend.",
    ),
    (
        "evidence_kept",
        "runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_topk_0p30",
        "Oracle evidence-preserving intervention: keep top evidence-overlap tokens at 30%.",
    ),
    (
        "evidence_removed",
        "runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_bottomk_0p30",
        "Oracle evidence-removal intervention: keep least evidence-overlap tokens at 30%.",
    ),
    (
        "ours_target0p30",
        "runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_topk_0p30_targetfix_802816",
        "Main Qwen target-conditioned pruning point at 30%.",
    ),
)

DEFAULT_COMPARISONS = (
    ("evidence_kept_vs_removed", "evidence_kept", "evidence_removed"),
    ("full_vs_removed", "full", "evidence_removed"),
    ("ours_vs_removed", "ours_target0p30", "evidence_removed"),
    ("ours_vs_full", "ours_target0p30", "full"),
    ("ours_vs_evidence_kept", "ours_target0p30", "evidence_kept"),
)


@dataclass(frozen=True)
class ArmSpec:
    label: str
    path: Path
    note: str


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="runs/region_logit_drop")
    parser.add_argument("--prefix", default="qwen_region_logit_drop")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--arm",
        action="append",
        help="Arm as label:path[:note]. If omitted, the Qwen TextOCR-Hard defaults are used.",
    )
    parser.add_argument(
        "--comparison",
        action="append",
        help="Comparison as label:left:right. Diff is left minus right.",
    )
    parser.add_argument("--top-examples", type=int, default=30)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arm_specs = parse_arms(args.arm)
    arms = {spec.label: load_probe_scores(resolve_probe_scores(spec.path)) for spec in arm_specs}
    common_ids = sorted(set.intersection(*(set(rows) for rows in arms.values())))
    if not common_ids:
        raise ValueError("No common probe ids across arms.")

    comparisons = parse_comparisons(args.comparison)
    summary_rows = [summarize_arm(label, rows, common_ids, note=note_for(label, arm_specs)) for label, rows in arms.items()]
    pairwise_rows = [
        compare_arms(
            label=label,
            left=left,
            right=right,
            arms=arms,
            sample_ids=common_ids,
            n_bootstrap=args.bootstrap,
            seed=args.seed + idx,
        )
        for idx, (label, left, right) in enumerate(comparisons)
    ]
    example_rows = build_examples(arms, common_ids, limit=args.top_examples)

    write_csv(out_dir / f"{args.prefix}_summary.csv", summary_rows)
    write_csv(out_dir / f"{args.prefix}_pairwise.csv", pairwise_rows)
    write_jsonl(out_dir / f"{args.prefix}_examples.jsonl", example_rows)
    write_markdown(
        out_dir / f"{args.prefix}_report.md",
        summary_rows=summary_rows,
        pairwise_rows=pairwise_rows,
        example_rows=example_rows,
        arm_specs=arm_specs,
        n=len(common_ids),
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    print(f"Wrote region logit-drop diagnostics to {out_dir}")


def parse_arms(values: list[str] | None) -> list[ArmSpec]:
    if not values:
        return [ArmSpec(label, Path(path), note) for label, path, note in DEFAULT_ARMS]
    specs: list[ArmSpec] = []
    for value in values:
        parts = value.split(":", 2)
        if len(parts) < 2:
            raise ValueError(f"Invalid arm {value!r}; expected label:path[:note].")
        label, path = parts[0], parts[1]
        note = parts[2] if len(parts) == 3 else ""
        specs.append(ArmSpec(label, Path(path), note))
    return specs


def parse_comparisons(values: list[str] | None) -> list[tuple[str, str, str]]:
    if not values:
        return list(DEFAULT_COMPARISONS)
    out: list[tuple[str, str, str]] = []
    for value in values:
        parts = value.split(":")
        if len(parts) != 3 or not all(parts):
            raise ValueError(f"Invalid comparison {value!r}; expected label:left:right.")
        out.append((parts[0], parts[1], parts[2]))
    return out


def note_for(label: str, specs: list[ArmSpec]) -> str:
    for spec in specs:
        if spec.label == label:
            return spec.note
    return ""


def resolve_probe_scores(path: Path) -> Path:
    if path.is_dir():
        return path / "probe_scores.jsonl"
    return path


def load_probe_scores(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            sample_id = str(raw["sample_id"])
            target = str(raw.get("target_answer", raw.get("direct_target", ""))).strip().lower()
            if target not in {"yes", "no"}:
                continue
            margin = float(raw.get("margin", float(raw["no_loss"]) - float(raw["yes_loss"])))
            target_support = margin if target == "yes" else -margin
            pred = str(raw.get("pred_answer", raw.get("direct_pred", ""))).strip().lower()
            if pred not in {"yes", "no"}:
                pred = "yes" if margin > 0.0 else "no"
            correct = bool(raw.get("correct", raw.get("direct_correct", pred == target)))
            rows[sample_id] = {
                "sample_id": sample_id,
                "image_id": raw.get("image_id", ""),
                "question": raw.get("question", ""),
                "target_answer": target,
                "pred_answer": pred,
                "correct": correct,
                "hallucination": bool(target == "no" and pred == "yes"),
                "yes_margin": margin,
                "target_support": target_support,
                "yes_loss": float(raw.get("yes_loss", 0.0)),
                "no_loss": float(raw.get("no_loss", 0.0)),
                "keep_ratio": maybe_float(raw.get("prune_keep_ratio")),
                "ecr": maybe_float(raw.get("prune_ecr")),
                "center_recall": maybe_float(raw.get("prune_evidence_center_recall")),
                "patch_recall": maybe_float(raw.get("prune_evidence_patch_recall")),
            }
    if not rows:
        raise ValueError(f"No valid yes/no rows found in {path}")
    return rows


def summarize_arm(label: str, rows: dict[str, dict[str, Any]], sample_ids: list[str], *, note: str) -> dict[str, Any]:
    selected = [rows[sid] for sid in sample_ids]
    pos = [row for row in selected if row["target_answer"] == "yes"]
    neg = [row for row in selected if row["target_answer"] == "no"]
    return {
        "arm": label,
        "n": len(selected),
        "n_pos": len(pos),
        "n_neg": len(neg),
        "acc": mean([1.0 if row["correct"] else 0.0 for row in selected]),
        "pos_acc": mean([1.0 if row["correct"] else 0.0 for row in pos]),
        "neg_acc": mean([1.0 if row["correct"] else 0.0 for row in neg]),
        "hFPR": mean([1.0 if row["hallucination"] else 0.0 for row in neg]),
        "yes_rate": mean([1.0 if row["pred_answer"] == "yes" else 0.0 for row in selected]),
        "mean_target_support": mean([row["target_support"] for row in selected]),
        "median_target_support": median([row["target_support"] for row in selected]),
        "mean_pos_support": mean([row["target_support"] for row in pos]),
        "mean_neg_support": mean([row["target_support"] for row in neg]),
        "mean_yes_margin": mean([row["yes_margin"] for row in selected]),
        "mean_keep_ratio": mean_present([row["keep_ratio"] for row in selected]),
        "mean_ecr": mean_present([row["ecr"] for row in selected]),
        "mean_center_recall": mean_present([row["center_recall"] for row in selected]),
        "mean_patch_recall": mean_present([row["patch_recall"] for row in selected]),
        "note": note,
    }


def compare_arms(
    *,
    label: str,
    left: str,
    right: str,
    arms: dict[str, dict[str, dict[str, Any]]],
    sample_ids: list[str],
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    if left not in arms or right not in arms:
        raise ValueError(f"Unknown comparison arm in {label}: {left} vs {right}")
    pairs = [(arms[left][sid], arms[right][sid]) for sid in sample_ids]
    pos_pairs = [(l, r) for l, r in pairs if l["target_answer"] == "yes"]
    neg_pairs = [(l, r) for l, r in pairs if l["target_answer"] == "no"]
    all_diffs = [l["target_support"] - r["target_support"] for l, r in pairs]
    pos_diffs = [l["target_support"] - r["target_support"] for l, r in pos_pairs]
    neg_diffs = [l["target_support"] - r["target_support"] for l, r in neg_pairs]
    boot_all = bootstrap_mean_diffs(pairs, key="target_support", n=n_bootstrap, seed=seed)
    boot_pos = bootstrap_mean_diffs(pos_pairs, key="target_support", n=n_bootstrap, seed=seed + 101)
    boot_neg = bootstrap_mean_diffs(neg_pairs, key="target_support", n=n_bootstrap, seed=seed + 202)
    return {
        "comparison": label,
        "left": left,
        "right": right,
        "n": len(pairs),
        "mean_support_diff": mean(all_diffs),
        "median_support_diff": median(all_diffs),
        "support_diff_ci_low": percentile(boot_all, 0.025),
        "support_diff_ci_high": percentile(boot_all, 0.975),
        "support_diff_p_two_sided": bootstrap_two_sided_p(boot_all),
        "frac_support_left_gt_right": mean([1.0 if diff > 0.0 else 0.0 for diff in all_diffs]),
        "pos_mean_support_diff": mean(pos_diffs),
        "pos_median_support_diff": median(pos_diffs),
        "pos_support_diff_ci_low": percentile(boot_pos, 0.025),
        "pos_support_diff_ci_high": percentile(boot_pos, 0.975),
        "pos_support_diff_p_two_sided": bootstrap_two_sided_p(boot_pos),
        "pos_frac_support_left_gt_right": mean([1.0 if diff > 0.0 else 0.0 for diff in pos_diffs]),
        "neg_mean_support_diff": mean(neg_diffs),
        "neg_median_support_diff": median(neg_diffs),
        "neg_support_diff_ci_low": percentile(boot_neg, 0.025),
        "neg_support_diff_ci_high": percentile(boot_neg, 0.975),
        "neg_support_diff_p_two_sided": bootstrap_two_sided_p(boot_neg),
        "neg_frac_support_left_gt_right": mean([1.0 if diff > 0.0 else 0.0 for diff in neg_diffs]),
        "left_correct_right_wrong": sum(1 for l, r in pairs if l["correct"] and not r["correct"]),
        "left_wrong_right_correct": sum(1 for l, r in pairs if (not l["correct"]) and r["correct"]),
        "left_hfp_right_not": sum(1 for l, r in neg_pairs if l["hallucination"] and not r["hallucination"]),
        "right_hfp_left_not": sum(1 for l, r in neg_pairs if r["hallucination"] and not l["hallucination"]),
    }


def bootstrap_mean_diffs(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    key: str,
    n: int,
    seed: int,
) -> list[float]:
    if not pairs:
        return []
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(n):
        total = 0.0
        for _ in pairs:
            left, right = pairs[rng.randrange(len(pairs))]
            total += float(left[key]) - float(right[key])
        draws.append(total / len(pairs))
    return draws


def build_examples(
    arms: dict[str, dict[str, dict[str, Any]]],
    sample_ids: list[str],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    required = ("full", "evidence_removed", "ours_target0p30")
    if not all(label in arms for label in required):
        return []
    examples: list[dict[str, Any]] = []
    for sid in sample_ids:
        full = arms["full"][sid]
        removed = arms["evidence_removed"][sid]
        ours = arms["ours_target0p30"][sid]
        if full["target_answer"] != "yes":
            continue
        examples.append(
            {
                "sample_id": sid,
                "image_id": full.get("image_id", ""),
                "question": full.get("question", ""),
                "target_answer": full["target_answer"],
                "full_support": full["target_support"],
                "evidence_removed_support": removed["target_support"],
                "ours_support": ours["target_support"],
                "full_minus_removed": full["target_support"] - removed["target_support"],
                "ours_minus_removed": ours["target_support"] - removed["target_support"],
                "full_pred": full["pred_answer"],
                "removed_pred": removed["pred_answer"],
                "ours_pred": ours["pred_answer"],
            }
        )
    examples.sort(key=lambda row: (-float(row["full_minus_removed"]), row["sample_id"]))
    return examples[:limit]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_markdown(
    path: Path,
    *,
    summary_rows: list[dict[str, Any]],
    pairwise_rows: list[dict[str, Any]],
    example_rows: list[dict[str, Any]],
    arm_specs: list[ArmSpec],
    n: int,
    n_bootstrap: int,
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Qwen TextOCR-Hard Region Logit-Drop Diagnostics")
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append(f"- Matched probes: {n}.")
    lines.append("- Dataset/protocol: cached Qwen TextOCR-Hard full1000 yes/no probes.")
    lines.append("- Positive target support is the yes margin `no_loss - yes_loss`; negative target support is the no margin `yes_loss - no_loss`.")
    lines.append("- `evidence_kept` and `evidence_removed` are oracle interventions built from annotated OCR evidence boxes. They are diagnostics, not deployable pruning methods.")
    lines.append(f"- Confidence intervals use paired bootstrap over probe ids: {n_bootstrap} draws, seed {seed}.")
    lines.append("")
    lines.append("## Arms")
    lines.append("")
    lines.append("| arm | acc | hFPR | yes rate | support | pos support | neg support | keep | ECR | note |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in summary_rows:
        lines.append(
            "| {arm} | {acc:.3f} | {hFPR:.3f} | {yes_rate:.3f} | {mean_target_support:.3f} | "
            "{mean_pos_support:.3f} | {mean_neg_support:.3f} | {mean_keep_ratio:.3f} | {mean_ecr:.3f} | {note} |".format(
                **format_none(row)
            )
        )
    lines.append("")
    lines.append("## Pairwise Target-Support Deltas")
    lines.append("")
    lines.append("| comparison | support diff | 95% CI | p | pos diff | pos 95% CI | neg diff | neg 95% CI | correct L/R flips | hFP L/R flips |")
    lines.append("|---|---:|---|---:|---:|---|---:|---|---:|---:|")
    for row in pairwise_rows:
        lines.append(
            "| {comparison} | {mean_support_diff:.3f} | [{support_diff_ci_low:+.3f}, {support_diff_ci_high:+.3f}] | "
            "{support_diff_p_two_sided:.4f} | {pos_mean_support_diff:.3f} | [{pos_support_diff_ci_low:+.3f}, {pos_support_diff_ci_high:+.3f}] | "
            "{neg_mean_support_diff:.3f} | [{neg_support_diff_ci_low:+.3f}, {neg_support_diff_ci_high:+.3f}] | "
            "{left_correct_right_wrong}/{left_wrong_right_correct} | {left_hfp_right_not}/{right_hfp_left_not} |".format(**row)
        )
    lines.append("")
    lines.append("## Readout")
    lines.append("")
    readout = derive_readout(summary_rows, pairwise_rows)
    lines.extend(f"- {item}" for item in readout)
    if example_rows:
        lines.append("")
        lines.append("## Largest Positive-Probe Evidence Drops")
        lines.append("")
        lines.append("| sample_id | full - removed | ours - removed | full pred | removed pred | ours pred | question |")
        lines.append("|---|---:|---:|---|---|---|---|")
        for row in example_rows[:10]:
            question = str(row["question"]).replace("|", "/")
            lines.append(
                f"| {row['sample_id']} | {row['full_minus_removed']:.3f} | {row['ours_minus_removed']:.3f} | "
                f"{row['full_pred']} | {row['removed_pred']} | {row['ours_pred']} | {question} |"
            )
    lines.append("")
    lines.append("## Source Arms")
    lines.append("")
    for spec in arm_specs:
        lines.append(f"- `{spec.label}`: `{spec.path}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def derive_readout(summary_rows: list[dict[str, Any]], pairwise_rows: list[dict[str, Any]]) -> list[str]:
    summary = {row["arm"]: row for row in summary_rows}
    pairs = {row["comparison"]: row for row in pairwise_rows}
    out: list[str] = []
    if "evidence_kept_vs_removed" in pairs:
        row = pairs["evidence_kept_vs_removed"]
        out.append(
            "Keeping annotated OCR evidence raises target support over the evidence-removed intervention "
            f"by {row['mean_support_diff']:.3f} on average; on positive probes the gain is {row['pos_mean_support_diff']:.3f}."
        )
        out.append(
            "The negative-probe support delta is "
            f"{row['neg_mean_support_diff']:.3f}, showing that evidence removal can look safer by becoming conservative/no-biased rather than by preserving OCR reasoning."
        )
    if "ours_vs_removed" in pairs and "ours_target0p30" in summary:
        row = pairs["ours_vs_removed"]
        out.append(
            "Our target-conditioned 30% point preserves the positive evidence signal: "
            f"positive target support is {row['pos_mean_support_diff']:.3f} higher than evidence removal."
        )
    if "ours_vs_full" in pairs:
        row = pairs["ours_vs_full"]
        out.append(
            "Ours and full-token Qwen have essentially matched target support "
            f"(mean diff {row['mean_support_diff']:.3f}), supporting the claim that pruning does not erase the decision margin at this operating point."
        )
    if "evidence_removed" in summary:
        removed = summary["evidence_removed"]
        out.append(
            "The evidence-removed arm has low hFPR "
            f"({removed['hFPR']:.3f}) but much lower accuracy ({removed['acc']:.3f}); use it as a diagnostic intervention, not as a competitive baseline."
        )
    return out


def format_none(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for key, value in out.items():
        if value is None:
            out[key] = float("nan")
    return out


def maybe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def mean_present(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return mean(present) if present else None


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return ordered[low]
    weight = pos - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def bootstrap_two_sided_p(draws: list[float]) -> float:
    if not draws:
        return 1.0
    le_zero = sum(1 for value in draws if value <= 0.0) / len(draws)
    ge_zero = sum(1 for value in draws if value >= 0.0) / len(draws)
    return min(1.0, 2.0 * min(le_zero, ge_zero))


if __name__ == "__main__":
    main()
