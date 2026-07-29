#!/usr/bin/env python
"""Summarize TextOCR-Hard input-space bbox occlusion diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recap.images import probe_to_visual


VIEWS = ("orig", "evidence_masked", "random_masked")
COMPARISONS = (
    ("orig_vs_evidence", "orig", "evidence_masked"),
    ("random_vs_evidence", "random_masked", "evidence_masked"),
    ("orig_vs_random", "orig", "random_masked"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--prefix", default="qwen_textocr_bbox_occlusion")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--top-examples", type=int, default=8)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "occlusion_report"
    examples_dir = out_dir / "examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    examples_dir.mkdir(parents=True, exist_ok=True)

    scores = load_scores(run_dir / "probe_scores.jsonl")
    probes = load_probes(run_dir / "probes.jsonl")
    common_ids = sorted(
        sample_id
        for sample_id, by_probe in scores.items()
        if all(view in by_probe for view in VIEWS)
    )
    if not common_ids:
        raise ValueError(f"No complete {VIEWS} sample groups found in {run_dir}")

    summary_rows = [summarize_view(view, scores, common_ids) for view in VIEWS]
    pairwise_rows = [
        compare_views(
            label=label,
            left=left,
            right=right,
            scores=scores,
            sample_ids=common_ids,
            n_bootstrap=args.bootstrap,
            seed=args.seed + index,
        )
        for index, (label, left, right) in enumerate(COMPARISONS)
    ]
    example_rows = build_examples(
        scores=scores,
        probes=probes,
        sample_ids=common_ids,
        examples_dir=examples_dir,
        limit=args.top_examples,
    )

    write_csv(out_dir / f"{args.prefix}_summary.csv", summary_rows)
    write_csv(out_dir / f"{args.prefix}_pairwise.csv", pairwise_rows)
    write_jsonl(out_dir / f"{args.prefix}_examples.jsonl", example_rows)
    write_markdown(
        out_dir / f"{args.prefix}_report.md",
        summary_rows=summary_rows,
        pairwise_rows=pairwise_rows,
        example_rows=example_rows,
        n=len(common_ids),
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    print(f"Wrote TextOCR bbox occlusion report to {out_dir}")


def load_scores(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            target = str(raw.get("target_answer", "")).strip().lower()
            if target not in {"yes", "no"}:
                continue
            margin = float(raw.get("margin", float(raw["no_loss"]) - float(raw["yes_loss"])))
            pred = str(raw.get("pred_answer", "")).strip().lower()
            if pred not in {"yes", "no"}:
                pred = "yes" if margin >= 0.0 else "no"
            target_support = margin if target == "yes" else -margin
            row = {
                "sample_id": str(raw["sample_id"]),
                "probe": str(raw["probe"]),
                "image_id": raw.get("image_id", ""),
                "question": raw.get("question", ""),
                "target_answer": target,
                "pred_answer": pred,
                "correct": bool(raw.get("correct", pred == target)),
                "hallucination": bool(target == "no" and pred == "yes"),
                "yes_margin": margin,
                "target_support": target_support,
                "yes_loss": float(raw.get("yes_loss", 0.0)),
                "no_loss": float(raw.get("no_loss", 0.0)),
                "source_text": raw.get("source_text", ""),
                "target_text": raw.get("target_text", ""),
                "image": raw.get("image", ""),
            }
            grouped.setdefault(row["sample_id"], {})[row["probe"]] = row
    return grouped


def load_probes(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            grouped.setdefault(str(row["sample_id"]), {})[str(row["probe"])] = row
    return grouped


def summarize_view(view: str, scores: dict[str, dict[str, dict[str, Any]]], sample_ids: list[str]) -> dict[str, Any]:
    rows = [scores[sid][view] for sid in sample_ids]
    pos = [row for row in rows if row["target_answer"] == "yes"]
    neg = [row for row in rows if row["target_answer"] == "no"]
    return {
        "view": view,
        "n": len(rows),
        "n_pos": len(pos),
        "n_neg": len(neg),
        "acc": mean([1.0 if row["correct"] else 0.0 for row in rows]),
        "pos_acc": mean([1.0 if row["correct"] else 0.0 for row in pos]),
        "neg_acc": mean([1.0 if row["correct"] else 0.0 for row in neg]),
        "hFPR": mean([1.0 if row["hallucination"] else 0.0 for row in neg]),
        "yes_rate": mean([1.0 if row["pred_answer"] == "yes" else 0.0 for row in rows]),
        "mean_target_support": mean([row["target_support"] for row in rows]),
        "median_target_support": median([row["target_support"] for row in rows]),
        "mean_pos_support": mean([row["target_support"] for row in pos]),
        "mean_neg_support": mean([row["target_support"] for row in neg]),
    }


def compare_views(
    *,
    label: str,
    left: str,
    right: str,
    scores: dict[str, dict[str, dict[str, Any]]],
    sample_ids: list[str],
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    pairs = [(scores[sid][left], scores[sid][right]) for sid in sample_ids]
    pos_pairs = [(lrow, rrow) for lrow, rrow in pairs if lrow["target_answer"] == "yes"]
    neg_pairs = [(lrow, rrow) for lrow, rrow in pairs if lrow["target_answer"] == "no"]
    all_diffs = [lrow["target_support"] - rrow["target_support"] for lrow, rrow in pairs]
    pos_diffs = [lrow["target_support"] - rrow["target_support"] for lrow, rrow in pos_pairs]
    neg_diffs = [lrow["target_support"] - rrow["target_support"] for lrow, rrow in neg_pairs]
    boot_all = bootstrap_mean_diffs(pairs, n=n_bootstrap, seed=seed)
    boot_pos = bootstrap_mean_diffs(pos_pairs, n=n_bootstrap, seed=seed + 101)
    boot_neg = bootstrap_mean_diffs(neg_pairs, n=n_bootstrap, seed=seed + 202)
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
        "pos_support_diff_ci_low": percentile(boot_pos, 0.025),
        "pos_support_diff_ci_high": percentile(boot_pos, 0.975),
        "pos_support_diff_p_two_sided": bootstrap_two_sided_p(boot_pos),
        "neg_mean_support_diff": mean(neg_diffs),
        "neg_support_diff_ci_low": percentile(boot_neg, 0.025),
        "neg_support_diff_ci_high": percentile(boot_neg, 0.975),
        "neg_support_diff_p_two_sided": bootstrap_two_sided_p(boot_neg),
        "left_correct_right_wrong": sum(1 for lrow, rrow in pairs if lrow["correct"] and not rrow["correct"]),
        "left_wrong_right_correct": sum(1 for lrow, rrow in pairs if (not lrow["correct"]) and rrow["correct"]),
        "left_hfp_right_not": sum(1 for lrow, rrow in neg_pairs if lrow["hallucination"] and not rrow["hallucination"]),
        "right_hfp_left_not": sum(1 for lrow, rrow in neg_pairs if rrow["hallucination"] and not lrow["hallucination"]),
    }


def build_examples(
    *,
    scores: dict[str, dict[str, dict[str, Any]]],
    probes: dict[str, dict[str, dict[str, Any]]],
    sample_ids: list[str],
    examples_dir: Path,
    limit: int,
) -> list[dict[str, Any]]:
    ranked: list[tuple[float, str]] = []
    for sid in sample_ids:
        by_probe = scores[sid]
        if by_probe["orig"]["target_answer"] != "yes":
            continue
        evidence_drop = by_probe["orig"]["target_support"] - by_probe["evidence_masked"]["target_support"]
        random_drop = by_probe["orig"]["target_support"] - by_probe["random_masked"]["target_support"]
        ranked.append((evidence_drop - max(0.0, random_drop), sid))
    ranked.sort(reverse=True)

    rows: list[dict[str, Any]] = []
    for rank, (_, sid) in enumerate(ranked[:limit], start=1):
        by_score = scores[sid]
        by_probe = probes.get(sid, {})
        try:
            triptych = make_triptych(by_probe, title=f"{sid} | {by_score['orig'].get('target_text', '')}")
        except Exception:
            continue
        out_path = examples_dir / f"{rank:02d}_{safe_name(sid)}.jpg"
        triptych.save(out_path, quality=90)
        rows.append(
            {
                "rank": rank,
                "sample_id": sid,
                "image_id": by_score["orig"].get("image_id", ""),
                "question": by_score["orig"].get("question", ""),
                "target_text": by_score["orig"].get("target_text", ""),
                "source_text": by_score["orig"].get("source_text", ""),
                "orig_support": by_score["orig"]["target_support"],
                "evidence_masked_support": by_score["evidence_masked"]["target_support"],
                "random_masked_support": by_score["random_masked"]["target_support"],
                "evidence_drop": by_score["orig"]["target_support"] - by_score["evidence_masked"]["target_support"],
                "random_drop": by_score["orig"]["target_support"] - by_score["random_masked"]["target_support"],
                "triptych": str(out_path),
            }
        )
    return rows


def make_triptych(by_probe: dict[str, dict[str, Any]], *, title: str) -> Image.Image:
    panels = [
        render_panel(probe_to_visual(by_probe["orig"], strict=True)[0], "original"),
        render_panel(probe_to_visual(by_probe["evidence_masked"], strict=True)[0], "evidence masked"),
        render_panel(probe_to_visual(by_probe["random_masked"], strict=True)[0], "random masked"),
    ]
    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels) + 32
    out = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(out)
    draw.text((8, 8), title[:140], fill=(0, 0, 0))
    x = 0
    for panel in panels:
        out.paste(panel, (x, 32))
        x += panel.width
    return out


def render_panel(image: Image.Image, label: str, size: int = 384) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail((size, size))
    panel = Image.new("RGB", (size, size + 28), "white")
    x = (size - image.width) // 2
    panel.paste(image, (x, 28))
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, size, 27), fill=(245, 245, 245))
    draw.text((8, 7), label, fill=(0, 0, 0))
    return panel


def write_markdown(
    path: Path,
    *,
    summary_rows: list[dict[str, Any]],
    pairwise_rows: list[dict[str, Any]],
    example_rows: list[dict[str, Any]],
    n: int,
    n_bootstrap: int,
    seed: int,
) -> None:
    lines = [
        "# TextOCR-Hard BBox Occlusion Diagnostics",
        "",
        f"- Matched base probes: {n}",
        f"- Bootstrap resamples: {n_bootstrap}",
        f"- Seed: {seed}",
        "- Positive target support is the yes margin; negative target support is the no margin.",
        "- This is an input-space causal-style diagnostic, not a deployable pruning method.",
        "",
        "## View Summary",
        "",
        table_md(
            summary_rows,
            [
                "view",
                "n",
                "acc",
                "hFPR",
                "yes_rate",
                "mean_target_support",
                "mean_pos_support",
                "mean_neg_support",
            ],
        ),
        "",
        "## Paired Logit-Drop Tests",
        "",
        table_md(
            pairwise_rows,
            [
                "comparison",
                "left",
                "right",
                "n",
                "mean_support_diff",
                "support_CI",
                "support_p",
                "pos_support_diff",
                "pos_support_CI",
                "neg_support_diff",
                "neg_support_CI",
                "correct_flip_L_R",
                "hFP_flip_L_R",
            ],
        ),
        "",
        "## Readout",
        "",
    ]
    pair_by_name = {row["comparison"]: row for row in pairwise_rows}
    if "orig_vs_evidence" in pair_by_name:
        row = pair_by_name["orig_vs_evidence"]
        lines.append(
            f"- Evidence masking changes target support by {fmt_signed(row['mean_support_diff'])} "
            f"overall: {fmt_signed(row['pos_mean_support_diff'])} on positive probes and "
            f"{fmt_signed(row['neg_mean_support_diff'])} on near-miss negative probes."
        )
    if "random_vs_evidence" in pair_by_name:
        row = pair_by_name["random_vs_evidence"]
        lines.append(
            f"- Random-masked support exceeds evidence-masked support by {fmt_signed(row['mean_support_diff'])} "
            "overall; this is the input-space locality control."
        )
    if "orig_vs_random" in pair_by_name:
        row = pair_by_name["orig_vs_random"]
        lines.append(
            f"- Original vs random-mask target support differs by {fmt_signed(row['mean_support_diff'])}; "
            "this is the main locality control."
        )
    lines.extend(
        [
            "",
            "## Example Triptychs",
            "",
            table_md(example_rows, ["rank", "sample_id", "target_text", "evidence_drop", "random_drop", "triptych"]),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def table_md(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    out = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        cells = []
        for col in columns:
            if col == "support_CI":
                cells.append(interval(row["support_diff_ci_low"], row["support_diff_ci_high"]))
            elif col == "pos_support_CI":
                cells.append(interval(row["pos_support_diff_ci_low"], row["pos_support_diff_ci_high"]))
            elif col == "neg_support_CI":
                cells.append(interval(row["neg_support_diff_ci_low"], row["neg_support_diff_ci_high"]))
            elif col == "support_p":
                cells.append(p(row["support_diff_p_two_sided"]))
            elif col == "pos_support_diff":
                cells.append(format_cell(row.get("pos_mean_support_diff", "")))
            elif col == "correct_flip_L_R":
                cells.append(f"{row['left_correct_right_wrong']}/{row['left_wrong_right_correct']}")
            elif col == "hFP_flip_L_R":
                cells.append(f"{row['left_hfp_right_not']}/{row['right_hfp_left_not']}")
            elif col == "neg_support_diff":
                cells.append(format_cell(row.get("neg_mean_support_diff", "")))
            else:
                cells.append(format_cell(row.get(col, "")))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def bootstrap_mean_diffs(pairs: list[tuple[dict[str, Any], dict[str, Any]]], *, n: int, seed: int) -> list[float]:
    if not pairs:
        return [0.0]
    rng = random.Random(seed)
    out = []
    size = len(pairs)
    for _ in range(n):
        total = 0.0
        for _ in range(size):
            left, right = pairs[rng.randrange(size)]
            total += left["target_support"] - right["target_support"]
        out.append(total / size)
    out.sort()
    return out


def bootstrap_two_sided_p(values: list[float]) -> float:
    if not values:
        return 1.0
    le = sum(1 for value in values if value <= 0.0)
    ge = sum(1 for value in values if value >= 0.0)
    return min(1.0, 2.0 * min(le, ge) / len(values))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    pos = q * (len(values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def interval(low: Any, high: Any) -> str:
    return f"[{fmt_signed(low)}, {fmt_signed(high)}]"


def p(value: Any) -> str:
    try:
        number = float(value)
    except Exception:
        return str(value)
    if number < 0.0001:
        return "<1e-4"
    return f"{number:.4f}"


def fmt_signed(value: Any) -> str:
    try:
        number = float(value)
    except Exception:
        return str(value)
    return f"{number:+.3f}"


def format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    text = str(value)
    return text.replace("|", "/")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:120]


if __name__ == "__main__":
    main()
