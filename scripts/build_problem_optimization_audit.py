#!/usr/bin/env python
"""Build reviewer-risk audit tables motivated by problem.md.

This script only consumes cached probe-level results and probe construction
files. It does not run model inference.
"""

from __future__ import annotations

import csv
import json
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "problem_optimization_audit"
BOOTSTRAP = 10000
SEED = 20260716


@dataclass(frozen=True)
class RunSpec:
    key: str
    model: str
    method: str
    path: Path


@dataclass(frozen=True)
class PairSpec:
    name: str
    left: str
    right: str
    claim: str


RUNS = {
    "qwen_full": RunSpec(
        "qwen_full",
        "Qwen3-VL-8B",
        "Full",
        ROOT / "runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_topk_1p00",
    ),
    "qwen_target20": RunSpec(
        "qwen_target20",
        "Qwen3-VL-8B",
        "Target (20%)",
        ROOT / "runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_topk_0p20",
    ),
    "qwen_target30": RunSpec(
        "qwen_target30",
        "Qwen3-VL-8B",
        "Target (30%)",
        ROOT / "runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_topk_0p30_targetfix_802816",
    ),
    "qwen_random30": RunSpec(
        "qwen_random30",
        "Qwen3-VL-8B",
        "Random (30%)",
        ROOT / "runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_random_0p30",
    ),
    "qwen_grid30": RunSpec(
        "qwen_grid30",
        "Qwen3-VL-8B",
        "Grid (30%)",
        ROOT / "runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_grid_0p30",
    ),
    "llava_full": RunSpec(
        "llava_full",
        "LLaVA-1.5-7B",
        "Full",
        ROOT / "runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_direct",
    ),
    "llava_protected40": RunSpec(
        "llava_protected40",
        "LLaVA-1.5-7B",
        "Protected (40%)",
        ROOT / "runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_embed_protected_topk_0p40",
    ),
    "llava_visionzip40": RunSpec(
        "llava_visionzip40",
        "LLaVA-1.5-7B",
        "VisionZip port (40%)",
        ROOT / "runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_visionzip_0p40",
    ),
    "internvl_full": RunSpec(
        "internvl_full",
        "InternVL3.5-8B",
        "Full calibrated",
        ROOT / "runs/internvl_textocr_hard/calibrated_test_full_devthr",
    ),
    "internvl_soft_hfpr": RunSpec(
        "internvl_soft_hfpr",
        "InternVL3.5-8B",
        "Soft evidence (50%, risk-selected)",
        ROOT / "runs/internvl_textocr_hard/calibrated_test_target_soft_evidence0p50_b0p05_hfprconstr_devthr",
    ),
    "internvl_soft_default": RunSpec(
        "internvl_soft_default",
        "InternVL3.5-8B",
        "Soft evidence (50%, default)",
        ROOT / "runs/internvl_textocr_hard/calibrated_test_target_soft_evidence0p50_b0p05_devthr",
    ),
    "internvl_grid50": RunSpec(
        "internvl_grid50",
        "InternVL3.5-8B",
        "Grid (50%)",
        ROOT / "runs/internvl_textocr_hard/calibrated_test_grid0p50_devthr",
    ),
}

PAIRS = [
    PairSpec("qwen_target30_vs_full", "qwen_target30", "qwen_full", "full-prefix parity"),
    PairSpec("qwen_target30_vs_random30", "qwen_target30", "qwen_random30", "same-budget random"),
    PairSpec("qwen_target30_vs_grid30", "qwen_target30", "qwen_grid30", "same-budget grid"),
    PairSpec("llava_protected40_vs_full", "llava_protected40", "llava_full", "full-prefix comparison"),
    PairSpec("llava_protected40_vs_visionzip40", "llava_protected40", "llava_visionzip40", "external-method comparison"),
    PairSpec("internvl_soft_default_vs_full", "internvl_soft_default", "internvl_full", "symmetric-calibration full-prefix comparison"),
    PairSpec("internvl_soft_default_vs_grid50", "internvl_soft_default", "internvl_grid50", "symmetric-calibration same-budget grid"),
    PairSpec("internvl_soft_hfpr_vs_soft_default", "internvl_soft_hfpr", "internvl_soft_default", "operating-point calibration"),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    loaded = {key: load_run(spec) for key, spec in RUNS.items()}

    cluster_rows = [cluster_compare(pair, loaded[pair.left], loaded[pair.right]) for pair in PAIRS]
    hard_rows, hard_summary = hard_negative_quality()
    random_rows = random_seed_status()
    internvl_rows = internvl_operating_point_notes(loaded)

    write_csv(OUT_DIR / "image_cluster_pairwise_stats.csv", cluster_rows)
    write_csv(OUT_DIR / "hard_negative_quality_examples.csv", hard_rows)
    write_csv(OUT_DIR / "hard_negative_quality_summary.csv", hard_summary)
    write_csv(OUT_DIR / "random_seed_status.csv", random_rows)
    write_csv(OUT_DIR / "internvl_operating_point_notes.csv", internvl_rows)
    (OUT_DIR / "problem_optimization_audit.md").write_text(
        markdown(cluster_rows, hard_summary, random_rows, internvl_rows),
        encoding="utf-8",
    )
    print(f"Wrote {OUT_DIR / 'problem_optimization_audit.md'}")


def load_run(spec: RunSpec) -> dict[str, Any]:
    scores = read_jsonl(spec.path / "probe_scores.jsonl")
    by_id = {row["sample_id"]: row for row in scores}
    return {"spec": spec, "scores": scores, "by_id": by_id}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def cluster_compare(pair: PairSpec, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_by_id = left["by_id"]
    right_by_id = right["by_id"]
    common = sorted(set(left_by_id) & set(right_by_id))
    acc_by_image: dict[str, list[float]] = defaultdict(list)
    hfpr_by_image: dict[str, list[float]] = defaultdict(list)
    ecr_by_image: dict[str, list[float]] = defaultdict(list)
    keep_by_image: dict[str, list[float]] = defaultdict(list)
    for sample_id in common:
        lrow = left_by_id[sample_id]
        rrow = right_by_id[sample_id]
        image_id = str(lrow.get("image_id") or sample_id.split(":")[0])
        acc_by_image[image_id].append(float(bool(lrow.get("correct"))) - float(bool(rrow.get("correct"))))
        if lrow.get("binary_polarity") == "negative":
            hfpr_by_image[image_id].append(float(lrow.get("pred_answer") == "yes") - float(rrow.get("pred_answer") == "yes"))
        ecr_by_image[image_id].append(ecr_value(lrow) - ecr_value(rrow))
        keep_by_image[image_id].append(keep_value(lrow))

    acc_clusters = cluster_means(acc_by_image)
    hfpr_clusters = cluster_means(hfpr_by_image)
    ecr_clusters = cluster_means(ecr_by_image)
    keep_clusters = cluster_means(keep_by_image)
    return {
        "comparison": pair.name,
        "claim": pair.claim,
        "model": left["spec"].model,
        "left": left["spec"].method,
        "right": right["spec"].method,
        "n_probes": len(common),
        "n_images": len(acc_clusters),
        "acc_diff": fmt(mean(acc_clusters)),
        "acc_cluster_ci": ci_text(cluster_bootstrap(acc_clusters, BOOTSTRAP, SEED + len(pair.name))),
        "hFPR_diff": fmt(mean(hfpr_clusters)),
        "hFPR_cluster_ci": ci_text(cluster_bootstrap(hfpr_clusters, BOOTSTRAP, SEED + 17 + len(pair.name))),
        "ECR_diff": fmt(mean(ecr_clusters)),
        "ECR_cluster_ci": ci_text(cluster_bootstrap(ecr_clusters, BOOTSTRAP, SEED + 31 + len(pair.name))),
        "left_mean_keep": fmt(mean(keep_clusters)),
        "bootstrap_unit": "image_id",
    }


def cluster_means(values_by_cluster: dict[str, list[float]]) -> list[float]:
    return [mean(values) for values in values_by_cluster.values() if values]


def cluster_bootstrap(values: list[float], n: int, seed: int) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    draws = []
    for _ in range(n):
        sample = [values[rng.randrange(len(values))] for _ in values]
        draws.append(mean(sample))
    draws.sort()
    return draws[int(0.025 * n)], draws[int(0.975 * n)]


def hard_negative_quality() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    probe_path = ROOT / "data/textocr_val_hard_probes_500img.jsonl"
    region_path = ROOT / "data/textocr_val_regions_500.jsonl"
    probes = read_jsonl(probe_path)
    regions = {row["image_id"]: row for row in read_jsonl(region_path)}
    rows = []
    distances = []
    norm_distances = []
    edit_types = Counter()
    collision_count = 0
    for probe in probes:
        if probe.get("binary_polarity") != "negative":
            continue
        source = normalize(probe.get("source_text", ""))
        target = normalize(probe.get("target_text", ""))
        dist = levenshtein(source.casefold(), target.casefold())
        distances.append(dist)
        norm_distances.append(dist / max(1, len(source)))
        edit_type = classify_edit(source, target, dist)
        edit_types[edit_type] += 1
        present = present_tokens(regions.get(probe.get("image_id", ""), {}))
        collision = target.casefold() in present
        collision_count += int(collision)
        rows.append(
            {
                "sample_id": probe.get("sample_id", ""),
                "image_id": probe.get("image_id", ""),
                "source_text": source,
                "target_text": target,
                "source_len": len(source),
                "target_len": len(target),
                "edit_distance": dist,
                "normalized_edit_distance": fmt(dist / max(1, len(source))),
                "edit_type": edit_type,
                "target_present_in_same_image_ocr": int(collision),
                "token_area": fmt(probe.get("token_area", "")),
                "token_area_rank": probe.get("token_area_rank", ""),
            }
        )
    summary = [
        {"metric": "negative_probe_count", "value": len(rows)},
        {"metric": "image_count", "value": len({row["image_id"] for row in rows})},
        {"metric": "mean_edit_distance", "value": fmt(mean(distances))},
        {"metric": "median_edit_distance", "value": fmt(statistics.median(distances))},
        {"metric": "mean_normalized_edit_distance", "value": fmt(mean(norm_distances))},
        {"metric": "target_present_in_same_image_ocr_count", "value": collision_count},
        {"metric": "target_present_in_same_image_ocr_rate", "value": fmt(collision_count / max(1, len(rows)))},
    ]
    for edit_type, count in sorted(edit_types.items()):
        summary.append({"metric": f"edit_type_{edit_type}_count", "value": count})
        summary.append({"metric": f"edit_type_{edit_type}_rate", "value": fmt(count / max(1, len(rows)))})
    return rows, summary


def random_seed_status() -> list[dict[str, Any]]:
    random_groups = [
        (
            "qwen_random20",
            [
                ROOT / "runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_random_0p20",
            ],
        ),
        (
            "qwen_random30",
            sorted(
                ROOT.glob(
                    "runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_random_0p30*"
                )
            ),
        ),
        (
            "llava_random40",
            sorted(ROOT.glob("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_random_0p40*")),
        ),
        (
            "internvl_random50",
            sorted(ROOT.glob("runs/internvl_textocr_hard/calibrated_test_random0p50*_devthr")),
        ),
    ]
    rows = []
    for group, paths in random_groups:
        valid_paths = [path for path in paths if (path / "probe_scores.jsonl").exists()]
        per_seed = [random_run_metrics(path) for path in valid_paths]
        first = per_seed[0] if per_seed else {}
        status = (
            "multi-seed random summarized"
            if len(per_seed) >= 2
            else "single cached realization; multi-seed random still pending"
        )
        rows.append(
            {
                "run": group,
                "paths": ";".join(str(path.relative_to(ROOT)) for path in valid_paths),
                "cached_random_realizations": len(per_seed),
                "n": int(first.get("n", 0)),
                "acc": fmt(first.get("acc", 0.0)),
                "hFPR": fmt(first.get("hFPR", 0.0)),
                "keep_ratio": fmt(first.get("keep_ratio", 0.0)),
                "ECR": fmt(first.get("ECR", 0.0)),
                "acc_mean": fmt(metric_mean(per_seed, "acc")),
                "acc_std": fmt(metric_std(per_seed, "acc")),
                "hFPR_mean": fmt(metric_mean(per_seed, "hFPR")),
                "hFPR_std": fmt(metric_std(per_seed, "hFPR")),
                "keep_mean": fmt(metric_mean(per_seed, "keep_ratio")),
                "keep_std": fmt(metric_std(per_seed, "keep_ratio")),
                "ECR_mean": fmt(metric_mean(per_seed, "ECR")),
                "ECR_std": fmt(metric_std(per_seed, "ECR")),
                "status": status,
            }
        )
    return rows


def random_run_metrics(path: Path) -> dict[str, float]:
    scores = read_jsonl(path / "probe_scores.jsonl")
    neg = [row for row in scores if row.get("binary_polarity") == "negative"]
    return {
        "n": float(len(scores)),
        "acc": mean([float(bool(row.get("correct"))) for row in scores]),
        "hFPR": mean([float(row.get("pred_answer") == "yes") for row in neg]),
        "keep_ratio": mean([keep_value(row) for row in scores]),
        "ECR": mean([ecr_value(row) for row in scores]),
    }


def metric_mean(rows: list[dict[str, float]], key: str) -> float:
    return mean([row[key] for row in rows]) if rows else 0.0


def metric_std(rows: list[dict[str, float]], key: str) -> float:
    values = [row[key] for row in rows]
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def internvl_operating_point_notes(loaded: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for key in ["internvl_soft_hfpr", "internvl_soft_default", "internvl_full", "internvl_grid50"]:
        run = loaded[key]
        scores = run["scores"]
        neg = [row for row in scores if row.get("binary_polarity") == "negative"]
        rows.append(
            {
                "run": key,
                "method": run["spec"].method,
                "n": len(scores),
                "acc": fmt(mean([float(bool(row.get("correct"))) for row in scores])),
                "hFPR": fmt(mean([float(row.get("pred_answer") == "yes") for row in neg])),
                "keep_ratio": fmt(mean([keep_value(row) for row in scores])),
                "ECR": fmt(mean([ecr_value(row) for row in scores])),
                "note": internvl_note(key),
            }
        )
    return rows


def internvl_note(key: str) -> str:
    if key == "internvl_soft_hfpr":
        return "risk-selected operating point; same token selector family but decision/threshold selected for lower hard-negative false positives"
    if key == "internvl_soft_default":
        return "default calibrated soft-evidence row; useful to expose calibration sensitivity"
    if key == "internvl_full":
        return "full-prefix calibrated reference"
    return "same-budget spatial baseline"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def keep_value(row: dict[str, Any]) -> float:
    if "prune_keep_ratio" in row:
        return float(row.get("prune_keep_ratio", 1.0) or 1.0)
    return 1.0


def ecr_value(row: dict[str, Any]) -> float:
    if "prune_ecr" in row:
        return float(row.get("prune_ecr", 0.0) or 0.0)
    # Full-prefix runs keep every visual token, so the annotated evidence region
    # is fully available even though no pruning trace field is emitted.
    return 1.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown(
    cluster_rows: list[dict[str, Any]],
    hard_summary: list[dict[str, Any]],
    random_rows: list[dict[str, Any]],
    internvl_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# problem.md Optimization Audit",
        "",
        "This audit targets low-cost reviewer risks identified in `problem.md`: image-level statistical dependence, hard-negative construction quality, single-realization random controls, and InternVL operating-point ambiguity.",
        "",
        "## Image-Cluster Paired Statistics",
        "",
        "Bootstrap confidence intervals resample `image_id` clusters rather than treating the two probes from each image as independent.",
        "",
        md_table(cluster_rows),
        "",
        "## Hard-Negative Construction Quality",
        "",
        md_table(hard_summary),
        "",
        "## Random Baseline Seed Status",
        "",
        md_table(random_rows),
        "",
        "## InternVL Soft-Evidence Operating Points",
        "",
        md_table(internvl_rows),
        "",
        "## Immediate Paper Implication",
        "",
        "- Use image-cluster CIs when discussing TextOCR-Hard paired significance.",
        "- State that current random baselines are single cached realizations unless multi-seed runs are added.",
        "- Report the hard-negative collision rate with same-image OCR tokens; zero collisions supports near-miss validity.",
        "- Clarify that InternVL soft-evidence rows are different operating points/threshold choices, not different evidence coverage.",
    ]
    return "\n".join(lines) + "\n"


def md_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    cols = list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(escape_md(row.get(col, "")) for col in cols) + " |")
    return "\n".join(out)


def escape_md(value: Any) -> str:
    return str(value).replace("|", "\\|")


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def ci_text(ci: tuple[float, float]) -> str:
    return f"[{fmt(ci[0])},{fmt(ci[1])}]"


def fmt(value: Any) -> str:
    if value == "":
        return ""
    try:
        numeric = float(value)
        if abs(numeric) < 0.0005:
            numeric = 0.0
        return f"{numeric:.3f}"
    except Exception:
        return str(value)


def normalize(value: Any) -> str:
    return str(value or "").strip()


def present_tokens(region_row: dict[str, Any]) -> set[str]:
    tokens = set()
    for item in region_row.get("ocr_tokens", []):
        if isinstance(item, dict):
            text = normalize(item.get("text", "")).strip("\"'`.,;:!?()[]{}")
            if text:
                tokens.add(text.casefold())
    return tokens


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + int(ca != cb)))
        prev = curr
    return prev[-1]


def classify_edit(source: str, target: str, dist: int) -> str:
    if dist == 0:
        return "identical"
    if len(source) == len(target) and dist == 1:
        return "single_substitution"
    if len(source) == len(target) + 1 and dist == 1:
        return "single_deletion"
    if len(source) + 1 == len(target) and dist == 1:
        return "single_insertion"
    if dist == 1:
        return "single_edit_other"
    return "multi_edit_or_confusable_sequence"


if __name__ == "__main__":
    main()
