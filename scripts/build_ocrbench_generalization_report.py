#!/usr/bin/env python
"""Summarize OCRBench answer-verification generalization runs."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "ocrbench_generalization"
INPUT_PATH = ROOT / "data" / "ocrbench_yesno_probes_100img.jsonl"
META_PATH = ROOT / "data" / "ocrbench_yesno_probes_100img.meta.json"
VALIDATION_PATH = OUT_DIR / "ocrbench100_image_validation.json"


@dataclass(frozen=True)
class RunSpec:
    model: str
    method: str
    ratio: float | str
    path: Path
    implementation: str
    note: str


RUN_SPECS = [
    RunSpec("Qwen3-VL-8B", "direct", "full", Path("runs/ocrbench_generalization/qwen3_8b_ocrbench100_direct"), "full", "unpruned likelihood baseline"),
    RunSpec("Qwen3-VL-8B", "target_embed_topk", 0.20, Path("runs/ocrbench_generalization/qwen3_8b_ocrbench100_target_embed_topk_0p20"), "ours/proxy", "target-conditioned visual token salience"),
    RunSpec("Qwen3-VL-8B", "target_embed_topk", 0.30, Path("runs/ocrbench_generalization/qwen3_8b_ocrbench100_target_embed_topk_0p30"), "ours/proxy", "target-conditioned visual token salience"),
    RunSpec("Qwen3-VL-8B", "target_embed_grid_topk", 0.30, Path("runs/ocrbench_generalization/qwen3_8b_ocrbench100_target_embed_grid_topk_0p30"), "ours OCR-safe", "50% grid floor, then target-conditioned salience fill"),
    RunSpec("Qwen3-VL-8B", "target_embed_grid_topk", 0.40, Path("runs/ocrbench_generalization/qwen3_8b_ocrbench100_target_embed_grid_topk_0p40"), "ours OCR-safe", "50% grid floor, then target-conditioned salience fill"),
    RunSpec("Qwen3-VL-8B", "target_embed_grid_topk", 0.50, Path("runs/ocrbench_generalization/qwen3_8b_ocrbench100_target_embed_grid_topk_0p50"), "ours OCR-safe", "50% grid floor, then target-conditioned salience fill"),
    RunSpec("Qwen3-VL-8B", "grid", 0.30, Path("runs/ocrbench_generalization/qwen3_8b_ocrbench100_grid_0p30"), "spatial baseline", "uniform grid keep baseline"),
    RunSpec("Qwen3-VL-8B", "random", 0.30, Path("runs/ocrbench_generalization/qwen3_8b_ocrbench100_random_0p30"), "random baseline", "random keep baseline"),
    RunSpec("LLaVA-1.5-7B", "direct", "full", Path("runs/ocrbench_generalization/llava15_7b_ocrbench100_direct"), "full", "unpruned likelihood baseline"),
    RunSpec("LLaVA-1.5-7B", "embed_protected_topk", 0.40, Path("runs/ocrbench_generalization/llava15_7b_ocrbench100_embed_protected_topk_0p40"), "ours/proxy", "evidence protection is inactive on OCRBench because no boxes are available"),
    RunSpec("LLaVA-1.5-7B", "grid", 0.40, Path("runs/ocrbench_generalization/llava15_7b_ocrbench100_grid_0p40"), "spatial baseline", "uniform grid keep baseline"),
    RunSpec("LLaVA-1.5-7B", "VisionZip", 0.40, Path("runs/ocrbench_generalization/llava15_7b_ocrbench100_visionzip_0p40"), "official-algorithm port", "LLaVA backend method-specific port"),
    RunSpec("LLaVA-1.5-7B", "FastV", 0.40, Path("runs/ocrbench_generalization/llava15_7b_ocrbench100_fastv_0p40"), "official-algorithm port", "LLaVA backend method-specific port"),
    RunSpec("InternVL3.5-8B", "direct", "full", Path("runs/ocrbench_generalization/internvl35_8b_ocrbench100_direct"), "full", "unpruned likelihood baseline"),
    RunSpec("InternVL3.5-8B", "target_embed_topk", 0.50, Path("runs/ocrbench_generalization/internvl35_8b_ocrbench100_target_embed_topk_0p50"), "ours/proxy", "target-conditioned visual token salience"),
    RunSpec("InternVL3.5-8B", "target_embed_grid_topk", 0.50, Path("runs/ocrbench_generalization/internvl35_8b_ocrbench100_target_embed_grid_topk_0p50"), "ours OCR-safe", "50% grid floor, then target-conditioned salience fill"),
    RunSpec("InternVL3.5-8B", "target_embed_grid_topk", 0.60, Path("runs/ocrbench_generalization/internvl35_8b_ocrbench100_target_embed_grid_topk_0p60"), "ours OCR-safe", "50% grid floor, then target-conditioned salience fill"),
    RunSpec("InternVL3.5-8B", "embed_topk", 0.50, Path("runs/ocrbench_generalization/internvl35_8b_ocrbench100_embed_topk_0p50"), "protocol proxy", "generic embedding salience baseline; not an official external method"),
    RunSpec("InternVL3.5-8B", "grid", 0.50, Path("runs/ocrbench_generalization/internvl35_8b_ocrbench100_grid_0p50"), "spatial baseline", "uniform grid keep baseline"),
]


SUMMARY_COLUMNS = [
    "model",
    "method",
    "ratio",
    "implementation",
    "status",
    "n",
    "n_images",
    "acc",
    "hFPR",
    "pos_acc",
    "neg_acc",
    "yes_rate",
    "keep_ratio",
    "kept_visual",
    "full_visual",
    "ECR",
    "path",
    "note",
]

TYPE_COLUMNS = [
    "model",
    "method",
    "ratio",
    "question_type",
    "n",
    "acc",
    "hFPR",
    "pos_acc",
    "neg_acc",
    "yes_rate",
]

PAIR_COLUMNS = [
    "model",
    "method",
    "ratio",
    "n",
    "n_negative",
    "acc_diff_vs_direct",
    "acc_sign_p",
    "hFPR_diff_vs_direct",
    "hFPR_sign_p",
    "candidate_better_acc",
    "candidate_worse_acc",
    "candidate_worse_hFPR",
    "candidate_better_hFPR",
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    probe_meta = load_probe_meta(INPUT_PATH)
    summary_rows = [row_for_run(spec, probe_meta) for spec in RUN_SPECS]
    type_rows = build_type_rows(summary_rows, probe_meta)
    pair_rows = build_pair_rows()
    write_csv(OUT_DIR / "ocrbench_generalization_summary.csv", summary_rows, SUMMARY_COLUMNS)
    write_csv(OUT_DIR / "ocrbench_generalization_by_type.csv", type_rows, TYPE_COLUMNS)
    write_csv(OUT_DIR / "ocrbench_generalization_pairwise.csv", pair_rows, PAIR_COLUMNS)
    (OUT_DIR / "ocrbench_generalization_report.md").write_text(
        build_markdown(summary_rows, type_rows, pair_rows),
        encoding="utf-8",
    )
    print(f"Wrote {OUT_DIR / 'ocrbench_generalization_report.md'}")
    print(f"Wrote {OUT_DIR / 'ocrbench_generalization_summary.csv'}")
    print(f"Wrote {OUT_DIR / 'ocrbench_generalization_by_type.csv'}")


def row_for_run(spec: RunSpec, probe_meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
    run_dir = ROOT / spec.path
    metrics_path = run_dir / "metrics.json"
    scores_path = run_dir / "probe_scores.jsonl"
    traces_path = run_dir / "prune_traces.jsonl"
    row: dict[str, Any] = {
        "model": spec.model,
        "method": spec.method,
        "ratio": spec.ratio,
        "implementation": spec.implementation,
        "status": "done" if metrics_path.exists() and scores_path.exists() else "pending",
        "n": "",
        "n_images": "",
        "acc": "",
        "hFPR": "",
        "pos_acc": "",
        "neg_acc": "",
        "yes_rate": "",
        "keep_ratio": 1.0 if spec.method == "direct" else "",
        "kept_visual": "",
        "full_visual": "",
        "ECR": "",
        "path": str(spec.path),
        "note": spec.note,
    }
    if row["status"] != "done":
        return row
    scores = attach_probe_meta(read_jsonl(scores_path), probe_meta)
    metrics = read_json(metrics_path)
    traces = read_jsonl(traces_path) if traces_path.exists() else []
    row.update(summarize_scores(scores, metrics))
    row.update(summarize_pruning(scores, traces, direct=spec.method == "direct"))
    return row


def summarize_scores(rows: list[dict[str, Any]], metrics: dict[str, Any]) -> dict[str, Any]:
    pos = [row for row in rows if target_answer(row) == "yes"]
    neg = [row for row in rows if target_answer(row) == "no"]
    return {
        "n": len(rows),
        "n_images": len({str(row.get("image_id", row.get("image", ""))) for row in rows if row.get("image_id") or row.get("image")}),
        "acc": float(metrics.get("direct_accuracy", mean_bool(row.get("correct") for row in rows))),
        "hFPR": float(metrics.get("direct_hallucination_fpr", mean_bool(row.get("pred_answer") == "yes" for row in neg))),
        "pos_acc": mean_bool(row.get("correct") for row in pos),
        "neg_acc": mean_bool(row.get("correct") for row in neg),
        "yes_rate": mean_bool(row.get("pred_answer") == "yes" for row in rows),
    }


def summarize_pruning(rows: list[dict[str, Any]], traces: list[dict[str, Any]], *, direct: bool) -> dict[str, Any]:
    if direct:
        return {"keep_ratio": 1.0, "kept_visual": "", "full_visual": "", "ECR": ""}
    if traces:
        wanted = {(str(row.get("sample_id", "")), str(row.get("probe", ""))) for row in rows}
        trace_rows = [
            row
            for row in traces
            if not wanted or (str(row.get("sample_id", "")), str(row.get("probe", ""))) in wanted
        ]
        evidence_rows = [row for row in trace_rows if row.get("has_evidence")]
        full = mean_number(row.get("full_visual_tokens") for row in trace_rows)
        kept = mean_number(row.get("kept_visual_tokens") for row in trace_rows)
        return {
            "keep_ratio": kept / full if is_number(full) and is_number(kept) and full else "",
            "kept_visual": kept,
            "full_visual": full,
            "ECR": mean_number(row.get("ecr") for row in evidence_rows),
        }
    evidence_scores = [row for row in rows if is_number(metric(row, "ecr")) and bool(row.get("has_evidence", False))]
    return {
        "keep_ratio": mean_number(metric(row, "keep_ratio") for row in rows),
        "kept_visual": mean_number(metric(row, "kept_visual_tokens") for row in rows),
        "full_visual": mean_number(metric(row, "full_visual_tokens") for row in rows),
        "ECR": mean_number(metric(row, "ecr") for row in evidence_scores),
    }


def build_type_rows(summary_rows: list[dict[str, Any]], probe_meta: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    done_specs = [spec for spec, row in zip(RUN_SPECS, summary_rows) if row["status"] == "done"]
    for spec in done_specs:
        scores = attach_probe_meta(read_jsonl(ROOT / spec.path / "probe_scores.jsonl"), probe_meta)
        by_type: dict[str, list[dict[str, Any]]] = {}
        for row in scores:
            question_type = str(row.get("ocrbench_question_type", "unknown"))
            by_type.setdefault(question_type, []).append(row)
        for question_type, rows in sorted(by_type.items()):
            stats = summarize_scores(rows, {})
            out.append(
                {
                    "model": spec.model,
                    "method": spec.method,
                    "ratio": spec.ratio,
                    "question_type": question_type,
                    "n": stats["n"],
                    "acc": stats["acc"],
                    "hFPR": stats["hFPR"],
                    "pos_acc": stats["pos_acc"],
                    "neg_acc": stats["neg_acc"],
                    "yes_rate": stats["yes_rate"],
                }
            )
    return out


def build_pair_rows() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    specs_by_model: dict[str, list[RunSpec]] = {}
    for spec in RUN_SPECS:
        specs_by_model.setdefault(spec.model, []).append(spec)
    for model, specs in specs_by_model.items():
        direct_specs = [spec for spec in specs if spec.method == "direct"]
        if not direct_specs:
            continue
        direct = direct_specs[0]
        direct_scores = load_score_map(ROOT / direct.path / "probe_scores.jsonl")
        if not direct_scores:
            continue
        for spec in specs:
            if spec.method == "direct":
                continue
            cand_scores = load_score_map(ROOT / spec.path / "probe_scores.jsonl")
            if not cand_scores:
                continue
            out.append(compare_pair(model, spec, cand_scores, direct_scores))
    return out


def compare_pair(
    model: str,
    spec: RunSpec,
    cand_scores: dict[str, dict[str, Any]],
    direct_scores: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    ids = sorted(set(cand_scores) & set(direct_scores))
    rows = [(cand_scores[sid], direct_scores[sid]) for sid in ids]
    negative_rows = [(crow, drow) for crow, drow in rows if target_answer(crow) == "no" or target_answer(drow) == "no"]
    cand_acc = mean_bool(crow.get("correct") for crow, _ in rows)
    direct_acc = mean_bool(drow.get("correct") for _, drow in rows)
    cand_hfpr = mean_bool(crow.get("pred_answer") == "yes" for crow, _ in negative_rows)
    direct_hfpr = mean_bool(drow.get("pred_answer") == "yes" for _, drow in negative_rows)
    acc_better, acc_worse = discordance(rows, "correct")
    hfpr_worse, hfpr_better = yes_discordance_on_negatives(negative_rows)
    return {
        "model": model,
        "method": spec.method,
        "ratio": spec.ratio,
        "n": len(rows),
        "n_negative": len(negative_rows),
        "acc_diff_vs_direct": cand_acc - direct_acc,
        "acc_sign_p": exact_sign_p(acc_better, acc_worse),
        "hFPR_diff_vs_direct": cand_hfpr - direct_hfpr,
        "hFPR_sign_p": exact_sign_p(hfpr_worse, hfpr_better),
        "candidate_better_acc": acc_better,
        "candidate_worse_acc": acc_worse,
        "candidate_worse_hFPR": hfpr_worse,
        "candidate_better_hFPR": hfpr_better,
    }


def build_markdown(
    summary_rows: list[dict[str, Any]],
    type_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
) -> str:
    meta = read_json(META_PATH) if META_PATH.exists() else {}
    validation = read_json(VALIDATION_PATH) if VALIDATION_PATH.exists() else {}
    done_rows = [row for row in summary_rows if row["status"] == "done"]
    lines = [
        "# OCRBench Generalization Check",
        "",
        "Scope: OCRBench answer-verification yes/no probes. This is a pruning-compatible robustness check, not OCRBench's native open-ended generation metric.",
        "",
        "## Dataset",
        "",
        f"- Source: `{meta.get('dataset_path', 'echo840/OCRBench')}` split `{meta.get('split', 'test')}`.",
        f"- Size: {meta.get('num_images', '')} images / {meta.get('num_probes', '')} probes; balanced positive/negative answer verification.",
        f"- Image validation: missing visual probes = {validation.get('missing_visual_count', '')}, missing rate = {fmt(validation.get('missing_visual_rate', ''))}.",
        "- Evidence note: OCRBench subset has no bbox/OCR regions, so ECR is intentionally blank; TextOCR remains the region-preservation benchmark.",
        "",
        "## Run Matrix",
        "",
        "| model | method | ratio | implementation | status | acc | hFPR | pos acc | neg acc | yes rate | keep | path |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            "| {model} | {method} | {ratio} | {impl} | {status} | {acc} | {hfpr} | {pos} | {neg} | {yes} | {keep} | `{path}` |".format(
                model=row["model"],
                method=row["method"],
                ratio=fmt(row["ratio"]),
                impl=row["implementation"],
                status=row["status"],
                acc=fmt(row["acc"]),
                hfpr=fmt(row["hFPR"]),
                pos=fmt(row["pos_acc"]),
                neg=fmt(row["neg_acc"]),
                yes=fmt(row["yes_rate"]),
                keep=fmt(row["keep_ratio"]),
                path=row["path"],
            )
        )

    lines.extend(["", "## Paired Deltas vs Direct", ""])
    if pair_rows:
        lines.extend(
            [
                "| model | method | ratio | n | acc diff | acc p | hFPR diff | hFPR p |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in pair_rows:
            lines.append(
                "| {model} | {method} | {ratio} | {n} | {acc} | {acc_p} | {hfpr} | {hfpr_p} |".format(
                    model=row["model"],
                    method=row["method"],
                    ratio=fmt(row["ratio"]),
                    n=row["n"],
                    acc=fmt_signed(row["acc_diff_vs_direct"]),
                    acc_p=fmt_p(row["acc_sign_p"]),
                    hfpr=fmt_signed(row["hFPR_diff_vs_direct"]),
                    hfpr_p=fmt_p(row["hFPR_sign_p"]),
                )
            )
    else:
        lines.append("No paired rows are available yet; run `scripts/run_ocrbench_generalization.sh` after GPU resources are available.")

    lines.extend(["", "## By-Type Output", ""])
    if type_rows:
        lines.append("Detailed per-question-type results are written to `runs/ocrbench_generalization/ocrbench_generalization_by_type.csv`.")
        preview = type_rows[: min(20, len(type_rows))]
        lines.extend(
            [
                "",
                "| model | method | ratio | question type | n | acc | hFPR |",
                "|---|---|---:|---|---:|---:|---:|",
            ]
        )
        for row in preview:
            lines.append(
                "| {model} | {method} | {ratio} | {typ} | {n} | {acc} | {hfpr} |".format(
                    model=row["model"],
                    method=row["method"],
                    ratio=fmt(row["ratio"]),
                    typ=row["question_type"],
                    n=row["n"],
                    acc=fmt(row["acc"]),
                    hfpr=fmt(row["hFPR"]),
                )
            )
    else:
        lines.append("Per-type rows will appear after the model runs finish.")

    pending = [row for row in summary_rows if row["status"] != "done"]
    lines.extend(
        [
            "",
            "## Readout",
            "",
            f"- Completed rows: {len(done_rows)} / {len(summary_rows)}.",
            f"- Pending rows: {len(pending)}.",
            "- LLaVA VisionZip/FastV rows are official-algorithm ports; InternVL external-style rows remain protocol proxies.",
            "- Use this table as an OCR domain-shift check alongside TextOCR-Hard, not as standalone OCRBench leaderboard evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def load_probe_meta(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out = {}
    for row in read_jsonl(path):
        out[str(row.get("sample_id", row.get("id", "")))] = row
    return out


def attach_probe_meta(rows: list[dict[str, Any]], probe_meta: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        meta = probe_meta.get(str(row.get("sample_id", "")), {})
        merged = dict(meta)
        merged.update(row)
        out.append(merged)
    return out


def load_score_map(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {str(row["sample_id"]): row for row in read_jsonl(path)}


def target_answer(row: dict[str, Any]) -> str:
    value = str(row.get("target_answer", row.get("answer", ""))).strip().lower()
    return "yes" if value in {"yes", "true", "1"} else "no"


def metric(row: dict[str, Any], key: str) -> Any:
    aliases = {
        "keep_ratio": ("prune_keep_ratio",),
        "ecr": ("prune_ecr",),
        "kept_visual_tokens": ("prune_kept_visual_tokens",),
        "full_visual_tokens": ("prune_full_visual_tokens",),
    }
    for name in aliases.get(key, (key, f"prune_{key}")):
        if name in row:
            return row[name]
    return ""


def discordance(rows: list[tuple[dict[str, Any], dict[str, Any]]], key: str) -> tuple[int, int]:
    cand_true_direct_false = 0
    direct_true_cand_false = 0
    for cand, direct in rows:
        cval = bool(cand.get(key, False))
        dval = bool(direct.get(key, False))
        if cval and not dval:
            cand_true_direct_false += 1
        elif dval and not cval:
            direct_true_cand_false += 1
    return cand_true_direct_false, direct_true_cand_false


def yes_discordance_on_negatives(rows: list[tuple[dict[str, Any], dict[str, Any]]]) -> tuple[int, int]:
    cand_worse = 0
    cand_better = 0
    for cand, direct in rows:
        c_yes = str(cand.get("pred_answer", "")) == "yes"
        d_yes = str(direct.get("pred_answer", "")) == "yes"
        if c_yes and not d_yes:
            cand_worse += 1
        elif d_yes and not c_yes:
            cand_better += 1
    return cand_worse, cand_better


def exact_sign_p(left: int, right: int) -> float:
    n = left + right
    if n == 0:
        return 1.0
    k = min(left, right)
    terms = [
        math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2.0)
        for i in range(k + 1)
    ]
    max_term = max(terms)
    tail = math.exp(max_term) * sum(math.exp(term - max_term) for term in terms)
    return min(1.0, 2.0 * tail)


def mean_bool(values: Any) -> float:
    vals = [bool(value) for value in values]
    return sum(1 for value in vals if value) / len(vals) if vals else 0.0


def mean_number(values: Any) -> float | str:
    vals = [float(value) for value in values if is_number(value)]
    return sum(vals) / len(vals) if vals else ""


def is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, str) and value.strip() != "":
        try:
            float(value)
            return True
        except ValueError:
            return False
    return False


def fmt(value: Any) -> str:
    if value == "" or value is None:
        return ""
    if is_number(value):
        return f"{float(value):.3f}"
    return str(value)


def fmt_signed(value: Any) -> str:
    if value == "" or value is None:
        return ""
    return f"{float(value):+.3f}"


def fmt_p(value: Any) -> str:
    if value == "" or value is None:
        return ""
    val = float(value)
    if val < 0.0001:
        return "<1e-4"
    return f"{val:.4f}"


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


if __name__ == "__main__":
    main()
