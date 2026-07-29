#!/usr/bin/env python
"""Build an official-baseline extension check for LLaVA and InternVL.

This report is intentionally conservative: an external method is marked as an
official-algorithm port only when the local backend contains a method-specific
path rather than a protocol-compatible proxy selector.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "official_baseline_extension"


@dataclass(frozen=True)
class RunSpec:
    model: str
    method: str
    ratio: float | str
    path: Path
    implementation: str
    note: str


LLAVA_OFFICIAL_RUNS = [
    RunSpec("LLaVA-1.5-7B", "VisionZip", 0.20, Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_visionzip_0p20"), "official-algorithm port", "CLS-attention dominant tokens plus contextual merge"),
    RunSpec("LLaVA-1.5-7B", "FastV", 0.20, Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_fastv_0p20"), "official-algorithm port", "layer-attention pruning at K=3"),
    RunSpec("LLaVA-1.5-7B", "VisionZip", 0.30, Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_visionzip_0p30"), "official-algorithm port", "CLS-attention dominant tokens plus contextual merge"),
    RunSpec("LLaVA-1.5-7B", "FastV", 0.30, Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_fastv_0p30"), "official-algorithm port", "layer-attention pruning at K=3"),
    RunSpec("LLaVA-1.5-7B", "VisionZip", 0.40, Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_visionzip_0p40"), "official-algorithm port", "CLS-attention dominant tokens plus contextual merge"),
    RunSpec("LLaVA-1.5-7B", "FastV", 0.40, Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_fastv_0p40"), "official-algorithm port", "layer-attention pruning at K=3"),
    RunSpec("LLaVA-1.5-7B", "VisionZip", 0.50, Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_visionzip_0p50"), "official-algorithm port", "CLS-attention dominant tokens plus contextual merge"),
    RunSpec("LLaVA-1.5-7B", "FastV", 0.50, Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_fastv_0p50"), "official-algorithm port", "layer-attention pruning at K=3"),
]


LLAVA_OURS_RUNS = [
    RunSpec("LLaVA-1.5-7B", "ours_protected_embed", 0.20, Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_embed_protected_topk_0p20"), "ours", "same-budget evidence-protected comparator"),
    RunSpec("LLaVA-1.5-7B", "ours_protected_embed", 0.30, Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_embed_protected_topk_0p30"), "ours", "same-budget evidence-protected comparator"),
    RunSpec("LLaVA-1.5-7B", "ours_protected_embed", 0.40, Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_embed_protected_topk_0p40"), "ours", "same-budget evidence-protected comparator"),
    RunSpec("LLaVA-1.5-7B", "ours_protected_embed", 0.50, Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_embed_protected_topk_0p50"), "ours", "same-budget evidence-protected comparator"),
]


INTERNVL_PROXY_RUNS = [
    RunSpec("InternVL3.5-8B", "FastV/TopV-style proxy", 0.50, Path("runs/internvl_textocr_hard/calibrated_test_embed0p50_devthr"), "protocol proxy", "single-pass embedding salience top-k, calibrated test split"),
    RunSpec("InternVL3.5-8B", "ours_soft_evidence", 0.50, Path("runs/internvl_textocr_hard/calibrated_test_target_soft_evidence0p50_b0p05_hfprconstr_devthr"), "ours", "risk-constrained soft-evidence point, calibrated test split"),
]


UNSUPPORTED_INTERNVL_ROWS = [
    {
        "model": "InternVL3.5-8B",
        "method": "VisionZip",
        "ratio": "",
        "implementation": "not claimed",
        "status": "unsupported",
        "n": "",
        "acc": "",
        "hFPR": "",
        "yes_rate": "",
        "pos_acc": "",
        "neg_acc": "",
        "keep_ratio": "",
        "ECR": "",
        "AnchorECR": "",
        "CenterR": "",
        "PatchR": "",
        "path": "",
        "note": "Official VisionZip is CLIP-ViT CLS-attention plus contextual merge. Current InternVL-HF backend exposes tiled image features/placeholder tokens, not the same official CLS-attention path.",
    },
    {
        "model": "InternVL3.5-8B",
        "method": "FastV",
        "ratio": "",
        "implementation": "not claimed",
        "status": "not implemented",
        "n": "",
        "acc": "",
        "hFPR": "",
        "yes_rate": "",
        "pos_acc": "",
        "neg_acc": "",
        "keep_ratio": "",
        "ECR": "",
        "AnchorECR": "",
        "CenterR": "",
        "PatchR": "",
        "path": "",
        "note": "Official FastV is a LLaVA decoder-layer attention hook. InternVL backend has no method-specific FastV branch; existing InternVL rows remain proxies unless this is separately ported and validated.",
    },
]


COLUMNS = [
    "model",
    "method",
    "ratio",
    "implementation",
    "status",
    "n",
    "acc",
    "hFPR",
    "yes_rate",
    "pos_acc",
    "neg_acc",
    "keep_ratio",
    "ECR",
    "AnchorECR",
    "CenterR",
    "PatchR",
    "path",
    "note",
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [row_for_run(spec) for spec in [*LLAVA_OURS_RUNS, *LLAVA_OFFICIAL_RUNS, *INTERNVL_PROXY_RUNS]]
    rows.extend(UNSUPPORTED_INTERNVL_ROWS)
    pair_rows = build_pair_rows()
    write_csv(OUT_DIR / "official_baseline_extension_summary.csv", rows)
    write_pair_csv(OUT_DIR / "official_baseline_extension_pairwise.csv", pair_rows)
    (OUT_DIR / "official_baseline_extension_report.md").write_text(build_markdown(rows, pair_rows), encoding="utf-8")
    print(f"Wrote {OUT_DIR / 'official_baseline_extension_report.md'}")
    print(f"Wrote {OUT_DIR / 'official_baseline_extension_summary.csv'}")


def row_for_run(spec: RunSpec) -> dict[str, Any]:
    run_dir = ROOT / spec.path
    metrics_path = run_dir / "metrics.json"
    score_path = run_dir / "probe_scores.jsonl"
    trace_path = trace_path_for(run_dir, metrics_path)
    row: dict[str, Any] = {
        "model": spec.model,
        "method": spec.method,
        "ratio": spec.ratio,
        "implementation": spec.implementation,
        "status": "done" if metrics_path.exists() and score_path.exists() else "missing",
        "n": "",
        "acc": "",
        "hFPR": "",
        "yes_rate": "",
        "pos_acc": "",
        "neg_acc": "",
        "keep_ratio": "",
        "ECR": "",
        "CenterR": "",
        "PatchR": "",
        "path": str(spec.path),
        "note": spec.note,
    }
    if row["status"] != "done":
        return row
    scores = read_jsonl(score_path)
    metrics = read_json(metrics_path)
    traces = read_jsonl(trace_path) if trace_path.exists() else []
    row.update(summarize_scores(scores, metrics))
    row.update(summarize_pruning(scores, traces))
    return row


def trace_path_for(run_dir: Path, metrics_path: Path) -> Path:
    direct = run_dir / "prune_traces.jsonl"
    if direct.exists() or not metrics_path.exists():
        return direct
    metrics = read_json(metrics_path)
    input_score = metrics.get("yesno_input_score")
    if input_score:
        candidate = (ROOT / str(input_score)).parent / "prune_traces.jsonl"
        if candidate.exists():
            return candidate
    return direct


def summarize_scores(rows: list[dict[str, Any]], metrics: dict[str, Any]) -> dict[str, Any]:
    pos = [row for row in rows if row.get("binary_polarity") == "positive"]
    neg = [row for row in rows if row.get("binary_polarity") == "negative"]
    return {
        "n": len(rows),
        "acc": float(metrics.get("direct_accuracy", mean_bool(row.get("correct") for row in rows))),
        "hFPR": float(metrics.get("direct_hallucination_fpr", mean_bool(row.get("pred_answer") == "yes" for row in neg))),
        "yes_rate": mean_bool(row.get("pred_answer") == "yes" for row in rows),
        "pos_acc": mean_bool(row.get("correct") for row in pos),
        "neg_acc": mean_bool(row.get("correct") for row in neg),
    }


def summarize_pruning(scores: list[dict[str, Any]], traces: list[dict[str, Any]]) -> dict[str, Any]:
    if traces:
        wanted = {(str(row.get("sample_id", "")), str(row.get("probe", ""))) for row in scores}
        rows = [
            row
            for row in traces
            if not wanted or (str(row.get("sample_id", "")), str(row.get("probe", ""))) in wanted
        ]
        evidence_rows = [row for row in rows if row.get("has_evidence")]
        full = mean_number(row.get("full_visual_tokens") for row in rows)
        kept = mean_number(row.get("kept_visual_tokens") for row in rows)
        exhaustive_merge = bool(evidence_rows) and all(
            int(row.get("visionzip_contextual_tokens", 0)) > 0 for row in evidence_rows
        )
        anchor_ecr = mean_number(row.get("anchor_ecr", row.get("ecr")) for row in evidence_rows)
        return {
            "keep_ratio": kept / full if is_number(full) and full else "",
            "ECR": 1.0 if exhaustive_merge else anchor_ecr,
            "AnchorECR": anchor_ecr,
            "CenterR": 1.0 if exhaustive_merge else mean_number(
                row.get("evidence_center_recall") for row in evidence_rows
            ),
            "PatchR": 1.0 if exhaustive_merge else mean_number(
                row.get("evidence_patch_recall") for row in evidence_rows
            ),
        }
    evidence_scores = [row for row in scores if is_number(metric(row, "ecr"))]
    return {
        "keep_ratio": mean_number(metric(row, "keep_ratio") for row in scores),
        "ECR": mean_number(metric(row, "ecr") for row in evidence_scores),
        "AnchorECR": mean_number(metric(row, "ecr") for row in evidence_scores),
        "CenterR": mean_number(metric(row, "evidence_center_recall") for row in evidence_scores),
        "PatchR": mean_number(metric(row, "evidence_patch_recall") for row in evidence_scores),
    }


def metric(row: dict[str, Any], key: str) -> Any:
    aliases = {
        "keep_ratio": ("prune_keep_ratio",),
        "ecr": ("prune_ecr",),
        "evidence_center_recall": ("prune_evidence_center_recall",),
        "evidence_patch_recall": ("prune_evidence_patch_recall",),
    }
    for name in aliases.get(key, (key, f"prune_{key}")):
        if name in row:
            return row[name]
    return ""


def build_pair_rows() -> list[dict[str, Any]]:
    ours_by_ratio = {float(spec.ratio): spec for spec in LLAVA_OURS_RUNS}
    out: list[dict[str, Any]] = []
    for spec in LLAVA_OFFICIAL_RUNS:
        ours = ours_by_ratio.get(float(spec.ratio))
        if ours is None:
            continue
        out.append(compare_pair(ours, spec))
    return out


def compare_pair(left: RunSpec, right: RunSpec) -> dict[str, Any]:
    left_scores = load_sample_scores(ROOT / left.path / "sample_scores.jsonl")
    right_scores = load_sample_scores(ROOT / right.path / "sample_scores.jsonl")
    ids = sorted(set(left_scores) & set(right_scores))
    rows = [(left_scores[sid], right_scores[sid]) for sid in ids]
    negative_rows = [(lrow, rrow) for lrow, rrow in rows if is_negative(lrow) or is_negative(rrow)]

    left_acc = mean_bool(lrow.get("direct_correct") for lrow, _ in rows)
    right_acc = mean_bool(rrow.get("direct_correct") for _, rrow in rows)
    left_hfpr = mean_bool(lrow.get("hallucination") for lrow, _ in negative_rows)
    right_hfpr = mean_bool(rrow.get("hallucination") for _, rrow in negative_rows)
    acc_b, acc_c = discordance(rows, "direct_correct")
    hfpr_b, hfpr_c = discordance(negative_rows, "hallucination")
    return {
        "ratio": left.ratio,
        "left_method": left.method,
        "right_method": right.method,
        "n": len(rows),
        "n_negative": len(negative_rows),
        "left_acc": left_acc,
        "right_acc": right_acc,
        "acc_diff": left_acc - right_acc,
        "acc_sign_p": exact_sign_p(acc_b, acc_c),
        "acc_left_better": acc_b,
        "acc_left_worse": acc_c,
        "left_hFPR": left_hfpr,
        "right_hFPR": right_hfpr,
        "hFPR_diff": left_hfpr - right_hfpr,
        "hFPR_sign_p": exact_sign_p(hfpr_b, hfpr_c),
        "hFPR_left_worse": hfpr_b,
        "hFPR_left_better": hfpr_c,
    }


def load_sample_scores(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {str(row["sample_id"]): row for row in read_jsonl(path)}


def is_negative(row: dict[str, Any]) -> bool:
    return bool(row.get("target_is_negative", False))


def discordance(rows: list[tuple[dict[str, Any], dict[str, Any]]], key: str) -> tuple[int, int]:
    left_true_right_false = 0
    right_true_left_false = 0
    for left, right in rows:
        lval = bool(left.get(key, False))
        rval = bool(right.get(key, False))
        if lval and not rval:
            left_true_right_false += 1
        elif rval and not lval:
            right_true_left_false += 1
    return left_true_right_false, right_true_left_false


def exact_sign_p(left_true_right_false: int, right_true_left_false: int) -> float:
    n = left_true_right_false + right_true_left_false
    if n == 0:
        return 1.0
    k = min(left_true_right_false, right_true_left_false)
    terms = [
        math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2.0)
        for i in range(k + 1)
    ]
    max_term = max(terms)
    tail = math.exp(max_term) * sum(math.exp(term - max_term) for term in terms)
    return min(1.0, 2.0 * tail)


def build_markdown(rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Official Baseline Extension Check",
        "",
        "Scope: TextOCR-HardYesNo. This report separates official-algorithm ports from protocol-compatible proxies.",
        "",
        "## Status Matrix",
        "",
        "| model | method | implementation | status | evidence |",
        "|---|---|---|---|---|",
        "| LLaVA-1.5-7B | VisionZip | official-algorithm port | done | 0.20/0.30/0.40/0.50 budget curve exists |",
        "| LLaVA-1.5-7B | FastV | official-algorithm port | done | 0.20/0.30/0.40/0.50 budget curve exists |",
        "| InternVL3.5-8B | VisionZip | not claimed | unsupported | official CLIP CLS-attention path is not available in current InternVL backend |",
        "| InternVL3.5-8B | FastV | not claimed | not implemented | no InternVL decoder-layer official FastV branch exists |",
        "| InternVL3.5-8B | FastV/TopV-style proxy | protocol proxy | done | calibrated `embed0p50` row exists; must not be called official |",
        "",
        "## LLaVA Official Budget Curve",
        "",
        "| method | ratio | acc | hFPR | pos acc | neg acc | yes rate | keep | Lineage ECR | Anchor ECR | path |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted([row for row in rows if row["model"].startswith("LLaVA")], key=sort_key):
        if row["status"] != "done":
            continue
        lines.append(
            "| {method} | {ratio} | {acc} | {hfpr} | {pos} | {neg} | {yes} | {keep} | {ecr} | {anchor_ecr} | `{path}` |".format(
                method=row["method"],
                ratio=fmt(row["ratio"]),
                acc=fmt(row["acc"]),
                hfpr=fmt(row["hFPR"]),
                pos=fmt(row["pos_acc"]),
                neg=fmt(row["neg_acc"]),
                yes=fmt(row["yes_rate"]),
                keep=fmt(row["keep_ratio"]),
                ecr=fmt(row["ECR"]),
                anchor_ecr=fmt(row["AnchorECR"]),
                path=row["path"],
            )
        )

    lines.extend(
        [
            "",
            "## LLaVA Paired Checks vs Ours",
            "",
            "| right method | ratio | n | acc diff | acc p | hFPR diff | hFPR p | reading |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in sorted(pair_rows, key=lambda item: (float(item["ratio"]), str(item["right_method"]))):
        lines.append(
            "| {method} | {ratio} | {n} | {acc_diff} | {acc_p} | {hfpr_diff} | {hfpr_p} | {reading} |".format(
                method=row["right_method"],
                ratio=fmt(row["ratio"]),
                n=row["n"],
                acc_diff=fmt_signed(row["acc_diff"]),
                acc_p=fmt_p(row["acc_sign_p"]),
                hfpr_diff=fmt_signed(row["hFPR_diff"]),
                hfpr_p=fmt_p(row["hFPR_sign_p"]),
                reading=pair_reading(row),
            )
        )

    lines.extend(
        [
            "",
            "## InternVL Official-Port Check",
            "",
            "| method | implementation | status | acc | hFPR | keep | ECR | path / note |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in [row for row in rows if row["model"].startswith("InternVL")]:
        path_or_note = f"`{row['path']}`" if row.get("path") else str(row["note"])
        lines.append(
            "| {method} | {implementation} | {status} | {acc} | {hfpr} | {keep} | {ecr} | {path_or_note} |".format(
                method=row["method"],
                implementation=row["implementation"],
                status=row["status"],
                acc=fmt(row["acc"]),
                hfpr=fmt(row["hFPR"]),
                keep=fmt(row["keep_ratio"]),
                ecr=fmt(row["ECR"]),
                path_or_note=path_or_note,
            )
        )

    lines.extend(
        [
            "",
            "## Reviewer-Facing Readout",
            "",
            "- LLaVA now has the official-baseline extension check: both VisionZip and FastV are evaluated at 0.20/0.30/0.40/0.50 under the same TextOCR-Hard protocol.",
            "- InternVL should remain labelled as proxy-only for external methods. Calling its `embed0p50` row official would be misleading.",
            "- FastV hFPR must be interpreted with positive accuracy and yes rate: on LLaVA it collapses to all-no, so hFPR=0 is not a useful standalone win.",
            "- For the paper, use LLaVA official ports as the main external-method evidence and InternVL proxy rows as cross-model sanity checks.",
            "",
            "## Source Hooks",
            "",
            "- LLaVA official paths: `recap/llava_pruned_backend.py` has method-specific `visionzip` and `fastv` branches.",
            "- InternVL backend: `recap/internvl_pruned_backend.py` only uses generic selector dispatch and has no method-specific official branch.",
            "- CLI wording: `run-llava-pruned` advertises LLaVA-only official ports, while `run-internvl-pruned` lists only generic non-oracle selector variants.",
        ]
    )
    return "\n".join(lines) + "\n"


def sort_key(row: dict[str, Any]) -> tuple[float, int, str]:
    order = {"ours_protected_embed": 0, "VisionZip": 1, "FastV": 2}
    ratio = float(row["ratio"]) if is_number(row["ratio"]) else 99.0
    return ratio, order.get(str(row["method"]), 99), str(row["method"])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_pair_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "ratio",
        "left_method",
        "right_method",
        "n",
        "n_negative",
        "left_acc",
        "right_acc",
        "acc_diff",
        "acc_sign_p",
        "acc_left_better",
        "acc_left_worse",
        "left_hFPR",
        "right_hFPR",
        "hFPR_diff",
        "hFPR_sign_p",
        "hFPR_left_worse",
        "hFPR_left_better",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mean_bool(values: Any) -> float:
    vals = [bool(value) for value in values]
    return sum(1 for value in vals if value) / len(vals) if vals else 0.0


def mean_number(values: Any) -> float | str:
    vals = [float(value) for value in values if is_number(value)]
    return sum(vals) / len(vals) if vals else ""


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) or (isinstance(value, str) and value != "")


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


def pair_reading(row: dict[str, Any]) -> str:
    acc = "acc higher" if float(row["acc_diff"]) > 0 else "acc lower/tied"
    hfpr = "hFPR lower" if float(row["hFPR_diff"]) < 0 else "hFPR higher"
    return f"{acc}; {hfpr}"


if __name__ == "__main__":
    main()
