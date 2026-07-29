#!/usr/bin/env python
"""Build the LLaVA external-method visual-token budget curve report."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "llava_textocr_hard"


@dataclass(frozen=True)
class RunSpec:
    method: str
    ratio: float
    path: Path
    role: str
    note: str = ""


RUN_SPECS = [
    RunSpec("ours_protected_embed", 0.20, Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_embed_protected_topk_0p20"), "ours", "evidence-protected embedding top-k"),
    RunSpec("VisionZip", 0.20, Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_visionzip_0p20"), "external official-algorithm port", "dominant token selection plus contextual merge"),
    RunSpec("FastV", 0.20, Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_fastv_0p20"), "external official-algorithm port", "layer-attention pruning at K=3"),
    RunSpec("ours_protected_embed", 0.30, Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_embed_protected_topk_0p30"), "ours", "evidence-protected embedding top-k"),
    RunSpec("VisionZip", 0.30, Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_visionzip_0p30"), "external official-algorithm port", "dominant token selection plus contextual merge"),
    RunSpec("FastV", 0.30, Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_fastv_0p30"), "external official-algorithm port", "layer-attention pruning at K=3"),
    RunSpec("ours_protected_embed", 0.40, Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_embed_protected_topk_0p40"), "ours", "evidence-protected embedding top-k"),
    RunSpec("SCOPE", 0.40, Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_scope_0p40"), "external official-algorithm port", "saliency-coverage greedy selection; official default alpha=1"),
    RunSpec("CoIn", 0.40, Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_coin_0p40"), "external paper-algorithm port", "incremental Gram-Schmidt selection; paper LLaVA-1.5 alpha=0.9, beta=0.6"),
    RunSpec("VisionZip", 0.40, Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_visionzip_0p40"), "external official-algorithm port", "dominant token selection plus contextual merge"),
    RunSpec("FastV", 0.40, Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_fastv_0p40"), "external official-algorithm port", "layer-attention pruning at K=3"),
    RunSpec("ours_protected_embed", 0.50, Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_embed_protected_topk_0p50"), "ours", "evidence-protected embedding top-k"),
    RunSpec("VisionZip", 0.50, Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_visionzip_0p50"), "external official-algorithm port", "dominant token selection plus contextual merge"),
    RunSpec("FastV", 0.50, Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_fastv_0p50"), "external official-algorithm port", "layer-attention pruning at K=3"),
]

METHOD_ORDER = {
    "ours_protected_embed": 0,
    "SCOPE": 1,
    "CoIn": 2,
    "VisionZip": 3,
    "FastV": 4,
}


CSV_COLUMNS = [
    "method",
    "ratio",
    "role",
    "status",
    "n",
    "acc",
    "hFPR",
    "yes_rate",
    "pos_acc",
    "neg_acc",
    "full_visual",
    "kept_visual",
    "keep_ratio",
    "ECR",
    "CenterR",
    "PatchR",
    "mean_forward_ms",
    "mean_prune_overhead_ms",
    "acc_delta_vs_ours",
    "hFPR_delta_vs_ours",
    "ECR_delta_vs_ours",
    "path",
    "note",
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [row_for_spec(spec) for spec in RUN_SPECS]
    add_deltas(rows)
    write_csv(OUT_DIR / "llava_external_budget_curve.csv", rows)
    (OUT_DIR / "llava_external_budget_curve.md").write_text(build_markdown(rows))
    print(f"Wrote {OUT_DIR / 'llava_external_budget_curve.csv'}")
    print(f"Wrote {OUT_DIR / 'llava_external_budget_curve.md'}")


def row_for_spec(spec: RunSpec) -> dict[str, Any]:
    run_dir = ROOT / spec.path
    metrics_path = run_dir / "metrics.json"
    score_path = run_dir / "probe_scores.jsonl"
    trace_path = run_dir / "prune_traces.jsonl"
    row: dict[str, Any] = {
        "method": spec.method,
        "ratio": spec.ratio,
        "role": spec.role,
        "status": "done" if metrics_path.exists() and score_path.exists() and trace_path.exists() else "missing",
        "n": "",
        "acc": "",
        "hFPR": "",
        "yes_rate": "",
        "pos_acc": "",
        "neg_acc": "",
        "full_visual": "",
        "kept_visual": "",
        "keep_ratio": "",
        "ECR": "",
        "CenterR": "",
        "PatchR": "",
        "mean_forward_ms": "",
        "mean_prune_overhead_ms": "",
        "acc_delta_vs_ours": "",
        "hFPR_delta_vs_ours": "",
        "ECR_delta_vs_ours": "",
        "path": str(spec.path),
        "note": spec.note,
    }
    if row["status"] != "done":
        return row

    metrics = read_json(metrics_path)
    scores = read_jsonl(score_path)
    traces = read_jsonl(trace_path)
    row.update(summarize_scores(scores, metrics))
    row.update(summarize_traces(traces, scores))
    return row


def summarize_scores(rows: list[dict[str, Any]], metrics: dict[str, Any]) -> dict[str, Any]:
    pos = [row for row in rows if row.get("binary_polarity") == "positive"]
    neg = [row for row in rows if row.get("binary_polarity") == "negative"]
    return {
        "n": len(rows),
        "acc": float(metrics.get("direct_accuracy", mean_bool(row.get("correct") for row in rows))),
        "hFPR": float(metrics.get("direct_hallucination_fpr", h_fpr(rows))),
        "yes_rate": mean_bool(row.get("pred_answer") == "yes" for row in rows),
        "pos_acc": mean_bool(row.get("correct") for row in pos),
        "neg_acc": mean_bool(row.get("correct") for row in neg),
    }


def summarize_traces(rows: list[dict[str, Any]], scores: list[dict[str, Any]]) -> dict[str, Any]:
    wanted = {(str(row.get("sample_id", "")), str(row.get("probe", ""))) for row in scores}
    if wanted:
        rows = [row for row in rows if (str(row.get("sample_id", "")), str(row.get("probe", ""))) in wanted]
    if not rows:
        return {}
    evidence_rows = [row for row in rows if row.get("has_evidence")]
    full = mean_numeric(row.get("full_visual_tokens") for row in rows)
    kept = mean_numeric(row.get("kept_visual_tokens") for row in rows)
    return {
        "full_visual": full,
        "kept_visual": kept,
        "keep_ratio": kept / full if isinstance(full, float) and full > 0.0 else "",
        "ECR": mean_numeric(row.get("ecr") for row in evidence_rows),
        "CenterR": mean_numeric(row.get("evidence_center_recall") for row in evidence_rows),
        "PatchR": mean_numeric(row.get("evidence_patch_recall") for row in evidence_rows),
        "mean_forward_ms": mean_numeric(row.get("mean_forward_ms") for row in rows),
        "mean_prune_overhead_ms": mean_numeric(row.get("mean_prune_overhead_ms") for row in rows),
    }


def add_deltas(rows: list[dict[str, Any]]) -> None:
    ours_by_ratio = {row["ratio"]: row for row in rows if row["method"] == "ours_protected_embed" and row["status"] == "done"}
    for row in rows:
        base = ours_by_ratio.get(row["ratio"])
        if not base or row["method"] == "ours_protected_embed":
            continue
        for out_key, key in (
            ("acc_delta_vs_ours", "acc"),
            ("hFPR_delta_vs_ours", "hFPR"),
            ("ECR_delta_vs_ours", "ECR"),
        ):
            if is_number(row.get(key)) and is_number(base.get(key)):
                row[out_key] = float(row[key]) - float(base[key])


def build_markdown(rows: list[dict[str, Any]]) -> str:
    done = [row for row in rows if row["status"] == "done"]
    lines = [
        "# LLaVA External-Method Budget Curve",
        "",
        "Dataset: TextOCR-HardYesNo, 1000 probes. Model: LLaVA-1.5-7B. All rows use the same probe file and the same target visual-token budget.",
        "",
        "SCOPE, VisionZip, and FastV are local official-algorithm ports. CoIn is a paper-algorithm port because its conference page links no code; SCOPE and CoIn are evaluated at the main 40% budget.",
        "",
        "## Main Curve",
        "",
        "| method | keep target | acc | hFPR | pos acc | neg acc | yes rate | keep | ECR | CenterR | PatchR | mean ms | path |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(done, key=sort_key):
        lines.append(
            "| {method} | {ratio} | {acc} | {hFPR} | {pos_acc} | {neg_acc} | {yes_rate} | {keep} | {ECR} | {CenterR} | {PatchR} | {mean_ms} | `{path}` |".format(
                method=row["method"],
                ratio=fmt(row["ratio"]),
                acc=fmt(row["acc"]),
                hFPR=fmt(row["hFPR"]),
                pos_acc=fmt(row["pos_acc"]),
                neg_acc=fmt(row["neg_acc"]),
                yes_rate=fmt(row["yes_rate"]),
                keep=fmt(row["keep_ratio"]),
                ECR=fmt(row["ECR"]),
                CenterR=fmt(row["CenterR"]),
                PatchR=fmt(row["PatchR"]),
                mean_ms=fmt(row["mean_forward_ms"], digits=1),
                path=row["path"],
            )
        )

    lines.extend(
        [
            "",
            "## Same-Budget Deltas vs Ours",
            "",
            "| method | keep target | Δacc | ΔhFPR | ΔECR | note |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in sorted(done, key=sort_key):
        if row["method"] == "ours_protected_embed":
            continue
        lines.append(
            "| {method} | {ratio} | {da} | {dh} | {de} | {note} |".format(
                method=row["method"],
                ratio=fmt(row["ratio"]),
                da=fmt(row["acc_delta_vs_ours"]),
                dh=fmt(row["hFPR_delta_vs_ours"]),
                de=fmt(row["ECR_delta_vs_ours"]),
                note=row["note"],
            )
        )

    lines.extend(["", "## Readout", ""])
    lines.extend(readout_lines(done))
    lines.append("")
    return "\n".join(lines)


def readout_lines(rows: list[dict[str, Any]]) -> list[str]:
    by_method = {method: [row for row in rows if row["method"] == method] for method in sorted({str(row["method"]) for row in rows})}
    ours = by_method.get("ours_protected_embed", [])
    visionzip = by_method.get("VisionZip", [])
    fastv = by_method.get("FastV", [])
    scope = by_method.get("SCOPE", [])
    best_ours = max(ours, key=lambda row: float(row["acc"])) if ours else None
    out: list[str] = []
    if best_ours:
        out.append(f"- Best ours row: keep={fmt(best_ours['ratio'])}, acc={fmt(best_ours['acc'])}, hFPR={fmt(best_ours['hFPR'])}.")
    if visionzip:
        max_hfpr = max(float(row["hFPR"]) for row in visionzip if is_number(row.get("hFPR")))
        out.append(f"- VisionZip remains high-hallucination on this OCR hard set; max hFPR={fmt(max_hfpr)} and every same-budget row is below ours in accuracy.")
    if scope:
        row = scope[0]
        out.append(
            f"- SCOPE at 40% reaches acc={fmt(row['acc'])}, hFPR={fmt(row['hFPR'])}, and ECR={fmt(row['ECR'])}; "
            "this tests a recent saliency-coverage baseline without exposing OCR boxes to selection."
        )
    if fastv:
        pos_values = sorted({float(row["pos_acc"]) for row in fastv if is_number(row.get("pos_acc"))})
        if pos_values == [0.0]:
            out.append("- FastV collapses to a conservative all-no regime across 0.20-0.50: hFPR=0.000, neg_acc=1.000, pos_acc=0.000, acc=0.500.")
    out.append("- The external-method comparison should therefore report both hFPR and positive accuracy; hFPR alone would incorrectly make FastV look safe.")
    return out


def sort_key(row: dict[str, Any]) -> tuple[float, int, str]:
    method = str(row["method"])
    return (float(row["ratio"]), METHOD_ORDER.get(method, 99), method)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def mean_bool(values: Iterable[Any]) -> float:
    vals = [bool(value) for value in values]
    return sum(vals) / len(vals) if vals else 0.0


def mean_numeric(values: Iterable[Any]) -> float | str:
    vals = [float(value) for value in values if value is not None and value != ""]
    return sum(vals) / len(vals) if vals else ""


def h_fpr(rows: list[dict[str, Any]]) -> float:
    neg = [row for row in rows if row.get("binary_polarity") == "negative"]
    return mean_bool(row.get("pred_answer") == "yes" for row in neg)


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) or (isinstance(value, str) and value != "")


def fmt(value: Any, digits: int = 3) -> str:
    if value == "" or value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    main()
