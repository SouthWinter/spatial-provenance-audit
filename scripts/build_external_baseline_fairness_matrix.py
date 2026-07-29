#!/usr/bin/env python3
"""Build structured evidence for external-baseline fairness.

This script turns the existing official-baseline extension and parity audit into
paper-facing CSV tables. It deliberately separates three evidence types:

- method-specific official-algorithm ports;
- protocol-compatible proxy rows;
- unsupported / not-claimed ports.

The goal is to make the baseline comparison auditable without inflating proxy
rows into official implementations.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "problem_optimization_audit" / "external_baseline_fairness"
OFFICIAL_DIR = ROOT / "runs" / "official_baseline_extension"
HARD_CSV = ROOT / "runs" / "hard_evidence" / "hard_evidence_summary.csv"
PAPER_EVIDENCE = ROOT / "runs" / "paper_evidence"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    official_rows = read_csv(OFFICIAL_DIR / "official_baseline_extension_summary.csv")
    pair_rows = read_csv(OFFICIAL_DIR / "official_baseline_extension_pairwise.csv")
    hard_rows = read_csv(HARD_CSV)

    matrix = build_matrix(official_rows, hard_rows)
    budget_curve = build_budget_curve(official_rows)
    paired = build_paired_readout(pair_rows)

    write_csv(OUT_DIR / "baseline_fairness_matrix.csv", matrix)
    write_csv(OUT_DIR / "baseline_budget_curve_summary.csv", budget_curve)
    write_csv(OUT_DIR / "baseline_pairwise_readout.csv", paired)
    PAPER_EVIDENCE.mkdir(parents=True, exist_ok=True)
    write_csv(PAPER_EVIDENCE / "table_external_baseline_fairness_matrix.csv", matrix)
    write_csv(PAPER_EVIDENCE / "table_external_baseline_budget_curve.csv", budget_curve)
    write_csv(PAPER_EVIDENCE / "table_external_baseline_paired_readout.csv", paired)
    (OUT_DIR / "baseline_fairness_report.md").write_text(
        build_markdown(matrix, budget_curve, paired),
        encoding="utf-8",
    )
    print(f"Wrote {OUT_DIR / 'baseline_fairness_report.md'}")


def build_matrix(
    official_rows: list[dict[str, str]],
    hard_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    llava_budgets = budgets_for(
        official_rows,
        model="LLaVA-1.5-7B",
        methods={"FastV", "VisionZip"},
        implementation="official-algorithm port",
    )
    qwen_proxy = proxy_rows(hard_rows, model="Qwen3-VL-8B")
    llava_proxy = proxy_rows(hard_rows, model="LLaVA-1.5-7B")
    internvl_proxy = proxy_rows(hard_rows, model="InternVL3.5-8B")

    rows = [
        {
            "backbone": "LLaVA-1.5-7B",
            "baseline": "FastV",
            "implementation_class": "official-algorithm port",
            "result_status": "done",
            "budgets_evaluated": ",".join(llava_budgets.get("FastV", [])),
            "same_protocol": "yes",
            "same_budget_curve": "yes",
            "metrics_available": "accuracy,hFPR,yes_rate,pos_acc,neg_acc,keep_ratio,ECR,CenterR,PatchR,paired_sign_test",
            "n": "1000",
            "source_evidence": "third_party/fastv@d165972; recap/llava_pruned_backend.py; official_baseline_extension_summary.csv",
            "safe_claim": "method-specific FastV port is evaluated on LLaVA under the TextOCR-Hard protocol",
            "remaining_gap": "FastV collapses to all-no on this setting, so hFPR=0 must not be read as evidence safety",
        },
        {
            "backbone": "LLaVA-1.5-7B",
            "baseline": "VisionZip",
            "implementation_class": "official-algorithm port",
            "result_status": "done",
            "budgets_evaluated": ",".join(llava_budgets.get("VisionZip", [])),
            "same_protocol": "yes",
            "same_budget_curve": "yes",
            "metrics_available": "accuracy,hFPR,yes_rate,pos_acc,neg_acc,keep_ratio,ECR,CenterR,PatchR,paired_sign_test",
            "n": "1000",
            "source_evidence": "third_party/VisionZip@8f86b55; recap/llava_pruned_backend.py; official_baseline_extension_summary.csv",
            "safe_claim": "method-specific VisionZip port is evaluated on LLaVA under the TextOCR-Hard protocol",
            "remaining_gap": "coverage is LLaVA-specific; Qwen3 has a separate native port, while InternVL VisionZip remains unsupported",
        },
        {
            "backbone": "Qwen3-VL-8B",
            "baseline": "FastV",
            "implementation_class": "not claimed",
            "result_status": "unsupported",
            "budgets_evaluated": "",
            "same_protocol": "no",
            "same_budget_curve": "no",
            "metrics_available": "",
            "n": "",
            "source_evidence": "external_baseline_parity_audit.md; recap/qwen_pruned_backend.py",
            "safe_claim": "No official Qwen FastV row is claimed",
            "remaining_gap": "native FastV adaptation for Qwen is not implemented in the current backend",
        },
        {
            "backbone": "Qwen3-VL-8B",
            "baseline": "VisionZip",
            "implementation_class": "official-algorithm port",
            "result_status": "done",
            "budgets_evaluated": "0.301",
            "same_protocol": "yes",
            "same_budget_curve": "single operating point",
            "metrics_available": "accuracy,hFPR,yes_rate,pos_acc,neg_acc,keep_ratio,ECR,CenterR,PatchR,native_port_gates",
            "n": "1000",
            "source_evidence": "qwen3_visionzip_native_port_gate_report.md; qwen3_visionzip_textocr_readout.md; recap/qwen_pruned_backend.py",
            "safe_claim": "Qwen3 VisionZip is evaluated as a Qwen3-specific native official-algorithm port at the 0.30 TextOCR-Hard budget",
            "remaining_gap": "single Qwen3 budget point; it improves ECR/hFPR but does not beat our Qwen Target 0.30 accuracy",
        },
        {
            "backbone": "Qwen3-VL-8B",
            "baseline": "FastV/TopV-style salience",
            "implementation_class": "protocol-compatible proxy",
            "result_status": "done",
            "budgets_evaluated": budgets_from_proxy(qwen_proxy, "FastV/TopV-style proxy"),
            "same_protocol": "yes",
            "same_budget_curve": "single operating point",
            "metrics_available": metrics_from_rows(qwen_proxy, "FastV/TopV-style proxy"),
            "n": n_from_rows(qwen_proxy, "FastV/TopV-style proxy"),
            "source_evidence": "hard_evidence_summary.csv; generic embedding-salience selector",
            "safe_claim": "Qwen has a protocol-compatible salience proxy, not an official FastV port",
            "remaining_gap": "native FastV adaptation for Qwen is not claimed",
        },
        {
            "backbone": "Qwen3-VL-8B",
            "baseline": "VisionZip-style diversity",
            "implementation_class": "protocol-compatible proxy",
            "result_status": "done",
            "budgets_evaluated": budgets_from_proxy(qwen_proxy, "VisionZip-style proxy"),
            "same_protocol": "yes",
            "same_budget_curve": "single operating point",
            "metrics_available": metrics_from_rows(qwen_proxy, "VisionZip-style proxy"),
            "n": n_from_rows(qwen_proxy, "VisionZip-style proxy"),
            "source_evidence": "hard_evidence_summary.csv; generic relevance/diversity selector",
            "safe_claim": "Qwen has a protocol-compatible VisionZip-style proxy used only as a mechanism check; the native VisionZip row should be preferred for official-baseline claims",
            "remaining_gap": "legacy proxy is not an official VisionZip port and should not be mixed with the native-port row",
        },
        {
            "backbone": "LLaVA-1.5-7B",
            "baseline": "FastV/TopV-style salience",
            "implementation_class": "protocol-compatible proxy",
            "result_status": "done",
            "budgets_evaluated": budgets_from_proxy(llava_proxy, "FastV/TopV-style proxy"),
            "same_protocol": "yes",
            "same_budget_curve": "single operating point",
            "metrics_available": metrics_from_rows(llava_proxy, "FastV/TopV-style proxy"),
            "n": n_from_rows(llava_proxy, "FastV/TopV-style proxy"),
            "source_evidence": "hard_evidence_summary.csv; generic embedding-salience selector",
            "safe_claim": "LLaVA proxy rows are separate from the official FastV port and are used only as mechanism checks",
            "remaining_gap": "official-port rows should be preferred for external-method claims",
        },
        {
            "backbone": "InternVL3.5-8B",
            "baseline": "FastV/TopV-style salience",
            "implementation_class": "protocol-compatible proxy",
            "result_status": "done",
            "budgets_evaluated": budgets_from_proxy(internvl_proxy, "FastV/TopV-style proxy"),
            "same_protocol": "yes",
            "same_budget_curve": "single operating point",
            "metrics_available": metrics_from_rows(internvl_proxy, "FastV/TopV-style proxy"),
            "n": n_from_rows(internvl_proxy, "FastV/TopV-style proxy"),
            "source_evidence": "hard_evidence_summary.csv; calibrated generic embedding-salience selector",
            "safe_claim": "InternVL has a calibrated protocol proxy, not an official FastV/TopV port",
            "remaining_gap": "native method-specific attention hook is not implemented",
        },
        {
            "backbone": "InternVL3.5-8B",
            "baseline": "VisionZip",
            "implementation_class": "not claimed",
            "result_status": "unsupported",
            "budgets_evaluated": "",
            "same_protocol": "no",
            "same_budget_curve": "no",
            "metrics_available": "",
            "n": "",
            "source_evidence": "official_baseline_extension_report.md; recap/internvl_pruned_backend.py",
            "safe_claim": "No official InternVL VisionZip row is claimed",
            "remaining_gap": "official CLIP-ViT CLS-attention plus contextual-merge path is not available in the current InternVL backend",
        },
        {
            "backbone": "InternVL3.5-8B",
            "baseline": "FastV",
            "implementation_class": "not claimed",
            "result_status": "unsupported",
            "budgets_evaluated": "",
            "same_protocol": "no",
            "same_budget_curve": "no",
            "metrics_available": "",
            "n": "",
            "source_evidence": "official_baseline_extension_report.md; recap/internvl_pruned_backend.py",
            "safe_claim": "No official InternVL FastV row is claimed",
            "remaining_gap": "official decoder-layer attention hook is not implemented in the current InternVL backend",
        },
    ]
    return rows


def build_budget_curve(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if row.get("model") != "LLaVA-1.5-7B":
            continue
        if row.get("status") != "done":
            continue
        if row.get("method") not in {"ours_protected_embed", "FastV", "VisionZip"}:
            continue
        out.append(
            {
                "backbone": row["model"],
                "method": row["method"],
                "implementation": row["implementation"],
                "ratio": fmt(row["ratio"]),
                "n": row["n"],
                "acc": fmt(row["acc"]),
                "hFPR": fmt(row["hFPR"]),
                "yes_rate": fmt(row["yes_rate"]),
                "pos_acc": fmt(row["pos_acc"]),
                "neg_acc": fmt(row["neg_acc"]),
                "keep_ratio": fmt(row["keep_ratio"]),
                "ECR": fmt(row["ECR"]),
                "CenterR": fmt(row["CenterR"]),
                "PatchR": fmt(row["PatchR"]),
                "source": row["path"],
                "note": row["note"],
            }
        )
    return sorted(out, key=lambda row: (float(row["ratio"]), row["method"]))


def build_paired_readout(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        out.append(
            {
                "backbone": "LLaVA-1.5-7B",
                "left": row["left_method"],
                "right": row["right_method"],
                "ratio": fmt(row["ratio"]),
                "n": row["n"],
                "acc_diff_left_minus_right": fmt_signed(row["acc_diff"]),
                "acc_p": p(row["acc_sign_p"]),
                "hFPR_diff_left_minus_right": fmt_signed(row["hFPR_diff"]),
                "hFPR_p": p(row["hFPR_sign_p"]),
                "interpretation": pair_interpretation(row),
            }
        )
    return sorted(out, key=lambda row: (float(row["ratio"]), row["right"]))


def build_markdown(
    matrix: list[dict[str, Any]],
    budget_curve: list[dict[str, Any]],
    paired: list[dict[str, Any]],
) -> str:
    lines = [
        "# External Baseline Fairness Matrix",
        "",
        "This report structures the external-baseline evidence needed for the `problem.md` parity concern. It separates official-algorithm ports from protocol-compatible proxies and unsupported rows.",
        "",
        "## Fairness Matrix",
        "",
        table_md(
            matrix,
            [
                "backbone",
                "baseline",
                "implementation_class",
                "result_status",
                "budgets_evaluated",
                "same_protocol",
                "same_budget_curve",
                "metrics_available",
                "safe_claim",
                "remaining_gap",
            ],
        ),
        "",
        "## LLaVA Official-Port Budget Curve",
        "",
        table_md(
            budget_curve,
            [
                "method",
                "implementation",
                "ratio",
                "acc",
                "hFPR",
                "yes_rate",
                "keep_ratio",
                "ECR",
                "note",
            ],
        ),
        "",
        "## Same-Budget Paired Readout",
        "",
        table_md(
            paired,
            [
                "left",
                "right",
                "ratio",
                "n",
                "acc_diff_left_minus_right",
                "acc_p",
                "hFPR_diff_left_minus_right",
                "hFPR_p",
                "interpretation",
            ],
        ),
        "",
        "## Manuscript Boundary",
        "",
        "Safe claim: official FastV/VisionZip comparisons are complete for LLaVA under the TextOCR-Hard protocol, and Qwen3 VisionZip has a scoped native official-algorithm port at the 0.30 TextOCR-Hard budget. Qwen3 FastV and InternVL FastV/VisionZip remain proxy or unsupported according to backend coverage.",
        "",
        "Unsafe claim: official FastV/VisionZip results are available on all three backbones or Qwen3 VisionZip beats our Qwen Target 0.30 accuracy. The current evidence does not support those statements.",
    ]
    return "\n".join(lines) + "\n"


def budgets_for(
    rows: list[dict[str, str]],
    *,
    model: str,
    methods: set[str],
    implementation: str,
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {method: [] for method in methods}
    for row in rows:
        if row.get("model") != model:
            continue
        if row.get("method") not in methods:
            continue
        if row.get("implementation") != implementation:
            continue
        if row.get("status") != "done":
            continue
        out[row["method"]].append(fmt(row["ratio"]))
    return {key: sorted(values, key=float) for key, values in out.items()}


def proxy_rows(rows: list[dict[str, str]], *, model: str) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("model") == model
        and row.get("family") == "external_proxy"
        and row.get("status") == "done"
    ]


def budgets_from_proxy(rows: list[dict[str, str]], role: str) -> str:
    values = sorted({fmt(row["keep_ratio"]) for row in rows if row.get("role") == role}, key=float)
    return ",".join(values)


def metrics_from_rows(rows: list[dict[str, str]], role: str) -> str:
    if not any(row.get("role") == role for row in rows):
        return ""
    return "accuracy,hFPR,yes_rate,pos_acc,neg_acc,keep_ratio,ECR,CenterR,PatchR"


def n_from_rows(rows: list[dict[str, str]], role: str) -> str:
    values = sorted({row["n"] for row in rows if row.get("role") == role})
    return ",".join(values)


def pair_interpretation(row: dict[str, str]) -> str:
    right = row.get("right_method", "")
    acc = float(row["acc_diff"])
    hfpr = float(row["hFPR_diff"])
    if right == "FastV" and hfpr > 0:
        return "ours has higher accuracy, but FastV is all-no; hFPR=0 is a collapse, not evidence safety"
    if acc > 0 and hfpr < 0:
        return "ours is higher accuracy and lower hFPR"
    if acc > 0:
        return "ours is higher accuracy; inspect hFPR/yes-rate trade-off"
    return "no clear quality win"


def table_md(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "(empty)"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def fmt(value: str) -> str:
    if value == "":
        return ""
    try:
        return f"{float(value):.3f}"
    except Exception:
        return value


def fmt_signed(value: str) -> str:
    return f"{float(value):+.3f}"


def p(value: str) -> str:
    val = float(value)
    if val < 1e-4:
        return "<1e-4"
    return f"{val:.4f}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
