#!/usr/bin/env python3
"""Build a reviewer-facing audit of external-baseline parity.

The goal is not to manufacture a stronger claim than the evidence supports.
Instead, this audit makes the implementation boundary explicit:

- LLaVA has method-specific FastV and VisionZip branches.
- Qwen and InternVL currently use protocol-compatible proxy selectors.
- InternVL official ports are not claimed because the backend lacks the
  method-specific attention/CLS-token paths those algorithms require.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "problem_optimization_audit"
OFFICIAL_DIR = ROOT / "runs" / "official_baseline_extension"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = read_csv(OFFICIAL_DIR / "official_baseline_extension_summary.csv")
    pair_rows = read_csv(OFFICIAL_DIR / "official_baseline_extension_pairwise.csv")
    source = source_audit()
    audit = {
        "llava_official_done": llava_official_done(summary_rows),
        "qwen_official_done": False,
        "internvl_official_done": False,
        "source_audit": source,
        "safe_claim": (
            "External official-algorithm ports are evaluated for LLaVA where the "
            "backend exposes method-specific FastV and VisionZip paths; Qwen and "
            "InternVL external rows are protocol-compatible proxies and must be "
            "labelled as such."
        ),
        "remaining_risk": (
            "A reviewer may still ask for native official ports on Qwen or InternVL. "
            "The current evidence does not close that request; it only prevents an "
            "overclaim and provides a fair LLaVA official-port comparison."
        ),
    }
    (OUT_DIR / "external_baseline_parity_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (OUT_DIR / "external_baseline_parity_audit.md").write_text(
        build_markdown(summary_rows, pair_rows, source),
        encoding="utf-8",
    )
    print(f"Wrote {OUT_DIR / 'external_baseline_parity_audit.md'}")


def llava_official_done(rows: list[dict[str, str]]) -> bool:
    wanted = {
        ("VisionZip", "0.2"),
        ("VisionZip", "0.3"),
        ("VisionZip", "0.4"),
        ("VisionZip", "0.5"),
        ("FastV", "0.2"),
        ("FastV", "0.3"),
        ("FastV", "0.4"),
        ("FastV", "0.5"),
    }
    done = {
        (row["method"], ratio_key(row["ratio"]))
        for row in rows
        if row.get("model") == "LLaVA-1.5-7B"
        and row.get("implementation") == "official-algorithm port"
        and row.get("status") == "done"
    }
    return wanted.issubset(done)


def source_audit() -> dict[str, Any]:
    llava = (ROOT / "recap" / "llava_pruned_backend.py").read_text(encoding="utf-8")
    qwen = (ROOT / "recap" / "qwen_pruned_backend.py").read_text(encoding="utf-8")
    internvl = (ROOT / "recap" / "internvl_pruned_backend.py").read_text(encoding="utf-8")
    cli = (ROOT / "recap" / "cli.py").read_text(encoding="utf-8")
    return {
        "llava_has_visionzip_branch": bool(re.search(r"def _forward_visionzip_pruned_llava", llava)),
        "llava_has_fastv_branch": bool(re.search(r"def _forward_fastv_pruned_llava", llava)),
        "llava_records_visionzip_commit": "JIA-Lab-research/VisionZip@8f86b55" in llava,
        "llava_records_fastv_commit": "pkunlp-icler/fastv@d165972" in llava,
        "qwen_has_visionzip_branch": "visionzip" in qwen.lower(),
        "qwen_has_fastv_branch": "fastv" in qwen.lower(),
        "internvl_has_visionzip_branch": "visionzip" in internvl.lower(),
        "internvl_has_fastv_branch": "fastv" in internvl.lower(),
        "llava_cli_mentions_official_ports": "LLaVA-only official-algorithm ports: visionzip, fastv, scope" in cli,
        "internvl_cli_mentions_official_ports": "official" in cli_section(cli, "run-internvl-pruned", "run-llava-direct").lower(),
    }


def build_markdown(summary_rows: list[dict[str, str]], pair_rows: list[dict[str, str]], source: dict[str, Any]) -> str:
    lines = [
        "# External Baseline Parity Audit",
        "",
        "This audit addresses the `problem.md` concern that external pruning baselines may be asymmetric across backbones.",
        "",
        "## Claim Boundary",
        "",
        "| Backbone | Official FastV/VisionZip status | What we can safely claim | What remains open |",
        "| --- | --- | --- | --- |",
        "| LLaVA-1.5-7B | Done for FastV and VisionZip at 0.20/0.30/0.40/0.50 | Main official-baseline evidence can be based on LLaVA | None for this backbone under TextOCR-Hard protocol |",
        "| Qwen3-VL-8B | Not claimed | Only protocol-compatible proxy rows | Native official ports would require separate method-specific adaptation |",
        "| InternVL3.5-8B | Not claimed | Only calibrated proxy rows | Native official ports would require method-specific access to the relevant attention/CLS-token path |",
        "",
        "## Source-Level Evidence",
        "",
        "| Check | Value |",
        "| --- | ---: |",
    ]
    for key, value in source.items():
        lines.append(f"| `{key}` | {value} |")

    lines.extend(
        [
            "",
            "## LLaVA Official-Port Results",
            "",
            "| Method | Ratio | Acc. | hFPR | Yes rate | ECR | Path |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in sorted(llava_official_rows(summary_rows), key=lambda item: (float(item["ratio"]), item["method"])):
        lines.append(
            "| {method} | {ratio} | {acc} | {hfpr} | {yes} | {ecr} | `{path}` |".format(
                method=row["method"],
                ratio=fmt(row["ratio"]),
                acc=fmt(row["acc"]),
                hfpr=fmt(row["hFPR"]),
                yes=fmt(row["yes_rate"]),
                ecr=fmt(row["ECR"]),
                path=row["path"],
            )
        )

    lines.extend(
        [
            "",
            "## Same-Budget Paired Readout",
            "",
            "| Baseline | Ratio | N | Acc. diff vs ours | hFPR diff vs ours | Interpretation |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in sorted(pair_rows, key=lambda item: (float(item["ratio"]), item["right_method"])):
        lines.append(
            "| {method} | {ratio} | {n} | {acc} | {hfpr} | {interp} |".format(
                method=row["right_method"],
                ratio=fmt(row["ratio"]),
                n=row["n"],
                acc=fmt_signed(row["acc_diff"]),
                hfpr=fmt_signed(row["hFPR_diff"]),
                interp=interpret_pair(row),
            )
        )

    lines.extend(
        [
            "",
            "## Safe Manuscript Wording",
            "",
            "Use wording like: *We evaluate method-specific FastV and VisionZip ports on LLaVA, where the required attention hooks are available. For Qwen and InternVL, we additionally report protocol-compatible proxy rows and label them as proxies rather than official implementations.*",
            "",
            "Avoid wording like: *We run official FastV/VisionZip on all three backbones.* The current code and evidence do not support that claim.",
            "",
            "## Decision",
            "",
            "This substantially reduces the overclaim risk but does not fully solve external-baseline parity. The paper can safely claim strong LLaVA official-port coverage and scoped Qwen/InternVL proxy checks. A fully closed Strong-Accept version would still need native official ports or collaboration-grade faithful adaptations for Qwen and InternVL.",
        ]
    )
    return "\n".join(lines) + "\n"


def llava_official_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("model") == "LLaVA-1.5-7B"
        and row.get("implementation") == "official-algorithm port"
        and row.get("status") == "done"
    ]


def interpret_pair(row: dict[str, str]) -> str:
    acc = float(row["acc_diff"])
    hfpr = float(row["hFPR_diff"])
    if row["right_method"] == "FastV" and hfpr > 0:
        return "Ours has higher accuracy, but FastV is all-no; hFPR alone is misleading."
    if acc > 0 and hfpr < 0:
        return "Ours is better on both accuracy and hFPR."
    if acc > 0:
        return "Ours has higher accuracy; inspect hFPR/yes-rate trade-off."
    return "No clear win."


def cli_section(text: str, start: str, end: str) -> str:
    start_idx = text.find(start)
    if start_idx < 0:
        return ""
    end_idx = text.find(end, start_idx + len(start))
    return text[start_idx:] if end_idx < 0 else text[start_idx:end_idx]


def ratio_key(value: str) -> str:
    return str(float(value)).rstrip("0").rstrip(".")


def fmt(value: str) -> str:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return value


def fmt_signed(value: str) -> str:
    return f"{float(value):+.3f}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    main()
