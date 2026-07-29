#!/usr/bin/env python
"""Compare the local SCOPE selector against the pinned official source."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from recap.llava_pruned_backend import _scope_select_indices


SCOPE_ROOT = ROOT / "third_party" / "SCOPE"
OFFICIAL_SOURCE = SCOPE_ROOT / "scope" / "clip_encoder.py"
SMOKE_TRACE = (
    ROOT
    / "runs"
    / "llava_textocr_hard"
    / "llava15_7b_textocr_hard_scope_0p40_smoke2"
    / "prune_traces.jsonl"
)
OUT_DIR = ROOT / "runs" / "official_baseline_extension" / "scope_port_parity"


def main() -> None:
    official_scope = load_official_scope()
    cases = compare_selectors(official_scope)
    smoke = audit_smoke_trace()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=SCOPE_ROOT,
        text=True,
    ).strip()
    source_sha256 = hashlib.sha256(OFFICIAL_SOURCE.read_bytes()).hexdigest()
    passed = all(case["exact_index_match"] for case in cases) and all(smoke.values())
    report = {
        "status": "pass" if passed else "fail",
        "official_repository": "https://github.com/kinredon/SCOPE",
        "official_commit": commit,
        "official_source": str(OFFICIAL_SOURCE.relative_to(ROOT)),
        "official_source_sha256": source_sha256,
        "official_defaults": {"ALPHA": 1.0, "COMBINED": "multi"},
        "cases": cases,
        "smoke_checks": smoke,
        "claim_boundary": (
            "Source-compatible official-algorithm port in the Hugging Face LLaVA backend; "
            "not a number copied from the authors' evaluation logs."
        ),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "scope_port_parity.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT_DIR / "scope_port_parity.md").write_text(build_markdown(report))
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


def load_official_scope():
    tree = ast.parse(OFFICIAL_SOURCE.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "SCOPE"]
    if len(functions) != 1:
        raise RuntimeError(f"Expected one official SCOPE function, found {len(functions)}")
    module = ast.Module(body=functions, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {"torch": torch, "os": os}
    exec(compile(module, str(OFFICIAL_SOURCE), "exec"), namespace)
    return namespace["SCOPE"]


def compare_selectors(official_scope) -> list[dict[str, Any]]:
    old_alpha = os.environ.get("ALPHA")
    old_combined = os.environ.get("COMBINED")
    os.environ["ALPHA"] = "1.0"
    os.environ["COMBINED"] = "multi"
    cases: list[dict[str, Any]] = []
    try:
        for seed, batch_size, num_tokens, width, keep_count in (
            (3, 1, 8, 5, 1),
            (7, 1, 17, 9, 7),
            (11, 2, 13, 6, 5),
            (19, 3, 9, 4, 9),
        ):
            generator = torch.Generator().manual_seed(seed)
            features = torch.randn(batch_size, num_tokens, width, generator=generator)
            attention = torch.rand(batch_size, num_tokens, generator=generator)
            expected, _ = official_scope(features.clone(), keep_count, attention.clone())
            actual = _scope_select_indices(features, attention, keep_count=keep_count, alpha=1.0)
            cases.append(
                {
                    "seed": seed,
                    "batch_size": batch_size,
                    "num_tokens": num_tokens,
                    "feature_width": width,
                    "keep_count": keep_count,
                    "exact_index_match": bool(torch.equal(actual.cpu(), expected.cpu())),
                }
            )
    finally:
        restore_env("ALPHA", old_alpha)
        restore_env("COMBINED", old_combined)
    return cases


def audit_smoke_trace() -> dict[str, bool]:
    rows = [json.loads(line) for line in SMOKE_TRACE.read_text().splitlines() if line.strip()]
    return {
        "two_smoke_rows": len(rows) == 2,
        "fixed_231_of_576_budget": all(
            int(row.get("full_visual_tokens", 0)) == 576 and int(row.get("kept_visual_tokens", 0)) == 231
            for row in rows
        ),
        "scope_source_recorded": all(row.get("scope_commit") == "kinredon/SCOPE@6bf7306" for row in rows),
        "original_order_materialized": all(
            row.get("kept_indices") == sorted(row.get("kept_indices", [])) for row in rows
        ),
        "paired_probe_mask_stable": len(rows) == 2 and rows[0].get("kept_indices") == rows[1].get("kept_indices"),
    }


def restore_env(key: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SCOPE Port Parity Audit",
        "",
        f"Status: **{report['status']}**",
        "",
        f"Official source: `{report['official_source']}` at `{report['official_commit']}`.",
        "",
        "| Seed | Batch | Tokens | Width | Keep | Exact index match |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for case in report["cases"]:
        lines.append(
            f"| {case['seed']} | {case['batch_size']} | {case['num_tokens']} | "
            f"{case['feature_width']} | {case['keep_count']} | {case['exact_index_match']} |"
        )
    lines.extend(["", "## Smoke Checks", ""])
    lines.extend(f"- `{key}`: {value}" for key, value in report["smoke_checks"].items())
    lines.extend(["", "## Claim Boundary", "", report["claim_boundary"], ""])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
