#!/usr/bin/env python
"""Write a compact status report for the remaining executable experiment queue."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "future_experiments"


@dataclass(frozen=True)
class ExpectedRun:
    stage: str
    name: str
    path: Path
    kind: str = "score"
    note: str = ""


EXPECTED_RUNS = [
    ExpectedRun(
        "relation_direct",
        "llava_vsr_profile_fast",
        Path("runs/rice_v5/llava15_7b_profile_fast_vsr_other_relations"),
        note="LLaVA direct VSR spatial/profile-fast coverage.",
    ),
    ExpectedRun(
        "relation_direct",
        "llava_whatsup_profile_fast",
        Path("runs/rice_v5/llava15_7b_profile_fast_whatsup_controlled_other_relations"),
        note="LLaVA direct What'sUp controlled/profile-fast coverage.",
    ),
    ExpectedRun(
        "relation_direct",
        "llava_gsr_coco_profile_fast",
        Path("runs/rice_v5/llava15_7b_profile_fast_gsrbench_coco_spatial_two"),
        note="LLaVA direct GSR-Bench COCO spatial/profile-fast coverage.",
    ),
    ExpectedRun(
        "relation_direct",
        "internvl_whatsup_profile_fast",
        Path("runs/rice_v5/internvl3_5_8b_profile_fast_whatsup_controlled_other_relations"),
        note="Completes InternVL relation-direct coverage for available Tier-0 data.",
    ),
    ExpectedRun(
        "causal_textocr",
        "llava_evidence_topk0p40",
        Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_topk_0p40"),
        kind="pruned",
        note="LLaVA evidence sufficiency under matched 40% keep.",
    ),
    ExpectedRun(
        "causal_textocr",
        "llava_bottomk0p40",
        Path("runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_bottomk_0p40"),
        kind="pruned",
        note="LLaVA anti-evidence necessity under matched 40% keep.",
    ),
    ExpectedRun(
        "causal_textocr",
        "internvl_evidence_topk0p50_raw",
        Path("runs/internvl_textocr_hard/internvl35_8b_textocr_hard_full1000_topk_0p50"),
        kind="pruned",
        note="InternVL raw evidence sufficiency under matched 50% keep.",
    ),
    ExpectedRun(
        "causal_textocr",
        "internvl_bottomk0p50_raw",
        Path("runs/internvl_textocr_hard/internvl35_8b_textocr_hard_full1000_bottomk_0p50"),
        kind="pruned",
        note="InternVL raw anti-evidence necessity under matched 50% keep.",
    ),
    ExpectedRun(
        "causal_textocr",
        "internvl_evidence_topk0p50_cal",
        Path("runs/internvl_textocr_hard/calibrated_test_topk0p50_devthr"),
        note="InternVL calibrated-test evidence sufficiency.",
    ),
    ExpectedRun(
        "causal_textocr",
        "internvl_bottomk0p50_cal",
        Path("runs/internvl_textocr_hard/calibrated_test_bottomk0p50_devthr"),
        note="InternVL calibrated-test anti-evidence necessity.",
    ),
    ExpectedRun(
        "spatial_pruned",
        "llava_gsr_embed0p40",
        Path("runs/llava_prune/llava15_7b_gsr_coco_spatial_profile_fast_embed_topk_0p40"),
        kind="pruned",
        note="LLaVA GSR-Bench pruning transfer: semantic selector.",
    ),
    ExpectedRun(
        "spatial_pruned",
        "llava_gsr_grid0p40",
        Path("runs/llava_prune/llava15_7b_gsr_coco_spatial_profile_fast_grid_0p40"),
        kind="pruned",
        note="LLaVA GSR-Bench pruning transfer: spatial baseline.",
    ),
    ExpectedRun(
        "spatial_pruned",
        "internvl_gsr_target0p50",
        Path("runs/internvl_prune/internvl35_8b_gsr_coco_spatial_profile_fast_target_embed_topk_0p50"),
        kind="pruned",
        note="InternVL GSR-Bench pruning transfer: target-conditioned selector.",
    ),
    ExpectedRun(
        "spatial_pruned",
        "internvl_gsr_grid0p50",
        Path("runs/internvl_prune/internvl35_8b_gsr_coco_spatial_profile_fast_grid_0p50"),
        kind="pruned",
        note="InternVL GSR-Bench pruning transfer: spatial baseline.",
    ),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [row_for_run(run) for run in EXPECTED_RUNS]
    write_csv(OUT_DIR / "future_experiment_status.csv", rows)
    (OUT_DIR / "future_experiment_status.md").write_text(markdown(rows))
    print(f"Wrote {OUT_DIR / 'future_experiment_status.md'}")
    print(f"Wrote {OUT_DIR / 'future_experiment_status.csv'}")


def row_for_run(run: ExpectedRun) -> dict[str, str]:
    run_dir = ROOT / run.path
    required = ["metrics.json", "probe_scores.jsonl"]
    if run.kind == "pruned":
        required.append("prune_traces.jsonl")
    missing = [name for name in required if not (run_dir / name).exists()]
    return {
        "stage": run.stage,
        "name": run.name,
        "status": "done" if not missing else "missing",
        "kind": run.kind,
        "path": str(run.path),
        "missing": ",".join(missing),
        "note": run.note,
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["stage", "name", "status", "kind", "path", "missing", "note"])
        writer.writeheader()
        writer.writerows(rows)


def markdown(rows: list[dict[str, str]]) -> str:
    done = sum(1 for row in rows if row["status"] == "done")
    lines = [
        "# Future Experiment Status",
        "",
        f"Executable local queue: {done}/{len(rows)} done.",
        "",
        "| stage | name | status | missing | path | note |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['stage']} | {row['name']} | {row['status']} | "
            f"{row['missing'] or '-'} | `{row['path']}` | {row['note']} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
