#!/usr/bin/env python3
"""Freeze the auto-screened V3 text-replacement probes for model evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACK = ROOT / "runs/problem_optimization_audit/text_replacement_control_pack_v3"
EXPECTED_PROBES = {
    "counterfactual_source_absent",
    "counterfactual_replacement_present",
    "counterfactual_sham_source",
    "counterfactual_sham_replacement",
    "counterfactual_erase_source",
    "counterfactual_erase_replacement",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probes", default=str(DEFAULT_PACK / "probes.jsonl"))
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_PACK / "human_qc_launch/text_replacement_human_qc_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_PACK / "cross_backbone_eval"),
    )
    args = parser.parse_args()

    manifest_rows = read_csv(Path(args.manifest))
    selected_ids = {
        str(row["image_id"])
        for row in manifest_rows
        if truthy(row.get("auto_control_render_pass", ""))
    }
    if len(selected_ids) != len(manifest_rows):
        raise ValueError(
            f"Expected every launch row to pass automatic controls; "
            f"got {len(selected_ids)}/{len(manifest_rows)}"
        )

    selected = [
        row for row in read_jsonl(Path(args.probes))
        if str(row.get("image_id", "")) in selected_ids
    ]
    validate(selected, selected_ids)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "probes.jsonl", selected)
    summary = {
        "manifest_rows": len(manifest_rows),
        "selected_images": len(selected_ids),
        "selected_probes": len(selected),
        "probes_per_image": 6,
        "selection_rule": "V3 automatic replacement/sham/erase control-render pass",
        "human_qc_status": "pending; model scores must be filtered after QC before claim promotion",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


def validate(rows: list[dict[str, Any]], selected_ids: set[str]) -> None:
    by_image: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_image.setdefault(str(row.get("image_id", "")), []).append(row)
        image = ROOT / str(row.get("image", ""))
        if not image.is_file():
            raise FileNotFoundError(f"Missing counterfactual image: {image}")
    if set(by_image) != selected_ids:
        missing = sorted(selected_ids - set(by_image))
        extra = sorted(set(by_image) - selected_ids)
        raise ValueError(f"Probe/manifest image mismatch: missing={missing[:5]}, extra={extra[:5]}")
    for image_id, image_rows in by_image.items():
        probes = Counter(str(row.get("probe", "")) for row in image_rows)
        if set(probes) != EXPECTED_PROBES or any(count != 1 for count in probes.values()):
            raise ValueError(f"Incomplete six-probe control set for {image_id}: {dict(probes)}")


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
