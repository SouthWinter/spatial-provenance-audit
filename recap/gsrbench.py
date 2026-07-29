"""Prepare the non-overlapping two-object GSR-Bench extension for RECAP.

GSR-Bench extends What'sUp with grounding annotations.  RECAP consumes only
the RGB image and relation caption, so this adapter reuses the official
What'sUp metadata/assets while exposing the COCO-Spatial-Two and
GQA-Spatial-Two splits that are not part of our controlled What'sUp run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from recap.whatsup import prepare_whatsup


@dataclass(frozen=True)
class GSRBenchConfig:
    name: str
    whatsup_dataset: str
    paper_split: str


GSRBENCH_CONFIGS: dict[str, GSRBenchConfig] = {
    "coco_spatial_two": GSRBenchConfig(
        name="coco_spatial_two",
        whatsup_dataset="coco_qa_two_obj",
        paper_split="COCO-Spatial-Two",
    ),
    "gqa_spatial_two": GSRBenchConfig(
        name="gqa_spatial_two",
        whatsup_dataset="vg_qa_two_obj",
        paper_split="GQA-Spatial-Two",
    ),
}

GSRBENCH_GROUPS: dict[str, tuple[str, ...]] = {
    "external_two_object": ("coco_spatial_two", "gqa_spatial_two"),
}


def prepare_gsrbench(
    *,
    dataset: str = "external_two_object",
    root_dir: str = "data/whatsup",
    metadata_file: str | None = None,
    image_root: str | None = None,
    coco_root: str | None = None,
    gqa_root: str | None = None,
    download: bool = False,
    extract: bool = True,
    metadata_only: bool = False,
    left_right_only: bool = False,
    relations: set[str] | None = None,
    relation_families: set[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Convert GSR-Bench's external two-object splits to canonical JSONL rows.

    The controlled Subset A/B images are intentionally unavailable here
    because they are already evaluated as What'sUp in the main experiment.
    ``limit`` follows the existing What'sUp behavior and applies per split.
    """

    names = _expand_dataset_names(dataset)
    if metadata_file and len(names) != 1:
        raise ValueError("--metadata-file requires one concrete GSR-Bench split.")

    canonical: list[dict[str, Any]] = []
    for name in names:
        config = GSRBENCH_CONFIGS[name]
        split_image_root = _split_image_root(
            config,
            image_root=image_root,
            coco_root=coco_root,
            gqa_root=gqa_root,
        )
        rows = prepare_whatsup(
            dataset=config.whatsup_dataset,
            root_dir=root_dir,
            metadata_file=metadata_file,
            image_root=split_image_root,
            download=download,
            extract=extract,
            metadata_only=metadata_only,
            left_right_only=left_right_only,
            relations=relations,
            relation_families=relation_families,
            positive_only=False,
            limit=limit,
        )
        canonical.extend(_mark_as_gsrbench(row, config=config) for row in rows)

    print(f"GSR-Bench {dataset}: kept {len(canonical)} binary relation rows from {len(names)} split(s)")
    return canonical


def _split_image_root(
    config: GSRBenchConfig,
    *,
    image_root: str | None,
    coco_root: str | None,
    gqa_root: str | None,
) -> str | None:
    if image_root:
        return image_root
    if config.name == "coco_spatial_two" and coco_root:
        root = Path(coco_root).expanduser()
        val_dir = root / "val2017"
        return str(val_dir if val_dir.is_dir() else root)
    if config.name == "gqa_spatial_two" and gqa_root:
        root = Path(gqa_root).expanduser()
        vg_dir = root / "vg_images"
        return str(vg_dir if vg_dir.is_dir() else root)
    return None


def _expand_dataset_names(dataset: str) -> tuple[str, ...]:
    key = str(dataset).strip().lower().replace("-", "_")
    if key in GSRBENCH_GROUPS:
        return GSRBENCH_GROUPS[key]
    if key in GSRBENCH_CONFIGS:
        return (key,)
    choices = sorted([*GSRBENCH_CONFIGS, *GSRBENCH_GROUPS])
    raise ValueError(f"Unknown GSR-Bench split '{dataset}'. Use one of: {choices}")


def _mark_as_gsrbench(row: dict[str, Any], *, config: GSRBenchConfig) -> dict[str, Any]:
    sample = dict(row)
    source_id = str(sample.get("id", ""))
    sample["id"] = f"gsrbench-{config.name}-{source_id}"
    sample["dataset"] = "GSR-Bench"
    sample["source_dataset"] = config.name
    sample["source_split_name"] = config.paper_split
    sample["evaluation_scope"] = "external_two_object"
    sample["uses_grounding_annotations"] = False
    sample["grounding_protocol"] = "rgb_only"
    sample["overlaps_main_whatsup_controlled"] = False
    return sample
