"""VSR preparation for RECAP.

VSR rows are image-caption truth judgments. RECAP keeps left/right statements by
default and converts the original true/false label into a yes/no relation QA.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from recap.caption_parse import parse_spatial_caption
from recap.io import jsonable
from recap.relations import bool_to_answer, normalize_relation, relation_family

VSR_DATASETS = {
    "random": "cambridgeltl/vsr_random",
    "zeroshot": "cambridgeltl/vsr_zeroshot",
}


def prepare_vsr(
    *,
    split: str = "test",
    dataset_path: str = "cambridgeltl/vsr_random",
    metadata_file: str | None = None,
    coco_root: str | None = None,
    output_image_dir: str | None = None,
    download_images: bool = True,
    local_files_only: bool = False,
    cache_dir: str | None = None,
    hf_endpoint: str | None = None,
    limit: int | None = None,
    left_right_only: bool = True,
    relations: set[str] | None = None,
    relation_families: set[str] | None = None,
    allow_missing_images: bool = False,
) -> list[dict[str, Any]]:
    dataset_path = VSR_DATASETS.get(dataset_path, dataset_path)
    hf_endpoint = hf_endpoint or os.environ.get("HF_ENDPOINT") or None
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint

    if metadata_file:
        rows = _read_table(metadata_file, split=split)
    else:
        rows = _load_hf_rows(
            dataset_path=dataset_path,
            split=split,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            limit=limit,
        )
    if limit is not None:
        rows = rows[:limit]

    output_dir = Path(output_image_dir).expanduser() if output_image_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    canonical: list[dict[str, Any]] = []
    copied_or_downloaded: dict[str, str] = {}
    skipped_parse_failed = 0
    skipped_relation_filter = 0
    relation_counts: Counter[str] = Counter()
    for idx, row in enumerate(rows):
        caption = str(row.get("caption", row.get("sentence", row.get("text", "")))).strip()
        relation_hint = str(row.get("relation", "")).strip()
        parsed = parse_spatial_caption(caption, relation_hint=relation_hint, left_right_only=False)
        if parsed is None:
            skipped_parse_failed += 1
            continue
        relation = normalize_relation(parsed.relation)
        relation_counts[relation or "<empty>"] += 1
        if left_right_only and relation not in {"left_of", "right_of"}:
            skipped_relation_filter += 1
            continue
        if not _keep_relation(relation, relations=relations, relation_families=relation_families):
            skipped_relation_filter += 1
            continue

        image_name = str(row.get("image", row.get("image_id", row.get("image_file", "")))).strip()
        image_link = str(row.get("image_link", row.get("url", ""))).strip()
        image_path = _materialize_image(
            image_name=image_name,
            image_link=image_link,
            coco_root=coco_root,
            output_dir=output_dir,
            download_images=download_images,
            allow_missing_images=allow_missing_images,
            cache=copied_or_downloaded,
        )

        label = row.get("label", row.get("answer", row.get("target", False)))
        answer_bool = _label_to_bool(label)
        item = {
            "id": f"vsr-{_safe_id(dataset_path)}-{split}-{idx}",
            "dataset": "VSR",
            "source_dataset": dataset_path,
            "split": split,
            "image": image_path,
            "image_id": image_name,
            "image_url": image_link,
            "question": f"Is this statement true? {caption}",
            "source_caption": caption,
            "source_relation": relation_hint,
            "subject": parsed.subject,
            "object": parsed.object,
            "relation": relation,
            "answer": bool_to_answer(answer_bool),
            "binary_polarity": "positive" if answer_bool else "negative",
            "bbox_source": "none",
        }
        if "reference_frame" in row:
            item["reference_frame"] = row["reference_frame"]
        canonical.append(jsonable(item))

    print(
        f"VSR {dataset_path}/{split}: kept {len(canonical)} rows"
        + (f", filtered {skipped_relation_filter} non-left/right rows" if skipped_relation_filter else "")
        + (f", parse_failed {skipped_parse_failed} rows" if skipped_parse_failed else "")
        + (f", materialized {len(copied_or_downloaded)} images" if output_dir and download_images else "")
    )
    if left_right_only and relation_counts:
        kept_lr = relation_counts["left_of"] + relation_counts["right_of"]
        print(f"VSR relation filter summary: left/right={kept_lr}, other_relations={sum(relation_counts.values()) - kept_lr}")
    return canonical


def _keep_relation(relation: str, *, relations: set[str] | None, relation_families: set[str] | None) -> bool:
    if relations and relation not in relations:
        return False
    if relation_families and relation_family(relation) not in relation_families:
        return False
    return True


def _load_hf_rows(
    *,
    dataset_path: str,
    split: str,
    cache_dir: str | None,
    local_files_only: bool,
    limit: int | None,
) -> list[dict[str, Any]]:
    try:
        from datasets import DownloadConfig, load_dataset
    except ImportError as exc:
        raise ImportError("prepare-vsr needs `datasets` unless --metadata-file is provided.") from exc

    kwargs: dict[str, Any] = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if local_files_only:
        kwargs["download_config"] = DownloadConfig(local_files_only=True)
    ds = load_dataset(dataset_path, split=split, **kwargs)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return [dict(row) for row in ds]


def _read_table(path: str, *, split: str) -> list[dict[str, Any]]:
    source = Path(path).expanduser()
    suffix = source.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with source.open("r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
        if isinstance(payload, dict):
            payload = payload.get(split, payload.get("data", payload.get("examples", payload)))
        if isinstance(payload, list):
            return [dict(row) for row in payload]
        raise ValueError(f"Unsupported VSR JSON payload in {source}")
    if suffix == ".csv":
        with source.open("r", encoding="utf-8-sig", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("Reading local parquet VSR metadata needs `pandas`.") from exc
        return pd.read_parquet(source).to_dict("records")
    raise ValueError(f"Unsupported VSR metadata extension: {source.suffix}")


def _materialize_image(
    *,
    image_name: str,
    image_link: str,
    coco_root: str | None,
    output_dir: Path | None,
    download_images: bool,
    allow_missing_images: bool,
    cache: dict[str, str],
) -> str:
    key = image_name or image_link
    if key in cache:
        return cache[key]

    local = _find_coco_image(image_name, coco_root)
    if output_dir is not None and local is not None:
        target = output_dir / local.name
        if not target.exists():
            shutil.copyfile(local, target)
        cache[key] = str(target.resolve())
        return cache[key]
    if local is not None:
        cache[key] = str(local.resolve())
        return cache[key]

    if output_dir is not None and download_images and image_link:
        name = Path(image_name or image_link.split("?")[0]).name
        if not name:
            name = f"vsr_image_{len(cache):06d}.jpg"
        target = output_dir / name
        if not target.exists():
            _download_file(image_link, target)
        cache[key] = str(target.resolve())
        return cache[key]

    if not allow_missing_images:
        raise FileNotFoundError(
            "Could not materialize VSR image. "
            f"image={image_name!r}, image_link={image_link!r}, coco_root={coco_root!r}, output_dir={str(output_dir) if output_dir else ''!r}. "
            "Pass a valid --coco-root containing COCO images, pass --image-dir to download/copy images, "
            "or use --metadata-only/--allow-missing-images only for non-visual debugging."
        )

    return image_name or image_link


def _find_coco_image(image_name: str, coco_root: str | None) -> Path | None:
    if not image_name or not coco_root:
        return None
    root = Path(coco_root).expanduser()
    candidates = [
        root / image_name,
        root / "train2017" / image_name,
        root / "val2017" / image_name,
        root / "test2017" / image_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _download_file(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "rice-v5-data-prep"})
    with urllib.request.urlopen(request, timeout=60) as response, target.open("wb") as f:
        shutil.copyfileobj(response, f)


def _label_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in value.lower()).strip("-")
