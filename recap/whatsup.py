"""WhatsUp/What'sUp preparation for RECAP.

The official WhatsUp benchmark stores each image with caption options. The first
caption is the correct statement. RECAP converts every left/right caption option
into a binary yes/no relation sample.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from recap.caption_parse import parse_spatial_caption
from recap.io import jsonable
from recap.relations import bool_to_answer, normalize_relation, relation_family


@dataclass(frozen=True)
class WhatupConfig:
    name: str
    metadata_name: str
    metadata_id: str
    archive_name: str
    archive_id: str
    image_dir: str
    kind: str


WHATUP_CONFIGS: dict[str, WhatupConfig] = {
    "controlled_a": WhatupConfig(
        name="controlled_a",
        metadata_name="controlled_images_dataset.json",
        metadata_id="1ap8mmmpQjLIjPGuplkpBgc1hoEHCj4hm",
        archive_name="controlled_images.tar.gz",
        archive_id="19KGYVQjrV3syb00GgcavB2nZTW5NXX0H",
        image_dir="controlled_images",
        kind="controlled",
    ),
    "controlled_b": WhatupConfig(
        name="controlled_b",
        metadata_name="controlled_clevr_dataset.json",
        metadata_id="1unNNosLbdy9NDjgj4l8fsQP3WiAAGA6z",
        archive_name="controlled_clevr.tar.gz",
        archive_id="13jdBpg8t3NqW3jrL6FK8HO93vwsUjDxG",
        image_dir="controlled_clevr",
        kind="controlled",
    ),
    "coco_qa_one_obj": WhatupConfig(
        name="coco_qa_one_obj",
        metadata_name="coco_qa_one_obj.json",
        metadata_id="1RsMdpE9mmwnK4zzMPpC1-wTU_hNis-dq",
        archive_name="val2017.zip",
        archive_id="1zp5vBRRM4_nSik6o9PeVspDvOsHgPT4l",
        image_dir="val2017",
        kind="coco",
    ),
    "coco_qa_two_obj": WhatupConfig(
        name="coco_qa_two_obj",
        metadata_name="coco_qa_two_obj.json",
        metadata_id="1TCEoM0mgFmz8T4cF7PQ3XJmO6JjtiQ-s",
        archive_name="val2017.zip",
        archive_id="1zp5vBRRM4_nSik6o9PeVspDvOsHgPT4l",
        image_dir="val2017",
        kind="coco",
    ),
    "vg_qa_one_obj": WhatupConfig(
        name="vg_qa_one_obj",
        metadata_name="vg_qa_one_obj.json",
        metadata_id="1ARMRzRdohs9QTr1gpIfzyUzvW20wYp_p",
        archive_name="vg_images.tar.gz",
        archive_id="1idW7Buoz7fQm4-670n-oERw9U-2JLJvE",
        image_dir="vg_images",
        kind="vg",
    ),
    "vg_qa_two_obj": WhatupConfig(
        name="vg_qa_two_obj",
        metadata_name="vg_qa_two_obj.json",
        metadata_id="1sjVG5O3QMY8s118k7kQM8zzDZH12i_95",
        archive_name="vg_images.tar.gz",
        archive_id="1idW7Buoz7fQm4-670n-oERw9U-2JLJvE",
        image_dir="vg_images",
        kind="vg",
    ),
}

WHATUP_GROUPS = {
    "all_controlled": ("controlled_a", "controlled_b"),
    "all_qa": ("coco_qa_one_obj", "coco_qa_two_obj", "vg_qa_one_obj", "vg_qa_two_obj"),
    "all": tuple(WHATUP_CONFIGS),
}


def prepare_whatsup(
    *,
    dataset: str = "controlled_a",
    root_dir: str = "data/whatsup",
    metadata_file: str | None = None,
    image_root: str | None = None,
    download: bool = False,
    extract: bool = True,
    metadata_only: bool = False,
    left_right_only: bool = True,
    relations: set[str] | None = None,
    relation_families: set[str] | None = None,
    positive_only: bool = False,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    names = _expand_dataset_names(dataset)
    if metadata_file and len(names) != 1:
        raise ValueError("--metadata-file can only be used with one concrete WhatsUp dataset name.")

    root = Path(root_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)

    canonical: list[dict[str, Any]] = []
    for name in names:
        config = WHATUP_CONFIGS[name]
        metadata_path = (
            Path(metadata_file).expanduser()
            if metadata_file
            else _ensure_assets(
                config,
                root=root,
                download=download,
                download_archive=not metadata_only and image_root is None,
                extract=extract and not metadata_only,
            )
        )
        rows = _read_json(metadata_path)
        if limit is not None:
            rows = rows[:limit]
        canonical.extend(
            _canonicalize_dataset(
                rows,
                config=config,
                root=root,
                image_root=Path(image_root).expanduser() if image_root else None,
                metadata_only=metadata_only,
                left_right_only=left_right_only,
                relations=relations,
                relation_families=relation_families,
                positive_only=positive_only,
            )
        )

    print(f"WhatsUp {dataset}: kept {len(canonical)} rows from {len(names)} source set(s)")
    return canonical


def _expand_dataset_names(dataset: str) -> tuple[str, ...]:
    key = dataset.lower()
    if key in WHATUP_GROUPS:
        return WHATUP_GROUPS[key]
    if key in WHATUP_CONFIGS:
        return (key,)
    choices = sorted([*WHATUP_CONFIGS, *WHATUP_GROUPS])
    raise ValueError(f"Unknown WhatsUp dataset '{dataset}'. Use one of: {choices}")


def _ensure_assets(
    config: WhatupConfig,
    *,
    root: Path,
    download: bool,
    download_archive: bool,
    extract: bool,
) -> Path:
    metadata_path = root / config.metadata_name
    archive_path = root / config.archive_name
    if download:
        if not metadata_path.exists():
            _gdown(config.metadata_id, metadata_path)
        if download_archive and not archive_path.exists():
            _gdown(config.archive_id, archive_path)
    if extract and archive_path.exists() and not (root / config.image_dir).exists():
        _extract_archive(archive_path, root)
    if not metadata_path.exists():
        raise FileNotFoundError(f"WhatsUp metadata not found: {metadata_path}. Rerun with --download or pass --metadata-file.")
    return metadata_path


def _canonicalize_dataset(
    rows: list[Any],
    *,
    config: WhatupConfig,
    root: Path,
    image_root: Path | None,
    metadata_only: bool,
    left_right_only: bool,
    relations: set[str] | None,
    relation_families: set[str] | None,
    positive_only: bool,
) -> list[dict[str, Any]]:
    canonical: list[dict[str, Any]] = []
    skipped = 0
    for idx, row in enumerate(rows):
        for option_idx, case in enumerate(_iter_caption_cases(row, config=config)):
            if positive_only and not case["answer"]:
                continue
            parsed = parse_spatial_caption(case["caption"], left_right_only=left_right_only)
            if parsed is None:
                skipped += 1
                continue
            relation = normalize_relation(parsed.relation)
            if left_right_only and relation not in {"left_of", "right_of"}:
                skipped += 1
                continue
            if not _keep_relation(relation, relations=relations, relation_families=relation_families):
                skipped += 1
                continue
            image_path = case["image"]
            if not metadata_only:
                image_path = str(_resolve_image(image_path, config=config, root=root, image_root=image_root))
            answer = bool(case["answer"])
            canonical.append(
                jsonable(
                    {
                        "id": f"whatsup-{config.name}-{idx}-{option_idx}",
                        "dataset": "WhatsUp",
                        "source_dataset": config.name,
                        "split": "test",
                        "image": image_path,
                        "image_id": case.get("image_id", ""),
                        "question": f"Is this statement true? {case['caption']}",
                        "source_caption": case["caption"],
                        "source_caption_options": case.get("caption_options", []),
                        "option_index": option_idx,
                        "choice_group_id": f"whatsup-{config.name}-{idx}",
                        "choice_index": option_idx,
                        "choice_is_correct": answer,
                        "task_form": "caption_option_binary",
                        "reference_frame": "viewer",
                        "subject": parsed.subject,
                        "object": parsed.object,
                        "relation": relation,
                        "answer": bool_to_answer(answer),
                        "binary_polarity": "positive" if answer else "negative",
                        "bbox_source": "none",
                    }
                )
            )
    if skipped:
        print(f"WhatsUp {config.name}: skipped {skipped} caption options outside the requested relation set")
    return canonical


def _keep_relation(relation: str, *, relations: set[str] | None, relation_families: set[str] | None) -> bool:
    if relations and relation not in relations:
        return False
    if relation_families and relation_family(relation) not in relation_families:
        return False
    return True


def _iter_caption_cases(row: Any, *, config: WhatupConfig) -> Iterable[dict[str, Any]]:
    if isinstance(row, dict):
        image = str(row.get("image_path", row.get("image", row.get("image_id", ""))))
        captions = row.get("caption_options", row.get("captions", row.get("options", [])))
        if isinstance(captions, str):
            captions = [captions]
        for idx, caption in enumerate(captions):
            yield {
                "image": image,
                "image_id": row.get("image_id", image),
                "caption": str(caption),
                "caption_options": captions,
                "answer": idx == 0,
            }
        return

    if isinstance(row, (list, tuple)) and len(row) >= 3:
        image_id = str(row[0])
        image = _image_name_from_id(image_id, config=config)
        captions = [str(caption) for caption in row[1:]]
        for idx, caption in enumerate(captions):
            yield {
                "image": image,
                "image_id": image_id,
                "caption": caption,
                "caption_options": captions,
                "answer": idx == 0,
            }
        return

    raise ValueError(f"Unsupported WhatsUp row format for {config.name}: {row}")


def _image_name_from_id(image_id: str, *, config: WhatupConfig) -> str:
    if config.kind == "coco":
        if image_id.lower().endswith((".jpg", ".jpeg", ".png")):
            return image_id
        return f"{int(image_id):012d}.jpg" if image_id.isdigit() else f"{image_id}.jpg"
    if config.kind == "vg":
        return image_id if image_id.lower().endswith((".jpg", ".jpeg", ".png")) else f"{image_id}.jpg"
    return image_id


def _resolve_image(image: str, *, config: WhatupConfig, root: Path, image_root: Path | None) -> Path:
    raw = Path(image)
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    if image_root is not None:
        candidates.extend([image_root / image, image_root / raw.name])
    candidates.extend(
        [
            root / image,
            root / config.image_dir / image,
            root / config.image_dir / raw.name,
        ]
    )
    if image.startswith("data/"):
        candidates.append(root / image.removeprefix("data/"))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"WhatsUp image not found for '{image}'. Searched: {searched}")


def _read_json(path: Path) -> list[Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "examples", "test"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError(f"Unsupported WhatsUp JSON payload in {path}")


def _gdown(file_id: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    commands = [
        [sys.executable, "-m", "gdown", "--id", file_id, "--output", str(output)],
        ["gdown", "--id", file_id, "--output", str(output)],
    ]
    last_error: Exception | None = None
    for command in commands:
        try:
            subprocess.run(command, check=True)
            return
        except (OSError, subprocess.CalledProcessError) as exc:
            last_error = exc
    raise RuntimeError("Could not download WhatsUp asset with gdown. Install it via `pip install gdown`.") from last_error


def _extract_archive(archive_path: Path, root: Path) -> None:
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as zf:
            _safe_extract_zip(zf, root)
        return
    if archive_path.name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive_path) as tf:
            _safe_extract_tar(tf, root)
        return
    raise ValueError(f"Unsupported WhatsUp archive type: {archive_path}")


def _safe_extract_zip(zf: zipfile.ZipFile, root: Path) -> None:
    root = root.resolve()
    for name in zf.namelist():
        target = (root / name).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"Refusing to extract path outside {root}: {name}") from exc
    zf.extractall(root)


def _safe_extract_tar(tf: tarfile.TarFile, root: Path) -> None:
    root = root.resolve()
    for member in tf.getmembers():
        target = (root / member.name).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"Refusing to extract path outside {root}: {member.name}") from exc
    tf.extractall(root)
