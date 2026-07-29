#!/usr/bin/env python
"""Build blank-image and collision-free image-mismatch TextOCR-Hard controls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from PIL import Image


Normalizer = Callable[[str], str]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/textocr_val_hard_confirmation_500img_seed20260720.jsonl",
    )
    parser.add_argument(
        "--regions",
        default="data/textocr_val_regions_full.jsonl",
        help="TextOCR regions containing all OCR strings for collision checks.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/textocr_image_free_controls/confirmation",
    )
    parser.add_argument("--seeds", default="101,202,303")
    parser.add_argument("--blank-value", type=int, default=127)
    args = parser.parse_args()

    if not 0 <= args.blank_value <= 255:
        raise ValueError("--blank-value must be in [0, 255]")

    probes = read_jsonl(args.input)
    groups = group_probe_pairs(probes)
    region_tokens = load_region_tokens(args.regions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    blank_dir = output_dir / "blank_images"
    blank_paths = build_blank_images(groups, blank_dir, args.blank_value)
    blank_rows = transform_rows(
        probes,
        groups,
        {image_id: image_id for image_id in groups},
        {image_id: blank_paths[image_id] for image_id in groups},
        control_type="blank",
        seed=None,
    )
    blank_output = output_dir / "blank.jsonl"
    write_jsonl(blank_output, blank_rows)

    mapping_rows: list[dict[str, Any]] = []
    outputs = {"blank": file_record(blank_output)}
    for seed in parse_seeds(args.seeds):
        mapping = build_collision_free_derangement(groups, region_tokens, seed)
        validate_derangement(mapping, groups, region_tokens)
        mismatched_paths = {
            source_id: str(groups[target_id][0]["image"])
            for source_id, target_id in mapping.items()
        }
        rows = transform_rows(
            probes,
            groups,
            mapping,
            mismatched_paths,
            control_type="image_mismatch",
            seed=seed,
        )
        output = output_dir / f"image_mismatch_seed{seed}.jsonl"
        write_jsonl(output, rows)
        outputs[f"image_mismatch_seed{seed}"] = file_record(output)
        for source_id in sorted(mapping):
            target_id = mapping[source_id]
            mapping_rows.append(
                {
                    "seed": seed,
                    "source_image_id": source_id,
                    "control_image_id": target_id,
                    "source_image": groups[source_id][0]["image"],
                    "control_image": groups[target_id][0]["image"],
                }
            )

    mapping_path = output_dir / "image_mismatch_mapping.csv"
    write_csv(mapping_path, mapping_rows)
    outputs["mapping"] = file_record(mapping_path)

    summary = {
        "input": str(Path(args.input).resolve()),
        "input_sha256": sha256_file(args.input),
        "regions": str(Path(args.regions).resolve()),
        "regions_sha256": sha256_file(args.regions),
        "num_probes": len(probes),
        "num_images": len(groups),
        "seeds": parse_seeds(args.seeds),
        "blank_value": args.blank_value,
        "collision_normalizers": list(NORMALIZERS),
        "pair_contract": (
            "Positive and negative probes from one source image share the same "
            "control image; image_id remains the source group for paired analysis."
        ),
        "outputs": outputs,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(probes)} probes x {1 + len(summary['seeds'])} controls "
        f"for {len(groups)} source images to {output_dir}"
    )


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty mapping")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must contain distinct integers")
    return seeds


def group_probe_pairs(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        image_id = str(row.get("image_id", ""))
        if not image_id:
            raise ValueError(f"Missing image_id in {row.get('sample_id')}")
        groups[image_id].append(row)
    for image_id, pair in groups.items():
        labels = sorted(str(row.get("target_answer", "")).lower() for row in pair)
        images = {str(row.get("image", "")) for row in pair}
        if len(pair) != 2 or labels != ["no", "yes"] or len(images) != 1:
            raise ValueError(
                f"Expected one positive/negative pair with one image for {image_id}"
            )
    return dict(groups)


def build_blank_images(
    groups: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    value: int,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for image_id, pair in groups.items():
        source = Path(pair[0]["image"])
        output = output_dir / f"{image_id}.jpg"
        with Image.open(source) as image:
            blank = Image.new("RGB", image.size, (value, value, value))
        blank.save(output, format="JPEG", quality=90, optimize=True)
        paths[image_id] = str(output.resolve())
    return paths


def load_region_tokens(path: str | Path) -> dict[str, dict[str, set[str]]]:
    by_image: dict[str, dict[str, set[str]]] = {}
    for row in read_jsonl(path):
        image_id = str(row.get("image_id", ""))
        if not image_id:
            continue
        strings = [
            str(token.get("text", ""))
            for token in row.get("ocr_tokens", [])
            if isinstance(token, dict)
        ]
        by_image[image_id] = {
            name: {normalizer(text) for text in strings if normalizer(text)}
            for name, normalizer in NORMALIZERS.items()
        }
    return by_image


def forbidden_strings(pair: list[dict[str, Any]]) -> dict[str, set[str]]:
    strings = {
        str(row.get(key, ""))
        for row in pair
        for key in ("target_text", "source_text")
    }
    return {
        name: {normalizer(text) for text in strings if normalizer(text)}
        for name, normalizer in NORMALIZERS.items()
    }


def assignment_allowed(
    source_id: str,
    target_id: str,
    groups: dict[str, list[dict[str, Any]]],
    region_tokens: dict[str, dict[str, set[str]]],
) -> bool:
    if source_id == target_id or target_id not in region_tokens:
        return False
    forbidden = forbidden_strings(groups[source_id])
    observed = region_tokens[target_id]
    return all(not forbidden[name].intersection(observed[name]) for name in NORMALIZERS)


def build_collision_free_derangement(
    groups: dict[str, list[dict[str, Any]]],
    region_tokens: dict[str, dict[str, set[str]]],
    seed: int,
    max_attempts: int = 1000,
) -> dict[str, str]:
    image_ids = sorted(groups)
    allowed = {
        source_id: [
            target_id
            for target_id in image_ids
            if assignment_allowed(source_id, target_id, groups, region_tokens)
        ]
        for source_id in image_ids
    }
    if any(not choices for choices in allowed.values()):
        missing = [source_id for source_id, choices in allowed.items() if not choices]
        raise RuntimeError(f"No valid mismatch candidates for {missing[:5]}")

    rng = random.Random(seed)
    for _ in range(max_attempts):
        jitter = {image_id: rng.random() for image_id in image_ids}
        order = sorted(image_ids, key=lambda image_id: (len(allowed[image_id]), jitter[image_id]))
        available = set(image_ids)
        mapping: dict[str, str] = {}
        for source_id in order:
            choices = [target_id for target_id in allowed[source_id] if target_id in available]
            if not choices:
                break
            target_id = rng.choice(choices)
            mapping[source_id] = target_id
            available.remove(target_id)
        if len(mapping) == len(image_ids):
            return mapping
    raise RuntimeError(f"Could not construct a valid derangement for seed {seed}")


def validate_derangement(
    mapping: dict[str, str],
    groups: dict[str, list[dict[str, Any]]],
    region_tokens: dict[str, dict[str, set[str]]],
) -> None:
    image_ids = set(groups)
    if set(mapping) != image_ids or set(mapping.values()) != image_ids:
        raise AssertionError("Mismatch mapping is not a permutation")
    for source_id, target_id in mapping.items():
        if not assignment_allowed(source_id, target_id, groups, region_tokens):
            raise AssertionError(f"Invalid mismatch {source_id} -> {target_id}")


def transform_rows(
    rows: list[dict[str, Any]],
    groups: dict[str, list[dict[str, Any]]],
    mapping: dict[str, str],
    paths: dict[str, str],
    *,
    control_type: str,
    seed: int | None,
) -> list[dict[str, Any]]:
    transformed = []
    for row in rows:
        source_id = str(row["image_id"])
        control_id = mapping[source_id]
        updated = dict(row)
        updated["original_image"] = row["image"]
        updated["control_type"] = control_type
        updated["control_seed"] = seed
        updated["control_image_id"] = control_id if control_type != "blank" else None
        updated["image"] = paths[source_id]
        transformed.append(updated)
    for source_id, pair in groups.items():
        transformed_paths = {
            row["image"]
            for row in transformed
            if str(row["image_id"]) == source_id
        }
        if len(pair) != 2 or len(transformed_paths) != 1:
            raise AssertionError(f"Pair path mismatch for {source_id}")
    return transformed


def file_record(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    return {
        "path": str(source.resolve()),
        "sha256": sha256_file(source),
        "bytes": source.stat().st_size,
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_trim(text: str) -> str:
    return text.strip()


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip())


def nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text.strip())


def nfkc_casefold(text: str) -> str:
    return nfkc(text).casefold()


def nfkc_casefold_nospace(text: str) -> str:
    return re.sub(r"\s+", "", nfkc_casefold(text))


def nfkc_casefold_alnum(text: str) -> str:
    return "".join(character for character in nfkc_casefold(text) if character.isalnum())


def ascii_fold_alnum(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return "".join(character for character in folded.casefold() if character.isalnum())


NORMALIZERS: dict[str, Normalizer] = {
    "raw_trim": raw_trim,
    "nfc": nfc,
    "nfkc": nfkc,
    "nfkc_casefold": nfkc_casefold,
    "nfkc_casefold_nospace": nfkc_casefold_nospace,
    "nfkc_casefold_alnum": nfkc_casefold_alnum,
    "ascii_fold_alnum": ascii_fold_alnum,
}


if __name__ == "__main__":
    main()
