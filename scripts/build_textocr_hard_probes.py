#!/usr/bin/env python
"""Build harder TextOCR exact-text yes/no probes from OCR word boxes."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any


DEVELOPMENT_V1 = "development-v1"
CONFIRMATION_V2 = "confirmation-v2"
CONSTRUCTION_VERSIONS = (DEVELOPMENT_V1, CONFIRMATION_V2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="TextOCR region JSONL from recap.textocr_regions.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit-images", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--min-token-len", type=int, default=4)
    parser.add_argument("--max-token-len", type=int, default=18)
    parser.add_argument("--min-area", type=float, default=0.00005)
    parser.add_argument("--max-area", type=float, default=0.0012)
    parser.add_argument("--pool-per-image", type=int, default=8)
    parser.add_argument(
        "--construction-version",
        choices=CONSTRUCTION_VERSIONS,
        default=CONFIRMATION_V2,
        help=(
            "Versioned decoy construction. development-v1 reproduces the locked "
            "development split; confirmation-v2 is the Unicode-safe revision used "
            "for the locked confirmation split."
        ),
    )
    parser.add_argument(
        "--expected-sha256",
        default="",
        help="Optional expected output SHA-256; fail after writing if it differs.",
    )
    parser.add_argument("--require-unique-token", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-negative", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--exclude-probes",
        default="",
        help="Optional probe JSONL whose image IDs must not appear in the output.",
    )
    parser.add_argument(
        "--shuffle-images",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Deterministically shuffle source images before selecting eligible examples.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = read_jsonl(args.input)
    excluded_image_ids = (
        image_ids_from_probes(read_jsonl(args.exclude_probes)) if args.exclude_probes else set()
    )
    if args.shuffle_images:
        rng.shuffle(rows)

    probes: list[dict[str, Any]] = []
    selected_images = 0
    for row_index, row in enumerate(rows):
        if str(row.get("image_id", "")) in excluded_image_ids:
            continue
        token = choose_hard_token(
            row,
            min_len=args.min_token_len,
            max_len=args.max_token_len,
            min_area=args.min_area,
            max_area=args.max_area,
            pool_size=args.pool_per_image,
            require_unique=args.require_unique_token,
            rng=rng,
        )
        if token is None:
            continue

        text, box, area, area_rank = token
        row_id = str(row.get("id", row_index))
        probes.append(
            make_probe(
                row,
                sample_id=f"{row_id}:hard-small-pos",
                question=f'Does the image contain the exact text "{text}"? Answer yes or no.',
                target_answer="yes",
                target_text=text,
                evidence_box=box,
                hard_type="small_positive",
                source_text=text,
                token_area=area,
                token_area_rank=area_rank,
            )
        )

        if args.include_negative:
            present = present_token_set(row)
            decoy = mutate_token(
                text,
                present=present,
                rng=rng,
                construction_version=args.construction_version,
            )
            if decoy:
                probes.append(
                    make_probe(
                        row,
                        sample_id=f"{row_id}:hard-nearmiss-neg",
                        question=f'Does the image contain the exact text "{decoy}"? Answer yes or no.',
                        target_answer="no",
                        target_text=decoy,
                        evidence_box=box,
                        hard_type="near_miss_negative",
                        source_text=text,
                        token_area=area,
                        token_area_rank=area_rank,
                    )
                )

        selected_images += 1
        if args.limit_images is not None and selected_images >= args.limit_images:
            break

    write_jsonl(args.output, probes)
    digest = sha256_file(args.output)
    if args.expected_sha256 and digest.lower() != args.expected_sha256.lower():
        raise RuntimeError(
            f"Output SHA-256 mismatch for {args.output}: expected "
            f"{args.expected_sha256.lower()}, got {digest}"
        )
    print(
        f"Wrote {len(probes)} hard TextOCR probes from {selected_images} images "
        f"to {args.output}; excluded {len(excluded_image_ids)} image IDs; "
        f"construction={args.construction_version}; sha256={digest}"
    )


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def image_ids_from_probes(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row.get("image_id", "")) for row in rows if row.get("image_id")}


def choose_hard_token(
    row: dict[str, Any],
    *,
    min_len: int,
    max_len: int,
    min_area: float,
    max_area: float,
    pool_size: int,
    require_unique: bool,
    rng: random.Random,
) -> tuple[str, list[float], float, int] | None:
    candidates: list[tuple[str, list[float], float]] = []
    token_counts = token_counter(row)
    for item in row.get("ocr_tokens", []):
        if not isinstance(item, dict):
            continue
        text = normalize_token(item.get("text", ""))
        box = normalize_box(item.get("bbox"))
        if box is None or not is_usable_token(text, min_len=min_len, max_len=max_len):
            continue
        if require_unique and token_counts.get(text.casefold(), 0) != 1:
            continue
        area = box_area(box)
        if area < min_area or area > max_area:
            continue
        candidates.append((text, box, area))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[2])
    pool = candidates[: max(1, min(pool_size, len(candidates)))]
    text, box, area = rng.choice(pool)
    rank = 1 + candidates.index((text, box, area))
    return text, box, area, rank


def make_probe(
    row: dict[str, Any],
    *,
    sample_id: str,
    question: str,
    target_answer: str,
    target_text: str,
    evidence_box: list[float],
    hard_type: str,
    source_text: str,
    token_area: float,
    token_area_rank: int,
) -> dict[str, Any]:
    polarity = "positive" if target_answer == "yes" else "negative"
    return {
        "id": sample_id,
        "sample_id": sample_id,
        "probe": "orig",
        "probe_count": 1,
        "rice_view": "original",
        "dataset": "TextOCR-HardYesNo",
        "source_dataset": row.get("source_dataset", row.get("dataset", "TextOCR")),
        "image": row.get("image", ""),
        "image_id": row.get("image_id", ""),
        "question": question,
        "target_answer": target_answer,
        "task_family": "ocr_text",
        "relation": "ocr_exact_text_visible",
        "base_relation": "ocr_exact_text_visible",
        "bbox_source": row.get("bbox_source", "textocr_word_annotations"),
        "has_bbox": True,
        "base_has_bbox": True,
        "answer": target_answer,
        "target_text": target_text,
        "source_text": source_text,
        "binary_polarity": polarity,
        "hard_type": hard_type,
        "token_area": token_area,
        "token_area_rank": token_area_rank,
        "evidence_regions": [evidence_box],
        "ocr_regions": [evidence_box],
        "evidence_region_count": 1,
    }


def present_token_set(row: dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for item in row.get("ocr_tokens", []):
        if isinstance(item, dict):
            text = normalize_token(item.get("text", ""))
            if text:
                tokens.add(text.casefold())
    return tokens


def token_counter(row: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in row.get("ocr_tokens", []):
        if isinstance(item, dict):
            text = normalize_token(item.get("text", ""))
            if text:
                key = text.casefold()
                counts[key] = counts.get(key, 0) + 1
    return counts


def mutate_token(
    text: str,
    *,
    present: set[str],
    rng: random.Random,
    construction_version: str = CONFIRMATION_V2,
) -> str:
    candidates: list[str] = []
    for index, char in enumerate(text):
        replacements = replacement_chars(char, construction_version=construction_version)
        for repl in replacements:
            mutated = text[:index] + repl + text[index + 1 :]
            if mutated != text and mutated.casefold() not in present:
                candidates.append(mutated)
    if len(text) >= 4:
        for index, char in enumerate(text):
            if char.isalnum():
                mutated = text[:index] + text[index + 1 :]
                if len(mutated) >= 3 and mutated.casefold() not in present:
                    candidates.append(mutated)
    # Deduplicate while preserving deterministic order before random choice.
    deduped = list(dict.fromkeys(candidates))
    return rng.choice(deduped) if deduped else ""


def replacement_chars(
    char: str,
    *,
    construction_version: str = CONFIRMATION_V2,
) -> list[str]:
    if construction_version not in CONSTRUCTION_VERSIONS:
        raise ValueError(f"Unknown construction version: {construction_version}")
    confusables = {
        "0": ["O", "8", "6"],
        "1": ["I", "l", "7"],
        "2": ["Z", "3"],
        "3": ["8", "5"],
        "4": ["A", "9"],
        "5": ["S", "6"],
        "6": ["8", "5"],
        "7": ["1", "T"],
        "8": ["B", "3", "0"],
        "9": ["8", "4"],
        "a": ["o", "e", "u"],
        "b": ["h", "d", "8"],
        "c": ["e", "o"],
        "d": ["b", "cl"],
        "e": ["c", "a", "o"],
        "f": ["t", "l"],
        "g": ["q", "9"],
        "h": ["b", "n"],
        "i": ["l", "1"],
        "j": ["i"],
        "k": ["h"],
        "l": ["1", "I", "i"],
        "m": ["rn", "n"],
        "n": ["h", "m"],
        "o": ["0", "a", "e"],
        "p": ["b", "q"],
        "q": ["g", "p"],
        "r": ["n"],
        "s": ["5", "S"],
        "t": ["f", "l"],
        "u": ["v", "n"],
        "v": ["u", "y"],
        "w": ["vv", "m"],
        "x": ["k"],
        "y": ["v"],
        "z": ["2", "s"],
    }
    lower = char.lower()
    reps = confusables.get(lower, [])
    unicode_safe = construction_version == CONFIRMATION_V2
    if char.isalpha() and (char.isascii() or not unicode_safe):
        shifted = chr(ord(lower) + 1) if lower != "z" else "a"
        reps.append(shifted)
    elif char.isdigit() and (char.isascii() or not unicode_safe):
        reps.append(str((int(char) + 1) % 10))
    if char.isupper():
        return [value.upper() if len(value) == 1 else value.upper() for value in reps]
    return reps


def normalize_token(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip("\"'`.,;:!?()[]{}")


def normalize_box(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(coord) for coord in value[:4]]
    except Exception:
        return None
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        return None
    return [x1, y1, x2, y2]


def box_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def is_usable_token(text: str, *, min_len: int, max_len: int) -> bool:
    if len(text) < min_len or len(text) > max_len:
        return False
    if not any(char.isalnum() for char in text):
        return False
    punctuation = sum(1 for char in text if not char.isalnum())
    if punctuation / max(1, len(text)) > 0.35:
        return False
    return True


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
