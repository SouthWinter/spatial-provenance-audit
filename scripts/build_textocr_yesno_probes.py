#!/usr/bin/env python
"""Build yes/no TextOCR probes with OCR region evidence."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--min-token-len", type=int, default=3)
    parser.add_argument("--max-regions-negative", type=int, default=80)
    parser.add_argument("--include-negative", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = read_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]

    vocab_by_image: list[set[str]] = []
    vocab: list[str] = []
    for row in rows:
        tokens = {normalize_token(item.get("text", "")) for item in row.get("ocr_tokens", []) if isinstance(item, dict)}
        tokens = {token for token in tokens if len(token) >= args.min_token_len}
        vocab_by_image.append(tokens)
        vocab.extend(sorted(tokens))
    vocab = sorted(set(vocab))

    probes: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        positive = choose_positive(row, min_len=args.min_token_len, rng=rng)
        if positive is None:
            continue
        token, box = positive
        base = base_probe(row, sample_id=f"{row.get('id', idx)}:ocr-pos", target_answer="yes")
        base["question"] = f'Is the text "{token}" visible in the image? Answer yes or no.'
        base["answer"] = "yes"
        base["target_text"] = token
        base["binary_polarity"] = "positive"
        base["evidence_regions"] = [box]
        base["ocr_regions"] = [box]
        base["evidence_region_count"] = 1
        probes.append(base)

        if args.include_negative:
            decoy = choose_decoy(vocab, vocab_by_image[idx], rng=rng)
            if decoy:
                regions = row.get("evidence_regions") or row.get("ocr_regions") or []
                regions = [region for region in regions if valid_box(region)]
                if args.max_regions_negative > 0:
                    regions = regions[: args.max_regions_negative]
                neg = base_probe(row, sample_id=f"{row.get('id', idx)}:ocr-neg", target_answer="no")
                neg["question"] = f'Is the text "{decoy}" visible in the image? Answer yes or no.'
                neg["answer"] = "no"
                neg["target_text"] = decoy
                neg["binary_polarity"] = "negative"
                neg["evidence_regions"] = regions
                neg["ocr_regions"] = regions
                neg["evidence_region_count"] = len(regions)
                probes.append(neg)

    write_jsonl(args.output, probes)
    print(f"Wrote {len(probes)} TextOCR yes/no probes to {args.output}")


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


def base_probe(row: dict[str, Any], *, sample_id: str, target_answer: str) -> dict[str, Any]:
    return {
        "id": sample_id,
        "sample_id": sample_id,
        "probe": "orig",
        "probe_count": 1,
        "rice_view": "original",
        "dataset": "TextOCR-YesNo",
        "source_dataset": row.get("source_dataset", row.get("dataset", "TextOCR")),
        "image": row.get("image", ""),
        "image_id": row.get("image_id", ""),
        "question": "",
        "target_answer": target_answer,
        "task_family": "ocr_text",
        "relation": "ocr_text_visible",
        "base_relation": "ocr_text_visible",
        "bbox_source": row.get("bbox_source", "textocr_word_annotations"),
        "has_bbox": True,
        "base_has_bbox": True,
    }


def choose_positive(row: dict[str, Any], *, min_len: int, rng: random.Random) -> tuple[str, list[float]] | None:
    candidates: list[tuple[str, list[float], float]] = []
    for item in row.get("ocr_tokens", []):
        if not isinstance(item, dict):
            continue
        token = normalize_token(item.get("text", ""))
        box = item.get("bbox")
        if len(token) < min_len or not valid_box(box):
            continue
        x1, y1, x2, y2 = [float(value) for value in box[:4]]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        candidates.append((token, [x1, y1, x2, y2], area))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[2], reverse=True)
    pool = candidates[: min(20, len(candidates))]
    token, box, _ = rng.choice(pool)
    return token, box


def choose_decoy(vocab: list[str], present: set[str], *, rng: random.Random) -> str:
    candidates = [token for token in vocab if token not in present]
    return rng.choice(candidates) if candidates else ""


def normalize_token(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def valid_box(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return False
    try:
        x1, y1, x2, y2 = [float(coord) for coord in value[:4]]
    except Exception:
        return False
    return x2 > x1 and y2 > y1


if __name__ == "__main__":
    main()
