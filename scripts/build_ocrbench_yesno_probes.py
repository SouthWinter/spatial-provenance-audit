#!/usr/bin/env python
"""Build OCRBench answer-verification yes/no probes.

The generated probes are intentionally compatible with the existing RECAP
yes/no likelihood runners. They are not a replacement for OCRBench's native
open-ended generation metric; they are a cross-dataset OCR robustness check for
the pruning pipeline.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_DATASET = "echo840/OCRBench"
DEFAULT_SPLIT = "test"
QUESTION_TYPES = (
    "Regular Text Recognition",
    "Irregular Text Recognition",
    "Artistic Text Recognition",
    "Handwriting Recognition",
    "Digit String Recognition",
    "Non-Semantic Text Recognition",
    "Scene Text-centric VQA",
    "Doc-oriented VQA",
    "Key Information Extraction",
    "Handwritten Mathematical Expression Recognition",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--output", default="data/ocrbench_yesno_probes_100img.jsonl")
    parser.add_argument("--image-dir", default="data/ocrbench_subset/images")
    parser.add_argument("--meta-output", default="")
    parser.add_argument("--samples-per-type", type=int, default=10)
    parser.add_argument("--limit-images", type=int, default=None)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = load_ocrbench_rows(
        dataset_path=args.dataset_path,
        split=args.split,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        streaming=args.streaming,
    )
    grouped = group_rows(rows)
    selected = select_stratified(
        grouped,
        samples_per_type=args.samples_per_type,
        limit_images=args.limit_images,
        rng=rng,
    )
    answer_pools = answer_pools_by_type(rows)
    probes = build_probes(
        selected,
        answer_pools=answer_pools,
        image_dir=Path(args.image_dir),
        rng=rng,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, probes)
    meta_path = Path(args.meta_output) if args.meta_output else output.with_suffix(".meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta = build_meta(
        dataset_path=args.dataset_path,
        split=args.split,
        selected=selected,
        probes=probes,
        samples_per_type=args.samples_per_type,
        limit_images=args.limit_images,
        seed=args.seed,
        image_dir=Path(args.image_dir),
    )
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(probes)} OCRBench yes/no probes from {len(selected)} images to {output}")
    print(f"Wrote metadata to {meta_path}")


def load_ocrbench_rows(
    *,
    dataset_path: str,
    split: str,
    cache_dir: str | None,
    local_files_only: bool,
    streaming: bool,
) -> list[dict[str, Any]]:
    from datasets import DownloadConfig, load_dataset

    kwargs: dict[str, Any] = {
        "split": split,
        "cache_dir": cache_dir,
        "streaming": streaming,
    }
    if local_files_only:
        kwargs["download_config"] = DownloadConfig(local_files_only=True)
    dataset = load_dataset(dataset_path, **kwargs)
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(dataset):
        answer = first_answer(row.get("answer"))
        question = str(row.get("question", "")).strip()
        question_type = str(row.get("question_type", "")).strip()
        if not answer or not question or not question_type:
            continue
        rows.append(
            {
                "row_index": index,
                "dataset": str(row.get("dataset", "")),
                "question": question,
                "question_type": question_type,
                "answer": answer,
                "answers": answer_list(row.get("answer")),
                "image": row.get("image"),
            }
        )
    return rows


def first_answer(value: Any) -> str:
    answers = answer_list(value)
    return answers[0] if answers else ""


def answer_list(value: Any) -> list[str]:
    if isinstance(value, str):
        raw = [value]
    elif isinstance(value, (list, tuple)):
        raw = list(value)
    else:
        raw = []
    answers = []
    for item in raw:
        text = normalize_answer(str(item))
        if text:
            answers.append(text)
    return list(dict.fromkeys(answers))


def normalize_answer(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value).strip())
    return text.strip()


def group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["question_type"])].append(row)
    return grouped


def select_stratified(
    grouped: dict[str, list[dict[str, Any]]],
    *,
    samples_per_type: int,
    limit_images: int | None,
    rng: random.Random,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for question_type in QUESTION_TYPES:
        candidates = list(grouped.get(question_type, []))
        if not candidates:
            continue
        rng.shuffle(candidates)
        selected.extend(candidates[:samples_per_type])
    remaining_types = sorted(key for key in grouped if key not in QUESTION_TYPES)
    for question_type in remaining_types:
        candidates = list(grouped[question_type])
        rng.shuffle(candidates)
        selected.extend(candidates[:samples_per_type])
    selected.sort(key=lambda row: (str(row["question_type"]), int(row["row_index"])))
    if limit_images is not None:
        selected = selected[:limit_images]
    return selected


def answer_pools_by_type(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    pools: dict[str, list[str]] = defaultdict(list)
    global_pool: list[str] = []
    for row in rows:
        answer = str(row["answer"])
        pools[str(row["question_type"])].append(answer)
        global_pool.append(answer)
    out = {key: list(dict.fromkeys(values)) for key, values in pools.items()}
    out["__global__"] = list(dict.fromkeys(global_pool))
    return out


def build_probes(
    rows: list[dict[str, Any]],
    *,
    answer_pools: dict[str, list[str]],
    image_dir: Path,
    rng: random.Random,
) -> list[dict[str, Any]]:
    image_dir.mkdir(parents=True, exist_ok=True)
    probes: list[dict[str, Any]] = []
    for local_index, row in enumerate(rows):
        image_path = save_image(row, image_dir=image_dir)
        sample_prefix = f"ocrbench-{int(row['row_index']):04d}-{slug(row['question_type'])}"
        answer = str(row["answer"])
        decoy = choose_decoy(row, answer_pools=answer_pools, rng=rng)
        probes.append(make_probe(row, sample_id=f"{sample_prefix}:answer-pos", image_path=image_path, candidate=answer, target_answer="yes", local_index=local_index))
        probes.append(make_probe(row, sample_id=f"{sample_prefix}:answer-neg", image_path=image_path, candidate=decoy, target_answer="no", local_index=local_index))
    return probes


def save_image(row: dict[str, Any], *, image_dir: Path) -> str:
    image = row.get("image")
    if image is None:
        raise ValueError(f"OCRBench row has no image: {row}")
    filename = f"ocrbench_{int(row['row_index']):04d}_{slug(row['question_type'])}.jpg"
    path = image_dir / filename
    if not path.exists():
        image.convert("RGB").save(path, format="JPEG", quality=95)
    return str(path.resolve())


def choose_decoy(row: dict[str, Any], *, answer_pools: dict[str, list[str]], rng: random.Random) -> str:
    answer = str(row["answer"]).casefold()
    pools = [answer_pools.get(str(row["question_type"]), []), answer_pools.get("__global__", [])]
    for pool in pools:
        candidates = [item for item in pool if item.casefold() != answer and item]
        if candidates:
            return rng.choice(candidates)
    return mutate_answer(str(row["answer"]))


def mutate_answer(answer: str) -> str:
    if not answer:
        return "not shown"
    if answer[-1].isdigit():
        replacement = str((int(answer[-1]) + 1) % 10)
        return answer[:-1] + replacement
    if answer[-1].isalpha():
        replacement = "X" if answer[-1].upper() != "X" else "Z"
        return answer[:-1] + replacement
    return answer + "X"


def make_probe(
    row: dict[str, Any],
    *,
    sample_id: str,
    image_path: str,
    candidate: str,
    target_answer: str,
    local_index: int,
) -> dict[str, Any]:
    polarity = "positive" if target_answer == "yes" else "negative"
    question = (
        f'Given the image, answer the OCR question: "{row["question"]}". '
        f'Is the answer "{candidate}"? Answer yes or no.'
    )
    return {
        "id": sample_id,
        "sample_id": sample_id,
        "probe": "orig",
        "probe_count": 1,
        "rice_view": "original",
        "dataset": "OCRBench-AnswerVerify",
        "source_dataset": f"OCRBench/{row['dataset']}",
        "image": image_path,
        "image_id": f"ocrbench-{row['row_index']}",
        "question": question,
        "target_answer": target_answer,
        "answer": target_answer,
        "target_text": candidate,
        "source_text": row["answer"],
        "answer_options": row["answers"],
        "candidate_answer": candidate,
        "ocrbench_question": row["question"],
        "ocrbench_question_type": row["question_type"],
        "task_family": "ocr_text",
        "relation": "ocr_answer_verification",
        "base_relation": "ocr_answer_verification",
        "binary_polarity": polarity,
        "hard_type": "ocrbench_answer_positive" if target_answer == "yes" else "ocrbench_answer_decoy_negative",
        "bbox_source": "none",
        "has_bbox": False,
        "base_has_bbox": False,
        "evidence_regions": [],
        "ocr_regions": [],
        "evidence_region_count": 0,
        "ocrbench_row_index": int(row["row_index"]),
        "ocrbench_local_index": int(local_index),
    }


def build_meta(
    *,
    dataset_path: str,
    split: str,
    selected: list[dict[str, Any]],
    probes: list[dict[str, Any]],
    samples_per_type: int,
    limit_images: int | None,
    seed: int,
    image_dir: Path,
) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    for row in selected:
        counts[str(row["question_type"])] += 1
    return {
        "dataset_path": dataset_path,
        "split": split,
        "task": "OCRBench answer-verification yes/no subset",
        "native_metric_note": "This is not OCRBench's native open-ended generation score; it is a pruning-compatible yes/no robustness probe set.",
        "num_images": len(selected),
        "num_probes": len(probes),
        "samples_per_type": samples_per_type,
        "limit_images": limit_images,
        "seed": seed,
        "image_dir": str(image_dir.resolve()),
        "question_type_counts": dict(sorted(counts.items())),
    }


def slug(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", str(value).strip().lower()).strip("-")
    return text or "unknown"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
