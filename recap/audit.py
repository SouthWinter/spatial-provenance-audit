"""Dataset audit helpers for RECAP canonical JSONL files."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Any

from recap.relations import relation_family, normalize_relation


def audit_samples(samples: list[dict[str, Any]], *, examples_per_relation: int = 3) -> dict[str, Any]:
    relation_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    answer_counts: Counter[str] = Counter()
    dataset_counts: Counter[str] = Counter()
    source_dataset_counts: Counter[str] = Counter()
    task_form_counts: Counter[str] = Counter()
    reference_frame_counts: Counter[str] = Counter()
    polarity_counts: Counter[str] = Counter()
    relation_answer_counts: dict[str, Counter[str]] = defaultdict(Counter)
    family_answer_counts: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    choice_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_image_count = 0

    for sample in samples:
        relation = normalize_relation(sample.get("relation", "")) or "<empty>"
        family = relation_family(relation) or "<empty>"
        answer = _answer_key(sample)
        dataset = str(sample.get("dataset", "") or "<empty>")
        source_dataset = str(sample.get("source_dataset", "") or "<empty>")
        task_form = str(sample.get("task_form", "") or "<empty>")
        reference_frame = str(sample.get("reference_frame", "") or "<empty>")
        polarity = str(sample.get("binary_polarity", "") or "<empty>")

        relation_counts[relation] += 1
        family_counts[family] += 1
        answer_counts[answer] += 1
        dataset_counts[dataset] += 1
        source_dataset_counts[source_dataset] += 1
        task_form_counts[task_form] += 1
        reference_frame_counts[reference_frame] += 1
        polarity_counts[polarity] += 1
        relation_answer_counts[relation][answer] += 1
        family_answer_counts[family][answer] += 1
        if not str(sample.get("image", sample.get("image_path", sample.get("image_base64", "")) or "")).strip():
            missing_image_count += 1

        if len(examples[relation]) < examples_per_relation:
            examples[relation].append(_sample_example(sample))

        group_id = infer_choice_group_id(sample)
        if group_id:
            choice_groups[group_id].append(sample)

    return {
        "num_rows": len(samples),
        "datasets": _counter_dict(dataset_counts),
        "source_datasets": _counter_dict(source_dataset_counts),
        "relations": _counter_dict(relation_counts),
        "relation_families": _counter_dict(family_counts),
        "answers": _counter_dict(answer_counts),
        "task_forms": _counter_dict(task_form_counts),
        "reference_frames": _counter_dict(reference_frame_counts),
        "binary_polarity": _counter_dict(polarity_counts),
        "missing_image_field_count": missing_image_count,
        "by_relation": {
            relation: {
                "count": count,
                "family": relation_family(relation),
                "answers": _counter_dict(relation_answer_counts[relation]),
            }
            for relation, count in _sorted_counter_items(relation_counts)
        },
        "by_relation_family": {
            family: {
                "count": count,
                "answers": _counter_dict(family_answer_counts[family]),
            }
            for family, count in _sorted_counter_items(family_counts)
        },
        "choice_groups": summarize_choice_groups(choice_groups),
        "examples_by_relation": {relation: rows for relation, rows in sorted(examples.items())},
    }


def summarize_choice_groups(groups: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    if not groups:
        return {
            "num_groups": 0,
            "group_size_distribution": {},
            "positive_count_distribution": {},
            "relation_set_distribution": {},
        }

    size_counts: Counter[str] = Counter()
    positive_counts: Counter[str] = Counter()
    relation_set_counts: Counter[str] = Counter()
    complete_groups = 0
    for rows in groups.values():
        size_counts[str(len(rows))] += 1
        positives = sum(1 for row in rows if _answer_key(row) == "yes")
        positive_counts[str(positives)] += 1
        if positives == 1 and len(rows) > 1:
            complete_groups += 1
        relation_set = ",".join(sorted({normalize_relation(row.get("relation", "")) or "<empty>" for row in rows}))
        relation_set_counts[relation_set] += 1

    return {
        "num_groups": len(groups),
        "complete_one_positive_groups": complete_groups,
        "group_size_distribution": _counter_dict(size_counts),
        "positive_count_distribution": _counter_dict(positive_counts),
        "relation_set_distribution": _counter_dict(relation_set_counts),
    }


def infer_choice_group_id(sample: dict[str, Any]) -> str:
    explicit = str(sample.get("choice_group_id", "") or "").strip()
    if explicit:
        return explicit

    sample_id = str(sample.get("id", sample.get("sample_id", "")) or "").strip()
    option_index = sample.get("option_index", sample.get("choice_index"))
    if sample_id and option_index is not None:
        suffix = f"-{option_index}"
        if sample_id.endswith(suffix):
            return sample_id[: -len(suffix)]

    captions = sample.get("source_caption_options", [])
    if isinstance(captions, (list, tuple)) and captions:
        payload = {
            "dataset": sample.get("dataset", ""),
            "source_dataset": sample.get("source_dataset", ""),
            "image_id": sample.get("image_id", sample.get("image", "")),
            "captions": list(captions),
        }
        digest = hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
        return f"choice-{digest}"

    return ""


def _answer_key(sample: dict[str, Any]) -> str:
    answer = sample.get("target_answer", sample.get("answer", sample.get("label", "")))
    text = str(answer).strip().lower()
    if text in {"true", "1"}:
        return "yes"
    if text in {"false", "0"}:
        return "no"
    return text or "<empty>"


def _sample_example(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": sample.get("id", sample.get("sample_id", "")),
        "dataset": sample.get("dataset", ""),
        "source_dataset": sample.get("source_dataset", ""),
        "relation": normalize_relation(sample.get("relation", "")),
        "answer": _answer_key(sample),
        "question": sample.get("question", ""),
        "source_caption": sample.get("source_caption", ""),
        "image": sample.get("image", sample.get("image_path", "")),
    }


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: int(value) for key, value in _sorted_counter_items(counter)}


def _sorted_counter_items(counter: Counter[str]) -> list[tuple[str, int]]:
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))
