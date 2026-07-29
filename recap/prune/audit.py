"""Dataset audits for pruning experiments."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from recap.prune.metrics import evidence_regions_from_sample, image_size_from_sample
from recap.relations import normalize_relation, relation_family


def audit_prune_samples(samples: list[dict[str, Any]], *, examples_per_family: int = 3) -> dict[str, Any]:
    dataset_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    evidence_counts: Counter[str] = Counter()
    bbox_source_counts: Counter[str] = Counter()
    image_size_counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for sample in samples:
        dataset = str(sample.get("dataset", sample.get("source_dataset", "")) or "<empty>")
        family = infer_task_family(sample)
        tags = infer_risk_tags(sample)
        evidence_regions = evidence_regions_from_sample(sample)
        evidence_key = "has_evidence" if evidence_regions else "missing_evidence"
        bbox_source = str(sample.get("bbox_source", "") or "<empty>")
        image_size = image_size_from_sample(sample)
        image_size_key = f"{int(image_size[0])}x{int(image_size[1])}" if image_size else "<unknown>"

        dataset_counts[dataset] += 1
        family_counts[family] += 1
        evidence_counts[evidence_key] += 1
        bbox_source_counts[bbox_source] += 1
        image_size_counts[image_size_key] += 1
        for tag in tags:
            tag_counts[tag] += 1

        if len(examples[family]) < examples_per_family:
            examples[family].append(
                {
                    "id": sample.get("id", sample.get("sample_id", "")),
                    "dataset": dataset,
                    "task_family": family,
                    "risk_tags": tags,
                    "relation": normalize_relation(sample.get("relation", "")),
                    "question": sample.get("question", sample.get("source_question", "")),
                    "evidence_region_count": len(evidence_regions),
                    "image": sample.get("image", sample.get("image_path", "")),
                }
            )

    return {
        "num_samples": len(samples),
        "datasets": _counter_dict(dataset_counts),
        "task_families": _counter_dict(family_counts),
        "risk_tags": _counter_dict(tag_counts),
        "evidence": _counter_dict(evidence_counts),
        "bbox_sources": _counter_dict(bbox_source_counts),
        "image_sizes": _counter_dict(image_size_counts),
        "examples_by_task_family": {key: value for key, value in sorted(examples.items())},
    }


def infer_task_family(sample: dict[str, Any]) -> str:
    text = _sample_text(sample)
    dataset = str(sample.get("dataset", sample.get("source_dataset", "")) or "").lower()
    relation = normalize_relation(sample.get("relation", ""))
    rel_family = relation_family(relation)
    if any(key in dataset or key in text for key in ("ocr", "textvqa", "docvqa", "text", "read", "word")):
        return "ocr_text"
    if any(key in dataset or key in text for key in ("document", "doc", "table", "chart", "receipt", "invoice")):
        return "document"
    if any(key in text for key in ("how many", "number of", "count", "counting")):
        return "counting"
    if rel_family in {"left_right", "vertical", "topology", "depth"}:
        return "spatial"
    if any(key in dataset or key in text for key in ("refcoco", "ground", "locat", "bounding box", "where is")):
        return "grounding"
    if any(key in dataset or key in text for key in ("video", "frame", "temporal", "before", "after")):
        return "temporal"
    return "general"


def infer_risk_tags(sample: dict[str, Any]) -> list[str]:
    text = _sample_text(sample)
    tags: list[str] = []
    task_family = infer_task_family(sample)
    if task_family != "general":
        tags.append(task_family)
    keyword_tags = {
        "small_object": ("small", "tiny", "far away", "distant"),
        "multi_object": ("between", "among", "multiple", "several", "all of"),
        "spatial_relation": ("left", "right", "above", "below", "front", "behind", "inside", "on top"),
        "visual_text": ("text", "word", "letter", "read", "sign", "label"),
        "counting": ("how many", "count", "number of"),
    }
    for tag, needles in keyword_tags.items():
        if any(needle in text for needle in needles):
            tags.append(tag)
    if evidence_regions_from_sample(sample):
        tags.append("has_evidence")
    return sorted(set(tags)) or ["low_risk_default"]


def _sample_text(sample: dict[str, Any]) -> str:
    fields = (
        sample.get("dataset", ""),
        sample.get("source_dataset", ""),
        sample.get("question", ""),
        sample.get("source_question", ""),
        sample.get("source_caption", ""),
        sample.get("relation", ""),
    )
    return " ".join(str(field) for field in fields if field is not None).lower()


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: int(value) for key, value in sorted(counter.items(), key=lambda item: (-item[1], item[0]))}
