#!/usr/bin/env python
"""Run native open-answer OCR/document QA generation for Qwen full vs pruned prefixes.

The script intentionally mirrors ``run_qwen_ocrbench_open_answer_generation.py``
but loads original TextVQA/DocVQA-style questions instead of converting them to
yes/no probes. This is a direct stress test for the "constructed binary task"
criticism in problem.md.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import string
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recap.io import jsonable, read_jsonl, write_json, write_jsonl
from scripts.run_qwen_ocrbench_open_answer_generation import (
    generate_full,
    generate_pruned,
    mean,
)
from recap.qwen_direct_backend import _load_qwen_direct
from recap.qwen_pruned_backend import PruneConfig


DATASET_SPECS: dict[str, dict[str, str]] = {
    "textvqa_val_lite": {
        "dataset_path": "lmms-lab/LMMs-Eval-Lite",
        "dataset_name": "textvqa_val",
        "split": "lite",
        "metric": "textvqa_accuracy",
    },
    "docvqa_val_lite": {
        "dataset_path": "lmms-lab/LMMs-Eval-Lite",
        "dataset_name": "docvqa_val",
        "split": "lite",
        "metric": "anls",
    },
    "textvqa_val": {
        "dataset_path": "lmms-lab/textvqa",
        "dataset_name": "",
        "split": "validation",
        "metric": "textvqa_accuracy",
        "data_glob": "data/validation-*.parquet",
    },
    "docvqa_val": {
        "dataset_path": "lmms-lab/DocVQA",
        "dataset_name": "DocVQA",
        "split": "validation",
        "metric": "anls",
        "data_glob": "DocVQA/validation-*.parquet",
    },
}


FOCUS_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "answer",
        "are",
        "associated",
        "at",
        "be",
        "been",
        "being",
        "called",
        "can",
        "could",
        "did",
        "displayed",
        "do",
        "document",
        "does",
        "for",
        "from",
        "given",
        "had",
        "has",
        "have",
        "he",
        "her",
        "hers",
        "him",
        "his",
        "how",
        "image",
        "in",
        "is",
        "it",
        "may",
        "me",
        "mentioned",
        "might",
        "mine",
        "my",
        "of",
        "on",
        "or",
        "our",
        "ours",
        "photo",
        "picture",
        "please",
        "read",
        "reads",
        "say",
        "says",
        "she",
        "should",
        "shown",
        "single",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "these",
        "they",
        "this",
        "those",
        "to",
        "using",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "word",
        "words",
        "would",
        "written",
        "you",
        "your",
        "yours",
    }
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="textvqa_val_lite", choices=sorted(DATASET_SPECS))
    parser.add_argument("--work-dir", default="runs/open_ocr_qa/qwen3_8b_textvqa_lite_target_grid0p70")
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--selector", default="target_embed_grid_topk")
    parser.add_argument(
        "--selector-target-source",
        default="question",
        choices=("question", "focus"),
        help="Use the full question or answer-free content terms as selector-side text.",
    )
    parser.add_argument("--keep-ratio", type=float, default=0.70)
    parser.add_argument("--budget-mode", default="fixed", choices=("fixed", "evidence_saturation"))
    parser.add_argument("--rho-min", type=float, default=0.30)
    parser.add_argument("--rho-max", type=float, default=0.70)
    parser.add_argument("--saturation-temperature", type=float, default=0.12)
    parser.add_argument("--saturation-mass-target", type=float, default=0.72)
    parser.add_argument("--saturation-cell-target", type=float, default=0.75)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--attn-implementation", default="")
    parser.add_argument("--min-pixels", type=int, default=50176)
    parser.add_argument("--max-pixels", type=int, default=802816)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--sample-start",
        type=int,
        default=0,
        help="Zero-based inclusive sample offset, used only to shard long evaluation runs.",
    )
    parser.add_argument(
        "--sample-end",
        type=int,
        default=None,
        help="Zero-based exclusive sample offset, used only to shard long evaluation runs.",
    )
    parser.add_argument(
        "--reuse-full-rows",
        default="",
        help="Optional JSONL from a matched run whose full-prefix answers are reused by sample_id.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from per-sample partial JSONL checkpoints in work-dir.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Flush partial checkpoints after this many newly completed samples.",
    )
    parser.add_argument("--use-fast-processor", action="store_true")
    args = parser.parse_args()

    if args.checkpoint_every < 1:
        raise ValueError("checkpoint-every must be at least 1")

    samples = load_samples(args.task, limit=args.limit)
    if args.sample_start < 0:
        raise ValueError("sample-start must be non-negative")
    sample_end = len(samples) if args.sample_end is None else args.sample_end
    if sample_end < args.sample_start or sample_end > len(samples):
        raise ValueError(
            f"sample range [{args.sample_start}, {sample_end}) is invalid for {len(samples)} samples"
        )
    samples = samples[args.sample_start:sample_end]
    if not samples:
        raise ValueError("requested sample range is empty")
    reused_full_rows = load_reused_full_rows(
        args.reuse_full_rows,
        samples=samples,
        task=args.task,
        max_new_tokens=args.max_new_tokens,
    )
    out_dir = Path(args.work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, traces, complete = load_generation_checkpoint(
        out_dir,
        samples=samples,
        resume=args.resume,
    )
    if complete:
        metrics = summarize(rows, args=args, task_metric=DATASET_SPECS[args.task]["metric"])
        write_json(out_dir / "metrics.json", metrics)
        write_markdown(out_dir / "open_ocr_qa_generation_report.md", metrics, rows)
        print(f"Checkpoint already complete: {len(rows)} rows in {out_dir}")
        return

    import torch

    model, processor, tokenizer, process_vision_info, input_device = _load_qwen_direct(
        pretrained=args.pretrained,
        device=args.device,
        device_map=args.device_map,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation or None,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        use_fast_processor=args.use_fast_processor,
        torch_module=torch,
    )
    prune_config = PruneConfig(
        selector=args.selector,
        keep_ratio=args.keep_ratio,
        budget_mode=args.budget_mode,
        rho_min=args.rho_min,
        rho_max=args.rho_max,
        saturation_temperature=args.saturation_temperature,
        saturation_mass_target=args.saturation_mass_target,
        saturation_cell_target=args.saturation_cell_target,
    )

    completed_ids = {str(row["sample_id"]) for row in rows}
    pending_samples = [sample for sample in samples if str(sample["sample_id"]) not in completed_ids]
    row_partial, trace_partial = checkpoint_paths(out_dir)
    with row_partial.open("a", encoding="utf-8") as row_handle, trace_partial.open(
        "a", encoding="utf-8"
    ) as trace_handle:
        for offset, sample in enumerate(
            tqdm(
                pending_samples,
                desc=f"{args.task} open generation",
                initial=len(rows),
                total=len(samples),
            ),
            start=1,
        ):
            probe = selector_probe(sample, target_source=args.selector_target_source)
            reused_row = reused_full_rows.get(sample["sample_id"])
            if reused_row is None:
                full_answer = generate_full(
                    probe,
                    model=model,
                    processor=processor,
                    tokenizer=tokenizer,
                    process_vision_info=process_vision_info,
                    input_device=input_device,
                    min_pixels=args.min_pixels,
                    max_pixels=args.max_pixels,
                    max_new_tokens=args.max_new_tokens,
                )
            else:
                full_answer = str(reused_row["full_answer"])
            pruned_answer, trace = generate_pruned(
                probe,
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                process_vision_info=process_vision_info,
                input_device=input_device,
                min_pixels=args.min_pixels,
                max_pixels=args.max_pixels,
                max_new_tokens=args.max_new_tokens,
                prune_config=prune_config,
            )
            full_metrics = score_task_answer(full_answer, sample["gold_answers"], sample["metric"])
            pruned_metrics = score_task_answer(pruned_answer, sample["gold_answers"], sample["metric"])
            row = {
                **drop_image(sample),
                "full_answer": full_answer,
                "pruned_answer": pruned_answer,
                "full_score": full_metrics["primary"],
                "pruned_score": pruned_metrics["primary"],
                "score_delta_pruned_minus_full": pruned_metrics["primary"] - full_metrics["primary"],
                "full_exact": full_metrics["exact"],
                "pruned_exact": pruned_metrics["exact"],
                "full_anls": full_metrics["anls"],
                "pruned_anls": pruned_metrics["anls"],
                "full_textvqa_accuracy": full_metrics["textvqa_accuracy"],
                "pruned_textvqa_accuracy": pruned_metrics["textvqa_accuracy"],
                "selector": args.selector,
                "budget_mode": args.budget_mode,
                "keep_ratio": args.keep_ratio,
                "effective_keep_ratio": float(trace.get("effective_keep_ratio", 0.0) or 0.0),
                "target_text_token_count": int(trace.get("target_text_token_count", 0) or 0),
                "selector_target_source": args.selector_target_source,
                "selector_focus_terms": list(probe.get("selector_target_texts", [])),
            }
            rows.append(row)
            traces.append(trace)
            append_checkpoint(row_handle, row)
            append_checkpoint(trace_handle, trace)
            if offset % args.checkpoint_every == 0:
                row_handle.flush()
                trace_handle.flush()
        row_handle.flush()
        trace_handle.flush()

    rows, traces = order_completed_rows(rows, traces, samples=samples)
    metrics = summarize(rows, args=args, task_metric=DATASET_SPECS[args.task]["metric"])
    write_jsonl(out_dir / "open_ocr_qa_generation.jsonl", rows)
    write_jsonl(out_dir / "prune_traces.jsonl", traces)
    write_json(out_dir / "metrics.json", metrics)
    write_markdown(out_dir / "open_ocr_qa_generation_report.md", metrics, rows)
    row_partial.unlink(missing_ok=True)
    trace_partial.unlink(missing_ok=True)
    print(
        f"Wrote {len(rows)} {args.task} open-answer rows to {out_dir}; "
        f"full_score={metrics['full_score']:.4f}, pruned_score={metrics['pruned_score']:.4f}, "
        f"delta={metrics['score_delta_pruned_minus_full']:+.4f}"
    )


def checkpoint_paths(out_dir: Path) -> tuple[Path, Path]:
    return (
        out_dir / "open_ocr_qa_generation.partial.jsonl",
        out_dir / "prune_traces.partial.jsonl",
    )


def load_generation_checkpoint(
    out_dir: Path,
    *,
    samples: list[dict[str, Any]],
    resume: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    final_rows = out_dir / "open_ocr_qa_generation.jsonl"
    final_traces = out_dir / "prune_traces.jsonl"
    row_partial, trace_partial = checkpoint_paths(out_dir)
    if not resume:
        if row_partial.exists() or trace_partial.exists():
            raise FileExistsError(f"Partial checkpoint exists in {out_dir}; pass --resume")
        return [], [], False

    if final_rows.exists() and final_traces.exists():
        rows = read_jsonl(final_rows)
        traces = read_jsonl(final_traces)
        rows, traces = order_completed_rows(rows, traces, samples=samples)
        return rows, traces, len(rows) == len(samples)

    if row_partial.exists() != trace_partial.exists():
        raise ValueError(f"Incomplete checkpoint pair in {out_dir}")
    if not row_partial.exists():
        return [], [], False
    rows = read_jsonl(row_partial)
    traces = read_jsonl(trace_partial)
    rows, traces = order_completed_rows(rows, traces, samples=samples, allow_partial=True)
    # A process can stop between the two append calls. Canonicalize the common
    # completed prefix before reopening both files in append mode.
    write_jsonl(row_partial, rows)
    write_jsonl(trace_partial, traces)
    return rows, traces, False


def order_completed_rows(
    rows: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    *,
    samples: list[dict[str, Any]],
    allow_partial: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    row_by_id = {str(row.get("sample_id", "")): row for row in rows}
    trace_by_id = {str(trace.get("sample_id", "")): trace for trace in traces}
    if len(row_by_id) != len(rows) or len(trace_by_id) != len(traces):
        raise ValueError("Checkpoint contains empty or duplicate sample IDs")
    if not allow_partial and len(rows) != len(traces):
        raise ValueError(f"Checkpoint row/trace count mismatch: {len(rows)} != {len(traces)}")
    if not allow_partial and set(row_by_id) != set(trace_by_id):
        raise ValueError("Checkpoint row and trace sample IDs differ")
    sample_ids = [str(sample["sample_id"]) for sample in samples]
    unknown = (set(row_by_id) | set(trace_by_id)).difference(sample_ids)
    if unknown:
        raise ValueError(f"Checkpoint contains {len(unknown)} samples outside the requested dataset")
    if not allow_partial and len(rows) != len(samples):
        raise ValueError(f"Completed output has {len(rows)} rows; expected {len(samples)}")
    common_ids = set(row_by_id) & set(trace_by_id)
    ordered_ids = [sample_id for sample_id in sample_ids if sample_id in common_ids]
    if allow_partial and ordered_ids != sample_ids[: len(ordered_ids)]:
        raise ValueError("Partial checkpoint common rows are not a contiguous dataset prefix")
    extra_ids = (set(row_by_id) | set(trace_by_id)).difference(common_ids)
    expected_extras = set(sample_ids[len(ordered_ids) : len(ordered_ids) + len(extra_ids)])
    if allow_partial and extra_ids != expected_extras:
        raise ValueError("Partial checkpoint mismatch is not confined to the trailing dataset prefix")
    return [row_by_id[sample_id] for sample_id in ordered_ids], [trace_by_id[sample_id] for sample_id in ordered_ids]


def append_checkpoint(handle, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(jsonable(payload), ensure_ascii=False) + "\n")


def load_reused_full_rows(
    path: str,
    *,
    samples: list[dict[str, Any]],
    task: str,
    max_new_tokens: int,
) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Reused full-row file does not exist: {source}")
    metrics_path = source.parent / "metrics.json"
    if metrics_path.is_file():
        source_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if source_metrics.get("task") != task:
            raise ValueError(f"Reused rows are for task {source_metrics.get('task')!r}, not {task!r}")
        if int(source_metrics.get("max_new_tokens", -1)) != max_new_tokens:
            raise ValueError(
                "Reused rows use max_new_tokens="
                f"{source_metrics.get('max_new_tokens')}, not {max_new_tokens}"
            )
    rows: dict[str, dict[str, Any]] = {}
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id", ""))
            if not sample_id or "full_answer" not in row:
                raise ValueError(f"Invalid reused row at {source}:{line_number}")
            if sample_id in rows:
                raise ValueError(f"Duplicate sample_id in reused rows: {sample_id}")
            rows[sample_id] = row
    missing = [str(sample["sample_id"]) for sample in samples if str(sample["sample_id"]) not in rows]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Reused full-row file is missing {len(missing)} requested samples: {preview}")
    return rows


def load_samples(task: str, *, limit: int | None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    spec = DATASET_SPECS[task]
    data_glob = spec.get("data_glob", "")
    if data_glob:
        from huggingface_hub import snapshot_download

        snapshot = Path(
            snapshot_download(
                spec["dataset_path"],
                repo_type="dataset",
                allow_patterns=["README.md", data_glob],
            )
        )
        files = sorted(snapshot.glob(data_glob))
        if not files:
            raise FileNotFoundError(f"No validation parquet files matching {snapshot / data_glob}")
        dataset = load_dataset(
            "parquet",
            data_files={spec["split"]: [str(path) for path in files]},
            split=spec["split"],
        )
        from datasets import Image as DatasetImage

        dataset = dataset.cast_column("image", DatasetImage(decode=False))
    else:
        kwargs: dict[str, Any] = {}
        if spec["dataset_name"]:
            kwargs["name"] = spec["dataset_name"]
        dataset = load_dataset(spec["dataset_path"], **kwargs, split=spec["split"])
    rows: list[dict[str, Any]] = []
    for idx, doc in enumerate(dataset):
        sample = sample_from_doc(task, idx, doc, metric=spec["metric"])
        if sample is not None:
            rows.append(sample)
        if limit is not None and len(rows) >= limit:
            break
    return rows


def sample_from_doc(task: str, idx: int, doc: dict[str, Any], *, metric: str) -> dict[str, Any] | None:
    image = doc.get("image")
    if image is None:
        return None
    question = str(doc.get("question", "")).strip()
    answers = collect_answers(doc)
    if not question or not answers:
        return None
    question_id = doc.get("question_id", doc.get("questionId", doc.get("id", idx)))
    return {
        "sample_id": f"{task}:{question_id}",
        "dataset": task,
        "question_id": str(question_id),
        "image": image,
        "question": format_question(question, task=task),
        "raw_question": question,
        "gold_answers": answers,
        "metric": metric,
    }


def collect_answers(doc: dict[str, Any]) -> list[str]:
    raw = doc.get("answers", doc.get("answer", doc.get("label", [])))
    if isinstance(raw, str):
        raw = [raw]
    if raw is None:
        raw = []
    answers: list[str] = []
    if isinstance(raw, (list, tuple)):
        for item in raw:
            text = str(item).strip()
            if text:
                answers.append(text)
    return answers


def format_question(question: str, *, task: str) -> str:
    if task.startswith("textvqa"):
        return f"{question}\nAnswer the question using a single word or phrase."
    return f"{question}\nAnswer the question using a single word or phrase."


def selector_probe(sample: dict[str, Any], *, target_source: str = "question") -> dict[str, Any]:
    if target_source == "focus":
        selector_target_texts = extract_focus_terms(str(sample.get("raw_question", sample["question"])))
        target_text = selector_target_texts[0]
    elif target_source == "question":
        selector_target_texts = [str(sample["question"])]
        target_text = str(sample["question"])
    else:
        raise ValueError(f"Unknown selector target source: {target_source!r}")
    return {
        "sample_id": sample["sample_id"],
        "id": sample["sample_id"],
        "dataset": sample["dataset"],
        "image": materialize_image(sample["image"]),
        "question": sample["question"],
        "target_text": target_text,
        "source_text": sample["question"],
        "selector_target_texts": selector_target_texts,
        "selector_target_source": target_source,
        "task_family": "open_ocr_qa",
        "evidence_regions": [],
        "ocr_regions": [],
    }


def materialize_image(image: Any):
    if image is None:
        return None
    if hasattr(image, "convert"):
        return image.convert("RGB")
    if isinstance(image, dict):
        from PIL import Image

        raw_bytes = image.get("bytes")
        path = image.get("path")
        if raw_bytes is not None:
            with Image.open(io.BytesIO(raw_bytes)) as loaded:
                return loaded.convert("RGB")
        if path:
            with Image.open(path) as loaded:
                return loaded.convert("RGB")
    raise TypeError(f"Unsupported image payload: {type(image).__name__}")


def extract_focus_terms(question: str, *, max_terms: int = 8) -> list[str]:
    """Extract answer-free field/entity cues while preserving their prompt spelling."""
    words = re.findall(r"[^\W_]+(?:[-'][^\W_]+)*", str(question), flags=re.UNICODE)
    terms: list[str] = []
    seen: set[str] = set()
    for word in words:
        key = word.casefold()
        if key in FOCUS_STOPWORDS or key in seen:
            continue
        if len(key) == 1 and not key.isdigit():
            continue
        terms.append(word)
        seen.add(key)
        if len(terms) >= max_terms:
            break
    return terms or [str(question).strip()]


def drop_image(sample: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in sample.items() if key != "image"}


def score_task_answer(prediction: str, gold_answers: list[str], metric: str) -> dict[str, float]:
    pred_norm = normalize_answer(prediction)
    gold_norms = [normalize_answer(answer) for answer in gold_answers if normalize_answer(answer)]
    exact = float(any(pred_norm == gold for gold in gold_norms))
    anls = max((anls_score(pred_norm, gold) for gold in gold_norms), default=0.0)
    textvqa_acc = textvqa_accuracy(prediction, gold_answers)
    if metric == "textvqa_accuracy":
        primary = textvqa_acc
    elif metric == "anls":
        primary = anls
    else:
        primary = exact
    return {
        "primary": primary,
        "exact": exact,
        "anls": anls,
        "textvqa_accuracy": textvqa_acc,
    }


def textvqa_accuracy(prediction: str, gold_answers: list[str]) -> float:
    pred = evalai_normalize(prediction)
    answers = [evalai_normalize(answer) for answer in gold_answers]
    if not answers:
        return 0.0
    scores: list[float] = []
    for idx, _answer in enumerate(answers):
        other_answers = [answers[j] for j in range(len(answers)) if j != idx]
        scores.append(min(1.0, other_answers.count(pred) / 3.0))
    return mean(scores)


def evalai_normalize(text: str) -> str:
    # Lightweight EvalAI-style normalization for standalone runs.
    text = str(text).lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_answer(text: str) -> str:
    text = str(text).lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def anls_score(prediction: str, gold: str) -> float:
    if not prediction and not gold:
        return 1.0
    if not prediction or not gold:
        return 0.0
    distance = levenshtein(prediction, gold)
    normalized = distance / max(len(prediction), len(gold), 1)
    return 1.0 - normalized if normalized < 0.5 else 0.0


def levenshtein(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, char_left in enumerate(left, start=1):
        current = [i]
        for j, char_right in enumerate(right, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (char_left != char_right)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def summarize(rows: list[dict[str, Any]], *, args: argparse.Namespace, task_metric: str) -> dict[str, Any]:
    return {
        "task": args.task,
        "note": "Greedy free-form generation on original open-answer OCR/document QA questions; selector text is derived from the question only.",
        "n": len(rows),
        "pretrained": args.pretrained,
        "selector": args.selector,
        "selector_target_source": args.selector_target_source,
        "keep_ratio": args.keep_ratio,
        "max_new_tokens": args.max_new_tokens,
        "reused_full_rows": args.reuse_full_rows or None,
        "primary_metric": task_metric,
        "full_score": mean(float(row["full_score"]) for row in rows),
        "pruned_score": mean(float(row["pruned_score"]) for row in rows),
        "score_delta_pruned_minus_full": mean(float(row["score_delta_pruned_minus_full"]) for row in rows),
        "full_exact": mean(float(row["full_exact"]) for row in rows),
        "pruned_exact": mean(float(row["pruned_exact"]) for row in rows),
        "full_anls": mean(float(row["full_anls"]) for row in rows),
        "pruned_anls": mean(float(row["pruned_anls"]) for row in rows),
        "full_textvqa_accuracy": mean(float(row["full_textvqa_accuracy"]) for row in rows),
        "pruned_textvqa_accuracy": mean(float(row["pruned_textvqa_accuracy"]) for row in rows),
        "mean_effective_keep_ratio": mean(float(row["effective_keep_ratio"]) for row in rows),
        "mean_target_text_token_count": mean(float(row["target_text_token_count"]) for row in rows),
    }


def write_markdown(path: Path, metrics: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        f"# {metrics['task']} Native Open-Answer Generation",
        "",
        metrics["note"],
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| n | {metrics['n']} |",
        f"| primary metric | {metrics['primary_metric']} |",
        f"| selector target source | {metrics['selector_target_source']} |",
        f"| full score | {metrics['full_score']:.3f} |",
        f"| pruned score | {metrics['pruned_score']:.3f} |",
        f"| delta score | {metrics['score_delta_pruned_minus_full']:+.3f} |",
        f"| full exact | {metrics['full_exact']:.3f} |",
        f"| pruned exact | {metrics['pruned_exact']:.3f} |",
        f"| full ANLS | {metrics['full_anls']:.3f} |",
        f"| pruned ANLS | {metrics['pruned_anls']:.3f} |",
        f"| full TextVQA acc. | {metrics['full_textvqa_accuracy']:.3f} |",
        f"| pruned TextVQA acc. | {metrics['pruned_textvqa_accuracy']:.3f} |",
        f"| effective keep | {metrics['mean_effective_keep_ratio']:.3f} |",
        f"| selector question-token count | {metrics['mean_target_text_token_count']:.3f} |",
        "",
        "## Score Flips",
        "",
        "| sample_id | gold | full | pruned | full score | pruned score |",
        "|---|---|---|---|---:|---:|",
    ]
    flips = [row for row in rows if abs(float(row["score_delta_pruned_minus_full"])) > 1e-9]
    for row in flips[:40]:
        lines.append(
            f"| {row['sample_id']} | {'; '.join(row['gold_answers'])} | "
            f"{row['full_answer']} | {row['pruned_answer']} | "
            f"{float(row['full_score']):.3f} | {float(row['pruned_score']):.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
