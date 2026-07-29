#!/usr/bin/env python3
"""Run native TextVQA/DocVQA generation with full and pruned LLaVA prefixes."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recap.io import write_json, write_jsonl
from recap.llava_direct_backend import (
    _format_llava_prompt,
    _load_llava_direct,
    _model_dtype,
    _move_inputs,
)
from recap.llava_pruned_backend import _llava_token_grid_shape
from recap.prune.budgets import fixed_keep_count, removal_fraction
from recap.prune.metrics import make_token_grid
from recap.prune.selectors import select_indices
from recap.qwen_pruned_backend import (
    _embedding_relevance_and_uniqueness,
    _selector_impl,
    _target_text_positions,
)
from scripts.run_qwen_open_ocr_qa_generation import (
    DATASET_SPECS,
    append_checkpoint,
    checkpoint_paths,
    drop_image,
    load_generation_checkpoint,
    load_samples,
    mean,
    order_completed_rows,
    score_task_answer,
    selector_probe,
)


SUPPORTED_SELECTORS = {"grid", "random", "target_embed_topk", "target_embed_grid_topk"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="textvqa_val_lite", choices=sorted(DATASET_SPECS))
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--selector", default="target_embed_topk", choices=sorted(SUPPORTED_SELECTORS))
    parser.add_argument("--keep-ratio", type=float, default=0.40)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="float16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--revision", default="main")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--reuse-full-rows", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    args = parser.parse_args()

    if not 0.0 < args.keep_ratio <= 1.0:
        raise ValueError("keep_ratio must be in (0, 1]")
    if args.checkpoint_every < 1:
        raise ValueError("checkpoint-every must be at least 1")
    samples = load_samples(args.task, limit=args.limit)
    reused = load_reused_full_rows(
        args.reuse_full_rows,
        samples=samples,
        task=args.task,
        max_new_tokens=args.max_new_tokens,
        pretrained=args.pretrained,
    )
    out_dir = Path(args.work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, traces, complete = load_generation_checkpoint(
        out_dir,
        samples=samples,
        resume=args.resume,
    )
    if complete:
        metrics = summarize(rows, traces, args=args, task_metric=DATASET_SPECS[args.task]["metric"])
        write_json(out_dir / "metrics.json", metrics)
        write_markdown(out_dir / "open_ocr_qa_generation_report.md", metrics, rows)
        print(f"Checkpoint already complete: {len(rows)} rows in {out_dir}")
        return

    import torch

    model, processor, input_device = _load_llava_direct(
        pretrained=args.pretrained,
        revision=args.revision,
        device=args.device,
        device_map=args.device_map,
        dtype=args.dtype,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation or None,
        torch_module=torch,
    )
    tokenizer = processor.tokenizer

    completed_ids = {str(row["sample_id"]) for row in rows}
    pending_samples = [sample for sample in samples if str(sample["sample_id"]) not in completed_ids]
    row_partial, trace_partial = checkpoint_paths(out_dir)
    with row_partial.open("a", encoding="utf-8") as row_handle, trace_partial.open(
        "a", encoding="utf-8"
    ) as trace_handle:
        for offset, sample in enumerate(
            tqdm(
                pending_samples,
                desc=f"LLaVA {args.task} open generation",
                initial=len(rows),
                total=len(samples),
            ),
            start=1,
        ):
            probe = selector_probe(sample, target_source="question")
            raw_question = str(sample.get("raw_question", sample["question"])).strip()
            probe["target_text"] = raw_question
            probe["selector_target_texts"] = [raw_question]
            reused_row = reused.get(sample["sample_id"])
            if reused_row is None:
                full_answer = generate_full(
                    probe,
                    model=model,
                    processor=processor,
                    tokenizer=tokenizer,
                    input_device=input_device,
                    max_new_tokens=args.max_new_tokens,
                    torch_module=torch,
                )
            else:
                full_answer = str(reused_row["full_answer"])
            pruned_answer, trace = generate_pruned(
                probe,
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                input_device=input_device,
                selector=args.selector,
                keep_ratio=args.keep_ratio,
                seed=args.seed,
                max_new_tokens=args.max_new_tokens,
                torch_module=torch,
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
                "keep_ratio": args.keep_ratio,
                "effective_keep_ratio": trace["effective_keep_ratio"],
                "target_text_token_count": trace["target_text_token_count"],
                "selector_target_source": "question",
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
    metrics = summarize(rows, traces, args=args, task_metric=DATASET_SPECS[args.task]["metric"])
    write_jsonl(out_dir / "open_ocr_qa_generation.jsonl", rows)
    write_jsonl(out_dir / "prune_traces.jsonl", traces)
    write_json(out_dir / "metrics.json", metrics)
    write_markdown(out_dir / "open_ocr_qa_generation_report.md", metrics, rows)
    row_partial.unlink(missing_ok=True)
    trace_partial.unlink(missing_ok=True)
    print(
        f"Wrote {len(rows)} LLaVA {args.task} rows to {out_dir}; "
        f"full={metrics['full_score']:.4f}, pruned={metrics['pruned_score']:.4f}, "
        f"delta={metrics['score_delta_pruned_minus_full']:+.4f}"
    )


def generate_full(
    probe: dict[str, Any],
    *,
    model,
    processor,
    tokenizer,
    input_device,
    max_new_tokens: int,
    torch_module,
) -> str:
    prompt = generation_prompt(processor, probe)
    inputs = processor(text=[prompt], images=[probe["image"]], return_tensors="pt")
    inputs = _move_inputs(inputs, input_device, _model_dtype(model), torch_module)
    prompt_len = int(inputs["input_ids"].shape[1])
    with torch_module.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0, prompt_len:]
    return clean_generation(tokenizer.decode(generated, skip_special_tokens=True))


def generate_pruned(
    probe: dict[str, Any],
    *,
    model,
    processor,
    tokenizer,
    input_device,
    selector: str,
    keep_ratio: float,
    seed: int,
    max_new_tokens: int,
    torch_module,
) -> tuple[str, dict[str, Any]]:
    prompt = generation_prompt(processor, probe)
    inputs = processor(text=[prompt], images=[probe["image"]], return_tensors="pt")
    inputs = _move_inputs(inputs, input_device, _model_dtype(model), torch_module)
    prepared, trace = prepare_pruned_prefix(
        probe,
        model=model,
        processor=processor,
        inputs=inputs,
        selector=selector,
        keep_ratio=keep_ratio,
        seed=seed,
        torch_module=torch_module,
    )
    start = time.perf_counter()
    with torch_module.inference_mode():
        output_ids = model.generate(
            inputs_embeds=prepared["inputs_embeds"],
            attention_mask=prepared["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    if output_ids.ndim != 2 or int(output_ids.shape[0]) != 1:
        raise ValueError(f"Unexpected generated token shape: {tuple(output_ids.shape)}")
    trace["generation_ms"] = (time.perf_counter() - start) * 1000.0
    return clean_generation(tokenizer.decode(output_ids[0], skip_special_tokens=True)), trace


def prepare_pruned_prefix(
    probe: dict[str, Any],
    *,
    model,
    processor,
    inputs,
    selector: str,
    keep_ratio: float,
    seed: int,
    torch_module,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if selector not in SUPPORTED_SELECTORS:
        raise ValueError(f"Unsupported open-generation selector: {selector}")
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    pixel_values = inputs["pixel_values"]
    image_sizes = inputs.get("image_sizes")
    if tuple(input_ids.shape[:1]) != (1,) or int(pixel_values.shape[0]) != 1:
        raise ValueError("LLaVA open generation supports one sample and one image at a time")

    llava_model = model.model
    image_token_id = int(model.config.image_token_index)
    vision_start = time.perf_counter()
    inputs_embeds = llava_model.get_input_embeddings()(input_ids)
    image_positions = torch_module.argwhere(input_ids[0].eq(image_token_id)).squeeze(1)
    num_visual_tokens = int(image_positions.numel())
    rows, cols = _llava_token_grid_shape(model, processor, num_visual_tokens)
    token_boxes = make_token_grid(rows, cols)
    image_features_list = llava_model.get_image_features(
        pixel_values=pixel_values,
        vision_feature_layer=model.config.vision_feature_layer,
        vision_feature_select_strategy=model.config.vision_feature_select_strategy,
        image_sizes=image_sizes,
    )
    image_features = torch_module.cat(image_features_list, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
    image_mask = llava_model.get_placeholder_mask(
        input_ids,
        inputs_embeds=inputs_embeds,
        image_features=image_features,
    )
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)
    synchronize_if_cuda(inputs_embeds, torch_module=torch_module)
    vision_ms = (time.perf_counter() - vision_start) * 1000.0
    if num_visual_tokens != int(image_features.shape[0]) or len(token_boxes) != num_visual_tokens:
        raise ValueError("Visual placeholder, feature, and grid token counts do not match")

    labels = torch_module.full_like(input_ids, -100)
    target_start = time.perf_counter()
    tokenizer = tokenizer_from_processor(processor)
    target_positions = _target_text_positions(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        tokenizer=tokenizer,
        probe=probe,
    )
    if int(target_positions.numel()) == 0:
        # Vicuna can encode a word differently after a newline than at a
        # standalone string boundary. Match that real prompt context, then
        # remove the newline token from the selector span.
        targets = list(probe.get("selector_target_texts", []))
        target_positions = contextual_newline_target_positions(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            tokenizer=tokenizer,
            target_texts=targets,
        )
    synchronize_if_cuda(inputs_embeds, torch_module=torch_module)
    target_text_ms = (time.perf_counter() - target_start) * 1000.0
    if int(target_positions.numel()) == 0:
        raise ValueError(f"Question target span was not found for {probe['sample_id']}")

    score_start = time.perf_counter()
    relevance, uniqueness = _embedding_relevance_and_uniqueness(
        inputs_embeds=inputs_embeds,
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        image_positions=image_positions,
        image_token_id=image_token_id,
        token_grid_h=rows,
        token_grid_w=cols,
        text_positions_override=target_positions,
    )
    synchronize_if_cuda(inputs_embeds, torch_module=torch_module)
    score_compute_ms = (time.perf_counter() - score_start) * 1000.0

    selector_impl, score_source = _selector_impl(selector)
    keep_count = fixed_keep_count(num_visual_tokens, keep_ratio)
    select_start = time.perf_counter()
    kept_indices = select_indices(
        selector_impl,
        num_tokens=num_visual_tokens,
        keep_count=keep_count,
        token_boxes=token_boxes,
        scores=relevance,
        relevance=relevance,
        uniqueness=uniqueness,
        evidence_regions=[],
        relation="",
        seed=seed,
        salt=f"{probe['sample_id']}:{keep_ratio}:{selector}",
        hybrid_core_ratio=0.50,
        hybrid_context_ratio=0.25,
        evidence_boost=0.10,
    )
    synchronize_if_cuda(inputs_embeds, torch_module=torch_module)
    selector_ms = (time.perf_counter() - select_start) * 1000.0

    materialize_start = time.perf_counter()
    keep_sequence = torch_module.ones(input_ids.shape[1], dtype=torch_module.bool, device=input_ids.device)
    drop_positions = image_positions.to(input_ids.device)
    keep_sequence[drop_positions] = False
    kept_tensor = torch_module.tensor(kept_indices, dtype=torch_module.long, device=input_ids.device)
    keep_sequence[drop_positions[kept_tensor]] = True
    pruned_inputs_embeds = inputs_embeds[:, keep_sequence, :]
    pruned_attention_mask = attention_mask[:, keep_sequence] if attention_mask is not None else None
    synchronize_if_cuda(pruned_inputs_embeds, torch_module=torch_module)
    materialize_ms = (time.perf_counter() - materialize_start) * 1000.0

    trace = {
        "sample_id": probe["sample_id"],
        "selector": selector,
        "selector_impl": selector_impl,
        "score_source": score_source,
        "target_text_token_count": int(target_positions.numel()),
        "target_keep_ratio": keep_ratio,
        "effective_keep_ratio": len(kept_indices) / num_visual_tokens,
        "full_sequence_tokens": int(input_ids.shape[1]),
        "pruned_sequence_tokens": int(pruned_inputs_embeds.shape[1]),
        "full_visual_tokens": num_visual_tokens,
        "kept_visual_tokens": len(kept_indices),
        "removal_fraction": removal_fraction(num_visual_tokens, len(kept_indices)),
        "kept_indices": kept_indices,
        "grid_h": rows,
        "grid_w": cols,
        "vision_ms": vision_ms,
        "target_text_ms": target_text_ms,
        "score_compute_ms": score_compute_ms,
        "selector_ms": selector_ms,
        "prune_materialize_ms": materialize_ms,
        "prune_overhead_ms": score_compute_ms + selector_ms + materialize_ms,
    }
    return {"inputs_embeds": pruned_inputs_embeds, "attention_mask": pruned_attention_mask}, trace


def generation_prompt(processor, probe: dict[str, Any]) -> str:
    prompt, _ = _format_llava_prompt(
        processor=processor,
        context=str(probe["question"]).replace("<image>", ""),
        continuation="",
        num_images=1,
        chat_template=None,
    )
    return prompt


def tokenizer_from_processor(processor):
    return getattr(processor, "tokenizer", processor)


def synchronize_if_cuda(tensor, *, torch_module) -> None:
    if tensor.device.type == "cuda":
        torch_module.cuda.synchronize(tensor.device)


def contextual_newline_target_positions(*, input_ids, labels, attention_mask, tokenizer, target_texts):
    import torch

    prompt_mask = labels[0].eq(-100)
    if attention_mask is not None:
        prompt_mask = prompt_mask & attention_mask[0].bool()
    prompt_positions = torch.argwhere(prompt_mask).squeeze(1)
    prompt_ids = input_ids[0, prompt_positions].detach().cpu().tolist()
    matched: list[int] = []
    for target_text in target_texts:
        encoded = tokenizer(f"\n{str(target_text).strip()}", add_special_tokens=False)
        pattern = [int(token_id) for token_id in encoded.get("input_ids", [])]
        for start in subsequence_starts(prompt_ids, pattern):
            for offset in range(len(pattern)):
                position = int(prompt_positions[start + offset].item())
                token_id = int(input_ids[0, position].item())
                piece = tokenizer.decode(
                    [token_id],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                if str(piece).strip():
                    matched.append(position)
    return torch.tensor(sorted(set(matched)), dtype=torch.long, device=input_ids.device)


def subsequence_starts(values: list[int], pattern: list[int]) -> list[int]:
    if not pattern or len(pattern) > len(values):
        return []
    width = len(pattern)
    return [idx for idx in range(len(values) - width + 1) if values[idx : idx + width] == pattern]


def clean_generation(text: str) -> str:
    answer = str(text).strip()
    for marker in ("ASSISTANT:", "Assistant:", "assistant:"):
        if marker in answer:
            answer = answer.split(marker)[-1].strip()
    return answer


def load_reused_full_rows(
    path: str,
    *,
    samples: list[dict[str, Any]],
    task: str,
    max_new_tokens: int,
    pretrained: str,
) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    source = Path(path)
    metrics_path = source.parent / "metrics.json"
    if not source.is_file() or not metrics_path.is_file():
        raise FileNotFoundError("Reused LLaVA rows require generation JSONL and sibling metrics.json")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    required = {
        "backend": "llava",
        "task": task,
        "max_new_tokens": max_new_tokens,
        "pretrained": pretrained,
    }
    mismatches = {key: (metrics.get(key), value) for key, value in required.items() if metrics.get(key) != value}
    if mismatches:
        raise ValueError(f"Reused full rows do not match this run: {mismatches}")
    rows = {str(row["sample_id"]): row for row in read_jsonl(source)}
    missing = [sample["sample_id"] for sample in samples if sample["sample_id"] not in rows]
    if missing:
        raise ValueError(f"Reused full rows are missing {len(missing)} samples")
    return rows


def summarize(
    rows: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    task_metric: str,
) -> dict[str, Any]:
    return {
        "backend": "llava",
        "task": args.task,
        "note": "Greedy original-question generation; selector receives the question only and never the answer.",
        "n": len(rows),
        "pretrained": args.pretrained,
        "selector": args.selector,
        "selector_target_source": "question",
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
        "mean_prune_overhead_ms": mean(float(trace["prune_overhead_ms"]) for trace in traces),
        "mean_generation_ms": mean(float(trace["generation_ms"]) for trace in traces),
    }


def write_markdown(path: Path, metrics: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        f"# LLaVA {metrics['task']} Native Open-Answer Generation",
        "",
        metrics["note"],
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| n | {metrics['n']} |",
        f"| primary metric | {metrics['primary_metric']} |",
        f"| full score | {metrics['full_score']:.3f} |",
        f"| pruned score | {metrics['pruned_score']:.3f} |",
        f"| delta score | {metrics['score_delta_pruned_minus_full']:+.3f} |",
        f"| effective keep | {metrics['mean_effective_keep_ratio']:.3f} |",
        f"| selector target tokens | {metrics['mean_target_text_token_count']:.1f} |",
        f"| selector/materialization ms | {metrics['mean_prune_overhead_ms']:.1f} |",
        f"| pruned generation ms | {metrics['mean_generation_ms']:.1f} |",
        "",
        "## Changed Scores",
        "",
        "| sample | gold | full | pruned | full score | pruned score |",
        "|---|---|---|---|---:|---:|",
    ]
    for row in [row for row in rows if abs(float(row["score_delta_pruned_minus_full"])) > 1e-9][:40]:
        lines.append(
            f"| {row['sample_id']} | {'; '.join(row['gold_answers'])} | {row['full_answer']} | "
            f"{row['pruned_answer']} | {row['full_score']:.3f} | {row['pruned_score']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    main()
