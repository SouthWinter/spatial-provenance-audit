#!/usr/bin/env python
"""Batch LLM-prefill benchmark for LLaVA and InternVL visual-token pruning."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recap.images import probe_to_visual, validate_probe_images
from recap.internvl_direct_backend import (
    _load_internvl_direct,
    _model_dtype as _internvl_model_dtype,
    _move_inputs as _move_internvl_inputs,
)
from recap.internvl_pruned_backend import _internvl_patch_token_grid, _internvl_token_boxes
from recap.io import read_jsonl
from recap.llava_direct_backend import (
    _format_llava_prompt,
    _load_llava_direct,
    _model_dtype as _llava_model_dtype,
    _move_inputs as _move_llava_inputs,
)
from recap.llava_pruned_backend import _llava_token_grid_shape
from recap.probes import build_probe_dataset
from recap.prune.budgets import fixed_keep_count, removal_fraction
from recap.prune.metrics import (
    evidence_coverage,
    evidence_regions_from_sample,
    make_token_grid,
)
from recap.prune.selectors import select_indices
from recap.qwen_pruned_backend import (
    _embedding_relevance_and_uniqueness,
    _evidence_relevance,
    _selector_impl,
    _target_text_positions,
)
from scripts.benchmark_qwen_prune_efficiency import mean, median, safe_speedup


@dataclass(frozen=True)
class BatchPruneConfig:
    selector: str
    keep_ratio: float
    seed: int = 13
    hybrid_core_ratio: float = 0.50
    hybrid_context_ratio: float = 0.25
    evidence_boost: float = 0.10


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-family", required=True, choices=["llava", "internvl"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--is-probes", action="store_true", help="Treat --input as an already-built probe JSONL.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--selector", default="grid")
    parser.add_argument("--keep-ratio", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--hybrid-core-ratio", type=float, default=0.50)
    parser.add_argument("--hybrid-context-ratio", type=float, default=0.25)
    parser.add_argument("--evidence-boost", type=float, default=0.10)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--low-cpu-mem-usage", action="store_true")
    parser.add_argument("--attn-implementation", default="")
    parser.add_argument("--chat-template", default=None)
    parser.add_argument("--min-patches", type=int, default=1)
    parser.add_argument("--max-patches", type=int, default=12)
    parser.add_argument("--target-delimiter", default=" ")
    parser.add_argument("--keep-non-left-right", action="store_true")
    parser.add_argument("--strict-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--use-cache", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = read_jsonl(args.input)
    if args.is_probes:
        probes = [dict(sample) for sample in samples if str(sample.get("probe", "orig")) == "orig"]
    else:
        probes = build_probe_dataset(samples, require_left_right=not args.keep_non_left_right, probe_mode="profile_fast")
        probes = [probe for probe in probes if probe.get("probe") == "orig"]
    if args.limit is not None:
        probes = probes[: args.limit]
    for probe in probes:
        probe["probe_count"] = 1

    image_report = validate_probe_images(probes)
    if args.strict_images and image_report["missing_visual_count"]:
        raise FileNotFoundError(f"Missing images for visual probes: {image_report}")
    if image_report["missing_visual_count"]:
        print(f"[BatchPrefill] WARNING missing images for visual probes: {image_report}", flush=True)

    prune_config = BatchPruneConfig(
        selector=args.selector,
        keep_ratio=args.keep_ratio,
        seed=args.seed,
        hybrid_core_ratio=args.hybrid_core_ratio,
        hybrid_context_ratio=args.hybrid_context_ratio,
        evidence_boost=args.evidence_boost,
    )

    import torch

    model, processor, input_device = load_model(args, torch)

    rows = []
    for start in tqdm(range(0, len(probes), args.batch_size), desc=f"{args.model_family} batch prune prefill"):
        batch_probes = probes[start : start + args.batch_size]
        if len(batch_probes) < args.batch_size:
            continue
        calls = [
            prepare_probe_calls(
                probe,
                model_family=args.model_family,
                model=model,
                processor=processor,
                input_device=input_device,
                prune_config=prune_config,
                target_delimiter=args.target_delimiter,
                chat_template=args.chat_template,
                min_patches=args.min_patches,
                max_patches=args.max_patches,
                strict_images=args.strict_images,
                torch_module=torch,
            )
            for probe in batch_probes
        ]
        full_batch = collate_calls([call["full_call"] for call in calls])
        pruned_batch = collate_calls([call["pruned_call"] for call in calls])

        for _ in range(max(0, args.warmup)):
            run_language_batch(model.model, full_batch, use_cache=args.use_cache)
            run_language_batch(model.model, pruned_batch, use_cache=args.use_cache)

        full_measurements = [
            time_language_batch_with_memory(model.model, full_batch, use_cache=args.use_cache)
            for _ in range(args.repeat)
        ]
        pruned_measurements = [
            time_language_batch_with_memory(model.model, pruned_batch, use_cache=args.use_cache)
            for _ in range(args.repeat)
        ]
        full_ms = [float(measurement["ms"]) for measurement in full_measurements]
        pruned_ms = [float(measurement["ms"]) for measurement in pruned_measurements]
        full_median = median(full_ms)
        pruned_median = median(pruned_ms)
        full_samples_per_s = args.batch_size * 1000.0 / full_median if full_median > 0.0 else 0.0
        pruned_samples_per_s = args.batch_size * 1000.0 / pruned_median if pruned_median > 0.0 else 0.0
        full_tokens_per_s = full_batch["token_count"] * 1000.0 / full_median if full_median > 0.0 else 0.0
        pruned_tokens_per_s = pruned_batch["token_count"] * 1000.0 / pruned_median if pruned_median > 0.0 else 0.0
        score_compute_ms = sum(float(call["score_compute_ms"]) for call in calls)
        target_text_ms = sum(float(call["target_text_ms"]) for call in calls)
        selector_ms = sum(float(call["selector_ms"]) for call in calls)
        prune_materialize_ms = sum(float(call["prune_materialize_ms"]) for call in calls)
        prune_overhead_ms = score_compute_ms + selector_ms + prune_materialize_ms
        saved_prefill_ms = max(0.0, full_median - pruned_median)
        full_visual_tokens = sum(int(call["full_visual_tokens"]) for call in calls)
        kept_visual_tokens = sum(int(call["kept_visual_tokens"]) for call in calls)
        rows.append(
            {
                "batch_index": start // args.batch_size,
                "batch_size": args.batch_size,
                "model_family": args.model_family,
                "selector": args.selector,
                "keep_ratio": args.keep_ratio,
                "full_ms": full_median,
                "pruned_ms": pruned_median,
                "speedup": safe_speedup(full_median, pruned_median),
                "full_samples_per_s": full_samples_per_s,
                "pruned_samples_per_s": pruned_samples_per_s,
                "sample_throughput_speedup": safe_speedup(pruned_samples_per_s, full_samples_per_s),
                "full_tokens_per_s": full_tokens_per_s,
                "pruned_tokens_per_s": pruned_tokens_per_s,
                "full_peak_allocated_gb": median([float(m["peak_allocated_gb"]) for m in full_measurements]),
                "pruned_peak_allocated_gb": median([float(m["peak_allocated_gb"]) for m in pruned_measurements]),
                "full_peak_reserved_gb": median([float(m["peak_reserved_gb"]) for m in full_measurements]),
                "pruned_peak_reserved_gb": median([float(m["peak_reserved_gb"]) for m in pruned_measurements]),
                "full_incremental_peak_allocated_gb": median(
                    [float(m["incremental_peak_allocated_gb"]) for m in full_measurements]
                ),
                "pruned_incremental_peak_allocated_gb": median(
                    [float(m["incremental_peak_allocated_gb"]) for m in pruned_measurements]
                ),
                "full_tokens": full_batch["token_count"],
                "pruned_tokens": pruned_batch["token_count"],
                "token_keep_ratio": pruned_batch["token_count"] / float(full_batch["token_count"]),
                "full_visual_tokens": full_visual_tokens,
                "kept_visual_tokens": kept_visual_tokens,
                "visual_keep_ratio": kept_visual_tokens / float(full_visual_tokens) if full_visual_tokens else 1.0,
                "mean_ecr": mean([float(call["ecr"]) for call in calls]),
                "prepare_ms": sum(float(call["prepare_ms"]) for call in calls),
                "build_inputs_ms": sum(float(call["build_inputs_ms"]) for call in calls),
                "vision_feature_ms": sum(float(call["vision_feature_ms"]) for call in calls),
                "target_text_ms": target_text_ms,
                "score_compute_ms": score_compute_ms,
                "selector_ms": selector_ms,
                "prune_materialize_ms": prune_materialize_ms,
                "prune_overhead_ms": prune_overhead_ms,
                "selector_over_saved_prefill_pct": 100.0 * selector_ms / saved_prefill_ms if saved_prefill_ms > 0.0 else 0.0,
                "prune_over_saved_prefill_pct": 100.0 * prune_overhead_ms / saved_prefill_ms if saved_prefill_ms > 0.0 else 0.0,
                "mean_prepare_ms_per_sample": mean([float(call["prepare_ms"]) for call in calls]),
                "mean_selector_ms_per_sample": mean([float(call["selector_ms"]) for call in calls]),
                "full_ms_values": full_ms,
                "pruned_ms_values": pruned_ms,
                "full_memory_values": full_measurements,
                "pruned_memory_values": pruned_measurements,
            }
        )

    write_jsonl(out_dir / "batch_prefill_rows.jsonl", rows)
    summary = summarize(rows, args=vars(args))
    (out_dir / "batch_prefill_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "batch_prefill_summary.md").write_text(markdown(summary), encoding="utf-8")
    print(f"Wrote {len(rows)} batch rows to {out_dir}")
    print(
        "Mean batch prefill speedup: "
        f"{summary['mean_speedup']:.3f}x keep={summary['mean_visual_keep_ratio']:.4f}",
        flush=True,
    )


def load_model(args, torch_module):
    if args.model_family == "llava":
        return _load_llava_direct(
            pretrained=args.pretrained,
            revision=args.revision,
            device=args.device,
            device_map=args.device_map,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            attn_implementation=args.attn_implementation or None,
            torch_module=torch_module,
        )
    return _load_internvl_direct(
        pretrained=args.pretrained,
        revision=args.revision,
        device=args.device,
        device_map=args.device_map,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        attn_implementation=args.attn_implementation or None,
        torch_module=torch_module,
    )


def prepare_probe_calls(
    probe: dict[str, Any],
    *,
    model_family: str,
    model,
    processor,
    input_device,
    prune_config: BatchPruneConfig,
    target_delimiter: str,
    chat_template: str | None,
    min_patches: int,
    max_patches: int,
    strict_images: bool,
    torch_module,
) -> dict[str, Any]:
    visuals = probe_to_visual(probe, strict=strict_images)
    if not isinstance(visuals, list):
        visuals = [visuals]
    pil_visuals = [visual for visual in visuals if isinstance(visual, Image.Image)]
    if model_family == "llava":
        return prepare_llava_probe_calls(
            probe,
            visuals=pil_visuals,
            continuation=f"{target_delimiter}yes",
            model=model,
            processor=processor,
            input_device=input_device,
            prune_config=prune_config,
            chat_template=chat_template,
            torch_module=torch_module,
        )
    return prepare_internvl_probe_calls(
        probe,
        visuals=pil_visuals,
        continuation=f"{target_delimiter}yes",
        model=model,
        processor=processor,
        input_device=input_device,
        prune_config=prune_config,
        min_patches=min_patches,
        max_patches=max_patches,
        torch_module=torch_module,
    )


def prepare_llava_probe_calls(
    probe: dict[str, Any],
    *,
    visuals: list[Image.Image],
    continuation: str,
    model,
    processor,
    input_device,
    prune_config: BatchPruneConfig,
    chat_template: str | None,
    torch_module,
) -> dict[str, Any]:
    prepare_start = time.perf_counter()
    build_start = time.perf_counter()
    context = str(probe["question"]).replace("<image>", "")
    prompt, prompt_and_continuation = _format_llava_prompt(
        processor=processor,
        context=context,
        continuation=continuation,
        num_images=len(visuals),
        chat_template=chat_template,
    )
    if visuals:
        inputs = processor(text=[prompt_and_continuation], images=visuals, return_tensors="pt")
        prompt_inputs = processor(text=[prompt], images=visuals, return_tensors="pt")
    else:
        inputs = processor(text=[prompt_and_continuation], return_tensors="pt")
        prompt_inputs = processor(text=[prompt], return_tensors="pt")
    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    inputs = _move_llava_inputs(inputs, input_device, _llava_model_dtype(model), torch_module)
    labels = inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100
    build_inputs_ms = (time.perf_counter() - build_start) * 1000.0

    if not visuals or "pixel_values" not in inputs or inputs.get("pixel_values") is None:
        return no_prune_calls(inputs=inputs, prepare_start=prepare_start, build_inputs_ms=build_inputs_ms)
    if inputs["input_ids"].shape[0] != 1 or int(inputs["pixel_values"].shape[0]) != 1:
        raise ValueError("The LLaVA batch prefill benchmark supports one image per probe.")

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    llava_model = model.model
    image_token_id = int(model.config.image_token_index)
    vision_start = time.perf_counter()
    with torch_module.inference_mode():
        inputs_embeds = llava_model.get_input_embeddings()(input_ids)
        image_features_list = llava_model.get_image_features(
            pixel_values=inputs["pixel_values"],
            vision_feature_layer=model.config.vision_feature_layer,
            vision_feature_select_strategy=model.config.vision_feature_select_strategy,
            image_sizes=inputs.get("image_sizes"),
        )
        image_features = torch_module.cat(image_features_list, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        special_image_mask = llava_model.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            image_features=image_features,
        )
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
    sync_tensor_device(input_ids)
    vision_feature_ms = (time.perf_counter() - vision_start) * 1000.0

    image_positions = torch_module.argwhere(input_ids[0].eq(image_token_id)).squeeze(1)
    full_visual_tokens = int(image_positions.numel())
    if full_visual_tokens != int(image_features.shape[0]):
        raise ValueError(
            f"Prompt has {full_visual_tokens} image tokens but vision tower returned {int(image_features.shape[0])} features."
        )
    token_grid_h, token_grid_w = _llava_token_grid_shape(model, processor, full_visual_tokens)
    token_boxes = make_token_grid(token_grid_h, token_grid_w)
    return prune_calls_from_embeddings(
        probe,
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        image_positions=image_positions,
        image_token_id=image_token_id,
        token_boxes=token_boxes,
        token_grid_h=token_grid_h,
        token_grid_w=token_grid_w,
        prune_config=prune_config,
        prepare_start=prepare_start,
        build_inputs_ms=build_inputs_ms,
        vision_feature_ms=vision_feature_ms,
        tokenizer=getattr(processor, "tokenizer", processor),
    )


def prepare_internvl_probe_calls(
    probe: dict[str, Any],
    *,
    visuals: list[Image.Image],
    continuation: str,
    model,
    processor,
    input_device,
    prune_config: BatchPruneConfig,
    min_patches: int,
    max_patches: int,
    torch_module,
) -> dict[str, Any]:
    prepare_start = time.perf_counter()
    build_start = time.perf_counter()
    context = str(probe["question"]).replace("<image>", "")
    user_content: list[dict[str, Any]] = [{"type": "image"} for _ in visuals]
    user_content.append({"type": "text", "text": context})
    prompt_messages = [{"role": "user", "content": user_content}]
    full_messages = prompt_messages + [{"role": "assistant", "content": continuation}]
    prompt_text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    full_text = processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
    processor_kwargs = {"min_patches": min_patches, "max_patches": max_patches}
    image_inputs = visuals or None
    inputs = processor(images=image_inputs, text=full_text, return_tensors="pt", **processor_kwargs)
    prompt_inputs = processor(images=image_inputs, text=prompt_text, return_tensors="pt", **processor_kwargs)
    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    inputs = _move_internvl_inputs(inputs, input_device, _internvl_model_dtype(model), torch_module)
    labels = inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100
    build_inputs_ms = (time.perf_counter() - build_start) * 1000.0

    if not visuals or "pixel_values" not in inputs or inputs.get("pixel_values") is None:
        return no_prune_calls(inputs=inputs, prepare_start=prepare_start, build_inputs_ms=build_inputs_ms)
    if inputs["input_ids"].shape[0] != 1 or len(visuals) != 1:
        raise ValueError("The InternVL batch prefill benchmark supports one image per probe.")

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    internvl_model = model.model
    image_token_id = int(model.config.image_token_id)
    vision_start = time.perf_counter()
    with torch_module.inference_mode():
        inputs_embeds = internvl_model.get_input_embeddings()(input_ids)
        image_features = internvl_model.get_image_features(
            pixel_values=inputs["pixel_values"],
            vision_feature_layer=model.config.vision_feature_layer,
            vision_feature_select_strategy=model.config.vision_feature_select_strategy,
        ).to(inputs_embeds.device, inputs_embeds.dtype)
        special_image_mask = internvl_model.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            image_features=image_features,
        )
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
    sync_tensor_device(input_ids)
    vision_feature_ms = (time.perf_counter() - vision_start) * 1000.0

    image_positions = torch_module.argwhere(input_ids[0].eq(image_token_id)).squeeze(1)
    full_visual_tokens = int(image_positions.numel())
    num_image_patches = int(image_features.shape[0])
    tokens_per_patch = int(image_features.shape[1])
    if full_visual_tokens != num_image_patches * tokens_per_patch:
        raise ValueError(
            f"Prompt has {full_visual_tokens} image tokens but vision tower returned "
            f"{num_image_patches}x{tokens_per_patch} features."
        )
    patch_grid_h, patch_grid_w = _internvl_patch_token_grid(model, tokens_per_patch)
    token_grid_h, token_grid_w = patch_grid_h, patch_grid_w * max(1, num_image_patches)
    token_boxes, _, _, _ = _internvl_token_boxes(
        image=visuals[0],
        processor=processor,
        num_image_patches=num_image_patches,
        patch_grid_h=patch_grid_h,
        patch_grid_w=patch_grid_w,
        min_patches=min_patches,
        max_patches=max_patches,
    )
    return prune_calls_from_embeddings(
        probe,
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        image_positions=image_positions,
        image_token_id=image_token_id,
        token_boxes=token_boxes,
        token_grid_h=token_grid_h,
        token_grid_w=token_grid_w,
        prune_config=prune_config,
        prepare_start=prepare_start,
        build_inputs_ms=build_inputs_ms,
        vision_feature_ms=vision_feature_ms,
        tokenizer=getattr(processor, "tokenizer", processor),
    )


def prune_calls_from_embeddings(
    probe: dict[str, Any],
    *,
    input_ids,
    labels,
    attention_mask,
    inputs_embeds,
    image_positions,
    image_token_id: int,
    token_boxes,
    token_grid_h: int,
    token_grid_w: int,
    prune_config: BatchPruneConfig,
    prepare_start: float,
    build_inputs_ms: float,
    vision_feature_ms: float,
    tokenizer,
) -> dict[str, Any]:
    import torch

    full_visual_tokens = int(image_positions.numel())
    if len(token_boxes) != full_visual_tokens:
        raise ValueError(f"Token geometry has {len(token_boxes)} boxes but prompt has {full_visual_tokens} image tokens.")

    evidence_regions = evidence_regions_from_sample(probe)
    selector_impl, score_source = _selector_impl(prune_config.selector)
    target_text_ms = 0.0
    score_compute_start = time.perf_counter()
    if score_source in {"embedding", "target_embedding"}:
        text_positions_override = None
        if score_source == "target_embedding":
            target_text_start = time.perf_counter()
            text_positions_override = _target_text_positions(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                tokenizer=tokenizer,
                probe=probe,
            )
            sync_tensor_device(input_ids)
            target_text_ms = (time.perf_counter() - target_text_start) * 1000.0
        relevance, uniqueness = _embedding_relevance_and_uniqueness(
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            image_positions=image_positions,
            image_token_id=image_token_id,
            token_grid_h=token_grid_h,
            token_grid_w=token_grid_w,
            text_positions_override=text_positions_override,
        )
    else:
        relevance = _evidence_relevance(token_boxes, evidence_regions)
        uniqueness = [0.0 for _ in token_boxes]
    sync_tensor_device(input_ids)
    score_compute_ms = (time.perf_counter() - score_compute_start) * 1000.0

    selector_start = time.perf_counter()
    keep_count = fixed_keep_count(full_visual_tokens, prune_config.keep_ratio)
    kept_indices = select_indices(
        selector_impl,
        num_tokens=full_visual_tokens,
        keep_count=keep_count,
        token_boxes=token_boxes,
        scores=relevance,
        relevance=relevance,
        uniqueness=uniqueness,
        evidence_regions=evidence_regions,
        relation=str(probe.get("relation", probe.get("base_relation", ""))),
        seed=prune_config.seed,
        salt=f"{probe.get('sample_id', probe.get('id', ''))}:{probe.get('probe', '')}:{prune_config.keep_ratio}:{prune_config.selector}",
        hybrid_core_ratio=prune_config.hybrid_core_ratio,
        hybrid_context_ratio=prune_config.hybrid_context_ratio,
        evidence_boost=prune_config.evidence_boost,
    )
    sync_tensor_device(input_ids)
    selector_ms = (time.perf_counter() - selector_start) * 1000.0

    prune_materialize_start = time.perf_counter()
    keep_sequence = torch.ones(input_ids.shape[1], dtype=torch.bool, device=input_ids.device)
    drop_positions = image_positions.to(input_ids.device)
    keep_sequence[drop_positions] = False
    kept_tensor = torch.tensor(kept_indices, dtype=torch.long, device=input_ids.device)
    keep_sequence[drop_positions[kept_tensor]] = True
    pruned_inputs_embeds = inputs_embeds[:, keep_sequence, :]
    pruned_attention_mask = attention_mask[:, keep_sequence] if attention_mask is not None else None
    sync_tensor_device(pruned_inputs_embeds)
    prune_materialize_ms = (time.perf_counter() - prune_materialize_start) * 1000.0

    return {
        "full_call": {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask},
        "pruned_call": {"inputs_embeds": pruned_inputs_embeds, "attention_mask": pruned_attention_mask},
        "full_visual_tokens": full_visual_tokens,
        "kept_visual_tokens": len(kept_indices),
        "visual_removal_fraction": removal_fraction(full_visual_tokens, len(kept_indices)),
        "ecr": evidence_coverage(kept_indices, token_boxes, evidence_regions),
        "prepare_ms": (time.perf_counter() - prepare_start) * 1000.0,
        "build_inputs_ms": build_inputs_ms,
        "vision_feature_ms": vision_feature_ms,
        "target_text_ms": target_text_ms,
        "score_compute_ms": score_compute_ms,
        "selector_ms": selector_ms,
        "prune_materialize_ms": prune_materialize_ms,
    }


def no_prune_calls(*, inputs, prepare_start: float, build_inputs_ms: float) -> dict[str, Any]:
    import torch

    input_ids = inputs["input_ids"]
    embeds = torch.zeros((input_ids.shape[0], input_ids.shape[1], 1), dtype=torch.float16, device=input_ids.device)
    return {
        "full_call": {"inputs_embeds": embeds, "attention_mask": inputs.get("attention_mask")},
        "pruned_call": {"inputs_embeds": embeds, "attention_mask": inputs.get("attention_mask")},
        "full_visual_tokens": 0,
        "kept_visual_tokens": 0,
        "visual_removal_fraction": 0.0,
        "ecr": 0.0,
        "prepare_ms": (time.perf_counter() - prepare_start) * 1000.0,
        "build_inputs_ms": build_inputs_ms,
        "vision_feature_ms": 0.0,
        "target_text_ms": 0.0,
        "score_compute_ms": 0.0,
        "selector_ms": 0.0,
        "prune_materialize_ms": 0.0,
    }


def collate_calls(calls: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    batch_size = len(calls)
    max_len = max(int(call["inputs_embeds"].shape[1]) for call in calls)
    hidden = int(calls[0]["inputs_embeds"].shape[-1])
    device = calls[0]["inputs_embeds"].device
    dtype = calls[0]["inputs_embeds"].dtype
    attention_dtype = calls[0]["attention_mask"].dtype if calls[0]["attention_mask"] is not None else torch.long
    embeds = torch.zeros((batch_size, max_len, hidden), dtype=dtype, device=device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=attention_dtype, device=device)
    token_count = 0
    for idx, call in enumerate(calls):
        length = int(call["inputs_embeds"].shape[1])
        embeds[idx, :length, :] = call["inputs_embeds"][0]
        attention_mask[idx, :length] = 1
        token_count += length
    return {
        "inputs_embeds": embeds,
        "attention_mask": attention_mask,
        "cache_position": torch.arange(max_len, device=device),
        "token_count": token_count,
    }


def run_language_batch(backbone_model, batch: dict[str, Any], *, use_cache: bool):
    with __import__("torch").inference_mode():
        kwargs = {
            "attention_mask": batch["attention_mask"],
            "inputs_embeds": batch["inputs_embeds"],
            "cache_position": batch["cache_position"],
            "use_cache": use_cache,
        }
        try:
            outputs = backbone_model.language_model(**kwargs)
        except TypeError:
            kwargs.pop("use_cache", None)
            outputs = backbone_model.language_model(**kwargs)
    sync_tensor_device(batch["inputs_embeds"])
    return outputs


def time_language_batch_with_memory(backbone_model, batch: dict[str, Any], *, use_cache: bool) -> dict[str, float]:
    import torch

    sync_tensor_device(batch["inputs_embeds"])
    device = batch["inputs_embeds"].device
    track_cuda = device.type == "cuda" and torch.cuda.is_available()
    base_allocated = 0
    base_reserved = 0
    if track_cuda:
        torch.cuda.synchronize(device)
        base_allocated = torch.cuda.memory_allocated(device)
        base_reserved = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    outputs = run_language_batch(backbone_model, batch, use_cache=use_cache)
    del outputs
    sync_tensor_device(batch["inputs_embeds"])
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if not track_cuda:
        return {
            "ms": elapsed_ms,
            "peak_allocated_gb": 0.0,
            "peak_reserved_gb": 0.0,
            "incremental_peak_allocated_gb": 0.0,
            "incremental_peak_reserved_gb": 0.0,
        }
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    gb = 1024.0**3
    return {
        "ms": elapsed_ms,
        "peak_allocated_gb": peak_allocated / gb,
        "peak_reserved_gb": peak_reserved / gb,
        "incremental_peak_allocated_gb": max(0, peak_allocated - base_allocated) / gb,
        "incremental_peak_reserved_gb": max(0, peak_reserved - base_reserved) / gb,
    }


def summarize(rows: list[dict[str, Any]], *, args: dict[str, Any]) -> dict[str, Any]:
    out = {"args": args, "num_batches": len(rows), "num_samples": len(rows) * int(args["batch_size"])}
    for key in (
        "full_ms",
        "pruned_ms",
        "speedup",
        "full_samples_per_s",
        "pruned_samples_per_s",
        "sample_throughput_speedup",
        "full_tokens_per_s",
        "pruned_tokens_per_s",
        "full_peak_allocated_gb",
        "pruned_peak_allocated_gb",
        "full_peak_reserved_gb",
        "pruned_peak_reserved_gb",
        "full_incremental_peak_allocated_gb",
        "pruned_incremental_peak_allocated_gb",
        "prepare_ms",
        "build_inputs_ms",
        "vision_feature_ms",
        "target_text_ms",
        "score_compute_ms",
        "selector_ms",
        "prune_materialize_ms",
        "prune_overhead_ms",
        "selector_over_saved_prefill_pct",
        "prune_over_saved_prefill_pct",
        "mean_prepare_ms_per_sample",
        "mean_selector_ms_per_sample",
        "token_keep_ratio",
        "visual_keep_ratio",
        "mean_ecr",
    ):
        values = [float(row[key]) for row in rows]
        out[f"mean_{key}"] = mean(values)
        out[f"median_{key}"] = median(values)
    out["mean_time_reduction_pct"] = (
        100.0 * (out["mean_full_ms"] - out["mean_pruned_ms"]) / out["mean_full_ms"]
        if out["mean_full_ms"] > 0.0
        else 0.0
    )
    out["mean_peak_allocated_reduction_pct"] = (
        100.0 * (out["mean_full_peak_allocated_gb"] - out["mean_pruned_peak_allocated_gb"])
        / out["mean_full_peak_allocated_gb"]
        if out["mean_full_peak_allocated_gb"] > 0.0
        else 0.0
    )
    out["mean_incremental_peak_allocated_reduction_pct"] = (
        100.0
        * (out["mean_full_incremental_peak_allocated_gb"] - out["mean_pruned_incremental_peak_allocated_gb"])
        / out["mean_full_incremental_peak_allocated_gb"]
        if out["mean_full_incremental_peak_allocated_gb"] > 0.0
        else 0.0
    )
    return out


def markdown(summary: dict[str, Any]) -> str:
    title = f"{summary['args']['model_family']} batch prefill pruning benchmark"
    lines = [
        f"# {title}",
        "",
        f"Samples: {summary['num_samples']}",
        f"Batch size: {summary['args']['batch_size']}",
        f"Selector: {summary['args']['selector']} @ keep={summary['args']['keep_ratio']}",
        "",
        "| metric | mean | median |",
        "|---|---:|---:|",
    ]
    for label, key in (
        ("Full batch prefill ms", "full_ms"),
        ("Pruned batch prefill ms", "pruned_ms"),
        ("Batch prefill speedup", "speedup"),
        ("Full samples/s", "full_samples_per_s"),
        ("Pruned samples/s", "pruned_samples_per_s"),
        ("Sample throughput speedup", "sample_throughput_speedup"),
        ("Full tokens/s", "full_tokens_per_s"),
        ("Pruned tokens/s", "pruned_tokens_per_s"),
        ("Full peak allocated GB", "full_peak_allocated_gb"),
        ("Pruned peak allocated GB", "pruned_peak_allocated_gb"),
        ("Full incremental peak allocated GB", "full_incremental_peak_allocated_gb"),
        ("Pruned incremental peak allocated GB", "pruned_incremental_peak_allocated_gb"),
        ("Prepare overhead ms/batch", "prepare_ms"),
        ("Build inputs ms/batch", "build_inputs_ms"),
        ("Vision feature ms/batch", "vision_feature_ms"),
        ("Target text ms/batch", "target_text_ms"),
        ("Score compute ms/batch", "score_compute_ms"),
        ("Selector ms/batch", "selector_ms"),
        ("Prune materialize ms/batch", "prune_materialize_ms"),
        ("Prune overhead ms/batch", "prune_overhead_ms"),
        ("Selector / saved prefill %", "selector_over_saved_prefill_pct"),
        ("Prune overhead / saved prefill %", "prune_over_saved_prefill_pct"),
        ("Prepare overhead ms/sample", "mean_prepare_ms_per_sample"),
        ("Selector ms/sample", "mean_selector_ms_per_sample"),
        ("Token keep ratio", "token_keep_ratio"),
        ("Visual keep ratio", "visual_keep_ratio"),
        ("ECR", "mean_ecr"),
    ):
        lines.append(f"| {label} | {summary[f'mean_{key}']:.4f} | {summary[f'median_{key}']:.4f} |")
    lines.extend(
        [
            "",
            f"Mean batch prefill time reduction: {summary['mean_time_reduction_pct']:.2f}%",
            f"Mean peak allocated memory reduction: {summary['mean_peak_allocated_reduction_pct']:.2f}%",
            "Mean incremental peak allocated memory reduction: "
            f"{summary['mean_incremental_peak_allocated_reduction_pct']:.2f}%",
            "",
        ]
    )
    return "\n".join(lines)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sync_tensor_device(tensor) -> None:
    try:
        import torch

        if getattr(tensor, "is_cuda", False):
            torch.cuda.synchronize(tensor.device)
    except Exception:
        pass


if __name__ == "__main__":
    main()
