#!/usr/bin/env python
"""Benchmark Qwen-VL prefill speed with and without visual-token pruning.

This benchmark reuses the same internal pruning path as ``run-qwen-pruned`` but
does not score yes/no continuations. It measures the LLM prefill call after the
visual encoder has produced image features, then reports both isolated LLM
speedup and conservative end-to-end estimated speedup with the shared visual
encoding time included.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recap.images import encode_image_data_url, probe_to_visual, validate_probe_images
from recap.io import read_jsonl
from recap.probes import build_probe_dataset
from recap.prune.budgets import fixed_keep_count, removal_fraction
from recap.prune.metrics import evidence_coverage, evidence_regions_from_sample, make_token_grid
from recap.prune.selectors import select_indices
from recap.qwen_direct_backend import _load_qwen_direct
from recap.qwen_pruned_backend import (
    PruneConfig,
    _effective_keep_ratio,
    _evidence_relevance,
    _spatial_uniqueness,
    _sync_tensor_device,
    _token_grid_shape,
)


DEFAULT_EXTRA_TEXT = (
    " Carefully inspect the image and preserve all spatial evidence before answering. "
    "Focus on the mentioned subject, object, and their relative location."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Canonical RECAP JSONL.")
    parser.add_argument("--is-probes", action="store_true", help="Treat --input as an already-built probe JSONL.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--selector", default="hybrid")
    parser.add_argument("--keep-ratio", type=float, default=0.50)
    parser.add_argument("--budget-mode", default="fixed", choices=["fixed", "sensitivity_policy", "risk_adaptive", "risk_bucket"])
    parser.add_argument("--budget-ratios", default="")
    parser.add_argument("--budget-ratio-key", default="keep_ratio")
    parser.add_argument("--rho-min", type=float, default=0.15)
    parser.add_argument("--rho-max", type=float, default=0.70)
    parser.add_argument("--hybrid-core-ratio", type=float, default=0.50)
    parser.add_argument("--hybrid-context-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--attn-implementation", default="")
    parser.add_argument("--min-pixels", type=int, default=50176)
    parser.add_argument("--max-pixels", type=int, default=50176)
    parser.add_argument("--use-fast-processor", action="store_true")
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--keep-non-left-right", action="store_true")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--decode-steps", type=int, default=0, help="If >0, also benchmark cached greedy decode steps.")
    parser.add_argument("--text-repeat", type=int, default=0, help="Repeat an extra instruction to lengthen text prefill.")
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
    if image_report["missing_visual_count"]:
        raise FileNotFoundError(f"Missing images for visual probes: {image_report}")

    budget_ratios = load_budget_ratios(Path(args.budget_ratios), args.budget_ratio_key) if args.budget_ratios else {}
    if args.budget_mode == "sensitivity_policy" and not budget_ratios:
        raise ValueError("--budget-mode sensitivity_policy requires --budget-ratios")

    prune_config = PruneConfig(
        selector=args.selector,
        keep_ratio=args.keep_ratio,
        budget_mode=args.budget_mode,
        rho_min=args.rho_min,
        rho_max=args.rho_max,
        seed=args.seed,
        hybrid_core_ratio=args.hybrid_core_ratio,
        hybrid_context_ratio=args.hybrid_context_ratio,
    )

    import torch

    model, processor, _, process_vision_info, input_device = _load_qwen_direct(
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

    rows = []
    for probe in tqdm(probes, desc="Qwen prune efficiency"):
        sample_id = str(probe.get("sample_id", probe.get("id", "")))
        row = benchmark_probe(
            probe,
            model=model,
            processor=processor,
            process_vision_info=process_vision_info,
            input_device=input_device,
            system_prompt=args.system_prompt,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            prune_config=prune_config,
            sample_budget_ratio=budget_ratios.get(sample_id),
            warmup=args.warmup,
            repeat=args.repeat,
            decode_steps=args.decode_steps,
            text_repeat=args.text_repeat,
            use_cache=args.use_cache,
        )
        rows.append(row)

    write_jsonl(out_dir / "efficiency_rows.jsonl", rows)
    summary = summarize_rows(rows, args=vars(args))
    (out_dir / "efficiency_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "efficiency_summary.md").write_text(summary_markdown(summary), encoding="utf-8")
    print(f"Wrote {len(rows)} benchmark rows to {out_dir}")
    print(
        "Mean speedup: "
        f"language={summary['mean_language_speedup']:.3f}x "
        f"end_to_end_est={summary['mean_end_to_end_est_speedup']:.3f}x "
        f"keep={summary['mean_visual_keep_ratio']:.4f}"
    )


def load_budget_ratios(path: Path, key: str) -> dict[str, float]:
    out = {}
    for row in read_jsonl(path):
        sample_id = str(row.get("sample_id", row.get("id", "")))
        if sample_id and key in row:
            out[sample_id] = float(row[key])
    return out


def benchmark_probe(
    probe: dict[str, Any],
    *,
    model,
    processor,
    process_vision_info,
    input_device,
    system_prompt: str,
    min_pixels: int,
    max_pixels: int,
    prune_config: PruneConfig,
    sample_budget_ratio: float | None,
    warmup: int,
    repeat: int,
    decode_steps: int,
    text_repeat: int,
    use_cache: bool,
) -> dict[str, Any]:
    import torch

    inputs = build_inputs(
        probe,
        processor=processor,
        process_vision_info=process_vision_info,
        input_device=input_device,
        system_prompt=system_prompt,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        text_repeat=text_repeat,
    )
    qwen_model = model.model
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    pixel_values = inputs["pixel_values"]
    image_grid_thw = inputs["image_grid_thw"]

    _sync_tensor_device(input_ids)
    vision_start = time.perf_counter()
    with torch.inference_mode():
        inputs_embeds = qwen_model.get_input_embeddings()(input_ids)
        image_embeds_list, deepstack_image_embeds = qwen_model.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds_list, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = qwen_model.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
    _sync_tensor_device(input_ids)
    vision_ms = (time.perf_counter() - vision_start) * 1000.0

    image_positions = torch.argwhere((input_ids[0] == model.config.image_token_id)).squeeze(1)
    full_visual_tokens = int(image_positions.numel())
    token_grid_h, token_grid_w = _token_grid_shape(image_grid_thw[0], qwen_model.config.vision_config.spatial_merge_size)
    token_boxes = make_token_grid(token_grid_h, token_grid_w)
    if len(token_boxes) != full_visual_tokens:
        raise ValueError(f"Token grid has {len(token_boxes)} boxes but prompt has {full_visual_tokens} image tokens.")
    evidence_regions = evidence_regions_from_sample(probe)
    relevance = _evidence_relevance(token_boxes, evidence_regions)
    uniqueness = _spatial_uniqueness(token_boxes, relevance)
    effective_keep_ratio = _effective_keep_ratio(prune_config, None, sample_budget_ratio)
    keep_count = fixed_keep_count(full_visual_tokens, effective_keep_ratio)
    kept_indices = select_indices(
        prune_config.selector,
        num_tokens=full_visual_tokens,
        keep_count=keep_count,
        token_boxes=token_boxes,
        scores=relevance,
        relevance=relevance,
        uniqueness=uniqueness,
        evidence_regions=evidence_regions,
        relation=str(probe.get("relation", probe.get("base_relation", ""))),
        seed=prune_config.seed,
        salt=f"{probe.get('sample_id', probe.get('id', ''))}:{probe.get('probe', '')}:{effective_keep_ratio}:{prune_config.selector}",
        hybrid_core_ratio=prune_config.hybrid_core_ratio,
        hybrid_context_ratio=prune_config.hybrid_context_ratio,
    )

    position_ids, _ = qwen_model.get_rope_index(input_ids, image_grid_thw, attention_mask=attention_mask)
    full_call = {
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "inputs_embeds": inputs_embeds,
        "cache_position": torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device),
        "visual_pos_masks": input_ids == model.config.image_token_id,
        "deepstack_visual_embeds": [x.to(inputs_embeds.device) for x in deepstack_image_embeds],
    }

    keep_sequence = torch.ones(input_ids.shape[1], dtype=torch.bool, device=input_ids.device)
    drop_positions = image_positions.to(input_ids.device)
    keep_sequence[drop_positions] = False
    kept_tensor = torch.tensor(kept_indices, dtype=torch.long, device=input_ids.device)
    keep_sequence[drop_positions[kept_tensor]] = True
    pruned_inputs_embeds = inputs_embeds[:, keep_sequence, :]
    pruned_call = {
        "position_ids": position_ids[:, :, keep_sequence],
        "attention_mask": attention_mask[:, keep_sequence] if attention_mask is not None else None,
        "inputs_embeds": pruned_inputs_embeds,
        "cache_position": torch.arange(pruned_inputs_embeds.shape[1], device=pruned_inputs_embeds.device),
        "visual_pos_masks": (input_ids == model.config.image_token_id)[:, keep_sequence],
        "deepstack_visual_embeds": [x[kept_tensor.to(x.device)].to(pruned_inputs_embeds.device) for x in deepstack_image_embeds],
    }

    for _ in range(max(0, warmup)):
        run_language(qwen_model, full_call, use_cache=use_cache)
        run_language(qwen_model, pruned_call, use_cache=use_cache)

    full_ms = [time_language(qwen_model, full_call, use_cache=use_cache) for _ in range(repeat)]
    pruned_ms = [time_language(qwen_model, pruned_call, use_cache=use_cache) for _ in range(repeat)]

    full_language_ms = median(full_ms)
    pruned_language_ms = median(pruned_ms)
    full_decode = decode_benchmark(
        qwen_model,
        model.lm_head,
        full_call,
        decode_steps=decode_steps,
        repeat=repeat,
        use_cache=use_cache,
    )
    pruned_decode = decode_benchmark(
        qwen_model,
        model.lm_head,
        pruned_call,
        decode_steps=decode_steps,
        repeat=repeat,
        use_cache=use_cache,
    )
    full_decode_ms = full_decode["median_decode_ms_total"]
    pruned_decode_ms = pruned_decode["median_decode_ms_total"]
    full_decode_per_token_ms = full_decode["median_decode_ms_per_token"]
    pruned_decode_per_token_ms = pruned_decode["median_decode_ms_per_token"]
    end_to_end_full_ms = vision_ms + full_language_ms
    end_to_end_pruned_ms = vision_ms + pruned_language_ms
    generation_full_ms = vision_ms + full_language_ms + full_decode_ms
    generation_pruned_ms = vision_ms + pruned_language_ms + pruned_decode_ms
    full_sequence_tokens = int(inputs_embeds.shape[1])
    pruned_sequence_tokens = int(pruned_inputs_embeds.shape[1])
    kept_visual_tokens = len(kept_indices)
    return {
        "sample_id": str(probe.get("sample_id", probe.get("id", ""))),
        "relation": str(probe.get("relation", probe.get("base_relation", ""))),
        "text_repeat": int(text_repeat),
        "use_cache": bool(use_cache),
        "full_sequence_tokens": full_sequence_tokens,
        "pruned_sequence_tokens": pruned_sequence_tokens,
        "sequence_keep_ratio": pruned_sequence_tokens / float(full_sequence_tokens) if full_sequence_tokens else 0.0,
        "full_visual_tokens": full_visual_tokens,
        "kept_visual_tokens": kept_visual_tokens,
        "visual_keep_ratio": kept_visual_tokens / float(full_visual_tokens) if full_visual_tokens else 0.0,
        "visual_removal_fraction": removal_fraction(full_visual_tokens, kept_visual_tokens),
        "effective_keep_ratio": float(effective_keep_ratio),
        "ecr": evidence_coverage(kept_indices, token_boxes, evidence_regions),
        "vision_ms": vision_ms,
        "full_language_ms": full_language_ms,
        "pruned_language_ms": pruned_language_ms,
        "language_speedup": safe_speedup(full_language_ms, pruned_language_ms),
        "decode_steps": int(decode_steps),
        "full_decode_ms_total": full_decode_ms,
        "pruned_decode_ms_total": pruned_decode_ms,
        "full_decode_ms_per_token": full_decode_per_token_ms,
        "pruned_decode_ms_per_token": pruned_decode_per_token_ms,
        "decode_speedup": safe_speedup(full_decode_per_token_ms, pruned_decode_per_token_ms),
        "end_to_end_full_est_ms": end_to_end_full_ms,
        "end_to_end_pruned_est_ms": end_to_end_pruned_ms,
        "end_to_end_est_speedup": safe_speedup(end_to_end_full_ms, end_to_end_pruned_ms),
        "generation_full_est_ms": generation_full_ms,
        "generation_pruned_est_ms": generation_pruned_ms,
        "generation_est_speedup": safe_speedup(generation_full_ms, generation_pruned_ms),
        "full_language_ms_values": full_ms,
        "pruned_language_ms_values": pruned_ms,
        "full_decode_ms_values": full_decode["decode_ms_total_values"],
        "pruned_decode_ms_values": pruned_decode["decode_ms_total_values"],
    }


def build_inputs(
    probe: dict[str, Any],
    *,
    processor,
    process_vision_info,
    input_device,
    system_prompt: str,
    min_pixels: int,
    max_pixels: int,
    text_repeat: int,
):
    context = str(probe["question"]).replace("<image>", "")
    if text_repeat > 0:
        context = context + DEFAULT_EXTRA_TEXT * int(text_repeat)
    visuals = probe_to_visual(probe, strict=True)
    if not isinstance(visuals, list):
        visuals = [visuals]

    content: list[dict[str, Any]] = []
    for visual in visuals:
        if isinstance(visual, Image.Image):
            content.append(
                {
                    "type": "image",
                    "image": encode_image_data_url(visual),
                    "max_pixels": max_pixels,
                    "min_pixels": min_pixels,
                }
            )
    content.append({"type": "text", "text": context})
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    return processor(text=text, images=image_inputs, videos=video_inputs, return_tensors="pt").to(input_device)


def run_language(qwen_model, call: dict[str, Any], *, use_cache: bool):
    import torch

    with torch.inference_mode():
        outputs = qwen_model.language_model(
            input_ids=None,
            position_ids=call["position_ids"],
            attention_mask=call["attention_mask"],
            past_key_values=None,
            inputs_embeds=call["inputs_embeds"],
            cache_position=call["cache_position"],
            visual_pos_masks=call["visual_pos_masks"],
            deepstack_visual_embeds=call["deepstack_visual_embeds"],
            use_cache=use_cache,
        )
    _sync_tensor_device(call["inputs_embeds"])
    return outputs


def decode_benchmark(
    qwen_model,
    lm_head,
    call: dict[str, Any],
    *,
    decode_steps: int,
    repeat: int,
    use_cache: bool,
) -> dict[str, Any]:
    if decode_steps <= 0:
        return {
            "median_decode_ms_total": 0.0,
            "median_decode_ms_per_token": 0.0,
            "decode_ms_total_values": [],
        }
    values = [
        time_decode_steps(qwen_model, lm_head, call, decode_steps=decode_steps, use_cache=use_cache)
        for _ in range(repeat)
    ]
    return {
        "median_decode_ms_total": median(values),
        "median_decode_ms_per_token": median([value / float(decode_steps) for value in values]),
        "decode_ms_total_values": values,
    }


def time_decode_steps(qwen_model, lm_head, call: dict[str, Any], *, decode_steps: int, use_cache: bool) -> float:
    import torch

    device = call["inputs_embeds"].device
    attention_dtype = call["attention_mask"].dtype if call["attention_mask"] is not None else torch.long
    cache_len = int(call["inputs_embeds"].shape[1])
    next_position = int(call["position_ids"].max().item()) + 1

    with torch.inference_mode():
        prefill = qwen_model.language_model(
            input_ids=None,
            position_ids=call["position_ids"],
            attention_mask=call["attention_mask"],
            past_key_values=None,
            inputs_embeds=call["inputs_embeds"],
            cache_position=call["cache_position"],
            visual_pos_masks=call["visual_pos_masks"],
            deepstack_visual_embeds=call["deepstack_visual_embeds"],
            use_cache=True,
        )
        next_token = lm_head(prefill.last_hidden_state[:, -1:, :]).argmax(dim=-1)
        past_key_values = prefill.past_key_values

    _sync_tensor_device(call["inputs_embeds"])
    start = time.perf_counter()
    for step in range(decode_steps):
        with torch.inference_mode():
            token_embeds = qwen_model.get_input_embeddings()(next_token)
            cache_position = torch.arange(cache_len + step, cache_len + step + 1, device=device)
            attention_mask = torch.ones((1, cache_len + step + 1), dtype=attention_dtype, device=device)
            position_ids = torch.full(
                (3, 1, 1),
                next_position + step,
                dtype=call["position_ids"].dtype,
                device=device,
            )
            outputs = qwen_model.language_model(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=token_embeds,
                cache_position=cache_position,
                visual_pos_masks=None,
                deepstack_visual_embeds=None,
                use_cache=use_cache,
            )
            next_token = lm_head(outputs.last_hidden_state[:, -1:, :]).argmax(dim=-1)
            past_key_values = outputs.past_key_values
    _sync_tensor_device(call["inputs_embeds"])
    return (time.perf_counter() - start) * 1000.0


def time_language(qwen_model, call: dict[str, Any], *, use_cache: bool) -> float:
    _sync_tensor_device(call["inputs_embeds"])
    start = time.perf_counter()
    outputs = run_language(qwen_model, call, use_cache=use_cache)
    del outputs
    _sync_tensor_device(call["inputs_embeds"])
    return (time.perf_counter() - start) * 1000.0


def summarize_rows(rows: list[dict[str, Any]], *, args: dict[str, Any]) -> dict[str, Any]:
    summary = {"args": args, "num_samples": len(rows)}
    numeric_keys = [
        "full_sequence_tokens",
        "pruned_sequence_tokens",
        "sequence_keep_ratio",
        "full_visual_tokens",
        "kept_visual_tokens",
        "visual_keep_ratio",
        "visual_removal_fraction",
        "ecr",
        "vision_ms",
        "full_language_ms",
        "pruned_language_ms",
        "language_speedup",
        "full_decode_ms_total",
        "pruned_decode_ms_total",
        "full_decode_ms_per_token",
        "pruned_decode_ms_per_token",
        "decode_speedup",
        "end_to_end_full_est_ms",
        "end_to_end_pruned_est_ms",
        "end_to_end_est_speedup",
        "generation_full_est_ms",
        "generation_pruned_est_ms",
        "generation_est_speedup",
    ]
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if key in row]
        summary[f"mean_{key}"] = mean(values)
        summary[f"median_{key}"] = median(values)
    summary["mean_language_ms_reduction_pct"] = pct_reduction(summary["mean_full_language_ms"], summary["mean_pruned_language_ms"])
    summary["mean_end_to_end_est_reduction_pct"] = pct_reduction(
        summary["mean_end_to_end_full_est_ms"],
        summary["mean_end_to_end_pruned_est_ms"],
    )
    return summary


def summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Qwen pruning efficiency benchmark",
        "",
        f"Samples: {summary['num_samples']}",
        "",
        "| metric | mean | median |",
        "|---|---:|---:|",
    ]
    for label, key in (
        ("Full sequence tokens", "full_sequence_tokens"),
        ("Pruned sequence tokens", "pruned_sequence_tokens"),
        ("Sequence keep ratio", "sequence_keep_ratio"),
        ("Full visual tokens", "full_visual_tokens"),
        ("Kept visual tokens", "kept_visual_tokens"),
        ("Visual keep ratio", "visual_keep_ratio"),
        ("ECR", "ecr"),
        ("Vision encoder ms", "vision_ms"),
        ("Full LLM prefill ms", "full_language_ms"),
        ("Pruned LLM prefill ms", "pruned_language_ms"),
        ("LLM prefill speedup", "language_speedup"),
        ("Full decode ms/token", "full_decode_ms_per_token"),
        ("Pruned decode ms/token", "pruned_decode_ms_per_token"),
        ("Decode speedup", "decode_speedup"),
        ("Full end-to-end estimated ms", "end_to_end_full_est_ms"),
        ("Pruned end-to-end estimated ms", "end_to_end_pruned_est_ms"),
        ("End-to-end estimated speedup", "end_to_end_est_speedup"),
        ("Full prefill+decode estimated ms", "generation_full_est_ms"),
        ("Pruned prefill+decode estimated ms", "generation_pruned_est_ms"),
        ("Prefill+decode estimated speedup", "generation_est_speedup"),
    ):
        lines.append(f"| {label} | {summary[f'mean_{key}']:.4f} | {summary[f'median_{key}']:.4f} |")
    lines.extend(
        [
            "",
            f"Mean LLM prefill time reduction: {summary['mean_language_ms_reduction_pct']:.2f}%",
            f"Mean estimated end-to-end time reduction: {summary['mean_end_to_end_est_reduction_pct']:.2f}%",
            "",
        ]
    )
    return "\n".join(lines)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_speedup(full_ms: float, pruned_ms: float) -> float:
    return full_ms / pruned_ms if pruned_ms > 0.0 else 0.0


def pct_reduction(full_ms: float, pruned_ms: float) -> float:
    return 100.0 * (full_ms - pruned_ms) / full_ms if full_ms > 0.0 else 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


if __name__ == "__main__":
    main()
