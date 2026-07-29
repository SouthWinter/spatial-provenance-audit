#!/usr/bin/env python
"""Batch LLM-prefill benchmark for Qwen visual-token pruning."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
from scripts.benchmark_qwen_prune_efficiency import build_inputs, load_budget_ratios, mean, median, safe_speedup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--is-probes", action="store_true", help="Treat --input as an already-built probe JSONL.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
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
    for start in tqdm(range(0, len(probes), args.batch_size), desc="Qwen batch prune prefill"):
        batch_probes = probes[start : start + args.batch_size]
        if len(batch_probes) < args.batch_size:
            continue
        calls = [
            prepare_probe_calls(
                probe,
                model=model,
                processor=processor,
                process_vision_info=process_vision_info,
                input_device=input_device,
                system_prompt=args.system_prompt,
                min_pixels=args.min_pixels,
                max_pixels=args.max_pixels,
                prune_config=prune_config,
                sample_budget_ratio=budget_ratios.get(str(probe.get("sample_id", probe.get("id", "")))),
            )
            for probe in batch_probes
        ]
        full_batch = collate_calls([call["full_call"] for call in calls])
        pruned_batch = collate_calls([call["pruned_call"] for call in calls])
        for _ in range(max(0, args.warmup)):
            run_language_batch(model.model, full_batch, use_cache=args.use_cache)
            run_language_batch(model.model, pruned_batch, use_cache=args.use_cache)
        full_measurements = [
            time_language_batch_with_memory(model.model, full_batch, use_cache=args.use_cache) for _ in range(args.repeat)
        ]
        pruned_measurements = [
            time_language_batch_with_memory(model.model, pruned_batch, use_cache=args.use_cache) for _ in range(args.repeat)
        ]
        full_ms = [float(measurement["ms"]) for measurement in full_measurements]
        pruned_ms = [float(measurement["ms"]) for measurement in pruned_measurements]
        full_median = median(full_ms)
        pruned_median = median(pruned_ms)
        full_samples_per_s = args.batch_size * 1000.0 / full_median if full_median > 0.0 else 0.0
        pruned_samples_per_s = args.batch_size * 1000.0 / pruned_median if pruned_median > 0.0 else 0.0
        full_tokens_per_s = full_batch["token_count"] * 1000.0 / full_median if full_median > 0.0 else 0.0
        pruned_tokens_per_s = pruned_batch["token_count"] * 1000.0 / pruned_median if pruned_median > 0.0 else 0.0
        evidence_score_ms = sum(float(call["evidence_score_ms"]) for call in calls)
        selector_ms = sum(float(call["selector_ms"]) for call in calls)
        prune_materialize_ms = sum(float(call["prune_materialize_ms"]) for call in calls)
        prune_overhead_ms = evidence_score_ms + selector_ms + prune_materialize_ms
        saved_prefill_ms = max(0.0, full_median - pruned_median)
        rows.append(
            {
                "batch_index": start // args.batch_size,
                "batch_size": args.batch_size,
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
                "full_visual_tokens": sum(call["full_visual_tokens"] for call in calls),
                "kept_visual_tokens": sum(call["kept_visual_tokens"] for call in calls),
                "visual_keep_ratio": sum(call["kept_visual_tokens"] for call in calls)
                / float(sum(call["full_visual_tokens"] for call in calls)),
                "mean_ecr": mean([call["ecr"] for call in calls]),
                "prepare_ms": sum(float(call["prepare_ms"]) for call in calls),
                "build_inputs_ms": sum(float(call["build_inputs_ms"]) for call in calls),
                "vision_feature_ms": sum(float(call["vision_feature_ms"]) for call in calls),
                "evidence_score_ms": evidence_score_ms,
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
        f"{summary['mean_speedup']:.3f}x keep={summary['mean_visual_keep_ratio']:.4f}"
    )


def prepare_probe_calls(
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
) -> dict[str, Any]:
    import torch

    prepare_start = time.perf_counter()
    build_start = time.perf_counter()
    qwen_model = model.model
    inputs = build_inputs(
        probe,
        processor=processor,
        process_vision_info=process_vision_info,
        input_device=input_device,
        system_prompt=system_prompt,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        text_repeat=0,
    )
    _sync_tensor_device(inputs["input_ids"])
    build_inputs_ms = (time.perf_counter() - build_start) * 1000.0
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    pixel_values = inputs["pixel_values"]
    image_grid_thw = inputs["image_grid_thw"]

    vision_start = time.perf_counter()
    with torch.inference_mode():
        inputs_embeds = qwen_model.get_input_embeddings()(input_ids)
        image_embeds_list, deepstack_image_embeds = qwen_model.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds_list, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = qwen_model.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
    _sync_tensor_device(input_ids)
    vision_feature_ms = (time.perf_counter() - vision_start) * 1000.0

    evidence_start = time.perf_counter()
    image_positions = torch.argwhere((input_ids[0] == model.config.image_token_id)).squeeze(1)
    full_visual_tokens = int(image_positions.numel())
    token_grid_h, token_grid_w = _token_grid_shape(image_grid_thw[0], qwen_model.config.vision_config.spatial_merge_size)
    token_boxes = make_token_grid(token_grid_h, token_grid_w)
    if len(token_boxes) != full_visual_tokens:
        raise ValueError(f"Token grid has {len(token_boxes)} boxes but prompt has {full_visual_tokens} image tokens.")
    evidence_regions = evidence_regions_from_sample(probe)
    relevance = _evidence_relevance(token_boxes, evidence_regions)
    uniqueness = _spatial_uniqueness(token_boxes, relevance)
    evidence_score_ms = (time.perf_counter() - evidence_start) * 1000.0

    selector_start = time.perf_counter()
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
    selector_ms = (time.perf_counter() - selector_start) * 1000.0

    prune_start = time.perf_counter()
    position_ids, _ = qwen_model.get_rope_index(input_ids, image_grid_thw, attention_mask=attention_mask)
    full_call = {
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "inputs_embeds": inputs_embeds,
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
        "visual_pos_masks": (input_ids == model.config.image_token_id)[:, keep_sequence],
        "deepstack_visual_embeds": [x[kept_tensor.to(x.device)].to(pruned_inputs_embeds.device) for x in deepstack_image_embeds],
    }
    _sync_tensor_device(input_ids)
    prune_materialize_ms = (time.perf_counter() - prune_start) * 1000.0
    return {
        "full_call": full_call,
        "pruned_call": pruned_call,
        "full_visual_tokens": full_visual_tokens,
        "kept_visual_tokens": len(kept_indices),
        "visual_removal_fraction": removal_fraction(full_visual_tokens, len(kept_indices)),
        "ecr": evidence_coverage(kept_indices, token_boxes, evidence_regions),
        "prepare_ms": (time.perf_counter() - prepare_start) * 1000.0,
        "build_inputs_ms": build_inputs_ms,
        "vision_feature_ms": vision_feature_ms,
        "evidence_score_ms": evidence_score_ms,
        "selector_ms": selector_ms,
        "prune_materialize_ms": prune_materialize_ms,
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
    position_ids = torch.ones((3, batch_size, max_len), dtype=calls[0]["position_ids"].dtype, device=device)
    visual_pos_masks = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
    token_count = 0
    for idx, call in enumerate(calls):
        length = int(call["inputs_embeds"].shape[1])
        embeds[idx, :length, :] = call["inputs_embeds"][0]
        attention_mask[idx, :length] = 1
        position_ids[:, idx : idx + 1, :length] = call["position_ids"]
        visual_pos_masks[idx, :length] = call["visual_pos_masks"][0]
        token_count += length
    deepstack_visual_embeds = []
    for layer_idx in range(len(calls[0]["deepstack_visual_embeds"])):
        deepstack_visual_embeds.append(torch.cat([call["deepstack_visual_embeds"][layer_idx] for call in calls], dim=0))
    return {
        "inputs_embeds": embeds,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "visual_pos_masks": visual_pos_masks,
        "deepstack_visual_embeds": deepstack_visual_embeds,
        "cache_position": torch.arange(max_len, device=device),
        "token_count": token_count,
    }


def run_language_batch(qwen_model, batch: dict[str, Any], *, use_cache: bool):
    import torch

    with torch.inference_mode():
        outputs = qwen_model.language_model(
            input_ids=None,
            position_ids=batch["position_ids"],
            attention_mask=batch["attention_mask"],
            past_key_values=None,
            inputs_embeds=batch["inputs_embeds"],
            cache_position=batch["cache_position"],
            visual_pos_masks=batch["visual_pos_masks"],
            deepstack_visual_embeds=batch["deepstack_visual_embeds"],
            use_cache=use_cache,
        )
    _sync_tensor_device(batch["inputs_embeds"])
    return outputs


def time_language_batch(qwen_model, batch: dict[str, Any], *, use_cache: bool) -> float:
    _sync_tensor_device(batch["inputs_embeds"])
    start = time.perf_counter()
    outputs = run_language_batch(qwen_model, batch, use_cache=use_cache)
    del outputs
    _sync_tensor_device(batch["inputs_embeds"])
    return (time.perf_counter() - start) * 1000.0


def time_language_batch_with_memory(qwen_model, batch: dict[str, Any], *, use_cache: bool) -> dict[str, float]:
    import torch

    _sync_tensor_device(batch["inputs_embeds"])
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
    outputs = run_language_batch(qwen_model, batch, use_cache=use_cache)
    del outputs
    _sync_tensor_device(batch["inputs_embeds"])
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
        "evidence_score_ms",
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
    out["mean_time_reduction_pct"] = 100.0 * (out["mean_full_ms"] - out["mean_pruned_ms"]) / out["mean_full_ms"]
    out["mean_peak_allocated_reduction_pct"] = 100.0 * (
        out["mean_full_peak_allocated_gb"] - out["mean_pruned_peak_allocated_gb"]
    ) / out["mean_full_peak_allocated_gb"] if out["mean_full_peak_allocated_gb"] > 0.0 else 0.0
    out["mean_incremental_peak_allocated_reduction_pct"] = 100.0 * (
        out["mean_full_incremental_peak_allocated_gb"] - out["mean_pruned_incremental_peak_allocated_gb"]
    ) / out["mean_full_incremental_peak_allocated_gb"] if out["mean_full_incremental_peak_allocated_gb"] > 0.0 else 0.0
    return out


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Qwen batch prefill pruning benchmark",
        "",
        f"Samples: {summary['num_samples']}",
        f"Batch size: {summary['args']['batch_size']}",
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
        ("Evidence score ms/batch", "evidence_score_ms"),
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


if __name__ == "__main__":
    main()
