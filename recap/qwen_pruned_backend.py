"""Direct Qwen-VL scorer with visual-token pruning before LLM prefill."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from PIL import Image
from tqdm import tqdm

from recap.images import encode_image_data_url, probe_to_visual, validate_probe_images
from recap.prune.budgets import fixed_keep_count, removal_fraction, risk_adaptive_keep_ratio, risk_bucket_keep_ratio
from recap.prune.metrics import (
    Box,
    evidence_center_recall,
    evidence_coverage,
    evidence_patch_recall,
    evidence_regions_from_sample,
    exhaustive_merge_source_indices,
    intersection_area,
    make_token_grid,
)
from recap.prune.selectors import select_indices
from recap.prune.saturation import SaturationConfig, evidence_saturation_decision
from recap.qwen_direct_backend import _load_qwen_direct
from recap.scoring import score_probe


@dataclass(frozen=True)
class PruneConfig:
    selector: str
    keep_ratio: float
    budget_mode: str = "fixed"
    rho_min: float = 0.15
    rho_max: float = 0.70
    seed: int = 13
    hybrid_core_ratio: float = 0.50
    hybrid_context_ratio: float = 0.25
    evidence_boost: float = 0.10
    embedding_relevance_weight: float = 0.85
    embedding_query_topk: int = 2
    saturation_temperature: float = 0.12
    saturation_mass_target: float = 0.72
    saturation_cell_target: float = 0.75

    def __post_init__(self) -> None:
        if not 0.0 <= self.embedding_relevance_weight <= 1.0:
            raise ValueError("embedding_relevance_weight must be in [0, 1].")
        if self.embedding_query_topk < 1:
            raise ValueError("embedding_query_topk must be positive.")


def score_probes_with_qwen_pruned(
    probes: list[dict[str, Any]],
    *,
    pretrained: str,
    prune_config: PruneConfig,
    risk_scores: dict[str, float] | None = None,
    budget_ratios: dict[str, float] | None = None,
    kept_indices_by_sample: dict[str, list[int]] | None = None,
    device: str = "cuda",
    device_map: str = "auto",
    dtype: str = "auto",
    attn_implementation: str | None = None,
    min_pixels: int = 50176,
    max_pixels: int = 50176,
    use_fast_processor: bool = False,
    system_prompt: str = "You are a helpful assistant.",
    target_delimiter: str = " ",
    debug_forward: bool = False,
    strict_images: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Score yes/no probes after pruning Qwen visual placeholder tokens."""
    import torch

    image_report = validate_probe_images(probes)
    if strict_images and image_report["missing_visual_count"]:
        raise FileNotFoundError(f"Missing images for RECAP visual probes: {image_report}")
    if image_report["missing_visual_count"]:
        print(f"[RECAP-QwenPruned] WARNING missing images for visual probes: {image_report}", flush=True)

    model, processor, _, process_vision_info, input_device = _load_qwen_direct(
        pretrained=pretrained,
        device=device,
        device_map=device_map,
        dtype=dtype,
        attn_implementation=attn_implementation,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        use_fast_processor=use_fast_processor,
        torch_module=torch,
    )

    scored: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for probe in tqdm(probes, desc="RECAP-QwenPruned probes"):
        sample_id = str(probe.get("sample_id", probe.get("id", "")))
        sample_risk = risk_scores.get(sample_id, 0.5) if risk_scores else None
        sample_budget_ratio = budget_ratios.get(sample_id) if budget_ratios else None
        sample_kept_indices = kept_indices_by_sample.get(sample_id) if kept_indices_by_sample else None
        yes_loss, yes_greedy, yes_trace = _score_pruned_continuation(
            probe,
            f"{target_delimiter}yes",
            model=model,
            processor=processor,
            process_vision_info=process_vision_info,
            input_device=input_device,
            system_prompt=system_prompt,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            prune_config=prune_config,
            sample_risk=sample_risk,
            sample_budget_ratio=sample_budget_ratio,
            sample_kept_indices=sample_kept_indices,
            debug_forward=debug_forward,
            strict_images=strict_images,
        )
        no_loss, no_greedy, no_trace = _score_pruned_continuation(
            probe,
            f"{target_delimiter}no",
            model=model,
            processor=processor,
            process_vision_info=process_vision_info,
            input_device=input_device,
            system_prompt=system_prompt,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            prune_config=prune_config,
            sample_risk=sample_risk,
            sample_budget_ratio=sample_budget_ratio,
            sample_kept_indices=sample_kept_indices,
            debug_forward=debug_forward,
            strict_images=strict_images,
        )
        record = score_probe(probe, yes_loss=yes_loss, no_loss=no_loss)
        record["yes_is_greedy"] = bool(yes_greedy)
        record["no_is_greedy"] = bool(no_greedy)
        record["model"] = "qwen_vl_pruned"
        record["pretrained"] = pretrained
        record.update(_score_prune_fields(yes_trace))
        scored.append(record)
        traces.append(_merge_trace_pair(probe, yes_trace, no_trace))

    return scored, traces


def summarize_prune_traces(traces: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate pruning traces into a compact metrics payload."""
    if not traces:
        return {
            "num_pruned_probes": 0.0,
            "mean_full_sequence_tokens": 0.0,
            "mean_pruned_sequence_tokens": 0.0,
            "mean_full_visual_tokens": 0.0,
            "mean_kept_visual_tokens": 0.0,
            "mean_keep_ratio": 0.0,
            "mean_removal_fraction": 0.0,
            "mean_ecr": 0.0,
            "mean_anchor_ecr": 0.0,
            "mean_evidence_center_recall": 0.0,
            "mean_evidence_patch_recall": 0.0,
            "mean_vision_ms": 0.0,
            "mean_target_text_ms": 0.0,
            "mean_score_compute_ms": 0.0,
            "mean_selector_ms": 0.0,
            "mean_prune_materialize_ms": 0.0,
            "mean_prune_overhead_ms": 0.0,
            "mean_language_ms": 0.0,
            "mean_forward_ms": 0.0,
        }
    full = [float(t.get("full_visual_tokens", 0.0)) for t in traces]
    kept = [float(t.get("kept_visual_tokens", 0.0)) for t in traces]
    ratios = [k / n for k, n in zip(kept, full) if n > 0.0]
    return {
        "num_pruned_probes": float(len(traces)),
        "mean_full_sequence_tokens": _mean([float(t.get("full_sequence_tokens", 0.0)) for t in traces]),
        "mean_pruned_sequence_tokens": _mean([float(t.get("pruned_sequence_tokens", 0.0)) for t in traces]),
        "mean_full_visual_tokens": _mean(full),
        "mean_kept_visual_tokens": _mean(kept),
        "mean_keep_ratio": _mean(ratios),
        "mean_removal_fraction": _mean([float(t.get("removal_fraction", 0.0)) for t in traces]),
        "mean_ecr": _mean([float(t.get("ecr", 0.0)) for t in traces if t.get("has_evidence")]),
        "mean_anchor_ecr": _mean(
            [float(t.get("anchor_ecr", t.get("ecr", 0.0))) for t in traces if t.get("has_evidence")]
        ),
        "mean_evidence_center_recall": _mean(
            [float(t.get("evidence_center_recall", 0.0)) for t in traces if t.get("has_evidence")]
        ),
        "mean_evidence_patch_recall": _mean(
            [float(t.get("evidence_patch_recall", 0.0)) for t in traces if t.get("has_evidence")]
        ),
        "mean_vision_ms": _mean([float(t.get("mean_vision_ms", t.get("vision_ms", 0.0))) for t in traces]),
        "mean_target_text_ms": _mean(
            [float(t.get("mean_target_text_ms", t.get("target_text_ms", 0.0))) for t in traces]
        ),
        "mean_score_compute_ms": _mean(
            [float(t.get("mean_score_compute_ms", t.get("score_compute_ms", 0.0))) for t in traces]
        ),
        "mean_selector_ms": _mean([float(t.get("mean_selector_ms", t.get("selector_ms", 0.0))) for t in traces]),
        "mean_prune_materialize_ms": _mean(
            [float(t.get("mean_prune_materialize_ms", t.get("prune_materialize_ms", 0.0))) for t in traces]
        ),
        "mean_prune_overhead_ms": _mean(
            [float(t.get("mean_prune_overhead_ms", t.get("prune_overhead_ms", 0.0))) for t in traces]
        ),
        "mean_language_ms": _mean([float(t.get("mean_language_ms", t.get("language_ms", 0.0))) for t in traces]),
        "mean_forward_ms": _mean([float(t.get("mean_forward_ms", 0.0)) for t in traces]),
    }


def _score_pruned_continuation(
    probe: dict[str, Any],
    continuation: str,
    *,
    model,
    processor,
    process_vision_info,
    input_device,
    system_prompt: str,
    min_pixels: int,
    max_pixels: int,
    prune_config: PruneConfig,
    sample_risk: float | None,
    sample_budget_ratio: float | None,
    sample_kept_indices: list[int] | None,
    debug_forward: bool,
    strict_images: bool,
) -> tuple[float, bool, dict[str, Any]]:
    import torch

    context = str(probe["question"]).replace("<image>", "")
    visuals = probe_to_visual(probe, strict=strict_images)
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

    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]
    full_messages = prompt_messages + [{"role": "assistant", "content": continuation}]

    if debug_forward:
        print(
            "[RECAP-QwenPruned] scoring "
            f"{probe.get('sample_id')}::{probe.get('probe')} continuation={continuation!r}",
            flush=True,
        )
    prompt_text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    full_text = processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)

    image_inputs, video_inputs = process_vision_info(full_messages)
    prompt_inputs = processor(text=prompt_text, images=image_inputs, videos=video_inputs, return_tensors="pt")
    inputs = processor(text=full_text, images=image_inputs, videos=video_inputs, return_tensors="pt")
    prompt_inputs = prompt_inputs.to(input_device)
    inputs = inputs.to(input_device)

    labels = inputs["input_ids"].clone()
    prompt_len = prompt_inputs["input_ids"].shape[1]
    labels[:, :prompt_len] = -100

    if "pixel_values" not in inputs or inputs.get("pixel_values") is None:
        start = time.perf_counter()
        with torch.inference_mode():
            outputs = model(**inputs, labels=labels)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        trace = _empty_trace(probe, prune_config, elapsed_ms=elapsed_ms)
        return float(outputs["loss"].item()), _greedy_from_labels(outputs["logits"], labels), trace

    start = time.perf_counter()
    with torch.inference_mode():
        logits, pruned_labels, trace = _forward_pruned_qwen(
            model=model,
            processor=processor,
            inputs=inputs,
            labels=labels,
            probe=probe,
            prune_config=prune_config,
            sample_risk=sample_risk,
            sample_budget_ratio=sample_budget_ratio,
            sample_kept_indices=sample_kept_indices,
        )
        loss = model.loss_function(
            logits=logits,
            labels=pruned_labels,
            vocab_size=model.config.text_config.vocab_size,
        )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    trace["forward_ms"] = elapsed_ms
    return float(loss.item()), _greedy_from_labels(logits, pruned_labels), trace


def _forward_pruned_qwen(
    *,
    model,
    processor,
    inputs,
    labels,
    probe: dict[str, Any],
    prune_config: PruneConfig,
    sample_risk: float | None,
    sample_budget_ratio: float | None,
    sample_kept_indices: list[int] | None,
):
    import torch

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    pixel_values = inputs["pixel_values"]
    image_grid_thw = inputs["image_grid_thw"]
    if input_ids.shape[0] != 1:
        raise ValueError("Qwen pruned scorer currently expects batch_size=1.")
    if image_grid_thw.shape[0] != 1:
        raise ValueError("Qwen pruned scorer currently expects one image per probe.")

    qwen_model = model.model
    _sync_tensor_device(input_ids)
    vision_start = time.perf_counter()
    inputs_embeds = qwen_model.get_input_embeddings()(input_ids)
    is_visionzip = _is_visionzip_selector(prune_config.selector)
    if is_visionzip:
        image_embeds, deepstack_image_embeds, visionzip_scores, visionzip_metric = _qwen3_visual_features_with_attention(
            qwen_model=qwen_model,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
    else:
        image_embeds_list, deepstack_image_embeds = qwen_model.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds_list, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        visionzip_scores = None
        visionzip_metric = None
    image_mask, _ = qwen_model.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
    _sync_tensor_device(input_ids)
    vision_ms = (time.perf_counter() - vision_start) * 1000.0

    image_positions = torch.argwhere((input_ids[0] == model.config.image_token_id)).squeeze(1)
    num_visual_tokens = int(image_positions.numel())
    token_grid_h, token_grid_w = _token_grid_shape(image_grid_thw[0], qwen_model.config.vision_config.spatial_merge_size)
    token_boxes = make_token_grid(token_grid_h, token_grid_w)
    if len(token_boxes) != num_visual_tokens:
        raise ValueError(f"Token grid has {len(token_boxes)} boxes but prompt has {num_visual_tokens} image tokens.")

    evidence_regions = evidence_regions_from_sample(probe)
    selector_impl, score_source = _selector_impl(prune_config.selector)
    target_text_token_count = 0
    target_text_ms = 0.0
    score_compute_start = time.perf_counter()
    if sample_kept_indices is not None:
        selector_impl = "provided_indices"
        score_source = "provided_indices"
        relevance = [0.0 for _ in token_boxes]
        uniqueness = [0.0 for _ in token_boxes]
    elif is_visionzip:
        selector_impl = "visionzip"
        score_source = "qwen3_vision_attention"
        relevance = visionzip_scores.detach().cpu().float().tolist()
        uniqueness = [0.0 for _ in token_boxes]
    elif score_source in {"embedding", "target_embedding"}:
        text_positions_override = None
        if score_source == "target_embedding":
            target_text_start = time.perf_counter()
            text_positions_override = _target_text_positions(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                tokenizer=getattr(processor, "tokenizer", processor),
                probe=probe,
            )
            _sync_tensor_device(input_ids)
            target_text_ms = (time.perf_counter() - target_text_start) * 1000.0
            target_text_token_count = int(text_positions_override.numel())
        relevance, uniqueness = _embedding_relevance_and_uniqueness(
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            image_positions=image_positions,
            image_token_id=model.config.image_token_id,
            token_grid_h=token_grid_h,
            token_grid_w=token_grid_w,
            text_positions_override=text_positions_override,
            relevance_weight=prune_config.embedding_relevance_weight,
            query_topk=prune_config.embedding_query_topk,
        )
    else:
        relevance = _evidence_relevance(token_boxes, evidence_regions)
        uniqueness = _spatial_uniqueness(token_boxes, relevance)
    _sync_tensor_device(input_ids)
    score_compute_ms = (time.perf_counter() - score_compute_start) * 1000.0
    selector_start = time.perf_counter()
    effective_keep_ratio = _effective_keep_ratio(prune_config, sample_risk, sample_budget_ratio)
    saturation_info: dict[str, Any] = {}
    if prune_config.budget_mode == "evidence_saturation":
        if selector_impl != "grid_topk":
            raise ValueError("evidence_saturation currently requires a grid_topk selector")
        effective_keep_ratio, saturation_info = evidence_saturation_decision(
            relevance,
            token_boxes,
            config=SaturationConfig(
                candidate_ratios=(prune_config.rho_min, prune_config.keep_ratio, prune_config.rho_max),
                temperature=prune_config.saturation_temperature,
                mass_target=prune_config.saturation_mass_target,
                cell_target=prune_config.saturation_cell_target,
                grid_ratio=prune_config.hybrid_core_ratio,
            ),
        )
    visionzip_info: dict[str, Any] = {}
    visionzip_features = None
    visionzip_deepstack = None
    if sample_kept_indices is not None:
        kept_indices = _sanitize_kept_indices(sample_kept_indices, num_visual_tokens)
        keep_count = len(kept_indices)
        effective_keep_ratio = keep_count / float(num_visual_tokens) if num_visual_tokens else 0.0
    elif is_visionzip:
        keep_count = fixed_keep_count(num_visual_tokens, effective_keep_ratio)
        kept_indices, visionzip_features, visionzip_deepstack, visionzip_info = _qwen3_visionzip_select_and_merge(
            image_embeds=image_embeds,
            deepstack_image_embeds=deepstack_image_embeds,
            salience=visionzip_scores,
            metric=visionzip_metric,
            keep_count=keep_count,
        )
    else:
        keep_count = fixed_keep_count(num_visual_tokens, effective_keep_ratio)
        kept_indices = select_indices(
            selector_impl,
            num_tokens=num_visual_tokens,
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
            evidence_boost=prune_config.evidence_boost,
        )
    _sync_tensor_device(input_ids)
    selector_ms = (time.perf_counter() - selector_start) * 1000.0

    prune_materialize_start = time.perf_counter()
    keep_sequence = torch.ones(input_ids.shape[1], dtype=torch.bool, device=input_ids.device)
    drop_positions = image_positions.to(input_ids.device)
    keep_sequence[drop_positions] = False
    kept_tensor = torch.tensor(kept_indices, dtype=torch.long, device=input_ids.device)
    keep_sequence[drop_positions[kept_tensor]] = True

    pruned_inputs_embeds = inputs_embeds[:, keep_sequence, :]
    pruned_attention_mask = attention_mask[:, keep_sequence] if attention_mask is not None else None
    pruned_labels = labels[:, keep_sequence]
    if visionzip_features is not None:
        pruned_image_mask = (input_ids == model.config.image_token_id)[:, keep_sequence]
        pruned_inputs_embeds = pruned_inputs_embeds.clone()
        pruned_inputs_embeds[pruned_image_mask] = visionzip_features.to(
            pruned_inputs_embeds.device, pruned_inputs_embeds.dtype
        )
    position_ids, _ = qwen_model.get_rope_index(input_ids, image_grid_thw, attention_mask=attention_mask)
    pruned_position_ids = position_ids[:, :, keep_sequence]
    visual_pos_masks = (input_ids == model.config.image_token_id)[:, keep_sequence]

    if visionzip_deepstack is not None:
        pruned_deepstack = [layer_embeds.to(pruned_inputs_embeds.device) for layer_embeds in visionzip_deepstack]
    else:
        pruned_deepstack = []
        for layer_embeds in deepstack_image_embeds:
            pruned_deepstack.append(layer_embeds[kept_tensor.to(layer_embeds.device)].to(pruned_inputs_embeds.device))

    cache_position = torch.arange(pruned_inputs_embeds.shape[1], device=pruned_inputs_embeds.device)
    _sync_tensor_device(pruned_inputs_embeds)
    prune_materialize_ms = (time.perf_counter() - prune_materialize_start) * 1000.0
    _sync_tensor_device(pruned_inputs_embeds)
    language_start = time.perf_counter()
    outputs = qwen_model.language_model(
        input_ids=None,
        position_ids=pruned_position_ids,
        attention_mask=pruned_attention_mask,
        past_key_values=None,
        inputs_embeds=pruned_inputs_embeds,
        cache_position=cache_position,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=pruned_deepstack,
    )
    logits = model.lm_head(outputs.last_hidden_state)
    _sync_tensor_device(pruned_inputs_embeds)
    language_ms = (time.perf_counter() - language_start) * 1000.0
    provenance_indices = (
        exhaustive_merge_source_indices(
            num_visual_tokens,
            kept_indices,
            contextual_tokens=int(visionzip_info.get("contextual_tokens", 0)),
        )
        if is_visionzip
        else kept_indices
    )
    coverage = evidence_coverage(provenance_indices, token_boxes, evidence_regions)
    center_recall = evidence_center_recall(provenance_indices, token_boxes, evidence_regions)
    patch_recall = evidence_patch_recall(provenance_indices, token_boxes, evidence_regions)
    anchor_coverage = evidence_coverage(kept_indices, token_boxes, evidence_regions)
    trace = {
        "sample_id": str(probe.get("sample_id", probe.get("id", ""))),
        "probe": str(probe.get("probe", "")),
        "selector": prune_config.selector,
        "selector_impl": selector_impl,
        "score_source": score_source,
        "target_text_token_count": target_text_token_count,
        "evidence_boost": float(prune_config.evidence_boost),
        "embedding_relevance_weight": float(prune_config.embedding_relevance_weight),
        "embedding_query_topk": int(prune_config.embedding_query_topk),
        "budget_mode": prune_config.budget_mode,
        "target_keep_ratio": float(prune_config.keep_ratio),
        "effective_keep_ratio": float(effective_keep_ratio),
        "budget_bucket": _budget_bucket_label(prune_config, sample_risk),
        "saturation_score_entropy": saturation_info.get("score_entropy"),
        "saturation_spatial_entropy": saturation_info.get("spatial_entropy"),
        "saturation_spatial_dispersion": saturation_info.get("spatial_dispersion"),
        "saturation_active_cell_count": saturation_info.get("active_cell_count"),
        "saturation_candidate_diagnostics": saturation_info.get("candidate_diagnostics", []),
        "risk": None if sample_risk is None else float(sample_risk),
        "sample_budget_ratio": None if sample_budget_ratio is None else float(sample_budget_ratio),
        "provided_indices": bool(sample_kept_indices is not None),
        "full_sequence_tokens": int(input_ids.shape[1]),
        "pruned_sequence_tokens": int(pruned_inputs_embeds.shape[1]),
        "full_visual_tokens": num_visual_tokens,
        "kept_visual_tokens": len(kept_indices),
        "removal_fraction": removal_fraction(num_visual_tokens, len(kept_indices)),
        "kept_indices": kept_indices,
        "provenance_indices": provenance_indices,
        "provenance_semantics": "source_lineage_union" if is_visionzip else "retained_token_origins",
        "has_evidence": bool(evidence_regions),
        "evidence_region_count": len(evidence_regions),
        "ecr": coverage,
        "anchor_ecr": anchor_coverage,
        "ecr_0_5": 1.0 if coverage >= 0.5 else 0.0,
        "evidence_center_recall": center_recall,
        "evidence_patch_recall": patch_recall,
        "grid_t": int(image_grid_thw[0][0].item()),
        "grid_h": int(image_grid_thw[0][1].item()),
        "grid_w": int(image_grid_thw[0][2].item()),
        "vision_ms": vision_ms,
        "target_text_ms": target_text_ms,
        "score_compute_ms": score_compute_ms,
        "selector_ms": selector_ms,
        "prune_materialize_ms": prune_materialize_ms,
        "prune_overhead_ms": score_compute_ms + selector_ms + prune_materialize_ms,
        "language_ms": language_ms,
        "visionzip_commit": "JIA-Lab-research/VisionZip@8f86b55",
        "visionzip_dominant_tokens": int(visionzip_info.get("dominant_tokens", 0)),
        "visionzip_contextual_tokens": int(visionzip_info.get("contextual_tokens", 0)),
        "visionzip_passthrough": bool(visionzip_info.get("passthrough", False)),
    }
    return logits, pruned_labels, trace


def _is_visionzip_selector(selector: str) -> bool:
    return selector.strip().lower() in {"visionzip", "official_visionzip", "qwen3_visionzip"}


def _qwen3_visual_features_with_attention(*, qwen_model, pixel_values, image_grid_thw):
    """Run Qwen3 visual tower while exposing last-layer token salience.

    The code mirrors Transformers' Qwen3VLVisionModel.forward. It captures
    attention in the final visual block and then applies the original merger and
    deepstack mergers. For strict keep=1 equivalence, load Qwen3 with eager
    attention so this helper and the direct path use the same attention kernel.
    """
    import torch
    import torch.nn.functional as F

    visual = qwen_model.visual
    hidden_states = visual.patch_embed(pixel_values.type(visual.dtype))
    hidden_states = hidden_states + visual.fast_pos_embed_interpolate(image_grid_thw)
    rotary_pos_emb = visual.rot_pos_emb(image_grid_thw)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0]).cumsum(
        dim=0,
        dtype=image_grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    deepstack_feature_lists = []
    salience = None
    metric = None
    last_layer = len(visual.blocks) - 1
    for layer_num, block in enumerate(visual.blocks):
        if layer_num == last_layer:
            hidden_states, attn_weights, key_metric = _qwen3_vision_block_with_attention(
                block=block,
                hidden_states=hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
            salience, metric = _qwen3_merge_attention_sources(
                attn_weights=attn_weights,
                key_metric=key_metric,
                spatial_merge_unit=visual.spatial_merge_size**2,
            )
        else:
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
        if layer_num in visual.deepstack_visual_indexes:
            deepstack_feature = visual.deepstack_merger_list[visual.deepstack_visual_indexes.index(layer_num)](
                hidden_states
            )
            deepstack_feature_lists.append(deepstack_feature)

    image_embeds = visual.merger(hidden_states)
    if salience is None or metric is None:
        raise RuntimeError("Qwen3 VisionZip failed to capture visual attention from the final vision block.")
    if int(salience.numel()) != int(image_embeds.shape[0]):
        raise ValueError(
            f"Qwen3 VisionZip salience has {int(salience.numel())} tokens but merged image embeds have "
            f"{int(image_embeds.shape[0])}."
        )
    return image_embeds, deepstack_feature_lists, salience, metric


def _qwen3_vision_block_with_attention(*, block, hidden_states, cu_seqlens, position_embeddings):
    """Forward one Qwen3 vision block and return merged hidden state plus attention."""
    import torch
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision, eager_attention_forward

    normed = block.norm1(hidden_states)
    attn_module = block.attn
    seq_length = normed.shape[0]
    query_states, key_states, value_states = (
        attn_module.qkv(normed).reshape(seq_length, 3, attn_module.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    )
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)
    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    value_states = value_states.transpose(0, 1).unsqueeze(0)

    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    q_splits, k_splits, v_splits = [
        torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
    ]
    outputs = []
    attn_chunks = []
    key_chunks = []
    for q, k, v in zip(q_splits, k_splits, v_splits):
        attn_output, attn_weights = eager_attention_forward(
            attn_module,
            q,
            k,
            v,
            attention_mask=None,
            scaling=attn_module.scaling,
            dropout=0.0 if not block.training else attn_module.attention_dropout,
            is_causal=False,
        )
        outputs.append(attn_output)
        attn_chunks.append(attn_weights)
        key_chunks.append(k)
    attn_output = torch.cat(outputs, dim=1).reshape(seq_length, -1).contiguous()
    attn_output = attn_module.proj(attn_output)
    hidden_states = hidden_states + attn_output
    hidden_states = hidden_states + block.mlp(block.norm2(hidden_states))

    if len(attn_chunks) != 1:
        raise ValueError("Qwen3 VisionZip currently expects one image/video chunk per probe.")
    attn_weights = attn_chunks[0]
    key_metric = key_chunks[0]
    return hidden_states, attn_weights, key_metric


def _qwen3_merge_attention_sources(*, attn_weights, key_metric, spatial_merge_unit: int):
    import torch

    if attn_weights.ndim != 4 or key_metric.ndim != 4:
        raise ValueError("Unexpected Qwen3 attention tensor shape for VisionZip.")
    raw_count = int(attn_weights.shape[-1])
    if raw_count % int(spatial_merge_unit) != 0:
        raise ValueError(f"Raw visual token count {raw_count} is not divisible by merge unit {spatial_merge_unit}.")

    received = attn_weights[0].mean(dim=0).sum(dim=0).float()
    salience = received.reshape(raw_count // spatial_merge_unit, spatial_merge_unit).mean(dim=1)
    salience = _minmax_tensor(salience)

    metric = key_metric[0].mean(dim=0).float()
    metric = metric.reshape(raw_count // spatial_merge_unit, spatial_merge_unit, metric.shape[-1]).mean(dim=1)
    metric = metric.unsqueeze(0)
    return salience, metric


def _qwen3_visionzip_select_and_merge(*, image_embeds, deepstack_image_embeds, salience, metric, keep_count: int):
    import torch

    num_tokens = int(image_embeds.shape[0])
    keep_count = max(1, min(int(keep_count), num_tokens))
    if keep_count >= num_tokens:
        return (
            list(range(num_tokens)),
            image_embeds,
            list(deepstack_image_embeds),
            {"dominant_tokens": num_tokens, "contextual_tokens": 0, "passthrough": True},
        )

    selector_start = time.perf_counter()
    dominant_count, contextual_count = _qwen3_visionzip_budget_split(num_tokens, keep_count)
    dominant_indices = salience.topk(dominant_count).indices.sort().values
    all_indices = torch.arange(num_tokens, device=image_embeds.device)
    dominant_mask = torch.zeros(num_tokens, dtype=torch.bool, device=image_embeds.device)
    dominant_mask[dominant_indices.to(image_embeds.device)] = True
    remaining = all_indices[~dominant_mask]

    contextual_features, contextual_indices, merge_plan = _qwen3_visionzip_contextual_tokens(
        features=image_embeds,
        metric=metric,
        remaining=remaining,
        contextual_count=contextual_count,
    )
    dominant_features = image_embeds[dominant_indices.to(image_embeds.device)]
    selected_indices = torch.cat([dominant_indices.to(image_embeds.device), contextual_indices.to(image_embeds.device)])
    selected_features = torch.cat([dominant_features, contextual_features], dim=0)
    order = selected_indices.argsort()
    selected_indices = selected_indices[order]
    selected_features = selected_features[order]

    zipped_deepstack = []
    for layer_embeds in deepstack_image_embeds:
        layer_contextual, _, _ = _qwen3_visionzip_contextual_tokens(
            features=layer_embeds,
            metric=metric,
            remaining=remaining.to(layer_embeds.device),
            contextual_count=contextual_count,
            merge_plan=merge_plan,
        )
        layer_dominant = layer_embeds[dominant_indices.to(layer_embeds.device)]
        layer_selected = torch.cat([layer_dominant, layer_contextual], dim=0)[order.to(layer_embeds.device)]
        zipped_deepstack.append(layer_selected)

    info = {
        "dominant_tokens": int(dominant_count),
        "contextual_tokens": int(contextual_count),
        "contextual_merge_is_exhaustive": bool(contextual_count > 0),
        "passthrough": False,
        "selector_ms": (time.perf_counter() - selector_start) * 1000.0,
    }
    return selected_indices.detach().cpu().tolist(), selected_features, zipped_deepstack, info


def _qwen3_visionzip_contextual_tokens(*, features, metric, remaining, contextual_count: int, merge_plan=None):
    import torch
    import torch.nn.functional as F

    if contextual_count <= 0 or int(remaining.numel()) == 0:
        empty = features[:0]
        empty_indices = torch.empty(0, dtype=torch.long, device=features.device)
        return empty, empty_indices, {"target_positions": empty_indices, "merge_mask": empty_indices.bool()}

    remaining = remaining.to(features.device)
    contextual_count = min(int(contextual_count), int(remaining.numel()))
    if merge_plan is None:
        metric_filtered = metric.to(features.device)[:, remaining, :]
        metric_normalized = F.normalize(metric_filtered, dim=-1)
        step = max(1, int(metric_normalized.shape[1]) // contextual_count)
        target_positions = torch.arange(0, metric_normalized.shape[1], step, device=features.device)[:contextual_count]
        all_filtered = torch.arange(metric_normalized.shape[1], device=features.device)
        merge_mask = ~torch.isin(all_filtered, target_positions)
        if bool(merge_mask.any()):
            tokens_to_merge = metric_normalized[:, merge_mask, :]
            target_tokens = metric_normalized[:, target_positions, :]
            assignment = torch.bmm(tokens_to_merge, target_tokens.transpose(1, 2)).argmax(dim=2)
        else:
            assignment = torch.empty((1, 0), dtype=torch.long, device=features.device)
        merge_plan = {
            "target_positions": target_positions,
            "merge_mask": merge_mask,
            "assignment": assignment,
        }
    else:
        target_positions = merge_plan["target_positions"].to(features.device)
        merge_mask = merge_plan["merge_mask"].to(features.device)
        assignment = merge_plan["assignment"].to(features.device)

    target_indices = remaining[target_positions]
    target_features = features[target_indices]
    if not bool(merge_mask.any()):
        return target_features, target_indices, merge_plan

    hidden_filtered = features[remaining]
    hidden_to_merge = hidden_filtered[merge_mask]
    assign_one_hot = torch.zeros(
        hidden_to_merge.shape[0],
        int(target_positions.numel()),
        dtype=features.dtype,
        device=features.device,
    )
    assign_one_hot.scatter_(1, assignment.reshape(-1, 1), 1)
    counts = assign_one_hot.sum(dim=0).clamp(min=1).unsqueeze(-1)
    aggregated = assign_one_hot.transpose(0, 1) @ hidden_to_merge / counts
    return target_features + aggregated, target_indices, merge_plan


def _qwen3_visionzip_budget_split(num_tokens: int, keep_count: int) -> tuple[int, int]:
    keep_count = max(1, min(int(keep_count), int(num_tokens)))
    if keep_count <= 2:
        return keep_count, 0
    contextual_count = max(1, int(round(keep_count * 10 / 64)))
    contextual_count = min(contextual_count, keep_count - 1)
    dominant_count = keep_count - contextual_count
    return dominant_count, contextual_count


def _effective_keep_ratio(config: PruneConfig, sample_risk: float | None, sample_budget_ratio: float | None = None) -> float:
    if config.budget_mode in {"fixed", "evidence_saturation"}:
        return config.keep_ratio
    if config.budget_mode == "sensitivity_policy":
        if sample_budget_ratio is None:
            return config.keep_ratio
        if sample_budget_ratio <= 0.0 or sample_budget_ratio > 1.0:
            raise ValueError(f"sample budget ratio must be in (0, 1], got {sample_budget_ratio}.")
        return float(sample_budget_ratio)
    if config.budget_mode == "risk_adaptive":
        risk = 0.5 if sample_risk is None else float(sample_risk)
        return risk_adaptive_keep_ratio(risk, rho_min=config.rho_min, rho_max=config.rho_max)
    if config.budget_mode == "risk_bucket":
        risk = 0.5 if sample_risk is None else float(sample_risk)
        return risk_bucket_keep_ratio(risk, rho_low=config.rho_min, rho_mid=config.keep_ratio, rho_high=config.rho_max)
    raise ValueError(f"Unknown budget_mode={config.budget_mode!r}.")


def _budget_bucket_label(config: PruneConfig, sample_risk: float | None) -> str:
    if config.budget_mode != "risk_bucket":
        return ""
    risk = 0.5 if sample_risk is None else float(sample_risk)
    if risk < 1.0 / 3.0:
        return "low"
    if risk < 2.0 / 3.0:
        return "mid"
    return "high"


def _selector_impl(selector: str) -> tuple[str, str]:
    selector = selector.strip().lower()
    if selector.startswith("target_embed_"):
        return selector.removeprefix("target_embed_"), "target_embedding"
    if selector.startswith("embed_"):
        return selector.removeprefix("embed_"), "embedding"
    return selector, "evidence_oracle"


def _token_grid_shape(grid_thw, spatial_merge_size: int) -> tuple[int, int]:
    t = int(grid_thw[0].item())
    h = int(grid_thw[1].item()) // int(spatial_merge_size)
    w = int(grid_thw[2].item()) // int(spatial_merge_size)
    if t != 1:
        raise ValueError(f"Only single-frame image pruning is supported, got t={t}.")
    return h, w


def _evidence_relevance(token_boxes: list[Box], evidence_regions: list[Box]) -> list[float]:
    if not evidence_regions:
        return _center_prior(token_boxes)
    scores: list[float] = []
    for token in token_boxes:
        token_area = max(1e-12, (token[2] - token[0]) * (token[3] - token[1]))
        overlap = sum(intersection_area(token, evidence) for evidence in evidence_regions)
        scores.append(overlap / token_area)
    if max(scores, default=0.0) <= 0.0:
        return _center_prior(token_boxes)
    return scores


def _center_prior(token_boxes: list[Box]) -> list[float]:
    out: list[float] = []
    for box in token_boxes:
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        out.append(1.0 - min(1.0, ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5 / 0.70710678118))
    return out


def _spatial_uniqueness(token_boxes: list[Box], relevance: list[float]) -> list[float]:
    if not token_boxes:
        return []
    max_rel = max(relevance) if relevance else 0.0
    if max_rel <= 0.0:
        return [0.0 for _ in token_boxes]
    evidence_centers = []
    for box, rel in zip(token_boxes, relevance):
        if rel > 0.0:
            evidence_centers.append(((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0))
    if not evidence_centers:
        return [0.0 for _ in token_boxes]
    out: list[float] = []
    for box in token_boxes:
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        nearest = min((cx - ex) ** 2 + (cy - ey) ** 2 for ex, ey in evidence_centers)
        out.append(min(1.0, nearest ** 0.5))
    return out


def _embedding_relevance_and_uniqueness(
    *,
    inputs_embeds,
    input_ids,
    labels,
    attention_mask,
    image_positions,
    image_token_id: int,
    token_grid_h: int,
    token_grid_w: int,
    text_positions_override=None,
    relevance_weight: float = 0.85,
    query_topk: int = 2,
) -> tuple[list[float], list[float]]:
    import torch
    import torch.nn.functional as F

    if not 0.0 <= relevance_weight <= 1.0:
        raise ValueError(f"relevance_weight must be in [0, 1], got {relevance_weight}.")
    if query_topk < 1:
        raise ValueError(f"query_topk must be positive, got {query_topk}.")

    visual = inputs_embeds[0, image_positions].float()
    if visual.numel() == 0:
        return [], []
    if text_positions_override is not None and int(text_positions_override.numel()) > 0:
        text_positions = text_positions_override.to(input_ids.device)
    else:
        text_mask = labels[0].eq(-100) & input_ids[0].ne(image_token_id)
        if attention_mask is not None:
            text_mask = text_mask & attention_mask[0].bool()
        text_positions = torch.argwhere(text_mask).squeeze(1)
    if text_positions.numel() == 0:
        token_boxes = make_token_grid(token_grid_h, token_grid_w)
        return _center_prior(token_boxes), [0.0 for _ in range(len(token_boxes))]

    text = inputs_embeds[0, text_positions].float()
    visual_normed = F.normalize(visual, dim=-1)
    text_normed = F.normalize(text, dim=-1)
    sims = visual_normed @ text_normed.transpose(0, 1)
    topk = min(int(query_topk), sims.shape[1])
    relevance = sims.topk(k=topk, dim=1).values.mean(dim=1)
    relevance = relevance_weight * _minmax_tensor(relevance) + (1.0 - relevance_weight) * _minmax_tensor(
        visual.norm(dim=-1)
    )
    uniqueness = _embedding_local_uniqueness(visual_normed, token_grid_h, token_grid_w)
    return relevance.detach().cpu().tolist(), uniqueness.detach().cpu().tolist()


def _target_text_positions(*, input_ids, labels, attention_mask, tokenizer, probe: dict[str, Any]):
    import torch

    target_texts = _target_texts_from_probe(probe)
    if not target_texts:
        return torch.empty(0, dtype=torch.long, device=input_ids.device)
    prompt_mask = labels[0].eq(-100)
    if attention_mask is not None:
        prompt_mask = prompt_mask & attention_mask[0].bool()
    prompt_positions = torch.argwhere(prompt_mask).squeeze(1)
    prompt_ids = input_ids[0, prompt_positions].detach().cpu().tolist()

    matched: list[int] = []
    for target_text in target_texts:
        target_matches: list[int] = []
        for token_ids in _target_token_id_candidates(tokenizer, target_text):
            for start in _subsequence_starts(prompt_ids, token_ids):
                target_matches.extend(
                    int(prompt_positions[start + offset].item()) for offset in range(len(token_ids))
                )
            if target_matches:
                break
        if not target_matches:
            target_matches = _decoded_substring_positions(
                tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                prompt_positions=prompt_positions,
                target_text=target_text,
            )
        matched.extend(target_matches)
    if not matched:
        return torch.empty(0, dtype=torch.long, device=input_ids.device)
    return torch.tensor(sorted(set(matched)), dtype=torch.long, device=input_ids.device)


def _target_texts_from_probe(probe: dict[str, Any]) -> list[str]:
    raw_targets = probe.get("selector_target_texts")
    if isinstance(raw_targets, str):
        raw_targets = [raw_targets]
    if isinstance(raw_targets, (list, tuple)):
        targets: list[str] = []
        seen: set[str] = set()
        for value in raw_targets:
            text = str(value).strip()
            key = text.casefold()
            if text and key not in seen:
                targets.append(text)
                seen.add(key)
        if targets:
            return targets
    fallback = _target_text_from_probe(probe)
    return [fallback] if fallback else []


def _target_text_from_probe(probe: dict[str, Any]) -> str:
    for key in ("target_text", "source_text"):
        value = probe.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    value = probe.get("target_answer")
    if value is not None and str(value).strip().lower() not in {"yes", "no"}:
        return str(value).strip()
    return ""


def _target_token_id_candidates(tokenizer, target_text: str) -> list[list[int]]:
    candidates: list[list[int]] = []
    for text in (target_text, f" {target_text}", f'"{target_text}"', f" '{target_text}'"):
        try:
            encoded = tokenizer(text, add_special_tokens=False)
            token_ids = list(encoded.get("input_ids", []))
        except Exception:
            token_ids = []
        token_ids = [int(token_id) for token_id in token_ids if token_id is not None]
        if token_ids and token_ids not in candidates:
            candidates.append(token_ids)
    return candidates


def _subsequence_starts(values: list[int], pattern: list[int]) -> list[int]:
    if not values or not pattern or len(pattern) > len(values):
        return []
    starts: list[int] = []
    width = len(pattern)
    for idx in range(0, len(values) - width + 1):
        if values[idx : idx + width] == pattern:
            starts.append(idx)
    return starts


def _decoded_substring_positions(*, tokenizer, prompt_ids: list[int], prompt_positions, target_text: str) -> list[int]:
    pieces = [_decode_one_token(tokenizer, token_id) for token_id in prompt_ids]
    joined = "".join(pieces)
    span = _find_text_span(joined, target_text)
    if span is None:
        return []
    start_char, end_char = span
    matched: list[int] = []
    cursor = 0
    for idx, piece in enumerate(pieces):
        next_cursor = cursor + len(piece)
        if next_cursor > start_char and cursor < end_char:
            matched.append(int(prompt_positions[idx].item()))
        cursor = next_cursor
    return matched


def _decode_one_token(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode(
            [int(token_id)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        try:
            return tokenizer.decode([int(token_id)], skip_special_tokens=False)
        except TypeError:
            return tokenizer.decode([int(token_id)])
    except Exception:
        return ""


def _find_text_span(text: str, needle: str) -> tuple[int, int] | None:
    if not text or not needle:
        return None
    start = text.find(needle)
    if start < 0:
        start = text.casefold().find(needle.casefold())
    if start < 0:
        return None
    return start, start + len(needle)


def _embedding_local_uniqueness(visual_normed, rows: int, cols: int):
    import torch

    if rows <= 0 or cols <= 0 or rows * cols != int(visual_normed.shape[0]):
        return torch.zeros(visual_normed.shape[0], dtype=torch.float32, device=visual_normed.device)
    grid = visual_normed.reshape(rows, cols, visual_normed.shape[-1])
    sums = torch.zeros((rows, cols), dtype=torch.float32, device=visual_normed.device)
    counts = torch.zeros((rows, cols), dtype=torch.float32, device=visual_normed.device)

    def add_pair(left, right, left_slice, right_slice) -> None:
        distance = 1.0 - (left * right).sum(dim=-1)
        sums[left_slice] += distance
        sums[right_slice] += distance
        counts[left_slice] += 1.0
        counts[right_slice] += 1.0

    if cols > 1:
        add_pair(grid[:, :-1], grid[:, 1:], (slice(None), slice(None, -1)), (slice(None), slice(1, None)))
    if rows > 1:
        add_pair(grid[:-1, :], grid[1:, :], (slice(None, -1), slice(None)), (slice(1, None), slice(None)))
    uniqueness = torch.where(counts > 0.0, sums / counts.clamp_min(1.0), torch.zeros_like(sums))
    return _minmax_tensor(uniqueness.reshape(-1))


def _minmax_tensor(values):
    import torch

    span = values.max() - values.min()
    if float(span.item()) <= 1e-8:
        return torch.zeros_like(values, dtype=torch.float32)
    return ((values - values.min()) / span).float()


def _greedy_from_labels(logits, labels) -> bool:
    import torch

    target_positions = torch.argwhere(labels[0] != -100).squeeze(1)
    if target_positions.numel() == 0:
        return False
    pred_positions = target_positions - 1
    valid = pred_positions >= 0
    if not bool(valid.any().item()):
        return False
    greedy_tokens = logits[0, pred_positions[valid]].argmax(dim=-1)
    target_tokens = labels[0, target_positions[valid]]
    return bool((greedy_tokens == target_tokens).all().item())


def _sync_tensor_device(tensor) -> None:
    try:
        import torch

        if getattr(tensor.device, "type", "") == "cuda":
            torch.cuda.synchronize(tensor.device)
    except Exception:
        pass


def _empty_trace(probe: dict[str, Any], config: PruneConfig, *, elapsed_ms: float) -> dict[str, Any]:
    return {
        "sample_id": str(probe.get("sample_id", probe.get("id", ""))),
        "probe": str(probe.get("probe", "")),
        "selector": config.selector,
        "selector_impl": _selector_impl(config.selector)[0],
        "score_source": _selector_impl(config.selector)[1],
        "target_text_token_count": 0,
        "evidence_boost": float(config.evidence_boost),
        "embedding_relevance_weight": float(config.embedding_relevance_weight),
        "embedding_query_topk": int(config.embedding_query_topk),
        "budget_mode": config.budget_mode,
        "target_keep_ratio": float(config.keep_ratio),
        "effective_keep_ratio": float(config.keep_ratio),
        "risk": None,
        "full_visual_tokens": 0,
        "kept_visual_tokens": 0,
        "removal_fraction": 0.0,
        "kept_indices": [],
        "has_evidence": False,
        "evidence_region_count": 0,
        "ecr": 0.0,
        "anchor_ecr": 0.0,
        "ecr_0_5": 0.0,
        "evidence_center_recall": 0.0,
        "evidence_patch_recall": 0.0,
        "vision_ms": 0.0,
        "target_text_ms": 0.0,
        "score_compute_ms": 0.0,
        "selector_ms": 0.0,
        "prune_materialize_ms": 0.0,
        "prune_overhead_ms": 0.0,
        "language_ms": 0.0,
        "forward_ms": elapsed_ms,
    }


def _sanitize_kept_indices(indices: list[int], num_visual_tokens: int) -> list[int]:
    kept = sorted({int(idx) for idx in indices if 0 <= int(idx) < num_visual_tokens})
    if not kept and num_visual_tokens > 0:
        raise ValueError("Provided kept_indices is empty after clipping.")
    return kept


def _score_prune_fields(trace: dict[str, Any]) -> dict[str, Any]:
    return {
        "prune_selector": trace.get("selector", ""),
        "prune_score_source": trace.get("score_source", ""),
        "prune_budget_mode": trace.get("budget_mode", ""),
        "prune_keep_ratio": trace.get("effective_keep_ratio", 0.0),
        "prune_full_visual_tokens": trace.get("full_visual_tokens", 0),
        "prune_kept_visual_tokens": trace.get("kept_visual_tokens", 0),
        "prune_removal_fraction": trace.get("removal_fraction", 0.0),
        "prune_ecr": trace.get("ecr", 0.0),
        "prune_anchor_ecr": trace.get("anchor_ecr", trace.get("ecr", 0.0)),
        "prune_provenance_semantics": trace.get("provenance_semantics", "retained_token_origins"),
        "prune_evidence_center_recall": trace.get("evidence_center_recall", 0.0),
        "prune_evidence_patch_recall": trace.get("evidence_patch_recall", 0.0),
        "prune_target_text_ms": trace.get("target_text_ms", 0.0),
        "prune_score_compute_ms": trace.get("score_compute_ms", 0.0),
        "prune_selector_ms": trace.get("selector_ms", 0.0),
        "prune_materialize_ms": trace.get("prune_materialize_ms", 0.0),
        "prune_overhead_ms": trace.get("prune_overhead_ms", 0.0),
    }


def _merge_trace_pair(probe: dict[str, Any], yes_trace: dict[str, Any], no_trace: dict[str, Any]) -> dict[str, Any]:
    trace = dict(yes_trace)
    trace["sample_id"] = str(probe.get("sample_id", probe.get("id", trace.get("sample_id", ""))))
    trace["probe"] = str(probe.get("probe", trace.get("probe", "")))
    trace["yes_forward_ms"] = float(yes_trace.get("forward_ms", 0.0))
    trace["no_forward_ms"] = float(no_trace.get("forward_ms", 0.0))
    trace["mean_forward_ms"] = (trace["yes_forward_ms"] + trace["no_forward_ms"]) / 2.0
    trace["yes_vision_ms"] = float(yes_trace.get("vision_ms", 0.0))
    trace["no_vision_ms"] = float(no_trace.get("vision_ms", 0.0))
    trace["mean_vision_ms"] = (trace["yes_vision_ms"] + trace["no_vision_ms"]) / 2.0
    trace["yes_language_ms"] = float(yes_trace.get("language_ms", 0.0))
    trace["no_language_ms"] = float(no_trace.get("language_ms", 0.0))
    trace["mean_language_ms"] = (trace["yes_language_ms"] + trace["no_language_ms"]) / 2.0
    for key in (
        "target_text_ms",
        "score_compute_ms",
        "selector_ms",
        "prune_materialize_ms",
        "prune_overhead_ms",
    ):
        yes_key = f"yes_{key}"
        no_key = f"no_{key}"
        mean_key = f"mean_{key}"
        trace[yes_key] = float(yes_trace.get(key, 0.0))
        trace[no_key] = float(no_trace.get(key, 0.0))
        trace[mean_key] = (trace[yes_key] + trace[no_key]) / 2.0
    return trace


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
