"""Direct LLaVA scorer with visual-token pruning before LLM prefill."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from PIL import Image
from tqdm import tqdm

from recap.images import probe_to_visual, validate_probe_images
from recap.llava_direct_backend import (
    _format_llava_prompt,
    _load_llava_direct,
    _model_dtype,
    _move_inputs,
)
from recap.prune.budgets import fixed_keep_count, removal_fraction
from recap.prune.anchorprune import (
    AnchorPruneConfig as AnchorPruneSelectorConfig,
    UPSTREAM_COMMIT as ANCHORPRUNE_COMMIT,
    anchorprune_select,
)
from recap.prune.metrics import (
    evidence_center_recall,
    evidence_coverage,
    evidence_patch_recall,
    evidence_regions_from_sample,
    exhaustive_merge_source_indices,
    make_token_grid,
)
from recap.prune.positions import pruned_position_ids, validate_position_mode
from recap.prune.selectors import select_indices
from recap.qwen_pruned_backend import (
    _embedding_relevance_and_uniqueness,
    _greedy_from_labels,
    _mean,
    _sanitize_kept_indices,
    _selector_impl,
    _target_text_positions,
)
from recap.scoring import score_probe


@dataclass(frozen=True)
class LlavaPruneConfig:
    selector: str
    keep_ratio: float
    seed: int = 13
    hybrid_core_ratio: float = 0.50
    hybrid_context_ratio: float = 0.25
    evidence_boost: float = 0.10
    scope_alpha: float = 1.0
    coin_alpha: float = 0.90
    coin_beta: float = 0.60
    anchorprune_k_min: int = 0
    anchorprune_k_min_ratio: float = 0.15625
    anchorprune_tau: float = 0.20
    anchorprune_patience: int = 3
    anchorprune_kmax_ratio: float = 0.50
    anchorprune_clip_model: str = "openai/clip-vit-large-patch14-336"
    kept_indices_by_sample: dict[str, list[int]] | None = None
    position_mode: str = "compact"

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_mode", validate_position_mode(self.position_mode))
        if float(self.scope_alpha) < 0.0:
            raise ValueError(f"SCOPE alpha must be non-negative, got {self.scope_alpha}.")
        if int(self.anchorprune_k_min) < 0:
            raise ValueError("AnchorPrune k_min must be non-negative; zero enables ratio-based selection.")
        if float(self.anchorprune_k_min_ratio) <= 0.0:
            raise ValueError("AnchorPrune k_min ratio must be positive.")


def score_probes_with_llava_pruned(
    probes: list[dict[str, Any]],
    *,
    pretrained: str,
    prune_config: LlavaPruneConfig,
    revision: str = "main",
    device: str = "cuda",
    device_map: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
    attn_implementation: str | None = None,
    chat_template: str | None = None,
    target_delimiter: str = " ",
    debug_forward: bool = False,
    strict_images: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import torch

    image_report = validate_probe_images(probes)
    if strict_images and image_report["missing_visual_count"]:
        raise FileNotFoundError(f"Missing images for RECAP visual probes: {image_report}")
    if image_report["missing_visual_count"]:
        print(f"[RECAP-LLaVAPruned] WARNING missing images for visual probes: {image_report}", flush=True)

    effective_attn_implementation = attn_implementation
    if (
        _is_visionzip_selector(prune_config.selector)
        or _is_fastv_selector(prune_config.selector)
        or _is_scope_selector(prune_config.selector)
        or _is_anchorprune_selector(prune_config.selector)
    ) and not effective_attn_implementation:
        effective_attn_implementation = "eager"

    model, processor, input_device = _load_llava_direct(
        pretrained=pretrained,
        revision=revision,
        device=device,
        device_map=device_map,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        attn_implementation=effective_attn_implementation,
        torch_module=torch,
    )
    if _is_anchorprune_selector(prune_config.selector):
        _prepare_anchorprune_auxiliaries(
            model.model,
            clip_model=prune_config.anchorprune_clip_model,
        )

    if prune_config.position_mode != "compact" and (
        _is_visionzip_selector(prune_config.selector)
        or _is_fastv_selector(prune_config.selector)
        or _is_scope_selector(prune_config.selector)
    ):
        raise ValueError(
            "--position-mode preserve is defined only for prefill token-deletion selectors; "
            "VisionZip, FastV, and SCOPE retain their native position handling."
        )

    scored: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for probe in tqdm(probes, desc="RECAP-LLaVAPruned probes"):
        visuals = probe_to_visual(probe, strict=strict_images)
        if not isinstance(visuals, list):
            visuals = [visuals]
        visuals = [visual for visual in visuals if isinstance(visual, Image.Image)]

        yes_loss, yes_greedy, yes_trace = _score_pruned_continuation(
            probe,
            visuals,
            f"{target_delimiter}yes",
            model=model,
            processor=processor,
            input_device=input_device,
            prune_config=prune_config,
            chat_template=chat_template,
            debug_forward=debug_forward,
            torch_module=torch,
        )
        no_loss, no_greedy, no_trace = _score_pruned_continuation(
            probe,
            visuals,
            f"{target_delimiter}no",
            model=model,
            processor=processor,
            input_device=input_device,
            prune_config=prune_config,
            chat_template=chat_template,
            debug_forward=debug_forward,
            torch_module=torch,
        )
        record = score_probe(probe, yes_loss=yes_loss, no_loss=no_loss)
        record["yes_is_greedy"] = bool(yes_greedy)
        record["no_is_greedy"] = bool(no_greedy)
        record["model"] = "llava_pruned"
        record["pretrained"] = pretrained
        record.update(_score_prune_fields(yes_trace))
        scored.append(record)
        traces.append(_merge_trace_pair(probe, yes_trace, no_trace))

    return scored, traces


def summarize_llava_prune_traces(traces: list[dict[str, Any]]) -> dict[str, float]:
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
    visuals: list[Image.Image],
    continuation: str,
    *,
    model,
    processor,
    input_device,
    prune_config: LlavaPruneConfig,
    chat_template: str | None,
    debug_forward: bool,
    torch_module,
) -> tuple[float, bool, dict[str, Any]]:
    context = str(probe["question"]).replace("<image>", "")
    prompt, prompt_and_continuation = _format_llava_prompt(
        processor=processor,
        context=context,
        continuation=continuation,
        num_images=len(visuals),
        chat_template=chat_template,
    )
    if debug_forward:
        print(
            f"[RECAP-LLaVAPruned] scoring {probe.get('sample_id')}::{probe.get('probe')} continuation={continuation!r}",
            flush=True,
        )

    if visuals:
        model_inputs = processor(text=[prompt_and_continuation], images=visuals, return_tensors="pt")
        prompt_inputs = processor(text=[prompt], images=visuals, return_tensors="pt")
    else:
        model_inputs = processor(text=[prompt_and_continuation], return_tensors="pt")
        prompt_inputs = processor(text=[prompt], return_tensors="pt")
    prompt_len = prompt_inputs["input_ids"].shape[1]
    model_inputs = _move_inputs(model_inputs, input_device, _model_dtype(model), torch_module)
    labels = model_inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100

    if not visuals or "pixel_values" not in model_inputs or model_inputs.get("pixel_values") is None:
        start = time.perf_counter()
        with torch_module.inference_mode():
            outputs = model(**model_inputs, labels=labels)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        trace = _empty_trace(probe, prune_config, elapsed_ms=elapsed_ms)
        return float(outputs["loss"].item()), _greedy_from_labels(outputs["logits"], labels), trace

    start = time.perf_counter()
    with torch_module.inference_mode():
        logits, pruned_labels, trace = _forward_pruned_llava(
            model=model,
            processor=processor,
            inputs=model_inputs,
            labels=labels,
            probe=probe,
            prune_config=prune_config,
        )
        loss = model.loss_function(
            logits=logits,
            labels=pruned_labels,
            vocab_size=model.config.text_config.vocab_size,
        )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    trace["forward_ms"] = elapsed_ms
    return float(loss.item()), _greedy_from_labels(logits, pruned_labels), trace


def _forward_pruned_llava(
    *,
    model,
    processor,
    inputs,
    labels,
    probe: dict[str, Any],
    prune_config: LlavaPruneConfig,
):
    import torch

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    pixel_values = inputs["pixel_values"]
    image_sizes = inputs.get("image_sizes")
    if input_ids.shape[0] != 1:
        raise ValueError("LLaVA pruned scorer currently expects batch_size=1.")
    if int(pixel_values.shape[0]) != 1:
        raise ValueError("LLaVA pruned scorer currently supports one image per probe.")

    llava_model = model.model
    image_token_id = int(model.config.image_token_index)
    vision_start = time.perf_counter()
    inputs_embeds = llava_model.get_input_embeddings()(input_ids)
    image_positions = torch.argwhere(input_ids[0].eq(image_token_id)).squeeze(1)
    num_visual_tokens = int(image_positions.numel())
    token_grid_h, token_grid_w = _llava_token_grid_shape(model, processor, num_visual_tokens)
    token_boxes = make_token_grid(token_grid_h, token_grid_w)
    if len(token_boxes) != num_visual_tokens:
        raise ValueError(f"Token grid has {len(token_boxes)} boxes but prompt has {num_visual_tokens} image tokens.")

    sample_id = str(probe.get("sample_id", probe.get("id", "")))
    evidence_regions = evidence_regions_from_sample(probe)
    if _is_visionzip_selector(prune_config.selector):
        return _forward_visionzip_pruned_llava(
            model=model,
            llava_model=llava_model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=pixel_values,
            image_positions=image_positions,
            inputs_embeds=inputs_embeds,
            probe=probe,
            prune_config=prune_config,
            token_boxes=token_boxes,
            evidence_regions=evidence_regions,
            vision_start=vision_start,
        )
    if _is_scope_selector(prune_config.selector):
        return _forward_scope_pruned_llava(
            model=model,
            llava_model=llava_model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=pixel_values,
            image_positions=image_positions,
            inputs_embeds=inputs_embeds,
            probe=probe,
            prune_config=prune_config,
            token_boxes=token_boxes,
            evidence_regions=evidence_regions,
        )

    is_anchorprune = _is_anchorprune_selector(prune_config.selector)
    anchorprune_query = str(probe.get("question", ""))
    if "<image>" not in anchorprune_query:
        anchorprune_query = f"<image>\n{anchorprune_query}"
    anchorprune_keep_count = fixed_keep_count(num_visual_tokens, prune_config.keep_ratio)
    configured_anchor_k_min = int(prune_config.anchorprune_k_min)
    anchor_k_min = configured_anchor_k_min or max(
        1,
        int(round(anchorprune_keep_count * float(prune_config.anchorprune_k_min_ratio))),
    )
    anchorprune_cache_key = (
        sample_id,
        str(probe.get("probe", "")),
        anchorprune_query,
        anchorprune_keep_count,
        anchor_k_min,
        float(prune_config.anchorprune_tau),
        int(prune_config.anchorprune_patience),
        float(prune_config.anchorprune_kmax_ratio),
    )
    anchorprune_selection_cache = getattr(llava_model, "_anchorprune_selection_cache", {})
    cached_anchorprune_selection = anchorprune_selection_cache.get(anchorprune_cache_key) if is_anchorprune else None
    anchorprune_cache_hit = cached_anchorprune_selection is not None
    anchorprune_signals = None
    if is_anchorprune and not anchorprune_cache_hit:
        image_features, anchorprune_signals = _anchorprune_vision_signals(
            llava_model,
            pixel_values,
            vision_feature_layer=model.config.vision_feature_layer,
            vision_feature_select_strategy=model.config.vision_feature_select_strategy,
        )
    else:
        image_features_list = llava_model.get_image_features(
            pixel_values=pixel_values,
            vision_feature_layer=model.config.vision_feature_layer,
            vision_feature_select_strategy=model.config.vision_feature_select_strategy,
            image_sizes=image_sizes,
        )
        image_features = torch.cat(image_features_list, dim=0)
    image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
    special_image_mask = llava_model.get_placeholder_mask(
        input_ids,
        inputs_embeds=inputs_embeds,
        image_features=image_features,
    )
    inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
    _sync_tensor_device(input_ids)
    vision_ms = (time.perf_counter() - vision_start) * 1000.0

    if num_visual_tokens != int(image_features.shape[0]):
        raise ValueError(
            f"Prompt has {num_visual_tokens} image tokens but vision tower returned {int(image_features.shape[0])} features."
        )
    if _is_fastv_selector(prune_config.selector):
        return _forward_fastv_pruned_llava(
            model=model,
            llava_model=llava_model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            image_positions=image_positions,
            inputs_embeds=inputs_embeds,
            probe=probe,
            prune_config=prune_config,
            token_boxes=token_boxes,
            evidence_regions=evidence_regions,
            vision_ms=vision_ms,
        )

    is_coin = _is_coin_selector(prune_config.selector)
    if is_coin:
        selector_impl, score_source = "coin", "coin_prompt_alignment_norm_coverage"
    elif is_anchorprune:
        selector_impl, score_source = "anchorprune", "official_clip_priority_cls_attention_novelty"
    else:
        selector_impl, score_source = _selector_impl(prune_config.selector)
    target_text_token_count = 0
    target_text_ms = 0.0
    score_compute_start = time.perf_counter()
    if is_coin:
        relevance = []
        uniqueness = []
    elif is_anchorprune:
        if anchorprune_cache_hit:
            relevance = []
            target_text_token_count = int(cached_anchorprune_selection[2])
        else:
            if anchorprune_signals is None:
                raise RuntimeError("AnchorPrune vision signals were not initialized.")
            target_text_start = time.perf_counter()
            relevance, target_text_token_count = _anchorprune_query_priority(
                llava_model,
                anchorprune_query,
                anchorprune_signals["clip_patch_features"],
            )
            _sync_tensor_device(relevance)
            target_text_ms = (time.perf_counter() - target_text_start) * 1000.0
        uniqueness = []
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
            image_token_id=image_token_id,
            token_grid_h=token_grid_h,
            token_grid_w=token_grid_w,
            text_positions_override=text_positions_override,
        )
    else:
        relevance = _evidence_relevance(token_boxes, evidence_regions)
        uniqueness = [0.0 for _ in token_boxes]
    _sync_tensor_device(input_ids)
    score_compute_ms = (time.perf_counter() - score_compute_start) * 1000.0

    selector_start = time.perf_counter()
    sample_kept_indices = None
    if prune_config.kept_indices_by_sample is not None:
        sample_kept_indices = prune_config.kept_indices_by_sample.get(sample_id)
    if sample_kept_indices is not None:
        kept_indices = _sanitize_kept_indices(sample_kept_indices, num_visual_tokens)
        keep_count = len(kept_indices)
    elif is_coin:
        keep_count = fixed_keep_count(num_visual_tokens, prune_config.keep_ratio)
        prompt_text_mask = labels[0].eq(-100) & input_ids[0].ne(image_token_id)
        if attention_mask is not None:
            prompt_text_mask = prompt_text_mask & attention_mask[0].bool()
        prompt_text_positions = torch.argwhere(prompt_text_mask).squeeze(1)
        if int(prompt_text_positions.numel()) == 0:
            raise ValueError("CoIn requires at least one non-image prompt token.")
        selected = _coin_select_indices(
            image_features.unsqueeze(0),
            inputs_embeds[:, prompt_text_positions, :],
            keep_count=keep_count,
            alpha=prune_config.coin_alpha,
            beta=prune_config.coin_beta,
        )
        kept_indices = sorted(int(index) for index in selected[0].tolist())
        target_text_token_count = int(prompt_text_positions.numel())
    elif is_anchorprune:
        keep_count = anchorprune_keep_count
        if anchorprune_cache_hit:
            selected = torch.tensor(cached_anchorprune_selection[0], dtype=torch.long, device=input_ids.device)
            anchor_indices = torch.tensor(cached_anchorprune_selection[1], dtype=torch.long, device=input_ids.device)
        else:
            if anchorprune_signals is None:
                raise RuntimeError("AnchorPrune selection signals were not initialized.")
            selected, anchor_indices = anchorprune_select(
                relevance=relevance,
                features=anchorprune_signals["clip_patch_features"],
                importance=anchorprune_signals["importance_prior"],
                config=AnchorPruneSelectorConfig(
                    k_total=keep_count,
                    k_min=anchor_k_min,
                    tau=float(prune_config.anchorprune_tau),
                    patience=int(prune_config.anchorprune_patience),
                    kmax_ratio=float(prune_config.anchorprune_kmax_ratio),
                ),
                expansion_features=anchorprune_signals["vision_features"],
            )
            anchorprune_selection_cache[anchorprune_cache_key] = (
                selected.detach().cpu().tolist(),
                anchor_indices.detach().cpu().tolist(),
                int(target_text_token_count),
            )
        kept_indices = sorted(int(index) for index in selected.tolist())
    else:
        keep_count = fixed_keep_count(num_visual_tokens, prune_config.keep_ratio)
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
            salt=f"{probe.get('sample_id', probe.get('id', ''))}:{probe.get('probe', '')}:{prune_config.keep_ratio}:{prune_config.selector}",
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
    cache_position = torch.arange(pruned_inputs_embeds.shape[1], device=pruned_inputs_embeds.device)
    language_kwargs = {
        "attention_mask": pruned_attention_mask,
        "inputs_embeds": pruned_inputs_embeds,
        "cache_position": cache_position,
    }
    if prune_config.position_mode == "preserve":
        language_kwargs["position_ids"] = pruned_position_ids(keep_sequence, mode="preserve")
    _sync_tensor_device(pruned_inputs_embeds)
    prune_materialize_ms = (time.perf_counter() - prune_materialize_start) * 1000.0
    _sync_tensor_device(pruned_inputs_embeds)
    language_start = time.perf_counter()
    outputs = llava_model.language_model(**language_kwargs)
    logits = model.lm_head(outputs.last_hidden_state)
    _sync_tensor_device(pruned_inputs_embeds)
    language_ms = (time.perf_counter() - language_start) * 1000.0

    coverage = evidence_coverage(kept_indices, token_boxes, evidence_regions)
    center_recall = evidence_center_recall(kept_indices, token_boxes, evidence_regions)
    patch_recall = evidence_patch_recall(kept_indices, token_boxes, evidence_regions)
    trace = {
        "sample_id": sample_id,
        "probe": str(probe.get("probe", "")),
        "selector": prune_config.selector,
        "selector_impl": selector_impl,
        "score_source": score_source,
        "position_mode": prune_config.position_mode,
        "target_text_token_count": target_text_token_count,
        "evidence_boost": float(prune_config.evidence_boost),
        "coin_alpha": float(prune_config.coin_alpha) if is_coin else None,
        "coin_beta": float(prune_config.coin_beta) if is_coin else None,
        "anchorprune_commit": ANCHORPRUNE_COMMIT if is_anchorprune else None,
        "anchorprune_anchor_size": int(anchor_indices.numel()) if is_anchorprune else None,
        "anchorprune_k_min": int(anchor_k_min) if is_anchorprune else None,
        "anchorprune_tau": float(prune_config.anchorprune_tau) if is_anchorprune else None,
        "anchorprune_patience": int(prune_config.anchorprune_patience) if is_anchorprune else None,
        "anchorprune_kmax_ratio": float(prune_config.anchorprune_kmax_ratio) if is_anchorprune else None,
        "anchorprune_clip_model": prune_config.anchorprune_clip_model if is_anchorprune else None,
        "anchorprune_cache_hit": bool(anchorprune_cache_hit) if is_anchorprune else None,
        "budget_mode": "provided_indices" if sample_kept_indices is not None else "fixed",
        "target_keep_ratio": float(prune_config.keep_ratio),
        "effective_keep_ratio": float(keep_count / num_visual_tokens) if num_visual_tokens else 0.0,
        "full_sequence_tokens": int(input_ids.shape[1]),
        "pruned_sequence_tokens": int(pruned_inputs_embeds.shape[1]),
        "full_visual_tokens": num_visual_tokens,
        "kept_visual_tokens": len(kept_indices),
        "provided_indices": bool(sample_kept_indices is not None),
        "removal_fraction": removal_fraction(num_visual_tokens, len(kept_indices)),
        "kept_indices": kept_indices,
        "has_evidence": bool(evidence_regions),
        "evidence_region_count": len(evidence_regions),
        "ecr": coverage,
        "ecr_0_5": 1.0 if coverage >= 0.5 else 0.0,
        "evidence_center_recall": center_recall,
        "evidence_patch_recall": patch_recall,
        "grid_h": token_grid_h,
        "grid_w": token_grid_w,
        "vision_ms": vision_ms,
        "target_text_ms": target_text_ms,
        "score_compute_ms": score_compute_ms,
        "selector_ms": selector_ms,
        "prune_materialize_ms": prune_materialize_ms,
        "prune_overhead_ms": score_compute_ms + selector_ms + prune_materialize_ms,
        "language_ms": language_ms,
    }
    return logits, pruned_labels, trace


def _is_visionzip_selector(selector: str) -> bool:
    return selector.strip().lower() in {"visionzip", "official_visionzip"}


def _is_fastv_selector(selector: str) -> bool:
    return selector.strip().lower() in {"fastv", "official_fastv"}


def _is_scope_selector(selector: str) -> bool:
    return selector.strip().lower() in {"scope", "official_scope"}


def _is_coin_selector(selector: str) -> bool:
    return selector.strip().lower() in {"coin", "paper_coin"}


def _is_anchorprune_selector(selector: str) -> bool:
    return selector.strip().lower() in {"anchorprune", "official_anchorprune"}


def _prepare_anchorprune_auxiliaries(llava_model, *, clip_model: str) -> None:
    """Load the CLIP components used by the pinned official LLaVA adapter."""
    if getattr(llava_model, "_anchorprune_clip_model", None) == clip_model:
        return

    import torch
    from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast, CLIPVisionModelWithProjection

    vision_device = next(llava_model.vision_tower.parameters()).device
    tokenizer = CLIPTokenizerFast.from_pretrained(clip_model)
    text_tower = CLIPTextModelWithProjection.from_pretrained(clip_model, low_cpu_mem_usage=True)
    text_tower.requires_grad_(False)
    text_tower.eval()
    text_tower = text_tower.to(device=vision_device)

    vision_with_projection = CLIPVisionModelWithProjection.from_pretrained(clip_model, low_cpu_mem_usage=True)
    visual_projection = vision_with_projection.visual_projection.to(device=vision_device)
    visual_projection.requires_grad_(False)
    visual_projection.eval()
    del vision_with_projection
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    llava_model._anchorprune_clip_model = clip_model
    llava_model._anchorprune_text_tokenizer = tokenizer
    llava_model._anchorprune_text_tower = text_tower
    llava_model._anchorprune_visual_projection = visual_projection
    llava_model._anchorprune_text_cache = {}
    llava_model._anchorprune_selection_cache = {}


def _anchorprune_vision_signals(
    llava_model,
    pixel_values,
    *,
    vision_feature_layer,
    vision_feature_select_strategy: str,
):
    """Extract the three signals used by AnchorPrune's official LLaVA adapter."""
    import torch
    import torch.nn.functional as functional

    if not isinstance(vision_feature_layer, int):
        raise ValueError("AnchorPrune LLaVA parity currently requires one vision feature layer.")
    if vision_feature_select_strategy not in {"default", "full"}:
        raise ValueError(f"Unexpected LLaVA vision feature strategy: {vision_feature_select_strategy}.")

    outputs = llava_model.vision_tower(
        pixel_values,
        output_hidden_states=True,
        output_attentions=True,
        return_dict=True,
    )
    selected_image_feature = outputs.hidden_states[vision_feature_layer]
    if vision_feature_select_strategy == "default":
        selected_image_feature = selected_image_feature[:, 1:]
    projected_image_features = llava_model.multi_modal_projector(selected_image_feature)[0]

    layer_index = -2
    effective_layer = layer_index if layer_index >= 0 else len(outputs.attentions) + layer_index
    vision_features = outputs.hidden_states[effective_layer + 1][0, 1:, :].float()
    attention = outputs.attentions[effective_layer]
    importance_prior = attention[0, :, 0, 1:].mean(dim=0).float()
    importance_prior = importance_prior / importance_prior.sum().clamp_min(1e-12)

    clip_patch_tokens = outputs.hidden_states[vision_feature_layer][:, 1:, :]
    post_layernorm = llava_model.vision_tower.vision_model.post_layernorm
    normalized_tokens = post_layernorm(clip_patch_tokens.to(dtype=next(post_layernorm.parameters()).dtype))
    projection = llava_model._anchorprune_visual_projection
    clip_patch_features = projection(normalized_tokens.to(dtype=projection.weight.dtype))
    clip_patch_features = functional.normalize(clip_patch_features.float(), dim=-1)[0]

    if not (
        projected_image_features.shape[0]
        == vision_features.shape[0]
        == importance_prior.shape[0]
        == clip_patch_features.shape[0]
    ):
        raise ValueError("AnchorPrune signal tensors do not share the LLaVA visual-token length.")
    return projected_image_features, {
        "clip_patch_features": clip_patch_features,
        "vision_features": vision_features,
        "importance_prior": importance_prior,
    }


def _anchorprune_query_priority(llava_model, query_text: str, clip_patch_features):
    """Match the official negated CLIP patch-text anchoring priority."""
    import torch
    import torch.nn.functional as functional

    cache = llava_model._anchorprune_text_cache
    cache_key = str(query_text)
    text_embeddings = cache.get(cache_key)
    tokenizer = llava_model._anchorprune_text_tokenizer
    text_tower = llava_model._anchorprune_text_tower
    tokenized = tokenizer(text=[cache_key], return_tensors="pt")
    token_count = int(tokenized.input_ids.shape[1])
    if text_embeddings is None:
        max_positions = int(text_tower.config.max_position_embeddings)
        segments = (token_count - 1) // max_positions + 1
        padding = max_positions * segments - token_count
        model_inputs = {
            key: torch.cat([value, value.new_zeros((value.shape[0], padding))], dim=1)
            .reshape(-1, max_positions)
            .to(device=next(text_tower.parameters()).device, non_blocking=True)
            for key, value in tokenized.items()
        }
        text_embeddings = text_tower(**model_inputs, return_dict=True).text_embeds.float()
        if len(cache) >= 256:
            cache.pop(next(iter(cache)))
        cache[cache_key] = text_embeddings
    text_embeddings = functional.normalize(
        text_embeddings.to(device=clip_patch_features.device, dtype=torch.float32, non_blocking=True),
        dim=-1,
    )
    raw_similarity = clip_patch_features @ text_embeddings.t()
    priority = (-raw_similarity).mean(dim=-1)
    minimum = priority.min()
    maximum = priority.max()
    normalized = (priority - minimum + 1e-6) / (maximum - minimum + 1e-12)
    return normalized, token_count


def _coin_select_indices(visual_features, text_features, *, keep_count: int, alpha: float = 0.9, beta: float = 0.6):
    """Paper-based port of CoIn's incremental Gram--Schmidt selector."""
    import torch
    import torch.nn.functional as F

    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError(f"CoIn alpha must be in [0, 1], got {alpha}.")
    if not 0.0 <= float(beta) <= 1.0:
        raise ValueError(f"CoIn beta must be in [0, 1], got {beta}.")
    if visual_features.ndim != 3 or text_features.ndim != 3:
        raise ValueError("CoIn expects batched visual and text feature tensors.")
    if visual_features.shape[0] != text_features.shape[0] or visual_features.shape[-1] != text_features.shape[-1]:
        raise ValueError("CoIn visual and text features must share batch and embedding dimensions.")

    batch_size, num_tokens, _ = visual_features.shape
    keep_count = max(1, min(int(keep_count), int(num_tokens)))
    work_features = visual_features.float()
    normalized = F.normalize(work_features, dim=-1)
    mean_text = text_features.float().mean(dim=1)
    alignment = F.cosine_similarity(work_features, mean_text.unsqueeze(1), dim=-1)
    saliency = work_features.norm(dim=-1)
    informativeness = float(beta) * _batch_minmax(saliency) + (1.0 - float(beta)) * _batch_minmax(alignment)

    selected = torch.zeros(batch_size, num_tokens, dtype=torch.bool, device=visual_features.device)
    selected_indices = torch.empty(batch_size, keep_count, dtype=torch.long, device=visual_features.device)
    projection_coefficients = torch.zeros(
        batch_size,
        num_tokens,
        keep_count,
        dtype=normalized.dtype,
        device=visual_features.device,
    )
    orthonormal_basis = []
    batch_indices = torch.arange(batch_size, device=visual_features.device)
    best = informativeness.argmax(dim=1)

    for step in range(keep_count):
        selected[batch_indices, best] = True
        selected_indices[:, step] = best
        candidate = normalized[batch_indices, best]
        if orthonormal_basis:
            basis = torch.stack(orthonormal_basis, dim=1)
            candidate = candidate - torch.bmm(
                torch.bmm(candidate.unsqueeze(1), basis.transpose(1, 2)), basis
            ).squeeze(1)
        candidate = F.normalize(candidate, dim=-1)
        orthonormal_basis.append(candidate)
        projection_coefficients[:, :, step] = torch.bmm(normalized, candidate.unsqueeze(-1)).squeeze(-1)

        if step + 1 < keep_count:
            squared_distance = (1.0 - projection_coefficients[:, :, : step + 1].square().sum(dim=-1)).clamp(min=0.0)
            coverage_gain = squared_distance.sqrt()
            scores = float(alpha) * coverage_gain + (1.0 - float(alpha)) * informativeness
            best = scores.masked_fill(selected, float("-inf")).argmax(dim=1)

    return selected_indices


def _batch_minmax(values):
    import torch

    minimum = values.amin(dim=1, keepdim=True)
    value_range = values.amax(dim=1, keepdim=True) - minimum
    return torch.where(value_range > 0, (values - minimum) / value_range.clamp(min=1e-12), torch.zeros_like(values))


def _forward_fastv_pruned_llava(
    *,
    model,
    llava_model,
    input_ids,
    attention_mask,
    labels,
    image_positions,
    inputs_embeds,
    probe: dict[str, Any],
    prune_config: LlavaPruneConfig,
    token_boxes,
    evidence_regions,
    vision_ms: float,
):
    language_start = time.perf_counter()
    hidden_states, pruned_labels, kept_indices, fastv_info = _fastv_language_forward(
        language_model=llava_model.language_model,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=labels,
        image_positions=image_positions,
        keep_ratio=prune_config.keep_ratio,
        fastv_k=3,
    )
    logits = model.lm_head(hidden_states)
    _sync_tensor_device(hidden_states)
    language_ms = (time.perf_counter() - language_start) * 1000.0

    coverage = evidence_coverage(kept_indices, token_boxes, evidence_regions)
    center_recall = evidence_center_recall(kept_indices, token_boxes, evidence_regions)
    patch_recall = evidence_patch_recall(kept_indices, token_boxes, evidence_regions)
    trace = {
        "sample_id": str(probe.get("sample_id", probe.get("id", ""))),
        "probe": str(probe.get("probe", "")),
        "selector": prune_config.selector,
        "selector_impl": "fastv",
        "score_source": "official_fastv_layer_attention",
        "position_mode": "official_fastv_preserve",
        "target_text_token_count": 0,
        "evidence_boost": float(prune_config.evidence_boost),
        "budget_mode": "fixed",
        "target_keep_ratio": float(prune_config.keep_ratio),
        "effective_keep_ratio": float(prune_config.keep_ratio),
        "full_sequence_tokens": int(input_ids.shape[1]),
        "pruned_sequence_tokens": int(hidden_states.shape[1]),
        "full_visual_tokens": int(image_positions.numel()),
        "kept_visual_tokens": len(kept_indices),
        "removal_fraction": removal_fraction(int(image_positions.numel()), len(kept_indices)),
        "kept_indices": kept_indices,
        "has_evidence": bool(evidence_regions),
        "evidence_region_count": len(evidence_regions),
        "ecr": coverage,
        "ecr_0_5": 1.0 if coverage >= 0.5 else 0.0,
        "evidence_center_recall": center_recall,
        "evidence_patch_recall": patch_recall,
        "grid_h": int(round(math.sqrt(int(image_positions.numel())))) if int(image_positions.numel()) else 0,
        "grid_w": int(round(math.sqrt(int(image_positions.numel())))) if int(image_positions.numel()) else 0,
        "vision_ms": vision_ms,
        "target_text_ms": 0.0,
        "score_compute_ms": float(fastv_info["attention_capture_ms"]),
        "selector_ms": float(fastv_info["selector_ms"]),
        "prune_materialize_ms": float(fastv_info["prune_materialize_ms"]),
        "prune_overhead_ms": float(fastv_info["selector_ms"]) + float(fastv_info["prune_materialize_ms"]),
        "language_ms": language_ms,
        "fastv_commit": "pkunlp-icler/fastv@d165972",
        "fastv_k": int(fastv_info["fastv_k"]),
        "fastv_r": float(prune_config.keep_ratio),
    }
    return logits, pruned_labels, trace


def _fastv_language_forward(
    *,
    language_model,
    inputs_embeds,
    attention_mask,
    labels,
    image_positions,
    keep_ratio: float,
    fastv_k: int,
):
    import torch
    from transformers.models.llama.modeling_llama import create_causal_mask

    hidden_states = inputs_embeds
    labels_current = labels
    attention_mask_current = attention_mask
    cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
    position_ids = cache_position.unsqueeze(0)
    last_attention = None
    pruned = False
    kept_indices = list(range(int(image_positions.numel())))
    selector_ms = 0.0
    prune_materialize_ms = 0.0
    attention_capture_ms = 0.0

    for layer_idx, decoder_layer in enumerate(language_model.layers[: language_model.config.num_hidden_layers]):
        if layer_idx == fastv_k and not pruned and hidden_states.shape[1] > 1:
            selector_start = time.perf_counter()
            if last_attention is None:
                raise RuntimeError("FastV did not capture attention before the pruning layer.")
            image_attention_score = last_attention.mean(dim=1)[0, -1, image_positions.to(last_attention.device)]
            keep_count = fixed_keep_count(int(image_positions.numel()), keep_ratio)
            top_image_indices = image_attention_score.topk(keep_count).indices.to(image_positions.device)
            kept_indices = sorted(int(idx.item()) for idx in top_image_indices)
            kept_positions = image_positions[top_image_indices]
            text_positions = torch.ones(hidden_states.shape[1], dtype=torch.bool, device=hidden_states.device)
            text_positions[image_positions.to(hidden_states.device)] = False
            keep_sequence = text_positions
            keep_sequence[kept_positions.to(hidden_states.device)] = True
            keep_indexs = torch.argwhere(keep_sequence).squeeze(1)
            selector_ms = (time.perf_counter() - selector_start) * 1000.0

            materialize_start = time.perf_counter()
            hidden_states = hidden_states[:, keep_indexs, :]
            labels_current = labels_current[:, keep_indexs]
            if attention_mask_current is not None:
                attention_mask_current = attention_mask_current[:, keep_indexs]
            cache_position = cache_position[keep_indexs]
            position_ids = cache_position.unsqueeze(0)
            prune_materialize_ms = (time.perf_counter() - materialize_start) * 1000.0
            pruned = True

        causal_mask = create_causal_mask(
            config=language_model.config,
            input_embeds=hidden_states,
            attention_mask=attention_mask_current,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )
        position_embeddings = language_model.rotary_emb(hidden_states, position_ids)
        capture_attention = layer_idx == fastv_k - 1
        if capture_attention:
            capture_start = time.perf_counter()
            hidden_states, last_attention = _fastv_decoder_layer_forward(
                decoder_layer=decoder_layer,
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                capture_attention=True,
            )
            attention_capture_ms += (time.perf_counter() - capture_start) * 1000.0
        else:
            hidden_states, _ = _fastv_decoder_layer_forward(
                decoder_layer=decoder_layer,
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                capture_attention=False,
            )

    hidden_states = language_model.norm(hidden_states)
    info = {
        "fastv_k": fastv_k,
        "selector_ms": selector_ms,
        "prune_materialize_ms": prune_materialize_ms,
        "attention_capture_ms": attention_capture_ms,
    }
    return hidden_states, labels_current, kept_indices, info


def _fastv_decoder_layer_forward(
    *,
    decoder_layer,
    hidden_states,
    attention_mask,
    position_ids,
    cache_position,
    position_embeddings,
    capture_attention: bool,
):
    residual = hidden_states
    hidden_states = decoder_layer.input_layernorm(hidden_states)
    attn_output, attn_weights = decoder_layer.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=False,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
    )
    if capture_attention and attn_weights is None:
        raise RuntimeError("FastV requires eager LLaMA attention weights.")
    hidden_states = residual + attn_output
    residual = hidden_states
    hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
    hidden_states = decoder_layer.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states, attn_weights if capture_attention else None


def _forward_visionzip_pruned_llava(
    *,
    model,
    llava_model,
    input_ids,
    attention_mask,
    labels,
    pixel_values,
    image_positions,
    inputs_embeds,
    probe: dict[str, Any],
    prune_config: LlavaPruneConfig,
    token_boxes,
    evidence_regions,
    vision_start: float,
):
    import torch

    num_visual_tokens = int(image_positions.numel())
    keep_count = fixed_keep_count(num_visual_tokens, prune_config.keep_ratio)
    zip_features, anchor_indices, zip_info = _visionzip_image_features(
        llava_model=llava_model,
        pixel_values=pixel_values,
        keep_count=keep_count,
        output_device=inputs_embeds.device,
        output_dtype=inputs_embeds.dtype,
    )
    _sync_tensor_device(zip_features)
    vision_ms = (time.perf_counter() - vision_start) * 1000.0

    prune_materialize_start = time.perf_counter()
    pruned_inputs_embeds, pruned_attention_mask, pruned_labels = _replace_visual_span(
        inputs_embeds=inputs_embeds,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        image_positions=image_positions,
        image_features=zip_features,
    )
    cache_position = torch.arange(pruned_inputs_embeds.shape[1], device=pruned_inputs_embeds.device)
    _sync_tensor_device(pruned_inputs_embeds)
    prune_materialize_ms = (time.perf_counter() - prune_materialize_start) * 1000.0

    language_start = time.perf_counter()
    outputs = llava_model.language_model(
        attention_mask=pruned_attention_mask,
        inputs_embeds=pruned_inputs_embeds,
        cache_position=cache_position,
    )
    logits = model.lm_head(outputs.last_hidden_state)
    _sync_tensor_device(pruned_inputs_embeds)
    language_ms = (time.perf_counter() - language_start) * 1000.0

    provenance_indices = exhaustive_merge_source_indices(
        num_visual_tokens,
        anchor_indices,
        contextual_tokens=int(zip_info["contextual_tokens"]),
    )
    coverage = evidence_coverage(provenance_indices, token_boxes, evidence_regions)
    center_recall = evidence_center_recall(provenance_indices, token_boxes, evidence_regions)
    patch_recall = evidence_patch_recall(provenance_indices, token_boxes, evidence_regions)
    anchor_coverage = evidence_coverage(anchor_indices, token_boxes, evidence_regions)
    trace = {
        "sample_id": str(probe.get("sample_id", probe.get("id", ""))),
        "probe": str(probe.get("probe", "")),
        "selector": prune_config.selector,
        "selector_impl": "visionzip",
        "score_source": "official_visionzip_cls_attention",
        "position_mode": "compressed_span_compact",
        "target_text_token_count": 0,
        "evidence_boost": float(prune_config.evidence_boost),
        "budget_mode": "fixed",
        "target_keep_ratio": float(prune_config.keep_ratio),
        "effective_keep_ratio": float(prune_config.keep_ratio),
        "full_sequence_tokens": int(input_ids.shape[1]),
        "pruned_sequence_tokens": int(pruned_inputs_embeds.shape[1]),
        "full_visual_tokens": num_visual_tokens,
        "kept_visual_tokens": int(zip_features.shape[0]),
        "removal_fraction": removal_fraction(num_visual_tokens, int(zip_features.shape[0])),
        "kept_indices": anchor_indices,
        "provenance_indices": provenance_indices,
        "provenance_semantics": "source_lineage_union",
        "has_evidence": bool(evidence_regions),
        "evidence_region_count": len(evidence_regions),
        "ecr": coverage,
        "anchor_ecr": anchor_coverage,
        "ecr_0_5": 1.0 if coverage >= 0.5 else 0.0,
        "evidence_center_recall": center_recall,
        "evidence_patch_recall": patch_recall,
        "grid_h": int(round(math.sqrt(num_visual_tokens))) if num_visual_tokens else 0,
        "grid_w": int(round(math.sqrt(num_visual_tokens))) if num_visual_tokens else 0,
        "vision_ms": vision_ms,
        "target_text_ms": 0.0,
        "score_compute_ms": 0.0,
        "selector_ms": float(zip_info["selector_ms"]),
        "prune_materialize_ms": prune_materialize_ms,
        "prune_overhead_ms": float(zip_info["selector_ms"]) + prune_materialize_ms,
        "language_ms": language_ms,
        "visionzip_commit": "JIA-Lab-research/VisionZip@8f86b55",
        "visionzip_dominant_patch_tokens": int(zip_info["dominant_patch_tokens"]),
        "visionzip_contextual_tokens": int(zip_info["contextual_tokens"]),
        "visionzip_includes_cls_token": True,
    }
    return logits, pruned_labels, trace


def _forward_scope_pruned_llava(
    *,
    model,
    llava_model,
    input_ids,
    attention_mask,
    labels,
    pixel_values,
    image_positions,
    inputs_embeds,
    probe: dict[str, Any],
    prune_config: LlavaPruneConfig,
    token_boxes,
    evidence_regions,
):
    import torch

    num_visual_tokens = int(image_positions.numel())
    keep_count = fixed_keep_count(num_visual_tokens, prune_config.keep_ratio)
    scope_features, kept_indices, scope_info = _scope_image_features(
        llava_model=llava_model,
        pixel_values=pixel_values,
        keep_count=keep_count,
        alpha=prune_config.scope_alpha,
        output_device=inputs_embeds.device,
        output_dtype=inputs_embeds.dtype,
    )
    _sync_tensor_device(scope_features)
    vision_ms = float(scope_info["vision_ms"])

    prune_materialize_start = time.perf_counter()
    pruned_inputs_embeds, pruned_attention_mask, pruned_labels = _replace_visual_span(
        inputs_embeds=inputs_embeds,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        image_positions=image_positions,
        image_features=scope_features,
    )
    cache_position = torch.arange(pruned_inputs_embeds.shape[1], device=pruned_inputs_embeds.device)
    _sync_tensor_device(pruned_inputs_embeds)
    prune_materialize_ms = (time.perf_counter() - prune_materialize_start) * 1000.0

    language_start = time.perf_counter()
    outputs = llava_model.language_model(
        attention_mask=pruned_attention_mask,
        inputs_embeds=pruned_inputs_embeds,
        cache_position=cache_position,
    )
    logits = model.lm_head(outputs.last_hidden_state)
    _sync_tensor_device(pruned_inputs_embeds)
    language_ms = (time.perf_counter() - language_start) * 1000.0

    coverage = evidence_coverage(kept_indices, token_boxes, evidence_regions)
    center_recall = evidence_center_recall(kept_indices, token_boxes, evidence_regions)
    patch_recall = evidence_patch_recall(kept_indices, token_boxes, evidence_regions)
    trace = {
        "sample_id": str(probe.get("sample_id", probe.get("id", ""))),
        "probe": str(probe.get("probe", "")),
        "selector": prune_config.selector,
        "selector_impl": "scope",
        "score_source": "official_scope_cls_attention_coverage",
        "position_mode": "compressed_span_compact",
        "target_text_token_count": 0,
        "evidence_boost": float(prune_config.evidence_boost),
        "budget_mode": "fixed",
        "target_keep_ratio": float(prune_config.keep_ratio),
        "effective_keep_ratio": float(prune_config.keep_ratio),
        "full_sequence_tokens": int(input_ids.shape[1]),
        "pruned_sequence_tokens": int(pruned_inputs_embeds.shape[1]),
        "full_visual_tokens": num_visual_tokens,
        "kept_visual_tokens": int(scope_features.shape[0]),
        "removal_fraction": removal_fraction(num_visual_tokens, int(scope_features.shape[0])),
        "kept_indices": kept_indices,
        "has_evidence": bool(evidence_regions),
        "evidence_region_count": len(evidence_regions),
        "ecr": coverage,
        "ecr_0_5": 1.0 if coverage >= 0.5 else 0.0,
        "evidence_center_recall": center_recall,
        "evidence_patch_recall": patch_recall,
        "grid_h": int(round(math.sqrt(num_visual_tokens))) if num_visual_tokens else 0,
        "grid_w": int(round(math.sqrt(num_visual_tokens))) if num_visual_tokens else 0,
        "vision_ms": vision_ms,
        "target_text_ms": 0.0,
        "score_compute_ms": 0.0,
        "selector_ms": float(scope_info["selector_ms"]),
        "prune_materialize_ms": prune_materialize_ms,
        "prune_overhead_ms": float(scope_info["selector_ms"]) + prune_materialize_ms,
        "language_ms": language_ms,
        "scope_commit": "kinredon/SCOPE@6bf7306",
        "scope_alpha": float(scope_info["alpha"]),
        "scope_combination": "multiplicative",
        "scope_vision_encoder_ms": float(scope_info["vision_encoder_ms"]),
        "scope_projector_ms": float(scope_info["projector_ms"]),
    }
    return logits, pruned_labels, trace


def _scope_image_features(*, llava_model, pixel_values, keep_count: int, alpha: float, output_device, output_dtype):
    vision_tower = llava_model.vision_tower
    tower_param = next(vision_tower.parameters())
    tower_inputs = pixel_values.to(device=tower_param.device, dtype=tower_param.dtype)
    vision_start = time.perf_counter()
    image_outputs = vision_tower(tower_inputs, output_hidden_states=True, output_attentions=True)
    _sync_tensor_device(image_outputs.last_hidden_state)
    vision_encoder_ms = (time.perf_counter() - vision_start) * 1000.0
    if not image_outputs.attentions:
        raise RuntimeError("SCOPE requires CLIP vision attentions, but the vision tower returned none.")
    attn_weights = image_outputs.attentions[-2]
    if attn_weights is None:
        raise RuntimeError("SCOPE requires CLIP vision attentions; load LLaVA with attn_implementation='eager'.")
    hidden_states = image_outputs.hidden_states[-2]
    if hidden_states.shape[0] != 1:
        raise ValueError("SCOPE scorer currently expects one image per probe.")

    patch_features = hidden_states[:, 1:, :]
    cls_attention = attn_weights[:, :, 0, 1:].sum(dim=1)
    selector_start = time.perf_counter()
    selected = _scope_select_indices(patch_features, cls_attention, keep_count=keep_count, alpha=alpha)
    _sync_tensor_device(selected)
    selector_ms = (time.perf_counter() - selector_start) * 1000.0

    projector_start = time.perf_counter()
    selected_sorted = selected.sort(dim=1).values
    gather_index = selected_sorted.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
    selected_hidden = patch_features.gather(1, gather_index)
    projected = llava_model.multi_modal_projector(selected_hidden)[0].to(
        device=output_device,
        dtype=output_dtype,
    )
    _sync_tensor_device(projected)
    projector_ms = (time.perf_counter() - projector_start) * 1000.0
    kept_indices = [int(index) for index in selected_sorted[0].tolist()]
    info = {
        "vision_ms": vision_encoder_ms + projector_ms,
        "vision_encoder_ms": vision_encoder_ms,
        "projector_ms": projector_ms,
        "selector_ms": selector_ms,
        "alpha": float(alpha),
    }
    return projected, kept_indices, info


def _scope_select_indices(visual_features, cls_attention, *, keep_count: int, alpha: float = 1.0):
    """Faithful tensor port of SCOPE's official multiplicative selector."""
    import torch

    batch_size, num_tokens, _ = visual_features.shape
    keep_count = max(1, min(int(keep_count), int(num_tokens)))
    normalized = visual_features / visual_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    cosine_similarity = torch.bmm(normalized, normalized.transpose(1, 2))
    selected = torch.zeros(batch_size, num_tokens, dtype=torch.bool, device=visual_features.device)
    selected_indices = torch.empty(batch_size, keep_count, dtype=torch.long, device=visual_features.device)
    current_coverage = torch.zeros(
        batch_size,
        num_tokens,
        dtype=visual_features.dtype,
        device=visual_features.device,
    )
    saliency = cls_attention.to(dtype=visual_features.dtype).pow(float(alpha))
    batch_indices = torch.arange(batch_size, device=visual_features.device)

    for step in range(keep_count):
        unselected = ~selected
        gains = torch.maximum(
            torch.zeros(1, dtype=visual_features.dtype, device=visual_features.device),
            cosine_similarity.masked_fill(~unselected.unsqueeze(1), 0) - current_coverage.unsqueeze(2),
        ).sum(dim=1)
        gains = (gains * saliency).masked_fill(~unselected, float("-inf"))
        best = gains.argmax(dim=1)
        selected[batch_indices, best] = True
        selected_indices[:, step] = best
        current_coverage = torch.maximum(current_coverage, cosine_similarity[batch_indices, best])

    return selected_indices


def _visionzip_image_features(*, llava_model, pixel_values, keep_count: int, output_device, output_dtype):
    import torch

    selector_start = time.perf_counter()
    vision_tower = llava_model.vision_tower
    tower_param = next(vision_tower.parameters())
    tower_inputs = pixel_values.to(device=tower_param.device, dtype=tower_param.dtype)
    image_outputs = vision_tower(tower_inputs, output_hidden_states=True, output_attentions=True)
    if not image_outputs.attentions:
        raise RuntimeError("VisionZip requires CLIP vision attentions, but the vision tower returned none.")
    attn_weights = image_outputs.attentions[-2]
    if attn_weights is None:
        raise RuntimeError("VisionZip requires CLIP vision attentions; load LLaVA with attn_implementation='eager'.")
    hidden_states = image_outputs.hidden_states[-2]
    metric = _clip_key_metric(vision_tower, image_outputs.hidden_states[-3])

    if hidden_states.shape[0] != 1:
        raise ValueError("VisionZip scorer currently expects one image per probe.")
    num_patches = int(hidden_states.shape[1] - 1)
    dominant_patch_count, contextual_count = _visionzip_budget_split(num_patches, keep_count)

    cls_attention = attn_weights[:, :, 0, 1:].sum(dim=1)
    dominant_patch_indices = cls_attention.topk(dominant_patch_count, dim=1).indices[0]
    cls_and_dominant_positions = torch.cat(
        [
            torch.zeros(1, dtype=dominant_patch_indices.dtype, device=dominant_patch_indices.device),
            dominant_patch_indices + 1,
        ]
    )
    dominant_tokens = hidden_states[:, cls_and_dominant_positions, :]

    all_positions = torch.arange(hidden_states.shape[1], device=hidden_states.device)
    filtered_mask = torch.ones(hidden_states.shape[1], dtype=torch.bool, device=hidden_states.device)
    filtered_mask[cls_and_dominant_positions] = False
    filtered_positions = all_positions[filtered_mask]
    contextual_tokens, contextual_patch_indices = _visionzip_contextual_tokens(
        hidden_states=hidden_states,
        metric=metric,
        filtered_positions=filtered_positions,
        contextual_count=contextual_count,
    )
    selected_hidden = torch.cat([dominant_tokens, contextual_tokens], dim=1)
    projected = llava_model.multi_modal_projector(selected_hidden)[0].to(device=output_device, dtype=output_dtype)
    coverage_indices = sorted(
        set(int(idx.item()) for idx in dominant_patch_indices)
        | set(int(idx.item()) for idx in contextual_patch_indices)
    )
    info = {
        "selector_ms": (time.perf_counter() - selector_start) * 1000.0,
        "dominant_patch_tokens": dominant_patch_count,
        "contextual_tokens": int(contextual_tokens.shape[1]),
        "contextual_merge_is_exhaustive": bool(contextual_tokens.shape[1] > 0),
    }
    return projected, coverage_indices, info


def _visionzip_contextual_tokens(*, hidden_states, metric, filtered_positions, contextual_count: int):
    import torch

    if contextual_count <= 0 or int(filtered_positions.numel()) == 0:
        empty_hidden = hidden_states[:, :0, :]
        empty_indices = torch.empty(0, dtype=torch.long, device=hidden_states.device)
        return empty_hidden, empty_indices

    contextual_count = min(contextual_count, int(filtered_positions.numel()))
    metric_filtered = metric[:, filtered_positions, :]
    hidden_filtered = hidden_states[:, filtered_positions, :]
    metric_normalized = metric_filtered / metric_filtered.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    step = max(1, int(metric_normalized.shape[1]) // contextual_count)
    target_indices = torch.arange(0, metric_normalized.shape[1], step, device=metric_normalized.device)[:contextual_count]
    target_hidden = hidden_filtered[:, target_indices, :]
    contextual_patch_indices = filtered_positions[target_indices] - 1

    all_filtered = torch.arange(metric_normalized.shape[1], device=metric_normalized.device)
    merge_mask = ~torch.isin(all_filtered, target_indices)
    if not bool(merge_mask.any()):
        return target_hidden, contextual_patch_indices

    tokens_to_merge = metric_normalized[:, merge_mask, :]
    target_tokens = metric_normalized[:, target_indices, :]
    similarity = torch.bmm(tokens_to_merge, target_tokens.transpose(1, 2))
    assign_one_hot = torch.zeros(
        tokens_to_merge.shape[0],
        tokens_to_merge.shape[1],
        contextual_count,
        dtype=hidden_filtered.dtype,
        device=metric_normalized.device,
    )
    assign_one_hot.scatter_(2, similarity.argmax(dim=2).unsqueeze(-1), 1)
    counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)
    hidden_to_merge = hidden_filtered[:, merge_mask, :]
    aggregated_hidden = torch.bmm(assign_one_hot.transpose(1, 2), hidden_to_merge) / counts
    return target_hidden + aggregated_hidden, contextual_patch_indices


def _visionzip_budget_split(num_patches: int, keep_count: int) -> tuple[int, int]:
    keep_count = max(1, min(int(keep_count), int(num_patches)))
    if keep_count <= 2:
        return max(1, keep_count - 1), 0
    contextual_count = max(1, int(round(keep_count * 10 / 64)))
    contextual_count = min(contextual_count, keep_count - 2)
    dominant_patch_count = keep_count - contextual_count - 1
    dominant_patch_count = max(1, min(dominant_patch_count, num_patches))
    contextual_count = max(0, min(contextual_count, num_patches - dominant_patch_count))
    return dominant_patch_count, contextual_count


def _clip_key_metric(vision_tower, layer_input):
    layer = vision_tower.vision_model.encoder.layers[-2]
    normed = layer.layer_norm1(layer_input)
    key_states = layer.self_attn.k_proj(normed)
    bsz, seq_len, _ = key_states.shape
    key_states = key_states.view(bsz, seq_len, layer.self_attn.num_heads, layer.self_attn.head_dim)
    return key_states.transpose(1, 2).mean(1)


def _replace_visual_span(*, inputs_embeds, input_ids, attention_mask, labels, image_positions, image_features):
    import torch

    if int(image_positions.numel()) == 0:
        raise ValueError("VisionZip pruning requires image placeholder tokens.")
    first = int(image_positions[0].item())
    last = int(image_positions[-1].item())
    expected = torch.arange(first, last + 1, device=image_positions.device, dtype=image_positions.dtype)
    if not torch.equal(image_positions, expected):
        raise ValueError("VisionZip pruning expects contiguous LLaVA image placeholder tokens.")

    image_features = image_features.unsqueeze(0).to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
    pruned_inputs_embeds = torch.cat(
        [inputs_embeds[:, :first, :], image_features, inputs_embeds[:, last + 1 :, :]],
        dim=1,
    )
    ignore_labels = torch.full(
        (labels.shape[0], image_features.shape[1]),
        -100,
        dtype=labels.dtype,
        device=labels.device,
    )
    pruned_labels = torch.cat([labels[:, :first], ignore_labels, labels[:, last + 1 :]], dim=1)
    if attention_mask is None:
        pruned_attention_mask = None
    else:
        image_attention = torch.ones(
            (attention_mask.shape[0], image_features.shape[1]),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        pruned_attention_mask = torch.cat(
            [attention_mask[:, :first], image_attention, attention_mask[:, last + 1 :]],
            dim=1,
        )
    return pruned_inputs_embeds, pruned_attention_mask, pruned_labels


def _llava_token_grid_shape(model, processor, num_visual_tokens: int) -> tuple[int, int]:
    image_size = getattr(getattr(model.config, "vision_config", None), "image_size", None)
    patch_size = getattr(getattr(model.config, "vision_config", None), "patch_size", None)
    if image_size and patch_size:
        side = int(image_size) // int(patch_size)
        if side > 0 and side * side == num_visual_tokens:
            return side, side

    crop_size = getattr(getattr(processor, "image_processor", None), "crop_size", None)
    if isinstance(crop_size, dict) and patch_size:
        height = int(crop_size.get("height", crop_size.get("shortest_edge", 0)) or 0)
        width = int(crop_size.get("width", crop_size.get("shortest_edge", 0)) or 0)
        rows = height // int(patch_size) if height else 0
        cols = width // int(patch_size) if width else 0
        if rows > 0 and cols > 0 and rows * cols == num_visual_tokens:
            return rows, cols

    side = int(math.sqrt(num_visual_tokens))
    if side * side == num_visual_tokens:
        return side, side
    rows = max(1, side)
    cols = int(math.ceil(num_visual_tokens / rows))
    return rows, cols


def _target_text_positions_llava(*, input_ids, labels, attention_mask, tokenizer, probe: dict[str, Any]):
    import torch

    target_text = _target_text_from_probe(probe)
    if not target_text:
        return torch.empty(0, dtype=torch.long, device=input_ids.device)
    prompt_mask = labels[0].eq(-100)
    if attention_mask is not None:
        prompt_mask = prompt_mask & attention_mask[0].bool()
    prompt_positions = torch.argwhere(prompt_mask).squeeze(1)
    prompt_ids = input_ids[0, prompt_positions].detach().cpu().tolist()
    matched: list[int] = []
    for token_ids in _target_token_id_candidates(tokenizer, target_text):
        for start in _subsequence_starts(prompt_ids, token_ids):
            matched.extend(int(prompt_positions[start + offset].item()) for offset in range(len(token_ids)))
        if matched:
            break
    if not matched:
        return torch.empty(0, dtype=torch.long, device=input_ids.device)
    return torch.tensor(sorted(set(matched)), dtype=torch.long, device=input_ids.device)


def _target_text_from_probe(probe: dict[str, Any]) -> str:
    for key in ("target_text", "target_answer", "source_text"):
        value = probe.get(key)
        if value is not None and str(value).strip():
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


def _evidence_relevance(token_boxes, evidence_regions):
    from recap.qwen_pruned_backend import _center_prior, _evidence_relevance as qwen_evidence_relevance

    if not evidence_regions:
        return _center_prior(token_boxes)
    return qwen_evidence_relevance(token_boxes, evidence_regions)


def _sync_tensor_device(tensor) -> None:
    import torch

    if getattr(tensor, "is_cuda", False):
        torch.cuda.synchronize(tensor.device)


def _empty_trace(probe: dict[str, Any], config: LlavaPruneConfig, *, elapsed_ms: float) -> dict[str, Any]:
    return {
        "sample_id": str(probe.get("sample_id", probe.get("id", ""))),
        "probe": str(probe.get("probe", "")),
        "selector": config.selector,
        "selector_impl": config.selector,
        "score_source": "",
        "position_mode": config.position_mode,
        "target_text_token_count": 0,
        "evidence_boost": float(config.evidence_boost),
        "budget_mode": "fixed",
        "target_keep_ratio": float(config.keep_ratio),
        "effective_keep_ratio": 1.0,
        "full_sequence_tokens": 0,
        "pruned_sequence_tokens": 0,
        "full_visual_tokens": 0,
        "kept_visual_tokens": 0,
        "removal_fraction": 0.0,
        "kept_indices": [],
        "has_evidence": False,
        "evidence_region_count": 0,
        "ecr": 0.0,
        "anchor_ecr": 0.0,
        "evidence_center_recall": 0.0,
        "evidence_patch_recall": 0.0,
        "grid_h": 0,
        "grid_w": 0,
        "vision_ms": 0.0,
        "target_text_ms": 0.0,
        "score_compute_ms": 0.0,
        "selector_ms": 0.0,
        "prune_materialize_ms": 0.0,
        "prune_overhead_ms": 0.0,
        "language_ms": elapsed_ms,
        "forward_ms": elapsed_ms,
    }


def _merge_trace_pair(probe: dict[str, Any], yes_trace: dict[str, Any], no_trace: dict[str, Any]) -> dict[str, Any]:
    trace = dict(yes_trace)
    trace["yes_forward_ms"] = float(yes_trace.get("forward_ms", 0.0))
    trace["no_forward_ms"] = float(no_trace.get("forward_ms", 0.0))
    trace["yes_vision_ms"] = float(yes_trace.get("vision_ms", 0.0))
    trace["no_vision_ms"] = float(no_trace.get("vision_ms", 0.0))
    trace["yes_language_ms"] = float(yes_trace.get("language_ms", 0.0))
    trace["no_language_ms"] = float(no_trace.get("language_ms", 0.0))
    trace["mean_forward_ms"] = (trace["yes_forward_ms"] + trace["no_forward_ms"]) / 2.0
    trace["mean_vision_ms"] = (trace["yes_vision_ms"] + trace["no_vision_ms"]) / 2.0
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
    trace["sample_id"] = str(probe.get("sample_id", trace.get("sample_id", "")))
    return trace


def _score_prune_fields(trace: dict[str, Any]) -> dict[str, Any]:
    return {
        "prune_selector": trace.get("selector", ""),
        "prune_score_source": trace.get("score_source", ""),
        "prune_position_mode": trace.get("position_mode", ""),
        "prune_budget_mode": trace.get("budget_mode", ""),
        "prune_keep_ratio": trace.get("effective_keep_ratio", 1.0),
        "prune_removal_fraction": trace.get("removal_fraction", 0.0),
        "prune_full_visual_tokens": trace.get("full_visual_tokens", 0),
        "prune_kept_visual_tokens": trace.get("kept_visual_tokens", 0),
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
