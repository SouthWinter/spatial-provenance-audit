"""Direct InternVL scorer with visual-token pruning before LLM prefill."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from PIL import Image
from tqdm import tqdm

from recap.images import probe_to_visual, validate_probe_images
from recap.internvl_direct_backend import (
    _load_internvl_direct,
    _model_dtype,
    _move_inputs,
)
from recap.prune.budgets import fixed_keep_count, removal_fraction
from recap.prune.metrics import (
    Box,
    evidence_center_recall,
    evidence_coverage,
    evidence_patch_recall,
    evidence_regions_from_sample,
    make_token_grid,
)
from recap.prune.positions import pruned_position_ids, validate_position_mode
from recap.prune.selectors import select_indices
from recap.qwen_pruned_backend import (
    _embedding_relevance_and_uniqueness,
    _evidence_relevance,
    _greedy_from_labels,
    _mean,
    _sanitize_kept_indices,
    _selector_impl,
    _target_text_positions,
)
from recap.scoring import score_probe


@dataclass(frozen=True)
class InternVLPruneConfig:
    selector: str
    keep_ratio: float
    seed: int = 13
    hybrid_core_ratio: float = 0.50
    hybrid_context_ratio: float = 0.25
    evidence_boost: float = 0.10
    kept_indices_by_sample: dict[str, list[int]] | None = None
    position_mode: str = "compact"

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_mode", validate_position_mode(self.position_mode))


def score_probes_with_internvl_pruned(
    probes: list[dict[str, Any]],
    *,
    pretrained: str,
    prune_config: InternVLPruneConfig,
    revision: str = "main",
    device: str = "cuda",
    device_map: str = "auto",
    dtype: str = "bfloat16",
    trust_remote_code: bool = False,
    low_cpu_mem_usage: bool = False,
    attn_implementation: str | None = None,
    min_patches: int = 1,
    max_patches: int = 12,
    target_delimiter: str = " ",
    debug_forward: bool = False,
    strict_images: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import torch

    image_report = validate_probe_images(probes)
    if strict_images and image_report["missing_visual_count"]:
        raise FileNotFoundError(f"Missing images for RECAP visual probes: {image_report}")
    if image_report["missing_visual_count"]:
        print(f"[RECAP-InternVLPruned] WARNING missing images for visual probes: {image_report}", flush=True)

    model, processor, input_device = _load_internvl_direct(
        pretrained=pretrained,
        revision=revision,
        device=device,
        device_map=device_map,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=low_cpu_mem_usage,
        attn_implementation=attn_implementation,
        torch_module=torch,
    )

    scored: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for probe in tqdm(probes, desc="RECAP-InternVLPruned probes"):
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
            min_patches=min_patches,
            max_patches=max_patches,
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
            min_patches=min_patches,
            max_patches=max_patches,
            debug_forward=debug_forward,
            torch_module=torch,
        )
        record = score_probe(probe, yes_loss=yes_loss, no_loss=no_loss)
        record["yes_is_greedy"] = bool(yes_greedy)
        record["no_is_greedy"] = bool(no_greedy)
        record["model"] = "internvl_pruned"
        record["pretrained"] = pretrained
        record.update(_score_prune_fields(yes_trace))
        scored.append(record)
        traces.append(_merge_trace_pair(probe, yes_trace, no_trace))

    return scored, traces


def summarize_internvl_prune_traces(traces: list[dict[str, Any]]) -> dict[str, float]:
    if not traces:
        return {
            "num_pruned_probes": 0.0,
            "mean_full_sequence_tokens": 0.0,
            "mean_pruned_sequence_tokens": 0.0,
            "mean_full_visual_tokens": 0.0,
            "mean_kept_visual_tokens": 0.0,
            "mean_keep_ratio": 0.0,
            "mean_removal_fraction": 0.0,
            "mean_num_image_patches": 0.0,
            "mean_ecr": 0.0,
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
        "mean_num_image_patches": _mean([float(t.get("num_image_patches", 0.0)) for t in traces]),
        "mean_ecr": _mean([float(t.get("ecr", 0.0)) for t in traces if t.get("has_evidence")]),
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
    prune_config: InternVLPruneConfig,
    min_patches: int,
    max_patches: int,
    debug_forward: bool,
    torch_module,
) -> tuple[float, bool, dict[str, Any]]:
    context = str(probe["question"]).replace("<image>", "")
    user_content: list[dict[str, Any]] = []
    for _ in visuals:
        user_content.append({"type": "image"})
    user_content.append({"type": "text", "text": context})

    prompt_messages = [{"role": "user", "content": user_content}]
    full_messages = prompt_messages + [{"role": "assistant", "content": continuation}]
    if debug_forward:
        print(
            f"[RECAP-InternVLPruned] scoring {probe.get('sample_id')}::{probe.get('probe')} continuation={continuation!r}",
            flush=True,
        )

    prompt_text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    full_text = processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
    image_inputs = visuals or None

    processor_kwargs = {"min_patches": min_patches, "max_patches": max_patches}
    inputs = processor(images=image_inputs, text=full_text, return_tensors="pt", **processor_kwargs)
    prompt_inputs = processor(images=image_inputs, text=prompt_text, return_tensors="pt", **processor_kwargs)

    inputs = _move_inputs(inputs, input_device, _model_dtype(model), torch_module)
    prompt_inputs = prompt_inputs.to(input_device)
    labels = inputs["input_ids"].clone()
    prompt_len = prompt_inputs["input_ids"].shape[1]
    labels[:, :prompt_len] = -100

    if not visuals or "pixel_values" not in inputs or inputs.get("pixel_values") is None:
        start = time.perf_counter()
        with torch_module.inference_mode():
            outputs = model(**inputs, labels=labels)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        trace = _empty_trace(probe, prune_config, elapsed_ms=elapsed_ms)
        return float(outputs["loss"].item()), _greedy_from_labels(outputs["logits"], labels), trace
    if len(visuals) != 1:
        raise ValueError("InternVL pruned scorer currently supports one image per probe.")

    start = time.perf_counter()
    with torch_module.inference_mode():
        logits, pruned_labels, trace = _forward_pruned_internvl(
            model=model,
            processor=processor,
            inputs=inputs,
            labels=labels,
            probe=probe,
            image=visuals[0],
            prune_config=prune_config,
            min_patches=min_patches,
            max_patches=max_patches,
        )
        loss = model.loss_function(
            logits=logits,
            labels=pruned_labels,
            vocab_size=model.config.text_config.vocab_size,
        )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    trace["forward_ms"] = elapsed_ms
    return float(loss.item()), _greedy_from_labels(logits, pruned_labels), trace


def _forward_pruned_internvl(
    *,
    model,
    processor,
    inputs,
    labels,
    probe: dict[str, Any],
    image: Image.Image,
    prune_config: InternVLPruneConfig,
    min_patches: int,
    max_patches: int,
):
    import torch

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    pixel_values = inputs["pixel_values"]
    if input_ids.shape[0] != 1:
        raise ValueError("InternVL pruned scorer currently expects batch_size=1.")

    internvl_model = model.model
    image_token_id = int(model.config.image_token_id)
    _sync_tensor_device(input_ids)
    vision_start = time.perf_counter()
    inputs_embeds = internvl_model.get_input_embeddings()(input_ids)
    image_features = internvl_model.get_image_features(
        pixel_values=pixel_values,
        vision_feature_layer=model.config.vision_feature_layer,
        vision_feature_select_strategy=model.config.vision_feature_select_strategy,
    ).to(inputs_embeds.device, inputs_embeds.dtype)
    special_image_mask = internvl_model.get_placeholder_mask(
        input_ids,
        inputs_embeds=inputs_embeds,
        image_features=image_features,
    )
    inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
    _sync_tensor_device(input_ids)
    vision_ms = (time.perf_counter() - vision_start) * 1000.0

    image_positions = torch.argwhere(input_ids[0].eq(image_token_id)).squeeze(1)
    num_visual_tokens = int(image_positions.numel())
    num_image_patches = int(image_features.shape[0])
    tokens_per_patch = int(image_features.shape[1])
    if num_visual_tokens != num_image_patches * tokens_per_patch:
        raise ValueError(
            f"Prompt has {num_visual_tokens} image tokens but vision tower returned "
            f"{num_image_patches}x{tokens_per_patch} features."
        )

    patch_grid_h, patch_grid_w = _internvl_patch_token_grid(model, tokens_per_patch)
    token_grid_h, token_grid_w = patch_grid_h, patch_grid_w * max(1, num_image_patches)
    token_boxes, tile_rows, tile_cols, has_thumbnail = _internvl_token_boxes(
        image=image,
        processor=processor,
        num_image_patches=num_image_patches,
        patch_grid_h=patch_grid_h,
        patch_grid_w=patch_grid_w,
        min_patches=min_patches,
        max_patches=max_patches,
    )
    if len(token_boxes) != num_visual_tokens:
        raise ValueError(f"Token geometry has {len(token_boxes)} boxes but prompt has {num_visual_tokens} image tokens.")

    evidence_regions = evidence_regions_from_sample(probe)
    selector_impl, score_source = _selector_impl(prune_config.selector)
    target_text_token_count = 0
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
    sample_id = str(probe.get("sample_id", probe.get("id", "")))
    sample_kept_indices = None
    if prune_config.kept_indices_by_sample is not None:
        sample_kept_indices = prune_config.kept_indices_by_sample.get(sample_id)
    if sample_kept_indices is not None:
        kept_indices = _sanitize_kept_indices(sample_kept_indices, num_visual_tokens)
        keep_count = len(kept_indices)
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
            salt=f"{sample_id}:{probe.get('probe', '')}:{prune_config.keep_ratio}:{prune_config.selector}",
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
    outputs = internvl_model.language_model(**language_kwargs)
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
        "patch_grid_h": patch_grid_h,
        "patch_grid_w": patch_grid_w,
        "num_image_patches": num_image_patches,
        "tile_rows": tile_rows,
        "tile_cols": tile_cols,
        "has_thumbnail_patch": has_thumbnail,
        "vision_ms": vision_ms,
        "target_text_ms": target_text_ms,
        "score_compute_ms": score_compute_ms,
        "selector_ms": selector_ms,
        "prune_materialize_ms": prune_materialize_ms,
        "prune_overhead_ms": score_compute_ms + selector_ms + prune_materialize_ms,
        "language_ms": language_ms,
    }
    return logits, pruned_labels, trace


def _internvl_patch_token_grid(model, tokens_per_patch: int) -> tuple[int, int]:
    vision_config = getattr(model.config, "vision_config", None)
    image_size = _pair_int(getattr(vision_config, "image_size", None), default=(448, 448))
    patch_size = _pair_int(getattr(vision_config, "patch_size", None), default=(14, 14))
    downsample_ratio = float(getattr(model.config, "downsample_ratio", 0.5) or 0.5)
    rows = int((image_size[0] // patch_size[0]) * downsample_ratio)
    cols = int((image_size[1] // patch_size[1]) * downsample_ratio)
    if rows > 0 and cols > 0 and rows * cols == tokens_per_patch:
        return rows, cols
    side = int(tokens_per_patch**0.5)
    if side * side == tokens_per_patch:
        return side, side
    return 1, tokens_per_patch


def _internvl_token_boxes(
    *,
    image: Image.Image,
    processor,
    num_image_patches: int,
    patch_grid_h: int,
    patch_grid_w: int,
    min_patches: int,
    max_patches: int,
) -> tuple[list[Box], int, int, bool]:
    if num_image_patches <= 1:
        return make_token_grid(patch_grid_h, patch_grid_w), 1, 1, False

    tile_cols, tile_rows = _infer_tile_canvas(
        image=image,
        processor=processor,
        min_patches=min_patches,
        max_patches=max_patches,
    )
    num_tile_patches = tile_rows * tile_cols
    has_thumbnail = num_image_patches == num_tile_patches + 1 and num_tile_patches > 1
    if num_image_patches not in {num_tile_patches, num_tile_patches + 1}:
        tile_rows, tile_cols = 1, num_image_patches
        num_tile_patches = num_image_patches
        has_thumbnail = False

    local_boxes = make_token_grid(patch_grid_h, patch_grid_w)
    boxes: list[Box] = []
    for patch_idx in range(num_tile_patches):
        row = patch_idx // tile_cols
        col = patch_idx % tile_cols
        x1 = col / tile_cols
        y1 = row / tile_rows
        x2 = (col + 1) / tile_cols
        y2 = (row + 1) / tile_rows
        boxes.extend(_map_local_boxes(local_boxes, (x1, y1, x2, y2)))
    if has_thumbnail:
        boxes.extend(local_boxes)
    return boxes, tile_rows, tile_cols, has_thumbnail


def _infer_tile_canvas(*, image: Image.Image, processor, min_patches: int, max_patches: int) -> tuple[int, int]:
    try:
        from transformers.models.got_ocr2.image_processing_got_ocr2_fast import get_optimal_tiled_canvas

        image_processor = getattr(processor, "image_processor", None)
        size = getattr(image_processor, "size", None)
        patch_h, patch_w = _size_hw(size, default=(448, 448))
        width, height = image.size
        cols, rows = get_optimal_tiled_canvas((height, width), (patch_h, patch_w), int(min_patches), int(max_patches))
        return max(1, int(cols)), max(1, int(rows))
    except Exception:
        return 1, max(1, int(max_patches))


def _map_local_boxes(local_boxes: list[Box], outer: Box) -> list[Box]:
    ox1, oy1, ox2, oy2 = outer
    ow = ox2 - ox1
    oh = oy2 - oy1
    return [
        (
            ox1 + box[0] * ow,
            oy1 + box[1] * oh,
            ox1 + box[2] * ow,
            oy1 + box[3] * oh,
        )
        for box in local_boxes
    ]


def _pair_int(value: Any, *, default: tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, dict):
        return int(value.get("height", default[0])), int(value.get("width", default[1]))
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return int(value[0]), int(value[1])
    if isinstance(value, int):
        return value, value
    return default


def _size_hw(value: Any, *, default: tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, dict):
        return int(value.get("height", default[0])), int(value.get("width", default[1]))
    height = getattr(value, "height", None)
    width = getattr(value, "width", None)
    if height is not None and width is not None:
        return int(height), int(width)
    return _pair_int(value, default=default)


def _sync_tensor_device(tensor) -> None:
    try:
        import torch

        if getattr(tensor.device, "type", "") == "cuda":
            torch.cuda.synchronize(tensor.device)
    except Exception:
        pass


def _empty_trace(probe: dict[str, Any], config: InternVLPruneConfig, *, elapsed_ms: float) -> dict[str, Any]:
    return {
        "sample_id": str(probe.get("sample_id", probe.get("id", ""))),
        "probe": str(probe.get("probe", "")),
        "selector": config.selector,
        "selector_impl": _selector_impl(config.selector)[0],
        "score_source": _selector_impl(config.selector)[1],
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
        "evidence_center_recall": 0.0,
        "evidence_patch_recall": 0.0,
        "grid_h": 0,
        "grid_w": 0,
        "num_image_patches": 0,
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
    trace["probe"] = str(probe.get("probe", trace.get("probe", "")))
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
        "prune_num_image_patches": trace.get("num_image_patches", 0),
        "prune_ecr": trace.get("ecr", 0.0),
        "prune_evidence_center_recall": trace.get("evidence_center_recall", 0.0),
        "prune_evidence_patch_recall": trace.get("evidence_patch_recall", 0.0),
        "prune_target_text_ms": trace.get("target_text_ms", 0.0),
        "prune_score_compute_ms": trace.get("score_compute_ms", 0.0),
        "prune_selector_ms": trace.get("selector_ms", 0.0),
        "prune_materialize_ms": trace.get("prune_materialize_ms", 0.0),
        "prune_overhead_ms": trace.get("prune_overhead_ms", 0.0),
    }
