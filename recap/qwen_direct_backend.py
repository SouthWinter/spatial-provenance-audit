"""Direct Qwen-VL scorer for RECAP probes.

This backend directly loads a Hugging Face checkpoint and only implements the
yes/no loglikelihood calls required by RECAP. The loader supports Qwen2.5-VL and
Qwen3-VL when the installed Transformers version provides the corresponding
model class.
"""

from __future__ import annotations

from typing import Any

from PIL import Image
from tqdm import tqdm

from recap.images import encode_image_data_url, probe_to_visual, validate_probe_images
from recap.scoring import score_probe


def score_probes_with_qwen_direct(
    probes: list[dict[str, Any]],
    *,
    pretrained: str,
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
) -> list[dict[str, Any]]:
    import torch
    image_report = validate_probe_images(probes)
    if strict_images and image_report["missing_visual_count"]:
        raise FileNotFoundError(f"Missing images for RECAP visual probes: {image_report}")
    if image_report["missing_visual_count"]:
        print(f"[RECAP-QwenDirect] WARNING missing images for visual probes: {image_report}", flush=True)

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
    for probe in tqdm(probes, desc="RECAP-QwenDirect probes"):
        yes_loss, yes_greedy = _score_continuation(
            probe,
            f"{target_delimiter}yes",
            model=model,
            processor=processor,
            process_vision_info=process_vision_info,
            input_device=input_device,
            system_prompt=system_prompt,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            debug_forward=debug_forward,
            strict_images=strict_images,
        )
        no_loss, no_greedy = _score_continuation(
            probe,
            f"{target_delimiter}no",
            model=model,
            processor=processor,
            process_vision_info=process_vision_info,
            input_device=input_device,
            system_prompt=system_prompt,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            debug_forward=debug_forward,
            strict_images=strict_images,
        )
        record = score_probe(probe, yes_loss=yes_loss, no_loss=no_loss)
        record["yes_is_greedy"] = bool(yes_greedy)
        record["no_is_greedy"] = bool(no_greedy)
        record["model"] = "qwen_vl_direct"
        record["pretrained"] = pretrained
        scored.append(record)

    return scored


def _score_continuation(
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
    debug_forward: bool,
    strict_images: bool = False,
) -> tuple[float, bool]:
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
        print(f"[RECAP-QwenDirect] scoring {probe.get('sample_id')}::{probe.get('probe')} continuation={continuation!r}", flush=True)
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

    with torch.inference_mode():
        outputs = model(**inputs, labels=labels)

    logits = outputs["logits"]
    continuation_tokens = inputs["input_ids"][:, prompt_len:]
    if continuation_tokens.numel() == 0:
        is_greedy = False
    else:
        greedy_tokens = logits[:, prompt_len - 1 : -1].argmax(dim=-1)
        is_greedy = bool((greedy_tokens == continuation_tokens).all().item())
    return float(outputs["loss"].item()), is_greedy


def _load_qwen_direct(
    *,
    pretrained: str,
    device: str,
    device_map: str,
    dtype: str,
    attn_implementation: str | None,
    min_pixels: int,
    max_pixels: int,
    use_fast_processor: bool,
    torch_module,
):
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, AutoTokenizer

    model_kwargs: dict[str, Any] = {"dtype": _resolve_dtype(dtype, torch_module)}
    normalized_device_map = str(device_map).lower()
    if normalized_device_map not in {"none", "null", "disabled"}:
        model_kwargs["device_map"] = device_map
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    model_cls = _resolve_qwen_model_class(pretrained)
    print(
        f"[RECAP-QwenDirect] Loading {model_cls.__name__} from {pretrained} with kwargs={model_kwargs}",
        flush=True,
    )
    model = model_cls.from_pretrained(pretrained, **model_kwargs)
    print("[RECAP-QwenDirect] from_pretrained returned", flush=True)
    model.eval()
    print("[RECAP-QwenDirect] eval mode set", flush=True)
    if "device_map" not in model_kwargs:
        print(f"[RECAP-QwenDirect] Moving model to {device}", flush=True)
        model.to(device)
        print(f"[RECAP-QwenDirect] Model moved to {device}", flush=True)

    print(f"[RECAP-QwenDirect] Loading processor/tokenizer with use_fast={use_fast_processor}", flush=True)
    processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels, use_fast=use_fast_processor)
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    print(f"[RECAP-QwenDirect] Processor/tokenizer loaded: {type(processor).__name__}, {type(tokenizer).__name__}", flush=True)

    input_device = _resolve_input_device(model, device)
    print(f"[RECAP-QwenDirect] Input tensors will be moved to {input_device}", flush=True)
    return model, processor, tokenizer, process_vision_info, input_device


def _resolve_qwen_model_class(pretrained: str):
    import transformers

    normalized = str(pretrained).lower()
    candidates: list[str] = []
    if "qwen3" in normalized:
        candidates.extend(["Qwen3VLForConditionalGeneration", "Qwen3_VLForConditionalGeneration"])
    candidates.extend(
        [
            "Qwen2_5_VLForConditionalGeneration",
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
        ]
    )
    missing: list[str] = []
    for name in candidates:
        model_cls = getattr(transformers, name, None)
        if model_cls is not None:
            return model_cls
        missing.append(name)
    raise ImportError(
        "Could not find a supported Qwen-VL model class in transformers. "
        f"Tried: {', '.join(missing)}. "
        "For Qwen3-VL, upgrade transformers on the run machine to a version "
        "that exports Qwen3VLForConditionalGeneration."
    )


def _resolve_dtype(dtype: str, torch_module):
    normalized = str(dtype).lower()
    if normalized == "auto":
        return "auto"
    if normalized in {"bf16", "bfloat16"}:
        return torch_module.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch_module.float16
    if normalized in {"fp32", "float32"}:
        return torch_module.float32
    raise ValueError(f"Unsupported dtype '{dtype}'. Use auto, bfloat16, float16, or float32.")


def _resolve_input_device(model, fallback: str):
    try:
        device = model.device
        if getattr(device, "type", None) != "meta":
            return device
    except Exception:
        pass
    for parameter in model.parameters():
        if getattr(parameter.device, "type", None) != "meta":
            return parameter.device
    return fallback


