"""Direct InternVL HF scorer for RECAP probes.

This file implements the small yes/no likelihood path RECAP needs using the
native HF ``InternVLForConditionalGeneration`` interface.
"""

from __future__ import annotations

from typing import Any

from PIL import Image
from tqdm import tqdm

from recap.images import probe_to_visual, validate_probe_images
from recap.scoring import score_probe


def score_probes_with_internvl_direct(
    probes: list[dict[str, Any]],
    *,
    pretrained: str,
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
) -> list[dict[str, Any]]:
    import torch

    image_report = validate_probe_images(probes)
    if strict_images and image_report["missing_visual_count"]:
        raise FileNotFoundError(f"Missing images for RECAP visual probes: {image_report}")
    if image_report["missing_visual_count"]:
        print(f"[RECAP-InternVLDirect] WARNING missing images for visual probes: {image_report}", flush=True)

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
    for probe in tqdm(probes, desc="RECAP-InternVLDirect probes"):
        visuals = probe_to_visual(probe, strict=strict_images)
        if not isinstance(visuals, list):
            visuals = [visuals]
        visuals = [visual for visual in visuals if isinstance(visual, Image.Image)]

        yes_loss, yes_greedy = _score_continuation(
            probe,
            visuals,
            f"{target_delimiter}yes",
            model=model,
            processor=processor,
            input_device=input_device,
            min_patches=min_patches,
            max_patches=max_patches,
            debug_forward=debug_forward,
            torch_module=torch,
        )
        no_loss, no_greedy = _score_continuation(
            probe,
            visuals,
            f"{target_delimiter}no",
            model=model,
            processor=processor,
            input_device=input_device,
            min_patches=min_patches,
            max_patches=max_patches,
            debug_forward=debug_forward,
            torch_module=torch,
        )
        record = score_probe(probe, yes_loss=yes_loss, no_loss=no_loss)
        record["yes_is_greedy"] = bool(yes_greedy)
        record["no_is_greedy"] = bool(no_greedy)
        record["model"] = "internvl_direct"
        record["pretrained"] = pretrained
        scored.append(record)

    return scored


def _score_continuation(
    probe: dict[str, Any],
    visuals: list[Image.Image],
    continuation: str,
    *,
    model,
    processor,
    input_device,
    min_patches: int,
    max_patches: int,
    debug_forward: bool,
    torch_module,
) -> tuple[float, bool]:
    context = str(probe["question"]).replace("<image>", "")
    user_content: list[dict[str, Any]] = []
    for _ in visuals:
        user_content.append({"type": "image"})
    user_content.append({"type": "text", "text": context})

    prompt_messages = [{"role": "user", "content": user_content}]
    full_messages = prompt_messages + [{"role": "assistant", "content": continuation}]

    if debug_forward:
        print(
            f"[RECAP-InternVLDirect] scoring {probe.get('sample_id')}::{probe.get('probe')} continuation={continuation!r}",
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

    with torch_module.inference_mode():
        outputs = model(**inputs, labels=labels)

    logits = outputs["logits"]
    continuation_tokens = inputs["input_ids"][:, prompt_len:]
    if continuation_tokens.numel() == 0:
        is_greedy = False
    else:
        greedy_tokens = logits[:, prompt_len - 1 : -1].argmax(dim=-1)
        is_greedy = bool((greedy_tokens == continuation_tokens).all().item())
    return float(outputs["loss"].item()), is_greedy


def _load_internvl_direct(
    *,
    pretrained: str,
    revision: str,
    device: str,
    device_map: str,
    dtype: str,
    trust_remote_code: bool,
    low_cpu_mem_usage: bool,
    attn_implementation: str | None,
    torch_module,
):
    from transformers import AutoProcessor

    model_cls = _resolve_internvl_model_class()
    model_kwargs: dict[str, Any] = {
        "revision": revision,
        "torch_dtype": _resolve_dtype(dtype, torch_module),
        "low_cpu_mem_usage": low_cpu_mem_usage,
        "trust_remote_code": trust_remote_code,
    }
    normalized_device_map = str(device_map).lower()
    if normalized_device_map not in {"", "none", "null", "disabled"}:
        model_kwargs["device_map"] = device_map
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    print(f"[RECAP-InternVLDirect] Loading {model_cls.__name__} from {pretrained} with kwargs={model_kwargs}", flush=True)
    model = model_cls.from_pretrained(pretrained, **model_kwargs).eval()
    if "device_map" not in model_kwargs:
        print(f"[RECAP-InternVLDirect] Moving model to {device}", flush=True)
        model.to(device)

    print("[RECAP-InternVLDirect] Loading processor", flush=True)
    processor = AutoProcessor.from_pretrained(pretrained, revision=revision, trust_remote_code=trust_remote_code)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    input_device = _resolve_input_device(model, device)
    print(f"[RECAP-InternVLDirect] Input tensors will be moved to {input_device}", flush=True)
    return model, processor, input_device


def _resolve_internvl_model_class():
    import transformers

    model_cls = getattr(transformers, "InternVLForConditionalGeneration", None)
    if model_cls is not None:
        return model_cls
    fallback = getattr(transformers, "AutoModelForImageTextToText", None) or getattr(transformers, "AutoModelForVision2Seq", None)
    if fallback is not None:
        return fallback
    raise ImportError(
        "Could not find InternVLForConditionalGeneration in transformers. "
        "Upgrade transformers on the run machine to the version used by the InternVL3.5 HF wrapper."
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


def _model_dtype(model):
    dtype = getattr(model, "dtype", None)
    if dtype is not None:
        return dtype
    for parameter in model.parameters():
        return parameter.dtype
    return None


def _move_inputs(inputs, device, dtype, torch_module):
    inputs = inputs.to(device)
    if dtype is None:
        return inputs
    for key, value in list(inputs.items()):
        if torch_module.is_tensor(value) and torch_module.is_floating_point(value):
            inputs[key] = value.to(dtype=dtype)
    return inputs
