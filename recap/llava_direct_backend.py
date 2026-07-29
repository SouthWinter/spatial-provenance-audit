"""Direct LLaVA HF scorer for RECAP probes.

The progress bar is per RECAP probe, matching the Qwen direct backend, while
each probe still scores both yes and no continuations.
"""

from __future__ import annotations

from typing import Any

from PIL import Image
from tqdm import tqdm

from recap.images import probe_to_visual, validate_probe_images
from recap.scoring import score_probe

DEFAULT_IMAGE_TOKEN = "<image>"
VICUNA_CHAT_TEMPLATE = "{% for message in messages %}{% if loop.index0 == 0 %}A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {{ message['content'] }} {% elif message['role'] == 'user' %}USER: {{ message['content'] }} {% else %} ASSISTANT: {{ message['content'] }}{{ eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"


def score_probes_with_llava_direct(
    probes: list[dict[str, Any]],
    *,
    pretrained: str,
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
) -> list[dict[str, Any]]:
    import torch

    image_report = validate_probe_images(probes)
    if strict_images and image_report["missing_visual_count"]:
        raise FileNotFoundError(f"Missing images for RECAP visual probes: {image_report}")
    if image_report["missing_visual_count"]:
        print(f"[RECAP-LLaVADirect] WARNING missing images for visual probes: {image_report}", flush=True)

    model, processor, input_device = _load_llava_direct(
        pretrained=pretrained,
        revision=revision,
        device=device,
        device_map=device_map,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
        torch_module=torch,
    )

    scored: list[dict[str, Any]] = []
    for probe in tqdm(probes, desc="RECAP-LLaVADirect probes"):
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
            chat_template=chat_template,
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
            chat_template=chat_template,
            debug_forward=debug_forward,
            torch_module=torch,
        )
        record = score_probe(probe, yes_loss=yes_loss, no_loss=no_loss)
        record["yes_is_greedy"] = bool(yes_greedy)
        record["no_is_greedy"] = bool(no_greedy)
        record["model"] = "llava_direct"
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
    chat_template: str | None,
    debug_forward: bool,
    torch_module,
) -> tuple[float, bool]:
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
            f"[RECAP-LLaVADirect] scoring {probe.get('sample_id')}::{probe.get('probe')} continuation={continuation!r}",
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

    with torch_module.inference_mode():
        outputs = model(**model_inputs, labels=labels)

    logits = outputs["logits"]
    continuation_tokens = model_inputs["input_ids"][:, prompt_len:]
    if continuation_tokens.numel() == 0:
        is_greedy = False
    else:
        greedy_tokens = logits[:, prompt_len - 1 : -1].argmax(dim=-1)
        is_greedy = bool((greedy_tokens == continuation_tokens).all().item())
    return float(outputs["loss"].item()), is_greedy


def _format_llava_prompt(
    *,
    processor,
    context: str,
    continuation: str,
    num_images: int,
    chat_template: str | None,
) -> tuple[str, str]:
    tokenizer = processor.tokenizer
    if chat_template is not None:
        tokenizer.chat_template = chat_template
        return _format_llava_text_prompt(
            tokenizer=tokenizer,
            context=context,
            continuation=continuation,
            num_images=num_images,
        )

    if hasattr(processor, "apply_chat_template"):
        content: list[dict[str, str]] = [{"type": "image"} for _ in range(num_images)]
        content.append({"type": "text", "text": context})
        prompt_messages = [{"role": "user", "content": content}]
        full_messages = prompt_messages + [{"role": "assistant", "content": [{"type": "text", "text": continuation}]}]
        try:
            prompt = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            prompt_and_continuation = processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
            if num_images == 0 or DEFAULT_IMAGE_TOKEN in prompt_and_continuation:
                return prompt, prompt_and_continuation
        except Exception:
            pass

    if tokenizer.chat_template is None:
        tokenizer.chat_template = VICUNA_CHAT_TEMPLATE
    return _format_llava_text_prompt(
        tokenizer=tokenizer,
        context=context,
        continuation=continuation,
        num_images=num_images,
    )


def _format_llava_text_prompt(*, tokenizer, context: str, continuation: str, num_images: int) -> tuple[str, str]:
    if num_images:
        image_tokens = " ".join([DEFAULT_IMAGE_TOKEN] * num_images)
        context = f"{image_tokens}\n{context}"
    messages = [{"role": "user", "content": context}, {"role": "assistant", "content": continuation}]
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    prompt_and_continuation = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return prompt, prompt_and_continuation


def _load_llava_direct(
    *,
    pretrained: str,
    revision: str,
    device: str,
    device_map: str,
    dtype: str,
    trust_remote_code: bool,
    attn_implementation: str | None,
    torch_module,
):
    from transformers import AutoConfig, AutoProcessor

    model_cls = _resolve_llava_model_class(pretrained, trust_remote_code=trust_remote_code)
    model_kwargs: dict[str, Any] = {
        "revision": revision,
        "torch_dtype": _resolve_dtype(dtype, torch_module),
        "trust_remote_code": trust_remote_code,
    }
    normalized_device_map = str(device_map).lower()
    if normalized_device_map not in {"", "none", "null", "disabled"}:
        model_kwargs["device_map"] = device_map
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    print(f"[RECAP-LLaVADirect] Loading {model_cls.__name__} from {pretrained} with kwargs={model_kwargs}", flush=True)
    model = model_cls.from_pretrained(pretrained, **model_kwargs)
    model.eval()
    if "device_map" not in model_kwargs:
        print(f"[RECAP-LLaVADirect] Moving model to {device}", flush=True)
        model.to(device)

    print("[RECAP-LLaVADirect] Loading processor", flush=True)
    processor = AutoProcessor.from_pretrained(pretrained, revision=revision, trust_remote_code=trust_remote_code)
    _configure_llava_processor(processor, model.config)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    input_device = _resolve_input_device(model, device)
    print(f"[RECAP-LLaVADirect] Input tensors will be moved to {input_device}", flush=True)
    return model, processor, input_device


def _configure_llava_processor(processor, config) -> None:
    vision_config = getattr(config, "vision_config", None)
    patch_size = getattr(vision_config, "patch_size", None)
    if patch_size is not None and getattr(processor, "patch_size", None) is None:
        processor.patch_size = patch_size
    feature_strategy = getattr(config, "vision_feature_select_strategy", None)
    if feature_strategy is not None and getattr(processor, "vision_feature_select_strategy", None) is None:
        processor.vision_feature_select_strategy = feature_strategy
    additional_tokens = getattr(config, "num_additional_image_tokens", None)
    if additional_tokens is not None and getattr(processor, "num_additional_image_tokens", None) is None:
        processor.num_additional_image_tokens = additional_tokens


def _resolve_llava_model_class(pretrained: str, *, trust_remote_code: bool):
    from transformers import AutoConfig
    import transformers

    config = AutoConfig.from_pretrained(pretrained, trust_remote_code=trust_remote_code)
    model_type = str(getattr(config, "model_type", "llava")).lower()
    candidates = {
        "llava": "LlavaForConditionalGeneration",
        "llava_next": "LlavaNextForConditionalGeneration",
        "llava_onevision": "LlavaOnevisionForConditionalGeneration",
    }
    class_name = candidates.get(model_type)
    if class_name:
        model_cls = getattr(transformers, class_name, None)
        if model_cls is not None:
            return model_cls
    fallback = getattr(transformers, "AutoModelForImageTextToText", None) or getattr(transformers, "AutoModelForVision2Seq", None)
    if fallback is not None:
        return fallback
    raise ImportError(f"Could not resolve a LLaVA-compatible model class for model_type={model_type!r}.")


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
