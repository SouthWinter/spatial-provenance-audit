#!/usr/bin/env python
"""Run native OCRBench open-answer generation for Qwen full vs pruned prefixes."""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recap.images import encode_image_data_url, probe_to_visual
from recap.io import write_json, write_jsonl
from recap.prune.budgets import fixed_keep_count, removal_fraction
from recap.prune.metrics import evidence_regions_from_sample, make_token_grid
from recap.prune.selectors import select_indices
from recap.prune.saturation import SaturationConfig, evidence_saturation_decision
from recap.qwen_direct_backend import _load_qwen_direct
from recap.qwen_pruned_backend import (
    PruneConfig,
    _embedding_relevance_and_uniqueness,
    _effective_keep_ratio,
    _selector_impl,
    _sync_tensor_device,
    _target_text_positions,
    _token_grid_shape,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/ocrbench_yesno_probes_100img.jsonl")
    parser.add_argument("--work-dir", default="runs/ocrbench_open_answer/qwen3_8b_open_answer_generate_target_grid0p30")
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--selector", default="target_embed_grid_topk")
    parser.add_argument("--keep-ratio", type=float, default=0.30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--attn-implementation", default="")
    parser.add_argument("--min-pixels", type=int, default=50176)
    parser.add_argument("--max-pixels", type=int, default=802816)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--use-fast-processor", action="store_true")
    args = parser.parse_args()

    samples = build_open_answer_samples(read_jsonl(args.input))
    if args.limit is not None:
        samples = samples[: args.limit]

    import torch

    model, processor, tokenizer, process_vision_info, input_device = _load_qwen_direct(
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
    prune_config = PruneConfig(selector=args.selector, keep_ratio=args.keep_ratio)

    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for sample in tqdm(samples, desc="OCRBench open generation"):
        probe = selector_probe(sample)
        full_answer = generate_full(
            probe,
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            process_vision_info=process_vision_info,
            input_device=input_device,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            max_new_tokens=args.max_new_tokens,
        )
        pruned_answer, trace = generate_pruned(
            probe,
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            process_vision_info=process_vision_info,
            input_device=input_device,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            max_new_tokens=args.max_new_tokens,
            prune_config=prune_config,
        )
        gold_answers = sample["gold_answers"]
        full_metrics = score_answer(full_answer, gold_answers)
        pruned_metrics = score_answer(pruned_answer, gold_answers)
        row = {
            **sample,
            "full_answer": full_answer,
            "pruned_answer": pruned_answer,
            "full_exact": full_metrics["exact"],
            "full_contains": full_metrics["contains"],
            "full_anls": full_metrics["anls"],
            "pruned_exact": pruned_metrics["exact"],
            "pruned_contains": pruned_metrics["contains"],
            "pruned_anls": pruned_metrics["anls"],
            "exact_delta_pruned_minus_full": pruned_metrics["exact"] - full_metrics["exact"],
            "anls_delta_pruned_minus_full": pruned_metrics["anls"] - full_metrics["anls"],
            "selector": args.selector,
            "keep_ratio": args.keep_ratio,
            "effective_keep_ratio": float(trace.get("effective_keep_ratio", 0.0) or 0.0),
            "target_text_token_count": int(trace.get("target_text_token_count", 0) or 0),
            "selector_target_source": "question_only",
        }
        rows.append(row)
        traces.append(trace)

    metrics = summarize(rows, args=args)
    out_dir = Path(args.work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "open_answer_generation.jsonl", rows)
    write_jsonl(out_dir / "prune_traces.jsonl", traces)
    write_json(out_dir / "metrics.json", metrics)
    write_markdown(out_dir / "open_answer_generation_report.md", metrics, rows)
    print(
        f"Wrote {len(rows)} native open-answer rows to {out_dir}; "
        f"full_exact={metrics['full_exact']:.4f}, pruned_exact={metrics['pruned_exact']:.4f}, "
        f"full_anls={metrics['full_anls']:.4f}, pruned_anls={metrics['pruned_anls']:.4f}"
    )


def generate_full(
    probe: dict[str, Any],
    *,
    model,
    processor,
    tokenizer,
    process_vision_info,
    input_device,
    min_pixels: int,
    max_pixels: int,
    max_new_tokens: int,
) -> str:
    import torch

    messages = prompt_messages(probe, min_pixels=min_pixels, max_pixels=max_pixels)
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=prompt_text, images=image_inputs, videos=video_inputs, return_tensors="pt").to(input_device)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=getattr(tokenizer, "eos_token_id", None),
        )
    generated = output_ids[:, inputs["input_ids"].shape[1] :]
    return decode_answer(processor, generated[0])


def generate_pruned(
    probe: dict[str, Any],
    *,
    model,
    processor,
    tokenizer,
    process_vision_info,
    input_device,
    min_pixels: int,
    max_pixels: int,
    max_new_tokens: int,
    prune_config: PruneConfig,
) -> tuple[str, dict[str, Any]]:
    import torch

    messages = prompt_messages(probe, min_pixels=min_pixels, max_pixels=max_pixels)
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=prompt_text, images=image_inputs, videos=video_inputs, return_tensors="pt").to(input_device)
    labels = torch.full_like(inputs["input_ids"], -100)

    qwen_model = model.model
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    pixel_values = inputs["pixel_values"]
    image_grid_thw = inputs["image_grid_thw"]
    inputs_embeds = qwen_model.get_input_embeddings()(input_ids)
    image_embeds_list, deepstack_image_embeds = qwen_model.get_image_features(pixel_values, image_grid_thw)
    image_embeds = torch.cat(image_embeds_list, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
    image_mask, _ = qwen_model.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    image_positions = torch.argwhere((input_ids[0] == model.config.image_token_id)).squeeze(1)
    num_visual_tokens = int(image_positions.numel())
    token_grid_h, token_grid_w = _token_grid_shape(image_grid_thw[0], qwen_model.config.vision_config.spatial_merge_size)
    token_boxes = make_token_grid(token_grid_h, token_grid_w)
    evidence_regions = evidence_regions_from_sample(probe)
    selector_impl, score_source = _selector_impl(prune_config.selector)
    target_text_token_count = 0
    if score_source in {"embedding", "target_embedding"}:
        text_positions_override = None
        if score_source == "target_embedding":
            text_positions_override = _target_text_positions(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                tokenizer=getattr(processor, "tokenizer", processor),
                probe=probe,
            )
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
        )
    else:
        relevance = [0.0 for _ in token_boxes]
        uniqueness = [0.0 for _ in token_boxes]

    effective_keep_ratio = _effective_keep_ratio(prune_config, None, None)
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
        salt=f"{probe.get('sample_id', probe.get('id', ''))}:open-generate:{effective_keep_ratio}:{prune_config.selector}",
        hybrid_core_ratio=prune_config.hybrid_core_ratio,
        hybrid_context_ratio=prune_config.hybrid_context_ratio,
        evidence_boost=prune_config.evidence_boost,
    )

    keep_sequence = torch.ones(input_ids.shape[1], dtype=torch.bool, device=input_ids.device)
    keep_sequence[image_positions.to(input_ids.device)] = False
    kept_tensor = torch.tensor(kept_indices, dtype=torch.long, device=input_ids.device)
    keep_sequence[image_positions.to(input_ids.device)[kept_tensor]] = True
    pruned_inputs_embeds = inputs_embeds[:, keep_sequence, :]
    pruned_attention_mask = attention_mask[:, keep_sequence] if attention_mask is not None else None
    position_ids, _ = qwen_model.get_rope_index(input_ids, image_grid_thw, attention_mask=attention_mask)
    pruned_position_ids = position_ids[:, :, keep_sequence]
    visual_pos_masks = (input_ids == model.config.image_token_id)[:, keep_sequence]
    pruned_deepstack = [
        layer_embeds[kept_tensor.to(layer_embeds.device)].to(pruned_inputs_embeds.device)
        for layer_embeds in deepstack_image_embeds
    ]

    generated: list[int] = []
    cache_position = torch.arange(pruned_inputs_embeds.shape[1], device=pruned_inputs_embeds.device)
    with torch.inference_mode():
        outputs = qwen_model.language_model(
            input_ids=None,
            position_ids=pruned_position_ids,
            attention_mask=pruned_attention_mask,
            past_key_values=None,
            inputs_embeds=pruned_inputs_embeds,
            cache_position=cache_position,
            use_cache=True,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=pruned_deepstack,
        )
        logits = model.lm_head(outputs.last_hidden_state)
        past_key_values = outputs.past_key_values
        next_token = logits[:, -1, :].argmax(dim=-1)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        for step in range(max_new_tokens):
            token_id = int(next_token.item())
            if eos_token_id is not None and token_id == int(eos_token_id):
                break
            generated.append(token_id)
            next_inputs_embeds = qwen_model.get_input_embeddings()(next_token[:, None])
            if pruned_attention_mask is None:
                step_attention_mask = None
            else:
                step_attention_mask = torch.cat(
                    [
                        pruned_attention_mask,
                        torch.ones((1, step + 1), dtype=pruned_attention_mask.dtype, device=pruned_attention_mask.device),
                    ],
                    dim=1,
                )
            step_cache_position = torch.tensor(
                [pruned_inputs_embeds.shape[1] + step],
                dtype=torch.long,
                device=pruned_inputs_embeds.device,
            )
            step_position_ids = pruned_position_ids[:, :, -1:] + (step + 1)
            step_outputs = qwen_model.language_model(
                input_ids=None,
                position_ids=step_position_ids,
                attention_mask=step_attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=next_inputs_embeds,
                cache_position=step_cache_position,
                use_cache=True,
                visual_pos_masks=None,
                deepstack_visual_embeds=None,
            )
            step_logits = model.lm_head(step_outputs.last_hidden_state)
            past_key_values = step_outputs.past_key_values
            next_token = step_logits[:, -1, :].argmax(dim=-1)

    _sync_tensor_device(pruned_inputs_embeds)
    trace = {
        "sample_id": str(probe.get("sample_id", probe.get("id", ""))),
        "selector": prune_config.selector,
        "score_source": score_source,
        "target_text_token_count": target_text_token_count,
        "target_keep_ratio": float(prune_config.keep_ratio),
        "effective_keep_ratio": float(effective_keep_ratio),
        "full_visual_tokens": num_visual_tokens,
        "kept_visual_tokens": len(kept_indices),
        "removal_fraction": removal_fraction(num_visual_tokens, len(kept_indices)),
        "kept_indices": kept_indices,
        "budget_mode": prune_config.budget_mode,
        "saturation_score_entropy": saturation_info.get("score_entropy"),
        "saturation_spatial_entropy": saturation_info.get("spatial_entropy"),
        "saturation_spatial_dispersion": saturation_info.get("spatial_dispersion"),
        "saturation_active_cell_count": saturation_info.get("active_cell_count"),
        "saturation_candidate_diagnostics": saturation_info.get("candidate_diagnostics", []),
    }
    return decode_answer(processor, generated), trace


def prompt_messages(probe: dict[str, Any], *, min_pixels: int, max_pixels: int) -> list[dict[str, Any]]:
    context = str(probe["question"]).replace("<image>", "")
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
    return [
        {"role": "system", "content": "You are a helpful OCR assistant. Answer with the shortest correct text only."},
        {"role": "user", "content": content},
    ]


def build_open_answer_samples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        if str(row.get("binary_polarity", "")) != "positive":
            continue
        image_id = str(row.get("image_id", row.get("sample_id", "")))
        grouped[image_id] = row
    samples: list[dict[str, Any]] = []
    for image_id, row in sorted(grouped.items()):
        answers = [str(value).strip() for value in row.get("answer_options", []) if str(value).strip()]
        if not answers:
            answer = str(row.get("source_text") or row.get("target_text") or "").strip()
            answers = [answer] if answer else []
        question = str(row.get("ocrbench_question") or "").strip()
        image = str(row.get("image") or "").strip()
        if not answers or not question or not image:
            continue
        samples.append(
            {
                "sample_id": f"{image_id}:open-answer-generate",
                "image_id": image_id,
                "image": image,
                "question": question,
                "gold_answers": answers,
                "ocrbench_question_type": row.get("ocrbench_question_type", ""),
                "source_dataset": row.get("source_dataset", ""),
                "ocrbench_row_index": row.get("ocrbench_row_index", ""),
            }
        )
    return samples


def selector_probe(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": sample["sample_id"],
        "id": sample["sample_id"],
        "dataset": "OCRBench-OpenAnswerGenerate",
        "image": sample["image"],
        "image_id": sample["image_id"],
        "question": sample["question"],
        "target_text": sample["question"],
        "source_text": sample["question"],
        "task_family": "ocr_text",
        "evidence_regions": [],
        "ocr_regions": [],
    }


def decode_answer(processor, token_ids: Any) -> str:
    if not isinstance(token_ids, list):
        try:
            token_ids = token_ids.detach().cpu().tolist()
        except AttributeError:
            token_ids = list(token_ids)
    text = processor.batch_decode([token_ids], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return clean_generation(text)


def clean_generation(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"^(answer|assistant)\s*[:：]\s*", "", text, flags=re.IGNORECASE).strip()
    return text.strip().strip("\"'")


def score_answer(prediction: str, gold_answers: list[str]) -> dict[str, float]:
    pred_norm = normalize_answer(prediction)
    gold_norms = [normalize_answer(answer) for answer in gold_answers]
    exact = float(any(pred_norm == gold for gold in gold_norms if gold))
    contains = float(any(gold and gold in pred_norm for gold in gold_norms))
    anls = max((anls_score(pred_norm, gold) for gold in gold_norms if gold), default=0.0)
    return {"exact": exact, "contains": contains, "anls": anls}


def normalize_answer(text: str) -> str:
    text = str(text).lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def anls_score(prediction: str, gold: str) -> float:
    if not prediction and not gold:
        return 1.0
    if not prediction or not gold:
        return 0.0
    distance = levenshtein(prediction, gold)
    normalized = distance / max(len(prediction), len(gold), 1)
    return 1.0 - normalized if normalized < 0.5 else 0.0


def levenshtein(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, char_left in enumerate(left, start=1):
        current = [i]
        for j, char_right in enumerate(right, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (char_left != char_right)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def summarize(rows: list[dict[str, Any]], *, args: argparse.Namespace) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_type.setdefault(str(row.get("ocrbench_question_type", "")), []).append(row)
    return {
        "task": "OCRBench native open-question generation",
        "note": "Greedy free-form generation on original OCRBench questions; selector receives the question text, not the gold answer.",
        "n": len(rows),
        "pretrained": args.pretrained,
        "selector": args.selector,
        "keep_ratio": args.keep_ratio,
        "max_new_tokens": args.max_new_tokens,
        "full_exact": mean(float(row["full_exact"]) for row in rows),
        "pruned_exact": mean(float(row["pruned_exact"]) for row in rows),
        "exact_delta_pruned_minus_full": mean(float(row["exact_delta_pruned_minus_full"]) for row in rows),
        "full_contains": mean(float(row["full_contains"]) for row in rows),
        "pruned_contains": mean(float(row["pruned_contains"]) for row in rows),
        "full_anls": mean(float(row["full_anls"]) for row in rows),
        "pruned_anls": mean(float(row["pruned_anls"]) for row in rows),
        "anls_delta_pruned_minus_full": mean(float(row["anls_delta_pruned_minus_full"]) for row in rows),
        "mean_effective_keep_ratio": mean(float(row["effective_keep_ratio"]) for row in rows),
        "mean_target_text_token_count": mean(float(row["target_text_token_count"]) for row in rows),
        "by_type": {
            key: {
                "n": len(type_rows),
                "full_exact": mean(float(row["full_exact"]) for row in type_rows),
                "pruned_exact": mean(float(row["pruned_exact"]) for row in type_rows),
                "full_anls": mean(float(row["full_anls"]) for row in type_rows),
                "pruned_anls": mean(float(row["pruned_anls"]) for row in type_rows),
            }
            for key, type_rows in sorted(by_type.items())
        },
    }


def write_markdown(path: Path, metrics: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# OCRBench Native Open-Question Generation",
        "",
        metrics["note"],
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| n | {metrics['n']} |",
        f"| full exact | {metrics['full_exact']:.3f} |",
        f"| pruned exact | {metrics['pruned_exact']:.3f} |",
        f"| delta exact | {metrics['exact_delta_pruned_minus_full']:+.3f} |",
        f"| full contains | {metrics['full_contains']:.3f} |",
        f"| pruned contains | {metrics['pruned_contains']:.3f} |",
        f"| full ANLS | {metrics['full_anls']:.3f} |",
        f"| pruned ANLS | {metrics['pruned_anls']:.3f} |",
        f"| delta ANLS | {metrics['anls_delta_pruned_minus_full']:+.3f} |",
        f"| effective keep | {metrics['mean_effective_keep_ratio']:.3f} |",
        f"| selector answer-token count | {metrics['mean_target_text_token_count']:.3f} |",
        "",
        "## By Type",
        "",
        "| Type | n | Full exact | Pruned exact | Full ANLS | Pruned ANLS |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, item in metrics["by_type"].items():
        lines.append(
            f"| {key} | {item['n']} | {item['full_exact']:.3f} | {item['pruned_exact']:.3f} | "
            f"{item['full_anls']:.3f} | {item['pruned_anls']:.3f} |"
        )
    failures = [row for row in rows if float(row["full_exact"]) != float(row["pruned_exact"])]
    lines.extend(
        [
            "",
            "## Exact-Match Flips",
            "",
            "| sample_id | type | gold | full | pruned |",
            "|---|---|---|---|---|",
        ]
    )
    for row in failures[:30]:
        lines.append(
            f"| {row['sample_id']} | {row['ocrbench_question_type']} | "
            f"{'; '.join(row['gold_answers'])} | {row['full_answer']} | {row['pruned_answer']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def mean(values: Any) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    main()
