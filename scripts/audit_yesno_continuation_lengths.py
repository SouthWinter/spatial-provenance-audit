#!/usr/bin/env python3
"""Audit yes/no continuation lengths under the three paper chat templates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from huggingface_hub import try_to_load_from_cache
from tokenizers import Tokenizer
from transformers import AutoProcessor


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "runs"
    / "problem_optimization_audit"
    / "yesno_continuation_length_audit"
)
QUESTION = 'Does the image contain the exact text "TOTAL"? Answer yes or no.'
ANSWERS = (" yes", " no")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen", required=True, help="Qwen3-VL checkpoint or local snapshot")
    parser.add_argument("--llava", required=True, help="LLaVA checkpoint, snapshot, or tokenizer.json")
    parser.add_argument("--internvl", required=True, help="InternVL checkpoint or local snapshot")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def template_delta(tokenizer, prompt: str, full: str) -> dict[str, Any]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    full_ids = tokenizer(full, add_special_tokens=False).input_ids
    return {
        "prompt_tokens": len(prompt_ids),
        "full_tokens": len(full_ids),
        "continuation_tokens": len(full_ids) - len(prompt_ids),
        "tail_token_ids": full_ids[-8:],
    }


def audit_qwen(path: str) -> list[dict[str, Any]]:
    processor = AutoProcessor.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=True,
    )
    prompt_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "placeholder"},
                {"type": "text", "text": QUESTION},
            ],
        },
    ]
    prompt = processor.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    rows = []
    for answer in ANSWERS:
        full = processor.apply_chat_template(
            prompt_messages + [{"role": "assistant", "content": answer}],
            tokenize=False,
            add_generation_prompt=False,
        )
        rows.append({"answer": answer.strip(), **template_delta(processor.tokenizer, prompt, full)})
    return rows


def audit_internvl(path: str) -> list[dict[str, Any]]:
    processor = AutoProcessor.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=True,
    )
    prompt_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": QUESTION},
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    rows = []
    for answer in ANSWERS:
        full = processor.apply_chat_template(
            prompt_messages + [{"role": "assistant", "content": answer}],
            tokenize=False,
            add_generation_prompt=False,
        )
        rows.append({"answer": answer.strip(), **template_delta(processor.tokenizer, prompt, full)})
    return rows


def resolve_llava_tokenizer_json(value: str) -> Path:
    path = Path(value).expanduser()
    candidates = [path, path / "tokenizer.json"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    cached = try_to_load_from_cache(value, "tokenizer.json")
    if isinstance(cached, str) and Path(cached).is_file():
        return Path(cached).resolve()
    raise FileNotFoundError(f"Could not resolve tokenizer.json for {value!r}")


def audit_llava(path: str) -> list[dict[str, Any]]:
    tokenizer = Tokenizer.from_file(str(resolve_llava_tokenizer_json(path)))
    question = f"<image>\n{QUESTION}"
    base = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's "
        f"questions. USER: {question} "
    )
    prompt = f"{base}ASSISTANT:"
    rows = []
    for answer in ANSWERS:
        full = f"{base} ASSISTANT: {answer}</s>"
        prompt_encoding = tokenizer.encode(prompt, add_special_tokens=True)
        full_encoding = tokenizer.encode(full, add_special_tokens=True)
        rows.append(
            {
                "answer": answer.strip(),
                "prompt_tokens": len(prompt_encoding.ids),
                "full_tokens": len(full_encoding.ids),
                "continuation_tokens": len(full_encoding.ids) - len(prompt_encoding.ids),
                "tail_token_ids": full_encoding.ids[-8:],
                "tail_tokens": full_encoding.tokens[-8:],
            }
        )
    return rows


def build_report(results: dict[str, Any]) -> str:
    lines = [
        "# Yes/No Continuation-Length Audit",
        "",
        "The scoring implementation masks prompt tokens and uses the model's mean "
        "cross-entropy over the remaining continuation tokens. The two candidates "
        "have equal continuation length within every model template.",
        "",
        "| Model | yes tokens | no tokens | Equal length |",
        "| --- | ---: | ---: | --- |",
    ]
    for model, rows in results["models"].items():
        yes, no = rows
        lines.append(
            f"| {model} | {yes['continuation_tokens']} | "
            f"{no['continuation_tokens']} | {yes['continuation_tokens'] == no['continuation_tokens']} |"
        )
    lines.extend(
        [
            "",
            "Counts include the answer token and chat-template closing tokens. Because "
            "yes and no have equal length within each backbone, candidate comparison is "
            "not affected by continuation-length imbalance.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    models = {
        "Qwen3-VL-8B": audit_qwen(args.qwen),
        "LLaVA-1.5-7B": audit_llava(args.llava),
        "InternVL3.5-8B": audit_internvl(args.internvl),
    }
    for model, rows in models.items():
        lengths = {row["continuation_tokens"] for row in rows}
        if len(lengths) != 1:
            raise RuntimeError(f"{model} has unequal yes/no continuation lengths: {rows}")
    results = {
        "question": QUESTION,
        "loss_reduction": "mean_over_unmasked_continuation_tokens",
        "models": models,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "yesno_continuation_length_audit.json").write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "yesno_continuation_length_audit.md").write_text(
        build_report(results),
        encoding="utf-8",
    )
    print(f"Wrote continuation-length audit to {args.output_dir}")


if __name__ == "__main__":
    main()
