#!/usr/bin/env python
"""Run a native-question OCRBench open-answer ranking check for Qwen-VL.

This is not the yes/no converted OCRBench probe. Each sample keeps OCRBench's
original question and evaluates whether the model assigns lower loss to the
gold open answer than to a decoy answer from the paired negative probe. The
pruned condition must not receive the gold answer as target text; it uses only
the open question text for selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recap.io import write_json, write_jsonl
from recap.qwen_direct_backend import _load_qwen_direct, _score_continuation
from recap.qwen_pruned_backend import PruneConfig, _score_pruned_continuation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/ocrbench_yesno_probes_100img.jsonl")
    parser.add_argument("--work-dir", default="runs/ocrbench_open_answer/qwen3_8b_open_answer_rank_target_grid0p30")
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--selector", default="target_embed_grid_topk")
    parser.add_argument("--keep-ratio", type=float, default=0.30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--attn-implementation", default="")
    parser.add_argument("--min-pixels", type=int, default=50176)
    parser.add_argument("--max-pixels", type=int, default=802816)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--use-fast-processor", action="store_true")
    args = parser.parse_args()

    samples = build_open_answer_samples(read_jsonl(args.input))
    if args.limit is not None:
        samples = samples[: args.limit]

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
    prune_config = PruneConfig(selector=args.selector, keep_ratio=args.keep_ratio)

    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for sample in samples:
        probe = selector_probe(sample)
        full_gold, _ = _score_continuation(
            probe,
            answer_continuation(sample["gold_answer"]),
            model=model,
            processor=processor,
            process_vision_info=process_vision_info,
            input_device=input_device,
            system_prompt="You are a helpful assistant.",
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            debug_forward=False,
            strict_images=True,
        )
        full_decoy, _ = _score_continuation(
            probe,
            answer_continuation(sample["decoy_answer"]),
            model=model,
            processor=processor,
            process_vision_info=process_vision_info,
            input_device=input_device,
            system_prompt="You are a helpful assistant.",
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            debug_forward=False,
            strict_images=True,
        )
        pruned_gold, _, gold_trace = _score_pruned_continuation(
            probe,
            answer_continuation(sample["gold_answer"]),
            model=model,
            processor=processor,
            process_vision_info=process_vision_info,
            input_device=input_device,
            system_prompt="You are a helpful assistant.",
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            prune_config=prune_config,
            sample_risk=None,
            sample_budget_ratio=None,
            debug_forward=False,
            strict_images=True,
        )
        pruned_decoy, _, decoy_trace = _score_pruned_continuation(
            probe,
            answer_continuation(sample["decoy_answer"]),
            model=model,
            processor=processor,
            process_vision_info=process_vision_info,
            input_device=input_device,
            system_prompt="You are a helpful assistant.",
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            prune_config=prune_config,
            sample_risk=None,
            sample_budget_ratio=None,
            debug_forward=False,
            strict_images=True,
        )
        row = {
            **sample,
            "full_gold_loss": full_gold,
            "full_decoy_loss": full_decoy,
            "full_margin": full_decoy - full_gold,
            "full_correct": full_gold < full_decoy,
            "pruned_gold_loss": pruned_gold,
            "pruned_decoy_loss": pruned_decoy,
            "pruned_margin": pruned_decoy - pruned_gold,
            "pruned_correct": pruned_gold < pruned_decoy,
            "margin_delta_pruned_minus_full": (pruned_decoy - pruned_gold) - (full_decoy - full_gold),
            "selector": args.selector,
            "keep_ratio": args.keep_ratio,
            "effective_keep_ratio": float(gold_trace.get("effective_keep_ratio", 0.0) or 0.0),
            "target_text_token_count": int(gold_trace.get("target_text_token_count", 0) or 0),
        }
        rows.append(row)
        traces.append({"sample_id": sample["sample_id"], "gold_trace": gold_trace, "decoy_trace": decoy_trace})

    metrics = summarize(rows, args=args)
    out_dir = Path(args.work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "open_answer_scores.jsonl", rows)
    write_jsonl(out_dir / "prune_traces.jsonl", traces)
    write_json(out_dir / "metrics.json", metrics)
    write_markdown(out_dir / "open_answer_report.md", metrics, rows)
    print(
        f"Wrote {len(rows)} open-answer ranking rows to {out_dir}; "
        f"full_acc={metrics['full_rank_acc']:.4f}, pruned_acc={metrics['pruned_rank_acc']:.4f}"
    )


def build_open_answer_samples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        image_id = str(row.get("image_id", row.get("sample_id", "")))
        polarity = str(row.get("binary_polarity", ""))
        grouped.setdefault(image_id, {})[polarity] = row
    samples: list[dict[str, Any]] = []
    for image_id, group in sorted(grouped.items()):
        pos = group.get("positive")
        neg = group.get("negative")
        if not pos or not neg:
            continue
        gold = str(pos.get("source_text") or pos.get("target_text") or "").strip()
        decoy = str(neg.get("candidate_answer") or neg.get("target_text") or "").strip()
        question = str(pos.get("ocrbench_question") or "").strip()
        image = str(pos.get("image") or "").strip()
        if not gold or not decoy or not question or not image:
            continue
        samples.append(
            {
                "sample_id": f"{image_id}:open-answer-rank",
                "image_id": image_id,
                "image": image,
                "question": question,
                "gold_answer": gold,
                "decoy_answer": decoy,
                "ocrbench_question_type": pos.get("ocrbench_question_type", ""),
                "source_dataset": pos.get("source_dataset", ""),
                "ocrbench_row_index": pos.get("ocrbench_row_index", ""),
            }
        )
    return samples


def selector_probe(sample: dict[str, Any]) -> dict[str, Any]:
    # Do not include gold_answer/source_text/target_answer here. For open-answer
    # evaluation the selector must not receive the answer string.
    return {
        "sample_id": sample["sample_id"],
        "id": sample["sample_id"],
        "dataset": "OCRBench-OpenAnswerRank",
        "image": sample["image"],
        "image_id": sample["image_id"],
        "question": sample["question"],
        "task_family": "ocr_text",
        "evidence_regions": [],
        "ocr_regions": [],
    }


def answer_continuation(answer: str) -> str:
    return " " + str(answer).strip()


def summarize(rows: list[dict[str, Any]], *, args: argparse.Namespace) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_type.setdefault(str(row.get("ocrbench_question_type", "")), []).append(row)
    return {
        "task": "OCRBench native open-question answer ranking",
        "note": "Gold-vs-decoy answer likelihood on original OCRBench questions; not yes/no verification and not free-form generation.",
        "n": len(rows),
        "pretrained": args.pretrained,
        "selector": args.selector,
        "keep_ratio": args.keep_ratio,
        "full_rank_acc": mean(float(row["full_correct"]) for row in rows),
        "pruned_rank_acc": mean(float(row["pruned_correct"]) for row in rows),
        "rank_acc_delta_pruned_minus_full": mean(float(row["pruned_correct"]) - float(row["full_correct"]) for row in rows),
        "mean_full_margin": mean(float(row["full_margin"]) for row in rows),
        "mean_pruned_margin": mean(float(row["pruned_margin"]) for row in rows),
        "mean_margin_delta_pruned_minus_full": mean(float(row["margin_delta_pruned_minus_full"]) for row in rows),
        "mean_effective_keep_ratio": mean(float(row["effective_keep_ratio"]) for row in rows),
        "mean_target_text_token_count": mean(float(row["target_text_token_count"]) for row in rows),
        "by_type": {
            key: {
                "n": len(type_rows),
                "full_rank_acc": mean(float(row["full_correct"]) for row in type_rows),
                "pruned_rank_acc": mean(float(row["pruned_correct"]) for row in type_rows),
                "mean_margin_delta": mean(float(row["margin_delta_pruned_minus_full"]) for row in type_rows),
            }
            for key, type_rows in sorted(by_type.items())
        },
    }


def write_markdown(path: Path, metrics: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# OCRBench Native Open-Question Answer Ranking",
        "",
        metrics["note"],
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| n | {metrics['n']} |",
        f"| full rank acc | {metrics['full_rank_acc']:.3f} |",
        f"| pruned rank acc | {metrics['pruned_rank_acc']:.3f} |",
        f"| delta acc | {metrics['rank_acc_delta_pruned_minus_full']:+.3f} |",
        f"| full margin | {metrics['mean_full_margin']:+.3f} |",
        f"| pruned margin | {metrics['mean_pruned_margin']:+.3f} |",
        f"| delta margin | {metrics['mean_margin_delta_pruned_minus_full']:+.3f} |",
        f"| effective keep | {metrics['mean_effective_keep_ratio']:.3f} |",
        f"| selector answer-token count | {metrics['mean_target_text_token_count']:.3f} |",
        "",
        "## By Type",
        "",
        "| Type | n | Full | Pruned | Delta margin |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, item in metrics["by_type"].items():
        lines.append(
            f"| {key} | {item['n']} | {item['full_rank_acc']:.3f} | "
            f"{item['pruned_rank_acc']:.3f} | {item['mean_margin_delta']:+.3f} |"
        )
    flips = [row for row in rows if bool(row["full_correct"]) != bool(row["pruned_correct"])]
    lines.extend(
        [
            "",
            "## Flips",
            "",
            "| sample_id | type | gold | decoy | full margin | pruned margin |",
            "|---|---|---|---|---:|---:|",
        ]
    )
    for row in flips[:20]:
        lines.append(
            f"| {row['sample_id']} | {row['ocrbench_question_type']} | {row['gold_answer']} | "
            f"{row['decoy_answer']} | {row['full_margin']:+.3f} | {row['pruned_margin']:+.3f} |"
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
