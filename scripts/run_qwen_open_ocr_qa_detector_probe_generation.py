#!/usr/bin/env python3
"""Run Qwen open-QA generation from selector-ready detector probes.

The input JSONL is expected to contain image paths, questions, gold answers,
and detector boxes in `evidence_regions`. This script is for the detector-in-
loop experiment: the pruned branch consumes those boxes during visual-token
selection. By default it reuses cached full-prefix answers from the probe file
and only runs the pruned branch, which keeps the detector-in-loop comparison
cheap and makes the detector source explicit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recap.io import write_json, write_jsonl
from recap.qwen_direct_backend import _load_qwen_direct
from recap.qwen_pruned_backend import PruneConfig
from scripts.run_qwen_ocrbench_open_answer_generation import generate_full, generate_pruned
from scripts.run_qwen_open_ocr_qa_generation import score_task_answer


TASK_METRIC = {
    "TextVQA-lite": "textvqa_accuracy",
    "textvqa_val_lite": "textvqa_accuracy",
    "DocVQA-lite": "anls",
    "docvqa_val_lite": "anls",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="runs/problem_optimization_audit/open_ocr_qa_easyocr_detector_boxes/easyocr_selector_probes.jsonl",
    )
    parser.add_argument(
        "--work-dir",
        default="runs/problem_optimization_audit/open_ocr_qa_detector_in_loop/qwen3_8b_easyocr_target_grid0p70",
    )
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--selector", default="target_embed_soft_evidence_topk")
    parser.add_argument("--keep-ratio", type=float, default=0.70)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--attn-implementation", default="")
    parser.add_argument("--min-pixels", type=int, default=50176)
    parser.add_argument("--max-pixels", type=int, default=802816)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--use-fast-processor", action="store_true")
    parser.add_argument(
        "--rerun-full",
        action="store_true",
        help="Regenerate the full-prefix answer instead of reusing cached full_answer from the probe.",
    )
    parser.add_argument(
        "--append-answer-instruction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append the same short-answer instruction used by the native open-QA runner.",
    )
    args = parser.parse_args()

    probes = read_jsonl(Path(args.input))
    if args.limit is not None:
        probes = probes[: args.limit]
    probes = [prepare_probe(probe, append_instruction=args.append_answer_instruction) for probe in probes]

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
    for probe in tqdm(probes, desc="open-QA detector-in-loop generation"):
        task = str(probe.get("dataset", ""))
        metric = TASK_METRIC.get(task, "anls" if "docvqa" in task.lower() else "textvqa_accuracy")
        gold_answers = [str(item) for item in probe.get("gold_answers", []) if str(item).strip()]

        if args.rerun_full or not str(probe.get("full_answer", "")).strip():
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
            full_source = "rerun"
        else:
            full_answer = str(probe.get("full_answer", ""))
            full_source = "cached_probe_full_answer"

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
        full_metrics = score_task_answer(full_answer, gold_answers, metric)
        pruned_metrics = score_task_answer(pruned_answer, gold_answers, metric)
        detector_box_count = len(probe.get("evidence_regions") or [])
        row = {
            "sample_id": probe.get("sample_id", probe.get("id", "")),
            "dataset": task,
            "question_id": probe.get("question_id", ""),
            "question": probe.get("question", ""),
            "gold_answers": gold_answers,
            "full_answer": full_answer,
            "pruned_answer": pruned_answer,
            "full_source": full_source,
            "primary_metric": metric,
            "full_score": full_metrics["primary"],
            "pruned_score": pruned_metrics["primary"],
            "score_delta_pruned_minus_full": pruned_metrics["primary"] - full_metrics["primary"],
            "full_exact": full_metrics["exact"],
            "pruned_exact": pruned_metrics["exact"],
            "full_anls": full_metrics["anls"],
            "pruned_anls": pruned_metrics["anls"],
            "full_textvqa_accuracy": full_metrics["textvqa_accuracy"],
            "pruned_textvqa_accuracy": pruned_metrics["textvqa_accuracy"],
            "selector": args.selector,
            "keep_ratio": args.keep_ratio,
            "effective_keep_ratio": float(trace.get("effective_keep_ratio", 0.0) or 0.0),
            "target_text_token_count": int(trace.get("target_text_token_count", 0) or 0),
            "detector_name": probe.get("detector_name", ""),
            "bbox_source": probe.get("bbox_source", ""),
            "detector_box_count": detector_box_count,
            "detector_elapsed_sec": float(probe.get("detector_elapsed_sec", 0.0) or 0.0),
            "selector_target_source": probe.get("selector_target_source", ""),
        }
        trace.update(
            {
                "dataset": task,
                "question_id": probe.get("question_id", ""),
                "detector_name": probe.get("detector_name", ""),
                "bbox_source": probe.get("bbox_source", ""),
                "detector_box_count": detector_box_count,
                "detector_elapsed_sec": float(probe.get("detector_elapsed_sec", 0.0) or 0.0),
                "selector_target_source": probe.get("selector_target_source", ""),
            }
        )
        rows.append(row)
        traces.append(trace)

    metrics = summarize(rows, args=args)
    out_dir = Path(args.work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "open_ocr_qa_detector_generation.jsonl", rows)
    write_jsonl(out_dir / "prune_traces.jsonl", traces)
    write_json(out_dir / "metrics.json", metrics)
    write_markdown(out_dir / "open_ocr_qa_detector_generation_report.md", metrics, rows)
    print(
        f"Wrote {len(rows)} detector-in-loop rows to {out_dir}; "
        f"full_score={metrics['full_score']:.4f}, pruned_score={metrics['pruned_score']:.4f}, "
        f"delta={metrics['score_delta_pruned_minus_full']:+.4f}"
    )


def prepare_probe(probe: dict[str, Any], *, append_instruction: bool) -> dict[str, Any]:
    out = dict(probe)
    question = str(out.get("question", "")).strip()
    instruction = "Answer the question using a single word or phrase."
    if append_instruction and instruction.lower() not in question.lower():
        question = f"{question}\n{instruction}"
    out["question"] = question
    out["target_text"] = question
    out["source_text"] = question
    out.setdefault("task_family", "open_ocr_qa")
    return out


def summarize(rows: list[dict[str, Any]], *, args: argparse.Namespace) -> dict[str, Any]:
    task_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        task_rows.setdefault(str(row.get("dataset", "")), []).append(row)
    return {
        "task": "Open OCR/DocVQA detector-in-loop generation",
        "note": "Pruned branch consumes selector-visible detector boxes from evidence_regions. Full branch is cached by default unless --rerun-full is used.",
        "n": len(rows),
        "input": args.input,
        "pretrained": args.pretrained,
        "selector": args.selector,
        "keep_ratio": args.keep_ratio,
        "max_new_tokens": args.max_new_tokens,
        "rerun_full": bool(args.rerun_full),
        "full_score": mean(float(row["full_score"]) for row in rows),
        "pruned_score": mean(float(row["pruned_score"]) for row in rows),
        "score_delta_pruned_minus_full": mean(float(row["score_delta_pruned_minus_full"]) for row in rows),
        "full_exact": mean(float(row["full_exact"]) for row in rows),
        "pruned_exact": mean(float(row["pruned_exact"]) for row in rows),
        "full_anls": mean(float(row["full_anls"]) for row in rows),
        "pruned_anls": mean(float(row["pruned_anls"]) for row in rows),
        "full_textvqa_accuracy": mean(float(row["full_textvqa_accuracy"]) for row in rows),
        "pruned_textvqa_accuracy": mean(float(row["pruned_textvqa_accuracy"]) for row in rows),
        "mean_effective_keep_ratio": mean(float(row["effective_keep_ratio"]) for row in rows),
        "mean_target_text_token_count": mean(float(row["target_text_token_count"]) for row in rows),
        "mean_detector_box_count": mean(float(row["detector_box_count"]) for row in rows),
        "rows_with_detector_boxes": sum(int(row["detector_box_count"]) > 0 for row in rows),
        "mean_detector_elapsed_ms": 1000.0 * mean(float(row["detector_elapsed_sec"]) for row in rows),
        "by_dataset": {
            key: summarize_group(group)
            for key, group in sorted(task_rows.items())
        },
    }


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "n": float(len(rows)),
        "full_score": mean(float(row["full_score"]) for row in rows),
        "pruned_score": mean(float(row["pruned_score"]) for row in rows),
        "score_delta_pruned_minus_full": mean(float(row["score_delta_pruned_minus_full"]) for row in rows),
        "mean_effective_keep_ratio": mean(float(row["effective_keep_ratio"]) for row in rows),
        "mean_detector_box_count": mean(float(row["detector_box_count"]) for row in rows),
        "rows_with_detector_boxes": float(sum(int(row["detector_box_count"]) > 0 for row in rows)),
    }


def write_markdown(path: Path, metrics: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Open OCR/DocVQA Detector-in-Loop Generation",
        "",
        metrics["note"],
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| n | {metrics['n']} |",
        f"| full score | {metrics['full_score']:.3f} |",
        f"| pruned score | {metrics['pruned_score']:.3f} |",
        f"| delta score | {metrics['score_delta_pruned_minus_full']:+.3f} |",
        f"| full exact | {metrics['full_exact']:.3f} |",
        f"| pruned exact | {metrics['pruned_exact']:.3f} |",
        f"| full ANLS | {metrics['full_anls']:.3f} |",
        f"| pruned ANLS | {metrics['pruned_anls']:.3f} |",
        f"| full TextVQA acc. | {metrics['full_textvqa_accuracy']:.3f} |",
        f"| pruned TextVQA acc. | {metrics['pruned_textvqa_accuracy']:.3f} |",
        f"| effective keep | {metrics['mean_effective_keep_ratio']:.3f} |",
        f"| detector boxes / row | {metrics['mean_detector_box_count']:.3f} |",
        f"| rows with detector boxes | {metrics['rows_with_detector_boxes']} |",
        f"| detector ms / row | {metrics['mean_detector_elapsed_ms']:.1f} |",
        "",
        "## By Dataset",
        "",
        "| Dataset | n | Full | Pruned | Delta | Keep | Boxes/row | Rows w/ boxes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, item in metrics["by_dataset"].items():
        lines.append(
            f"| {dataset} | {int(item['n'])} | {item['full_score']:.3f} | "
            f"{item['pruned_score']:.3f} | {item['score_delta_pruned_minus_full']:+.3f} | "
            f"{item['mean_effective_keep_ratio']:.3f} | {item['mean_detector_box_count']:.3f} | "
            f"{int(item['rows_with_detector_boxes'])} |"
        )
    flips = [row for row in rows if abs(float(row["score_delta_pruned_minus_full"])) > 1e-9]
    lines.extend(
        [
            "",
            "## Score Flips",
            "",
            "| Sample | Dataset | Gold | Full | Pruned | Full | Pruned | Boxes |",
            "|---|---|---|---|---|---:|---:|---:|",
        ]
    )
    for row in flips[:50]:
        lines.append(
            f"| {row['sample_id']} | {row['dataset']} | {'; '.join(row['gold_answers'])} | "
            f"{row['full_answer']} | {row['pruned_answer']} | "
            f"{float(row['full_score']):.3f} | {float(row['pruned_score']):.3f} | "
            f"{row['detector_box_count']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def mean(values: Any) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


if __name__ == "__main__":
    main()
