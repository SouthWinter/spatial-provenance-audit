#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
QWEN="${QWEN_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
LLAVA="${LLAVA_PATH:-llava-hf/llava-1.5-7b-hf}"
RUNS="$ROOT/runs/open_ocr_qa_full"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false

run_qwen() {
  local task=$1
  local task_slug=$2
  local selector=$3
  local reuse=$4
  "$PYTHON" "$ROOT/scripts/run_qwen_open_ocr_qa_generation.py" \
    --task "$task" \
    --work-dir "$RUNS/qwen3_8b_${task_slug}_${selector}0p70" \
    --pretrained "$QWEN" \
    --selector "$selector" \
    --keep-ratio 0.70 \
    --reuse-full-rows "$reuse" \
    --checkpoint-every 25 \
    --resume
}

run_llava() {
  local task=$1
  local task_slug=$2
  local selector=$3
  local reuse=$4
  "$PYTHON" "$ROOT/scripts/run_llava_open_ocr_qa_generation.py" \
    --task "$task" \
    --work-dir "$RUNS/llava15_7b_${task_slug}_${selector}0p70" \
    --pretrained "$LLAVA" \
    --selector "$selector" \
    --keep-ratio 0.70 \
    --reuse-full-rows "$reuse" \
    --checkpoint-every 25 \
    --resume
}

run_qwen textvqa_val textvqa_val random \
  "$RUNS/qwen3_8b_textvqa_val_target_grid0p70/open_ocr_qa_generation.jsonl"
run_qwen textvqa_val textvqa_val grid \
  "$RUNS/qwen3_8b_textvqa_val_target_grid0p70/open_ocr_qa_generation.jsonl"
run_qwen docvqa_val docvqa_val random \
  "$RUNS/qwen3_8b_docvqa_val_target_grid0p70/open_ocr_qa_generation.jsonl"
run_qwen docvqa_val docvqa_val grid \
  "$RUNS/qwen3_8b_docvqa_val_target_grid0p70/open_ocr_qa_generation.jsonl"

run_llava textvqa_val textvqa_val random \
  "$RUNS/llava15_7b_textvqa_val_target0p70/open_ocr_qa_generation.jsonl"
run_llava textvqa_val textvqa_val grid \
  "$RUNS/llava15_7b_textvqa_val_target0p70/open_ocr_qa_generation.jsonl"
run_llava docvqa_val docvqa_val random \
  "$RUNS/llava15_7b_docvqa_val_target0p70/open_ocr_qa_generation.jsonl"
run_llava docvqa_val docvqa_val grid \
  "$RUNS/llava15_7b_docvqa_val_target0p70/open_ocr_qa_generation.jsonl"
