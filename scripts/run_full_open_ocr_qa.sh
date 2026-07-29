#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "llama" ]]; then
  exec conda run --no-capture-output -n llama bash "$0" "$@"
fi

source scripts/env_local.sh

export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-1800}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUTPUT_ROOT="${FULL_OPEN_QA_ROOT:-runs/open_ocr_qa_full}"
LOG_ROOT="${OUTPUT_ROOT}/logs"
mkdir -p "${LOG_ROOT}"

run_stage() {
  local name="$1"
  shift
  echo "[$(date --iso-8601=seconds)] START ${name}"
  "$@" 2>&1 | tee -a "${LOG_ROOT}/${name}.log"
  echo "[$(date --iso-8601=seconds)] DONE  ${name}"
}

qwen_run() {
  local task="$1"
  local keep="$2"
  local work_dir="$3"
  shift 3
  python -u scripts/run_qwen_open_ocr_qa_generation.py \
    --task "${task}" \
    --work-dir "${work_dir}" \
    --pretrained "${QWEN3_PATH}" \
    --selector target_embed_grid_topk \
    --selector-target-source question \
    --keep-ratio "${keep}" \
    --max-new-tokens 32 \
    --dtype auto \
    --resume \
    --checkpoint-every 1 \
    "$@"
}

llava_run() {
  local task="$1"
  local keep="$2"
  local work_dir="$3"
  shift 3
  python -u scripts/run_llava_open_ocr_qa_generation.py \
    --task "${task}" \
    --work-dir "${work_dir}" \
    --pretrained "${LLAVA_PATH}" \
    --selector target_embed_topk \
    --keep-ratio "${keep}" \
    --max-new-tokens 32 \
    --dtype float16 \
    --attn-implementation eager \
    --resume \
    --checkpoint-every 1 \
    "$@"
}

QWEN_TEXT_70="${OUTPUT_ROOT}/qwen3_8b_textvqa_val_target_grid0p70"
QWEN_TEXT_30="${OUTPUT_ROOT}/qwen3_8b_textvqa_val_target_grid0p30"
QWEN_DOC_70="${OUTPUT_ROOT}/qwen3_8b_docvqa_val_target_grid0p70"
QWEN_DOC_30="${OUTPUT_ROOT}/qwen3_8b_docvqa_val_target_grid0p30"
LLAVA_TEXT_70="${OUTPUT_ROOT}/llava15_7b_textvqa_val_target0p70"
LLAVA_TEXT_40="${OUTPUT_ROOT}/llava15_7b_textvqa_val_target0p40"
LLAVA_DOC_70="${OUTPUT_ROOT}/llava15_7b_docvqa_val_target0p70"
LLAVA_DOC_40="${OUTPUT_ROOT}/llava15_7b_docvqa_val_target0p40"

run_stage qwen_textvqa_70 qwen_run textvqa_val 0.70 "${QWEN_TEXT_70}"
run_stage qwen_textvqa_30 qwen_run textvqa_val 0.30 "${QWEN_TEXT_30}" \
  --reuse-full-rows "${QWEN_TEXT_70}/open_ocr_qa_generation.jsonl"

run_stage qwen_docvqa_70 qwen_run docvqa_val 0.70 "${QWEN_DOC_70}"
run_stage qwen_docvqa_30 qwen_run docvqa_val 0.30 "${QWEN_DOC_30}" \
  --reuse-full-rows "${QWEN_DOC_70}/open_ocr_qa_generation.jsonl"

run_stage llava_textvqa_70 llava_run textvqa_val 0.70 "${LLAVA_TEXT_70}"
run_stage llava_textvqa_40 llava_run textvqa_val 0.40 "${LLAVA_TEXT_40}" \
  --reuse-full-rows "${LLAVA_TEXT_70}/open_ocr_qa_generation.jsonl"

run_stage llava_docvqa_70 llava_run docvqa_val 0.70 "${LLAVA_DOC_70}"
run_stage llava_docvqa_40 llava_run docvqa_val 0.40 "${LLAVA_DOC_40}" \
  --reuse-full-rows "${LLAVA_DOC_70}/open_ocr_qa_generation.jsonl"

run_stage build_report python scripts/build_full_open_ocr_qa_report.py --root "${OUTPUT_ROOT}"

echo "[$(date --iso-8601=seconds)] Full open-QA matrix complete."
