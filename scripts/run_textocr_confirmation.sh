#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

source "${SCRIPT_DIR}/env_local.sh"

# The llama environment contains an unused TensorFlow install whose protobuf
# stack is ABI-incompatible. Transformers only needs PyTorch for these runs.
export USE_TF=0
export TRANSFORMERS_NO_TF=1

INPUT="${INPUT:-data/textocr_val_hard_confirmation_500img_seed20260720.jsonl}"
OUT_ROOT="${OUT_ROOT:-runs/textocr_confirmation}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-17}"
QWEN_MIN_PIXELS="${QWEN_MIN_PIXELS:-802816}"
QWEN_MAX_PIXELS="${QWEN_MAX_PIXELS:-802816}"
LIMIT="${LIMIT:-}"

limit_args=()
if [[ -n "${LIMIT}" ]]; then
  limit_args+=(--limit "${LIMIT}")
fi

mkdir -p "${OUT_ROOT}/logs"

run_qwen() {
  local name="$1"
  local selector="$2"
  local ratio="$3"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" TOKENIZERS_PARALLELISM=false python -u -m recap.cli run-qwen-pruned \
    --input "${INPUT}" \
    --is-probes \
    --work-dir "${OUT_ROOT}/${name}" \
    --pretrained "${QWEN3_PATH}" \
    --selector "${selector}" \
    --keep-ratio "${ratio}" \
    --seed "${SEED}" \
    --device cuda \
    --device-map auto \
    --attn-implementation eager \
    --min-pixels "${QWEN_MIN_PIXELS}" \
    --max-pixels "${QWEN_MAX_PIXELS}" \
    ${limit_args[@]+"${limit_args[@]}"}
}

run_llava_pruned() {
  local name="$1"
  local selector="$2"
  local ratio="$3"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" TOKENIZERS_PARALLELISM=false python -u -m recap.cli run-llava-pruned \
    --input "${INPUT}" \
    --is-probes \
    --work-dir "${OUT_ROOT}/${name}" \
    --pretrained "${LLAVA_PATH}" \
    --selector "${selector}" \
    --keep-ratio "${ratio}" \
    --seed "${SEED}" \
    --device cuda \
    --device-map auto \
    --dtype float16 \
    --trust-remote-code \
    --attn-implementation eager \
    ${limit_args[@]+"${limit_args[@]}"}
}

run_llava_full() {
  CUDA_VISIBLE_DEVICES="${GPU_ID}" TOKENIZERS_PARALLELISM=false python -u -m recap.cli run-llava-direct \
    --input "${INPUT}" \
    --work-dir "${OUT_ROOT}/llava15_7b_full" \
    --pretrained "${LLAVA_PATH}" \
    --is-probes \
    --device cuda \
    --device-map auto \
    --dtype float16 \
    --trust-remote-code \
    --attn-implementation eager \
    ${limit_args[@]+"${limit_args[@]}"}
}

run_qwen qwen3_8b_full topk 1.00 >"${OUT_ROOT}/logs/qwen3_8b_full.log" 2>&1 &
qwen_full_pid=$!
run_qwen qwen3_8b_target_0p30 target_embed_topk 0.30 >"${OUT_ROOT}/logs/qwen3_8b_target_0p30.log" 2>&1 &
qwen_target_pid=$!
wait "${qwen_full_pid}"
wait "${qwen_target_pid}"

run_qwen qwen3_8b_random_0p30 random 0.30 >"${OUT_ROOT}/logs/qwen3_8b_random_0p30.log" 2>&1 &
qwen_random_pid=$!
run_qwen qwen3_8b_grid_0p30 grid 0.30 >"${OUT_ROOT}/logs/qwen3_8b_grid_0p30.log" 2>&1 &
qwen_grid_pid=$!
wait "${qwen_random_pid}"
wait "${qwen_grid_pid}"

run_llava_full >"${OUT_ROOT}/logs/llava15_7b_full.log" 2>&1 &
llava_full_pid=$!
run_llava_pruned llava15_7b_protected_0p40 embed_protected_topk 0.40 >"${OUT_ROOT}/logs/llava15_7b_protected_0p40.log" 2>&1 &
llava_protected_pid=$!
wait "${llava_full_pid}"
wait "${llava_protected_pid}"

run_llava_pruned llava15_7b_random_0p40 random 0.40 >"${OUT_ROOT}/logs/llava15_7b_random_0p40.log" 2>&1 &
llava_random_pid=$!
run_llava_pruned llava15_7b_visionzip_0p40 visionzip 0.40 >"${OUT_ROOT}/logs/llava15_7b_visionzip_0p40.log" 2>&1 &
llava_visionzip_pid=$!
wait "${llava_random_pid}"
wait "${llava_visionzip_pid}"
