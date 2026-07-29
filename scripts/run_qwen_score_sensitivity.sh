#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"
source "${SCRIPT_DIR}/env_local.sh"

export USE_TF=0
export TRANSFORMERS_NO_TF=1

INPUT="${INPUT:-data/textocr_val_hard_confirmation_500img_seed20260720.jsonl}"
OUT_ROOT="${OUT_ROOT:-runs/textocr_confirmation/qwen_score_sensitivity}"
GPU_ID="${GPU_ID:-0}"
mkdir -p "${OUT_ROOT}/logs"

run_variant() {
  local name="$1"
  local weight="$2"
  local query_topk="$3"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" TOKENIZERS_PARALLELISM=false python -u -m recap.cli run-qwen-pruned \
    --input "${INPUT}" \
    --is-probes \
    --work-dir "${OUT_ROOT}/${name}" \
    --pretrained "${QWEN3_PATH}" \
    --selector target_embed_topk \
    --keep-ratio 0.30 \
    --embedding-relevance-weight "${weight}" \
    --embedding-query-topk "${query_topk}" \
    --seed 17 \
    --device cuda \
    --device-map auto \
    --attn-implementation eager \
    --min-pixels 802816 \
    --max-pixels 802816
}

run_variant relevance_w100_top2 1.00 2 >"${OUT_ROOT}/logs/relevance_w100_top2.log" 2>&1 &
pid_a=$!
run_variant relevance_w070_top2 0.70 2 >"${OUT_ROOT}/logs/relevance_w070_top2.log" 2>&1 &
pid_b=$!
wait "${pid_a}"
wait "${pid_b}"

run_variant relevance_w050_top2 0.50 2 >"${OUT_ROOT}/logs/relevance_w050_top2.log" 2>&1 &
pid_a=$!
run_variant relevance_w085_top1 0.85 1 >"${OUT_ROOT}/logs/relevance_w085_top1.log" 2>&1 &
pid_b=$!
wait "${pid_a}"
wait "${pid_b}"

run_variant relevance_w085_top4 0.85 4 >"${OUT_ROOT}/logs/relevance_w085_top4.log" 2>&1
