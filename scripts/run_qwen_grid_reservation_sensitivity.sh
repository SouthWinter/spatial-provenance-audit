#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"
source "${SCRIPT_DIR}/env_local.sh"

export USE_TF=0
export TRANSFORMERS_NO_TF=1

INPUT="${INPUT:-data/textocr_val_hard_confirmation_500img_seed20260720.jsonl}"
OUT_ROOT="${OUT_ROOT:-runs/textocr_confirmation/qwen_grid_reservation_sensitivity}"
GPU_ID="${GPU_ID:-0}"
mkdir -p "${OUT_ROOT}/logs"

run_variant() {
  local name="$1"
  local grid_ratio="$2"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" TOKENIZERS_PARALLELISM=false python -u -m recap.cli run-qwen-pruned \
    --input "${INPUT}" \
    --is-probes \
    --work-dir "${OUT_ROOT}/${name}" \
    --pretrained "${QWEN3_PATH}" \
    --selector target_embed_grid_topk \
    --keep-ratio 0.30 \
    --hybrid-core-ratio "${grid_ratio}" \
    --seed 17 \
    --device cuda \
    --device-map auto \
    --attn-implementation eager \
    --min-pixels 802816 \
    --max-pixels 802816
}

run_variant grid_reservation_0p25 0.25 >"${OUT_ROOT}/logs/grid_reservation_0p25.log" 2>&1 &
pid_a=$!
run_variant grid_reservation_0p50 0.50 >"${OUT_ROOT}/logs/grid_reservation_0p50.log" 2>&1 &
pid_b=$!
wait "${pid_a}"
wait "${pid_b}"

run_variant grid_reservation_0p75 0.75 >"${OUT_ROOT}/logs/grid_reservation_0p75.log" 2>&1
