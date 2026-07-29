#!/usr/bin/env bash
set -euo pipefail

INPUT="${INPUT:-data/textocr_val_hard_confirmation_500img_seed20260720.jsonl}"
OUT_ROOT="${OUT_ROOT:-runs/efficiency/scope_repeated_exclusive}"
LLAVA_PATH="${LLAVA_PATH:-llava-hf/llava-1.5-7b-hf}"
GPU_ID="${GPU_ID:-0}"
LIMIT="${LIMIT:-100}"

export USE_TF=0
export TRANSFORMERS_NO_TF=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

mkdir -p "${OUT_ROOT}/logs"

run_point() {
  local method="$1"
  local selector="$2"
  local ratio="$3"
  local rep="$4"
  local out="${OUT_ROOT}/${method}_rep${rep}"
  if [[ -f "${out}/metrics.json" ]]; then
    echo "Skipping completed ${method}_rep${rep}"
    return
  fi
  python -u -m recap.cli run-llava-pruned \
    --input "${INPUT}" --is-probes \
    --work-dir "${out}" \
    --pretrained "${LLAVA_PATH}" \
    --selector "${selector}" --keep-ratio "${ratio}" \
    --seed 17 --device cuda --device-map auto --dtype float16 \
    --trust-remote-code --attn-implementation eager --limit "${LIMIT}" \
    >"${OUT_ROOT}/logs/${method}_rep${rep}.log" 2>&1
}

# Rotate method order across repetitions to reduce order/temperature bias.
run_point full grid 1.00 1
run_point protected target_embed_protected_topk 0.40 1
run_point scope scope 0.40 1

run_point scope scope 0.40 2
run_point full grid 1.00 2
run_point protected target_embed_protected_topk 0.40 2

run_point protected target_embed_protected_topk 0.40 3
run_point scope scope 0.40 3
run_point full grid 1.00 3

python scripts/build_scope_repeated_timing_report.py --runs-root "${OUT_ROOT}" --warmup 5

