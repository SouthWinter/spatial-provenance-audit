#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "llama" ]]; then
  exec conda run --no-capture-output -n llama bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"
source "${SCRIPT_DIR}/env_local.sh"

INPUT="${INPUT:-data/textocr_val_hard_confirmation_500img_seed20260720.jsonl}"
OUT_ROOT="${OUT_ROOT:-runs/efficiency/anchorprune_repeated_exclusive}"
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
  args=(
    python -u -m recap.cli run-llava-pruned
    --input "${INPUT}" --is-probes
    --work-dir "${out}"
    --pretrained "${LLAVA_PATH}"
    --selector "${selector}" --keep-ratio "${ratio}"
    --seed 17 --device cuda --device-map auto --dtype float16
    --trust-remote-code --attn-implementation eager --limit "${LIMIT}"
  )
  if [[ "${selector}" == "anchorprune" ]]; then
    args+=(--anchorprune-clip-model "${ANCHORPRUNE_CLIP_PATH}")
  fi
  "${args[@]}" >"${OUT_ROOT}/logs/${method}_rep${rep}.log" 2>&1
}

# Rotate method order across repetitions to reduce order and temperature bias.
run_point full grid 1.00 1
run_point target target_embed_topk 0.40 1
run_point anchorprune anchorprune 0.40 1

run_point anchorprune anchorprune 0.40 2
run_point full grid 1.00 2
run_point target target_embed_topk 0.40 2

run_point target target_embed_topk 0.40 3
run_point anchorprune anchorprune 0.40 3
run_point full grid 1.00 3

python scripts/build_anchorprune_repeated_timing_report.py --runs-root "${OUT_ROOT}" --warmup 5
