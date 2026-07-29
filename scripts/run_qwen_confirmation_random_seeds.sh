#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

source "${SCRIPT_DIR}/env_local.sh"

export USE_TF=0
export TRANSFORMERS_NO_TF=1

INPUT="${INPUT:-data/textocr_val_hard_confirmation_500img_seed20260720.jsonl}"
OUT_ROOT="${OUT_ROOT:-runs/textocr_confirmation}"
GPU_ID="${GPU_ID:-0}"
QWEN_MIN_PIXELS="${QWEN_MIN_PIXELS:-802816}"
QWEN_MAX_PIXELS="${QWEN_MAX_PIXELS:-802816}"
SEEDS="${SEEDS:-101 202 303 404 505}"

mkdir -p "${OUT_ROOT}/logs"

for seed in ${SEEDS}; do
  name="qwen3_8b_random_0p30_seed${seed}"
  work_dir="${OUT_ROOT}/${name}"
  log="${OUT_ROOT}/logs/${name}.log"

  if [[ -s "${work_dir}/probe_scores.jsonl" ]] &&
     [[ "$(wc -l <"${work_dir}/probe_scores.jsonl")" -eq 1000 ]]; then
    echo "[skip] ${name}: complete"
    continue
  fi

  echo "[run] ${name}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" TOKENIZERS_PARALLELISM=false \
    python -u -m recap.cli run-qwen-pruned \
      --input "${INPUT}" \
      --is-probes \
      --work-dir "${work_dir}" \
      --pretrained "${QWEN3_PATH}" \
      --selector random \
      --keep-ratio 0.30 \
      --seed "${seed}" \
      --device cuda \
      --device-map auto \
      --attn-implementation eager \
      --min-pixels "${QWEN_MIN_PIXELS}" \
      --max-pixels "${QWEN_MAX_PIXELS}" \
      >"${log}" 2>&1
done

echo "Completed confirmation random seeds: ${SEEDS}"
