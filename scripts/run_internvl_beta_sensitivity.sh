#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"
source "${SCRIPT_DIR}/env_local.sh"

export USE_TF=0
export TRANSFORMERS_NO_TF=1

INPUT="${INPUT:-data/textocr_val_hard_probes_500img.jsonl}"
OUT_ROOT="${OUT_ROOT:-runs/internvl_textocr_hard/beta_sensitivity}"
GPU_ID="${GPU_ID:-0}"
PARALLEL_RUNS="${PARALLEL_RUNS:-0}"
mkdir -p "${OUT_ROOT}/logs"

run_variant() {
  local name="$1"
  local beta="$2"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" TOKENIZERS_PARALLELISM=false python -u -m recap.cli run-internvl-pruned \
    --input "${INPUT}" \
    --is-probes \
    --work-dir "${OUT_ROOT}/${name}" \
    --pretrained "${INTERNVL_PATH}" \
    --selector target_embed_soft_evidence_topk \
    --keep-ratio 0.50 \
    --evidence-boost "${beta}" \
    --seed 17 \
    --device cuda \
    --device-map auto \
    --dtype bfloat16 \
    --trust-remote-code \
    --low-cpu-mem-usage \
    --attn-implementation eager
}

if [[ "${PARALLEL_RUNS}" == "1" ]]; then
  run_variant soft_evidence_beta_0p02 0.02 >"${OUT_ROOT}/logs/soft_evidence_beta_0p02.log" 2>&1 &
  pid_a=$!
  run_variant soft_evidence_beta_0p10 0.10 >"${OUT_ROOT}/logs/soft_evidence_beta_0p10.log" 2>&1 &
  pid_b=$!
  wait "${pid_a}"
  wait "${pid_b}"
else
  run_variant soft_evidence_beta_0p02 0.02 >"${OUT_ROOT}/logs/soft_evidence_beta_0p02.log" 2>&1
  run_variant soft_evidence_beta_0p10 0.10 >"${OUT_ROOT}/logs/soft_evidence_beta_0p10.log" 2>&1
fi

run_variant soft_evidence_beta_0p20 0.20 >"${OUT_ROOT}/logs/soft_evidence_beta_0p20.log" 2>&1
