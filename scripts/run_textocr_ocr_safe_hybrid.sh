#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

source "${SCRIPT_DIR}/env_local.sh"

CONDA_ENV="${CONDA_ENV:-llama}"
GPU_ID="${GPU_ID:-0}"
INPUT="${INPUT:-data/textocr_val_hard_probes_500img.jsonl}"
LOG_DIR="${LOG_DIR:-runs/textocr_ocr_safe_hybrid/logs}"
LIMIT="${LIMIT:-}"

QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-${QWEN3_PATH}}"
QWEN_DTYPE="${QWEN_DTYPE:-auto}"
QWEN_MIN_PIXELS="${QWEN_MIN_PIXELS:-802816}"
QWEN_MAX_PIXELS="${QWEN_MAX_PIXELS:-802816}"

INTERNVL_MODEL_PATH="${INTERNVL_MODEL_PATH:-${INTERNVL_PATH}}"
INTERNVL_DTYPE="${INTERNVL_DTYPE:-bfloat16}"
INTERNVL_MIN_PATCHES="${INTERNVL_MIN_PATCHES:-1}"
INTERNVL_MAX_PATCHES="${INTERNVL_MAX_PATCHES:-12}"

RUN_QWEN="${RUN_QWEN:-1}"
RUN_INTERNVL="${RUN_INTERNVL:-1}"
RUN_REPORTS="${RUN_REPORTS:-1}"

mkdir -p "${LOG_DIR}"

py() {
  conda run --no-capture-output -n "${CONDA_ENV}" python "$@"
}

log() {
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

run_to_log() {
  local name="$1"
  shift
  local log_path="${LOG_DIR}/${name}.log"
  log "${name} log=${log_path}"
  if "$@" >"${log_path}" 2>&1; then
    tr '\r' '\n' <"${log_path}" | tail -n 20
  else
    local status=$?
    tr '\r' '\n' <"${log_path}" | tail -n 80 || true
    return "${status}"
  fi
}

ratio_tag() {
  local ratio="$1"
  echo "${ratio/./p}"
}

has_score_run() {
  local dir="$1"
  [[ -s "${dir}/metrics.json" && -s "${dir}/probe_scores.jsonl" && -s "${dir}/sample_scores.jsonl" ]]
}

has_pruned_run() {
  local dir="$1"
  [[ -s "${dir}/metrics.json" && -s "${dir}/probe_scores.jsonl" && -s "${dir}/sample_scores.jsonl" && -s "${dir}/prune_traces.jsonl" ]]
}

add_limit_args() {
  if [[ -n "${LIMIT}" ]]; then
    printf '%s\n' --limit "${LIMIT}"
  fi
}

run_qwen_grid_topk() {
  local ratio="$1"
  local tag
  tag="$(ratio_tag "${ratio}")"
  local work_dir="runs/prune_textocr_hard_full1000/qwen3_8b_textocr_hard_full1000_target_embed_grid_topk_${tag}_targetfix_802816"
  if has_pruned_run "${work_dir}"; then
    log "skip Qwen OCR-safe ${work_dir}"
    return
  fi
  run_to_log "qwen_textocr_grid_topk_${tag}" \
    env CUDA_VISIBLE_DEVICES="${GPU_ID}" TOKENIZERS_PARALLELISM=false \
    conda run --no-capture-output -n "${CONDA_ENV}" python -u -m recap.cli run-qwen-pruned \
    --input "${INPUT}" \
    --is-probes \
    --work-dir "${work_dir}" \
    --pretrained "${QWEN_MODEL_PATH}" \
    --device cuda \
    --device-map auto \
    --dtype "${QWEN_DTYPE}" \
    --attn-implementation eager \
    --min-pixels "${QWEN_MIN_PIXELS}" \
    --max-pixels "${QWEN_MAX_PIXELS}" \
    --selector target_embed_grid_topk \
    --keep-ratio "${ratio}" \
    $(add_limit_args)
}

run_internvl_grid_topk() {
  local ratio="$1"
  local tag
  tag="$(ratio_tag "${ratio}")"
  local work_dir="runs/internvl_textocr_hard/internvl35_8b_textocr_hard_full1000_target_embed_grid_topk_${tag}"
  if has_pruned_run "${work_dir}"; then
    log "skip InternVL OCR-safe raw ${work_dir}"
    return
  fi
  run_to_log "internvl_textocr_grid_topk_${tag}" \
    env CUDA_VISIBLE_DEVICES="${GPU_ID}" TOKENIZERS_PARALLELISM=false \
    conda run --no-capture-output -n "${CONDA_ENV}" python -u -m recap.cli run-internvl-pruned \
    --input "${INPUT}" \
    --is-probes \
    --work-dir "${work_dir}" \
    --pretrained "${INTERNVL_MODEL_PATH}" \
    --device cuda \
    --device-map auto \
    --dtype "${INTERNVL_DTYPE}" \
    --min-patches "${INTERNVL_MIN_PATCHES}" \
    --max-patches "${INTERNVL_MAX_PATCHES}" \
    --selector target_embed_grid_topk \
    --keep-ratio "${ratio}" \
    $(add_limit_args)
}

calibrate_internvl_grid_topk() {
  local ratio="$1"
  local tag
  tag="$(ratio_tag "${ratio}")"
  local raw_dir="runs/internvl_textocr_hard/internvl35_8b_textocr_hard_full1000_target_embed_grid_topk_${tag}"
  local out_dir="runs/internvl_textocr_hard/calibrated_test_target_grid_topk${tag}_devthr"
  if has_score_run "${out_dir}"; then
    log "skip InternVL OCR-safe calibration ${out_dir}"
    return
  fi
  run_to_log "calibrate_internvl_textocr_grid_topk_${tag}" \
    conda run --no-capture-output -n "${CONDA_ENV}" python scripts/apply_yesno_threshold.py \
    --score "${raw_dir}/probe_scores.jsonl" \
    --out-dir "${out_dir}" \
    --threshold-split dev \
    --eval-split test
}

py -m recap.cli validate-images \
  --input "${INPUT}" \
  --is-probes \
  --output runs/textocr_ocr_safe_hybrid/image_validation.json

if [[ "${RUN_QWEN}" == "1" ]]; then
  run_qwen_grid_topk 0.30
  run_qwen_grid_topk 0.50
fi

if [[ "${RUN_INTERNVL}" == "1" ]]; then
  run_internvl_grid_topk 0.50
  run_internvl_grid_topk 0.60
  calibrate_internvl_grid_topk 0.50
  calibrate_internvl_grid_topk 0.60
fi

if [[ "${RUN_REPORTS}" == "1" ]]; then
  py scripts/build_textocr_hard_cross_model_report.py
fi

log "TextOCR OCR-safe hybrid queue completed"
