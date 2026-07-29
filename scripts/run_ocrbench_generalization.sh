#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

source "${SCRIPT_DIR}/env_local.sh"

CONDA_ENV="${CONDA_ENV:-llama}"
GPU_ID="${GPU_ID:-0}"
INPUT="${INPUT:-data/ocrbench_yesno_probes_100img.jsonl}"
OUT_ROOT="${OUT_ROOT:-runs/ocrbench_generalization}"
LOG_DIR="${LOG_DIR:-${OUT_ROOT}/logs}"
LIMIT="${LIMIT:-}"

QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-${QWEN3_PATH}}"
QWEN_MODEL_TAG="${QWEN_MODEL_TAG:-qwen3_8b}"
QWEN_MIN_PIXELS="${QWEN_MIN_PIXELS:-50176}"
QWEN_MAX_PIXELS="${QWEN_MAX_PIXELS:-50176}"
QWEN_DTYPE="${QWEN_DTYPE:-auto}"

LLAVA_MODEL_PATH="${LLAVA_MODEL_PATH:-${LLAVA_PATH}}"
LLAVA_MODEL_TAG="${LLAVA_MODEL_TAG:-llava15_7b}"
LLAVA_DTYPE="${LLAVA_DTYPE:-float16}"

INTERNVL_MODEL_PATH="${INTERNVL_MODEL_PATH:-${INTERNVL_PATH}}"
INTERNVL_MODEL_TAG="${INTERNVL_MODEL_TAG:-internvl35_8b}"
INTERNVL_DTYPE="${INTERNVL_DTYPE:-bfloat16}"
INTERNVL_MIN_PATCHES="${INTERNVL_MIN_PATCHES:-1}"
INTERNVL_MAX_PATCHES="${INTERNVL_MAX_PATCHES:-12}"

RUN_QWEN="${RUN_QWEN:-1}"
RUN_LLAVA="${RUN_LLAVA:-1}"
RUN_INTERNVL="${RUN_INTERNVL:-1}"
RUN_LLAVA_OFFICIAL="${RUN_LLAVA_OFFICIAL:-1}"
RUN_REPORTS="${RUN_REPORTS:-1}"

mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

py() {
  conda run --no-capture-output -n "${CONDA_ENV}" python "$@"
}

log() {
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

ratio_tag() {
  local ratio="$1"
  echo "${ratio/./p}"
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

run_qwen_direct() {
  local work_dir="${OUT_ROOT}/${QWEN_MODEL_TAG}_ocrbench100_direct"
  if has_score_run "${work_dir}"; then
    log "skip Qwen direct ${work_dir}"
    return
  fi
  run_to_log "qwen_direct_$(basename "${work_dir}")" \
    env CUDA_VISIBLE_DEVICES="${GPU_ID}" TOKENIZERS_PARALLELISM=false \
    conda run --no-capture-output -n "${CONDA_ENV}" python -u -m recap.cli run-qwen-direct \
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
    $(add_limit_args)
}

run_qwen_pruned() {
  local selector="$1"
  local ratio="$2"
  local tag="${selector}_$(ratio_tag "${ratio}")"
  local work_dir="${OUT_ROOT}/${QWEN_MODEL_TAG}_ocrbench100_${tag}"
  if has_pruned_run "${work_dir}"; then
    log "skip Qwen pruned ${work_dir}"
    return
  fi
  run_to_log "qwen_pruned_$(basename "${work_dir}")" \
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
    --selector "${selector}" \
    --keep-ratio "${ratio}" \
    $(add_limit_args)
}

run_llava_direct() {
  local work_dir="${OUT_ROOT}/${LLAVA_MODEL_TAG}_ocrbench100_direct"
  if has_score_run "${work_dir}"; then
    log "skip LLaVA direct ${work_dir}"
    return
  fi
  run_to_log "llava_direct_$(basename "${work_dir}")" \
    env CUDA_VISIBLE_DEVICES="${GPU_ID}" TOKENIZERS_PARALLELISM=false \
    conda run --no-capture-output -n "${CONDA_ENV}" python -u -m recap.cli run-llava-direct \
    --input "${INPUT}" \
    --is-probes \
    --work-dir "${work_dir}" \
    --pretrained "${LLAVA_MODEL_PATH}" \
    --device cuda \
    --device-map auto \
    --dtype "${LLAVA_DTYPE}" \
    --trust-remote-code \
    --attn-implementation eager \
    $(add_limit_args)
}

run_llava_pruned() {
  local selector="$1"
  local ratio="$2"
  local tag="${selector}_$(ratio_tag "${ratio}")"
  local work_dir="${OUT_ROOT}/${LLAVA_MODEL_TAG}_ocrbench100_${tag}"
  if has_pruned_run "${work_dir}"; then
    log "skip LLaVA pruned ${work_dir}"
    return
  fi
  run_to_log "llava_pruned_$(basename "${work_dir}")" \
    env CUDA_VISIBLE_DEVICES="${GPU_ID}" TOKENIZERS_PARALLELISM=false \
    conda run --no-capture-output -n "${CONDA_ENV}" python -u -m recap.cli run-llava-pruned \
    --input "${INPUT}" \
    --is-probes \
    --work-dir "${work_dir}" \
    --pretrained "${LLAVA_MODEL_PATH}" \
    --device cuda \
    --device-map auto \
    --dtype "${LLAVA_DTYPE}" \
    --trust-remote-code \
    --attn-implementation eager \
    --selector "${selector}" \
    --keep-ratio "${ratio}" \
    $(add_limit_args)
}

run_internvl_direct() {
  local work_dir="${OUT_ROOT}/${INTERNVL_MODEL_TAG}_ocrbench100_direct"
  if has_score_run "${work_dir}"; then
    log "skip InternVL direct ${work_dir}"
    return
  fi
  run_to_log "internvl_direct_$(basename "${work_dir}")" \
    env CUDA_VISIBLE_DEVICES="${GPU_ID}" TOKENIZERS_PARALLELISM=false \
    conda run --no-capture-output -n "${CONDA_ENV}" python -u -m recap.cli run-internvl-direct \
    --input "${INPUT}" \
    --is-probes \
    --work-dir "${work_dir}" \
    --pretrained "${INTERNVL_MODEL_PATH}" \
    --device cuda \
    --device-map auto \
    --dtype "${INTERNVL_DTYPE}" \
    --min-patches "${INTERNVL_MIN_PATCHES}" \
    --max-patches "${INTERNVL_MAX_PATCHES}" \
    $(add_limit_args)
}

run_internvl_pruned() {
  local selector="$1"
  local ratio="$2"
  local tag="${selector}_$(ratio_tag "${ratio}")"
  local work_dir="${OUT_ROOT}/${INTERNVL_MODEL_TAG}_ocrbench100_${tag}"
  if has_pruned_run "${work_dir}"; then
    log "skip InternVL pruned ${work_dir}"
    return
  fi
  run_to_log "internvl_pruned_$(basename "${work_dir}")" \
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
    --selector "${selector}" \
    --keep-ratio "${ratio}" \
    $(add_limit_args)
}

py -m recap.cli validate-images \
  --input "${INPUT}" \
  --is-probes \
  --output "${OUT_ROOT}/ocrbench100_image_validation.json"

if [[ "${RUN_QWEN}" == "1" ]]; then
  run_qwen_direct
  run_qwen_pruned target_embed_topk 0.20
  run_qwen_pruned target_embed_topk 0.30
  run_qwen_pruned target_embed_grid_topk 0.30
  run_qwen_pruned target_embed_grid_topk 0.40
  run_qwen_pruned target_embed_grid_topk 0.50
  run_qwen_pruned grid 0.30
  run_qwen_pruned random 0.30
fi

if [[ "${RUN_LLAVA}" == "1" ]]; then
  run_llava_direct
  run_llava_pruned embed_protected_topk 0.40
  run_llava_pruned grid 0.40
  if [[ "${RUN_LLAVA_OFFICIAL}" == "1" ]]; then
    run_llava_pruned visionzip 0.40
    run_llava_pruned fastv 0.40
  fi
fi

if [[ "${RUN_INTERNVL}" == "1" ]]; then
  run_internvl_direct
  run_internvl_pruned target_embed_topk 0.50
  run_internvl_pruned target_embed_grid_topk 0.50
  run_internvl_pruned target_embed_grid_topk 0.60
  run_internvl_pruned embed_topk 0.50
  run_internvl_pruned grid 0.50
fi

if [[ "${RUN_REPORTS}" == "1" ]]; then
  py scripts/build_ocrbench_generalization_report.py
fi

log "OCRBench generalization queue completed"
