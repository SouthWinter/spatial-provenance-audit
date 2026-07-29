#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

source "${SCRIPT_DIR}/env_local.sh"

CONDA_ENV="${CONDA_ENV:-llama}"
GPU_ID="${GPU_ID:-0}"
LLAVA_DTYPE="${LLAVA_DTYPE:-float16}"
INTERNVL_DTYPE="${INTERNVL_DTYPE:-bfloat16}"
INTERNVL_MIN_PATCHES="${INTERNVL_MIN_PATCHES:-1}"
INTERNVL_MAX_PATCHES="${INTERNVL_MAX_PATCHES:-12}"
QWEN_PIXELS="${QWEN_PIXELS:-50176}"
LIMIT="${LIMIT:-}"
RUN_RELATION_DIRECT="${RUN_RELATION_DIRECT:-1}"
RUN_TEXTOCR_CAUSAL="${RUN_TEXTOCR_CAUSAL:-1}"
RUN_SPATIAL_PRUNED="${RUN_SPATIAL_PRUNED:-1}"
RUN_REPORTS="${RUN_REPORTS:-1}"
LOG_DIR="${LOG_DIR:-runs/future_experiments/logs}"

mkdir -p runs/future_experiments "${LOG_DIR}"

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

has_score_run() {
  local dir="$1"
  [[ -s "${dir}/metrics.json" && -s "${dir}/probe_scores.jsonl" ]]
}

has_pruned_run() {
  local dir="$1"
  [[ -s "${dir}/metrics.json" && -s "${dir}/probe_scores.jsonl" && -s "${dir}/prune_traces.jsonl" ]]
}

limit_args=()
if [[ -n "${LIMIT}" ]]; then
  limit_args=(--limit "${LIMIT}")
fi

compact_metrics() {
  local dir="$1"
  if [[ -s "${dir}/metrics.json" ]]; then
    py -m recap.cli compact-metrics \
      --input "${dir}/metrics.json" \
      --output "${dir}/metrics_compact.json" \
      --include-by-family \
      --include-by-relation
  fi
}

run_llava_direct() {
  local input="$1"
  local work_dir="$2"
  if has_score_run "${work_dir}"; then
    log "skip LLaVA direct ${work_dir}"
    compact_metrics "${work_dir}"
    return
  fi
  log "run LLaVA direct ${work_dir}"
  run_to_log "llava_direct_$(basename "${work_dir}")" \
    env CUDA_VISIBLE_DEVICES="${GPU_ID}" TOKENIZERS_PARALLELISM=false \
    conda run --no-capture-output -n "${CONDA_ENV}" python -u -m recap.cli run-llava-direct \
    --input "${input}" \
    --work-dir "${work_dir}" \
    --pretrained "${LLAVA_PATH}" \
    --device cuda \
    --device-map auto \
    --dtype "${LLAVA_DTYPE}" \
    --trust-remote-code \
    --attn-implementation eager \
    --keep-non-left-right \
    --probe-mode profile_fast \
    ${limit_args[@]+"${limit_args[@]}"}
  compact_metrics "${work_dir}"
}

run_internvl_direct() {
  local input="$1"
  local work_dir="$2"
  if has_score_run "${work_dir}"; then
    log "skip InternVL direct ${work_dir}"
    compact_metrics "${work_dir}"
    return
  fi
  log "run InternVL direct ${work_dir}"
  run_to_log "internvl_direct_$(basename "${work_dir}")" \
    env CUDA_VISIBLE_DEVICES="${GPU_ID}" TOKENIZERS_PARALLELISM=false \
    conda run --no-capture-output -n "${CONDA_ENV}" python -u -m recap.cli run-internvl-direct \
    --input "${input}" \
    --work-dir "${work_dir}" \
    --pretrained "${INTERNVL_PATH}" \
    --device cuda \
    --device-map auto \
    --dtype "${INTERNVL_DTYPE}" \
    --min-patches "${INTERNVL_MIN_PATCHES}" \
    --max-patches "${INTERNVL_MAX_PATCHES}" \
    --keep-non-left-right \
    --probe-mode profile_fast \
    ${limit_args[@]+"${limit_args[@]}"}
  compact_metrics "${work_dir}"
}

run_llava_pruned_textocr() {
  local selector="$1"
  local ratio="$2"
  local tag="$3"
  local work_dir="runs/llava_textocr_hard/llava15_7b_textocr_hard_full1000_${tag}"
  if has_pruned_run "${work_dir}"; then
    log "skip LLaVA TextOCR pruned ${work_dir}"
    return
  fi
  log "run LLaVA TextOCR pruned ${work_dir}"
  run_to_log "llava_textocr_$(basename "${work_dir}")" \
    env CUDA_VISIBLE_DEVICES="${GPU_ID}" TOKENIZERS_PARALLELISM=false \
    conda run --no-capture-output -n "${CONDA_ENV}" python -u -m recap.cli run-llava-pruned \
    --input data/textocr_val_hard_probes_500img.jsonl \
    --is-probes \
    --work-dir "${work_dir}" \
    --pretrained "${LLAVA_PATH}" \
    --device cuda \
    --device-map auto \
    --dtype "${LLAVA_DTYPE}" \
    --trust-remote-code \
    --attn-implementation eager \
    --selector "${selector}" \
    --keep-ratio "${ratio}" \
    ${limit_args[@]+"${limit_args[@]}"}
}

run_internvl_pruned_textocr() {
  local selector="$1"
  local ratio="$2"
  local tag="$3"
  local work_dir="runs/internvl_textocr_hard/internvl35_8b_textocr_hard_full1000_${tag}"
  if has_pruned_run "${work_dir}"; then
    log "skip InternVL TextOCR pruned ${work_dir}"
    return
  fi
  log "run InternVL TextOCR pruned ${work_dir}"
  run_to_log "internvl_textocr_$(basename "${work_dir}")" \
    env CUDA_VISIBLE_DEVICES="${GPU_ID}" TOKENIZERS_PARALLELISM=false \
    conda run --no-capture-output -n "${CONDA_ENV}" python -u -m recap.cli run-internvl-pruned \
    --input data/textocr_val_hard_probes_500img.jsonl \
    --is-probes \
    --work-dir "${work_dir}" \
    --pretrained "${INTERNVL_PATH}" \
    --device cuda \
    --device-map auto \
    --dtype "${INTERNVL_DTYPE}" \
    --min-patches "${INTERNVL_MIN_PATCHES}" \
    --max-patches "${INTERNVL_MAX_PATCHES}" \
    --selector "${selector}" \
    --keep-ratio "${ratio}" \
    ${limit_args[@]+"${limit_args[@]}"}
}

calibrate_internvl_textocr() {
  local raw_dir="$1"
  local out_dir="$2"
  if has_score_run "${out_dir}"; then
    log "skip InternVL calibration ${out_dir}"
    return
  fi
  log "calibrate InternVL ${raw_dir} -> ${out_dir}"
  py scripts/apply_yesno_threshold.py \
    --score "${raw_dir}/probe_scores.jsonl" \
    --out-dir "${out_dir}" \
    --threshold-split dev \
    --eval-split test
}

run_llava_pruned_gsr() {
  local selector="$1"
  local ratio="$2"
  local tag="$3"
  local work_dir="runs/llava_prune/llava15_7b_gsr_coco_spatial_profile_fast_${tag}"
  if has_pruned_run "${work_dir}"; then
    log "skip LLaVA GSR pruned ${work_dir}"
    return
  fi
  log "run LLaVA GSR pruned ${work_dir}"
  run_to_log "llava_gsr_$(basename "${work_dir}")" \
    env CUDA_VISIBLE_DEVICES="${GPU_ID}" TOKENIZERS_PARALLELISM=false \
    conda run --no-capture-output -n "${CONDA_ENV}" python -u -m recap.cli run-llava-pruned \
    --input data/recap_gsrbench_coco_spatial_two.jsonl \
    --work-dir "${work_dir}" \
    --pretrained "${LLAVA_PATH}" \
    --device cuda \
    --device-map auto \
    --dtype "${LLAVA_DTYPE}" \
    --trust-remote-code \
    --attn-implementation eager \
    --selector "${selector}" \
    --keep-ratio "${ratio}" \
    --keep-non-left-right \
    --probe-mode profile_fast \
    ${limit_args[@]+"${limit_args[@]}"}
}

run_internvl_pruned_gsr() {
  local selector="$1"
  local ratio="$2"
  local tag="$3"
  local work_dir="runs/internvl_prune/internvl35_8b_gsr_coco_spatial_profile_fast_${tag}"
  if has_pruned_run "${work_dir}"; then
    log "skip InternVL GSR pruned ${work_dir}"
    return
  fi
  log "run InternVL GSR pruned ${work_dir}"
  run_to_log "internvl_gsr_$(basename "${work_dir}")" \
    env CUDA_VISIBLE_DEVICES="${GPU_ID}" TOKENIZERS_PARALLELISM=false \
    conda run --no-capture-output -n "${CONDA_ENV}" python -u -m recap.cli run-internvl-pruned \
    --input data/recap_gsrbench_coco_spatial_two.jsonl \
    --work-dir "${work_dir}" \
    --pretrained "${INTERNVL_PATH}" \
    --device cuda \
    --device-map auto \
    --dtype "${INTERNVL_DTYPE}" \
    --min-patches "${INTERNVL_MIN_PATCHES}" \
    --max-patches "${INTERNVL_MAX_PATCHES}" \
    --selector "${selector}" \
    --keep-ratio "${ratio}" \
    --keep-non-left-right \
    --probe-mode profile_fast \
    ${limit_args[@]+"${limit_args[@]}"}
}

if [[ "${RUN_RELATION_DIRECT}" == "1" ]]; then
  run_llava_direct data/rice_vsr_random_test_other_relations.jsonl \
    runs/rice_v5/llava15_7b_profile_fast_vsr_other_relations
  run_llava_direct data/rice_whatsup_controlled_other_relations.jsonl \
    runs/rice_v5/llava15_7b_profile_fast_whatsup_controlled_other_relations
  run_llava_direct data/recap_gsrbench_coco_spatial_two.jsonl \
    runs/rice_v5/llava15_7b_profile_fast_gsrbench_coco_spatial_two
  run_internvl_direct data/rice_whatsup_controlled_other_relations.jsonl \
    runs/rice_v5/internvl3_5_8b_profile_fast_whatsup_controlled_other_relations
fi

if [[ "${RUN_TEXTOCR_CAUSAL}" == "1" ]]; then
  run_llava_pruned_textocr topk 0.40 topk_0p40
  run_llava_pruned_textocr bottomk 0.40 bottomk_0p40
  run_internvl_pruned_textocr topk 0.50 topk_0p50
  run_internvl_pruned_textocr bottomk 0.50 bottomk_0p50
  calibrate_internvl_textocr \
    runs/internvl_textocr_hard/internvl35_8b_textocr_hard_full1000_topk_0p50 \
    runs/internvl_textocr_hard/calibrated_test_topk0p50_devthr
  calibrate_internvl_textocr \
    runs/internvl_textocr_hard/internvl35_8b_textocr_hard_full1000_bottomk_0p50 \
    runs/internvl_textocr_hard/calibrated_test_bottomk0p50_devthr
fi

if [[ "${RUN_SPATIAL_PRUNED}" == "1" ]]; then
  run_llava_pruned_gsr embed_topk 0.40 embed_topk_0p40
  run_llava_pruned_gsr grid 0.40 grid_0p40
  run_internvl_pruned_gsr target_embed_topk 0.50 target_embed_topk_0p50
  run_internvl_pruned_gsr grid 0.50 grid_0p50
fi

if [[ "${RUN_REPORTS}" == "1" ]]; then
  py scripts/build_hard_evidence_report.py
  py scripts/build_textocr_hard_cross_model_report.py
  py scripts/build_cross_model_batch_prefill_report.py
  py scripts/build_future_experiment_status.py
fi

log "future experiment queue completed"
