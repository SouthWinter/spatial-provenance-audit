#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

if [[ -f "${SCRIPT_DIR}/env_local.sh" ]]; then
  source "${SCRIPT_DIR}/env_local.sh"
fi

export USE_TF=0
export TRANSFORMERS_NO_TF=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL="${MODEL:-qwen}"
GPU_ID="${GPU_ID:-0}"
JOBS="${JOBS:-2}"
INPUT_DIR="${INPUT_DIR:-data/textocr_image_free_controls/confirmation}"
OUT_ROOT="${OUT_ROOT:-runs/problem_optimization_audit/image_free_controls}"
LIMIT="${LIMIT:-}"
QWEN_MIN_PIXELS="${QWEN_MIN_PIXELS:-802816}"
QWEN_MAX_PIXELS="${QWEN_MAX_PIXELS:-802816}"

case "${MODEL}" in
  qwen) : "${QWEN3_PATH:?Set QWEN3_PATH to the Qwen3-VL checkpoint}" ;;
  llava) : "${LLAVA_PATH:?Set LLAVA_PATH to the LLaVA checkpoint}" ;;
  internvl) : "${INTERNVL_PATH:?Set INTERNVL_PATH to the InternVL checkpoint}" ;;
esac

inputs=(
  "blank:${INPUT_DIR}/blank.jsonl"
  "image_mismatch_seed101:${INPUT_DIR}/image_mismatch_seed101.jsonl"
  "image_mismatch_seed202:${INPUT_DIR}/image_mismatch_seed202.jsonl"
  "image_mismatch_seed303:${INPUT_DIR}/image_mismatch_seed303.jsonl"
)

limit_args=()
if [[ -n "${LIMIT}" ]]; then
  limit_args+=(--limit "${LIMIT}")
fi

run_one() {
  local tag="$1"
  local input="$2"
  local work_dir="${OUT_ROOT}/${MODEL}/${tag}"
  local log_dir="${OUT_ROOT}/${MODEL}/logs"
  local expected_rows
  mkdir -p "${work_dir}" "${log_dir}"

  if [[ -n "${LIMIT}" ]]; then
    expected_rows="${LIMIT}"
  else
    expected_rows="$(wc -l < "${input}")"
  fi
  if [[ -f "${work_dir}/probe_scores.jsonl" ]]; then
    local rows
    rows="$(wc -l < "${work_dir}/probe_scores.jsonl")"
    if (( rows >= expected_rows )); then
      echo "skip ${MODEL}/${tag}: ${rows} completed probes"
      return
    fi
  fi

  case "${MODEL}" in
    qwen)
      CUDA_VISIBLE_DEVICES="${GPU_ID}" TOKENIZERS_PARALLELISM=false \
        python -u -m recap.cli run-qwen-pruned \
          --input "${input}" \
          --is-probes \
          --work-dir "${work_dir}" \
          --pretrained "${QWEN3_PATH}" \
          --selector topk \
          --keep-ratio 1.0 \
          --seed 17 \
          --device cuda \
          --device-map auto \
          --attn-implementation eager \
          --min-pixels "${QWEN_MIN_PIXELS}" \
          --max-pixels "${QWEN_MAX_PIXELS}" \
          ${limit_args[@]+"${limit_args[@]}"} \
          >"${log_dir}/${tag}.log" 2>&1
      ;;
    llava)
      CUDA_VISIBLE_DEVICES="${GPU_ID}" TOKENIZERS_PARALLELISM=false \
        python -u -m recap.cli run-llava-direct \
          --input "${input}" \
          --is-probes \
          --work-dir "${work_dir}" \
          --pretrained "${LLAVA_PATH}" \
          --device cuda \
          --device-map auto \
          --dtype float16 \
          --trust-remote-code \
          --attn-implementation eager \
          ${limit_args[@]+"${limit_args[@]}"} \
          >"${log_dir}/${tag}.log" 2>&1
      ;;
    internvl)
      CUDA_VISIBLE_DEVICES="${GPU_ID}" TOKENIZERS_PARALLELISM=false \
        python -u -m recap.cli run-internvl-direct \
          --input "${input}" \
          --is-probes \
          --work-dir "${work_dir}" \
          --pretrained "${INTERNVL_PATH}" \
          --device cuda \
          --device-map auto \
          --dtype bfloat16 \
          --min-patches 1 \
          --max-patches 12 \
          ${limit_args[@]+"${limit_args[@]}"} \
          >"${log_dir}/${tag}.log" 2>&1
      ;;
    *)
      echo "Unsupported MODEL=${MODEL}; expected qwen, llava, or internvl" >&2
      return 2
      ;;
  esac
}

pids=()
for item in "${inputs[@]}"; do
  tag="${item%%:*}"
  input="${item#*:}"
  run_one "${tag}" "${input}" &
  pids+=("$!")
  if (( ${#pids[@]} >= JOBS )); then
    for pid in "${pids[@]}"; do
      wait "${pid}"
    done
    pids=()
  fi
done

if (( ${#pids[@]} > 0 )); then
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
fi

echo "Completed ${MODEL} image-free controls under ${OUT_ROOT}/${MODEL}"
