#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

source "${SCRIPT_DIR}/env_local.sh"

CONDA_ENV="${CONDA_ENV:-llama}"
GPU_ID="${GPU_ID:-0}"
INPUT="${INPUT:-data/textocr_val_hard_probes_500img.jsonl}"
INDEX_DIR="${INDEX_DIR:-runs/textocr_deletion_restoration/qwen_target30_indices}"
OUT_ROOT="${OUT_ROOT:-runs/textocr_deletion_restoration/qwen_target30_runs}"
LOG_DIR="${LOG_DIR:-${OUT_ROOT}/logs}"
LIMIT="${LIMIT:-}"

QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-${QWEN3_PATH}}"
QWEN_DTYPE="${QWEN_DTYPE:-auto}"
QWEN_MIN_PIXELS="${QWEN_MIN_PIXELS:-802816}"
QWEN_MAX_PIXELS="${QWEN_MAX_PIXELS:-802816}"

VARIANTS="${VARIANTS:-selected,remove_evidence,restore_evidence_0p25,restore_evidence_0p50,restore_evidence_1p00,restore_random_0p25,restore_random_0p50,restore_random_1p00}"

mkdir -p "${LOG_DIR}"

add_limit_args() {
  if [[ -n "${LIMIT}" ]]; then
    printf '%s\n' --limit "${LIMIT}"
  fi
}

has_run() {
  local dir="$1"
  [[ -s "${dir}/metrics.json" && -s "${dir}/probe_scores.jsonl" && -s "${dir}/sample_scores.jsonl" && -s "${dir}/prune_traces.jsonl" ]]
}

run_variant() {
  local variant="$1"
  local index_file="${INDEX_DIR}/${variant}.jsonl"
  local work_dir="${OUT_ROOT}/${variant}"
  local log_path="${LOG_DIR}/${variant}.log"
  if [[ ! -s "${index_file}" ]]; then
    echo "Missing index file: ${index_file}" >&2
    return 2
  fi
  if has_run "${work_dir}"; then
    echo "[skip] ${variant} -> ${work_dir}"
    return
  fi
  echo "[run] ${variant} -> ${work_dir}"
  if env CUDA_VISIBLE_DEVICES="${GPU_ID}" TOKENIZERS_PARALLELISM=false USE_TF=0 TRANSFORMERS_NO_TF=1 \
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
      --selector target_embed_topk \
      --keep-ratio 0.30 \
      --kept-indices "${index_file}" \
      $(add_limit_args) >"${log_path}" 2>&1; then
    tail -n 20 "${log_path}"
  else
    local status=$?
    tail -n 80 "${log_path}" || true
    return "${status}"
  fi
}

IFS=',' read -r -a variant_array <<<"${VARIANTS}"
for variant in "${variant_array[@]}"; do
  run_variant "${variant}"
done

conda run --no-capture-output -n "${CONDA_ENV}" python scripts/build_textocr_deletion_restoration_report.py \
  --runs-root "${OUT_ROOT}" \
  --output-dir "${OUT_ROOT}"

echo "Qwen TextOCR-Hard deletion/restoration queue completed."
