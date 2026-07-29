#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

source "${SCRIPT_DIR}/env_local.sh"

MODEL_PATH="${MODEL_PATH:-${QWEN3_PATH}}"
MODEL_TAG="${MODEL_TAG:-qwen3_8b}"
INPUT="${INPUT:-data/textocr_val_yesno_probes_500.jsonl}"
OUT_ROOT="${OUT_ROOT:-runs/prune_textocr}"
RUN_TAG="${RUN_TAG:-textocr_yesno_500}"
GPU_ID="${GPU_ID:-0}"
MIN_PIXELS="${MIN_PIXELS:-802816}"
MAX_PIXELS="${MAX_PIXELS:-802816}"
LIMIT="${LIMIT:-}"
PARALLEL_JOBS="${PARALLEL_JOBS:-3}"

FULL_SELECTOR="${FULL_SELECTOR:-topk}"
FULL_RATIO="${FULL_RATIO:-1.00}"
FIXED_SELECTORS="${FIXED_SELECTORS:-topk hybrid}"
FIXED_RATIOS="${FIXED_RATIOS:-0.25 0.30 0.40 0.50}"
BASELINE_SELECTORS="${BASELINE_SELECTORS:-random grid}"
BASELINE_RATIOS="${BASELINE_RATIOS:-0.50}"

RUN_BUILD_POLICY="${RUN_BUILD_POLICY:-1}"
RUN_POLICY="${RUN_POLICY:-1}"
POLICY_DIR="${POLICY_DIR:-${OUT_ROOT}/${MODEL_TAG}_${RUN_TAG}_hybrid_ecr_policy_0p30_0p40_0p50}"

mkdir -p "${OUT_ROOT}"

ratio_tag() {
  local ratio="$1"
  echo "${ratio/./p}"
}

run_name() {
  local selector="$1"
  local ratio="$2"
  echo "${MODEL_TAG}_${RUN_TAG}_${selector}_$(ratio_tag "${ratio}")"
}

is_complete_run() {
  local work_dir="$1"
  [[ -s "${work_dir}/metrics.json" && -s "${work_dir}/sample_scores.jsonl" && -s "${work_dir}/prune_traces.jsonl" ]]
}

run_pruned() {
  local selector="$1"
  local ratio="$2"
  local name
  name="$(run_name "${selector}" "${ratio}")"
  local work_dir="${OUT_ROOT}/${name}"
  local log_file="${work_dir}/run.log"

  if is_complete_run "${work_dir}"; then
    echo "[skip] ${name}"
    return 0
  fi

  if [[ -d "${work_dir}" ]]; then
    echo "[redo] removing incomplete ${name}"
    rm -rf "${work_dir}"
  fi
  mkdir -p "${work_dir}"

  local args=(
    python -u -m recap.cli run-qwen-pruned
    --input "${INPUT}"
    --is-probes
    --work-dir "${work_dir}"
    --pretrained "${MODEL_PATH}"
    --device cuda
    --device-map auto
    --attn-implementation eager
    --min-pixels "${MIN_PIXELS}"
    --max-pixels "${MAX_PIXELS}"
    --selector "${selector}"
    --keep-ratio "${ratio}"
  )
  if [[ -n "${LIMIT}" ]]; then
    args+=(--limit "${LIMIT}")
  fi

  echo "[start] ${name}"
  if CUDA_VISIBLE_DEVICES="${GPU_ID}" TOKENIZERS_PARALLELISM=false "${args[@]}" >"${log_file}" 2>&1; then
    echo "[done] ${name}"
  else
    echo "[fail] ${name}; see ${log_file}" >&2
    return 1
  fi
}

run_policy() {
  local name="${MODEL_TAG}_${RUN_TAG}_hybrid_ecr_policy_0p30_0p40_0p50_scored"
  local work_dir="${OUT_ROOT}/${name}"
  local log_file="${work_dir}/run.log"

  if is_complete_run "${work_dir}"; then
    echo "[skip] ${name}"
    return 0
  fi
  if [[ -d "${work_dir}" ]]; then
    echo "[redo] removing incomplete ${name}"
    rm -rf "${work_dir}"
  fi
  mkdir -p "${work_dir}"

  local args=(
    python -u -m recap.cli run-qwen-pruned
    --input "${INPUT}"
    --is-probes
    --work-dir "${work_dir}"
    --pretrained "${MODEL_PATH}"
    --device cuda
    --device-map auto
    --attn-implementation eager
    --min-pixels "${MIN_PIXELS}"
    --max-pixels "${MAX_PIXELS}"
    --selector hybrid
    --keep-ratio 0.30
    --budget-mode sensitivity_policy
    --budget-ratios "${POLICY_DIR}/budget_ratios.jsonl"
    --budget-ratio-key keep_ratio
  )
  if [[ -n "${LIMIT}" ]]; then
    args+=(--limit "${LIMIT}")
  fi

  echo "[start] ${name}"
  if CUDA_VISIBLE_DEVICES="${GPU_ID}" TOKENIZERS_PARALLELISM=false "${args[@]}" >"${log_file}" 2>&1; then
    echo "[done] ${name}"
  else
    echo "[fail] ${name}; see ${log_file}" >&2
    return 1
  fi
}

summarize_and_pair() {
  python scripts/summarize_prune_main.py \
    --runs-dir "${OUT_ROOT}" \
    --csv-out "${OUT_ROOT}/${MODEL_TAG}_${RUN_TAG}_summary.csv" \
    --md-out "${OUT_ROOT}/${MODEL_TAG}_${RUN_TAG}_summary.md"

  local full_run
  full_run="$(run_name "${FULL_SELECTOR}" "${FULL_RATIO}")"
  local comparisons=()
  add_comparison() {
    local label="$1"
    local left="$2"
    local right="$3"
    if [[ -f "${OUT_ROOT}/${left}/sample_scores.jsonl" && -f "${OUT_ROOT}/${right}/sample_scores.jsonl" ]]; then
      comparisons+=("${label}:${left}:${right}")
    else
      echo "Skipping paired comparison ${label}; missing ${left} or ${right}." >&2
    fi
  }
  add_comparison "hybrid0p30_vs_full" "$(run_name hybrid 0.30)" "${full_run}"
  add_comparison "hybrid0p40_vs_full" "$(run_name hybrid 0.40)" "${full_run}"
  add_comparison "hybrid0p50_vs_full" "$(run_name hybrid 0.50)" "${full_run}"
  add_comparison "topk0p50_vs_full" "$(run_name topk 0.50)" "${full_run}"
  add_comparison "random0p50_vs_full" "$(run_name random 0.50)" "${full_run}"
  add_comparison "grid0p50_vs_full" "$(run_name grid 0.50)" "${full_run}"
  add_comparison \
    "ecr_policy_vs_full" \
    "${MODEL_TAG}_${RUN_TAG}_hybrid_ecr_policy_0p30_0p40_0p50_scored" \
    "${full_run}"
  add_comparison \
    "ecr_policy_vs_hybrid0p40" \
    "${MODEL_TAG}_${RUN_TAG}_hybrid_ecr_policy_0p30_0p40_0p50_scored" \
    "$(run_name hybrid 0.40)"

  if [[ "${#comparisons[@]}" -eq 0 ]]; then
    echo "No paired comparisons available yet."
    return
  fi

  local args=(
    python scripts/paired_prune_stats.py
    --runs-dir "${OUT_ROOT}"
    --out-csv "${OUT_ROOT}/${MODEL_TAG}_${RUN_TAG}_paired_stats.csv"
    --out-md "${OUT_ROOT}/${MODEL_TAG}_${RUN_TAG}_paired_stats.md"
  )
  for comparison in "${comparisons[@]}"; do
    args+=(--comparison "${comparison}")
  done
  "${args[@]}"
}

tasks=("${FULL_SELECTOR}:${FULL_RATIO}")
for selector in ${FIXED_SELECTORS}; do
  for ratio in ${FIXED_RATIOS}; do
    if [[ "${selector}" == "${FULL_SELECTOR}" && "${ratio}" == "${FULL_RATIO}" ]]; then
      continue
    fi
    tasks+=("${selector}:${ratio}")
  done
done
for selector in ${BASELINE_SELECTORS}; do
  for ratio in ${BASELINE_RATIOS}; do
    tasks+=("${selector}:${ratio}")
  done
done

pids=()
cleanup() {
  local running
  running="$(jobs -pr)"
  if [[ -n "${running}" ]]; then
    kill ${running} 2>/dev/null || true
  fi
}
trap cleanup INT TERM

fail=0
for task in "${tasks[@]}"; do
  while [[ "$(jobs -pr | wc -l)" -ge "${PARALLEL_JOBS}" ]]; do
    sleep 5
  done
  IFS=: read -r selector ratio <<<"${task}"
  run_pruned "${selector}" "${ratio}" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    fail=1
  fi
done
trap - INT TERM
if [[ "${fail}" -ne 0 ]]; then
  exit 1
fi

if [[ "${RUN_BUILD_POLICY}" == "1" ]]; then
  python scripts/build_ecr_budget_policy.py \
    --runs-dir "${OUT_ROOT}" \
    --trace-run "$(run_name hybrid 0.30)" \
    --score-run "0.30:$(run_name hybrid 0.30)" \
    --score-run "0.40:$(run_name hybrid 0.40)" \
    --score-run "0.50:$(run_name hybrid 0.50)" \
    --output-dir "${POLICY_DIR}" \
    --base-budget 0.30 \
    --mid-budget 0.40 \
    --high-budget 0.50 \
    --hfpr-target 0.05 \
    --keep-target 0.45
fi

if [[ "${RUN_POLICY}" == "1" ]]; then
  run_policy
fi

summarize_and_pair
