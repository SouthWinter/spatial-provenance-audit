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

# Fixed runs. Keep the no-prune topk@1.00 run first so paired stats have a
# stable baseline even if the script is interrupted later.
FULL_SELECTOR="${FULL_SELECTOR:-topk}"
FULL_RATIO="${FULL_RATIO:-1.00}"
FIXED_SELECTORS="${FIXED_SELECTORS:-topk hybrid}"
FIXED_RATIOS="${FIXED_RATIOS:-0.25 0.30 0.40 0.50}"
BASELINE_SELECTORS="${BASELINE_SELECTORS:-random grid}"
BASELINE_RATIOS="${BASELINE_RATIOS:-0.50}"
RUN_FIXED="${RUN_FIXED:-1}"

# Optional ECR policy stage. Run with RUN_BUILD_POLICY=1 after the 0.30/0.40/0.50
# hybrid runs finish; then RUN_POLICY=1 scores the learned budget assignment.
RUN_BUILD_POLICY="${RUN_BUILD_POLICY:-0}"
RUN_POLICY="${RUN_POLICY:-0}"
POLICY_DIR="${POLICY_DIR:-${OUT_ROOT}/${MODEL_TAG}_${RUN_TAG}_hybrid_ecr_policy_0p30_0p40_0p50}"

mkdir -p "${OUT_ROOT}"

ratio_tag() {
  local ratio="$1"
  echo "${ratio/./p}"
}

run_pruned() {
  local selector="$1"
  local ratio="$2"
  local extra_tag="${3:-}"
  local tag
  tag="$(ratio_tag "${ratio}")"
  local work_dir="${OUT_ROOT}/${MODEL_TAG}_${RUN_TAG}_${selector}_${tag}${extra_tag}"
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
  CUDA_VISIBLE_DEVICES="${GPU_ID}" TOKENIZERS_PARALLELISM=false "${args[@]}"
}

run_policy() {
  local work_dir="${OUT_ROOT}/${MODEL_TAG}_${RUN_TAG}_hybrid_ecr_policy_0p30_0p40_0p50_scored"
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
  CUDA_VISIBLE_DEVICES="${GPU_ID}" TOKENIZERS_PARALLELISM=false "${args[@]}"
}

summarize_and_pair() {
  python scripts/summarize_prune_main.py \
    --runs-dir "${OUT_ROOT}" \
    --csv-out "${OUT_ROOT}/${MODEL_TAG}_${RUN_TAG}_summary.csv" \
    --md-out "${OUT_ROOT}/${MODEL_TAG}_${RUN_TAG}_summary.md"

  local full_run="${MODEL_TAG}_${RUN_TAG}_${FULL_SELECTOR}_$(ratio_tag "${FULL_RATIO}")"
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
  add_comparison "hybrid0p30_vs_full" "${MODEL_TAG}_${RUN_TAG}_hybrid_0p30" "${full_run}"
  add_comparison "hybrid0p40_vs_full" "${MODEL_TAG}_${RUN_TAG}_hybrid_0p40" "${full_run}"
  add_comparison "hybrid0p50_vs_full" "${MODEL_TAG}_${RUN_TAG}_hybrid_0p50" "${full_run}"
  add_comparison "topk0p50_vs_full" "${MODEL_TAG}_${RUN_TAG}_topk_0p50" "${full_run}"
  add_comparison "random0p50_vs_full" "${MODEL_TAG}_${RUN_TAG}_random_0p50" "${full_run}"
  add_comparison "grid0p50_vs_full" "${MODEL_TAG}_${RUN_TAG}_grid_0p50" "${full_run}"
  add_comparison \
    "ecr_policy_vs_full" \
    "${MODEL_TAG}_${RUN_TAG}_hybrid_ecr_policy_0p30_0p40_0p50_scored" \
    "${full_run}"
  add_comparison \
    "ecr_policy_vs_hybrid0p40" \
    "${MODEL_TAG}_${RUN_TAG}_hybrid_ecr_policy_0p30_0p40_0p50_scored" \
    "${MODEL_TAG}_${RUN_TAG}_hybrid_0p40"

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

if [[ "${RUN_FIXED}" == "1" ]]; then
  run_pruned "${FULL_SELECTOR}" "${FULL_RATIO}"

  for selector in ${FIXED_SELECTORS}; do
    for ratio in ${FIXED_RATIOS}; do
      if [[ "${selector}" == "${FULL_SELECTOR}" && "${ratio}" == "${FULL_RATIO}" ]]; then
        continue
      fi
      run_pruned "${selector}" "${ratio}"
    done
  done

  for selector in ${BASELINE_SELECTORS}; do
    for ratio in ${BASELINE_RATIOS}; do
      run_pruned "${selector}" "${ratio}"
    done
  done
fi

if [[ "${RUN_BUILD_POLICY}" == "1" ]]; then
  python scripts/build_ecr_budget_policy.py \
    --runs-dir "${OUT_ROOT}" \
    --trace-run "${MODEL_TAG}_${RUN_TAG}_hybrid_0p30" \
    --score-run "0.30:${MODEL_TAG}_${RUN_TAG}_hybrid_0p30" \
    --score-run "0.40:${MODEL_TAG}_${RUN_TAG}_hybrid_0p40" \
    --score-run "0.50:${MODEL_TAG}_${RUN_TAG}_hybrid_0p50" \
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
