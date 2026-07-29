#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

source "${SCRIPT_DIR}/env_local.sh"

MODEL_PATH="${MODEL_PATH:-${QWEN_PATH}}"
MODEL_TAG="${MODEL_TAG:-qwen3_2b}"
INPUT="${INPUT:-data/recap_gsrbench_coco_spatial_two.jsonl}"
OUT_ROOT="${OUT_ROOT:-runs/prune_main}"
MIN_PIXELS="${MIN_PIXELS:-50176}"
MAX_PIXELS="${MAX_PIXELS:-50176}"
FIXED_RATIOS="${FIXED_RATIOS:-0.25 0.35 0.50 0.70 1.00}"
FIXED_SELECTORS="${FIXED_SELECTORS:-topk rise}"
BASELINE_SELECTORS="${BASELINE_SELECTORS:-random grid center}"
BASELINE_RATIOS="${BASELINE_RATIOS:-0.50}"
if [[ -z "${RISK_SCORES:-}" ]]; then
  if [[ "${MODEL_TAG}" == *"8b"* ]]; then
    RISK_SCORES="runs/rice_v5/qwen3_8b_recap_profile_gsrbench_coco_spatial_two/sample_scores.jsonl"
  else
    RISK_SCORES="runs/rice_v5/qwen3_2b_recap_profile_gsrbench_coco_spatial_two/sample_scores.jsonl"
  fi
fi
RISK_KEY="${RISK_KEY:-recap_evidence_risk}"
RUN_ADAPTIVE="${RUN_ADAPTIVE:-1}"
LIMIT="${LIMIT:-}"

run_fixed() {
  local selector="$1"
  local ratio="$2"
  local ratio_tag="${ratio/./p}"
  local work_dir="${OUT_ROOT}/${MODEL_TAG}_gsr_orig_${selector}_${ratio_tag}"
  local args=(
    python -u -m recap.cli run-qwen-pruned
    --input "${INPUT}"
    --work-dir "${work_dir}"
    --pretrained "${MODEL_PATH}"
    --device cuda
    --device-map auto
    --attn-implementation eager
    --min-pixels "${MIN_PIXELS}"
    --max-pixels "${MAX_PIXELS}"
    --keep-non-left-right
    --probe-mode profile_fast
    --orig-only
    --selector "${selector}"
    --keep-ratio "${ratio}"
  )
  if [[ -n "${LIMIT}" ]]; then
    args+=(--limit "${LIMIT}")
  fi
  "${args[@]}"
}

for selector in ${FIXED_SELECTORS}; do
  for ratio in ${FIXED_RATIOS}; do
    run_fixed "${selector}" "${ratio}"
  done
done

for selector in ${BASELINE_SELECTORS}; do
  for ratio in ${BASELINE_RATIOS}; do
    run_fixed "${selector}" "${ratio}"
  done
done

if [[ "${RUN_ADAPTIVE}" == "1" ]]; then
  adaptive_dir="${OUT_ROOT}/${MODEL_TAG}_gsr_orig_rise_adaptive_${RISK_KEY}_0p15_0p70"
  adaptive_args=(
    python -u -m recap.cli run-qwen-pruned
    --input "${INPUT}"
    --work-dir "${adaptive_dir}"
    --pretrained "${MODEL_PATH}"
    --device cuda
    --device-map auto
    --attn-implementation eager
    --min-pixels "${MIN_PIXELS}"
    --max-pixels "${MAX_PIXELS}"
    --keep-non-left-right
    --probe-mode profile_fast
    --orig-only
    --selector rise
    --keep-ratio 0.35
    --budget-mode risk_adaptive
    --risk-scores "${RISK_SCORES}"
    --risk-key "${RISK_KEY}"
    --rho-min 0.15
    --rho-max 0.70
  )
  if [[ -n "${LIMIT}" ]]; then
    adaptive_args+=(--limit "${LIMIT}")
  fi
  "${adaptive_args[@]}"
fi
