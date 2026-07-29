#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

source "${SCRIPT_DIR}/env_local.sh"

LIMIT="${LIMIT:-}"
GRID_SIZE="${GRID_SIZE:-24}"
KEEP_RATIOS="${KEEP_RATIOS:-0.15,0.25,0.35,0.5,0.7,1.0}"
SELECTORS="${SELECTORS:-random,grid,center}"
RUN_VSR="${RUN_VSR:-0}"
RUN_WHATSUP="${RUN_WHATSUP:-1}"
RUN_GSRBENCH="${RUN_GSRBENCH:-1}"
GSRBENCH_DATASET="${GSRBENCH_DATASET:-coco_spatial_two}"
RUN_TEXTOCR="${RUN_TEXTOCR:-1}"
TEXTOCR_LIMIT="${TEXTOCR_LIMIT:-20}"
TEXTOCR_MAX_REGIONS="${TEXTOCR_MAX_REGIONS:-80}"

mkdir -p data runs/prune_phase_a

inputs=()

if [[ "${RUN_VSR}" == "1" ]]; then
  vsr_args=(
    python -m recap.cli prepare-vsr
    --dataset-path random
    --split test
    --coco-root "${COCO_ROOT}"
    --image-dir data/vsr_random_images
    --relation-families "${RELATION_FAMILIES}"
    --output data/rice_vsr_random_test_other_relations.jsonl
  )
  if [[ -n "${LIMIT}" ]]; then
    vsr_args+=(--limit "${LIMIT}")
  fi
  "${vsr_args[@]}"
  inputs+=(rice_vsr_random_test_other_relations)
fi

if [[ "${RUN_WHATSUP}" == "1" ]]; then
  whatsup_args=(
    python -m recap.cli prepare-whatsup
    --dataset all_controlled
    --root-dir "${WHATSUP_ROOT}"
    --relation-families "${RELATION_FAMILIES}"
    --output data/rice_whatsup_controlled_other_relations.jsonl
  )
  if [[ "${WHATSUP_DOWNLOAD}" == "1" ]]; then
    whatsup_args+=(--download)
  fi
  if [[ -n "${LIMIT}" ]]; then
    whatsup_args+=(--limit "${LIMIT}")
  fi
  "${whatsup_args[@]}"
  inputs+=(rice_whatsup_controlled_other_relations)
fi

if [[ "${RUN_GSRBENCH}" == "1" ]]; then
  gsr_name="recap_gsrbench_${GSRBENCH_DATASET}"
  gsr_args=(
    python -m recap.cli prepare-gsrbench
    --dataset "${GSRBENCH_DATASET}"
    --root-dir "${GSRBENCH_ROOT}"
    --coco-root "${COCO_ROOT}"
    --relation-families "${RELATION_FAMILIES}"
    --output "data/${gsr_name}_raw.jsonl"
  )
  if [[ "${GSRBENCH_DOWNLOAD}" == "1" ]]; then
    gsr_args+=(--download)
  fi
  if [[ -n "${LIMIT}" ]]; then
    gsr_args+=(--limit "${LIMIT}")
  fi
  if [[ -n "${GSRBENCH_GQA_ROOT:-}" ]]; then
    gsr_args+=(--gqa-root "${GSRBENCH_GQA_ROOT}")
  fi
  "${gsr_args[@]}"
  if [[ "${GSRBENCH_DATASET}" == "coco_spatial_two" || "${GSRBENCH_DATASET}" == "external_two_object" ]]; then
    python -m recap.cli attach-coco-evidence \
      --input "data/${gsr_name}_raw.jsonl" \
      --output "data/${gsr_name}.jsonl" \
      --instances-file "${COCO_INSTANCES_VAL}" \
      --report-output "runs/prune_phase_a/${gsr_name}_coco_evidence_report.json"
  else
    cp "data/${gsr_name}_raw.jsonl" "data/${gsr_name}.jsonl"
  fi
  inputs+=("${gsr_name}")
fi

if [[ "${RUN_TEXTOCR}" == "1" ]]; then
  textocr_args=(
    python -m recap.cli prepare-textocr-regions
    --split val
    --output-dir data/textocr
    --image-root "${TEXTOCR_ROOT}"
    --output data/textocr_val_regions.jsonl
  )
  if [[ -n "${TEXTOCR_LIMIT}" ]]; then
    textocr_args+=(--limit "${TEXTOCR_LIMIT}")
  fi
  if [[ -n "${TEXTOCR_MAX_REGIONS}" ]]; then
    textocr_args+=(--max-regions "${TEXTOCR_MAX_REGIONS}")
  fi
  "${textocr_args[@]}"
  inputs+=(textocr_val_regions)
fi

for name in "${inputs[@]}"; do
  input_path="data/${name}.jsonl"
  python -m recap.cli prune-audit \
    --input "${input_path}" \
    --output "runs/prune_phase_a/${name}_prune_audit.json"

  python -m recap.cli prune-offline-baselines \
    --input "${input_path}" \
    --output "runs/prune_phase_a/${name}_offline_prune_baselines.jsonl" \
    --summary-output "runs/prune_phase_a/${name}_offline_prune_summary.json" \
    --grid-size "${GRID_SIZE}" \
    --keep-ratios "${KEEP_RATIOS}" \
    --selectors "${SELECTORS}"
done
