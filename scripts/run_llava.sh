#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-llava-hf/llava-1.5-7b-hf}"
MODEL_TAG="${MODEL_TAG:-llava15_7b_hf}"
LLAVA_DTYPE="${LLAVA_DTYPE:-float16}"
COCO_ROOT="${COCO_ROOT:-data/coco2017}"
WHATSUP_ROOT="${WHATSUP_ROOT:-data/whatsup}"
GSRBENCH_ROOT="${GSRBENCH_ROOT:-${WHATSUP_ROOT}}"
GSRBENCH_GQA_ROOT="${GSRBENCH_GQA_ROOT:-}"
GPU_ID="${GPU_ID:-0}"

RELATION_FAMILIES="${RELATION_FAMILIES:-left_right,vertical,topology,depth}"
PROBE_MODE="${PROBE_MODE:-recap}"
ORIG_ONLY="${ORIG_ONLY:-0}"
RUN_VSR="${RUN_VSR:-1}"
RUN_WHATSUP="${RUN_WHATSUP:-1}"
RUN_GSRBENCH="${RUN_GSRBENCH:-0}"
WHATSUP_DOWNLOAD="${WHATSUP_DOWNLOAD:-0}"
GSRBENCH_DOWNLOAD="${GSRBENCH_DOWNLOAD:-0}"
LIMIT="${LIMIT:-}"

run_compact_metrics() {
  local input_path="$1"
  local output_path="$2"
  local -a compact_args=(
    python -m recap.cli compact-metrics
    --input "${input_path}"
    --output "${output_path}"
  )
  if [[ "${INCLUDE_BY_FAMILY:-0}" == "1" ]]; then
    compact_args+=(--include-by-family)
  fi
  if [[ "${INCLUDE_BY_RELATION:-0}" == "1" ]]; then
    compact_args+=(--include-by-relation)
  fi
  "${compact_args[@]}"
}

run_recap() {
  local input_path="$1"
  local work_dir="$2"
  local -a pilot_args=(
    python -u -m recap.cli run-llava-direct
    --input "${input_path}"
    --work-dir "${work_dir}"
    --pretrained "${MODEL_PATH}"
    --device cuda
    --device-map auto
    --dtype "${LLAVA_DTYPE}"
    --trust-remote-code
    --attn-implementation eager
    --keep-non-left-right
    --probe-mode "${PROBE_MODE}"
  )
  if [[ -n "${LIMIT}" ]]; then
    pilot_args+=(--limit "${LIMIT}")
  fi
  if [[ "${ORIG_ONLY}" == "1" ]]; then
    pilot_args+=(--orig-only)
  fi
  CUDA_VISIBLE_DEVICES="${GPU_ID}" TOKENIZERS_PARALLELISM=false "${pilot_args[@]}"
}

mkdir -p data runs/rice_v5

if [[ "${RUN_VSR}" == "1" ]]; then
  python -m recap.cli prepare-vsr \
    --dataset-path random \
    --split test \
    --coco-root "${COCO_ROOT}" \
    --image-dir data/vsr_random_images \
    --relation-families "${RELATION_FAMILIES}" \
    --output data/rice_vsr_random_test_other_relations.jsonl

  python -m recap.cli validate-images \
    --input data/rice_vsr_random_test_other_relations.jsonl \
    --keep-non-left-right \
    --probe-mode "${PROBE_MODE}"

  python -m recap.cli audit-data \
    --input data/rice_vsr_random_test_other_relations.jsonl \
    --output data/rice_vsr_random_test_other_relations_audit.json

  run_recap \
    data/rice_vsr_random_test_other_relations.jsonl \
    "runs/rice_v5/${MODEL_TAG}_vsr_other_relations"

  run_compact_metrics \
    "runs/rice_v5/${MODEL_TAG}_vsr_other_relations/metrics.json" \
    "runs/rice_v5/${MODEL_TAG}_vsr_other_relations/metrics_compact.json"
fi

if [[ "${RUN_WHATSUP}" == "1" ]]; then
  if [[ "${WHATSUP_DOWNLOAD}" == "1" ]]; then
    python -m recap.cli prepare-whatsup \
      --dataset all_controlled \
      --root-dir "${WHATSUP_ROOT}" \
      --relation-families "${RELATION_FAMILIES}" \
      --output data/rice_whatsup_controlled_other_relations.jsonl \
      --download
  else
    python -m recap.cli prepare-whatsup \
      --dataset all_controlled \
      --root-dir "${WHATSUP_ROOT}" \
      --relation-families "${RELATION_FAMILIES}" \
      --output data/rice_whatsup_controlled_other_relations.jsonl
  fi

  python -m recap.cli validate-images \
    --input data/rice_whatsup_controlled_other_relations.jsonl \
    --keep-non-left-right \
    --probe-mode "${PROBE_MODE}"

  python -m recap.cli audit-data \
    --input data/rice_whatsup_controlled_other_relations.jsonl \
    --output data/rice_whatsup_controlled_other_relations_audit.json

  run_recap \
    data/rice_whatsup_controlled_other_relations.jsonl \
    "runs/rice_v5/${MODEL_TAG}_whatsup_controlled_other_relations"

  run_compact_metrics \
    "runs/rice_v5/${MODEL_TAG}_whatsup_controlled_other_relations/metrics.json" \
    "runs/rice_v5/${MODEL_TAG}_whatsup_controlled_other_relations/metrics_compact.json"

fi

if [[ "${RUN_GSRBENCH}" == "1" ]]; then
  gsr_args=(
    python -m recap.cli prepare-gsrbench
    --dataset external_two_object
    --root-dir "${GSRBENCH_ROOT}"
    --coco-root "${COCO_ROOT}"
    --relation-families "${RELATION_FAMILIES}"
    --output data/recap_gsrbench_external_two_object.jsonl
  )
  if [[ "${GSRBENCH_DOWNLOAD}" == "1" ]]; then
    gsr_args+=(--download)
  fi
  if [[ -n "${GSRBENCH_GQA_ROOT}" ]]; then
    gsr_args+=(--gqa-root "${GSRBENCH_GQA_ROOT}")
  fi
  "${gsr_args[@]}"

  python -m recap.cli validate-images \
    --input data/recap_gsrbench_external_two_object.jsonl \
    --keep-non-left-right \
    --probe-mode "${PROBE_MODE}"

  python -m recap.cli audit-data \
    --input data/recap_gsrbench_external_two_object.jsonl \
    --output data/recap_gsrbench_external_two_object_audit.json

  run_recap \
    data/recap_gsrbench_external_two_object.jsonl \
    "runs/rice_v5/${MODEL_TAG}_gsrbench_external_two_object"

  run_compact_metrics \
    "runs/rice_v5/${MODEL_TAG}_gsrbench_external_two_object/metrics.json" \
    "runs/rice_v5/${MODEL_TAG}_gsrbench_external_two_object/metrics_compact.json"
fi
