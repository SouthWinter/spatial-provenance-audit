#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"
source "${SCRIPT_DIR}/env_local.sh"

export USE_TF=0
export TRANSFORMERS_NO_TF=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1

GPU_ID="${GPU_ID:-0}"
SPLIT="${SPLIT:-confirmation}"
KEEP_RATIO="${KEEP_RATIO:-0.40}"
LIMIT="${LIMIT:-}"
OUT_ROOT="${OUT_ROOT:-runs/anchorprune_textocr}"
CLIP_MODEL="${CLIP_MODEL:-${ANCHORPRUNE_CLIP_PATH}}"

case "${SPLIT}" in
  development)
    INPUT="${INPUT:-data/textocr_val_hard_probes_500img.jsonl}"
    ;;
  confirmation)
    INPUT="${INPUT:-data/textocr_val_hard_confirmation_500img_seed20260720.jsonl}"
    ;;
  *)
    echo "SPLIT must be development or confirmation" >&2
    exit 2
    ;;
esac

ratio_tag="${KEEP_RATIO/./p}"
work_dir="${OUT_ROOT}/${SPLIT}_llava15_anchorprune_${ratio_tag}"
mkdir -p "${OUT_ROOT}/logs"

args=(
  python -u -m recap.cli run-llava-pruned
  --input "${INPUT}"
  --is-probes
  --work-dir "${work_dir}"
  --pretrained "${LLAVA_PATH}"
  --selector anchorprune
  --keep-ratio "${KEEP_RATIO}"
  --anchorprune-k-min-ratio 0.15625
  --anchorprune-tau 0.20
  --anchorprune-patience 3
  --anchorprune-kmax-ratio 0.50
  --anchorprune-clip-model "${CLIP_MODEL}"
  --seed 17
  --device cuda
  --device-map auto
  --dtype float16
  --trust-remote-code
  --attn-implementation eager
)
if [[ -n "${LIMIT}" ]]; then
  args+=(--limit "${LIMIT}")
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${args[@]}" 2>&1 | tee "${OUT_ROOT}/logs/${SPLIT}_llava15_anchorprune_${ratio_tag}.log"
