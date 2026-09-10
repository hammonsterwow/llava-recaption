#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash launch_4gpu.sh \
#     /path/to/blip_laion_cc_sbu_558k.json \
#     /path/to/LLaVA-Pretrain/images \
#     /path/to/output_dir
#
# Optional env vars:
#   MODEL=Qwen/Qwen2.5-VL-7B-Instruct
#   MIN_VTOK=256
#   MAX_VTOK=512
#   MAX_NEW=256
#   COMPUTE_DTYPE=fp16
#   MODEL_DTYPE=fp16
#   LIMIT=100

INPUT_JSON="${1:?input JSON required}"
IMAGE_ROOT="${2:?image root required}"
OUTPUT_DIR="${3:?output dir required}"

MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
MIN_VTOK="${MIN_VTOK:-256}"
MAX_VTOK="${MAX_VTOK:-512}"
MAX_NEW="${MAX_NEW:-256}"
COMPUTE_DTYPE="${COMPUTE_DTYPE:-fp16}"
MODEL_DTYPE="${MODEL_DTYPE:-fp16}"
LIMIT="${LIMIT:-}"

mkdir -p "${OUTPUT_DIR}/logs"

PIDS=()

for GPU in 0 1 2 3; do
  CMD=(
    python3.10 -u recaption.py
    --input-json "${INPUT_JSON}"
    --image-root "${IMAGE_ROOT}"
    --output-dir "${OUTPUT_DIR}"
    --model "${MODEL}"
    --shard-id "${GPU}"
    --num-shards 4
    --min-visual-tokens "${MIN_VTOK}"
    --max-visual-tokens "${MAX_VTOK}"
    --max-new-tokens "${MAX_NEW}"
    --compute-dtype "${COMPUTE_DTYPE}"
    --model-dtype "${MODEL_DTYPE}"
  )

  if [[ -n "${LIMIT}" ]]; then
    CMD+=(--limit "${LIMIT}")
  fi

  echo "[launch] GPU ${GPU}: ${CMD[*]}"

  CUDA_VISIBLE_DEVICES="${GPU}" \
    "${CMD[@]}" \
    > "${OUTPUT_DIR}/logs/gpu_${GPU}.log" 2>&1 &

  PIDS+=("$!")
done

echo "Started PIDs: ${PIDS[*]}"
echo "Logs: ${OUTPUT_DIR}/logs/gpu_*.log"

FAIL=0
for PID in "${PIDS[@]}"; do
  if ! wait "${PID}"; then
    FAIL=1
  fi
done

if [[ "${FAIL}" -ne 0 ]]; then
  echo "One or more workers failed. Inspect the logs."
  exit 1
fi

echo "All four workers finished."
