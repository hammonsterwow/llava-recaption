#!/usr/bin/env bash
set -euo pipefail

# Four-GPU launcher for the recaptioning job.
# Run this inside tmux for long jobs so SSH disconnects do not stop the work.

INPUT_JSON="${1:?input JSON required}"
IMAGE_ROOT="${2:?image root required}"
OUTPUT_DIR="${3:?output dir required}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
MIN_VTOK="${MIN_VTOK:-256}"
MAX_VTOK="${MAX_VTOK:-512}"
MAX_NEW="${MAX_NEW:-256}"
COMPUTE_DTYPE="${COMPUTE_DTYPE:-fp16}"
MODEL_DTYPE="${MODEL_DTYPE:-fp16}"
LIMIT="${LIMIT:-}"
PROMPT_FILE="${PROMPT_FILE:-${SCRIPT_DIR}/prompt_kakao.txt}"
GPU_LIST="${GPU_LIST:-0 1 2 3}"

mkdir -p "${OUTPUT_DIR}/logs"

if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "[ERROR] prompt file not found: ${PROMPT_FILE}"
  exit 1
fi

read -r -a GPUS <<< "${GPU_LIST}"
NUM_SHARDS="${#GPUS[@]}"

if [[ "${NUM_SHARDS}" -lt 1 ]]; then
  echo "[ERROR] GPU_LIST is empty"
  exit 1
fi

# Atomic output-directory lock. This prevents two launchers from appending to
# the same part_*.jsonl files at the same time.
LOCKDIR="${OUTPUT_DIR}/.launch_lock"
THIS_HOST="$(hostname)"

acquire_lock() {
  if mkdir "${LOCKDIR}" 2>/dev/null; then
    printf '%s\n' "$$" > "${LOCKDIR}/pid"
    printf '%s\n' "${THIS_HOST}" > "${LOCKDIR}/host"
    return 0
  fi

  local old_pid=""
  local old_host=""
  [[ -f "${LOCKDIR}/pid" ]] && old_pid="$(cat "${LOCKDIR}/pid" 2>/dev/null || true)"
  [[ -f "${LOCKDIR}/host" ]] && old_host="$(cat "${LOCKDIR}/host" 2>/dev/null || true)"

  if [[ "${old_host}" == "${THIS_HOST}" && "${old_pid}" =~ ^[0-9]+$ ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "[ERROR] another launcher is already using this output directory"
    echo "        output: ${OUTPUT_DIR}"
    echo "        pid: ${old_pid}"
    exit 1
  fi

  if [[ -n "${old_host}" && "${old_host}" != "${THIS_HOST}" ]]; then
    echo "[ERROR] lock belongs to another host (${old_host}); refusing to remove it automatically"
    echo "        lock: ${LOCKDIR}"
    exit 1
  fi

  echo "[WARN] stale launch lock found; removing it: ${LOCKDIR}"
  rm -rf "${LOCKDIR}"
  mkdir "${LOCKDIR}"
  printf '%s\n' "$$" > "${LOCKDIR}/pid"
  printf '%s\n' "${THIS_HOST}" > "${LOCKDIR}/host"
}

release_lock() {
  if [[ -d "${LOCKDIR}" ]]; then
    local lock_pid=""
    [[ -f "${LOCKDIR}/pid" ]] && lock_pid="$(cat "${LOCKDIR}/pid" 2>/dev/null || true)"
    if [[ "${lock_pid}" == "$$" ]]; then
      rm -rf "${LOCKDIR}"
    fi
  fi
}

acquire_lock
trap release_lock EXIT

# Record the exact generation configuration once per output directory. LIMIT is
# intentionally excluded because it may change between resume sessions.
# A mismatched model/prompt/sharding/configuration aborts before any worker is
# launched, preventing mixed datasets.
python3.10 - \
  "${OUTPUT_DIR}/run_config.json" \
  "${MODEL}" \
  "${NUM_SHARDS}" \
  "${MIN_VTOK}" \
  "${MAX_VTOK}" \
  "${MAX_NEW}" \
  "${COMPUTE_DTYPE}" \
  "${MODEL_DTYPE}" \
  "${PROMPT_FILE}" \
  "${INPUT_JSON}" \
  "${IMAGE_ROOT}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

(
    config_path,
    model,
    num_shards,
    min_vtok,
    max_vtok,
    max_new,
    compute_dtype,
    model_dtype,
    prompt_file,
    input_json,
    image_root,
) = sys.argv[1:]

prompt_path = Path(prompt_file).resolve()
prompt = prompt_path.read_text(encoding="utf-8").strip()

config = {
    "model": model,
    "num_shards": int(num_shards),
    "min_visual_tokens": int(min_vtok),
    "max_visual_tokens": int(max_vtok),
    "max_new_tokens": int(max_new),
    "compute_dtype": compute_dtype,
    "model_dtype": model_dtype,
    "prompt_file": str(prompt_path),
    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    "prompt": prompt,
    "input_json": str(Path(input_json).resolve()),
    "image_root": str(Path(image_root).resolve()),
}

path = Path(config_path)
if path.exists():
    old = json.loads(path.read_text(encoding="utf-8"))
    check_keys = [
        "model",
        "num_shards",
        "min_visual_tokens",
        "max_visual_tokens",
        "max_new_tokens",
        "compute_dtype",
        "model_dtype",
        "prompt_sha256",
        "input_json",
        "image_root",
    ]
    mismatches = {
        key: {"existing": old.get(key), "requested": config.get(key)}
        for key in check_keys
        if old.get(key) != config.get(key)
    }
    if mismatches:
        raise SystemExit(
            "[ERROR] output directory already contains results generated with "
            "different settings:\n" + json.dumps(mismatches, indent=2, ensure_ascii=False)
        )
else:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    print(f"[config] saved: {path}")
PY

PIDS=()

terminate_workers() {
  echo "[signal] stopping recaption workers..."
  for PID in "${PIDS[@]:-}"; do
    kill -TERM "${PID}" 2>/dev/null || true
  done
  for PID in "${PIDS[@]:-}"; do
    wait "${PID}" 2>/dev/null || true
  done
}

on_signal() {
  terminate_workers
  exit 130
}

trap on_signal INT TERM

for IDX in "${!GPUS[@]}"; do
  GPU="${GPUS[$IDX]}"

  CMD=(
    python3.10 -u "${SCRIPT_DIR}/recaption.py"
    --input-json "${INPUT_JSON}"
    --image-root "${IMAGE_ROOT}"
    --output-dir "${OUTPUT_DIR}"
    --model "${MODEL}"
    --shard-id "${IDX}"
    --num-shards "${NUM_SHARDS}"
    --min-visual-tokens "${MIN_VTOK}"
    --max-visual-tokens "${MAX_VTOK}"
    --max-new-tokens "${MAX_NEW}"
    --compute-dtype "${COMPUTE_DTYPE}"
    --model-dtype "${MODEL_DTYPE}"
    --prompt-file "${PROMPT_FILE}"
  )

  if [[ -n "${LIMIT}" ]]; then
    CMD+=(--limit "${LIMIT}")
  fi

  echo "[launch] physical GPU ${GPU}, shard ${IDX}/${NUM_SHARDS}: ${CMD[*]}"

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

echo "All workers finished."
