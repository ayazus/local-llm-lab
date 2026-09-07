#!/usr/bin/env bash
# Start the vLLM OpenAI-compatible server.
#
# Requires: Linux + NVIDIA GPU, `pip install vllm`.
# Foreground; Ctrl+C to stop. For background: `bash start_vllm.sh &`
#
# Env (all optional — see .env.example):
#   MODEL_PATH  VLLM_PORT  VLLM_SERVED_MODEL_NAME
#   VLLM_MAX_MODEL_LEN  VLLM_GPU_MEMORY_UTILIZATION
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-./models/qwen-14b-awq}"
PORT="${VLLM_PORT:-8000}"
SERVED_NAME="${VLLM_SERVED_MODEL_NAME:-qwen-14b-awq}"
MAX_LEN="${VLLM_MAX_MODEL_LEN:-24000}"
GPU_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"

echo "vLLM  model=${MODEL_PATH}  port=${PORT}  served-name=${SERVED_NAME}"
echo "      max-model-len=${MAX_LEN}  gpu-memory-utilization=${GPU_UTIL}"
echo

exec python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name "${SERVED_NAME}" \
    --quantization awq_marlin \
    --dtype float16 \
    --max-model-len "${MAX_LEN}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --host 0.0.0.0 \
    --port "${PORT}"
