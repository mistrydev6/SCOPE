#!/usr/bin/env bash
set -euo pipefail

export PYTHONNOUSERSITE=1

MODEL_PATH="UPDATE MODEL PATH HERE"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
PORT="${VLLM_PORT:-8000}"

exec vllm serve "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --served-model-name qwen3.5:9b \
  --tensor-parallel-size 1 \
  --max-model-len "$MAX_MODEL_LEN" \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --language-model-only \
  --max-num-seqs 64 
