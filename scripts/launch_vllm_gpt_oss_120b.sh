#!/bin/bash

mkdir -p /tmp/$USER-tiktoken-cache /tmp/$USER-tmp
chmod 700 /tmp/$USER-tiktoken-cache /tmp/$USER-tmp
export TIKTOKEN_CACHE_DIR=/tmp/$USER-tiktoken-cache
export TMPDIR=/tmp/$USER-tmp

ts=$(date +%Y%m%d_%H%M%S)

nohup python -m vllm.entrypoints.openai.api_server \
  --model "openai/gpt-oss-120b" \
  --tokenizer "openai/gpt-oss-120b" \
  --dtype auto \
  --port ${1} \
  --gpu-memory-utilization 0.95 \
  --tensor-parallel-size 2 \
  > vllm_logs/vllm_${1}_${ts}.log 2>&1 & echo $! > vllm_logs/vllm_${1}_${ts}.pid