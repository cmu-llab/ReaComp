#!/bin/bash

ts=$(date +%Y%m%d_%H%M%S)

nohup python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.6-35B-A3B \
  --tokenizer Qwen/Qwen3.6-35B-A3B \
  --dtype auto \
  --port ${1} \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.95 \
  --tensor-parallel-size ${2} \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  > vllm_logs/vllm_${1}_${ts}.log 2>&1 & echo $! > vllm_logs/vllm_${1}_${ts}.pid