#!/bin/bash

ts=$(date +%Y%m%d_%H%M%S)

nohup python -m vllm.entrypoints.openai.api_server \
  --model "openai/gpt-oss-120b" \
  --tokenizer "openai/gpt-oss-120b" \
  --dtype auto \
  --port ${1} \
  --gpu-memory-utilization 0.95 \
  --tensor-parallel-size 2 \
  > vllm_logs/vllm_${1}_${ts}.log 2>&1 & echo $! > vllm_logs/vllm_${1}_${ts}.pid