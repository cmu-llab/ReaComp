#!/bin/bash

mkdir -p /tmp/$USER-tiktoken-cache /tmp/$USER-tmp
chmod 700 /tmp/$USER-tiktoken-cache /tmp/$USER-tmp
export TIKTOKEN_CACHE_DIR=/tmp/$USER-tiktoken-cache
export TMPDIR=/tmp/$USER-tmp

ts=$(date +%Y%m%d_%H%M%S)

# Required vLLM tool-calling flags (vLLM >= v0.16.0 for PR #28729):
#   --enable-auto-tool-choice  enables tool_choice="auto"
#   --tool-call-parser openai  parses gpt-oss Harmony commentary channel
#   --reasoning-parser openai_gptoss  routes analysis-channel content into
#                                     message.reasoning_content
nohup python -m vllm.entrypoints.openai.api_server \
  --model "openai/gpt-oss-20b" \
  --tokenizer "openai/gpt-oss-20b" \
  --dtype auto \
  --port ${1} \
  --gpu-memory-utilization 0.95 \
  --tensor-parallel-size 1 \
  --enable-auto-tool-choice \
  --tool-call-parser openai \
  --reasoning-parser openai_gptoss \
  > vllm_logs/vllm_${1}_${ts}.log 2>&1 & echo $! > vllm_logs/vllm_${1}_${ts}.pid
