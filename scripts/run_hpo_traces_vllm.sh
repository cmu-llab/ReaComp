#!/usr/bin/env bash
# Autoresearch HPO — LLM HPO agent (baseline + reasoning traces for solver induction).
#
# One episode per task record: trial 0 trains the default nanochat configuration, then
# the model served by vLLM proposes one configuration per trial until the task's
# training-time budget is spent. Every trial trains for 300 s on the local GPU, so the
# pilot task (1 h) takes a bit over an hour and a full task 24 h plus the LLM time.
# Re-running the same command resumes unfinished episodes.
#
# Prerequisites:
#   bash scripts/setup_autoresearch_hpo.sh                  # training environment
#   bash scripts/launch_vllm_qwen3.6_35b_a3b.sh <PORT> <N>  # model server
# The model server and the training runs need separate GPUs: TRAIN_GPU selects the
# one training uses.
#
# Usage (from project root):
#   bash scripts/run_hpo_traces_vllm.sh
#   PORT=8004 bash scripts/run_hpo_traces_vllm.sh           # vLLM port (default 8000)
#   TRAIN_GPU=1 TASKS=data/autoresearch_hpo/tasks_full.jsonl bash scripts/run_hpo_traces_vllm.sh
#   TRAIN_GPU=2 TASKS=data/autoresearch_hpo/tasks_full.jsonl \
#       bash scripts/run_hpo_traces_vllm.sh --task-id autoresearch_hpo_24h_seed1   # one seed per GPU
#
# Upstream settings of the paper's "Karpathy Agent (14 HPs)" baseline:
#   PROMPT_STYLE=paper MAX_TOKENS=2048 bash scripts/run_hpo_traces_vllm.sh --parse-retries 0
#
# Dry run without a GPU (synthetic objective, NOT benchmark numbers):
#   TASKS=data/autoresearch_hpo/synthetic_tasks.jsonl bash scripts/run_hpo_traces_vllm.sh
#
# Next step: python -m openhands_agents.build_hpo_balanced_demos --episodes <RUN_DIR>
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
TASKS=${TASKS:-data/autoresearch_hpo/pilot_tasks.jsonl}

PORT=${PORT:-8000}
GPU_NODE=localhost
BASE_URL=http://${GPU_NODE}:${PORT}/v1
MODEL=${MODEL:-openai/Qwen/Qwen3.6-35B-A3B}
MODEL_TAG=${MODEL_TAG:-qwen3.6_35b_a3b}

PROMPT_STYLE=${PROMPT_STYLE:-cot}   # cot (reasoning traces) or paper (upstream prompt)
MAX_TOKENS=${MAX_TOKENS:-8192}
TEMPERATURE=${TEMPERATURE:-0.7}

# GPU index for the training runs (empty = whatever CUDA_VISIBLE_DEVICES already says)
TRAIN_GPU=${TRAIN_GPU:-}

STYLE_TAG=$([ "${PROMPT_STYLE}" == "cot" ] && echo "" || echo "_${PROMPT_STYLE}")
RUN_DIR=${RUN_DIR:-outputs/autoresearch_hpo_$(basename "${TASKS}" .jsonl)_hpo_agent_${MODEL_TAG}${STYLE_TAG}}
# ─────────────────────────────────────────────────────────────────────────────

[ -n "${TRAIN_GPU}" ] && export CUDA_VISIBLE_DEVICES="${TRAIN_GPU}"

echo "Autoresearch HPO agent run"
echo "  tasks     : $TASKS"
echo "  model     : $MODEL ($BASE_URL)"
echo "  prompt    : $PROMPT_STYLE"
echo "  train GPU : ${CUDA_VISIBLE_DEVICES:-<all visible>}"
echo "  run_dir   : $RUN_DIR"
echo ""

python -m openhands_agents.run_hpo_traces \
    --tasks-file "$TASKS" \
    --run-dir "$RUN_DIR" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --prompt-style "$PROMPT_STYLE" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    "$@"
