#!/usr/bin/env bash
# =============================================================================
# test_vllm.sh — Run the Symbolic Library Agent against a local vLLM server
#
# Usage:
#   bash scripts/test_vllm.sh
#   BASE_URL=http://myhost:9000/v1 bash scripts/test_vllm.sh
#
# All variables below can be overridden from the environment, e.g.:
#   MODEL=llama-3-70b OUTPUT_DIR=out/ bash scripts/test_vllm.sh
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration — override any of these from the environment
# =============================================================================

# vLLM server endpoint (OpenAI-compatible)
BASE_URL="http://localhost:8002/v1"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"

# Model name as registered in vLLM (must match --served-model-name)
MODEL="${MODEL:-openai/gpt-oss-120b}"

# API key for the vLLM server ("EMPTY" works for most local deployments)
VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"

# Per-task step budget (each SSL or BCR call costs 1.0–1.5 budget units)
BUDGET="${BUDGET:-15.0}"

# Directory that receives per-task trajectory.json + response.json
OUTPUT_DIR="${OUTPUT_DIR:-outputs/vllm_test}"

# Path to a custom JSONL tasks file (leave empty to use built-in examples)
TASKS_FILE="${TASKS_FILE:-}"

# Single built-in task index to run in the quick-smoke test (0-based)
SINGLE_TASK_INDEX="${SINGLE_TASK_INDEX:-0}"

# Set to "1" to also print the library stats summary after each run
SHOW_STATS="${SHOW_STATS:-1}"

# =============================================================================
# Helpers
# =============================================================================

BOLD="\033[1m"
GREEN="\033[32m"
YELLOW="\033[33m"
RED="\033[31m"
RESET="\033[0m"

log()  { echo -e "${BOLD}[test_vllm]${RESET} $*"; }
ok()   { echo -e "${GREEN}[OK]${RESET} $*"; }
warn() { echo -e "${YELLOW}[WARN]${RESET} $*"; }
fail() { echo -e "${RED}[FAIL]${RESET} $*"; exit 1; }

# Move to repo root regardless of where the script is invoked from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# =============================================================================
# Pre-flight checks
# =============================================================================

log "=== Pre-flight checks ==="

# 1. Python available
python --version >/dev/null 2>&1 || fail "python not found on PATH"
ok "python found: $(python --version)"

# 2. Dependencies installed
python -c "import anthropic, openai, dotenv" 2>/dev/null \
  || fail "Missing Python dependencies. Run:  pip install -r requirements.txt"
ok "Python dependencies present"

# 3. vLLM server reachable
log "Checking vLLM server at ${BASE_URL} ..."
HEALTH_URL="${BASE_URL%/v1}/health"
if curl -sf --max-time 5 "$HEALTH_URL" >/dev/null 2>&1; then
  ok "vLLM server is up ($HEALTH_URL)"
else
  # Fall back to a models probe (some servers don't expose /health)
  MODELS_URL="${BASE_URL}/models"
  if curl -sf --max-time 5 \
       -H "Authorization: Bearer ${VLLM_API_KEY}" \
       "$MODELS_URL" >/dev/null 2>&1; then
    ok "vLLM server is up ($MODELS_URL)"
  else
    fail "Cannot reach vLLM server at ${BASE_URL}. Is it running?\n" \
         "  Start with:  vllm serve ${MODEL} --port 8000"
  fi
fi

# 4. Confirm the requested model is listed
log "Verifying model '${MODEL}' is served ..."
MODELS_RESP=$(curl -sf --max-time 5 \
  -H "Authorization: Bearer ${VLLM_API_KEY}" \
  "${BASE_URL}/models" 2>/dev/null || echo "")
if echo "$MODELS_RESP" | grep -q "$MODEL"; then
  ok "Model '${MODEL}' found"
else
  warn "Could not confirm model '${MODEL}' in server response — proceeding anyway"
  warn "Server response: ${MODELS_RESP:0:200}"
fi

echo ""

# =============================================================================
# Build shared CLI flags
# =============================================================================

COMMON_FLAGS=(
  --base-url    "$BASE_URL"
  --model       "$MODEL"
  --budget      "$BUDGET"
  --output-dir  "$OUTPUT_DIR"
)
if [[ "$SHOW_STATS" == "1" ]]; then
  COMMON_FLAGS+=(--stats)
fi

export VLLM_API_KEY

# =============================================================================
# Run 1 — Single built-in task (quick smoke test)
# =============================================================================

log "=== Run 1: single built-in task (index ${SINGLE_TASK_INDEX}) ==="
python main.py \
  "${COMMON_FLAGS[@]}" \
  --task "$SINGLE_TASK_INDEX"
echo ""

# =============================================================================
# Run 2 — All built-in tasks in batch (shared library across tasks)
# =============================================================================

log "=== Run 2: all built-in tasks (batch, shared library) ==="
python main.py \
  "${COMMON_FLAGS[@]}"
echo ""

# =============================================================================
# Run 3 — Tasks from file (JSONL or JSON)
# =============================================================================

# If no custom file was provided, generate a small sample JSONL for the test
if [[ -z "$TASKS_FILE" ]]; then
  TASKS_FILE="$(mktemp /tmp/test_tasks_XXXXXX.jsonl)"
  GENERATED_FILE=1
  cat > "$TASKS_FILE" <<'JSONL'
{"prompt": "Given a list of integers, return a new list with each element doubled.", "type": "list_transform"}
{"prompt": "Given a list of integers, return only the even numbers.", "type": "list_transform"}
{"prompt": "Given a string, return it with every word capitalised.", "type": "string_transform"}
{"prompt": "Given a list of integers, return the running cumulative sum.", "type": "sequence"}
{"prompt": "Given a list of integers, return the sum of the squares of the odd numbers.", "type": "compositional"}
JSONL
  log "Generated sample tasks file: $TASKS_FILE"
else
  GENERATED_FILE=0
  log "Using provided tasks file: $TASKS_FILE"
fi

log "=== Run 3: tasks from file (${TASKS_FILE}) ==="
python main.py \
  "${COMMON_FLAGS[@]}" \
  --tasks-file "$TASKS_FILE"
echo ""

[[ "$GENERATED_FILE" == "1" ]] && rm -f "$TASKS_FILE"

# =============================================================================
# Summary
# =============================================================================

log "=== Output directory layout ==="
if command -v tree >/dev/null 2>&1; then
  tree -L 2 "$OUTPUT_DIR"
else
  find "$OUTPUT_DIR" -type f | sort
fi

echo ""
ok "All runs completed.  Results are in: ${OUTPUT_DIR}/"
