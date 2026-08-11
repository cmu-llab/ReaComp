#!/usr/bin/env bash
# Orchestrator (session helper): wait for the rsync env-copy to finish, then — if
# the NFS vLLM isn't already serving on :8000 — kill it and relaunch from the
# fast /scratch copy. Idempotent-ish; logs everything to a status file.
set -uo pipefail

RSYNC_PID=441387
NFS_VLLM_PID=458579
PORT=8000
STATUS=/tmp/vllm_switch_status.log
SCRATCH_PY=/scratch/arnaik/envs/vllm_devstral/bin/python
cd /home/arnaik/symbolic-library-agent

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$STATUS"; }

serving() { timeout 4 curl -s "http://localhost:${PORT}/v1/models" 2>/dev/null | grep -q "Qwen"; }

say "orchestrator start: waiting for rsync pid=$RSYNC_PID"
while kill -0 "$RSYNC_PID" 2>/dev/null; do
  if serving; then say "NFS vLLM already SERVING before rsync done — no switch needed. exit."; exit 0; fi
  sleep 30
done
say "rsync finished. scratch env size: $(du -sh /scratch/arnaik/envs/vllm_devstral 2>/dev/null | cut -f1)"

# Give the just-finished copy a moment; verify the scratch interpreter runs.
if [ ! -x "$SCRATCH_PY" ]; then say "ERROR: $SCRATCH_PY missing/not executable. abort switch."; exit 1; fi

if serving; then say "NFS vLLM is SERVING — leaving it, no switch."; exit 0; fi

say "NFS vLLM not serving. Killing NFS pid=$NFS_VLLM_PID and relaunching from scratch."
kill -9 "$NFS_VLLM_PID" 2>/dev/null
# also clear any stray api_server we own
pkill -9 -f "vllm.entrypoints.openai.api_server --model Qwen/Qwen3.6-35B-A3B" 2>/dev/null
sleep 5

ts=$(date +%Y%m%d_%H%M%S)
LOG=vllm_logs/vllm_8000_scratch_${ts}.log
say "launching scratch vLLM -> $LOG"
PYTHONUNBUFFERED=1 nohup "$SCRATCH_PY" -u -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.6-35B-A3B --tokenizer Qwen/Qwen3.6-35B-A3B \
  --dtype auto --port "$PORT" --max-model-len 262144 \
  --gpu-memory-utilization 0.95 --tensor-parallel-size 2 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  > "$LOG" 2>&1 &
NEWPID=$!
echo "$NEWPID" > "vllm_logs/vllm_8000_scratch_${ts}.pid"
echo "$LOG" > /tmp/current_vllm_log.txt
say "scratch vLLM launched pid=$NEWPID"

# Watch for it to serve (scratch import should be much faster than NFS)
for i in $(seq 1 60); do
  if [ ! -d /proc/$NEWPID ]; then say "scratch vLLM DIED. tail:"; tail -20 "$LOG" | tee -a "$STATUS"; exit 1; fi
  if serving; then say "scratch vLLM SERVING on :$PORT (after ~$((i*30))s)"; exit 0; fi
  sleep 30
done
say "scratch vLLM not serving after 30min. log size=$(stat -c%s "$LOG")"
