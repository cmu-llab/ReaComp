# Running TroVE on PBEBench-Lite

This guide covers launching the TroVE baseline against `openai/gpt-oss-20b`
served by vLLM. There are two paths:

- **Notebook (recommended on RunPod)** — `notebooks/run_trove_pbebench.ipynb`
  drives the whole flow (env setup → vLLM launch → TroVE run → analysis) from
  one place and mirrors logs to disk.
- **Shell scripts** — for SSH / tmux workflows where a notebook is awkward.

Both paths assume an L40S/H100-class GPU with ≥40 GB VRAM and ≥40 GB free disk
for the model cache.

---

## 0. Prerequisites

- `vLLM >= 0.16.0` — earlier versions do not ship the gpt-oss reasoning parser
  or auto tool-choice support.
- `typing_extensions >= 4.12.2` — older versions break vLLM startup with
  `cannot import name 'TypeIs' from typing_extensions`.
- `huggingface_hub` with a working transfer backend. If `xet` errors during
  download, set `HF_HUB_DISABLE_XET=1`.
- `HF_HOME` pointed at a persistent volume (e.g. `/workspace/hf-cache`) so the
  model is not re-downloaded across container restarts.

Quick install / repair on a fresh container:

```bash
python -m pip install -U "typing_extensions>=4.12.2" \
                        "huggingface_hub[hf_transfer]" hf_xet
```

---

## 1. Notebook path (RunPod)

```bash
git clone <repo-url> /workspace/symbolic-library-agent
cd /workspace/symbolic-library-agent
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Then open `notebooks/run_trove_pbebench.ipynb` and run the cells top-to-bottom:

1. **Env / cache setup** — sets `HF_HOME=/workspace/hf-cache` and disables xet.
2. **`pip install` cell** — refreshes `typing_extensions` and HF transfer.
3. **Launch vLLM** — backgrounds `scripts/launch_vllm_gpt_oss_20b.sh 8000` and
   tails `vllm_logs/`.
4. **Wait for server ready** — polls `/v1/models` until 200 OK.
5. **`tail_vllm_log(60)` helper** — re-runnable cell for spot-checking the
   server log at any time.
6. **Run TroVE** — `subprocess.Popen` of `main.py` with the PBEBench-Lite
   pilot tasks. Stdout is mirrored to `outputs/trove_pbebench_lite_smoke_<ts>.log`
   on disk in addition to the cell output, so you can SSH in and `tail -f` the
   run from another shell.
7. **Analyze** — calls `scripts/analyze_trove_run.py` on the output JSONL.

If the notebook cell stops responding, do **not** `pkill -f "main.py"` —
that pattern can match the vLLM process tree on some images. Instead:

```bash
ps -ef | awk '/python .*main.py/ && /--framework/ && /trove/ {print $2}' \
       | xargs -r kill
```

---

## 2. Shell-script path

Two scripts; run them in two terminals (or one tmux session with two panes).

### 2a. Launch vLLM

```bash
cd /workspace/symbolic-library-agent
mkdir -p vllm_logs
bash scripts/launch_vllm_gpt_oss_20b.sh 8000
# logs: vllm_logs/vllm_8000_<timestamp>.log
# pid : vllm_logs/vllm_8000_<timestamp>.pid
```

The script forwards three flags that are required for our IMPORT-with-tools
branch to work:

- `--enable-auto-tool-choice`
- `--tool-call-parser openai`
- `--reasoning-parser openai_gptoss`

Wait for `Application startup complete` in the log before continuing.

### 2b. Run TroVE

```bash
PORT=8000 bash scripts/run_trove_vllm.sh
```

Defaults (overridable via env vars or trailing flags):

| Env var      | Default                                   |
| ------------ | ----------------------------------------- |
| `PORT`       | `8000`                                    |
| `TASKS_FILE` | `data/pbebench/lite_pilot_tasks.jsonl`    |
| `OUT_FILE`   | `outputs/trove_pbebench_lite_pilot.jsonl` |

Pass through any extra `main.py` flag, e.g.:

```bash
PORT=8000 bash scripts/run_trove_vllm.sh --num-tasks 10  # quick sanity run
```

### 2c. Analyze

```bash
python scripts/analyze_trove_run.py outputs/trove_pbebench_lite_pilot.jsonl
```

Reports overall accuracy, final toolbox size, per-mode wins, IMPORT-mode
tool-call success rate, and the top-10 most-called toolbox functions.

---

## 3. Key flags (cheat sheet)

The TroVE-specific flags on `main.py` matter most:

| Flag                  | Default      | Purpose                                                 |
| --------------------- | ------------ | ------------------------------------------------------- |
| `--framework`         | —            | Set to `trove`                                          |
| `--trove-task-family` | `default`    | Set to `pbebench` to enable PBEBench few-shots & parser |
| `--trove-selection`   | `reward`     | `reward` (PBEBench) or `consistency` (original TroVE)   |
| `--trove-k`           | `5`          | Candidates per mode (1 disables sampling)               |
| `--trove-trim-every`  | `100`        | Set high (`9999`) for ≤100-task pilots                  |
| `--default-reward`    | —            | Set to `pbebench` for the PBEBench verifier             |
| `--max-programs`      | `5`          | PBEBench program-list length cap                        |

---

## 4. Resuming and cleanup

- Resume: just re-run the same command. `main.py` checkpoints to the output
  JSONL; if both the JSONL and `--debug-dir` are intact it will skip already-
  completed task indices.
- Force-restart: delete the output JSONL before running.
- vLLM cleanup:
  ```bash
  kill "$(cat vllm_logs/vllm_8000_*.pid)" 2>/dev/null || true
  pkill -f vllm.entrypoints.openai.api_server  # safe — only matches vLLM
  ```
