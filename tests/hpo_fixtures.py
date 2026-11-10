"""Shared fixtures for the autoresearch HPO tests.

Everything here runs on a CPU without a GPU, an LLM server or Apptainer:

  * ``synthetic_entry`` builds task records on the synthetic objective;
  * ``make_stub_repo`` writes a stand-in for the autoresearch checkout (a train.py
    that prints the upstream summary block without importing torch) and
    ``stub_launcher`` a stand-in for ``uv run python``;
  * ``FakeLLMClient`` stands in for the OpenAI client and ``FakeLLMServer`` for a
    vLLM server (a real HTTP endpoint, so the OpenAI client itself is exercised);
  * ``FakeSandbox`` stands in for ``ApptainerSandbox`` (bind mounts become symlinks
    under a temporary root);
  * ``REFERENCE_SOLVER`` is a small valid SOLVER_HPO.py.
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import random
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rewards.autoresearch_hpo import sample_config  # noqa: E402


def load_script(relative_path, module_name):
    """Import a script of the repository that is not part of a package."""
    path = os.path.join(PROJECT_ROOT, relative_path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


generator = load_script("data_generation/autoresearch_hpo_generator.py",
                        "autoresearch_hpo_generator")


def synthetic_entry(hours=0.5, landscape=0, seed=0, **hardware):
    """A synthetic task record (30 simulated minutes by default: at least 6 trials)."""
    return generator.build_task("synthetic", seed, hours, landscape_seed=landscape, **hardware)


def quiet(function, *args, **kwargs):
    """Call ``function`` with stdout and stderr captured; returns (result, out, err)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        result = function(*args, **kwargs)
    return result, out.getvalue(), err.getvalue()


def exit_status(main, argv):
    """(exit status, stderr) of a command-line entry point, argparse errors included."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            status = main(argv)
        except SystemExit as exc:
            status = exc.code
    return status, err.getvalue()


def write_tasks(path, entries):
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    return path


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ──────────────────────────────────────────────────────────────────────────────
# Stand-in for the autoresearch checkout and its launcher
# ──────────────────────────────────────────────────────────────────────────────

STUB_PREPARE = '''\
"""Stand-in for the upstream prepare.py (tests only)."""
MAX_SEQ_LEN = 2048
TIME_BUDGET = 300  # training seconds per trial
'''

# The hyperparameter block mimics the upstream layout: comments on the lines, an
# expression instead of a literal, assignments spread over the file.
STUB_TRAIN = '''\
"""Stand-in for the upstream train.py (tests only): no torch, no GPU.

STUB_MODE in the environment selects the outcome.
"""
import os
import subprocess
import sys
import time

from prepare import MAX_SEQ_LEN, TIME_BUDGET

# Model architecture
ASPECT_RATIO = 64       # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 128          # target head dimension for attention
WINDOW_PATTERN = "SSSL" # sliding window pattern

# Optimization
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial

# Model size
DEPTH = 8               # number of transformer layers
DEVICE_BATCH_SIZE = 128 # per-device batch size

assert TOTAL_BATCH_SIZE % (DEVICE_BATCH_SIZE * MAX_SEQ_LEN) == 0

mode = os.environ.get("STUB_MODE", "ok")
if mode == "oom":
    sys.stderr.write("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB\\n")
    sys.exit(1)
if mode == "diverged":
    print("\\rstep 00012 (4.0%) | loss: nan    FAIL")
    sys.exit(1)
if mode == "assert":
    assert DEPTH < 0, "stub assertion"
if mode == "crash":
    sys.stderr.write("RuntimeError: something else\\n")
    sys.exit(3)
if mode == "silent":
    print("no summary block here")
    sys.exit(0)
if mode == "hang":
    # A grandchild that outlives a plain kill of this process, like the python
    # process behind `uv run`.
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    with open(os.environ["STUB_PID_FILE"], "w") as f:
        f.write(str(child.pid))
    time.sleep(120)

print(f"Time budget: {TIME_BUDGET}s")
print("\\rstep 00100 (100.0%) | loss: 3.1    ", end="", flush=True)
print()
print("---")
print(f"val_bpb:          {0.9 + DEPTH / 1000 + MATRIX_LR / 10:.6f}")
print(f"training_seconds: {300.1:.1f}")
print(f"total_seconds:    {341.7:.1f}")
print(f"peak_vram_mb:     {DEVICE_BATCH_SIZE * 350.5:.1f}")
print(f"mfu_percent:      {float(os.environ.get('CUDA_MEM_FRACTION', '-1')):.3f}")
print(f"total_tokens_M:   {TOTAL_BATCH_SIZE / 1e6:.1f}")
print(f"num_steps:        {100}")
print(f"num_params_M:     {26.3:.1f}")
print(f"depth:            {DEPTH}")
print(f"window_long:      {WINDOW_PATTERN.count('L')}")
'''

# Stand-in for `uv run python`: answers the device probe (`-c ...`) from STUB_DEVICE
# and runs anything else with the plain interpreter.
STUB_LAUNCHER = '''\
import json
import os
import sys

args = sys.argv[1:]
if args[:1] == ["-c"]:
    device = os.environ.get("STUB_DEVICE", "")
    if not device:
        sys.stderr.write("RuntimeError: Found no NVIDIA driver on your system.\\n")
        sys.exit(1)
    print("warning: some launcher noise")
    print("HPO_DEVICE " + device)
    sys.exit(0)
os.execv(sys.executable, [sys.executable] + args)
'''

H200 = {"gpu": "NVIDIA H200", "total_vram_gb": 139.8}
SMALL_GPU = {"gpu": "NVIDIA RTX 4090", "total_vram_gb": 23.99}
A100_80GB = {"gpu": "NVIDIA A100-SXM4-80GB", "total_vram_gb": 79.25}


def sha256_of(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def make_stub_repo(directory, train_source=STUB_TRAIN, prepare_source=STUB_PREPARE):
    """Write the stand-in checkout; returns the ``UPSTREAM`` entries for its files.

    The harness accepts one pin only (``rewards.autoresearch_hpo.UPSTREAM``); a test
    that wants the stand-in to count as the pinned sources patches it:
    ``mock.patch.dict(UPSTREAM, make_stub_repo(directory))``.
    """
    os.makedirs(directory, exist_ok=True)
    for name, source in (("train.py", train_source), ("prepare.py", prepare_source)):
        with open(os.path.join(directory, name), "w", encoding="utf-8") as f:
            f.write(source)
    return {"train_py_sha256": sha256_of(os.path.join(directory, "train.py")),
            "prepare_py_sha256": sha256_of(os.path.join(directory, "prepare.py"))}


def stub_launcher(directory):
    """Write the stand-in launcher; returns the ``run_command`` list."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "stub_launcher.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(STUB_LAUNCHER)
    return [sys.executable, path]


def device_env(device, **extra):
    """``extra_env`` for a NanochatObjective on the stand-in launcher."""
    env = {"STUB_DEVICE": json.dumps(device) if device else ""}
    env.update(extra)
    return env


def nanochat_entry(hours=0.5, seed=0, **hardware):
    """A nanochat task record (it names whatever ``UPSTREAM`` pins when it is built)."""
    return generator.build_task("nanochat", seed, hours, **hardware)


# ──────────────────────────────────────────────────────────────────────────────
# Stand-ins for the LLM
# ──────────────────────────────────────────────────────────────────────────────

def cot_answer(config, thought="The default leaves memory headroom; a small step first."):
    return f"{thought}\n\n```json\n{json.dumps(config)}\n```"


def sampled_answer(prompt, seed):
    """A well-formed answer whose configuration is a random draw from the search space."""
    rng = random.Random(f"fake_llm/{seed}/{len(prompt)}")
    return cot_answer(sample_config(rng))


class FakeLLMClient:
    """Stands in for ``openai.OpenAI``: ``client.chat.completions.create(...)``.

    ``answers`` is a list consumed one per request; an item is the answer text, a
    ``(content, reasoning)`` pair or ``(content, reasoning, finish_reason)`` triple
    (default ``"stop"``), an exception to raise, or a callable
    ``(prompt, seed) -> one of those``. When the list is used up the last item repeats.
    """

    def __init__(self, answers, reasoning_key="reasoning_content", usage=(100, 20, 5)):
        self.answers = list(answers)
        self.reasoning_key = reasoning_key
        self.usage = usage
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request):
        self.requests.append(request)
        answer = self.answers[min(len(self.requests) - 1, len(self.answers) - 1)]
        if callable(answer):
            answer = answer(request["messages"][0]["content"], request["seed"])
        if isinstance(answer, BaseException):
            raise answer
        content, reasoning, finish_reason = (
            (tuple(answer) + ("stop",))[:3] if isinstance(answer, tuple) else (answer, "", "stop"))
        message = SimpleNamespace(
            content=content, model_extra={self.reasoning_key: reasoning} if reasoning else {})
        prompt_tokens, completion_tokens, reasoning_tokens = self.usage
        usage = SimpleNamespace(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason=finish_reason)], usage=usage)


class FakeLLMServer:
    """A local OpenAI-compatible ``/v1/chat/completions`` endpoint (context manager).

    ``answer(prompt, seed) -> (content, reasoning)`` produces the reply; the reasoning
    is returned in the ``reasoning_content`` field, as vLLM does.
    """

    def __init__(self, answer):
        self.requests = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                server.requests.append({"path": self.path, "body": body})
                content, reasoning = answer(body["messages"][0]["content"], body.get("seed"))
                payload = json.dumps({
                    "id": "chatcmpl-test", "object": "chat.completion", "created": 0,
                    "model": body["model"],
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": content,
                                             "reasoning_content": reasoning}}],
                    "usage": {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80,
                              "completion_tokens_details": {"reasoning_tokens": 10}},
                }).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self._httpd.server_address[1]}/v1"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join()


# ──────────────────────────────────────────────────────────────────────────────
# Stand-ins for the solver and the sandbox
# ──────────────────────────────────────────────────────────────────────────────

REFERENCE_SOLVER = '''\
"""A small valid SOLVER_HPO.py for the tests: seeded local search around the incumbent."""
import math
import random

from rewards.autoresearch_hpo import SEARCH_SPACE, normalize_config


def solve_hpo(task, history):
    print("solvers may print; the harness must not mistake this for the answer")
    rng = random.Random(f"{task['seed']}/{len(history)}")
    finished = [t for t in history if t["status"] == "ok"]
    best = min(finished, key=lambda t: t["val_bpb"])["config"] if finished else task["starting_config"]
    config = dict(best)
    for hp in rng.sample(list(SEARCH_SPACE), 2):
        name = hp["name"]
        if hp["type"] == "cat":
            config[name] = rng.choice(hp["choices"])
        elif name in ("DEPTH", "ASPECT_RATIO", "DEVICE_BATCH_SIZE", "HEAD_DIM"):
            continue                    # keep the memory footprint of the incumbent
        elif hp["log"]:
            config[name] = config[name] * math.exp(rng.uniform(-0.3, 0.3))
        else:
            config[name] = config[name] + rng.uniform(-0.1, 0.1) * (hp["high"] - hp["low"])
    return {"config": normalize_config(config), "strategy": "local perturbation"}
'''


def write_solver(directory, source=REFERENCE_SOLVER, name="SOLVER_HPO.py"):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(source)
    return path


class FakeSandbox:
    """Stands in for ``ApptainerSandbox`` without a container.

    Every call builds a temporary root in which each bind mount is a symlink at its
    container path, rewrites the container paths in the code to that root and runs
    the code with the local interpreter. No isolation: tests only.
    """

    def __init__(self, timeout=30):
        self.timeout = timeout
        self.calls = []

    def run_code(self, code, lib_dir=None, timeout=None, extra_binds=None):
        self.calls.append({"code": code, "timeout": timeout, "extra_binds": extra_binds})
        with tempfile.TemporaryDirectory(prefix="fake_sandbox_") as root:
            for host_path, container_path, _mode in extra_binds or []:
                target = os.path.join(root, container_path.lstrip("/"))
                os.makedirs(os.path.dirname(target), exist_ok=True)
                os.symlink(host_path, target)
            script = os.path.join(root, "code.py")
            with open(script, "w", encoding="utf-8") as f:
                f.write(code.replace("/workspace", os.path.join(root, "workspace")))
            try:
                # -B: the bound directories are read-only in the real sandbox.
                proc = subprocess.run(
                    [sys.executable, "-I", "-B", script], capture_output=True, text=True,
                    cwd=root, timeout=timeout or self.timeout)
            except subprocess.TimeoutExpired:
                return False, "", f"Timeout after {timeout or self.timeout}s"
        return proc.returncode == 0, proc.stdout, proc.stderr
