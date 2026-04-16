# Apptainer sandbox

`openhands_agents/sandbox.py`

## How it works

Each `run_code()` call:

1. Writes code to a fresh `tmpdir` (`/tmp/oh_exec_XXX/code.py`) with a sys.path preamble
2. Launches: `apptainer exec --contain --no-home --writable-tmpfs --bind tmpdir:/exec:rw [--bind lib_dir:/exec/<pkg>:ro] sandbox.sif python /exec/code.py`
3. Returns `(ok: bool, stdout: str, stderr: str)` and cleans up the tmpdir

```python
sandbox = ApptainerSandbox(sif_path="/path/to/sandbox.sif", timeout=30)
ok, stdout, stderr = sandbox.run_code(code, lib_dir="/path/to/library")
```

## sys.path preamble

Every code execution is prepended with:
```python
import sys; sys.path.insert(0, '/exec')
```
This makes the library package importable as `from library import fn` (package is bound at `/exec/library`).

## Library bind mount

If `lib_dir` is provided:
- Host path: `lib_dir` (e.g. `/scratch/user/oh_packages/library`)
- Container path: `/exec/<basename(lib_dir)>` (e.g. `/exec/library`)
- Mount mode: `:ro` (read-only — generated code can import but not modify)

## Building the SIF

```bash
bash openhands_agents/scripts/build_sandbox.sh
```

The `sandbox.def` installs Python 3.11 + numpy, scipy, sympy. Edit `sandbox.def` to add more packages.

## Constraints

- No network access (`--contain --no-home`)
- No persistent state between runs (fresh tmpdir each call)
- Reward computation runs on **host** (outside container): `rewards/*.py` use the host Python environment
- Timeout: 30s default; overridable per-call with `timeout=N`
