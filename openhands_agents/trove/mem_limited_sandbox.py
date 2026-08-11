"""
Memory-limited sandbox for the PBEBench/SLR TroVE baseline.

ADDITIVE module (new file). Subclasses ApptainerSandbox without modifying it.

Why: some induced toolbox functions are brute-force search (e.g. nested
``itertools.product(chars, repeat=len)`` over candidate replace ops). On a hard,
long-cascade task a single candidate can balloon to ~19 GB RSS. With 8 workers
these blow past our SLURM cgroup cap (100 GB) and the kernel OOM-kills the whole
job — including the runner — leaving no traceback.

Fix: cap *virtual address space* per exec via ``resource.setrlimit(RLIMIT_AS)``
inside the container. A runaway candidate then raises ``MemoryError`` (caught as
a normal execution failure -> reward 0) instead of consuming host memory. This
changes only the failure mode of pathological candidates, not the result of any
candidate that would have fit in the cap, so it does not bias the baseline.
"""

from ..sandbox import ApptainerSandbox


def _mem_preamble(limit_bytes: int) -> str:
    # Best-effort: set an address-space rlimit; ignore platforms that disallow it.
    return (
        "import resource as _r\n"
        "try:\n"
        f"    _r.setrlimit(_r.RLIMIT_AS, ({limit_bytes}, {limit_bytes}))\n"
        "except Exception:\n"
        "    pass\n"
    )


class MemoryLimitedApptainerSandbox(ApptainerSandbox):
    """
    ApptainerSandbox that prepends an RLIMIT_AS cap to every executed program.

    Parameters
    ----------
    mem_limit_gb : float
        Per-exec virtual-memory cap in GiB (default 4). A candidate that exceeds
        it raises MemoryError -> counted as a failed execution.
    """

    def __init__(self, *args, mem_limit_gb: float = 4.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._mem_limit_bytes = int(mem_limit_gb * (1024 ** 3))

    def run_code(self, code, *args, **kwargs):
        guarded = _mem_preamble(self._mem_limit_bytes) + code
        return super().run_code(guarded, *args, **kwargs)
