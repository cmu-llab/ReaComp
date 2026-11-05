"""
Domain registry.

    from domains import get_domain, list_domains
    dom = get_domain("pbebench")
    tasks = dom.load_tasks("lite")

Domains are imported lazily so that optional dependencies of one domain
(Docker, SWI-Prolog, reasoning_gym, ...) never block using another.
Register a new domain by adding it to ``_REGISTRY`` (module path, class name)
or by calling ``register`` at import time from your own package.
"""
from __future__ import annotations

import importlib
from typing import Dict, List, Type

from .base import Domain, SolveOutcome, SplitSpec, REPO_ROOT  # noqa: F401

_REGISTRY: Dict[str, str] = {
    "pbebench": "domains.pbebench:PBEBenchDomain",
    "slr_bench": "domains.slr_bench:SLRBenchDomain",
    "list_functions": "domains.list_functions:ListFunctionsDomain",
    "envsetup": "domains.envsetup.domain:EnvSetupDomain",
}
_INSTANCES: Dict[str, Domain] = {}


def register(name: str, target: str) -> None:
    """Register ``name`` -> "module.path:ClassName"."""
    _REGISTRY[name] = target
    _INSTANCES.pop(name, None)


def list_domains() -> List[str]:
    return sorted(_REGISTRY)


def get_domain(name: str) -> Domain:
    if name not in _REGISTRY:
        raise KeyError(f"unknown domain {name!r}; registered: {list_domains()}")
    if name not in _INSTANCES:
        module_path, cls_name = _REGISTRY[name].split(":")
        cls: Type[Domain] = getattr(importlib.import_module(module_path), cls_name)
        _INSTANCES[name] = cls()
    return _INSTANCES[name]
