"""Classical HPO baselines for the autoresearch HPO domain, as episode policies.

  random   uniform / log-uniform sampling of the search space
  tpe      Optuna TPESampler(seed=seed)
  cmaes    Optuna CmaEsSampler(seed=seed)

TPE and CMA-ES follow the upstream benchmark's Optuna backends
(``backends/optuna_backend.py``, ``backends/cma_es_backend.py`` of
ferreirafabio/autoresearch-automl): the starting configuration is added to the study
as a completed trial, and a failed trial is told as a completed one with the failure
penalty (100.0) so the sampler learns to avoid the region. They need the optional
``optuna`` (and, for cmaes, ``cmaes``) packages.

SMAC, LLAMBO and Centaur from the paper are not implemented here.
"""
from __future__ import annotations

import random
from typing import Any, Callable, Dict, List

from rewards.autoresearch_hpo import (
    FAILURE_PENALTY, SEARCH_SPACE, STATUS_OK, EpisodeLogMismatch, normalize_config,
    sample_config,
)

BASELINES = ("random", "tpe", "cmaes")


def random_search(task: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Random search. Stateless: the draw depends on the task seed and the trial index."""
    rng = random.Random(f"random_search/{task['seed']}/{len(history)}")
    return {"config": sample_config(rng), "source": "random"}


class OptunaPolicy:
    """An Optuna sampler driven through ask / tell, one study per episode.

    A resumed episode replays its log: the seeded sampler is asked again for every
    logged trial and told the logged result, which puts it in the state it had when
    the run stopped, so the resumed run continues exactly as an uninterrupted one
    would have. A log the sampler does not reproduce (another Optuna version, a log
    written by something else) raises ``EpisodeLogMismatch`` instead of continuing
    with a sampler in a different state.
    """

    def __init__(self, sampler: str, seed: int):
        try:
            import optuna
        except ImportError as exc:
            raise ImportError(
                f"the {sampler!r} baseline needs Optuna: pip install optuna cmaes") from exc
        self._optuna = optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        if sampler == "tpe":
            sampler_obj = optuna.samplers.TPESampler(seed=seed)
        elif sampler == "cmaes":
            # CMA-ES does not handle the categorical WINDOW_PATTERN; Optuna samples it
            # independently (at random), as in the upstream backend. The flag only
            # silences the warning Optuna would print about this for every trial.
            sampler_obj = optuna.samplers.CmaEsSampler(seed=seed, warn_independent_sampling=False)
        else:
            raise ValueError(f"unknown Optuna sampler {sampler!r}")
        self.name = sampler
        self._study = optuna.create_study(sampler=sampler_obj, direction="minimize")
        dist = optuna.distributions
        self._distributions = {}
        for hp in SEARCH_SPACE:
            if hp["type"] == "cat":
                self._distributions[hp["name"]] = dist.CategoricalDistribution(hp["choices"])
            elif hp["type"] == "int":
                self._distributions[hp["name"]] = dist.IntDistribution(hp["low"], hp["high"], log=hp["log"])
            else:
                self._distributions[hp["name"]] = dist.FloatDistribution(hp["low"], hp["high"], log=hp["log"])
        self._seen = 0             # history entries already in the study
        self._pending = None       # the trial asked for whose result is not known yet

    @staticmethod
    def _value(trial: Dict[str, Any]) -> float:
        return trial["val_bpb"] if trial["status"] == STATUS_OK else FAILURE_PENALTY

    def __call__(self, task: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, Any]:
        for index in range(self._seen, len(history)):
            trial = history[index]
            if trial["config"] is None:
                continue               # an ask that failed: nothing was trained
            if index == 0:
                # The starting configuration, as upstream: a completed trial the
                # sampler did not propose.
                self._study.add_trial(self._optuna.trial.create_trial(
                    params=trial["config"], distributions=self._distributions,
                    value=self._value(trial)))
                continue
            if self._pending is None:
                self._pending = self._study.ask(self._distributions)    # replaying a log
            if normalize_config(dict(self._pending.params)) != trial["config"]:
                raise EpisodeLogMismatch(
                    f"trial {index} of the episode log is not what the {self.name} sampler "
                    f"proposes at that point (seed {task['seed']}, Optuna "
                    f"{self._optuna.__version__}); the log cannot be resumed with this "
                    f"sampler. Use another output directory.")
            # Told for the parameters the sampler proposed, as upstream does (the
            # trained config may differ from them by the power-of-two snapping).
            self._study.tell(self._pending, self._value(trial))
            self._pending = None
        self._seen = len(history)
        self._pending = self._study.ask(self._distributions)
        return {"config": dict(self._pending.params), "source": self.name}


def make_baseline(name: str, entry: Dict[str, Any]) -> Callable:
    """Build the baseline policy ``name`` for one task (seeded with the task's seed)."""
    if name == "random":
        return random_search
    if name in ("tpe", "cmaes"):
        return OptunaPolicy(name, entry["seed"])
    raise ValueError(f"unknown baseline {name!r}; choose from {BASELINES}")
