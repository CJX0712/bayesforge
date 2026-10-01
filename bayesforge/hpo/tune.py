"""Lightweight HPO for the BO internal hyper-parameters.

Runs a cheap inner cross-validation sweep over ``(kernel, acq)`` for GPBO and
returns the best combo. Kept small so it can be used inside an ablation without
blowing the CPU budget; the default demo does not call it.
"""

from __future__ import annotations

from typing import List, Tuple

from ..core.types import FunctionSpec
from ..optimizer.optimizers import GPBO
from ..pipeline.benchmark import benchmark


def tune_gpbo(specs: List[FunctionSpec], seeds: List[int], n_evals: int = 40,
              n_init: int = 5, base_seed: int = 7) -> Tuple[str, str]:
    kernels = ["rbf", "matern52"]
    acqs = ["ei", "lcb", "pi"]
    best, best_key = float("inf"), ("rbf", "ei")
    for k in kernels:
        for a in acqs:
            methods = [GPBO(kernel=k, acq=a)]
            rep = benchmark(specs, methods, seeds, n_evals, n_init, base_seed=base_seed)
            mean_sr = rep["summary_by_method"]["gp_bo"]["mean_simple_regret"]
            if mean_sr < best:
                best, best_key = mean_sr, (k, a)
    return best_key
