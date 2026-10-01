"""Evaluation metrics for black-box optimization."""

from __future__ import annotations

import numpy as np

from ..core.types import Observed


def simple_regret(obs: Observed, true_opt: float) -> float:
    """Final regret = best objective found - known global optimum (>=0 ideal)."""
    if len(obs) == 0:
        return float("inf")
    return float(obs.best()[1]) - true_opt


def cumulative_regret(obs: Observed, true_opt: float) -> float:
    """Sum of instantaneous regret over the evaluation order."""
    if len(obs) == 0:
        return float("inf")
    return float(np.sum(np.asarray(obs.ys, dtype=np.float64) - true_opt))


def regret_trace(obs: Observed, true_opt: float) -> np.ndarray:
    """Best-so-far regret trajectory (monotone non-increasing)."""
    ys = np.asarray(obs.ys, dtype=np.float64)
    best = np.minimum.accumulate(ys)
    return best - true_opt
