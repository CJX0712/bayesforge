"""Acquisition functions for GP-based Bayesian optimization.

Convention: every acquisition returns a scalar to **maximize** over candidates.
(We minimize the objective, so high acquisition == promising point.)
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr

from ..core.errors import acquisition_err
from ..core.types import Bounds


def _cdf(z: np.ndarray) -> np.ndarray:
    """Standard normal CDF (vectorised, exact)."""
    return ndtr(np.asarray(z, dtype=np.float64))


def _pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def expected_improvement(mean: np.ndarray, std: np.ndarray, best_y: float, kappa: float = 0.0) -> np.ndarray:
    s = np.maximum(std, 1e-12)
    z = (best_y - mean) / s
    return s * (z * _cdf(z) + _pdf(z))


def probability_of_improvement(mean: np.ndarray, std: np.ndarray, best_y: float, kappa: float = 0.0) -> np.ndarray:
    s = np.maximum(std, 1e-12)
    z = (best_y - mean) / s
    return _cdf(z)


def lower_confidence_bound(mean: np.ndarray, std: np.ndarray, best_y: float, kappa: float = 2.0) -> np.ndarray:
    # maximize (kappa*std - mean)  <=>  minimize LCB = mean - kappa*std
    return kappa * std - mean


_ACQ = {
    "ei": expected_improvement,
    "pi": probability_of_improvement,
    "lcb": lower_confidence_bound,
}


def get_acquisition(name: str):
    if name not in _ACQ:
        raise acquisition_err(f"unknown acquisition {name!r}")
    return _ACQ[name]


def portfolio(mean: np.ndarray, std: np.ndarray, best_y: float, kappa: float = 2.0) -> np.ndarray:
    """Robust blend: max over per-batch min-max normalized EI and LCB.

    Normalizing inside the candidate batch keeps the two heterogeneous scores
    on a comparable footing and avoids one acquisition silently dominating.
    """
    ei = expected_improvement(mean, std, best_y)
    lcb = lower_confidence_bound(mean, std, best_y, kappa)

    def _norm(a: np.ndarray) -> np.ndarray:
        lo, hi = float(a.min()), float(a.max())
        if hi - lo < 1e-12:
            return np.zeros_like(a)
        return (a - lo) / (hi - lo)

    return np.maximum(_norm(ei), _norm(lcb))


def optimize_acquisition(
    surrogate,
    acq_fn,
    bounds: Bounds,
    best_y: float,
    rng: np.random.Generator,
    kappa: float = 2.0,
    n_random: int = 1000,
    n_multistart: int = 5,
    init_candidates: np.ndarray | None = None,
    init_values: np.ndarray | None = None,
) -> np.ndarray:
    """Return the candidate x* that maximizes the acquisition on the box.

    ``init_candidates`` / ``init_values`` let a caller reuse an already-screened
    batch (and its acquisition values) instead of paying for screening twice —
    used by BayesFuse, which screens once and then decides which acquisition to
    refine with.
    """
    if hasattr(surrogate, "dim") and surrogate.dim != bounds.dim:
        raise acquisition_err("surrogate dim != bounds dim")

    def neg(x):
        m, s = surrogate.predict(x.reshape(1, -1))
        return -float(acq_fn(m[0], s[0], best_y, kappa))

    # random screening (reused when the caller supplies it)
    if init_candidates is None:
        cand = bounds.sample(n_random, rng)
        init_candidates = cand
    else:
        cand = np.asarray(init_candidates, dtype=np.float64)
    if init_values is None:
        m, s = surrogate.predict(cand)
        acq_vals = acq_fn(m, s, best_y, kappa)
    else:
        acq_vals = np.asarray(init_values, dtype=np.float64)
    order = np.argsort(-acq_vals)
    top = cand[order[:n_multistart]]
    starts = [top[i] for i in range(top.shape[0])]
    if starts:
        starts.append(cand[order[n_multistart % cand.shape[0]]])
    starts.append(bounds.sample(1, rng)[0])

    best_x = starts[0]
    best_val = np.inf
    for x0 in starts:
        try:
            res = minimize(neg, x0, method="L-BFGS-B", bounds=list(zip(bounds.lo, bounds.hi)))
        except Exception:
            continue
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
    return bounds.clip(np.asarray(best_x, dtype=np.float64))
