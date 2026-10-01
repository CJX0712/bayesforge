"""Acquisition-function invariants (author: 晨星).

Hard invariant: the analytic EI is cross-checked against a brute-force Monte
Carlo estimate ``E[max(best - y, 0)]`` with y ~ N(mean, std^2).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import ndtr

from bayesforge.acquisition.acq import (
    expected_improvement,
    get_acquisition,
    lower_confidence_bound,
    optimize_acquisition,
    portfolio,
    probability_of_improvement,
)
from bayesforge.core.errors import BayesForgeError
from bayesforge.surrogate.gp import NumpyGP


def test_unknown_acquisition_raises():
    with pytest.raises(BayesForgeError):
        get_acquisition("nope")


def test_ei_matches_monte_carlo():
    rng = np.random.default_rng(0)
    best_y = 0.0
    for mean, std in [(-1.0, 0.5), (0.0, 1.0), (1.0, 2.0), (-0.3, 0.2)]:
        an = float(expected_improvement(np.array([mean]), np.array([std]), best_y)[0])
        draws = rng.normal(mean, std, size=400_000)
        mc = float(np.mean(np.maximum(best_y - draws, 0.0)))
        assert an == pytest.approx(mc, abs=5e-3), f"mean={mean} std={std}: EI={an} MC={mc}"


def test_ei_is_nonnegative_and_zero_without_uncertainty():
    vals = expected_improvement(np.array([-2.0, -1.0, 3.0]), np.array([0.5, 1.0, 2.0]), 0.0)
    assert np.all(vals >= 0.0)
    z = expected_improvement(np.array([1.0]), np.array([1e-15]), 0.0)
    assert float(z[0]) == pytest.approx(0.0, abs=1e-6)


def test_pi_matches_normal_cdf_and_bounded():
    mean, std, best = np.array([0.5, -1.0]), np.array([1.0, 0.5]), 0.0
    got = probability_of_improvement(mean, std, best)
    want = ndtr((best - mean) / std)
    assert np.allclose(got, want)
    assert np.all(got >= 0.0) and np.all(got <= 1.0)


def test_lcb_is_kappa_linear_and_monotone():
    m, s = np.array([1.0, 2.0]), np.array([0.5, 0.5])
    a = lower_confidence_bound(m, s, 0.0, kappa=1.0)
    b = lower_confidence_bound(m, s, 0.0, kappa=3.0)
    assert np.all(b >= a)  # more exploration weight -> higher score
    assert np.allclose(b - a, 2.0 * s)


def test_portfolio_is_bounded_unit_range():
    rng = np.random.default_rng(1)
    m = rng.normal(size=500)
    s = rng.uniform(0.01, 2.0, size=500)
    p = portfolio(m, s, 0.0, kappa=2.0)
    assert np.all(p >= -1e-12) and np.all(p <= 1.0 + 1e-12)


def test_portfolio_degenerate_batch_is_finite():
    p = portfolio(np.zeros(10), np.zeros(10) + 1e-15, 0.0)
    assert np.all(np.isfinite(p))


# --------------------- acquisition optimiser ---------------------
def _fit_gp(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, size=(12, 2))
    y = np.sin(X[:, 0]) + 0.25 * X[:, 1] ** 2
    return NumpyGP(kernel="rbf", noise_floor=1e-6).fit(X, y)


@pytest.mark.parametrize("acq", ["ei", "pi", "lcb"])
def test_optimize_acquisition_returns_point_inside_bounds(acq, box2d):
    gp = _fit_gp()
    rng = np.random.default_rng(0)
    x = optimize_acquisition(gp, get_acquisition(acq), box2d, -1.0, rng)
    assert x.shape == (2,)
    assert np.all(x >= box2d.lo - 1e-9) and np.all(x <= box2d.hi + 1e-9)


def test_optimize_acquisition_beats_random_screening(box2d):
    """Brute-force cross-check: the optimiser must beat a large random sample."""
    gp = _fit_gp(seed=3)
    rng = np.random.default_rng(7)
    fn = get_acquisition("ei")
    x = optimize_acquisition(gp, fn, box2d, -1.0, rng)
    cand = box2d.sample(20_000, np.random.default_rng(11))
    m, s = gp.predict(cand)
    best_random = float(np.max(fn(m, s, -1.0)))
    m1, s1 = gp.predict(x.reshape(1, -1))
    got = float(fn(m1, s1, -1.0)[0])
    assert got >= best_random - 1e-9


def test_optimize_acquisition_dim_mismatch_raises(box2d):
    gp = _fit_gp()
    from bayesforge.core.types import Bounds
    bad = Bounds(lo=np.zeros(5), hi=np.ones(5))
    with pytest.raises(BayesForgeError):
        optimize_acquisition(gp, get_acquisition("ei"), bad, 0.0, np.random.default_rng(0))
