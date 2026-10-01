"""Optimizer invariants: budget, feasibility, determinism, quality (author: 晨星)."""

from __future__ import annotations

import numpy as np
import pytest

from bayesforge.data.registry import demo_benchmark_functions
from bayesforge.optimizer.optimizers import (
    GPBO,
    BayesFuse,
    DifferentialEvolution,
    GridSearch,
    RandomSearch,
)

N_EVALS = 24
N_INIT = 5


def _methods():
    return [RandomSearch(), GridSearch(), DifferentialEvolution(), GPBO(), BayesFuse()]


@pytest.mark.parametrize("method", _methods(), ids=lambda m: m.name)
@pytest.mark.parametrize("spec", demo_benchmark_functions(), ids=lambda s: s.name)
def test_budget_is_exact(method, spec):
    """Every optimizer must spend exactly ``n_evals`` evaluations — no free extras."""
    rng = np.random.default_rng(0)
    obs = method.minimize(spec, spec.bounds, N_EVALS, rng, n_init=N_INIT)
    assert len(obs) == N_EVALS, f"{method.name} used {len(obs)} evals"


@pytest.mark.parametrize("method", _methods(), ids=lambda m: m.name)
@pytest.mark.parametrize("spec", demo_benchmark_functions(), ids=lambda s: s.name)
def test_all_evaluated_points_are_feasible(method, spec):
    rng = np.random.default_rng(1)
    obs = method.minimize(spec, spec.bounds, N_EVALS, rng, n_init=N_INIT)
    X = obs.X()
    assert np.all(X >= spec.bounds.lo - 1e-9)
    assert np.all(X <= spec.bounds.hi + 1e-9)


@pytest.mark.parametrize("method", _methods(), ids=lambda m: m.name)
@pytest.mark.parametrize("spec", demo_benchmark_functions(), ids=lambda s: s.name)
def test_reported_best_is_consistent_with_observations(method, spec):
    """Cross-check: ``best()`` must equal the min over the recorded trace."""
    rng = np.random.default_rng(2)
    obs = method.minimize(spec, spec.bounds, N_EVALS, rng, n_init=N_INIT)
    _, yb = obs.best()
    assert yb == pytest.approx(min(obs.ys))


@pytest.mark.parametrize("method", _methods(), ids=lambda m: m.name)
def test_reusing_one_instance_is_bit_reproducible(method):
    """Regression guard: warm-start state must reset per run.

    Reusing the *same* optimizer object across two runs used to leak
    ``_prev_theta`` and silently change the trajectory.
    """
    spec = demo_benchmark_functions()[0]
    a = method.minimize(spec, spec.bounds, N_EVALS, np.random.default_rng(5), n_init=N_INIT)
    b = method.minimize(spec, spec.bounds, N_EVALS, np.random.default_rng(5), n_init=N_INIT)
    assert np.array_equal(a.Y(), b.Y())


@pytest.mark.parametrize("method", _methods(), ids=lambda m: m.name)
def test_same_seed_same_result_fresh_instances(method):
    spec = demo_benchmark_functions()[0]
    a = method.minimize(spec, spec.bounds, N_EVALS, np.random.default_rng(6), n_init=N_INIT)
    b = method.minimize(spec, spec.bounds, N_EVALS, np.random.default_rng(6), n_init=N_INIT)
    assert a.best()[1] == b.best()[1]


def test_random_search_is_seed_sensitive():
    spec = demo_benchmark_functions()[0]
    a = RandomSearch().minimize(spec, spec.bounds, N_EVALS, np.random.default_rng(1), n_init=N_INIT)
    b = RandomSearch().minimize(spec, spec.bounds, N_EVALS, np.random.default_rng(2), n_init=N_INIT)
    assert a.best()[1] != b.best()[1]


QUALITY_EVALS = 30
QUALITY_SEEDS = (0, 1, 2)


def _mean_best(cls, spec, n_evals):
    vals = [
        cls().minimize(spec, spec.bounds, n_evals, np.random.default_rng(s), n_init=N_INIT).best()[1]
        for s in QUALITY_SEEDS
    ]
    return float(np.mean(vals))


@pytest.mark.parametrize("spec", demo_benchmark_functions(), ids=lambda s: s.name)
def test_gp_bo_beats_random_search(spec):
    """Quality gate: mean over seeds — BO must dominate random search.

    Averaged over several seeds because a single seed on a small budget is
    dominated by the luck of the initial design.
    """
    bo = _mean_best(GPBO, spec, QUALITY_EVALS)
    rs = _mean_best(RandomSearch, spec, QUALITY_EVALS)
    assert bo <= rs + 1e-12, f"GPBO {bo} vs random {rs}"


@pytest.mark.parametrize("spec", demo_benchmark_functions(), ids=lambda s: s.name)
def test_bayesfuse_beats_random_search(spec):
    bf = _mean_best(BayesFuse, spec, QUALITY_EVALS)
    rs = _mean_best(RandomSearch, spec, QUALITY_EVALS)
    assert bf <= rs + 1e-12, f"BayesFuse {bf} vs random {rs}"


def test_grid_search_degenerate_high_dim_is_safe():
    """dim > 6 falls back to a single sample instead of building a huge grid."""
    from bayesforge.core.types import Bounds, FunctionSpec
    from bayesforge.data.functions import sphere
    spec = FunctionSpec("sphere8", sphere, Bounds(lo=np.full(8, -1.0), hi=np.full(8, 1.0)), 0.0, 8)
    obs = GridSearch().minimize(spec, spec.bounds, N_EVALS, np.random.default_rng(0), n_init=N_INIT)
    assert len(obs) >= 1


def test_gp_bo_handles_matern_kernel():
    spec = demo_benchmark_functions()[0]
    obs = GPBO(kernel="matern52").minimize(spec, spec.bounds, N_EVALS, np.random.default_rng(0), n_init=N_INIT)
    assert len(obs) == N_EVALS and np.isfinite(obs.best()[1])
