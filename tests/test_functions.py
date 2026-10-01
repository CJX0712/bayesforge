"""Benchmark-function invariants: known global optima (author: 晨星).

The hard invariant here is analytic-vs-declared: each :class:`FunctionSpec`
declares a global optimum, and we re-evaluate the function at the textbook
argmin to confirm the declaration is right. A wrong ``optimum`` would silently
corrupt every regret number downstream, so this is checked explicitly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bayesforge.data.registry import all_functions, demo_benchmark_functions, demo_functions

# Textbook global minimisers (rounded); tolerance accounts for the rounding.
KNOWN_MINIMISERS = {
    "sphere3": [np.zeros(3)],
    "rosenbrock3": [np.ones(3)],
    "ackley3": [np.zeros(3)],
    "levy3": [np.ones(3)],
    "branin": [np.array([-math.pi, 12.275]), np.array([math.pi, 2.275]), np.array([9.42478, 2.475])],
    "six_hump_camel": [np.array([0.0898, -0.7126]), np.array([-0.0898, 0.7126])],
    "goldstein_price": [np.array([0.0, -1.0])],
    "michalewicz2": [np.array([2.20, 1.57])],
    "hartmann3": [np.array([0.114614, 0.555649, 0.852547])],
    "hartmann6": [np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])],
}

# michalewicz2 argmin is quoted to 2 decimals, hence the looser tolerance.
TOL = {"michalewicz2": 2e-3, "six_hump_camel": 1e-6, "hartmann3": 1e-5}
DEFAULT_TOL = 1e-6


@pytest.mark.parametrize("spec", all_functions(), ids=lambda s: s.name)
def test_declared_optimum_matches_known_minimiser(spec):
    vals = [float(spec.f(p)) for p in KNOWN_MINIMISERS[spec.name]]
    tol = TOL.get(spec.name, DEFAULT_TOL)
    assert min(vals) == pytest.approx(spec.optimum, abs=tol)


@pytest.mark.parametrize("spec", all_functions(), ids=lambda s: s.name)
def test_spec_dim_matches_argmin_dim(spec):
    for p in KNOWN_MINIMISERS[spec.name]:
        assert p.shape[0] == spec.dim == spec.bounds.dim


@pytest.mark.parametrize("spec", all_functions(), ids=lambda s: s.name)
def test_function_is_finite_across_domain(spec):
    rng = np.random.default_rng(0)
    for x in spec.bounds.sample(200, rng):
        assert np.isfinite(float(spec(x)))


@pytest.mark.parametrize("spec", all_functions(), ids=lambda s: s.name)
def test_no_sample_beats_declared_optimum(spec):
    """Sanity: the declared optimum is a true lower bound on sampled values."""
    rng = np.random.default_rng(3)
    ys = [float(spec(x)) for x in spec.bounds.sample(2000, rng)]
    tol = TOL.get(spec.name, DEFAULT_TOL)
    assert min(ys) >= spec.optimum - tol


@pytest.mark.parametrize("spec", demo_functions(), ids=lambda s: s.name)
def test_demo_functions_are_noise_free(spec):
    assert spec.noise_std == 0.0
    x = spec.bounds.sample(1, np.random.default_rng(0))[0]
    assert float(spec(x)) == float(spec(x))


def test_noisy_spec_is_seed_deterministic():
    """Even with noise on, the spec carries its own RNG (no global state)."""
    from bayesforge.core.types import FunctionSpec
    from bayesforge.data.functions import sphere

    s1 = FunctionSpec("sphere3", sphere, all_functions()[0].bounds, 0.0, 3, noise_std=0.1)
    s2 = FunctionSpec("sphere3", sphere, all_functions()[0].bounds, 0.0, 3, noise_std=0.1)
    x = np.array([1.0, 2.0, 3.0])
    assert float(s1(x)) == float(s2(x))  # independent instances, same stream
    assert float(s1(x)) != 9.0 + 5.0     # noise actually applied


def test_demo_benchmark_subset_is_contained():
    names = {s.name for s in demo_benchmark_functions()}
    assert names <= {s.name for s in all_functions()}
    assert len(names) == 4
