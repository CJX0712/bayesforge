"""Core invariants: Bounds, Observed, metrics, config, seeding (author: 晨星)."""

from __future__ import annotations

import numpy as np
import pytest

from bayesforge.core.config import Config, load_config
from bayesforge.core.errors import BayesForgeError
from bayesforge.core.seed import set_all
from bayesforge.core.types import Bounds, Observed
from bayesforge.eval.metrics import cumulative_regret, regret_trace, simple_regret


# ----------------------------- Bounds -----------------------------
def test_bounds_rejects_inverted_box():
    with pytest.raises(BayesForgeError):
        Bounds(lo=np.array([1.0]), hi=np.array([0.0]))


def test_bounds_rejects_shape_mismatch():
    with pytest.raises(BayesForgeError):
        Bounds(lo=np.array([0.0, 0.0]), hi=np.array([1.0]))


def test_bounds_sample_stays_inside_box(box2d, rng):
    pts = box2d.sample(500, rng)
    assert pts.shape == (500, 2)
    assert np.all(pts >= box2d.lo - 0.0)
    assert np.all(pts <= box2d.hi + 0.0)


def test_bounds_sample_is_seed_deterministic(box2d):
    a = box2d.sample(50, np.random.default_rng(7))
    b = box2d.sample(50, np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_bounds_clip_projects_outliers(box2d):
    x = np.array([-100.0, 100.0])
    c = box2d.clip(x)
    assert np.allclose(c, np.array([-5.0, 15.0]))


# ----------------------------- Observed -----------------------------
def test_observed_best_returns_minimum():
    o = Observed()
    o.add(np.array([0.0]), 3.0)
    o.add(np.array([1.0]), -1.0)
    o.add(np.array([2.0]), 2.0)
    xb, yb = o.best()
    assert yb == -1.0
    assert np.allclose(xb, np.array([1.0]))


def test_observed_empty_raises():
    with pytest.raises(ValueError):
        Observed().best()


def test_observed_X_Y_shapes():
    o = Observed()
    for i in range(4):
        o.add(np.array([float(i), float(i)]), float(i))
    assert o.X().shape == (4, 2)
    assert o.Y().shape == (4,)
    assert len(o) == 4


# ----------------------------- metrics -----------------------------
def test_simple_regret_is_nonnegative_and_exact():
    o = Observed()
    o.add(np.array([0.0]), 3.0)
    o.add(np.array([1.0]), 1.0)
    assert simple_regret(o, 0.5) == pytest.approx(0.5)


def test_simple_regret_empty_is_inf():
    assert simple_regret(Observed(), 0.0) == float("inf")


def test_regret_trace_is_monotone_nonincreasing(rng):
    o = Observed()
    ys = rng.random(30)
    for i, y in enumerate(ys):
        o.add(np.array([float(i)]), float(y))
    tr = regret_trace(o, 0.0)
    assert np.all(np.diff(tr) <= 1e-12)
    assert tr[-1] == pytest.approx(simple_regret(o, 0.0))


def test_cumulative_regret_matches_manual_sum():
    o = Observed()
    for y in (2.0, 1.0, 4.0):
        o.add(np.array([0.0]), y)
    assert cumulative_regret(o, 1.0) == pytest.approx((1.0 + 0.0 + 3.0))


# ----------------------------- config -----------------------------
def test_config_defaults_validate():
    cfg = load_config()
    cfg.validate()
    assert cfg.seed == 42 and cfg.kernel == "rbf" and cfg.acq == "ei"


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("BO_SEED", "7")
    monkeypatch.setenv("BO_N_EVALS", "30")
    monkeypatch.setenv("BO_KERNEL", "matern52")
    cfg = load_config()
    assert cfg.seed == 7 and cfg.n_evals == 30 and cfg.kernel == "matern52"


def test_config_rejects_bad_kernel():
    with pytest.raises(BayesForgeError):
        load_config(kernel="nope")


def test_config_rejects_n_evals_le_n_init():
    with pytest.raises(BayesForgeError):
        load_config(n_evals=5, n_init=5)


def test_config_rejects_negative_seed():
    with pytest.raises(BayesForgeError):
        Config(seed=-1).validate()


# ----------------------------- seeding -----------------------------
def test_set_all_makes_global_rng_reproducible():
    set_all(99)
    a = float(np.random.rand())
    set_all(99)
    b = float(np.random.rand())
    assert a == b
