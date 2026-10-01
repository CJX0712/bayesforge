"""GP surrogate invariants (author: 晨星).

Hard invariants:
  1. Analytic NLL gradient == central finite differences (the classic GPML
     ``-0.5 tr((aa^T - K^-1) dK/dtheta)`` identity). A sign error here silently
     destroys hyper-parameter learning, so it is pinned numerically.
  2. Noise-free GP interpolates its own training targets.
  3. Posterior variance is non-negative and shrinks at data points.
"""

from __future__ import annotations

import numpy as np
import pytest

from bayesforge.core.errors import BayesForgeError
from bayesforge.surrogate.gp import NumpyGP, _kernel, available_numpy


def _make_data(n: int = 12, d: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2.0, 2.0, size=(n, d))
    y = np.sin(X[:, 0]) + (0.5 * X[:, 1] ** 2 if d > 1 else 0.0)
    return X, y


# ----------------------------- kernel -----------------------------
@pytest.mark.parametrize("kern", ["rbf", "matern52"])
def test_kernel_is_symmetric_psd(kern):
    rng = np.random.default_rng(1)
    X = rng.uniform(-1, 1, size=(8, 3))
    K = _kernel(kern, X, X, np.array([1.0, 1.0, 1.0]), 1.0)
    assert np.allclose(K, K.T, atol=1e-12)
    w = np.linalg.eigvalsh(K)
    assert w.min() > -1e-8  # positive semi-definite
    assert np.allclose(np.diag(K), 1.0)  # unit signal variance


def test_kernel_rejects_unknown_name():
    with pytest.raises(BayesForgeError):
        NumpyGP(kernel="bogus")


# ----------------------------- gradient -----------------------------
def _nll(gp: NumpyGP, theta, Xs, ys, n, d, nf):
    val, _ = gp._neg_nll_rbf_grad(theta, Xs, ys, n, d, nf)
    return val


def _numeric_grad(gp: NumpyGP, theta, Xs, ys, n, d, nf, eps=1e-6):
    g = np.zeros_like(theta)
    for i in range(theta.size):
        tp, tm = theta.copy(), theta.copy()
        tp[i] += eps
        tm[i] -= eps
        g[i] = (_nll(gp, tp, Xs, ys, n, d, nf) - _nll(gp, tm, Xs, ys, n, d, nf)) / (2 * eps)
    return g


def test_analytic_gradient_matches_finite_differences():
    X, y = _make_data(n=10, d=2, seed=5)
    gp = NumpyGP(kernel="rbf", noise_floor=1e-6)
    gp.fit(X, y)
    n, d = X.shape
    Xs = (X - gp._xmean) / gp._xstd
    ys = (y - gp._ymean) / gp._ystd
    for theta in (np.zeros(d + 2), np.array([0.5, -0.5, 0.3, -2.0]),
                  np.array([-1.0, 1.0, -0.4, -4.0])):
        _, ag = gp._neg_nll_rbf_grad(theta, Xs, ys, n, d, 1e-6)
        ng = _numeric_grad(gp, theta, Xs, ys, n, d, 1e-6)
        assert np.max(np.abs(ag - ng)) < 1e-5, f"theta={theta} ag={ag} ng={ng}"


def test_gradient_descent_actually_lowers_nll():
    """Regression guard: with the correct sign, fitting beats theta=0."""
    X, y = _make_data(n=14, d=2, seed=11)
    gp = NumpyGP(kernel="rbf")
    gp.fit(X, y)
    n, d = X.shape
    Xs = (X - gp._xmean) / gp._xstd
    ys = (y - gp._ymean) / gp._ystd
    nll0 = _nll(gp, np.zeros(d + 2), Xs, ys, n, d, 1e-6)
    nllf = _nll(gp, gp._theta, Xs, ys, n, d, 1e-6)
    assert nllf <= nll0 + 1e-9


# ----------------------------- fit / predict -----------------------------
@pytest.mark.parametrize("kern", ["rbf", "matern52"])
def test_gp_interpolates_training_data(kern):
    X, y = _make_data(n=15, d=2, seed=2)
    gp = NumpyGP(kernel=kern, noise_floor=1e-8).fit(X, y)
    m, s = gp.predict(X)
    assert np.max(np.abs(m - y)) < 1e-2, "noise-free GP must interpolate its targets"


def test_gp_posterior_std_nonnegative_and_shrinks_at_data():
    X, y = _make_data(n=15, d=2, seed=4)
    gp = NumpyGP(kernel="rbf", noise_floor=1e-8).fit(X, y)
    rng = np.random.default_rng(0)
    Xq = rng.uniform(-2, 2, size=(50, 2))
    _, s_far = gp.predict(Xq)
    _, s_near = gp.predict(X)
    assert np.all(s_far >= 0.0)
    assert float(s_near.max()) < float(s_far.mean())


def test_gp_far_from_data_returns_prior_std():
    X, y = _make_data(n=10, d=1, seed=6)
    gp = NumpyGP(kernel="rbf").fit(X, y)
    _, s = gp.predict(np.array([[500.0]]))
    assert float(s[0]) > 0.5 * gp._ystd  # uncertainty reverts to prior


def test_gp_rejects_predict_before_fit():
    from bayesforge.core.errors import BayesForgeError as BFE
    with pytest.raises(BFE):
        NumpyGP().predict(np.array([[0.0]]))


def test_gp_fit_is_deterministic():
    X, y = _make_data(n=12, d=2, seed=8)
    g1 = NumpyGP(kernel="rbf").fit(X, y)
    g2 = NumpyGP(kernel="rbf").fit(X, y)
    assert np.array_equal(g1._theta, g2._theta)
    m1, _ = g1.predict(X)
    m2, _ = g2.predict(X)
    assert np.array_equal(m1, m2)


def test_gp_warm_start_does_not_degrade_fit():
    """Warm-start is a speed hint: it must not land in a worse optimum."""
    X, y = _make_data(n=12, d=2, seed=9)
    gp = NumpyGP(kernel="rbf")
    g_cold = NumpyGP(kernel="rbf").fit(X, y)
    g_warm = NumpyGP(kernel="rbf").fit(X, y, init_theta=g_cold._theta)
    n, d = X.shape
    Xs = (X - g_cold._xmean) / g_cold._xstd
    ys = (y - g_cold._ymean) / g_cold._ystd
    nll_cold = _nll(gp, g_cold._theta, Xs, ys, n, d, 1e-6)
    nll_warm = _nll(gp, g_warm._theta, Xs, ys, n, d, 1e-6)
    assert np.isfinite(nll_warm)
    assert nll_warm <= nll_cold + 1e-3


def test_gp_hyperparams_stay_in_bounds():
    """Guard against exp() overflow producing NaN thetas."""
    X, y = _make_data(n=20, d=3, seed=13)
    gp = NumpyGP(kernel="rbf").fit(X, y)
    assert np.all(np.isfinite(gp._theta))
    assert np.all(gp._theta <= 5.0 + 1e-9)
    assert gp._theta[-1] <= 1.0 + 1e-9
    assert np.isfinite(gp._ell).all() and np.all(gp._ell > 0)


def test_available_numpy():
    assert available_numpy() is True
