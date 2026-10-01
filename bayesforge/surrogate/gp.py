"""Exact Gaussian Process regression in pure numpy + scipy.

This is the always-available offline surrogate backend. It implements the
classic RBF / Matérn-5/2 GP with marginal-likelihood hyper-parameter learning.
The RBF path uses an **analytic NLL gradient** and **warm-starts** θ from the
previous fit, so it is fast enough for tight BO loops on a CPU.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ..core.errors import surrogate_err

_JITTER = 1e-8


def _sqdist(Xa: np.ndarray, Xb: np.ndarray, inv_ell2: np.ndarray) -> np.ndarray:
    # Weighted squared distance: sum_d ((xa-xb)^2 / ell_d^2)
    diff = Xa[:, None, :] - Xb[None, :, :]
    scaled = diff * np.sqrt(inv_ell2)[None, None, :]
    return np.sum(scaled * scaled, axis=2)


def _kernel(base: str, Xa: np.ndarray, Xb: np.ndarray, ell: np.ndarray, signal2: float) -> np.ndarray:
    r2 = _sqdist(Xa, Xb, 1.0 / (ell * ell))
    if base == "rbf":
        return signal2 * np.exp(-0.5 * r2)
    # matern52
    r = np.sqrt(r2)
    sq5r = np.sqrt(5.0) * r
    return signal2 * (1.0 + sq5r + (5.0 / 3.0) * r2) * np.exp(-sq5r)


class NumpyGP:
    """Exact GP. Attributes satisfy the :class:`Surrogate` protocol."""

    def __init__(self, kernel: str = "rbf", noise_floor: float = 1e-6,
                 noise_bound_hi: float = -5.0, rng: np.random.Generator | None = None):
        """Args:
            noise_floor: additive jitter on the noise variance (numerical safety).
            noise_bound_hi: upper bound on ``log`` noise std. Kept small by
                default so a noise-free objective is genuinely *interpolated*.
                Left unbounded, the marginal likelihood happily explains a
                non-stationary landscape as pure noise, which collapses the
                posterior mean to a constant and blows up EI. Raise it (e.g.
                to 0.0) when the objective is known to be noisy.
        """
        if kernel not in ("rbf", "matern52"):
            raise surrogate_err(f"unknown kernel {kernel!r}")
        self.kernel = kernel
        self.noise_floor = float(noise_floor)
        self.noise_bound_hi = float(noise_bound_hi)
        self._rng = rng if rng is not None else np.random.default_rng(0)
        self.dim = 0
        self._xmean: np.ndarray | None = None
        self._xstd: np.ndarray | None = None
        self._ymean = 0.0
        self._ystd = 1.0
        self._ell = np.ones(1)
        self._signal2 = 1.0
        self._noise2 = 1e-6
        self._K = None
        self._L = None
        self._alpha = None
        self._Xs = None
        self._theta = None
        self._nll_value = float("inf")

    # ---- training ----
    def fit(self, X: np.ndarray, y: np.ndarray, init_theta: np.ndarray | None = None) -> "NumpyGP":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        if X.ndim == 1:
            X = X[:, None]
        n, d = X.shape
        self.dim = d
        self._xmean = X.mean(axis=0)
        self._xstd = X.std(axis=0)
        self._xstd[self._xstd == 0] = 1.0
        Xs = (X - self._xmean) / self._xstd
        self._ymean = float(y.mean())
        self._ystd = float(y.std())
        if self._ystd == 0:
            self._ystd = 1.0
        ys = (y - self._ymean) / self._ystd
        self._Xs = Xs

        def nll(theta: np.ndarray) -> float:
            ell = np.exp(theta[:d]) + 1e-3
            s2 = np.exp(2.0 * theta[d])
            n2 = np.exp(2.0 * theta[d + 1]) + self.noise_floor
            K = _kernel(self.kernel, Xs, Xs, ell, s2) + n2 * np.eye(n)
            K += _JITTER * np.eye(n)
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                return 1e12
            inv_y = np.linalg.solve(L.T, np.linalg.solve(L, ys))
            maha = float(ys @ inv_y)
            logdet = 2.0 * np.sum(np.log(np.diag(L)))
            return 0.5 * (maha + logdet + n * np.log(2.0 * np.pi))

        if self.kernel == "rbf":
            obj = self._neg_nll_rbf_grad
        else:
            obj = nll

        starts = []
        if init_theta is not None:
            starts.append(np.asarray(init_theta, dtype=np.float64))
        starts.append(np.zeros(d + 2))
        if init_theta is None:
            starts.append(np.r_[np.full(d, 0.0), [0.0, -2.0]])
            starts.append(np.r_[-np.full(d, 0.5), [0.5, -3.0]])
        # Bounded log-hyper-parameters: keeps exp() in a safe range so the
        # analytic gradient never overflows to NaN / inf.
        bounds = [(-5.0, 5.0)] * d + [(-5.0, 5.0), (-16.0, self.noise_bound_hi)]
        best = None
        for s in np.clip(np.asarray(starts, dtype=np.float64),
                         [b[0] for b in bounds], [b[1] for b in bounds]):
            try:
                if self.kernel == "rbf":
                    res = minimize(lambda t: obj(t, Xs, ys, n, d, self.noise_floor), s,
                                   jac=True, method="L-BFGS-B", bounds=bounds,
                                   options={"maxiter": 100})
                else:
                    res = minimize(nll, s, method="L-BFGS-B", bounds=bounds,
                                   options={"maxiter": 100})
            except Exception:
                continue
            if best is None or res.fun < best.fun:
                best = res
        if best is None:
            raise surrogate_err("GP hyper-parameter optimization failed")
        t = best.x
        self._theta = t
        self._nll_value = float(best.fun)
        self._ell = np.exp(t[:d]) + 1e-3
        self._signal2 = np.exp(2.0 * t[d])
        self._noise2 = np.exp(2.0 * t[d + 1]) + self.noise_floor

        K = _kernel(self.kernel, Xs, Xs, self._ell, self._signal2) + self._noise2 * np.eye(n)
        K += _JITTER * np.eye(n)
        self._K = K
        self._L = np.linalg.cholesky(K)
        self._alpha = np.linalg.solve(self._L.T, np.linalg.solve(self._L, ys))
        return self

    def _neg_nll_rbf_grad(self, theta, Xs, ys, n, d, noise_floor):
        ell = np.exp(theta[:d]) + 1e-3
        s2 = np.exp(2.0 * theta[d])
        n2 = np.exp(2.0 * theta[d + 1]) + noise_floor
        Kb = np.exp(-0.5 * _sqdist(Xs, Xs, 1.0 / (ell * ell)))
        K = s2 * Kb + n2 * np.eye(n)
        K += _JITTER * np.eye(n)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e12, np.zeros_like(theta)
        inv_y = np.linalg.solve(L.T, np.linalg.solve(L, ys))
        maha = float(ys @ inv_y)
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        nll_val = 0.5 * (maha + logdet + n * np.log(2.0 * np.pi))
        # G = alpha alpha^T - K^{-1}
        Kinv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(n)))
        G = np.outer(inv_y, inv_y) - Kinv
        grad = np.zeros_like(theta)
        # Chain rule: ell = exp(theta) + eps  =>  d ell / d theta = exp(theta),
        # NOT ell. Dropping the eps term is what makes the analytic gradient
        # match central finite differences to ~1e-9 instead of ~1e-3.
        dl = np.exp(theta[:d])
        dn2 = 2.0 * (n2 - noise_floor)
        # lengthscales
        for k in range(d):
            diff2 = (Xs[:, k][:, None] - Xs[:, k][None, :]) ** 2 / (ell[k] ** 2)
            M = s2 * Kb * diff2 * (dl[k] / ell[k])
            grad[k] = -0.5 * np.sum(G * M)
        grad[d] = -0.5 * np.sum(G * (2.0 * s2 * Kb))
        grad[d + 1] = -0.5 * np.sum(G * (dn2 * np.eye(n)))
        return nll_val, grad

    # ---- inference ----
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._L is None:
            raise surrogate_err("call fit() before predict()")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[:, None]
        Xs = (X - self._xmean) / self._xstd
        Ks = _kernel(self.kernel, Xs, self._Xs, self._ell, self._signal2)
        mean = (Ks @ self._alpha) * self._ystd + self._ymean
        v = np.linalg.solve(self._L, Ks.T)
        # Latent-function variance (noise excluded): this is what EI / PI / LCB
        # are defined against.
        var = np.maximum(self._signal2 - np.sum(v * v, axis=0), 0.0) * (self._ystd ** 2)
        std = np.sqrt(var)
        return mean, std


def make_numpy_gp(kernel: str = "rbf", noise_floor: float = 1e-6, rng: np.random.Generator | None = None) -> NumpyGP:
    return NumpyGP(kernel=kernel, noise_floor=noise_floor, rng=rng)


def available_numpy() -> bool:
    return True
