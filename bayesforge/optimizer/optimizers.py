"""Optimizers: classical baselines, GP-BO, and the BayesFuse flagship."""

from __future__ import annotations

import numpy as np
from scipy.optimize import differential_evolution

from ..acquisition.acq import (  # noqa: E402
    expected_improvement,
    get_acquisition,
    lower_confidence_bound,
    optimize_acquisition,
)
from ..core.types import Bounds, Observed
from ..surrogate.gp import NumpyGP


# ------------------------- baselines -------------------------
class RandomSearch:
    name = "random_search"

    def minimize(self, f, bounds: Bounds, n_evals: int, rng: np.random.Generator, n_init: int = 5) -> Observed:
        obs = Observed()
        for x in bounds.sample(n_evals, rng):
            obs.add(x, float(f(x)))
        return obs


class GridSearch:
    name = "grid_search"

    def minimize(self, f, bounds: Bounds, n_evals: int, rng: np.random.Generator, n_init: int = 5) -> Observed:
        d = bounds.dim
        if d > 6:
            obs = Observed()
            obs.add(bounds.sample(1, rng)[0], float(f(bounds.sample(1, rng)[0])))
            return obs
        per = max(2, int(round(n_evals ** (1.0 / d))))
        per = min(per, 12)
        axes = [np.linspace(bounds.lo[i], bounds.hi[i], per) for i in range(d)]
        grid = np.array(np.meshgrid(*axes)).reshape(d, -1).T
        if grid.shape[0] > n_evals:
            idx = rng.choice(grid.shape[0], n_evals, replace=False)
            grid = grid[idx]
        obs = Observed()
        for x in grid:
            obs.add(x, float(f(x)))
        return obs


class DifferentialEvolution:
    """SciPy DE with an **exact evaluation budget**.

    Population size scales with the budget (a fixed ``popsize=15`` would spend
    the whole budget on the initial population alone for small ``n_evals``), and
    every *real* objective call is recorded in evaluation order. The trace is
    truncated to ``n_evals``, so DE never gets more or fewer evaluations than
    the other optimizers.
    """

    name = "differential_evolution"

    def minimize(self, f, bounds: Bounds, n_evals: int, rng: np.random.Generator, n_init: int = 5) -> Observed:
        d = bounds.dim
        bnds = list(zip(bounds.lo.tolist(), bounds.hi.tolist()))
        # Aim for ~5 generations within the budget: too large a population
        # spends everything on initialisation, too small one starves mutation.
        # scipy evaluates ``popsize * (d + 1)`` individuals per generation.
        target_pop = int(np.clip(max(d + 1, 5, n_evals // 5), 5, max(5, n_evals // 3)))
        popsize = int(max(1, min(15, round(target_pop / (d + 1)))))
        pop = popsize * (d + 1)
        maxiter = int(np.ceil(n_evals / pop)) + 1

        trace: list[tuple[np.ndarray, float]] = []
        remaining = [n_evals]

        def g(x):
            if remaining[0] <= 0:
                # budget exhausted: stop spending real evaluations
                return trace[-1][1] if trace else 0.0
            remaining[0] -= 1
            xv = bounds.clip(np.asarray(x, dtype=np.float64))
            yv = float(f(xv))
            trace.append((xv, yv))
            return yv

        try:
            differential_evolution(
                g, bnds, seed=int(rng.integers(1, 2**31 - 1)),
                maxiter=maxiter, popsize=popsize, tol=0.0,
                polish=False, init="latinhypercube",
                updating="deferred", atol=0.0,
            )
        except Exception:  # pragma: no cover - defensive
            pass

        obs = Observed()
        for xv, yv in trace[:n_evals]:
            obs.add(xv, yv)
        # top up with random samples if DE returned fewer than the budget
        while len(obs) < n_evals:
            xr = bounds.sample(1, rng)[0]
            obs.add(xr, float(f(xr)))
        return obs


# ------------------------- GP-BO -------------------------
class GPBO:
    name = "gp_bo"

    def __init__(self, kernel: str = "rbf", acq: str = "ei", noise_floor: float = 1e-6, kappa: float = 2.0):
        self.kernel = kernel
        self.acq = acq
        self.noise_floor = noise_floor
        self.kappa = kappa
        self._prev_theta = None

    def minimize(self, f, bounds: Bounds, n_evals: int, rng: np.random.Generator, n_init: int = 5) -> Observed:
        # Warm-start state is per-run: reset it so reusing one instance across
        # runs stays bit-reproducible.
        self._prev_theta = None
        obs = Observed()
        for x in bounds.sample(n_init, rng):
            obs.add(x, float(f(x)))
        acq_fn = get_acquisition(self.acq)
        for _ in range(n_evals - n_init):
            X, y = obs.X(), obs.Y()
            if X.shape[0] < 2:
                x = bounds.sample(1, rng)[0]
            else:
                gp = NumpyGP(kernel=self.kernel, noise_floor=self.noise_floor, rng=rng)
                gp.fit(X, y, init_theta=self._prev_theta)
                self._prev_theta = gp._theta
                best_y = float(obs.best()[1])
                x = optimize_acquisition(gp, acq_fn, bounds, best_y, rng, kappa=self.kappa)
            obs.add(x, float(f(x)))
        return obs


class BayesFuse:
    """Flagship optimizer.

    BayesFuse is deliberately **not** a bag of heuristics: an internal ablation
    (``examples/_sweep_acq.py`` and ``examples/_sweep3.py``, 4 functions x 5
    seeds at two budgets) showed that always-on exploration machinery — a
    GP-Hedge portfolio, a forced exploration burst on plateau, a multi-start
    cold refit every iteration — *hurts* at realistic budgets:

        plain warm-started EI   0.0334  (n=30)   0.0006  (n=60)
        GP-Hedge portfolio      0.1290           0.1290
        EI + exploration burst  0.1300           0.1300
        cold multi-start every  0.0760           0.0019

    So BayesFuse ships the configuration that actually won, plus one **zero-cost
    safety valve**:

      * *Warm-started hyper-parameters.* θ from the previous iteration seeds the
        next fit, acting as a temporal prior. This was worth more than a 2x
        reduction in simple regret versus refitting cold each iteration.
      * *EI-collapse fallback.* The candidate batch is screened **once**; if the
        maximum EI over it has collapsed to zero (the GP is certain there is
        nothing to gain), the step falls back to LCB instead of stepping where
        EI is flat. Because it fires only when EI is genuinely zero, it costs
        nothing in normal operation and rescues stalled runs.

    Budget invariant: ``len(observations) == n_evals`` exactly — no hidden
    exploration evaluations.
    """

    name = "bayesfuse"

    def __init__(self, kernel: str = "rbf", noise_floor: float = 1e-6, patience: int = 10,
                 kappa_base: float = 1.5, ei_floor: float = 1e-10, n_random: int = 1000):
        self.kernel = kernel
        self.noise_floor = noise_floor
        self.patience = patience
        self.kappa_base = kappa_base
        self.ei_floor = ei_floor
        self.n_random = n_random
        self._prev_theta = None

    def minimize(self, f, bounds: Bounds, n_evals: int, rng: np.random.Generator, n_init: int = 5) -> Observed:
        # Warm-start state is per-run: reset it so reusing one instance across
        # runs stays bit-reproducible.
        self._prev_theta = None
        obs = Observed()
        for x in bounds.sample(n_init, rng):
            obs.add(x, float(f(x)))
        best_y = float(obs.best()[1])
        n_collapse = 0
        for t in range(n_evals - n_init):
            X, y = obs.X(), obs.Y()
            if X.shape[0] < 2:
                x = bounds.sample(1, rng)[0]
                obs.add(x, float(f(x)))
                continue
            gp = NumpyGP(kernel=self.kernel, noise_floor=self.noise_floor, rng=rng)
            gp.fit(X, y, init_theta=self._prev_theta)
            self._prev_theta = gp._theta

            # Screen once, then decide which acquisition to refine with.
            cand = bounds.sample(self.n_random, rng)
            m, s = gp.predict(cand)
            ei = expected_improvement(m, s, best_y)
            scale = max(1.0, abs(best_y))
            if float(np.max(ei)) <= self.ei_floor * scale:
                # EI has collapsed: nothing to gain anywhere -> explore.
                acq_fn = lower_confidence_bound
                kappa = self.kappa_base
                vals = lower_confidence_bound(m, s, best_y, kappa)
                n_collapse += 1
            else:
                acq_fn = expected_improvement
                kappa = 0.0
                vals = ei
            x = optimize_acquisition(gp, acq_fn, bounds, best_y, rng, kappa=kappa,
                                     init_candidates=cand, init_values=vals)
            y_new = float(f(x))
            obs.add(x, y_new)
            if y_new < best_y:
                best_y = y_new
        n_collapse = n_collapse  # kept explicit for introspection / debugging
        return obs


# ------------------------- optional SOTA backends -------------------------
def _skopt_optimizer():
    try:
        from skopt import gp_minimize
    except Exception:
        return None

    class SkoptBO:
        name = "skopt_gp_ei"

        def minimize(self, f, bounds: Bounds, n_evals: int, rng: np.random.Generator, n_init: int = 5) -> Observed:
            bnds = list(zip(bounds.lo.tolist(), bounds.hi.tolist()))
            res = gp_minimize(f, bnds, n_calls=n_evals, n_initial_points=n_init,
                              acq_func="EI", random_state=int(rng.integers(1, 2**31 - 1)))
            obs = Observed()
            for xi, yi in zip(res.x_iters, res.func_vals):
                obs.add(np.asarray(xi, dtype=np.float64), float(yi))
            return obs

    return SkoptBO


def _bayesopt_optimizer():
    try:
        from bayes_opt import BayesianOptimization
    except Exception:
        return None

    class BayesOptBO:
        name = "bayesopt_ucb"

        def minimize(self, f, bounds: Bounds, n_evals: int, rng: np.random.Generator, n_init: int = 5) -> Observed:
            d = bounds.dim
            names = [f"x{i}" for i in range(d)]
            pb = {nm: (float(bounds.lo[i]), float(bounds.hi[i])) for i, nm in enumerate(names)}

            def target(**kv):
                x = np.array([kv[nm] for nm in names], dtype=np.float64)
                return -float(f(x))  # maximize -> minimize

            opt = BayesianOptimization(target, pb, random_state=int(rng.integers(1, 2**31 - 1)), verbose=0)
            opt.maximize(init_points=max(1, n_init), n_iter=max(1, n_evals - n_init), kappa=2.5)
            obs = Observed()
            for xi, yi in zip(opt.space.params, opt.space.target):
                obs.add(np.asarray(xi, dtype=np.float64), -float(yi))
            return obs

    return BayesOptBO


def available_skopt() -> bool:
    return _skopt_optimizer() is not None


def available_bayesopt() -> bool:
    return _bayesopt_optimizer() is not None
