"""Internal experiment: which acquisition strategy should back BayesFuse?

Compares (A) plain EI, (B) EI/PI/LCB GP-Hedge bandit, (C) EI + plateau restart
with a hyper-parameter reset (no random injection). Writes results to stdout.
Not part of the deliverable API — used to pick the shipped default.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bayesforge.acquisition.acq import (  # noqa: E402
    expected_improvement,
    lower_confidence_bound,
    optimize_acquisition,
    probability_of_improvement,
)
from bayesforge.core.types import Observed  # noqa: E402
from bayesforge.data.registry import demo_benchmark_functions  # noqa: E402
from bayesforge.surrogate.gp import NumpyGP  # noqa: E402

ARMS = {
    "ei": expected_improvement,
    "pi": probability_of_improvement,
    "lcb": lower_confidence_bound,
}


class PlainEI:
    name = "A_plain_ei"

    def minimize(self, f, bounds, n_evals, rng, n_init=5):
        obs = Observed()
        for x in bounds.sample(n_init, rng):
            obs.add(x, float(f(x)))
        prev = None
        for _ in range(n_evals - n_init):
            gp = NumpyGP(kernel="rbf").fit(obs.X(), obs.Y(), init_theta=prev)
            prev = gp._theta
            x = optimize_acquisition(gp, expected_improvement, bounds, obs.best()[1], rng)
            obs.add(x, float(f(x)))
        return obs


class Hedge:
    """GP-Hedge (Hoffman, Brochu & de Freitas, UAI 2011)."""

    name = "B_gp_hedge"

    def __init__(self, eta: float = 0.2):
        self.eta = eta

    def minimize(self, f, bounds, n_evals, rng, n_init=5):
        obs = Observed()
        for x in bounds.sample(n_init, rng):
            obs.add(x, float(f(x)))
        names = list(ARMS)
        gains = np.zeros(len(names))
        prev = None
        for _ in range(n_evals - n_init):
            gp = NumpyGP(kernel="rbf").fit(obs.X(), obs.Y(), init_theta=prev)
            prev = gp._theta
            best = obs.best()[1]
            z = self.eta * (gains - gains.max())
            w = np.exp(z)
            w /= w.sum()
            j = int(rng.choice(len(names), p=w))
            x = optimize_acquisition(gp, ARMS[names[j]], bounds, best, rng, kappa=2.0)
            y_new = float(f(x))
            reward = max(0.0, best - y_new)
            scale = max(1e-12, abs(best))
            gains[j] += reward / scale
            obs.add(x, y_new)
        return obs


class RestartEI:
    """EI + plateau-triggered hyper-parameter restart (no random injection)."""

    name = "C_restart_ei"

    def __init__(self, patience: int | None = None):
        self.patience = patience

    def minimize(self, f, bounds, n_evals, rng, n_init=5):
        obs = Observed()
        for x in bounds.sample(n_init, rng):
            obs.add(x, float(f(x)))
        patience = self.patience or max(4, n_evals // 5)
        best = obs.best()[1]
        since = 0
        prev = None
        boost = 0
        for _ in range(n_evals - n_init):
            gp = NumpyGP(kernel="rbf").fit(obs.X(), obs.Y(), init_theta=prev)
            prev = gp._theta
            if boost > 0:
                fn = lower_confidence_bound
                kappa = 3.0
                boost -= 1
            else:
                fn = expected_improvement
                kappa = 2.0
            x = optimize_acquisition(gp, fn, bounds, best, rng, kappa=kappa)
            y_new = float(f(x))
            obs.add(x, y_new)
            if y_new < best - 1e-9:
                best, since = y_new, 0
            else:
                since += 1
            if since >= patience:
                prev = None          # re-fit hyper-parameters from scratch
                boost = 3            # short exploration burst
                since = 0
        return obs


def main() -> int:
    n_evals, seeds = 30, list(range(5))
    specs = demo_benchmark_functions()
    variants = [PlainEI(), Hedge(), Hedge(eta=0.5), RestartEI()]
    print(f"{'function':<16}{'opt':>10} " + "".join(f"{v.name:>16}" for v in variants))
    totals = {v.name: [] for v in variants}
    for spec in specs:
        row = []
        for v in variants:
            srs = []
            for s in seeds:
                obs = v.minimize(spec, spec.bounds, n_evals, np.random.default_rng(s), n_init=5)
                srs.append(obs.best()[1] - spec.optimum)
            m = float(np.mean(srs))
            totals[v.name].append(m)
            row.append(f"{m:>16.4f}")
        print(f"{spec.name:<16}{spec.optimum:>10.4f} " + "".join(row))
    print(f"{'MEAN-SR':<16}{'':>10} " + "".join(f"{np.mean(totals[v.name]):>16.4f}" for v in variants))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
