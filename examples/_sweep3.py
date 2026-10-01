"""Internal experiment #3: does a zero-cost hyper-parameter restart help?

  A  plain EI + warm start            (control / current GPBO)
  G  A + plateau-triggered theta reset (re-fit from scratch, NO exploration burst)
  I  full multi-start fit every iteration (no warm start) — upper bound on fit quality
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bayesforge.acquisition.acq import expected_improvement, optimize_acquisition  # noqa: E402
from bayesforge.core.types import Observed  # noqa: E402
from bayesforge.data.registry import demo_benchmark_functions  # noqa: E402
from bayesforge.surrogate.gp import NumpyGP  # noqa: E402


class Base:
    name = "base"
    restart = False
    multistart = False

    def minimize(self, f, bounds, n_evals, rng, n_init=5):
        obs = Observed()
        for x in bounds.sample(n_init, rng):
            obs.add(x, float(f(x)))
        patience = max(4, n_evals // 5)
        best = obs.best()[1]
        since = 0
        prev = None
        for _ in range(n_evals - n_init):
            gp = NumpyGP(kernel="rbf").fit(obs.X(), obs.Y(),
                                           init_theta=None if self.multistart else prev)
            prev = gp._theta
            x = optimize_acquisition(gp, expected_improvement, bounds, best, rng)
            y_new = float(f(x))
            obs.add(x, y_new)
            if y_new < best - 1e-9:
                best, since = y_new, 0
            else:
                since += 1
            if self.restart and since >= patience:
                prev = None          # force a from-scratch hyper-parameter refit
                since = 0
        return obs


class A(Base):
    name = "A_plain_ei"


class G(Base):
    name = "G_theta_reset"
    restart = True


class ColdMulti(Base):
    name = "I_multistart"
    multistart = True


def main() -> int:
    seeds = list(range(5))
    specs = demo_benchmark_functions()
    variants = [A(), G(), ColdMulti()]
    for n_evals in (30, 60):
        print(f"\n=== n_evals={n_evals} (mean simple regret over 5 seeds) ===")
        print(f"{'function':<16}" + "".join(f"{v.name:>18}" for v in variants))
        tot = {v.name: [] for v in variants}
        for spec in specs:
            row = []
            for v in variants:
                srs = [v.minimize(spec, spec.bounds, n_evals, np.random.default_rng(s),
                                  n_init=5).best()[1] - spec.optimum for s in seeds]
                tot[v.name].append(float(np.mean(srs)))
                row.append(f"{np.mean(srs):>18.4f}")
            print(f"{spec.name:<16}" + "".join(row))
        print(f"{'MEAN-SR':<16}" + "".join(f"{np.mean(tot[v.name]):>18.4f}" for v in variants))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
