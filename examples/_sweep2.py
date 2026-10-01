"""Internal experiment #2: what actually improves on plain GP-EI?

Tests, at two budgets:
  A  plain EI (control)
  D  EI + Sobol (low-discrepancy) initial design
  E  EI + per-iteration kernel selection by marginal likelihood (RBF vs M52)
  F  D + E combined
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.stats import qmc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bayesforge.acquisition.acq import expected_improvement, optimize_acquisition  # noqa: E402
from bayesforge.core.types import Observed  # noqa: E402
from bayesforge.data.registry import demo_benchmark_functions  # noqa: E402
from bayesforge.surrogate.gp import NumpyGP  # noqa: E402


def _sobol(bounds, n, rng):
    """Scrambled Sobol sample mapped into the box (deterministic given seed)."""
    d = bounds.dim
    m = int(np.ceil(np.log2(max(n, 2))))
    eng = qmc.Sobol(d=d, scramble=True, seed=int(rng.integers(1, 2**31 - 1)))
    u = eng.random_base2(m)[:n]
    return bounds.lo[None, :] + u * (bounds.hi - bounds.lo)[None, :]


class Base:
    name = "base"
    sobol = False
    kernel_select = False

    def _init(self, bounds, n_init, rng):
        return _sobol(bounds, n_init, rng) if self.sobol else bounds.sample(n_init, rng)

    def _fit(self, X, y, prev):
        if not self.kernel_select:
            return NumpyGP(kernel="rbf").fit(X, y, init_theta=prev)
        best = None
        for k in ("rbf", "matern52"):
            g = NumpyGP(kernel=k).fit(X, y, init_theta=(prev if k == "rbf" else None))
            if best is None or g._nll_value < best._nll_value:
                best = g
        return best

    def minimize(self, f, bounds, n_evals, rng, n_init=5):
        obs = Observed()
        for x in self._init(bounds, n_init, rng):
            obs.add(x, float(f(x)))
        prev = None
        for _ in range(n_evals - n_init):
            gp = self._fit(obs.X(), obs.Y(), prev)
            prev = gp._theta if gp.kernel == "rbf" else None
            x = optimize_acquisition(gp, expected_improvement, bounds, obs.best()[1], rng)
            obs.add(x, float(f(x)))
        return obs


class A(Base):
    name = "A_plain_ei"


class D(Base):
    name = "D_sobol_init"
    sobol = True


class E(Base):
    name = "E_kernel_select"
    kernel_select = True


class F(Base):
    name = "F_sobol_kernel"
    sobol = True
    kernel_select = True


def main() -> int:
    seeds = list(range(5))
    specs = demo_benchmark_functions()
    variants = [A(), D(), E(), F()]
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
