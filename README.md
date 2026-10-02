# BayesForge

**Deterministic Bayesian optimization with an exact NumPy Gaussian process and the BayesFuse adaptive acquisition ensemble.**

Author: **晨星**

![CI](https://github.com/CJX0712/bayesforge/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Deps](https://img.shields.io/badge/deps-numpy%20%2B%20scipy-orange)
![Determinism](https://img.shields.io/badge/determinism-%CE%94%3C1e--9-brightgreen)

---

## Why this exists

Bayesian optimization (BO) is the standard tool for optimizing objectives that
are expensive, noisy and derivative-free — hyper-parameter tuning, experiment
design, simulation calibration, A/B parameter selection. Most BO libraries are
heavy, pull in a large dependency tree, and are only *approximately*
reproducible.

BayesForge is the opposite: **two dependencies** (`numpy`, `scipy`), an
**exact GP written from first principles**, and **bit-level determinism** —
the same seed twice produces the same trajectory to the last bit.

---

## Results

`python examples/run_demo.py` (4 functions × 4 methods × 3 seeds, `n_evals=35`,
wall ≈ 25 s):

| method | mean simple regret | std | gap vs random |
|---|---:|---:|---:|
| `random_search` | 1.4219 | 1.6730 | — |
| `grid_search` | 1.1458 | 0.6641 | 19.4 % |
| `differential_evolution` | 0.9993 | 1.3588 | 29.7 % |
| **`bayesfuse`** | **0.0537** | **0.1666** | **96.2 %** |

*Simple regret = best value found − known global optimum (lower is better).*

Determinism self-check inside the demo: `max |ΔSR| = 0.00e+00`. Running the demo
twice in **separate processes** produces a `benchmark.json` that is
**bit-identical** across all 48 result rows (ignoring wall-clock timings).

Against the strongest classic baseline, `bayesfuse` is **18.6× better** than
differential evolution at the same 35-evaluation budget.

---

## Install

```bash
pip install -e .            # core (numpy + scipy)
pip install -e ".[sota]"    # + optional scikit-optimize / bayesian-optimization backends
pip install -e ".[dev]"     # + pytest, ruff
```

## Usage

```bash
# run a benchmark, write benchmark.json
python -m bayesforge.cli run --seed 42 --n-evals 40 --seeds 3

# pretty-print an existing result
python -m bayesforge.cli table --json benchmark.json

# determinism self-check (runs the benchmark twice, compares)
python -m bayesforge.cli check
```

As a library:

```python
import numpy as np
from bayesforge.data.registry import all_functions
from bayesforge.optimizer.optimizers import BayesFuse

spec = [s for s in all_functions() if s.name == "branin"][0]
obs = BayesFuse().minimize(spec, spec.bounds, n_evals=40,
                           rng=np.random.default_rng(0), n_init=5)
x_best, y_best = obs.best()      # branin optimum = 0.397887
```

Plug in your own objective — anything callable `f(x) -> float`:

```python
from bayesforge.core.types import Bounds, FunctionSpec
import numpy as np

def my_objective(x):            # expensive real-world evaluation
    return float(np.sum((x - 0.3) ** 2) + 0.1 * np.sin(10 * x[0]))

spec = FunctionSpec("mine", my_objective,
                    Bounds(lo=np.zeros(3), hi=np.ones(3)),
                    optimum=float("nan"), dim=3)   # optimum unknown -> use 0.0 for a regret proxy
```

---

## What's inside

| Component | Detail |
|---|---|
| **Surrogate** | Exact GP, RBF / Matérn-5/2 with ARD lengthscales, Cholesky inference, **analytic NLL gradient** fed to L-BFGS-B, bounded `θ`, warm-started across BO iterations |
| **Acquisitions** | Expected improvement, probability of improvement, lower confidence bound, and a normalized EI/LCB portfolio |
| **Acquisition optimiser** | 1000-point random screening → top-k L-BFGS-B multistart → box clip |
| **Optimizers** | random search, grid search, differential evolution, GP-BO, **BayesFuse** |
| **Benchmarks** | 10 analytic functions with *known* global optima: sphere, Rosenbrock, Ackley, Levy, Branin, six-hump camel, Goldstein–Price, Michalewicz, Hartmann-3, Hartmann-6 |
| **Metrics** | simple regret, cumulative regret, monotone regret trace |
| **Determinism** | seed tuple `[base_seed, method, function, seed]`; no global RNG, no wall-clock, no instance-state leakage |

---

## Determinism, concretely

```bash
$ python -m bayesforge.cli check --n-evals 16
determinism check: max |SR diff| over 60 runs = 0.00e+00
PASS
```

Three design rules make this hold (all regression-tested):

1. Every optimizer gets a `Generator` derived from an explicit seed tuple.
2. The GP warm-start `θ` is reset at the top of each `minimize()` call — reusing
   an optimizer object is bit-identical to constructing a new one.
3. Even optional observation noise uses a per-`FunctionSpec` generator, not the
   global NumPy stream.

---

## Tests

```bash
pytest -q
```

The suite pins **independently checkable invariants**, not just "it runs":

- analytic ∂NLL/∂θ vs central finite differences (tolerance 1e-5);
- noise-free GP interpolates its own training targets;
- analytic EI vs a 400 000-draw Monte-Carlo estimate;
- the acquisition optimiser beats the best of 20 000 random candidates;
- every optimizer spends **exactly** `n_evals` evaluations and never leaves the box;
- same seed ⇒ identical trace (fresh instance *and* reused instance);
- declared global optima re-verified at the textbook argmin for all 10 functions.

`pytest.ini` promotes `RuntimeWarning` to an error, so numerical overflow in the
GP is a hard failure rather than a silent NaN.

---

## Layout

```
bayesforge/{core,data,surrogate,acquisition,optimizer,hpo,eval,pipeline}
tests/                      invariant + determinism + CLI smoke tests
examples/run_demo.py        timed end-to-end demo → benchmark.json
docs/architecture.md        design, invariants, complexity
docs/model_card.md          intended use, limitations, verification
```

---

## Limitations

- **Cubic in the budget.** Each BO step refits the GP: `O(n³)`. Designed for
  *expensive* objectives (tens to a few hundred evaluations), not for cheap ones
  where evolutionary methods win outright.
- **Box constraints only.** No linear/nonlinear constraint handling.
- **Single-objective.** No multi-objective or constrained Pareto support.
- **Modest dimension.** Reliable to roughly `d ≤ 10`; beyond that the GP
  surrogate degrades and the initial design dominates.
- **Sequential.** No batch/parallel candidate generation.

See [`docs/model_card.md`](docs/model_card.md) for the full statement.

## License

MIT — see [LICENSE](LICENSE).
