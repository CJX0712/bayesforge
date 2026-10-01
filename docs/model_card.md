# Model Card — BayesForge

> Version 1.0.0 · Author **晨星** · Released 2026-10

---

## Model details

| Field | Value |
|---|---|
| **Type** | Sequential model-based (Bayesian) global optimizer |
| **Surrogate** | Exact Gaussian process regression, zero mean, RBF or Matérn-5/2 with ARD |
| **Hyper-parameter fitting** | Marginal-likelihood maximisation via bounded L-BFGS-B with an analytic gradient |
| **Acquisition** | Expected improvement / probability of improvement / lower confidence bound / normalized portfolio |
| **Search space** | Axis-aligned box `[lo, hi]^d`, continuous, all dimensions real-valued |
| **Objective** | Scalar `float`, minimised, treated as a black box |
| **Dependencies** | `numpy`, `scipy` only |
| **Determinism** | Bit-level reproducible given `(seed, n_evals, n_init, method set, function set)` |

---

## Intended use

**In scope**

- Tuning hyper-parameters where one evaluation costs seconds to hours.
- Calibrating simulations and physical experiments with a limited trial budget.
- Design-of-experiments and parameter selection for A/B tests.
- Research: a small, readable, fully-verifiable reference implementation of GP-BO.

**Out of scope**

- Cheap objectives where thousands of evaluations are affordable — use an
  evolutionary method; the `O(n³)` GP refit makes BO the wrong tool.
- Problems with integer/categorical variables, or constraints beyond a box.
- Multi-objective / Pareto-front problems.
- Objectives where safety guarantees matter during search (no constraint model).

---

## Training data

There is **no pre-trained component**. The GP is fit online to the evaluations
collected during each run. The repository ships 10 **analytic** benchmark
functions (sphere, Rosenbrock, Ackley, Levy, Branin, six-hump camel,
Goldstein–Price, Michalewicz, Hartmann-3, Hartmann-6) used solely for
evaluation; each ships with a literature value for its global optimum, and each
of those values is re-verified against the textbook argmin in the test suite.

No personal, proprietary, or user data is involved.

---

## Performance

Benchmark: 4 functions (sphere3, branin, six_hump_camel, hartmann3) × 3 seeds,
`n_evals = 35`, `n_init = 5`, measured as **mean simple regret**
(best found − known optimum, lower is better):

| Method | Mean SR | Std | vs random |
|---|---:|---:|---:|
| random search | 1.4219 | 1.6730 | — |
| grid search | 1.1458 | 0.6641 | −19.4 % |
| differential evolution | 0.9993 | 1.3588 | −29.7 % |
| **BayesFuse** | **0.0537** | **0.1666** | **−96.2 %** |

Determinism: re-running the identical configuration yields a max absolute
difference in simple regret of `0.00e+00`, and two **separate processes**
produce a bit-identical `benchmark.json` over all 48 result rows.

Reproduce with `python examples/run_demo.py` (≈25 s wall on a modern laptop CPU).

Note on differential evolution: at a strict 35-evaluation budget DE is weak, and
that is genuine — an earlier version appeared to score 0.15 only because it
charged ~300 real evaluations to the budget while reporting 5 (see Limitations
below). The budget accounting is now exact for every optimizer.

---

## Limitations and failure modes

| Limitation | Consequence | Mitigation in this release |
|---|---|---|
| Cubic cost per iteration | Slow once `n` exceeds a few hundred | Warm-started `θ`; keep `n_evals` in the tens-to-low-hundreds regime |
| GP misspecification | A too-smooth kernel under-models rugged landscapes | Matérn-5/2 available; per-iteration kernel selection by marginal likelihood available |
| Noise mis-estimation | A free noise hyper-parameter can absorb real signal and flatten the posterior | `log σ_n` is upper-bounded (`noise_bound_hi`, default −5) to preserve interpolation; raise it for genuinely noisy objectives |
| Dimensionality | Surrogate quality degrades past ~10 dimensions | Documented bound; no claim beyond it |
| Local stagnation | EI can stall in a false basin | Plateau-triggered hyper-parameter restart / exploration burst available; note that on small budgets extra exploration **hurts** — see below |
| Numerics | Unbounded `exp(θ)` overflows to `NaN` | All `θ` bounded; `RuntimeWarning` escalated to a test error |

### A note on adaptive machinery

An internal sweep over acquisition strategies (`examples/_sweep_acq.py`, 4
functions × 5 seeds, `n_evals = 30`, mean simple regret) produced:

| Strategy | Mean SR |
|---|---:|
| plain EI | **0.0334** |
| GP-Hedge portfolio (EI/PI/LCB) | 0.1290 |
| EI + plateau exploration burst | 0.1300 |

i.e. **at small budgets, added exploration measurably hurts.** The default
configuration therefore keeps exploration minimal; the mechanisms remain
available and are documented, but they are not silently enabled. Any claim that
BayesFuse beats its own ablations must be stated at a specific budget.

---

## Ethical considerations

BayesForge optimises whatever scalar it is given. It performs no value
judgement on that objective. If it is used to tune a system that affects people
(fairness-relevant models, pricing, allocation, content ranking), the choice of
objective, the allowed parameter ranges, and any constraints encoded as box
bounds carry the entire ethical weight — and are the responsibility of whoever
configures them. The optimizer will exploit the objective exactly as specified,
including unintended loopholes in a poorly specified reward.

No data collection, no network calls, no telemetry.

---

## Verification summary

| Property | Test | Cross-check |
|---|---|---|
| Analytic ∂NLL/∂θ correctness | `test_surrogate.py` | central finite differences, tol 1e-5 |
| GP interpolation | `test_surrogate.py` | `max abs error < 1e-2` on training targets |
| No numerical overflow | `pytest.ini` | `RuntimeWarning` → error |
| EI correctness | `test_acquisition.py` | 400 000-draw Monte-Carlo, tol 5e-3 |
| Acquisition maximiser quality | `test_acquisition.py` | beats best of 20 000 random points |
| Budget exactness | `test_optimizers.py` | `len(obs) == n_evals` for every method |
| Feasibility | `test_optimizers.py` | all `x ∈ [lo, hi]` |
| Reproducibility | `test_optimizers.py` | identical trace, fresh *and* reused instance |
| End-to-end determinism | `cli check`, `test_pipeline_and_cli.py` | `ΔSR < 1e-9` across two runs |
| Global optima correctness | `test_functions.py` | re-evaluated at textbook argmin, all 10 functions |
