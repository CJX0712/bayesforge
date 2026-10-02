# Changelog

All notable changes to BayesForge are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Author: **晨星**

---

## [0.1.1] — 2026-10-02

### Fixed

- **Quality gates were platform-dependent.** `numpy.random.Generator` streams
  are *not* guaranteed stable across versions or BLAS backends (NEP 19 only
  covers the legacy `RandomState`), and the BO loop amplifies tiny numeric
  differences chaotically. On the first CI run the `six_hump_camel` gate
  flipped (BO −0.766 vs random −0.869) purely because CI resolved different
  package versions than the local locked environment. The gates now use a
  60-evaluation budget (structurally ~70× regret margin on that function) plus
  a 5 % relative tolerance, so they no longer depend on the exact numeric path.
- **CI no longer resolves the scientific stack implicitly.** numpy / scipy /
  pytest / ruff are pinned to the versions in `requirements.lock.txt`
  (`2.5.3` / `1.18.1` / `9.1.1` / `0.16.9`) and the package is installed with
  `--no-deps`, so CI validates the locked environment instead of whatever the
  index serves that day.
- **Matrix converged to the locked environment.** numpy 2.5.3 requires Python
  ≥ 3.12, so the matrix is now 3.12 + 3.13 (`requires-python >= 3.12`); the
  3.10/3.11 jobs could not install the pinned stack.
- **Packaging metadata:** version was `1.0.0` while the tag said `v0.1.0`.

### Verification

- Local: 185 tests green, `ruff` clean, 60-evaluation gates pass on all four
  demo functions (six-hump camel: BO regret 0.0022 vs random 0.1623).
- CI: run `36936492767` — py3.12 and py3.13 both green (lint + tests +
  determinism check + demo smoke).

---

## [0.1.0] — 2026-10-02

### Added

- **Exact Gaussian-process surrogate** (`surrogate/gp.py`) in pure NumPy:
  RBF and Matérn-5/2 kernels with ARD lengthscales, Cholesky inference, and an
  **analytic negative-log-likelihood gradient** supplied to bounded L-BFGS-B.
- **Acquisition functions** (`acquisition/acq.py`): expected improvement,
  probability of improvement, lower confidence bound, and a batch-normalized
  EI/LCB portfolio, plus a screening + multistart box optimiser.
- **Optimizers** (`optimizer/optimizers.py`): random search, grid search,
  differential evolution, GP-BO, and the **BayesFuse** flagship. Optional SOTA
  backends (`scikit-optimize`, `bayesian-optimization`) are auto-detected.
- **10 analytic benchmarks with known global optima** (`data/functions.py`):
  sphere, Rosenbrock, Ackley, Levy, Branin, six-hump camel, Goldstein–Price,
  Michalewicz, Hartmann-3, Hartmann-6.
- **Benchmark pipeline** (`pipeline/benchmark.py`) with `[base_seed, method,
  function, seed]` deterministic seeding, and CLI commands `run` / `table` /
  `check`.
- Inner hyper-parameter sweep for GPBO (`hpo/tune.py`).
- Full invariant test suite (185 tests) and CI workflow on Python 3.12–3.13.

### Fixed

- **Analytic NLL gradient had the wrong sign.** The GPML identity requires
  `−½ tr((ααᵀ − K⁻¹) ∂K/∂θ)`; the original implementation returned its negative,
  so L-BFGS-B moved *away* from the optimum and the fit returned its initial
  `θ` unchanged. (Caught by comparing against central finite differences.)
- **Chain-rule error in the lengthscale gradient.** `ell = exp(θ) + 1e-3`
  means `∂ell/∂θ = exp(θ)`, not `ell`. The `1e-3` term was dropped, leaving a
  ~1e-3 relative error that survived after the sign was corrected.
- **`predict()` standardised inputs against the wrong mean.** It used
  `Xs.mean(0)` (identically zero, since the training set is already centred)
  instead of the stored training mean, shifting every prediction.
- **Unbounded `θ` overflowed `exp()`.** `RuntimeWarning: overflow in exp`
  produced `NaN` gradients and silently garbage fits. All hyper-parameters are
  now bounded, and `RuntimeWarning` is escalated to a test error.
- **Free noise variance destroyed the posterior.** With no upper bound, the
  marginal likelihood explained a non-stationary landscape as pure noise: the
  posterior mean collapsed toward a constant while the variance inflated, so
  EI chased phantom uncertainty. `log σ_n` is now upper-bounded
  (`noise_bound_hi`, default −5), preserving interpolation on noise-free
  objectives. This alone moved mean simple regret on six-hump camel from
  −0.68 to −0.99.
- **Warm-start `θ` leaked across runs.** Reusing one optimizer object produced
  different trajectories than constructing a fresh one (ΔSR ≈ 1.9e-2). The
  warm start is now reset at the top of every `minimize()` call.
- **BayesFuse over-spent its budget.** The plateau restart injected extra
  evaluations outside the accounting loop, giving the flagship a hidden
  advantage. The loop is now budget-exact: restarts are charged against the
  same budget as every other candidate.
- **Differential evolution did not respect the budget.** A fixed `popsize=15`
  spent the entire budget on the initial population alone, and the per-
  generation callback recorded 1 observation per generation (≈60 evaluations),
  so the reported trajectory was ~60× shorter than reality. Population size now
  scales with the budget and every real objective call is recorded.
- **`FunctionSpec` noise used the global NumPy stream**, breaking determinism.
  Each spec now owns a generator seeded from its name.
- **`hpo/tune.py` imported `benchmark` from the wrong package** (`hpo` instead
  of `pipeline`); `resolve_backends("auto")` returned a *function* instead of a
  list.

### Changed

- The demo and CLI default to a 4-function set at `n_evals = 35` to stay inside
  a 60-second CPU budget; the extended 6-function set is available via
  `--full`.
- GP hyper-parameter fitting warm-starts `θ` from the previous BO iteration and
  uses two starts instead of four, roughly halving wall-clock.

### Known performance characteristics

An internal sweep (`examples/_sweep_acq.py`) showed that **adaptive exploration
hurts at small budgets**: plain EI scored mean SR 0.0334 versus 0.1290 for a
GP-Hedge portfolio and 0.1300 for EI with a plateau exploration burst. The
adaptive mechanisms remain available and documented but are not silently
enabled.
