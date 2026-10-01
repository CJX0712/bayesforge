# BayesForge — Architecture

> Author: **晨星** · Version 1.0.0 · Deterministic Bayesian optimization in pure NumPy

---

## 1. What this system is

BayesForge is a **black-box derivative-free optimizer** for expensive objectives:
given a box-bounded objective `f: [lo, hi]^d → ℝ` and a hard budget of `n_evals`
function evaluations, it returns the best input it can find. The core is an
**exact Gaussian-process surrogate** plus an **acquisition function** that trades
exploitation against exploration; the flagship `BayesFuse` adds an adaptive
acquisition portfolio and a plateau-triggered restart.

Everything is **deterministic**: two runs with the same seed produce
bit-identical trajectories. There is no wall-clock dependence, no `hash()`
ordering, and no hidden global RNG state.

---

## 2. Module map

```
bayesforge/
├── core/
│   ├── types.py        Bounds, Observed, FunctionSpec, BenchmarkResult
│   ├── config.py       Config dataclass + `BO_*` env overrides + validation
│   ├── errors.py       Error catalogue E100–E500 (single base class)
│   ├── seed.py         set_all(seed) global seeding helper
│   └── interfaces.py   Objective / Surrogate / Acquisition protocols
├── data/
│   ├── functions.py    10 analytic benchmarks with known global optima
│   └── registry.py     all_functions / demo_functions / demo_benchmark_functions
├── surrogate/
│   └── gp.py           Exact GP (RBF, Matérn-5/2), analytic NLL gradient
├── acquisition/
│   └── acq.py          EI / PI / LCB / portfolio + box-constrained maximiser
├── optimizer/
│   ├── optimizers.py   Random, Grid, DE, GPBO, BayesFuse (+ optional SOTA)
│   └── registry.py     core / sota / resolve_backends
├── hpo/
│   └── tune.py         Inner sweep over (kernel, acq) for GPBO
├── eval/
│   └── metrics.py      simple / cumulative regret, regret trace
├── pipeline/
│   └── benchmark.py    method × function × seed orchestration
└── cli.py              run / table / check
```

**Dependency direction is strictly one-way**: `core → data → surrogate →
acquisition → optimizer → pipeline → cli`. Nothing in `core` imports from
`optimizer`, so the surrogate and acquisition layers are independently testable
(and are — see §6).

---

## 3. The Bayesian optimization loop

```
1. Initial design   n_init points from a uniform sample of the box
2. repeat until budget exhausted:
     a. Fit GP posterior on all observations (X, y)
     b. Choose acquisition  a(x)  (EI / PI / LCB / portfolio / hedge)
     c. x* = argmax a(x) over the box   ← random screening + L-BFGS-B multistart
     d. Evaluate y = f(x*), append to observations
3. Return argmin over recorded y
```

### 3.1 Surrogate — exact GP

* Kernel: RBF `k = σ² exp(−½ r²)` or Matérn-5/2 `σ²(1+√5r+5r²/3)e^{−√5r}`,
  with **ARD** (one lengthscale per dimension).
* Inputs standardised per dimension, outputs standardised globally.
* Hyper-parameters `θ = [log ℓ₁…log ℓ_d, log σ, log σ_n]` fit by maximising the
  marginal likelihood (equivalently minimising the negative log-likelihood).

**NLL and its analytic gradient** (GPML identity):

```
NLL(θ)  = ½ yᵀK⁻¹y + log|K| + (n/2)log 2π
∂NLL/∂θ = ½ tr((ααᵀ − K⁻¹) ∂K/∂θ),   α = K⁻¹y
```

The gradient is supplied to L-BFGS-B as `jac=True`. This is pinned against
central finite differences in `tests/test_surrogate.py` (tolerance 1e-5).

**Two numerical decisions that matter** (both are regression-tested):

| Decision | Why |
|---|---|
| `θ` is **bounded** — `log ℓ ∈ [−5, 5]`, `log σ ∈ [−5, 5]`, `log σ_n ∈ [−16, noise_bound_hi]` | Without bounds, `exp(θ)` overflows to `inf`, the gradient becomes `NaN`, and the fit silently returns garbage. |
| `log σ_n` upper bound defaults to **−5** | If the noise variance is free, the marginal likelihood happily explains a non-stationary landscape as *pure noise*: the posterior mean collapses to a constant and the variance inflates, so EI is driven by phantom uncertainty and the search degenerates. Pinning the noise to a small value preserves the interpolation property that noise-free benchmarks require. |

Inference uses a Cholesky factorisation:

```
K = LLᵀ,  α = Lᵀ⁻¹L⁻¹y
μ(x*) = k*ᵀα                      (de-standardised)
σ²(x*) = σ² − ‖L⁻¹k*‖²            (latent variance, noise excluded)
```

The reported variance is the **latent function variance** — this is what EI, PI
and LCB are defined against.

### 3.2 Acquisition functions

All are written to be **maximised**; the objective is minimised, so a high score
means "promising".

| Name | Formula | Notes |
|---|---|---|
| EI | `s·(z·Φ(z) + φ(z))`, `z = (y_best − μ)/s` | Cross-checked against Monte-Carlo `E[max(y_best − y, 0)]` |
| PI | `Φ(z)` | Bounded in `[0, 1]` |
| LCB | `κ·s − μ` | Maximising this ≡ minimising `μ − κ·s` |
| portfolio | `max(norm(EI), norm(LCB))` | Batch min–max normalisation keeps heterogeneous scores comparable |

`optimize_acquisition` maximises on the box by **random screening
(1000 candidates) → top-k L-BFGS-B multistart → clip**. A brute-force invariant
test asserts the returned point scores at least as high as the best of 20 000
random candidates.

### 3.3 BayesFuse (flagship)

BayesFuse is **not** a bag of heuristics — every mechanism in it was selected by
an internal ablation, and several plausible-sounding ones were *rejected*
because they measured worse (see §8). Two mechanisms survived:

1. **Warm-started hyper-parameters.** θ from the previous iteration seeds the
   next fit, acting as a temporal prior across the search. This is worth more
   than a 2× reduction in simple regret versus refitting cold from multiple
   starts every iteration (0.0334 vs 0.0760 at `n_evals=30`).

2. **EI-collapse fallback.** The candidate batch is screened **once** per
   iteration; if the maximum EI over it has collapsed to zero — the GP is
   certain there is nothing left to gain — the step falls back to LCB instead
   of blindly stepping where EI is flat. Because it fires only when EI is
   genuinely zero, it costs nothing in normal operation and rescues stalled
   runs. Screening once and reusing the batch for whichever acquisition wins
   also avoids paying for two rounds of GP predictions.

BayesFuse obeys the budget invariant `len(observations) == n_evals` exactly —
there is no hidden exploration budget.

---

## 4. Baselines

| Optimizer | Role |
|---|---|
| `random_search` | Naive baseline; defines the 0% reference for `gap%_vs_rand` |
| `grid_search` | Classic uniform grid (degrades to a single sample for `d > 6`) |
| `differential_evolution` | Strong classic population method; population size scales with the budget and every real objective call is recorded so its budget is exact |
| `gp_bo` | Reference GP-BO with a single fixed acquisition |
| `bayesfuse` | Flagship |

Optional SOTA backends (`scikit-optimize`, `bayesian-optimization`) are detected
at import time and added by `--all`; **none are required** — the NumPy path is
always available.

---

## 5. Determinism model

Every optimizer receives a `numpy.random.Generator` seeded from the tuple

```
[base_seed, method_index, function_index, seed_index]
```

so a run is a pure function of `(base_seed, n_evals, n_init, method set,
function set)`. Three properties make this hold in practice:

1. **No global RNG** — even `FunctionSpec`'s optional observation noise carries
   its own generator, seeded from the function name.
2. **No optimizer-instance state across runs** — the GP warm-start `θ` is reset
   at the top of every `minimize()` call, so reusing one optimizer object is
   bit-identical to constructing a fresh one. *(This was a real bug: the warm
   start leaked across benchmark runs and changed trajectories by ~2e-2 in
   simple regret.)*
3. **No wall-clock or `hash()` ordering** in any decision path.

`python -m bayesforge.cli check` re-runs the benchmark twice and asserts the max
absolute difference in simple regret is `< 1e-9`.

---

## 6. Verification strategy

Each layer has an independently checkable invariant — not just "it runs":

| Layer | Invariant | Cross-check |
|---|---|---|
| Functions | declared `optimum` is correct | re-evaluate at textbook argmin |
| GP gradient | analytic ∂NLL/∂θ | central finite differences |
| GP fit | noise-free GP interpolates | `max|μ(X) − y| < 1e-2` |
| GP stability | `θ` finite and in bounds, no `NaN` | bounded L-BFGS-B |
| EI | analytic EI | Monte-Carlo over 400 000 draws |
| Acquisition opt. | found point beats screening | best of 20 000 random candidates |
| Optimizers | budget exactness | `len(obs) == n_evals` |
| Optimizers | feasibility | every `x ∈ [lo, hi]` |
| Optimizers | reproducibility | same seed ⇒ identical `y` trace |
| Pipeline | end-to-end determinism | run twice, `ΔSR < 1e-9` |

---

## 7. Complexity

Let `n` be the number of observations and `d` the dimension.

| Operation | Cost | Notes |
|---|---|---|
| Kernel build | `O(n²d)` | vectorised squared-distance matrix |
| Cholesky | `O(n³)` | recomputed per NLL evaluation |
| NLL + gradient | `O(n³)` | dominated by `K⁻¹` |
| Hyper-parameter fit | `O(R·I·n³)` | `R` restarts × `I` L-BFGS-B iterations |
| Posterior predict | `O(n²)` per batch | reuses the cached Cholesky |
| Acquisition screening | `O(m·n²)` | `m = 1000` candidates |

BO is therefore cubic in the evaluation budget — appropriate for *expensive*
objectives (the regime it targets), and the reason the default demo budget is
`n_evals = 35`. Warm-starting `θ` from the previous iteration cuts `R` from 3 to
2 and roughly halves wall-clock *while also improving search quality*.

---

## 8. Why BayesFuse looks the way it does (ablation record)

The flagship design was chosen by measurement, not taste. Every candidate below
was run on 4 functions × 5 seeds (sphere3, branin, six_hump_camel, hartmann3),
reported as **mean simple regret**.

**Round 1 — acquisition strategy** (`examples/_sweep_acq.py`, `n_evals=30`):

| Strategy | Mean SR |
|---|---:|
| plain EI | **0.0334** |
| GP-Hedge portfolio over EI / PI / LCB (η=0.2 and 0.5) | 0.1290 |
| EI + plateau exploration burst (κ=3) | 0.1300 |

**Round 2 — search-space tricks** (`examples/_sweep2.py`):

| Variant | n=30 | n=60 |
|---|---:|---:|
| plain EI (control) | **0.0334** | **0.0006** |
| + scrambled Sobol initial design | 0.0666 | 0.0016 |
| + per-iteration RBF/Matérn selection by marginal likelihood | 0.0434 | 0.0014 |
| + both | 0.0805 | 0.0022 |

**Round 3 — fitting schedule** (`examples/_sweep3.py`):

| Variant | n=30 | n=60 |
|---|---:|---:|
| warm-started θ (control) | **0.0334** | **0.0006** |
| + plateau-triggered θ reset | 0.0369 | 0.0011 |
| cold multi-start every iteration | 0.0760 | 0.0019 |

**Conclusion.** At realistic budgets every form of "extra exploration" measurably
hurts, including two well-published ones (GP-Hedge, low-discrepancy initial
designs). The only changes that helped were **numerical**: fixing the noise
bound and warm-starting the fit. So BayesFuse ships the plain-EI configuration
that won, plus the one fallback that only fires when EI is identically zero.

These scripts are kept in `examples/` precisely so the claim can be re-run and
falsified.

---

## 9. Bugs found and fixed during development

Recorded because several are subtle and each materially changed results:

| Bug | Symptom | Fix |
|---|---|---|
| Analytic ∂NLL/∂θ had the **wrong sign** | L-BFGS-B moved away from the optimum; fit returned its initial θ | `½ tr((ααᵀ − K⁻¹) ∂K/∂θ)` |
| Chain rule dropped the `1e-3` in `ell = exp(θ)+1e-3` | gradient accurate only to ~1e-3 | chain factor is `exp(θ)`, not `ell` |
| `predict()` standardised against `Xs.mean(0)` (≡ 0) | all predictions offset | use the stored training mean |
| Unbounded θ overflowed `exp()` | `NaN` gradients, silent garbage fits | bounded L-BFGS-B |
| **Free noise variance** swallowed the signal | posterior mean collapsed to a constant, variance inflated, EI chased phantom uncertainty (six-hump camel SR −0.68) | upper-bound `log σ_n` (default −5) → SR −0.99 |
| Warm-start θ leaked across runs | ΔSR ≈ 1.9e-2 when reusing an optimizer object | reset `_prev_theta` per `minimize()` |
| BayesFuse spent evaluations outside the loop | flagship got a hidden budget | budget-exact loop |
| DE charged ~300 real evals to a 35-eval budget | DE appeared 6× stronger than it is | record every real call; population scales with budget |
| `FunctionSpec` noise used the global NumPy stream | non-deterministic | per-spec generator seeded from the name |
