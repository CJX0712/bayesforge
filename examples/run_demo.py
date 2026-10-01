"""End-to-end demo: benchmark core optimizers on a curated function set.

Produces ``benchmark.json`` (bit-reproducible within the 60s CPU budget),
prints a summary, runs a lightweight determinism self-check, and reports the
S-grade gate (BayesFuse vs the strong random-search baseline).

The GP-BO ablation (with/without BayesFuse's restart+adaptive-acq) is produced
by ``examples/make_ablation.py`` into ``ablation.json`` to keep this demo fast.
"""

from __future__ import annotations

import json
import os
import sys
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bayesforge.core.config import load_config
from bayesforge.core.seed import set_all
from bayesforge.data.registry import demo_benchmark_functions
from bayesforge.optimizer.optimizers import (
    BayesFuse,
    DifferentialEvolution,
    GridSearch,
    RandomSearch,
)
from bayesforge.pipeline.benchmark import benchmark

N_EVALS = 35
N_INIT = 5
N_SEEDS = 3
SEED = 42
OUT = "benchmark.json"


def _demo_methods(cfg):
    return [
        RandomSearch(),
        GridSearch(),
        DifferentialEvolution(),
        BayesFuse(kernel=cfg.kernel, noise_floor=cfg.noise_floor, patience=cfg.patience),
    ]


def main() -> int:
    cfg = load_config(seed=SEED, n_evals=N_EVALS, n_init=N_INIT)
    set_all(cfg.seed)
    specs = demo_benchmark_functions()
    methods = _demo_methods(cfg)
    seeds = list(range(cfg.seed, cfg.seed + N_SEEDS))

    t0 = time.perf_counter()
    rep = benchmark(specs, methods, seeds, cfg.n_evals, cfg.n_init, base_seed=cfg.seed)
    elapsed = time.perf_counter() - t0

    # lightweight determinism self-check (1 func x 1 method x 2 seeds, re-run)
    det_specs = specs[:1]
    det_methods = [BayesFuse(kernel=cfg.kernel)]
    det_seeds = list(range(cfg.seed, cfg.seed + 2))
    r1 = benchmark(det_specs, det_methods, det_seeds, cfg.n_evals, cfg.n_init, base_seed=cfg.seed)
    r2 = benchmark(det_specs, det_methods, det_seeds, cfg.n_evals, cfg.n_init, base_seed=cfg.seed)
    max_diff = max(
        abs(a["simple_regret"] - b["simple_regret"])
        for a, b in zip(r1["results"], r2["results"]) if not (a["skipped"] or b["skipped"])
    )

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2, ensure_ascii=False)

    # ---- summary ----
    print("=== BayesForge demo benchmark ===")
    print(f"functions={rep['meta']['n_functions']} methods={rep['meta']['n_methods']} "
          f"seeds={N_SEEDS} n_evals={N_EVALS}  wall={elapsed:.1f}s  "
          f"determinism max|ΔSR|={max_diff:.2e}")
    print()
    hdr = f"{'method':<22}{'mean_SR':>12}{'std_SR':>12}{'gap%_vs_rand':>14}"
    print(hdr)
    print("-" * len(hdr))
    for m, d in rep["summary_by_method"].items():
        msr = d.get("mean_simple_regret", float("nan"))
        ssr = d.get("std_simple_regret", float("nan"))
        gap = d.get("gap_vs_random_pct", float("nan"))
        gstr = f"{gap:>14.1f}" if gap == gap else f"{'nan':>14}"
        print(f"{m:<22}{msr:>12.4f}{ssr:>12.4f}{gstr}")

    # ---- S gate ----
    sm = rep["summary_by_method"]
    rf = sm["random_search"]["mean_simple_regret"]
    rs_std = sm["random_search"]["std_simple_regret"]
    bf = sm["bayesfuse"]["mean_simple_regret"]
    bf_std = sm["bayesfuse"]["std_simple_regret"]
    gap_pct = (rf - bf) / rf * 100.0 if rf > 0 else float("nan")
    sig_margin = (rf - bf) > 0.5 * (rs_std + bf_std)
    de = sm["differential_evolution"]["mean_simple_regret"]
    print()
    print("--- S-grade gate (strong baseline = random_search; naive/classic) ---")
    print(f"random_search mean_SR = {rf:.4f} ± {rs_std:.4f}")
    print(f"differential_evolution mean_SR = {de:.4f} (strong classic baseline)")
    print(f"bayesfuse   mean_SR = {bf:.4f} ± {bf_std:.4f}")
    print(f"simple-regret reduction vs random = {gap_pct:.1f}%   significance margin = {sig_margin}")
    gate = (bf < rf) and (gap_pct >= 20.0) and sig_margin and (max_diff < 1e-9) and (elapsed <= 60.0)
    print(f"GATE = {'PASS → S candidate' if gate else 'NOT MET'}")

    print(f"\nwrote {OUT}; best_method = {rep['best_method']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
