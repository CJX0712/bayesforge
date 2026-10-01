"""Benchmark orchestration: run every optimizer on every function × seed.

Determinism: every optimizer gets a seed derived from
``[base_seed, method_index, func_index, seed_index]`` so two runs with the same
``base_seed`` are bit-identical (no Python ``hash()`` / no wall-clock).
"""

from __future__ import annotations

import time
from typing import Dict, List

import numpy as np

from ..core.types import BenchmarkResult, FunctionSpec
from ..eval.metrics import cumulative_regret, simple_regret


def run_one(method, func: FunctionSpec, n_evals: int, n_init: int, base_seed: int,
            m_idx: int, f_idx: int, s_idx: int) -> BenchmarkResult:
    rng = np.random.default_rng([int(base_seed), int(m_idx), int(f_idx), int(s_idx)])
    t0 = time.perf_counter()
    try:
        obs = method.minimize(func, func.bounds, n_evals, rng, n_init=n_init)
        elapsed = time.perf_counter() - t0
        sr = simple_regret(obs, func.optimum)
        cr = cumulative_regret(obs, func.optimum)
        best_y = float(obs.best()[1])
        return BenchmarkResult(
            method=method.name, func=func.name, dim=func.dim, n_evals=len(obs),
            best_y=best_y, true_opt=func.optimum, simple_regret=sr,
            cumulative_regret=cr, elapsed_sec=elapsed,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return BenchmarkResult(
            method=method.name, func=func.name, dim=func.dim, n_evals=0,
            best_y=float("nan"), true_opt=func.optimum, simple_regret=float("nan"),
            cumulative_regret=float("nan"), elapsed_sec=time.perf_counter() - t0,
            skipped=True, note=f"error: {exc}",
        )


def benchmark(specs: List[FunctionSpec], methods, seeds: List[int],
              n_evals: int, n_init: int, base_seed: int = 42) -> Dict:
    results: List[BenchmarkResult] = []
    for mi, method in enumerate(methods):
        for fi, func in enumerate(specs):
            for si, sd in enumerate(seeds):
                results.append(run_one(method, func, n_evals, n_init, base_seed, mi, fi, si))

    # aggregates
    by_method_func: Dict = {}
    for r in results:
        by_method_func.setdefault((r.method, r.func), []).append(r.simple_regret)

    summary_by_method: Dict[str, Dict] = {}
    for method in [m.name for m in methods]:
        vals = [r.simple_regret for r in results if r.method == method and not r.skipped]
        if vals:
            summary_by_method[method] = {
                "mean_simple_regret": float(np.mean(vals)),
                "std_simple_regret": float(np.std(vals)),
            }
        else:
            summary_by_method[method] = {"mean_simple_regret": float("nan"), "std_simple_regret": float("nan")}

    # gap vs random_search
    rand_mean = summary_by_method.get("random_search", {}).get("mean_simple_regret", float("nan"))
    for method, d in summary_by_method.items():
        m = d["mean_simple_regret"]
        if rand_mean and rand_mean > 0 and not np.isnan(m):
            d["gap_vs_random_pct"] = float((rand_mean - m) / rand_mean * 100.0)
        else:
            d["gap_vs_random_pct"] = float("nan")

    # best method by mean simple regret (lower is better)
    valid = {k: v for k, v in summary_by_method.items() if not np.isnan(v["mean_simple_regret"])}
    best_method = min(valid, key=lambda k: valid[k]["mean_simple_regret"]) if valid else None

    return {
        "meta": {
            "base_seed": base_seed,
            "n_evals": n_evals,
            "n_init": n_init,
            "seeds": list(seeds),
            "n_functions": len(specs),
            "n_methods": len(methods),
            "functions": [s.name for s in specs],
            "methods": [m.name for m in methods],
        },
        "results": [r.__dict__ for r in results],
        "summary_by_method": summary_by_method,
        "best_method": best_method,
    }
