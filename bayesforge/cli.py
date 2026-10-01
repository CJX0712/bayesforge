"""Command-line entry point for BayesForge.

Usage:
    python -m bayesforge.cli run      # run benchmark, write benchmark.json
    python -m bayesforge.cli table     # print summary of an existing benchmark.json
    python -m bayesforge.cli check     # determinism self-check (run twice, compare)
"""

from __future__ import annotations

import argparse
import json
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from .core.config import load_config
from .core.seed import set_all
from .data.registry import demo_benchmark_functions, demo_functions
from .optimizer.registry import all_optimizers, core_optimizers
from .pipeline.benchmark import benchmark


def _print_summary(rep: dict) -> None:
    print("\n=== BayesForge benchmark summary ===")
    meta = rep["meta"]
    print(
        f"seed={meta['base_seed']} n_evals={meta['n_evals']} n_init={meta['n_init']} "
        f"functions={meta['n_functions']} methods={meta['n_methods']} seeds={len(meta['seeds'])}"
    )
    print()
    hdr = f"{'method':<22}{'mean_SR':>12}{'std_SR':>12}{'gap%_vs_rand':>14}"
    print(hdr)
    print("-" * len(hdr))
    for m, d in rep["summary_by_method"].items():
        msr = d.get("mean_simple_regret", float("nan"))
        ssr = d.get("std_simple_regret", float("nan"))
        gap = d.get("gap_vs_random_pct", float("nan"))
        print(f"{m:<22}{msr:>12.4f}{ssr:>12.4f}{(gap if gap == gap else float('nan')):>14.1f}")
    print(f"\nbest_method = {rep['best_method']}")


def _cmd_run(args) -> None:
    cfg = load_config(
        seed=args.seed,
        n_evals=args.n_evals,
        n_init=args.n_init,
        kernel=args.kernel,
        acq=args.acq,
        patience=args.patience,
    )
    set_all(cfg.seed)
    specs = demo_functions() if args.full else demo_benchmark_functions()
    methods = all_optimizers(cfg) if args.all else core_optimizers(cfg)
    rep = benchmark(
        specs,
        methods,
        seeds=list(range(cfg.seed, cfg.seed + args.seeds)),
        n_evals=cfg.n_evals,
        n_init=cfg.n_init,
        base_seed=cfg.seed,
    )
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2, ensure_ascii=False)
    print(f"wrote {args.out}")
    _print_summary(rep)


def _cmd_table(args) -> None:
    with open(args.json, "r", encoding="utf-8") as fh:
        rep = json.load(fh)
    _print_summary(rep)


def _cmd_check(args) -> None:
    cfg = load_config(seed=args.seed, n_evals=args.n_evals, n_init=args.n_init)
    specs = demo_functions()
    methods = core_optimizers(cfg)
    seeds = list(range(cfg.seed, cfg.seed + 3))
    r1 = benchmark(specs, methods, seeds, cfg.n_evals, cfg.n_init, base_seed=cfg.seed)
    r2 = benchmark(specs, methods, seeds, cfg.n_evals, cfg.n_init, base_seed=cfg.seed)
    # compare core metrics
    diffs = []
    for a, b in zip(r1["results"], r2["results"]):
        if a["skipped"] or b["skipped"]:
            continue
        diffs.append(abs(a["simple_regret"] - b["simple_regret"]))
    max_diff = max(diffs) if diffs else 0.0
    print(f"determinism check: max |SR diff| over {len(diffs)} runs = {max_diff:.2e}")
    print("PASS" if max_diff < 1e-9 else "FAIL")


def main() -> None:
    p = argparse.ArgumentParser(prog="bayesforge", description="Bayesian optimization toolkit (author: 晨星)")
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("run")
    pr.add_argument("--seed", type=int, default=42)
    pr.add_argument("--n-evals", dest="n_evals", type=int, default=60)
    pr.add_argument("--n-init", dest="n_init", type=int, default=5)
    pr.add_argument("--seeds", type=int, default=3)
    pr.add_argument("--kernel", default="rbf")
    pr.add_argument("--acq", default="ei")
    pr.add_argument("--patience", type=int, default=10)
    pr.add_argument("--out", default="benchmark.json")
    pr.add_argument("--all", action="store_true", help="include optional SOTA backends (skopt/bayesopt)")
    pr.add_argument("--full", action="store_true", help="use the extended 6-function set (slower)")
    pr.set_defaults(func=_cmd_run)
    pt = sub.add_parser("table")
    pt.add_argument("--json", default="benchmark.json")
    pt.set_defaults(func=_cmd_table)
    pc = sub.add_parser("check")
    pc.add_argument("--seed", type=int, default=42)
    pc.add_argument("--n-evals", dest="n_evals", type=int, default=60)
    pc.add_argument("--n-init", dest="n_init", type=int, default=5)
    pc.set_defaults(func=_cmd_check)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
