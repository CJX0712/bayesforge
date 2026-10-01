"""Pipeline / registry / CLI smoke tests (author: 晨星)."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from bayesforge.core.config import load_config
from bayesforge.data.registry import demo_benchmark_functions
from bayesforge.hpo.tune import tune_gpbo
from bayesforge.optimizer.registry import (
    all_optimizers,
    core_optimizers,
    resolve_backends,
)
from bayesforge.pipeline.benchmark import benchmark, run_one


def _cfg():
    return load_config(seed=42, n_evals=20, n_init=5)


def test_benchmark_output_structure():
    cfg = _cfg()
    methods = core_optimizers(cfg)
    rep = benchmark(demo_benchmark_functions(), methods, [42, 43], cfg.n_evals, cfg.n_init, base_seed=42)
    assert set(rep) >= {"meta", "results", "summary_by_method", "best_method"}
    assert len(rep["results"]) == len(methods) * 4 * 2
    for m in methods:
        assert m.name in rep["summary_by_method"]
    assert rep["best_method"] in {m.name for m in methods}


def test_benchmark_gap_vs_random_is_defined():
    cfg = _cfg()
    rep = benchmark(demo_benchmark_functions(), core_optimizers(cfg), [42], cfg.n_evals, cfg.n_init)
    assert rep["summary_by_method"]["random_search"]["gap_vs_random_pct"] == pytest.approx(0.0)


def test_benchmark_is_bit_reproducible():
    cfg = _cfg()
    specs, methods, seeds = demo_benchmark_functions(), core_optimizers(cfg), [42, 43]
    r1 = benchmark(specs, methods, seeds, cfg.n_evals, cfg.n_init, base_seed=42)
    r2 = benchmark(specs, methods, seeds, cfg.n_evals, cfg.n_init, base_seed=42)
    for a, b in zip(r1["results"], r2["results"]):
        assert a["simple_regret"] == b["simple_regret"] or (a["skipped"] and b["skipped"])


def test_run_one_reports_skipped_on_failure():
    class Boom:
        name = "boom"

        def minimize(self, *a, **k):
            raise RuntimeError("kaboom")

    r = run_one(Boom(), demo_benchmark_functions()[0], 20, 5, 42, 0, 0, 0)
    assert r.skipped and "kaboom" in r.note


def test_registry_core_contains_expected_methods():
    names = {m.name for m in core_optimizers(_cfg())}
    assert names == {"random_search", "grid_search", "differential_evolution", "gp_bo", "bayesfuse"}


def test_all_optimizers_superset_of_core():
    core = {m.name for m in core_optimizers(_cfg())}
    assert core <= {m.name for m in all_optimizers(_cfg())}


def test_resolve_backends_auto_returns_list():
    out = resolve_backends("auto")
    assert isinstance(out, list) and len(out) >= 5


def test_resolve_backends_numpy():
    out = resolve_backends(["numpy"])
    assert {m.name for m in out} == {"gp_bo", "bayesfuse"}


def test_tune_gpbo_returns_valid_combo():
    k, a = tune_gpbo(demo_benchmark_functions()[:1], [42], n_evals=12, n_init=5)
    assert k in ("rbf", "matern52") and a in ("ei", "lcb", "pi")


# ----------------------------- CLI -----------------------------
def _cli(*args):
    return subprocess.run([sys.executable, "-m", "bayesforge.cli", *args],
                          capture_output=True, text=True, encoding="utf-8", timeout=600)


def test_cli_run_writes_json(tmp_path):
    out = tmp_path / "bm.json"
    r = _cli("run", "--seed", "7", "--n-evals", "16", "--seeds", "1", "--out", str(out))
    assert r.returncode == 0, r.stdout + r.stderr
    assert out.exists()
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["meta"]["base_seed"] == 7
    assert rep["best_method"]


def test_cli_table_reads_json(tmp_path):
    out = tmp_path / "bm.json"
    assert _cli("run", "--n-evals", "16", "--seeds", "1", "--out", str(out)).returncode == 0
    r = _cli("table", "--json", str(out))
    assert r.returncode == 0 and "best_method" in r.stdout


def test_cli_check_reports_pass(tmp_path):
    r = _cli("check", "--n-evals", "16")
    assert r.returncode == 0
    assert "PASS" in r.stdout, r.stdout + r.stderr
