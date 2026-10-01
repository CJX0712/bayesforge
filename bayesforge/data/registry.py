"""Benchmark function registry + file loaders.

The *demo* set is curated to keep the end-to-end run inside the 60s CPU budget
while spanning low- and high-dimensional, convex and multimodal, smooth and
rugged landscapes. All demo functions are **noise-free** so two runs with the
same seed are bit-identical (determinism gate).
"""

from __future__ import annotations

import csv
import math
from typing import List, Tuple

import numpy as np

from ..core.types import Bounds, FunctionSpec
from .functions import (
    ackley,
    branin,
    goldstein_price,
    hartmann,
    levy,
    michalewicz,
    rosenbrock,
    six_hump_camel,
    sphere,
)


def _b(lo, hi, n) -> Bounds:
    return Bounds(lo=np.full(n, lo), hi=np.full(n, hi))


def all_functions() -> List[FunctionSpec]:
    """Full catalogue (used by tests / extended runs)."""
    specs = [
        FunctionSpec("sphere3", sphere, _b(-5.0, 5.0, 3), 0.0, 3),
        FunctionSpec("rosenbrock3", rosenbrock, _b(-2.0, 2.0, 3), 0.0, 3),
        FunctionSpec("ackley3", ackley, _b(-5.0, 10.0, 3), 0.0, 3),
        FunctionSpec("levy3", levy, _b(-10.0, 10.0, 3), 0.0, 3),
        FunctionSpec("branin", branin, Bounds(lo=np.array([-5.0, 0.0]), hi=np.array([10.0, 15.0])), 0.397887357, 2),
        FunctionSpec(
            "six_hump_camel",
            six_hump_camel,
            Bounds(lo=np.array([-3.0, -2.0]), hi=np.array([3.0, 2.0])),
            -1.031628453,
            2,
        ),
        FunctionSpec("goldstein_price", goldstein_price, _b(-2.0, 2.0, 2), 3.0, 2),
        FunctionSpec(
            "michalewicz2",
            michalewicz,
            Bounds(lo=np.array([0.0, 0.0]), hi=np.array([math.pi, math.pi])),
            -1.801303410,
            2,
        ),
        FunctionSpec("hartmann3", lambda x: hartmann(x, 3), _b(0.0, 1.0, 3), -3.862779531, 3),
        FunctionSpec("hartmann6", lambda x: hartmann(x, 6), _b(0.0, 1.0, 6), -3.322368011, 6),
    ]
    return specs


def demo_functions() -> List[FunctionSpec]:
    """Curated, noise-free subset (used by extended tests / reference runs)."""
    keep = {
        "sphere3",
        "rosenbrock3",
        "ackley3",
        "branin",
        "six_hump_camel",
        "hartmann3",
    }
    return [s for s in all_functions() if s.name in keep]


def demo_benchmark_functions() -> List[FunctionSpec]:
    """Lightweight 4-function set for the *timed* 60s demo.

    Spans convex (sphere), rugged/multimodal (branin, six_hump_camel) and a
    higher-dimensional smooth basin (hartmann3, dim-3). hartmann6 (dim-6) is
    validated separately in the extended test suite.
    """
    keep = {"sphere3", "branin", "six_hump_camel", "hartmann3"}
    return [s for s in all_functions() if s.name in keep]


def load_csv_pairs(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load (X, y) observation pairs from a CSV with header; last col = y."""
    import os

    if not os.path.exists(path):
        from ..core.errors import data_err

        raise data_err(f"file not found: {path}")
    X, y = [], []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        next(reader, None)  # header (validated by column count below)
        for row in reader:
            if not row or all(c.strip() == "" for c in row):
                continue
            nums = [float(c) for c in row]
            X.append(nums[:-1])
            y.append(nums[-1])
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
