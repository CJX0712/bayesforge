"""Shared dataclasses for BayesForge."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Tuple

import numpy as np

# Minimization everywhere: lower objective value == better.
ObjectiveFn = Callable[[np.ndarray], float]


@dataclass
class Bounds:
    """Box constraints ``lo <= x_i <= hi`` for every dimension."""

    lo: np.ndarray
    hi: np.ndarray

    def __post_init__(self) -> None:
        self.lo = np.asarray(self.lo, dtype=np.float64)
        self.hi = np.asarray(self.hi, dtype=np.float64)
        if self.lo.shape != self.hi.shape:
            from .errors import data_err

            raise data_err("lo and hi must have equal shape")
        if np.any(self.lo >= self.hi):
            from .errors import data_err

            raise data_err("every lo must be strictly smaller than hi")

    @property
    def dim(self) -> int:
        return int(self.lo.shape[0])

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` uniform random points inside the box."""
        u = rng.random((n, self.dim))
        return self.lo[None, :] + u * (self.hi - self.lo)[None, :]

    def clip(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.lo, self.hi)


@dataclass
class Observed:
    """Accumulated observations of a black-box objective."""

    xs: List[np.ndarray] = field(default_factory=list)
    ys: List[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.ys)

    def add(self, x: np.ndarray, y: float) -> None:
        self.xs.append(np.asarray(x, dtype=np.float64).ravel())
        self.ys.append(float(y))

    def X(self) -> np.ndarray:
        if not self.xs:
            return np.zeros((0, 0), dtype=np.float64)
        return np.stack(self.xs, axis=0)

    def Y(self) -> np.ndarray:
        return np.asarray(self.ys, dtype=np.float64)

    def best(self) -> Tuple[np.ndarray, float]:
        if not self.ys:
            raise ValueError("no observations yet")
        i = int(np.argmin(self.ys))
        return self.xs[i], float(self.ys[i])


@dataclass
class BenchmarkResult:
    """Per-method, per-function outcome for one seed."""

    method: str
    func: str
    dim: int
    n_evals: int
    best_y: float
    true_opt: float
    simple_regret: float
    cumulative_regret: float
    elapsed_sec: float
    skipped: bool = False
    note: str = ""


@dataclass
class FunctionSpec:
    """A benchmark black-box function with a *known* global optimum."""

    name: str
    f: ObjectiveFn
    bounds: Bounds
    optimum: float
    dim: int
    noise_std: float = 0.0

    def __post_init__(self) -> None:
        # Self-contained noise stream (seeded from the function name) so even
        # noisy variants stay bit-reproducible without global RNG state.
        seed = sum((i + 1) * ord(c) for i, c in enumerate(self.name)) & 0xFFFFFFFF
        self._noise_rng = np.random.default_rng(seed)

    def __call__(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64).ravel()
        y = float(self.f(x))
        if self.noise_std > 0.0:
            y += float(self.noise_std) * float(self._noise_rng.standard_normal())
        return y
