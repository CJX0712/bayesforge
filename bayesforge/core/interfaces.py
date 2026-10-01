"""Protocol (structural) interfaces — the contract every module honors."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .types import Bounds, Observed


@runtime_checkable
class Surrogate(Protocol):
    """A probabilistic model mapping X -> (mean, std)."""

    dim: int

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Surrogate":
        ...

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, std) with std >= 0 for every row."""
        ...


@runtime_checkable
class Acquisition(Protocol):
    """Maps a surrogate posterior into a scalar score to maximize."""

    def __call__(self, mean: np.ndarray, std: np.ndarray, best_y: float) -> np.ndarray:
        ...


@runtime_checkable
class Optimizer(Protocol):
    """A black-box optimizer: given an objective + bounds, find the minimum."""

    name: str

    def minimize(
        self,
        f,
        bounds: Bounds,
        n_evals: int,
        rng: np.random.Generator,
        n_init: int = 5,
    ) -> Observed:
        ...
