"""Global deterministic seeding.

A single entry point :func:`set_all` fixes every RNG used across the system so
that two runs with the same seed produce bit-identical benchmark numbers.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

_SEED: Optional[int] = None


def set_all(seed: int = 42) -> np.random.Generator:
    """Seed python ``random``, ``numpy`` global state, and return a Generator.

    The returned ``Generator`` is the preferred source for all new randomness
    inside the system; legacy libraries that only read ``np.random`` are also
    covered via ``np.random.seed``.
    """
    global _SEED
    _SEED = int(seed)
    random.seed(_SEED)
    np.random.seed(_SEED)
    os.environ["PYTHONHASHSEED"] = str(_SEED)
    return np.random.default_rng(_SEED)


def get_seed() -> Optional[int]:
    return _SEED
