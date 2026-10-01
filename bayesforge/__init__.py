"""BayesForge — world-class Bayesian optimization system.

Author: 晨星 (CJX0712).  Repo: https://github.com/CJX0712/bayesforge
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "晨星"

from .core.config import Config, load_config
from .core.seed import set_all
from .data.registry import all_functions, demo_benchmark_functions, demo_functions
from .optimizer.optimizers import (
    GPBO,
    BayesFuse,
    DifferentialEvolution,
    GridSearch,
    RandomSearch,
)
from .pipeline.benchmark import benchmark

__all__ = [
    "Config",
    "load_config",
    "set_all",
    "all_functions",
    "demo_functions",
    "demo_benchmark_functions",
    "BayesFuse",
    "DifferentialEvolution",
    "GPBO",
    "GridSearch",
    "RandomSearch",
    "benchmark",
    "__version__",
    "__author__",
]
