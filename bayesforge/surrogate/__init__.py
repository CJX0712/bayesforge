"""BayesForge surrogate package."""

from .gp import NumpyGP, available_numpy, make_numpy_gp

__all__ = ["NumpyGP", "available_numpy", "make_numpy_gp"]
