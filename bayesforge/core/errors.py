"""Error catalogue for BayesForge (E100–E500).

All exceptions derive from :class:`BayesForgeError` so callers can catch the
whole family with a single ``except``.
"""

from __future__ import annotations


class BayesForgeError(Exception):
    """Base class for every BayesForge error."""


class ConfigError(BayesForgeError):
    """E100 — configuration / environment validation failure."""


class DataError(BayesForgeError):
    """E200 — data generation or loading failure."""


class SurrogateError(BayesForgeError):
    """E300 — surrogate model (e.g. GP) failure."""


class AcquisitionError(BayesForgeError):
    """E400 — acquisition function failure."""


class OptimizerError(BayesForgeError):
    """E500 — optimizer / pipeline failure."""


def _fmt(code: str, msg: str) -> str:
    return f"[{code}] {msg}"


def config_err(msg: str) -> ConfigError:
    return ConfigError(_fmt("E100", msg))


def data_err(msg: str) -> DataError:
    return DataError(_fmt("E200", msg))


def surrogate_err(msg: str) -> SurrogateError:
    return SurrogateError(_fmt("E300", msg))


def acquisition_err(msg: str) -> AcquisitionError:
    return AcquisitionError(_fmt("E400", msg))


def optimizer_err(msg: str) -> OptimizerError:
    return OptimizerError(_fmt("E500", msg))
