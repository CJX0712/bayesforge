"""Configuration with ``ENV_BO_*`` overrides and schema validation."""

from __future__ import annotations

import os
from dataclasses import dataclass

from .errors import config_err


@dataclass
class Config:
    """Runtime configuration. Every field may be overridden via env vars.

    Env mapping (prefix ``BO_``): ``BO_SEED``, ``BO_N_INIT``, ``BO_N_EVALS``,
    ``BO_NOISE_FLOOR``, ``BO_KERNEL``, ``BO_ACQ``, ``BO_PATIENCE``.
    """

    seed: int = 42
    n_init: int = 5
    n_evals: int = 60
    noise_floor: float = 1e-6
    kernel: str = "rbf"
    acq: str = "ei"
    patience: int = 10
    backend: str = "auto"

    _ENV_MAP = {
        "BO_SEED": ("seed", int),
        "BO_N_INIT": ("n_init", int),
        "BO_N_EVALS": ("n_evals", int),
        "BO_NOISE_FLOOR": ("noise_floor", float),
        "BO_KERNEL": ("kernel", str),
        "BO_ACQ": ("acq", str),
        "BO_PATIENCE": ("patience", int),
        "BO_BACKEND": ("backend", str),
    }

    @classmethod
    def from_env(cls, **overrides: object) -> "Config":
        kwargs: dict = {}
        for env_key, (attr, caster) in cls._ENV_MAP.items():
            val = os.environ.get(env_key)
            if val is not None:
                try:
                    kwargs[attr] = caster(val)
                except (ValueError, TypeError) as exc:
                    raise config_err(f"invalid env {env_key}={val!r}: {exc}") from exc
        kwargs.update(overrides)
        cfg = cls(**kwargs)  # type: ignore[arg-type]
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.seed < 0:
            raise config_err("seed must be >= 0")
        if self.n_init < 1:
            raise config_err("n_init must be >= 1")
        if self.n_evals <= self.n_init:
            raise config_err("n_evals must be > n_init")
        if self.noise_floor < 0:
            raise config_err("noise_floor must be >= 0")
        if self.kernel not in ("rbf", "matern52"):
            raise config_err(f"unknown kernel {self.kernel!r}")
        if self.acq not in ("ei", "pi", "lcb", "portfolio"):
            raise config_err(f"unknown acq {self.acq!r}")
        if self.patience < 1:
            raise config_err("patience must be >= 1")
        if self.backend not in ("auto", "numpy", "skopt", "bayesopt"):
            raise config_err(f"unknown backend {self.backend!r}")


def load_config(**overrides: object) -> Config:
    return Config.from_env(**overrides)
