"""Optimizer registry: always-available core + optional SOTA backends."""

from __future__ import annotations

from typing import List

from ..core.config import Config
from .optimizers import (
    GPBO,
    BayesFuse,
    DifferentialEvolution,
    GridSearch,
    RandomSearch,
    _bayesopt_optimizer,
    _skopt_optimizer,
)


def core_optimizers(cfg: Config) -> List:
    return [
        RandomSearch(),
        GridSearch(),
        DifferentialEvolution(),
        GPBO(kernel=cfg.kernel, acq=cfg.acq, noise_floor=cfg.noise_floor),
        BayesFuse(kernel=cfg.kernel, noise_floor=cfg.noise_floor, patience=cfg.patience),
    ]


def sota_optimizers() -> List:
    out: List = []
    sk = _skopt_optimizer()
    if sk is not None:
        out.append(sk())
    bo = _bayesopt_optimizer()
    if bo is not None:
        out.append(bo())
    return out


def resolve_backends(spec, cfg: Config | None = None) -> List:
    """Resolve a backend spec (str or tuple/list of str) to optimizer list."""
    if isinstance(spec, str):
        spec = [spec]
    spec = list(spec)
    if spec == ["auto"]:
        return core_optimizers(cfg if cfg is not None else Config())
    out: List = []
    for s in spec:
        if s == "numpy":
            out.append(GPBO())
            out.append(BayesFuse())
        elif s == "skopt":
            sk = _skopt_optimizer()
            if sk is not None:
                out.append(sk())
        elif s == "bayesopt":
            bo = _bayesopt_optimizer()
            if bo is not None:
                out.append(bo())
    return out


def all_optimizers(cfg: Config) -> List:
    opts = core_optimizers(cfg)
    opts.extend(sota_optimizers())
    return opts
