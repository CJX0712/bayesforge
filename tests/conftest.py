"""Shared pytest fixtures for BayesForge (author: 晨星)."""

from __future__ import annotations

import numpy as np
import pytest

from bayesforge.core.types import Bounds, FunctionSpec
from bayesforge.data.functions import branin, sphere


@pytest.fixture
def box2d() -> Bounds:
    return Bounds(lo=np.array([-5.0, 0.0]), hi=np.array([10.0, 15.0]))


@pytest.fixture
def branin_spec() -> FunctionSpec:
    return FunctionSpec("branin", branin, Bounds(lo=np.array([-5.0, 0.0]), hi=np.array([10.0, 15.0])),
                        0.397887357, 2)


@pytest.fixture
def sphere_spec() -> FunctionSpec:
    return FunctionSpec("sphere3", sphere, Bounds(lo=np.full(3, -5.0), hi=np.full(3, 5.0)), 0.0, 3)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)
