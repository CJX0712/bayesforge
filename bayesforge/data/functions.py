"""Benchmark black-box functions with *known* global optima (minimization).

Every function returns a scalar to minimize. ``true_opt`` records the analytic
global minimum so that simple regret = best_found - true_opt is exact.
"""

from __future__ import annotations

import math

import numpy as np


def sphere(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    return float(np.sum(x * x))


def rosenbrock(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    s = 0.0
    for i in range(len(x) - 1):
        s += 100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1.0 - x[i]) ** 2
    return float(s)


def ackley(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    d = len(x)
    a, b, c = 20.0, 0.2, 2.0 * math.pi
    sum_sq = float(np.sum(x * x))
    sum_cos = float(np.sum(np.cos(c * x)))
    return float(-a * math.exp(-b * math.sqrt(sum_sq / d)) - math.exp(sum_cos / d) + a + math.e)


def levy(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    d = len(x)
    w = 1.0 + (x - 1.0) / 4.0
    term1 = math.sin(math.pi * w[0]) ** 2
    term2 = 0.0
    for i in range(d - 1):
        term2 += (w[i] - 1.0) ** 2 * (1.0 + 10.0 * math.sin(math.pi * w[i] + 1.0) ** 2)
    term3 = (w[d - 1] - 1.0) ** 2 * (1.0 + math.sin(2.0 * math.pi * w[d - 1]) ** 2)
    return float(term1 + term2 + term3)


def branin(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    x1, x2 = float(x[0]), float(x[1])
    a = 1.0
    b = 5.1 / (4.0 * math.pi**2)
    c = 5.0 / math.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * math.pi)
    return float(a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1.0 - t) * math.cos(x1) + s)


def six_hump_camel(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    x1, x2 = float(x[0]), float(x[1])
    return float(
        (4.0 - 2.1 * x1**2 + (x1**4) / 3.0) * x1**2
        + x1 * x2
        + (-4.0 + 4.0 * x2**2) * x2**2
    )


def goldstein_price(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    x1, x2 = float(x[0]), float(x[1])
    a = 1.0 + (x1 + x2 + 1.0) ** 2 * (
        19.0 - 14.0 * x1 + 3.0 * x1**2 - 14.0 * x2 + 6.0 * x1 * x2 + 3.0 * x2**2
    )
    b = 30.0 + (2.0 * x1 - 3.0 * x2) ** 2 * (
        18.0 - 32.0 * x1 + 12.0 * x1**2 + 48.0 * x2 - 36.0 * x1 * x2 + 27.0 * x2**2
    )
    return float(a * b)


def michalewicz(x: np.ndarray, m: float = 10.0) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    d = len(x)
    s = 0.0
    for i in range(d):
        s += math.sin(x[i]) * math.sin(((i + 1) * x[i] ** 2) / math.pi) ** (2 * m)
    return float(-s)


def hartmann(x: np.ndarray, d: int = 3) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    if d == 3:
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array(
            [
                [3.0, 10.0, 30.0],
                [0.1, 10.0, 35.0],
                [3.0, 10.0, 30.0],
                [0.1, 10.0, 35.0],
            ]
        )
        P = 1e-4 * np.array(
            [
                [3689, 1170, 2673],
                [4699, 4387, 7470],
                [1091, 8732, 5547],
                [381, 5743, 8828],
            ]
        )
    elif d == 6:
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array(
            [
                [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
                [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
                [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
                [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
            ]
        )
        P = 1e-4 * np.array(
            [
                [1312, 1696, 5569, 124.0, 8283, 5886],
                [2329, 4135, 8307, 3736, 1004, 9991],
                [2348, 1451, 3522, 2883, 3047, 6650],
                [4047, 8828, 8732, 5743, 1091, 381],
            ]
        )
    else:
        raise ValueError("hartmann only supports d=3 or d=6")
    outer = 0.0
    for i in range(4):
        inner = 0.0
        for j in range(d):
            inner += A[i, j] * (x[j] - P[i, j]) ** 2
        outer += alpha[i] * math.exp(-inner)
    return float(-outer)
