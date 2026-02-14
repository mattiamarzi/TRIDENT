"""trident.models.BiCM

Implementation of the Bipartite Configuration Model (BiCM).

The BiCM is the canonical maximum-entropy ensemble of binary bipartite
graphs reproducing, in expectation, the degree sequence of both layers.
In exponential parametrization the link probabilities read

\begin{equation}
    p_{i\alpha} = \frac{x_i y_\alpha}{1 + x_i y_\alpha},
\end{equation}

where $x_i > 0$ and $y_\alpha > 0$ are Lagrange multipliers in exponential
form. They are fitted by enforcing

\begin{align}
    k_i^* &= \sum_{\alpha} p_{i\alpha},\\
    h_\alpha^* &= \sum_i p_{i\alpha}.
\end{align}

This module provides a lightweight solver based on alternating 1D root
finding (bisection) for each multiplier, exploiting monotonicity of the
constraint equations with respect to a single variable while keeping the
other layer fixed.

Notes
-----
The implementation prioritizes numerical robustness and ease of
distribution. For very large matrices, faster Newton or quasi-Newton
schemes may be preferable; however, the present solver is adequate for
unit tests, examples, and moderate-size datasets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BiCMResult:
    """Container for BiCM fitted parameters and diagnostics."""

    x: np.ndarray
    y: np.ndarray
    P: np.ndarray
    k_obs: np.ndarray
    h_obs: np.ndarray
    k_exp: np.ndarray
    h_exp: np.ndarray
    L_obs: float
    L_exp: float


def _p_from_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute $P$ from exponential parameters $x$ and $y$."""
    M = np.outer(x, y)
    return M / (1.0 + M)


def _solve_positive_root_bisect(
    target: float,
    y: np.ndarray,
    *,
    tol: float,
    max_iter: int,
) -> float:
    """Solve for $x \ge 0$ the equation $\sum_\alpha x y_\alpha/(1+x y_\alpha)=\text{target}$.

    The left-hand side is continuous and strictly increasing in $x$ for
    $y_\alpha \ge 0$, thus bisection provides a globally convergent scheme.
    """

    if target <= 0.0:
        return 0.0

    # Numerical guard: maximum achievable sum is len(y) as x -> +inf.
    m = float(y.size)
    if target >= m:
        # Return a large value, still finite, yielding probabilities close to 1.
        return 1e12

    y = np.asarray(y, dtype=float)
    if np.all(y == 0.0):
        return 0.0

    def f(x: float) -> float:
        z = x * y
        return float(np.sum(z / (1.0 + z)))

    lo = 0.0
    hi = 1.0
    while f(hi) < target:
        hi *= 2.0
        if hi > 1e12:
            break

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        val = f(mid)
        if abs(val - target) <= tol * max(1.0, target):
            return mid
        if val < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def bicm_solver(
    A: np.ndarray,
    *,
    tol: float = 1e-10,
    max_outer_iter: int = 200,
    max_bisect_iter: int = 80,
    return_probability_matrix: bool = True,
) -> dict:
    """Fit the BiCM on a binary biadjacency matrix.

    Parameters
    ----------
    A:
        Binary matrix with shape (N, M).
    tol:
        Relative tolerance used both in fitting constraints and in the
        bisection subproblems.
    max_outer_iter:
        Maximum number of alternating updates over the two layers.
    max_bisect_iter:
        Maximum number of bisection iterations per scalar subproblem.
    return_probability_matrix:
        If True, include the full $P$ matrix in the returned dictionary.

    Returns
    -------
    dict
        A result dictionary exposing fitted parameters and observed/expected
        degrees. The structure is kept intentionally simple so that it can be
        consumed by validators and user code.
    """

    A = (np.asarray(A) > 0).astype(np.uint8)
    n, m = A.shape
    k_obs = A.sum(axis=1).astype(float)
    h_obs = A.sum(axis=0).astype(float)

    # Initialize positive parameters with small values proportional to degrees.
    # This improves stability in sparse regimes.
    x = np.clip(k_obs / max(1.0, float(m)), 1e-12, 1.0)
    y = np.clip(h_obs / max(1.0, float(n)), 1e-12, 1.0)

    # Alternating updates: solve each scalar constraint exactly given the other layer.
    for _ in range(max_outer_iter):
        x_old = x.copy()
        y_old = y.copy()

        for i in range(n):
            x[i] = _solve_positive_root_bisect(
                float(k_obs[i]),
                y,
                tol=tol,
                max_iter=max_bisect_iter,
            )

        for a in range(m):
            y[a] = _solve_positive_root_bisect(
                float(h_obs[a]),
                x,
                tol=tol,
                max_iter=max_bisect_iter,
            )

        # Convergence in parameters (scale-invariant diagnostics are delicate,
        # so we use a relative criterion on the update itself).
        dx = float(np.max(np.abs(x - x_old) / np.maximum(1.0, np.abs(x_old))))
        dy = float(np.max(np.abs(y - y_old) / np.maximum(1.0, np.abs(y_old))))
        if max(dx, dy) <= tol:
            break

    P = _p_from_xy(x, y)
    k_exp = P.sum(axis=1)
    h_exp = P.sum(axis=0)
    L_obs = float(k_obs.sum())
    L_exp = float(k_exp.sum())

    res = dict(
        x=x,
        y=y,
        k_obs=k_obs,
        h_obs=h_obs,
        k_exp=k_exp,
        h_exp=h_exp,
        L_obs=L_obs,
        L_exp=L_exp,
        constraint="BiCM",
    )
    if return_probability_matrix:
        res["P"] = P
    return res


def probability_matrix_from_bicm(res_bicm: dict) -> np.ndarray:
    """Construct the BiCM probability matrix from a `bicm_solver` result."""

    if "P" in res_bicm and isinstance(res_bicm["P"], np.ndarray):
        return np.asarray(res_bicm["P"], dtype=float)
    x = np.asarray(res_bicm["x"], dtype=float)
    y = np.asarray(res_bicm["y"], dtype=float)
    return _p_from_xy(x, y)
