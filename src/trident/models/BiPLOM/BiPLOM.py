"""
BiPOLOM.py
==============================================

Public API
----------
biplom_solver(A, *, verbose=False, log_every=1000, **kwargs) -> dict
probability_matrix_from_biplom(res) -> ndarray
"""

from __future__ import annotations

from pathlib import Path
from itertools import combinations
try:
    # Numba is used exclusively as a performance accelerator.
    # The scientific correctness of the implementation does not rely on it.
    from numba import njit  # type: ignore
except Exception:  # pragma: no cover
    def njit(*_args, **_kwargs):
        """Fallback decorator used when numba is unavailable or broken."""

        def _decorator(func):
            return func

        return _decorator
import numpy as np

# original BiPLOM sources
from .bipclass       import BipartiteGraph1
from .miscellaneous  import bicmlo_from_fitnesses

# ────────────────────────────────────────────────────────────────────
# helpers for expectations
# ────────────────────────────────────────────────────────────────────
@njit(fastmath=True)
def _row_two_stars(k):
    return 0.5 * k * (k - 1)

@njit(fastmath=True)
def _prob_mat_numba(x, y, v, h_obs):
    N, M = x.size, y.size
    P = np.empty((N, M), dtype=np.float64)
    for i in range(N):
        xi = x[i]
        vi = v[i]
        for a in range(M):
            t = -(xi + y[a] + vi * h_obs[a])
            if t >= 0.0:
                P[i, a] = 1.0 / (1.0 + np.exp(-t))
            else:
                et = np.exp(t)
                P[i, a] = et / (1.0 + et)
    return P

# ────────────────────────────────────────────────────────────────────
# main wrapper
# ────────────────────────────────────────────────────────────────────
def biplom_solver(A: np.ndarray,
                  *,
                  verbose: bool | int = False,
                  log_every: int      = 1_000,
                  method: str         = "newton",
                  **solver_kw) -> dict:
    """
    Fit the BiPLOM-IV model on a binary bi-adjacency matrix *A*.

    Parameters
    ----------
    A : ndarray[int]      shape (N, M), 0/1
    verbose : bool|int    False → silent;
                          True  → progress every `log_every` its;
                          int   → use that int as `log_every`
    log_every : int       interval for progress prints if verbose
    method : str          'newton' | 'quasinewton' | 'fixed-point'
    **solver_kw           forwarded to BipartiteGraph1.solve_tool

    Returns
    -------
    dict  – fully compatible with Bi2SM diagnostics utilities.
    """

    A = (A > 0).astype(int)
    N, M = A.shape
    k_obs = A.sum(axis=1).astype(float)
    h_obs = A.sum(axis=0).astype(float)

    # ----------------------------------------------------------------
    # 1.  Build & solve the original BiPLOM object (exp mode, IV var.)
    # ----------------------------------------------------------------
    G = BipartiteGraph1(biadjacency=A, V_mod=4, method=method)
    # decide solver verbosity
    solver_verbose = bool(verbose)
    if isinstance(verbose, int) and verbose > 0:
        log_every = verbose
        solver_verbose = True

    G.solve_tool(method=method,
                 exp=True,
                 verbose=solver_verbose,
                 linsearch=True,
                 full_return=False,
                 **solver_kw)

    # ----------------------------------------------------------------
    # 2.  Collect fitnesses
    # ----------------------------------------------------------------
    theta_x = G.theta_x            # shape (N,)
    theta_y = G.theta_y            # shape (M,)
    theta_v = G.theta_V            # shape (N,)

    # ----------------------------------------------------------------
    # 3.  Probability matrix  P_{iα}
    # ----------------------------------------------------------------
    P = bicmlo_from_fitnesses(4, k_obs, h_obs, theta_x, theta_y, theta_v, None)

    # ----------------------------------------------------------------
    # 4.  Observed & expected statistics (same as other solvers)
    # ----------------------------------------------------------------
    k_exp = P.sum(axis=1)
    h_exp = P.sum(axis=0)

    row_P2 = (P ** 2).sum(axis=1)
    col_P2 = (P ** 2).sum(axis=0)

    S_i_obs = _row_two_stars(k_obs)
    T_a_obs = _row_two_stars(h_obs)
    S_i_exp = 0.5 * (k_exp ** 2 - row_P2)
    T_a_exp = 0.5 * (h_exp ** 2 - col_P2)

    res = dict(
        theta_x=theta_x, theta_y=theta_y, theta_v=theta_v,
        L_obs=k_obs.sum(), L_exp=k_exp.sum(),
        S_obs=S_i_obs.sum(), S_exp=S_i_exp.sum(),
        T_obs=T_a_obs.sum(), T_exp=T_a_exp.sum(),
        k_obs=k_obs, k_exp=k_exp,
        h_obs=h_obs, h_exp=h_exp,
        S_i_obs=S_i_obs, S_i_exp=S_i_exp,
        T_a_obs=T_a_obs, T_a_exp=T_a_exp,
        constraint="BiPLOM-IV",
        solver_steps=int(G.n_steps) if hasattr(G, "n_steps") else None,
        best_rel_change=float(G.error) if G.error is not None else None,
    )

    # optional console feedback harmonised with other solvers
    if verbose:
        print(f"[BiPLOM] finished in {res['solver_steps']} iterations   "
              f"Δ* = {res['best_rel_change']:.3e}")

    return res


# --------------------------------------------------------------------
# 5.  Public helper to rebuild P quickly (numba)
# --------------------------------------------------------------------
def probability_matrix_from_biplom(res: dict) -> np.ndarray:
    """
    Return the full N×M edge-probability matrix for a BiPLOM result.
    """
    return _prob_mat_numba(res["theta_x"], res["theta_y"], res["theta_v"], res["h_obs"])
