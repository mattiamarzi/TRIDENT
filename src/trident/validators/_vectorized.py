"""
_vectorized.py
══════════════
P-values of all pairs / triplets at once, for the methods "poisson", "normal" and "rna".

The loop validators build, for every tuple, q_α = P_iα P_jα (P_kα) and sum it.
Here the same sums are obtained for all tuples with array operations:

    pairs    : row-wise products P_i * P_j, as in the loop (identical p-values);
    triplets : μ = Σ q, Σ q², Σ q³ and the observed counts by matrix products,
               σ² = μ - Σ q²,  Σ q(1-q)(1-2q) = μ - 3 Σ q² + 2 Σ q³.

The p-value formulas and edge cases are those of the loop workers
(V = 0 → 1; zero variance → 1 if V ≤ μ else 0). Triplets whose variance is
close to zero (σ² ≤ 1e-6 μ, where the difference above loses digits) are
recomputed element-wise as in the loop. Memory is O(N·M + N²) per block.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, poisson

VECTOR_METHODS = ("poisson", "normal", "rna")
_ENGINES = ("auto", "vectorized", "loop")


def resolve_engine(method: str, engine: str) -> str:
    """Return "vectorized" or "loop" for the requested method and engine."""
    if engine not in _ENGINES:
        raise ValueError("engine must be one of: 'auto', 'vectorized', 'loop'.")
    if engine == "vectorized" and method not in VECTOR_METHODS:
        raise ValueError("engine='vectorized' supports the methods 'poisson', 'normal', 'rna'.")
    if engine == "auto":
        return "vectorized" if method in VECTOR_METHODS else "loop"
    return engine


# =============================================================================
# Moments of the Poisson–binomial counts
# =============================================================================
def pair_moments(A: np.ndarray, P: np.ndarray):
    """Observed counts V, mean μ, variance σ² and third central moment of all
    pairs i<j, in lexicographic order."""
    A = (np.asarray(A) > 0).astype(np.float64)
    P = np.asarray(P, dtype=float)
    n = A.shape[0]
    V, MU, VAR, M3 = [], [], [], []
    for i in range(n - 1):
        q = P[i] * P[i + 1:]                  # q_α = P_iα P_jα, one row per j
        var_arr = q * (1.0 - q)
        V.append(A[i + 1:] @ A[i])
        MU.append(q.sum(axis=1))
        VAR.append(var_arr.sum(axis=1))
        M3.append((var_arr * (1.0 - 2.0 * q)).sum(axis=1))
    if not V:
        e = np.empty(0)
        return e, e, e, e
    return tuple(np.concatenate(x) for x in (V, MU, VAR, M3))


def triplet_moments(A: np.ndarray, P: np.ndarray):
    """Observed counts V, mean μ, variance σ² and third central moment of all
    triplets i<j<k, in lexicographic order."""
    A = (np.asarray(A) > 0).astype(np.float64)
    P = np.asarray(P, dtype=float)
    P2, P3 = P ** 2, P ** 3
    n = A.shape[0]
    V, MU, Q2, Q3 = [], [], [], []
    for i in range(n - 2):
        W = P[i] * P[i + 1:n - 1]             # P_iα P_jα, one row per j
        Wa = A[i] * A[i + 1:n - 1]
        Wq2 = W * W
        keep = np.arange(i + 2, n)[None, :] > np.arange(i + 1, n - 1)[:, None]   # k > j
        V.append((Wa @ A[i + 2:].T)[keep])
        MU.append((W @ P[i + 2:].T)[keep])
        Q2.append((Wq2 @ P2[i + 2:].T)[keep])
        Q3.append(((Wq2 * W) @ P3[i + 2:].T)[keep])
    if not V:
        e = np.empty(0)
        return e, e, e, e
    V, MU, Q2, Q3 = (np.concatenate(x) for x in (V, MU, Q2, Q3))
    VAR = MU - Q2
    M3 = MU - 3.0 * Q2 + 2.0 * Q3

    # Near-degenerate variance: recompute as in the loop (sum of q(1-q) etc.)
    bad = np.flatnonzero(VAR <= 1e-6 * MU)
    if bad.size:
        I, J, K = triplet_index(n, bad)
        for s in range(0, bad.size, 4096):
            sl = slice(s, s + 4096)
            q = P[I[sl]] * P[J[sl]] * P[K[sl]]
            var_arr = q * (1.0 - q)
            MU[bad[sl]] = q.sum(axis=1)
            VAR[bad[sl]] = var_arr.sum(axis=1)
            M3[bad[sl]] = (var_arr * (1.0 - 2.0 * q)).sum(axis=1)
    return V, MU, VAR, M3


def triplet_index(n: int, pos: np.ndarray):
    """(i, j, k) of the triplets at positions `pos` of the lexicographic order."""
    first = np.array([(n - 1 - i) * (n - 2 - i) // 2 for i in range(n)])
    start = np.concatenate([[0], np.cumsum(first)])
    I = np.searchsorted(start, pos, side="right") - 1
    R = pos - start[I]                        # position among the pairs (j, k) of {i+1, ..., n-1}
    J = np.empty(pos.size, dtype=int)
    K = np.empty(pos.size, dtype=int)
    for t, (i, r) in enumerate(zip(I.tolist(), R.tolist())):   # few triplets: plain loop
        j = i + 1
        while r >= n - 1 - j:                 # n-1-j pairs start with j
            r -= n - 1 - j
            j += 1
        J[t], K[t] = j, j + 1 + r
    return I, J, K


# =============================================================================
# P-values
# =============================================================================
def pvalues(v, mu, var, m3, method: str) -> np.ndarray:
    """Upper-tail p-values with the formulas and edge cases of the loop workers."""
    v = np.asarray(v, dtype=float)
    p = np.ones(v.shape, dtype=float)
    pos = v > 0                                       # V = 0 → p = 1
    if method == "poisson":
        p[pos] = poisson.sf(v[pos] - 1, mu[pos])
        return p
    if method not in ("normal", "rna"):
        raise ValueError("method must be one of: 'poisson', 'normal', 'rna'.")
    deg = pos & ~(var > 0.0)                          # zero variance
    p[deg] = np.where(v[deg] <= mu[deg], 1.0, 0.0)
    ok = pos & (var > 0.0)
    sigma = np.sqrt(var[ok])
    z = (v[ok] + 0.5 - mu[ok]) / sigma
    if method == "normal":
        p[ok] = norm.sf(z)
        return p
    gamma = m3[ok] / sigma ** 3
    cdf = norm.cdf(z) + gamma * (1.0 - z ** 2) * norm.pdf(z) / 6.0
    p[ok] = 1.0 - np.clip(cdf, 0.0, 1.0)
    return p


# =============================================================================
# Observed counts as nested dictionaries (same keys and order as the set-based
# _pair_counts / _triplet_counts, computed with matrix products)
# =============================================================================
def pair_counts(A: np.ndarray, include_absent: bool = True) -> dict:
    A = (np.asarray(A) > 0).astype(np.float64)
    n = A.shape[0]
    C = np.rint(A @ A.T).astype(np.int64)
    out: dict = {}
    for i in range(n - 1):
        row = dict(zip(range(i + 1, n), C[i, i + 1:].tolist()))
        if not include_absent:
            row = {j: c for j, c in row.items() if c}
        if row:
            out[i] = row
    return out


def triplet_counts(A: np.ndarray, include_absent: bool = True) -> dict:
    A = (np.asarray(A) > 0).astype(np.float64)
    n = A.shape[0]
    out: dict = {}
    for i in range(n - 2):
        Ci = np.rint((A[i] * A[i + 1:n - 1]) @ A[i + 2:].T).astype(np.int64)   # rows j = i+1.., cols k = i+2..
        oi: dict = {}
        for jj, j in enumerate(range(i + 1, n - 1)):
            row = dict(zip(range(j + 1, n), Ci[jj, jj:].tolist()))
            if not include_absent:
                row = {k: c for k, c in row.items() if c}
            if row:
                oi[j] = row
        if oi:
            out[i] = oi
    return out


# =============================================================================
# Nested dictionaries with the keys of the loop validators
# =============================================================================
def nest_pairs(p: np.ndarray, n: int, counts: dict) -> dict:
    pl, out, pos = p.tolist(), {}, 0
    for i in range(n - 1):
        m = n - 1 - i
        row = dict(zip(range(i + 1, n), pl[pos:pos + m]))
        pos += m
        if i in counts:
            ci = counts[i]
            out[i] = row if len(ci) == m else {j: row[j] for j in ci}
    return out


def nest_triplets(p: np.ndarray, n: int, counts: dict) -> dict:
    pl, out, pos = p.tolist(), {}, 0
    for i in range(n - 2):
        ci = counts.get(i)
        oi = {}
        for j in range(i + 1, n - 1):
            m = n - 1 - j
            if ci is not None and j in ci:
                row = dict(zip(range(j + 1, n), pl[pos:pos + m]))
                cij = ci[j]
                oi[j] = row if len(cij) == m else {k: row[k] for k in cij}
            pos += m
        if ci is not None:
            out[i] = oi
    return out
