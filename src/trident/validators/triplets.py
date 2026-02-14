"""
triplets.py
════════════
Three–star (country–triplet) validator under a bipartite null model.

Given a binary bi-adjacency matrix A (rows: countries, cols: products)
and a corresponding probability matrix P (same shape), this module
computes p-values for the number of common neighbors (triple V-motifs)
of each country triplet (i, j, k) under several approximations:

    • "poisson" :  S ≈ Poisson(μ)
    • "normal"  :  CLT with continuity correction (+0.5)
    • "rna"     :  refined normal approx. (skewness correction)
    • "poibin"  :  exact Poisson–binomial via FFT  (recommended)

Crucially, we enumerate **all** unordered triplets i<j<k, even if their
observed count is zero, to align with classical multiple-testing setups.

The Poisson–binomial tail is P{ S ≥ v_obs } so that smaller p-values
indicate more significant positive co-occurrence.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Standard libs
# ---------------------------------------------------------------------------
import math
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.stats import poisson, norm, binom

# =============================================================================
# FFT-based Poisson–binomial (exact)  — stable and fast
# =============================================================================
class PoiBin:
    """
    Exact Poisson–binomial with FFT (Hong, 2013 style).

    Parameters
    ----------
    p : 1D array-like of float
        Success probabilities for independent Bernoulli summands.

    Notes
    -----
    - pmf_list[k] stores P[S = k] for k = 0..n.
    - cdf_list[k] stores P[S ≤ k] for k = 0..n.
    - pval(k) returns the **upper tail** P[S ≥ k].
    - Falls back to Binomial(n, p0) when all probabilities are equal.
    """
    def __init__(self, p):
        self.p = np.asarray(p, dtype=float)
        self.n = self.p.size
        self._check_prob()
        self.omega = 2 * np.pi / (self.n + 1)
        self.pmf_list = self._pmf_fft()
        self.cdf_list = np.cumsum(self.pmf_list)

    # ----- public API -------------------------------------------------
    def pmf(self, k: int) -> float:
        self._check_k(k)
        return float(self.pmf_list[k])

    def cdf(self, k: int) -> float:
        self._check_k(k)
        return float(self.cdf_list[k])

    def pval(self, k: int) -> float:
        """
        Upper tail (right-tail) p-value: P[S ≥ k].
        """
        if k <= 0:
            return 1.0
        # equal-probability shortcut → Binomial SF (stable)
        if np.allclose(self.p, self.p[0]):
            return float(binom.sf(k - 1, self.n, float(self.p[0])))
        return float(1.0 - self.cdf_list[k - 1])

    # ----- internals --------------------------------------------------
    def _pmf_fft(self):
        n = self.n
        chi = np.empty(n + 1, dtype=np.complex128)
        half = (n + 1) // 2

        # First half (l = 0..half), magnitude via log-abs for stability,
        # phase via sum of angles.
        for l in range(0, half + 1):
            z       = 1 - self.p + self.p * np.exp(1j * self.omega * l)
            abs_log = np.log(np.abs(z)).sum()
            phase   = np.angle(z).sum()
            chi[l]  = np.exp(abs_log) * (np.cos(phase) + 1j * np.sin(phase))

        # Mirror the remainder by conjugate symmetry
        chi[half + 1:] = np.conjugate(chi[1:n - half + 1][::-1])

        chi /= (n + 1)
        pmf = np.fft.fft(chi).real

        # Numerical guards
        pmf[pmf < 0] = 0.0
        s = pmf.sum()
        if not np.isclose(s, 1.0):
            pmf /= s
        return pmf

    def _check_k(self, k):
        if not (isinstance(k, (int, np.integer)) and 0 <= k <= self.n):
            raise ValueError("k must be an integer in [0, n].")

    def _check_prob(self):
        if self.p.ndim != 1:
            raise ValueError("p must be 1D.")
        if np.any(self.p < 0) or np.any(self.p > 1):
            raise ValueError("probabilities must lie in [0, 1].")


# =============================================================================
# Utilities
# =============================================================================
def _adj_from_matrix(A: np.ndarray) -> list[set[int]]:
    """
    Row adjacency list (countries → set of products).
    A must be binary (0/1) of shape (N_countries, M_products).
    """
    return [set(np.where(A[i])[0]) for i in range(A.shape[0])]


def _triplet_counts(adj: list[set[int]], include_absent: bool = True) -> dict[int, dict[int, dict[int, int]]]:
    """
    Observed triple V-motif multiplicities for all country triplets.

    Returns
    -------
    counts : dict
        counts[i][j][k] = |N(i) ∩ N(j) ∩ N(k)| for all i < j < k.
        If include_absent is True (default) we include zero counts too,
        so **every** triplet obtains a p-value.
    """
    out: dict[int, dict[int, dict[int, int]]] = {}
    n = len(adj)
    for i in range(n):
        for j in range(i + 1, n):
            ij = adj[i] & adj[j]
            if not include_absent and not ij:
                continue
            for k in range(j + 1, n):
                c = len(ij & adj[k])
                if include_absent or c:
                    out.setdefault(i, {}).setdefault(j, {})[k] = c
    return out


# =============================================================================
# Parallel worker (uses process globals for minimal pickling)
# =============================================================================
_Pmat   = None   # NxM probability matrix
_counts = None   # observed triplet counts dict
_method = None   # p-value method string

def _init_globals(P: np.ndarray, counts: dict, method: str):
    """
    Initializer for ProcessPoolExecutor workers.
    """
    global _Pmat, _counts, _method
    _Pmat   = P
    _counts = counts
    _method = method


def _worker(task: tuple[int, int, int]):
    """
    Compute one p-value for the triplet (i, j, k).
    Uses globals set by _init_globals.
    """
    i, j, k = task
    v_obs = _counts[i][j][k]
    if v_obs == 0:
        return i, j, k, 1.0     # P{S ≥ 0} = 1, skip all computations
    probs = _Pmat[i] * _Pmat[j] * _Pmat[k]   # q_α = P_{iα} P_{jα} P_{kα}
    mu    = probs.sum()

    if _method == "poisson":
        # Upper tail of Poisson(μ)
        return i, j, k, float(poisson.sf(v_obs - 1, mu))

    if _method == "normal":
        # CLT with continuity correction (+0.5)
        var = float((probs * (1.0 - probs)).sum())
        if var <= 0.0:
            # Degenerate: if v_obs ≤ μ then P{S ≥ v_obs} = 1 else 0
            return i, j, k, (1.0 if v_obs <= mu else 0.0)
        z = (v_obs + 0.5 - mu) / math.sqrt(var)
        return i, j, k, float(norm.sf(z))

    if _method == "rna":
        # Refined normal approximation (skewness correction)
        var_arr = probs * (1.0 - probs)
        sigma   = float(np.sqrt(var_arr.sum()))
        if sigma == 0.0:
            return i, j, k, (1.0 if v_obs <= mu else 0.0)
        gamma   = float((var_arr * (1.0 - 2.0 * probs)).sum() / (sigma ** 3))
        z       = (v_obs + 0.5 - mu) / sigma
        # First get CDF with skewness correction, then convert to upper tail
        cdf = float(norm.cdf(z) + gamma * (1.0 - z**2) * norm.pdf(z) / 6.0)
        cdf = min(max(cdf, 0.0), 1.0)
        return i, j, k, 1.0 - cdf

    if _method == "poibin":
        # Exact Poisson–binomial tail P{S ≥ v_obs}
        return i, j, k, float(PoiBin(probs).pval(v_obs))

    raise ValueError("method must be one of: 'poisson', 'normal', 'rna', 'poibin'.")


# =============================================================================
# Public validator
# =============================================================================
class TripletValidator:
    """
    Validate country triplets (three-stars / triple V-motifs) against a null model.

    Parameters
    ----------
    A : ndarray of shape (N, M), dtype {0,1}
        Binary bi-adjacency (rows: countries, cols: products).
    P : ndarray of shape (N, M)
        Model probability matrix aligned with A (same shape).
    n_jobs : int, default 4
        Number of parallel worker processes. Use -1 for all cores.
    progress : bool, default True
        Show a progress bar when computing p-values.
    include_absent : bool, default True
        If True, assign a p-value to every triplet i<j<k (including those
        with observed count 0). This matches the behavior of the
        “old” code and standard multiple-testing practice.
    """
    def __init__(self,
                 A: np.ndarray,
                 P: np.ndarray,
                 n_jobs: int = 4,
                 progress: bool = True,
                 include_absent: bool = True):

        self.A        = (A > 0).astype(np.uint8)
        self.P        = np.asarray(P, dtype=float)
        self.n_jobs   = max(1, (os.cpu_count() if n_jobs == -1 else int(n_jobs)))
        self.progress = bool(progress)

        # Basic checks
        if self.A.shape != self.P.shape:
            raise ValueError("A and P must have the same shape (N x M).")
        if self.P.min() < 0.0 or self.P.max() > 1.0:
            raise ValueError("P must contain probabilities in [0, 1].")

        # Observed counts & adjacency
        self.adj    = _adj_from_matrix(self.A)
        self.counts = _triplet_counts(self.adj, include_absent=include_absent)

    # .........................................................................
    def compute_pvals(self, method: str = "poibin") -> dict[int, dict[int, dict[int, float]]]:
        """
        Compute p-values for all triplets i<j<k under the requested method.

        Returns
        -------
        pvals : dict
            Nested dict pvals[i][j][k] = p-value for i<j<k.
        """
        tasks = [(i, j, k)
                 for i in self.counts
                 for j in self.counts[i]
                 for k in self.counts[i][j]]
        out: dict[int, dict[int, dict[int, float]]] = {}

        # Single-process path
        if self.n_jobs == 1:
            _init_globals(self.P, self.counts, method)
            iterator = tqdm(tasks) if self.progress else tasks
            for t in iterator:
                i, j, k, p = _worker(t)
                out.setdefault(i, {}).setdefault(j, {})[k] = float(p)
            return out

        # Multi-process path
        with ProcessPoolExecutor(max_workers=self.n_jobs,
                         initializer=_init_globals,
                         initargs=(self.P, self.counts, method)) as ex:
            chunk = 5_000
            iterator = ex.map(_worker, tasks, chunksize=chunk)
            if self.progress:
                iterator = tqdm(iterator, total=len(tasks))
            for i, j, k, p in iterator:
                out.setdefault(i, {}).setdefault(j, {})[k] = float(p)
        return out

    # .........................................................................
    @staticmethod
    def _threshold(pvals: np.ndarray, alpha: float, kind: str) -> float:
        """
        Multiple-testing threshold.

        kind ∈ {"global", "bonferroni", "fdr"}:
            - global     : α
            - bonferroni : α / m
            - fdr (BH)   : largest p_k with p_(k) ≤ (k/m) α
        """
        p = np.asarray(pvals, dtype=float)
        p = p[np.isfinite(p)]
        m = p.size
        if m == 0:
            return 0.0
        if kind == "global":
            return alpha
        if kind == "bonferroni":
            return alpha / m
        if kind == "fdr":
            p.sort()
            crit = alpha * np.arange(1, m + 1) / m
            idx  = np.where(p <= crit)[0]
            return float(p[idx[-1]]) if idx.size else 0.0
        raise ValueError("kind must be 'global', 'bonferroni', or 'fdr'.")

    def validate(self,
                 pvals_dict: dict[int, dict[int, dict[int, float]]],
                 alpha: float = 0.05,
                 method: str = "fdr",
                 names: list[str] | None = None) -> tuple[pd.DataFrame, float]:
        """
        Validate triplets at level α with the chosen multiple-testing rule.

        Returns
        -------
        df : DataFrame
            Columns: ('i', 'j', 'k', 'p') (indices or names).
        thr : float
            The p-value threshold actually used.
        """
        rec: list[tuple[int, int, int, float]] = []
        for i in pvals_dict:
            for j in pvals_dict[i]:
                for k, p in pvals_dict[i][j].items():
                    if p is None or not np.isfinite(p):
                        continue
                    rec.append((i, j, k, float(p)))

        if not rec:
            return pd.DataFrame(columns=["i", "j", "k", "p"]), 0.0

        flat = np.fromiter((r[-1] for r in rec), dtype=float)
        thr  = self._threshold(flat, alpha, method)
        df   = pd.DataFrame(rec, columns=["i", "j", "k", "p"])
        df   = df.query("p <= @thr").reset_index(drop=True)

        if names is not None:
            rep = {idx: nm for idx, nm in enumerate(names)}
            df[["i", "j", "k"]] = df[["i", "j", "k"]].replace(rep)

        return df, float(thr)


# -----------------------------------------------------------------------------
# Small stdlib import needed late to avoid shadowing by user environments
# -----------------------------------------------------------------------------
import os  # placed at the end intentionally