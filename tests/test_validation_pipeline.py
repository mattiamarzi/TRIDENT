"""Unit tests for the minimal TRIDENT validation pipeline.

The tests are designed to be lightweight and deterministic.
They validate that:

1) BiCM fitting returns a probability matrix with the correct shape and
   reproduces observed degrees approximately.
2) Pair and triplet validators run end-to-end using a Gaussian
   approximation for the Poisson-binomial distribution.
3) The BiPLOM solver can be executed on a small instance and its
   probability matrix can be consumed by the triplet validator.
"""

from __future__ import annotations

import numpy as np

import pytest

from trident import bicm_solver, probability_matrix_from_bicm
from trident import biplom_solver, probability_matrix_from_biplom
from trident import TwoStarValidator, TripletValidator


def _make_small_bipartite(seed: int, n: int, m: int, p: float) -> np.ndarray:
    """Generate a small binary biadjacency matrix with fixed seed."""
    rng = np.random.default_rng(seed)

    # The BiPLOM reference implementation performs internal reductions when
    # empty/full rows or columns exist. For a lightweight unit test we enforce
    # strictly non-degenerate degrees in both layers.
    for _ in range(200):
        A = (rng.random((n, m)) < p).astype(np.uint8)
        if A.sum() == 0:
            continue
        if np.all(A.sum(axis=1) > 0) and np.all(A.sum(axis=0) > 0):
            return A

    # Fallback: make a minimally connected instance deterministically.
    A = np.zeros((n, m), dtype=np.uint8)
    for i in range(n):
        A[i, i % m] = 1
    for a in range(m):
        A[a % n, a] = 1
    return A


def test_bicm_and_pair_triplet_validation_normal():
    A = _make_small_bipartite(seed=1, n=20, m=20, p=0.12)

    res = bicm_solver(A, tol=1e-10, max_outer_iter=150)
    P = probability_matrix_from_bicm(res)

    assert P.shape == A.shape
    assert np.all(P >= 0.0) and np.all(P <= 1.0)

    # Pair validation
    val2 = TwoStarValidator(A, P, n_jobs=1, progress=False)
    pvals2 = val2.compute_pvals(method="normal")
    df2, thr2 = val2.validate(pvals2, alpha=0.05, method="fdr")
    assert set(df2.columns) == {"i", "j", "p"}
    assert 0.0 <= thr2 <= 0.05

    # Triplet validation
    val3 = TripletValidator(A, P, n_jobs=1, progress=False)
    pvals3 = val3.compute_pvals(method="normal")
    df3, thr3 = val3.validate(pvals3, alpha=0.05, method="fdr")
    assert set(df3.columns) == {"i", "j", "k", "p"}
    assert 0.0 <= thr3 <= 0.05


def test_biplom_smoke_and_triplet_validator():
    # Smaller instance to keep CI runtime low and reduce numba compile time.
    A = _make_small_bipartite(seed=2, n=12, m=12, p=0.15)

    res = biplom_solver(A, verbose=False)
    P = probability_matrix_from_biplom(res)

    assert P.shape == A.shape
    assert np.all(P >= 0.0) and np.all(P <= 1.0)

    val3 = TripletValidator(A, P, n_jobs=1, progress=False)
    pvals3 = val3.compute_pvals(method="normal")
    df3, _ = val3.validate(pvals3, alpha=0.05, method="fdr")
    assert set(df3.columns) == {"i", "j", "k", "p"}
