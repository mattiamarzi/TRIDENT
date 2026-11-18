"""The vectorized p-values must match the loop validators.

For each method with a vectorized path ("poisson", "normal", "rna") and for
pairs and triplets: same keys, p-values equal up to floating-point rounding,
and the same set validated with the Benjamini–Hochberg rule. Includes
degenerate probabilities (silent regions, P = 0 and P = 1 entries) and
include_absent=False.
"""

from __future__ import annotations

import numpy as np
import pytest

from trident import TripletValidator, TwoStarValidator, bicm_solver, probability_matrix_from_bicm


def _flat(d):
    out = {}
    for i, di in d.items():
        for j, dj in di.items():
            if isinstance(dj, dict):
                for k, p in dj.items():
                    out[(i, j, k)] = p
            else:
                out[(i, j)] = dj
    return out


def _compare(validator, method):
    loop = _flat(validator.compute_pvals(method=method, engine="loop"))
    vec = _flat(validator.compute_pvals(method=method, engine="vectorized"))
    assert list(loop) == list(vec)
    a, b = np.array(list(loop.values())), np.array(list(vec.values()))
    assert np.max(np.abs(a - b)) < 1e-10
    keys = np.array(list(loop), dtype=object)
    ta, tb = validator._threshold(a, 0.05, "fdr"), validator._threshold(b, 0.05, "fdr")
    assert set(map(tuple, keys[a <= ta])) == set(map(tuple, keys[b <= tb]))


def _correlated(seed=1, n=14, m=500):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((m, 3)) @ rng.standard_normal((3, n)) + rng.standard_normal((m, n))
    A = (np.abs((z - z.mean(0)) / z.std(0)) > 1).T.astype(np.uint8)
    return A[:, A.sum(0) > 0]


@pytest.mark.parametrize("method", ["poisson", "normal", "rna"])
def test_fitted_bicm(method):
    A = _correlated()
    P = probability_matrix_from_bicm(bicm_solver(A, tol=1e-10, max_outer_iter=300))
    _compare(TwoStarValidator(A, P, n_jobs=1, progress=False), method)
    _compare(TripletValidator(A, P, n_jobs=1, progress=False), method)


@pytest.mark.parametrize("method", ["poisson", "normal", "rna"])
def test_degenerate_probabilities(method):
    rng = np.random.default_rng(2)
    n, m = 9, 120
    A = (rng.random((n, m)) < 0.3).astype(np.uint8)
    A[0] = 0                                   # silent region: V = 0
    P = rng.random((n, m))
    P[1] = 0.0                                 # zero mean and variance
    P[2, :40] = 1.0
    P[3] = 1.0                                 # q = 1 for some tuples: zero variance
    _compare(TwoStarValidator(A, P, n_jobs=1, progress=False), method)
    _compare(TripletValidator(A, P, n_jobs=1, progress=False), method)


def test_include_absent_false():
    A = _correlated(seed=3, n=10, m=200)
    A[:, :] = A * (np.random.default_rng(4).random(A.shape) < 0.5)
    P = probability_matrix_from_bicm(bicm_solver(A, tol=1e-10, max_outer_iter=300))
    _compare(TwoStarValidator(A, P, n_jobs=1, progress=False, include_absent=False), "rna")
    _compare(TripletValidator(A, P, n_jobs=1, progress=False, include_absent=False), "rna")


def test_engine_choice():
    A = _correlated(seed=5, n=6, m=80)
    P = np.clip(A.mean(1, keepdims=True) * np.ones_like(A, dtype=float), 0.01, 0.99)
    v = TwoStarValidator(A, P, n_jobs=1, progress=False)
    with pytest.raises(ValueError):
        v.compute_pvals(method="poibin", engine="vectorized")
    with pytest.raises(ValueError):
        v.compute_pvals(method="rna", engine="fast")
    assert _flat(v.compute_pvals(method="poibin")) == _flat(v.compute_pvals(method="poibin", engine="loop"))


@pytest.mark.parametrize("include_absent", [True, False])
def test_counts_match_set_based(include_absent):
    from trident.validators.pairs import _adj_from_matrix, _pair_counts
    from trident.validators.triplets import _triplet_counts

    A = _correlated(seed=6, n=12, m=150)
    A[2] = 0                                   # a silent region
    adj = _adj_from_matrix(A)
    v2 = TwoStarValidator(A, np.full(A.shape, 0.3), n_jobs=1, progress=False, include_absent=include_absent)
    v3 = TripletValidator(A, np.full(A.shape, 0.3), n_jobs=1, progress=False, include_absent=include_absent)
    ref2, ref3 = _pair_counts(adj, include_absent), _triplet_counts(adj, include_absent)
    assert v2.counts == ref2 and list(_flat(v2.counts)) == list(_flat(ref2))
    assert v3.counts == ref3 and list(_flat(v3.counts)) == list(_flat(ref3))
