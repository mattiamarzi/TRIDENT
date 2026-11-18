"""Independent references for inclusive motif-count tails."""

import math

import numpy as np
import pytest
from scipy.stats import binom

from trident import TripletValidator, TwoStarValidator
from trident.models.BiPLOM.bipclass import BipartiteGraph1


VALIDATORS = [(TwoStarValidator, 2), (TripletValidator, 3)]


def _single_tuple(cls, order, trials, observed, q):
    """Construct one motif count with a known Binomial(trials, q) null law."""
    a = np.zeros((order, trials), dtype=np.uint8)
    a[:, :observed] = 1
    p = np.full(a.shape, q ** (1.0 / order))
    return cls(a, p, n_jobs=1, progress=False)


def _only_value(result):
    while isinstance(result, dict):
        assert len(result) == 1
        result = next(iter(result.values()))
    return result


@pytest.mark.parametrize("cls,order", VALIDATORS)
@pytest.mark.parametrize("method", ["normal", "rna"])
@pytest.mark.parametrize("engine", ["auto", "loop", "vectorized"])
@pytest.mark.parametrize("observed", [50, 51])
def test_symmetric_binomial_inclusive_tail(cls, order, method, engine, observed):
    val = _single_tuple(cls, order, 100, observed, 0.5)
    actual = _only_value(val.compute_pvals(method, engine=engine))
    # Symmetry gives the exact mass at 50 without a probability-library oracle.
    center_mass = math.comb(100, 50) / 2**100
    exact = (1 + center_mass) / 2 if observed == 50 else (1 - center_mass) / 2
    reference = 0.539827837277029 if observed == 50 else 0.460172162722971
    assert actual == pytest.approx(reference, rel=1e-10, abs=1e-14)
    # This is an approximation-error bound for this analytic example, not a
    # relaxed floating-point comparison or a universal normal-error guarantee.
    assert abs(actual - exact) < 5e-5
    assert _only_value(val.compute_pvals("poibin")) == pytest.approx(
        exact, rel=1e-10, abs=1e-14
    )


@pytest.mark.parametrize("cls,order", VALIDATORS)
@pytest.mark.parametrize("method", ["normal", "rna"])
@pytest.mark.parametrize("engine", ["loop", "vectorized"])
def test_skewed_binomial_reference(cls, order, method, engine):
    trials, q, observed = 200, 0.125, 27
    val = _single_tuple(cls, order, trials, observed, q)
    mu = trials * q
    variance = trials * q * (1 - q)
    z = (observed - 0.5 - mu) / math.sqrt(variance)
    # erfc supplies an independent normal-tail evaluation; the skewness term
    # is evaluated from the analytic binomial moments, not package helpers.
    reference = math.erfc(z / math.sqrt(2)) / 2
    if method == "rna":
        skewness = (1 - 2*q) / math.sqrt(variance)
        density = math.exp(-z*z / 2) / math.sqrt(2*math.pi)
        reference -= skewness * (1 - z*z) * density / 6
    actual = _only_value(val.compute_pvals(method, engine=engine))
    assert actual == pytest.approx(reference, rel=1e-10, abs=1e-14)
    # The exact tail is an additional check on direction and skewness sign.
    bound = 0.01 if method == "normal" else 0.001
    assert abs(actual - binom.sf(observed - 1, trials, q)) < bound


@pytest.mark.parametrize("cls,order", VALIDATORS)
@pytest.mark.parametrize("method", ["normal", "rna"])
@pytest.mark.parametrize("engine", ["loop", "vectorized"])
@pytest.mark.parametrize("q,observed,expected", [
    (0.0, 0, 1.0), (0.5, 0, 1.0), (0.0, 1, 0.0), (1.0, 10, 1.0),
])
def test_zero_count_and_deterministic_tails(cls, order, method, engine, q, observed, expected):
    val = _single_tuple(cls, order, 10, observed, q)
    assert _only_value(val.compute_pvals(method, engine=engine)) == expected


@pytest.mark.parametrize("method", ["poisson", "normal", "rna", "poibin"])
@pytest.mark.parametrize("q,observed", [(0.5, 50), (0.125, 27), (0.0, 0), (0.0, 1), (1.0, 100)])
def test_legacy_projection_matches_triplet_validator(method, q, observed):
    trials = 200 if q == 0.125 else 100
    probability = q ** (1/3)
    graph = BipartiteGraph1()
    graph.V = np.ones(3)
    graph.cols_deg = np.zeros(trials)
    graph.v_adj_list = {0: {1: {2: observed}}}
    graph.projection_method = method
    # Very large finite odds give exactly deterministic probabilities without
    # undefined infinite fitnesses in the legacy probability parameterization.
    odds = probability / (1-probability) if probability < 1 else 1e20
    actual = graph.pval_calculator(0, np.full(3, odds), np.ones(trials), graph.V)[1][2]
    val = _single_tuple(TripletValidator, 3, trials, observed, q)
    expected = _only_value(val.compute_pvals(method, engine="loop"))
    assert actual == pytest.approx(expected, rel=1e-10, abs=1e-14)
