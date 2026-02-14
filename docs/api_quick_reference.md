# TRIDENT API quick reference

This page lists the public API that most users interact with.

## Solvers

### BiCM

```python
from trident import bicm_solver, probability_matrix_from_bicm

res = bicm_solver(A)
P   = probability_matrix_from_bicm(res)
```

Returned dictionary keys (common):

- `x`, `y`: fitted exponential parameters
- `P`: probability matrix (if `return_probability_matrix=True`)
- `k_obs`, `h_obs`: observed degrees in the two layers
- `k_exp`, `h_exp`: expected degrees

### BiPLOM

```python
from trident import biplom_solver, probability_matrix_from_biplom

res = biplom_solver(A)
P   = probability_matrix_from_biplom(res)
```

The BiPLOM solver returns fitted parameters and diagnostics; the exact
structure depends on the underlying implementation, but it always
contains the parameters required to reconstruct the probability matrix.

## Validators

### Pairs (two-star, V-motifs)

```python
from trident import TwoStarValidator

val = TwoStarValidator(A, P, n_jobs=1)
pvals = val.compute_pvals(method="normal")
df, thr = val.validate(pvals, alpha=0.05, method="fdr")
```

### Triplets (three-star)

```python
from trident import TripletValidator

val = TripletValidator(A, P, n_jobs=1)
pvals = val.compute_pvals(method="normal")
df, thr = val.validate(pvals, alpha=0.05, method="fdr")
```

## Methods for p-values

Both validators accept:

- `"poisson"`: Poisson approximation
- `"normal"`: Gaussian approximation with continuity correction
- `"rna"`: refined normal approximation (skewness correction)
- `"poibin"`: Poisson-binomial via FFT
