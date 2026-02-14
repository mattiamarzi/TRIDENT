# Usage

This page assumes TRIDENT is installed as

```bash
pip install trident
```

TRIDENT expects a binary bipartite biadjacency matrix $A$ of shape $(N, M)$.
Rows and columns can represent any two-mode system (for example, users-items,
regions-time, firms-products).

## Validate pairs with BiCM

```python
import numpy as np

from trident import bicm_solver, probability_matrix_from_bicm
from trident import TwoStarValidator

# A: binary biadjacency (N x M)
A = ...

# Fit BiCM and obtain probabilities
res = bicm_solver(A)
P   = probability_matrix_from_bicm(res)

# Compute p-values for all pairs i<j
validator = TwoStarValidator(A, P, n_jobs=1)
pvals = validator.compute_pvals(method="normal")

# Multiple-testing correction and validated pairs
df_pairs, thr = validator.validate(pvals, alpha=0.05, method="fdr")
print(df_pairs.head())
```

## Validate triplets with BiPLOM

BiPLOM is a second-order null model that controls for both node heterogeneity
and local pairwise co-activation tendencies, providing a stricter baseline
for triadic validation.

```python
from trident import biplom_solver, probability_matrix_from_biplom
from trident import TripletValidator

res = biplom_solver(A)
P   = probability_matrix_from_biplom(res)

validator = TripletValidator(A, P, n_jobs=1)
pvals = validator.compute_pvals(method="normal")

df_triplets, thr = validator.validate(pvals, alpha=0.05, method="fdr")
print(df_triplets.head())
```

## Multiple-testing correction

When validating pairs or triplets, TRIDENT applies a correction for multiple
hypothesis testing through the `method` argument of `validate(...)`.

Available options are:

- `method="fdr"` (default, recommended):
  applies the Benjamini–Hochberg false discovery rate control. This option is
  generally preferred when validating a large number of interactions, as it
  retains higher statistical power.

- `method="bonferroni"`:
  applies Bonferroni correction, controlling the family-wise error rate. This
  option is more conservative and may severely reduce the number of validated
  interactions in large systems.

Example:

```python
df_pairs, thr = validator.validate(
    pvals,
    alpha=0.05,
    method="fdr",
)
```

The same interface applies to both pairwise and triplet validation.

## Notes on performance

- For exact p-values, use `method="poibin"`.
- For large systems, computing all triplets is combinatorially expensive.
  In such cases, users commonly restrict the search space (for example, by
  screening candidate triplets using pairwise results).

## Citing

If you use TRIDENT in academic work, please cite the TRIDENT preprint and consider citing the reference BiCM implementation and the foundational papers listed in the main README.
