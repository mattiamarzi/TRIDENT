# TRIDENT

**TRIDENT** (TRIplets DEtectioN via staTistical validation) is a Python package
for the statistical validation of pairwise and triadic interactions in binary
bipartite data.

TRIDENT implements a maximum-entropy validation pipeline:

- **BiCM** (Bipartite Configuration Model) for validating pairs,
- **BiPLOM** (Bipartite Partial Local Overlap Model) for validating triplets,
- Poisson-binomial p-values (exact or approximated), with multiple-testing
  correction (Bonferroni or FDR, with FDR recommended) to produce validated
  networks and hypergraphs.

The methodology is described in the companion preprint
*Signatures of higher-order cooperation/coordination in the human brain*.

## Installation

```bash
pip install trident
```

For development:

```bash
git clone <repo-url>
cd TRIDENT
pip install -e ".[dev]"
pytest -q
```

## Quick start

### Validate pairs (BiCM)

```python
import numpy as np

from trident import bicm_solver, probability_matrix_from_bicm
from trident import TwoStarValidator

A = ...  # binary biadjacency matrix (N x M)

res = bicm_solver(A)
P   = probability_matrix_from_bicm(res)

val = TwoStarValidator(A, P, n_jobs=1)
pvals = val.compute_pvals(method="normal")
df_pairs, thr = val.validate(pvals, alpha=0.05, method="fdr")
```

### Validate triplets (BiPLOM)

```python
from trident import biplom_solver, probability_matrix_from_biplom
from trident import TripletValidator

res = biplom_solver(A)
P   = probability_matrix_from_biplom(res)

val = TripletValidator(A, P, n_jobs=1)
pvals = val.compute_pvals(method="normal")
df_triplets, thr = val.validate(pvals, alpha=0.05, method="fdr")
```

## Documentation

See the `docs/` folder:

- `docs/api_quick_reference.md`
- `docs/math.md`
- `docs/usage.md`

## Acknowledgements and citations

TRIDENT builds on and extends maximum-entropy methods for bipartite networks and statistically validated projections.

If you use TRIDENT, please also consider citing the reference BiCM implementation and the following foundational works:

- Reference implementation (BiCM module): https://github.com/mat701/BiCM
- N. Vallarano, M. Bruno, E. Marchese, G. Trapani, F. Saracco, T. Squartini, G. Cimini, M. Zanon, "Fast and scalable likelihood maximization for Exponential Random Graph Models with local constraints", Scientific Reports (2021).
- F. Saracco, R. Di Clemente, A. Gabrielli, T. Squartini, "Randomizing bipartite networks: the case of the World Trade Web", Scientific Reports 5, 10595 (2015).
- F. Saracco, M. J. Straka, R. Di Clemente, A. Gabrielli, G. Caldarelli, T. Squartini, "Inferring monopartite projections of bipartite networks: an entropy-based approach", New Journal of Physics 19, 053022 (2017).

## License

MIT (see `LICENSE`).
