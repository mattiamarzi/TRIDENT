"""TRIDENT

TRIDENT (TRIadic DEtectioN via staTistical validation) provides tools to
infer statistically validated pairwise and triadic interactions from
binary bipartite data using maximum-entropy null models.

The public API is intentionally small:

- `bicm_solver`, `probability_matrix_from_bicm`
- `biplom_solver`, `probability_matrix_from_biplom`
- `TwoStarValidator` (pairs) and `TripletValidator` (triplets)
"""

from .models.BiCM import bicm_solver, probability_matrix_from_bicm
from .models.BiPLOM.BiPLOM import biplom_solver, probability_matrix_from_biplom
from .validators.pairs import TwoStarValidator
from .validators.triplets import TripletValidator

__all__ = [
    "bicm_solver",
    "probability_matrix_from_bicm",
    "biplom_solver",
    "probability_matrix_from_biplom",
    "TwoStarValidator",
    "TripletValidator",
]

__version__ = "0.1.0"
