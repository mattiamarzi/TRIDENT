"""Model solvers shipped with TRIDENT."""

from .BiCM import bicm_solver, probability_matrix_from_bicm
try:
    from .BiPLOM.BiPLOM import biplom_solver, probability_matrix_from_biplom
except ModuleNotFoundError:  # pragma: no cover
    def biplom_solver(*_args, **_kwargs):
        """Raise an informative error if the optional BiPLOM dependency is missing."""

        raise ImportError(
            "The BiPLOM solver requires the optional 'bicm' dependency, "
            "which is not available in this environment."
        )

    def probability_matrix_from_biplom(*_args, **_kwargs):
        """Raise an informative error if the optional BiPLOM dependency is missing."""

        raise ImportError(
            "The BiPLOM solver requires the optional 'bicm' dependency, "
            "which is not available in this environment."
        )

__all__ = [
    "bicm_solver",
    "probability_matrix_from_bicm",
    "biplom_solver",
    "probability_matrix_from_biplom",
]
