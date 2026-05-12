"""Diagnostics for marginal and conditional velocity-field analyses."""

from intro_fm_diffusion.diagnostics.conditional_field import (
    conditional_field_mse_by_time,
    global_conditional_field_mse,
)
from intro_fm_diffusion.diagnostics.decomposition import posterior_irreducible_variance_by_time
from intro_fm_diffusion.diagnostics.marginal_field import (
    marginal_field_mse_by_time,
    marginal_velocity_grid_comparison,
)

__all__ = [
    "marginal_velocity_grid_comparison",
    "marginal_field_mse_by_time",
    "global_conditional_field_mse",
    "conditional_field_mse_by_time",
    "posterior_irreducible_variance_by_time",
]
