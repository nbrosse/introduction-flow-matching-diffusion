"""Notebook-level analysis facades built on diagnostics modules."""

from typing import Literal

import numpy as np
import torch

from intro_fm_diffusion.diagnostics.conditional_field import (
    conditional_field_mse_by_time,
    global_conditional_field_mse,
)
from intro_fm_diffusion.diagnostics.decomposition import posterior_irreducible_variance_by_time
from intro_fm_diffusion.diagnostics.marginal_field import (
    marginal_field_mse_by_time,
    marginal_velocity_grid_comparison,
)
from intro_fm_diffusion.notebook.training import resolve_device
from intro_fm_diffusion.paths.paths_gaussian import GaussianConditionalProbabilityPath
from intro_fm_diffusion.training import MLPVelocityField


def analyze_cfm_decomposition_by_time(
    path: GaussianConditionalProbabilityPath,
    velocity_model: MLPVelocityField,
    batch_size: int = 1000,
    num_batches: int = 100,
    num_posterior_samples: int = 1000,
    num_time_bins: int = 20,
    t_min: float = 0.0,
    t_max: float = 0.95,
    variance_num_samples: int = 100,
    variance_num_time_steps: int = 20,
    posterior_method: Literal["auto", "exact", "monte_carlo"] = "auto",
    device: torch.device | str | None = None,
) -> dict:
    """Build the full CFM decomposition bundle for plotting."""
    resolved_device = resolve_device(device=device, model=velocity_model)
    path = path.to(resolved_device)
    velocity_model = velocity_model.to(resolved_device)

    return {
        "marginal_by_time": marginal_field_mse_by_time(
            velocity_model=velocity_model,
            path=path,
            device=resolved_device,
            batch_size=batch_size,
            num_batches=num_batches,
            num_posterior_samples=num_posterior_samples,
            num_time_bins=num_time_bins,
            t_min=t_min,
            t_max=t_max,
            posterior_method=posterior_method,
        ),
        "conditional_by_time": conditional_field_mse_by_time(
            velocity_model=velocity_model,
            path=path,
            device=resolved_device,
            batch_size=batch_size,
            num_batches=num_batches,
            num_time_bins=num_time_bins,
            t_min=t_min,
            t_max=t_max,
        ),
        "irreducible_variance_by_time": posterior_irreducible_variance_by_time(
            path=path,
            device=resolved_device,
            num_samples=variance_num_samples,
            num_time_steps=variance_num_time_steps,
            num_posterior_samples=num_posterior_samples,
            t_max=t_max,
            posterior_method=posterior_method,
        ),
    }


def analyze_training_plateau_vs_global_cfm(
    path: GaussianConditionalProbabilityPath,
    velocity_model: MLPVelocityField,
    training_losses: list[float] | np.ndarray,
    plateau_window: int = 200,
    global_batch_size: int = 1000,
    global_num_batches: int = 200,
    global_t_min: float = 0.0,
    global_t_max: float = 1.0,
    device: torch.device | str | None = None,
) -> dict:
    """Compare late training plateau with an empirical global CFM estimate."""
    resolved_device = resolve_device(device=device, model=velocity_model)
    path = path.to(resolved_device)
    velocity_model = velocity_model.to(resolved_device)

    losses = np.asarray(training_losses, dtype=float)
    if losses.size == 0:
        raise ValueError("training_losses must not be empty.")

    resolved_window = min(int(plateau_window), losses.size)
    tail_losses = losses[-resolved_window:]
    return {
        "final_loss": float(losses[-1]),
        "plateau_window": resolved_window,
        "plateau_mean": float(tail_losses.mean()),
        "plateau_std": float(tail_losses.std()),
        "global_conditional_mse": global_conditional_field_mse(
            velocity_model=velocity_model,
            path=path,
            device=resolved_device,
            batch_size=global_batch_size,
            num_batches=global_num_batches,
            t_min=global_t_min,
            t_max=global_t_max,
        ),
    }


def analyze_velocity_fields_on_grid(
    path: GaussianConditionalProbabilityPath,
    velocity_model: MLPVelocityField,
    scale: float = 15.0,
    num_bins: int = 30,
    num_time_steps: int = 20,
    num_posterior_samples: int = 1000,
    posterior_method: Literal["auto", "exact", "monte_carlo"] = "auto",
    device: torch.device | str | None = None,
) -> dict:
    """Compute velocity-field comparison data for plotting."""
    resolved_device = resolve_device(device=device, model=velocity_model)
    path = path.to(resolved_device)
    velocity_model = velocity_model.to(resolved_device)
    return marginal_velocity_grid_comparison(
        velocity_model=velocity_model,
        path=path,
        device=resolved_device,
        scale=scale,
        num_bins=num_bins,
        num_time_steps=num_time_steps,
        num_posterior_samples=num_posterior_samples,
        posterior_method=posterior_method,
    )
