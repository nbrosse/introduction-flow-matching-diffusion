"""Diagnostics for conditional target velocity fields."""

import numpy as np
import torch
from tqdm import tqdm

from intro_fm_diffusion.paths.paths_base import ConditionalProbabilityPath


def _sample_uniform_times(
    batch_size: int,
    device: torch.device,
    t_min: float,
    t_max: float,
) -> torch.Tensor:
    if t_max <= t_min:
        raise ValueError(f"Expected t_max > t_min, got t_min={t_min} and t_max={t_max}.")
    return torch.rand(batch_size, 1, device=device) * (t_max - t_min) + t_min


def global_conditional_field_mse(
    velocity_model: torch.nn.Module,
    path: ConditionalProbabilityPath,
    device: torch.device,
    batch_size: int = 1000,
    num_batches: int = 100,
    t_min: float = 0.0,
    t_max: float = 1.0,
) -> dict:
    """Estimate global conditional-target MSE over time interval [t_min, t_max]."""
    batch_losses = []
    was_training = velocity_model.training
    velocity_model.eval()
    try:
        with torch.no_grad():
            for _ in tqdm(range(num_batches), desc="Computing global conditional-target MSE"):
                z = path.sample_conditioning_variable(batch_size).to(device)
                t = _sample_uniform_times(batch_size=batch_size, device=device, t_min=t_min, t_max=t_max)
                x = path.sample_conditional_path(z, t)
                u_conditional = path.conditional_velocity_field(x, z, t)
                u_learned = velocity_model(x, t)
                batch_losses.append(torch.mean((u_learned - u_conditional) ** 2).item())
    finally:
        velocity_model.train(was_training)

    losses = np.asarray(batch_losses, dtype=float)
    return {
        "mean_loss": float(losses.mean()),
        "std_loss": float(losses.std()),
        "batch_losses": losses,
        "batch_size": batch_size,
        "num_batches": num_batches,
        "t_min": t_min,
        "t_max": t_max,
    }


def conditional_field_mse_by_time(
    velocity_model: torch.nn.Module,
    path: ConditionalProbabilityPath,
    device: torch.device,
    batch_size: int = 1000,
    num_batches: int = 100,
    num_time_bins: int = 20,
    t_min: float = 0.0,
    t_max: float = 0.95,
) -> dict:
    """Estimate conditional-target MSE curve across time bins."""
    time_edges = np.linspace(t_min, t_max, num_time_bins + 1)
    time_bins = (time_edges[:-1] + time_edges[1:]) / 2
    mean_losses = []
    std_losses = []

    was_training = velocity_model.training
    velocity_model.eval()
    try:
        with torch.no_grad():
            for bin_idx in tqdm(range(num_time_bins), desc="Computing conditional-target MSE by time"):
                t_low = time_edges[bin_idx]
                t_high = time_edges[bin_idx + 1]
                bin_losses = []
                for _ in range(num_batches):
                    z = path.sample_conditioning_variable(batch_size).to(device)
                    t = torch.rand(batch_size, 1, device=device) * (t_high - t_low) + t_low
                    x = path.sample_conditional_path(z, t)
                    u_conditional = path.conditional_velocity_field(x, z, t)
                    u_learned = velocity_model(x, t)
                    bin_losses.append(torch.mean((u_learned - u_conditional) ** 2).item())

                losses = np.asarray(bin_losses, dtype=float)
                mean_losses.append(float(losses.mean()))
                std_losses.append(float(losses.std()))
    finally:
        velocity_model.train(was_training)

    return {
        "time_bins": time_bins,
        "mean_losses": np.array(mean_losses),
        "std_losses": np.array(std_losses),
        "time_edges": time_edges,
    }
