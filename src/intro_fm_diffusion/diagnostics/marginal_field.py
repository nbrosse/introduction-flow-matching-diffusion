"""Diagnostics for the marginal target velocity field."""

import logging
from typing import Literal

import numpy as np
import torch
from tqdm import tqdm

from intro_fm_diffusion.paths.paths_gaussian import GaussianConditionalProbabilityPath

logger = logging.getLogger(__name__)


def marginal_velocity_grid_comparison(
    velocity_model: torch.nn.Module,
    path: GaussianConditionalProbabilityPath,
    device: torch.device,
    scale: float = 15.0,
    num_bins: int = 30,
    num_time_steps: int = 20,
    num_posterior_samples: int = 1000,
    posterior_method: Literal["auto", "exact", "monte_carlo"] = "auto",
) -> dict:
    """Compare learned and target marginal velocity fields on a 2D grid."""
    xs = torch.linspace(-scale, scale, num_bins).to(device)
    ys = torch.linspace(-scale, scale, num_bins).to(device)
    xx, yy = torch.meshgrid(xs, ys, indexing="ij")
    xx_flat = xx.reshape(-1, 1)
    yy_flat = yy.reshape(-1, 1)
    xy = torch.cat([xx_flat, yy_flat], dim=-1)

    time_values = torch.linspace(0.0, 0.99, num_time_steps).to(device)
    learned_velocity_fields = []
    target_velocity_fields = []
    mse_errors = []
    cosine_similarities = []

    resolved_posterior_method = path.resolve_posterior_method(posterior_method)
    logger.info(
        "Computing velocity fields for %s steps using %s posterior method.",
        num_time_steps,
        resolved_posterior_method,
    )

    was_training = velocity_model.training
    velocity_model.eval()
    try:
        with torch.no_grad():
            for t_val in tqdm(time_values):
                bs = num_bins**2
                tt = t_val.view(1, 1).expand(bs, 1)
                learned_velocity = velocity_model(xy, tt)
                target_velocity = path.marginal_velocity_field(
                    xy,
                    tt,
                    num_posterior_samples=num_posterior_samples,
                    method=resolved_posterior_method,
                )

                learned_velocity_fields.append(learned_velocity.cpu())
                target_velocity_fields.append(target_velocity.cpu())

                learned_np = learned_velocity.detach().cpu().numpy()
                target_np = target_velocity.detach().cpu().numpy()
                mse_errors.append(torch.mean((learned_velocity - target_velocity) ** 2).item())

                dot_product = np.sum(learned_np * target_np, axis=1)
                learned_norm = np.linalg.norm(learned_np, axis=1)
                target_norm = np.linalg.norm(target_np, axis=1)
                cosine_sim = dot_product / (learned_norm * target_norm + 1e-8)
                cosine_similarities.append(float(np.mean(cosine_sim)))
    finally:
        velocity_model.train(was_training)

    return {
        "learned_velocity_fields": torch.stack(learned_velocity_fields).numpy(),
        "target_velocity_fields": torch.stack(target_velocity_fields).numpy(),
        "time_values": time_values.cpu().numpy(),
        "grid": {
            "xx": xx.cpu().numpy(),
            "yy": yy.cpu().numpy(),
            "xx_flat": xx_flat.cpu().numpy(),
            "yy_flat": yy_flat.cpu().numpy(),
            "num_bins": num_bins,
        },
        "metrics": {
            "mse_errors": np.array(mse_errors),
            "cosine_similarities": np.array(cosine_similarities),
        },
        "posterior_method": resolved_posterior_method,
    }


def marginal_field_mse_by_time(
    velocity_model: torch.nn.Module,
    path: GaussianConditionalProbabilityPath,
    device: torch.device,
    batch_size: int = 1000,
    num_batches: int = 100,
    num_posterior_samples: int = 1000,
    num_time_bins: int = 20,
    t_min: float = 0.0,
    t_max: float = 0.95,
    posterior_method: Literal["auto", "exact", "monte_carlo"] = "auto",
) -> dict:
    """Estimate marginal-target MSE as a function of time."""
    time_edges = np.linspace(t_min, t_max, num_time_bins + 1)
    time_bins = (time_edges[:-1] + time_edges[1:]) / 2
    mean_losses = []
    std_losses = []
    resolved_posterior_method = path.resolve_posterior_method(posterior_method)

    was_training = velocity_model.training
    velocity_model.eval()
    try:
        with torch.no_grad():
            for bin_idx in tqdm(range(num_time_bins), desc="Computing marginal-target MSE by time"):
                t_low = time_edges[bin_idx]
                t_high = time_edges[bin_idx + 1]
                bin_losses = []
                for _ in range(num_batches):
                    t = torch.rand(batch_size, 1, device=device) * (t_high - t_low) + t_low
                    x = path.sample_marginal_path(t)
                    u_target = path.marginal_velocity_field(
                        x,
                        t,
                        num_posterior_samples=num_posterior_samples,
                        method=resolved_posterior_method,
                    )
                    u_learned = velocity_model(x, t)
                    bin_losses.append(torch.mean((u_learned - u_target) ** 2).item())

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
        "posterior_method": resolved_posterior_method,
    }
