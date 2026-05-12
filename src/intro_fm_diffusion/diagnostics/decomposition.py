"""Decomposition diagnostics for conditional flow matching."""

from typing import Literal

import numpy as np
import torch
from tqdm import tqdm

from intro_fm_diffusion.paths.paths_gaussian import GaussianConditionalProbabilityPath


def posterior_irreducible_variance_by_time(
    path: GaussianConditionalProbabilityPath,
    device: torch.device,
    num_samples: int = 100,
    num_time_steps: int = 20,
    num_posterior_samples: int = 1000,
    t_max: float = 0.95,
    posterior_method: Literal["auto", "exact", "monte_carlo"] = "auto",
) -> dict:
    """Estimate irreducible conditional variance as a function of time."""
    time_values = torch.linspace(0.01, t_max, num_time_steps).to(device)
    variances = []
    resolved_posterior_method = path.resolve_posterior_method(posterior_method)

    for t_val in tqdm(time_values, desc=f"Computing irreducible variance ({resolved_posterior_method})"):
        t = t_val.view(1, 1).expand(num_samples, 1)
        x = path.sample_marginal_path(t)
        batch_variances = path.irreducible_variance(
            x,
            t,
            num_posterior_samples=num_posterior_samples,
            method=resolved_posterior_method,
        )
        variances.append(batch_variances.mean().item())

    return {
        "time_values": time_values.cpu().numpy(),
        "variances": np.array(variances),
        "mean_variance": float(np.mean(variances)),
        "posterior_method": resolved_posterior_method,
    }
