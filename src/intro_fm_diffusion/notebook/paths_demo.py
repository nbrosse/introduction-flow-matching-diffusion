"""Demo path builders used by final notebooks."""

import torch

from intro_fm_diffusion.density import GaussianMixture
from intro_fm_diffusion.paths.paths_gaussian import GaussianConditionalProbabilityPath
from intro_fm_diffusion.schedules import LinearAlpha, SquareRootBeta
from intro_fm_diffusion.utils import get_device


def build_symmetric_gaussian_mixture_path(
    nmodes: int = 5,
    target_std: float = 1.0,
    target_scale: float = 10.0,
    device: torch.device | str | None = None,
) -> GaussianConditionalProbabilityPath:
    """Build the canonical 2D Gaussian-mixture conditional path."""
    path = GaussianConditionalProbabilityPath(
        p_data=GaussianMixture.symmetric_2D(
            nmodes=nmodes,
            std=target_std,
            scale=target_scale,
        ),
        alpha=LinearAlpha(),
        beta=SquareRootBeta(),
    )
    return path.to(get_device(device))


def build_demo_gaussian_path(
    nmodes: int = 5,
    target_std: float = 1.0,
    target_scale: float = 10.0,
    device: torch.device | str | None = None,
) -> GaussianConditionalProbabilityPath:
    """Alias for the default demo path used in notebooks."""
    return build_symmetric_gaussian_mixture_path(
        nmodes=nmodes,
        target_std=target_std,
        target_scale=target_scale,
        device=device,
    )
