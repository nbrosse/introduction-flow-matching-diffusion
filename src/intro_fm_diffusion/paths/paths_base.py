from abc import ABC, abstractmethod

import torch

from intro_fm_diffusion.density import Sampleable


class ConditionalProbabilityPath(torch.nn.Module, ABC):
    """Abstract base class for conditional probability paths."""

    def __init__(self, p_simple: Sampleable, p_data: Sampleable, dim: int):
        super().__init__()
        self.p_simple = p_simple
        self.p_data = p_data
        self.dim = dim

    def sample_marginal_path(self, t: torch.Tensor) -> torch.Tensor:
        """
        Sample from p_t(x) = ∫ p_t(x|z)p(z)dz.

        Args:
            t: time tensor with shape (num_samples, 1)
        """
        num_samples = t.shape[0]
        z = self.sample_conditioning_variable(num_samples)
        return self.sample_conditional_path(z, t)

    @abstractmethod
    def sample_conditioning_variable(self, num_samples: int) -> torch.Tensor:
        """Sample conditioning variables z ~ p(z)."""

    @abstractmethod
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Sample x from p_t(x|z)."""

    @abstractmethod
    def conditional_velocity_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Evaluate u_t(x|z)."""

    @abstractmethod
    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Evaluate ∇_x log p_t(x|z)."""
