"""Canonical exports for path classes."""

from intro_fm_diffusion.paths.paths_base import ConditionalProbabilityPath
from intro_fm_diffusion.paths.paths_gaussian import GaussianConditionalProbabilityPath

__all__ = ["ConditionalProbabilityPath", "GaussianConditionalProbabilityPath"]
