from typing import Literal

import torch

from intro_fm_diffusion.density import Gaussian, GaussianMixture, Sampleable
from intro_fm_diffusion.paths.paths_base import ConditionalProbabilityPath
from intro_fm_diffusion.paths.posterior_exact import (
    marginal_score_exact as _marginal_score_exact,
    marginal_velocity_field_exact as _marginal_velocity_field_exact,
    posterior_covariance_exact as _posterior_covariance_exact,
    posterior_mean_exact as _posterior_mean_exact,
    posterior_mixture_parameters_exact as _posterior_mixture_parameters_exact,
    irreducible_variance_exact as _irreducible_variance_exact,
)
from intro_fm_diffusion.paths.posterior_is import compute_importance_weighted_expectation, compute_importance_weighted_values
from intro_fm_diffusion.paths.posterior_sampling import sample_posterior as _sample_posterior
from intro_fm_diffusion.schedules import Alpha, Beta


class GaussianConditionalProbabilityPath(ConditionalProbabilityPath):
    def __init__(self, p_data: Sampleable, alpha: Alpha, beta: Beta):
        p_simple = Gaussian.isotropic(p_data.dim, 1.0)
        super().__init__(p_simple, p_data, p_data.dim)
        self.alpha = alpha
        self.beta = beta
        self._eps = 1e-6

    def _safe_beta(self, t: torch.Tensor) -> torch.Tensor:
        return torch.clamp_min(self.beta(t), self._eps)

    def sample_conditioning_variable(self, num_samples: int) -> torch.Tensor:
        return self.p_data.sample(num_samples)

    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        noise = self.p_simple.sample(z.shape[0]).to(z.device)
        return z * self.alpha(t) + noise * self.beta(t)

    def conditional_velocity_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        c_t, d_t = self._velocity_coefficients(t)
        return c_t * z + d_t * x

    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        beta_t = self._safe_beta(t)
        return (self.alpha(t) * z - x) / beta_t**2

    def _log_conditional_path_density(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        dim = x.shape[1]
        mean = self.alpha(t) * z
        std = self._safe_beta(t)
        squared_distance = torch.sum((x - mean) ** 2 / std**2, dim=-1)
        return -0.5 * squared_distance - dim * torch.log(std.squeeze(-1))

    def _velocity_coefficients(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        beta_t = self._safe_beta(t)
        alpha_t = self.alpha(t)
        beta_dt = self.beta.dt(t)
        c_t = self.alpha.dt(t) - beta_dt * alpha_t / beta_t
        d_t = beta_dt / beta_t
        return c_t, d_t

    def has_exact_gaussian_mixture_posterior(self) -> bool:
        return isinstance(self.p_data, GaussianMixture)

    def resolve_posterior_method(self, method: Literal["auto", "exact", "monte_carlo"] = "auto") -> str:
        if method == "auto":
            return "exact" if self.has_exact_gaussian_mixture_posterior() else "monte_carlo"
        if method == "exact" and not self.has_exact_gaussian_mixture_posterior():
            raise NotImplementedError("Exact posterior formulas are implemented only for GaussianMixture p_data.")
        if method in {"exact", "monte_carlo"}:
            return method
        raise ValueError(f"Unknown posterior method: {method}")

    def posterior_mixture_parameters_exact(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _posterior_mixture_parameters_exact(self, x, t)

    def posterior_mean_exact(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return _posterior_mean_exact(self, x, t)

    def posterior_covariance_exact(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return _posterior_covariance_exact(self, x, t)

    def marginal_velocity_field_exact(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return _marginal_velocity_field_exact(self, x, t)

    def marginal_score_exact(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return _marginal_score_exact(self, x, t)

    def irreducible_variance_exact(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return _irreducible_variance_exact(self, x, t)

    def marginal_velocity_field(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        num_posterior_samples: int = 1000,
        method: Literal["auto", "exact", "monte_carlo"] = "auto",
    ) -> torch.Tensor:
        resolved_method = self.resolve_posterior_method(method)
        if resolved_method == "exact":
            return self.marginal_velocity_field_exact(x, t)
        return self.marginal_velocity_field_monte_carlo(x, t, num_posterior_samples)

    def marginal_score(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        num_posterior_samples: int = 1000,
        method: Literal["auto", "exact", "monte_carlo"] = "auto",
    ) -> torch.Tensor:
        resolved_method = self.resolve_posterior_method(method)
        if resolved_method == "exact":
            return self.marginal_score_exact(x, t)
        return self.marginal_score_monte_carlo(x, t, num_posterior_samples)

    def irreducible_variance(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        num_posterior_samples: int = 1000,
        method: Literal["auto", "exact", "monte_carlo"] = "auto",
    ) -> torch.Tensor:
        resolved_method = self.resolve_posterior_method(method)
        if resolved_method == "exact":
            return self.irreducible_variance_exact(x, t)
        return self.irreducible_variance_monte_carlo(x, t, num_posterior_samples)

    def marginal_velocity_field_monte_carlo(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        num_posterior_samples: int = 1000,
    ) -> torch.Tensor:
        return compute_importance_weighted_expectation(
            path=self,
            x=x,
            t=t,
            num_posterior_samples=num_posterior_samples,
            evaluator=lambda x_flat, z_flat, t_flat: self.conditional_velocity_field(x_flat, z_flat, t_flat),
        )

    def marginal_score_monte_carlo(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        num_posterior_samples: int = 1000,
    ) -> torch.Tensor:
        return compute_importance_weighted_expectation(
            path=self,
            x=x,
            t=t,
            num_posterior_samples=num_posterior_samples,
            evaluator=lambda x_flat, z_flat, t_flat: self.conditional_score(x_flat, z_flat, t_flat),
        )

    def irreducible_variance_monte_carlo(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        num_posterior_samples: int = 1000,
    ) -> torch.Tensor:
        u_cond, weights = compute_importance_weighted_values(
            path=self,
            x=x,
            t=t,
            num_posterior_samples=num_posterior_samples,
            evaluator=lambda x_flat, z_flat, t_flat: self.conditional_velocity_field(x_flat, z_flat, t_flat),
        )
        weight_view = weights.unsqueeze(-1)
        u_marginal = torch.sum(weight_view * u_cond, dim=1)
        squared_diff = torch.mean((u_cond - u_marginal.unsqueeze(1)) ** 2, dim=-1)
        return torch.sum(weights * squared_diff, dim=1)

    def log_conditional_posterior(self, z: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        log_p_x_given_z = self._log_conditional_path_density(x, z, t)
        log_p_z = self.p_data.log_density(z).squeeze(-1)
        return log_p_x_given_z + log_p_z

    def grad_log_conditional_posterior(self, z: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        alpha_t = self.alpha(t)
        beta_t = self._safe_beta(t)
        inv_beta_sq = 1.0 / (beta_t**2)
        grad_likelihood = (alpha_t * inv_beta_sq) * x - (alpha_t**2 * inv_beta_sq) * z

        if hasattr(self.p_data, "grad_log_density"):
            grad_prior = self.p_data.grad_log_density(z)
        else:
            raise NotImplementedError("Closed-form gradient requires p_data to have a grad_log_density(z) method")
        return grad_likelihood + grad_prior

    def sample_posterior(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        num_burnin: int = 1000,
        num_samples: int = 1000,
        method: Literal["mala", "metropolis"] = "mala",
        step_size: float | None = None,
        proposal_std: float | None = None,
        num_chains: int = 10,
        return_trajectory: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        return _sample_posterior(
            path=self,
            x=x,
            t=t,
            num_burnin=num_burnin,
            num_samples=num_samples,
            method=method,
            step_size=step_size,
            proposal_std=proposal_std,
            num_chains=num_chains,
            return_trajectory=return_trajectory,
        )
