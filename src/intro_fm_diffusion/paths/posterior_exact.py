import math

import torch


def batched_gaussian_log_prob(x: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
    """Evaluate log N(x; mean, cov) for broadcast-compatible batched inputs."""
    dim = x.shape[-1]
    diff = x - mean
    solve = torch.linalg.solve(cov, diff.unsqueeze(-1)).squeeze(-1)
    quadratic = torch.sum(diff * solve, dim=-1)
    _, logdet = torch.linalg.slogdet(cov)
    log_two_pi = x.new_tensor(2.0 * math.pi).log()
    return -0.5 * (quadratic + logdet + dim * log_two_pi)


def posterior_mixture_parameters_exact(path: object, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute exact posterior mixture p_t(z|x) for Gaussian-mixture p_data.

    Returns:
        responsibilities: (batch_size, nmodes)
        component_means: (batch_size, nmodes, dim)
        component_covs: (batch_size, nmodes, dim, dim)
    """
    path.resolve_posterior_method("exact")

    batch_size, dim = x.shape
    means = path.p_data.means.to(device=x.device, dtype=x.dtype)
    covs = path.p_data.covs.to(device=x.device, dtype=x.dtype)
    precisions = path.p_data.precisions.to(device=x.device, dtype=x.dtype)
    weights = path.p_data.weights.to(device=x.device, dtype=x.dtype)

    alpha_t = path.alpha(t).to(dtype=x.dtype)
    beta_t = path._safe_beta(t).to(dtype=x.dtype)
    alpha_sq = alpha_t**2
    beta_sq = beta_t**2

    eye = torch.eye(dim, device=x.device, dtype=x.dtype)
    eye_batched = eye.view(1, 1, dim, dim)

    posterior_precisions = precisions.unsqueeze(0) + (alpha_sq / beta_sq).view(batch_size, 1, 1, 1) * eye_batched
    component_covs = torch.linalg.inv(posterior_precisions)

    precision_means = torch.einsum("mij,mj->mi", precisions, means)
    natural_parameters = precision_means.unsqueeze(0) + (alpha_t / beta_sq).view(batch_size, 1, 1) * x.unsqueeze(1)
    component_means = torch.einsum("bmij,bmj->bmi", component_covs, natural_parameters)

    marginal_means = alpha_t.view(batch_size, 1, 1) * means.unsqueeze(0)
    marginal_covs = alpha_sq.view(batch_size, 1, 1, 1) * covs.unsqueeze(0) + beta_sq.view(batch_size, 1, 1, 1) * eye_batched
    component_log_probs = batched_gaussian_log_prob(x=x.unsqueeze(1), mean=marginal_means, cov=marginal_covs)
    log_weights = torch.log(weights.clamp_min(torch.finfo(weights.dtype).tiny)).unsqueeze(0) + component_log_probs
    responsibilities = torch.softmax(log_weights, dim=1)

    return responsibilities, component_means, component_covs


def posterior_mean_exact(path: object, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    responsibilities, component_means, _ = posterior_mixture_parameters_exact(path, x, t)
    return torch.sum(responsibilities.unsqueeze(-1) * component_means, dim=1)


def posterior_covariance_exact(path: object, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    responsibilities, component_means, component_covs = posterior_mixture_parameters_exact(path, x, t)
    mean = torch.sum(responsibilities.unsqueeze(-1) * component_means, dim=1)
    offsets = component_means - mean.unsqueeze(1)
    between_covs = offsets.unsqueeze(-1) * offsets.unsqueeze(-2)
    return torch.sum(responsibilities[:, :, None, None] * (component_covs + between_covs), dim=1)


def marginal_velocity_field_exact(path: object, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    posterior_mean = posterior_mean_exact(path, x, t)
    c_t, d_t = path._velocity_coefficients(t)
    return c_t * posterior_mean + d_t * x


def marginal_score_exact(path: object, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    posterior_mean = posterior_mean_exact(path, x, t)
    alpha_t = path.alpha(t)
    beta_t = path._safe_beta(t)
    return (alpha_t * posterior_mean - x) / beta_t**2


def irreducible_variance_exact(path: object, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    posterior_covariance = posterior_covariance_exact(path, x, t)
    c_t, _ = path._velocity_coefficients(t)
    trace_covariance = torch.diagonal(posterior_covariance, dim1=-2, dim2=-1).sum(dim=-1)
    return c_t.squeeze(-1) ** 2 * trace_covariance / path.dim
