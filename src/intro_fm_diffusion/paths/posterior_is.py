from collections.abc import Callable

import torch


def compute_importance_weighted_values(
    path: object,
    x: torch.Tensor,
    t: torch.Tensor,
    num_posterior_samples: int,
    evaluator: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build importance-weighted posterior particles for z|x,t using z~p_data.

    Returns:
        values: shape (batch_size, num_posterior_samples, ...)
        weights: shape (batch_size, num_posterior_samples)
    """
    batch_size, dim = x.shape
    z_samples = path.sample_conditioning_variable(num_posterior_samples).to(x.device)

    x_expanded = x.unsqueeze(1).expand(batch_size, num_posterior_samples, -1)
    z_expanded = z_samples.unsqueeze(0).expand(batch_size, -1, -1)
    t_expanded = t.unsqueeze(1).expand(batch_size, num_posterior_samples, -1)

    x_flat = x_expanded.reshape(-1, dim)
    z_flat = z_expanded.reshape(-1, dim)
    t_flat = t_expanded.reshape(-1, 1)

    value_flat = evaluator(x_flat, z_flat, t_flat)
    trailing_shape = tuple(value_flat.shape[1:])
    values = value_flat.reshape(batch_size, num_posterior_samples, *trailing_shape)

    log_weights = path._log_conditional_path_density(x_flat, z_flat, t_flat).reshape(batch_size, num_posterior_samples)
    weights = torch.softmax(log_weights, dim=1)
    return values, weights


def compute_importance_weighted_expectation(
    path: object,
    x: torch.Tensor,
    t: torch.Tensor,
    num_posterior_samples: int,
    evaluator: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Compute E[g(Z)|x,t] with importance weights from p_t(x|z)."""
    values, weights = compute_importance_weighted_values(
        path=path,
        x=x,
        t=t,
        num_posterior_samples=num_posterior_samples,
        evaluator=evaluator,
    )

    weight_view = weights.view(weights.shape[0], weights.shape[1], *([1] * (values.ndim - 2)))
    return torch.sum(weight_view * values, dim=1)
