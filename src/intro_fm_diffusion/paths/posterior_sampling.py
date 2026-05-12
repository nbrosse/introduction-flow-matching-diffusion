import logging
from typing import Literal

import torch


logger = logging.getLogger(__name__)


def sample_posterior(
    path: object,
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
    """Sample z~p_t(z|x) with MALA or random-walk Metropolis."""
    if method == "mala":
        return _sample_posterior_mala(
            path=path,
            x=x,
            t=t,
            num_burnin=num_burnin,
            num_samples=num_samples,
            step_size=step_size,
            num_chains=num_chains,
            return_trajectory=return_trajectory,
        )
    if method == "metropolis":
        return _sample_posterior_metropolis(
            path=path,
            x=x,
            t=t,
            num_burnin=num_burnin,
            num_samples=num_samples,
            proposal_std=proposal_std,
            num_chains=num_chains,
            return_trajectory=return_trajectory,
        )
    raise ValueError(f"Unknown MCMC method: {method}")


def _sample_posterior_mala(
    path: object,
    x: torch.Tensor,
    t: torch.Tensor,
    num_burnin: int = 1000,
    num_samples: int = 1000,
    step_size: float | None = None,
    num_chains: int = 10,
    return_trajectory: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    alpha_t = path.alpha(t)
    beta_t = path.beta(t)
    sigma_post = 1.0 / torch.sqrt(1.0 + (alpha_t / (beta_t + 1e-6)) ** 2)
    sigma_post_scalar = sigma_post.mean().item()
    heuristic_step_size = 0.3 * sigma_post_scalar
    if step_size is None:
        step_size = heuristic_step_size

    batch_size = x.shape[0]
    dim = x.shape[1]

    x_expanded = x.repeat_interleave(num_chains, dim=0)
    t_expanded = t.repeat_interleave(num_chains, dim=0)

    z_current = path.p_data.sample(batch_size * num_chains).to(x.device)
    z_samples = torch.zeros(batch_size * num_chains, num_samples, dim, device=x.device)

    if return_trajectory:
        trajectories = [torch.zeros(num_burnin + num_samples, dim, device=x.device) for _ in range(batch_size * num_chains)]
        for chain_idx in range(batch_size * num_chains):
            trajectories[chain_idx][0] = z_current[chain_idx]

    accept_count_window = 0
    global_accept_count = 0
    global_total_proposals = 0

    adaptation_interval = 50
    target_accept_low = 0.45
    target_accept_high = 0.65

    use_closed_form = hasattr(path.p_data, "grad_log_density")
    sample_idx = 0

    for step in range(num_burnin + num_samples):
        if use_closed_form:
            grad_log_prob = path.grad_log_conditional_posterior(z_current, x_expanded, t_expanded)
            log_prob_current = path.log_conditional_posterior(z_current, x_expanded, t_expanded)
        else:
            z_current.requires_grad_(True)
            log_prob_current = path.log_conditional_posterior(z_current, x_expanded, t_expanded)
            grad_log_prob = torch.autograd.grad(
                outputs=log_prob_current.sum(),
                inputs=z_current,
                create_graph=False,
                retain_graph=False,
            )[0]
            z_current = z_current.detach()

        noise = torch.randn_like(z_current)
        mean_forward = z_current + (step_size**2 / 2) * grad_log_prob
        z_proposed = mean_forward + step_size * noise

        if use_closed_form:
            grad_log_prob_proposed = path.grad_log_conditional_posterior(z_proposed, x_expanded, t_expanded)
            log_prob_proposed = path.log_conditional_posterior(z_proposed, x_expanded, t_expanded)
        else:
            z_proposed.requires_grad_(True)
            log_prob_proposed = path.log_conditional_posterior(z_proposed, x_expanded, t_expanded)
            grad_log_prob_proposed = torch.autograd.grad(
                outputs=log_prob_proposed.sum(),
                inputs=z_proposed,
                create_graph=False,
                retain_graph=False,
            )[0]
            z_proposed = z_proposed.detach()

        mean_backward = z_proposed + (step_size**2 / 2) * grad_log_prob_proposed
        log_q_forward = -0.5 * torch.sum((z_proposed - mean_forward) ** 2, dim=-1) / step_size**2
        log_q_backward = -0.5 * torch.sum((z_current - mean_backward) ** 2, dim=-1) / step_size**2

        log_alpha = (log_prob_proposed + log_q_backward) - (log_prob_current + log_q_forward)
        accept = torch.log(torch.rand(batch_size * num_chains, device=x.device)) < log_alpha
        z_current = torch.where(accept.unsqueeze(-1), z_proposed, z_current)

        accepts = accept.float().sum().item()
        accept_count_window += accepts
        global_accept_count += accepts
        global_total_proposals += batch_size * num_chains

        if step < num_burnin and (step + 1) % adaptation_interval == 0:
            acceptance_rate = accept_count_window / (adaptation_interval * batch_size * num_chains)
            if acceptance_rate < target_accept_low:
                step_size *= 0.9
            elif acceptance_rate > target_accept_high:
                step_size *= 1.1
            accept_count_window = 0

        if return_trajectory and step < num_burnin + num_samples - 1:
            for chain_idx in range(batch_size * num_chains):
                trajectories[chain_idx][step + 1] = z_current[chain_idx]

        if step >= num_burnin and sample_idx < num_samples:
            z_samples[:, sample_idx, :] = z_current
            sample_idx += 1
            if sample_idx >= num_samples:
                break

    final_acceptance_rate = global_accept_count / global_total_proposals
    logger.info(
        f"MALA final acceptance rate: {final_acceptance_rate:.2%}, "
        f"final step_size: {step_size:.4f} ({num_chains} chains)"
    )

    z_samples = z_samples.reshape(batch_size, num_chains, num_samples, dim)
    z_samples = z_samples.transpose(1, 2).reshape(batch_size, num_samples * num_chains, dim)
    if return_trajectory:
        return z_samples, trajectories
    return z_samples


def _sample_posterior_metropolis(
    path: object,
    x: torch.Tensor,
    t: torch.Tensor,
    num_burnin: int = 1000,
    num_samples: int = 1000,
    proposal_std: float | None = None,
    num_chains: int = 10,
    return_trajectory: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    batch_size, dim = x.shape
    d = x.shape[1]
    alpha_t = path.alpha(t)
    beta_t = path.beta(t)
    sigma_post = 1.0 / torch.sqrt(1.0 + (alpha_t / (beta_t + 1e-6)) ** 2)
    sigma_post_scalar = sigma_post.mean().item()

    heuristic_proposal_std = 2.4 * sigma_post_scalar / (d**0.5)
    if proposal_std is None:
        proposal_std = heuristic_proposal_std

    x_expanded = x.repeat_interleave(num_chains, dim=0)
    t_expanded = t.repeat_interleave(num_chains, dim=0)

    z_current = path.p_data.sample(batch_size * num_chains).to(x.device)
    log_prob_current = path.log_conditional_posterior(z_current, x_expanded, t_expanded)
    z_samples = torch.zeros(batch_size * num_chains, num_samples, dim, device=x.device)

    if return_trajectory:
        trajectories = [torch.zeros(num_burnin + num_samples, dim, device=x.device) for _ in range(batch_size * num_chains)]
        for chain_idx in range(batch_size * num_chains):
            trajectories[chain_idx][0] = z_current[chain_idx]

    accept_count_window = 0
    global_accept_count = 0
    global_total_proposals = 0

    adaptation_interval = 50
    target_accept_low = 0.20
    target_accept_high = 0.40
    sample_idx = 0

    for step in range(num_burnin + num_samples):
        z_proposed = z_current + proposal_std * torch.randn_like(z_current)
        log_prob_proposed = path.log_conditional_posterior(z_proposed, x_expanded, t_expanded)

        log_alpha = log_prob_proposed - log_prob_current
        accept = torch.log(torch.rand(batch_size * num_chains, device=x.device)) < log_alpha

        z_current = torch.where(accept.unsqueeze(-1), z_proposed, z_current)
        log_prob_current = torch.where(accept, log_prob_proposed, log_prob_current)

        accepts = accept.float().sum().item()
        accept_count_window += accepts
        global_accept_count += accepts
        global_total_proposals += batch_size * num_chains

        if step < num_burnin and (step + 1) % adaptation_interval == 0:
            window_acceptance_rate = accept_count_window / (adaptation_interval * batch_size * num_chains)
            if window_acceptance_rate < target_accept_low:
                proposal_std *= 0.9
            elif window_acceptance_rate > target_accept_high:
                proposal_std *= 1.1
            accept_count_window = 0

        if return_trajectory and step < num_burnin + num_samples - 1:
            for chain_idx in range(batch_size * num_chains):
                trajectories[chain_idx][step + 1] = z_current[chain_idx]

        if step >= num_burnin and sample_idx < num_samples:
            z_samples[:, sample_idx, :] = z_current
            sample_idx += 1
            if sample_idx >= num_samples:
                break

    final_acceptance_rate = global_accept_count / global_total_proposals if global_total_proposals > 0 else 0.0
    logger.info(
        f"Metropolis final acceptance rate: {final_acceptance_rate:.2%}, "
        f"final proposal_std: {proposal_std:.4f} ({num_chains} chains)"
    )

    z_samples = z_samples.reshape(batch_size, num_chains, num_samples, dim)
    z_samples = z_samples.transpose(1, 2).reshape(batch_size, num_samples * num_chains, dim)
    if return_trajectory:
        return z_samples, trajectories
    return z_samples
