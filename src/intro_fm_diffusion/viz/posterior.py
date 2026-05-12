"""Plots for posterior sampling diagnostics."""

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch

from intro_fm_diffusion.paths.paths_gaussian import GaussianConditionalProbabilityPath
from intro_fm_diffusion.viz.density import _infer_module_device, imshow_log_density


def plot_posterior_mcmc_grid(
    path: GaussianConditionalProbabilityPath,
    device: torch.device | None = None,
    scale: float = 15.0,
    num_trajectories: int = 10,
    num_posterior_samples: int = 1000,
    method: Literal["mala", "metropolis"] = "metropolis",
    num_burnin: int = 1000,
    num_chains: int = 10,
    seed: int = 42,
    show: bool = True,
):
    """Visualize posterior samples p_t(z | x_t) along random trajectories."""
    device = device or _infer_module_device(path)
    torch.manual_seed(seed)
    np.random.seed(seed)

    z = path.p_data.sample(num_trajectories).to(device)
    t = torch.rand(num_trajectories, 1).to(device) * 0.9 + 0.05
    x_0 = path.p_simple.sample(num_trajectories).to(device)
    x_t = path.alpha(t) * z + path.beta(t) * x_0

    z_samples = path.sample_posterior(
        x_t,
        t,
        num_burnin=num_burnin,
        num_samples=num_posterior_samples,
        method=method,
        num_chains=num_chains,
    )

    ncols = 5
    nrows = (num_trajectories + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for idx in range(num_trajectories):
        ax = axes[idx]
        x_loc = x_t[idx].cpu().numpy()
        z_origin = z[idx].cpu().numpy()
        t_val = t[idx].item()
        z_posterior_samples = z_samples[idx].cpu().numpy()

        ax.hist2d(
            z_posterior_samples[:, 0],
            z_posterior_samples[:, 1],
            bins=50,
            cmap="Purples",
            alpha=0.8,
            range=[[-scale, scale], [-scale, scale]],
        )
        imshow_log_density(
            density=path.p_data,
            x_bounds=[-scale, scale],
            y_bounds=[-scale, scale],
            bins=150,
            ax=ax,
            vmin=-10,
            alpha=0.25,
            cmap=plt.get_cmap("Blues"),
        )
        imshow_log_density(
            density=path.p_simple,
            x_bounds=[-scale, scale],
            y_bounds=[-scale, scale],
            bins=150,
            ax=ax,
            vmin=-10,
            alpha=0.25,
            cmap=plt.get_cmap("Reds"),
        )
        ax.scatter(
            [x_loc[0]],
            [x_loc[1]],
            c="red",
            s=200,
            marker="X",
            linewidths=3,
            edgecolors="darkred",
            zorder=10,
            label="$x_t$",
        )
        ax.scatter(
            [z_origin[0]],
            [z_origin[1]],
            c="green",
            s=150,
            marker="o",
            linewidths=2,
            edgecolors="black",
            zorder=9,
            alpha=0.7,
            label="$z$",
        )
        ax.scatter(
            [x_0[idx][0].cpu()],
            [x_0[idx][1].cpu()],
            c="orange",
            s=150,
            marker="o",
            linewidths=2,
            edgecolors="black",
            zorder=9,
            alpha=0.7,
            label="$x_0$",
        )
        ax.set_xlim(-scale, scale)
        ax.set_ylim(-scale, scale)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"t = {t_val:.3f}", fontsize=12, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.8)

        for spine in ax.spines.values():
            spine.set_edgecolor("gray")
            spine.set_linewidth(1.5)

    for idx in range(num_trajectories, len(axes)):
        axes[idx].axis("off")

    fig.suptitle(
        "Posterior Densities $p_t(z|x_t)$ Along Random Trajectories\n"
        "Purple: Posterior Samples | Blue: $p_{data}$ | "
        "Red X: $x_t$ | Green: $z$ | Orange: $x_0$",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes
