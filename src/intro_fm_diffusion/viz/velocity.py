"""Plots for velocity field comparisons."""

import matplotlib.pyplot as plt

from intro_fm_diffusion.paths.paths_base import ConditionalProbabilityPath
from intro_fm_diffusion.viz.density import imshow_log_density


def plot_learned_vs_marginal_velocity_grid(
    velocity_grid_analysis: dict,
    path: ConditionalProbabilityPath,
    scale: float = 15.0,
    show: bool = True,
):
    """Plot learned and target marginal velocity fields across time."""
    learned_velocity_fields = velocity_grid_analysis["learned_velocity_fields"]
    target_velocity_fields = velocity_grid_analysis["target_velocity_fields"]
    time_values = velocity_grid_analysis["time_values"]
    grid = velocity_grid_analysis["grid"]
    posterior_method = velocity_grid_analysis.get("posterior_method", "monte_carlo")

    time_indices = list(range(learned_velocity_fields.shape[0]))
    fig, axes = plt.subplots(2, len(time_indices), figsize=(4 * len(time_indices), 8))
    if len(time_indices) == 1:
        axes = axes.reshape(-1, 1)

    xx_flat = grid["xx_flat"]
    yy_flat = grid["yy_flat"]

    for col_idx, time_idx in enumerate(time_indices):
        t_val = time_values[time_idx]
        learned_velocity = learned_velocity_fields[time_idx]
        target_velocity = target_velocity_fields[time_idx]

        ax = axes[0, col_idx]
        ax.quiver(
            xx_flat,
            yy_flat,
            learned_velocity[:, 0],
            learned_velocity[:, 1],
            scale=150,
            alpha=0.6,
            width=0.003,
        )
        imshow_log_density(
            density=path.p_simple,
            x_bounds=[-scale, scale],
            y_bounds=[-scale, scale],
            bins=200,
            ax=ax,
            vmin=-10,
            alpha=0.15,
            cmap=plt.get_cmap("Reds"),
        )
        imshow_log_density(
            density=path.p_data,
            x_bounds=[-scale, scale],
            y_bounds=[-scale, scale],
            bins=200,
            ax=ax,
            vmin=-10,
            alpha=0.15,
            cmap=plt.get_cmap("Blues"),
        )
        ax.set_title(f"$u_{{t}}^{{\\theta}}(x)$ at t={t_val:.2f}", fontsize=14)
        ax.set_xlim(-scale, scale)
        ax.set_ylim(-scale, scale)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

        ax = axes[1, col_idx]
        ax.quiver(
            xx_flat,
            yy_flat,
            target_velocity[:, 0],
            target_velocity[:, 1],
            scale=150,
            alpha=0.5,
            width=0.003,
        )
        imshow_log_density(
            density=path.p_simple,
            x_bounds=[-scale, scale],
            y_bounds=[-scale, scale],
            bins=200,
            ax=ax,
            vmin=-10,
            alpha=0.15,
            cmap=plt.get_cmap("Reds"),
        )
        imshow_log_density(
            density=path.p_data,
            x_bounds=[-scale, scale],
            y_bounds=[-scale, scale],
            bins=200,
            ax=ax,
            vmin=-10,
            alpha=0.15,
            cmap=plt.get_cmap("Blues"),
        )
        ax.set_title(f"$u_{{t}}(x)$ at t={t_val:.2f}", fontsize=14)
        ax.set_xlim(-scale, scale)
        ax.set_ylim(-scale, scale)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    target_label = "Exact Gaussian-mixture posterior" if posterior_method == "exact" else "Monte Carlo"
    fig.suptitle(
        f"Velocity Field Evolution: Learned (top) vs Target ({target_label}) (bottom)",
        fontsize=16,
        y=0.995,
    )
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes
