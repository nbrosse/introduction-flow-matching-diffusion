"""Plots for CFM decomposition diagnostics."""

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def plot_cfm_decomposition_by_time(
    marginal_by_time: dict,
    conditional_by_time: dict,
    irreducible_variance_by_time: dict,
    show: bool = True,
):
    """Plot marginal/conditional curves with irreducible variance term."""
    fig, ax = plt.subplots(figsize=(12, 7))

    marginal_time_bins = marginal_by_time["time_bins"]
    marginal_mean_losses = marginal_by_time["mean_losses"]
    marginal_std_losses = marginal_by_time["std_losses"]

    conditional_time_bins = conditional_by_time["time_bins"]
    conditional_mean_losses = conditional_by_time["mean_losses"]
    conditional_std_losses = conditional_by_time["std_losses"]

    variance_time_values = irreducible_variance_by_time["time_values"]
    variances = irreducible_variance_by_time["variances"]

    ax.plot(
        conditional_time_bins,
        conditional_mean_losses,
        "o-",
        label=r"$\mathcal{L}_{\mathrm{CFM}}(\theta)$ (Conditional-target MSE)",
        linewidth=2.5,
        markersize=7,
        color="#d62728",
        zorder=3,
    )
    ax.fill_between(
        conditional_time_bins,
        conditional_mean_losses - conditional_std_losses,
        conditional_mean_losses + conditional_std_losses,
        alpha=0.2,
        color="#d62728",
    )

    ax.plot(
        marginal_time_bins,
        marginal_mean_losses,
        "s-",
        label=r"$\mathcal{L}_{\mathrm{FM}}(\theta)$ (Marginal-target MSE)",
        linewidth=2.5,
        markersize=7,
        color="#1f77b4",
        zorder=3,
    )
    ax.fill_between(
        marginal_time_bins,
        marginal_mean_losses - marginal_std_losses,
        marginal_mean_losses + marginal_std_losses,
        alpha=0.2,
        color="#1f77b4",
    )

    ax.plot(
        variance_time_values,
        variances,
        "^-",
        label=r"$\mathbb{E}[\mathrm{Tr}\,\mathrm{Cov}(u_t(X_t\mid Z)\mid X_t)]/d$",
        linewidth=2.5,
        markersize=7,
        color="#2ca02c",
        zorder=3,
    )

    variances_at_marginal_times = interp1d(
        variance_time_values,
        variances,
        kind="linear",
        fill_value="extrapolate",
    )(marginal_time_bins)
    marginal_plus_var = marginal_mean_losses + variances_at_marginal_times
    ax.plot(
        marginal_time_bins,
        marginal_plus_var,
        "x--",
        label=r"$\mathcal{L}_{\mathrm{FM}}(\theta) + \mathrm{Irreducible Variance}$",
        linewidth=2,
        markersize=8,
        color="#ff7f0e",
        alpha=0.7,
        zorder=2,
    )

    ax.set_xlabel("Time $t$", fontsize=13)
    ax.set_ylabel("Loss", fontsize=13)
    ax.set_title("CFM Decomposition by Time", fontsize=15, fontweight="bold", pad=15)
    ax.legend(fontsize=10, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")

    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax
