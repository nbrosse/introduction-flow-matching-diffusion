"""Notebook-facing plotting API."""

from intro_fm_diffusion.viz.decomposition import plot_cfm_decomposition_by_time
from intro_fm_diffusion.viz.density import imshow_log_density
from intro_fm_diffusion.viz.posterior import plot_posterior_mcmc_grid
from intro_fm_diffusion.viz.training import plot_late_training_plateau, plot_training_loss_linear_log
from intro_fm_diffusion.viz.velocity import plot_learned_vs_marginal_velocity_grid

__all__ = [
    "imshow_log_density",
    "plot_training_loss_linear_log",
    "plot_late_training_plateau",
    "plot_cfm_decomposition_by_time",
    "plot_learned_vs_marginal_velocity_grid",
    "plot_posterior_mcmc_grid",
]
