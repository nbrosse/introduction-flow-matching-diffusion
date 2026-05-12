"""Public API for the intro_fm_diffusion package."""

from intro_fm_diffusion.notebook import (
    DEFAULT_HIDDEN_LAYERS,
    ExperimentConfig,
    TrainingRun,
    analyze_cfm_decomposition_by_time,
    analyze_training_plateau_vs_global_cfm,
    analyze_velocity_fields_on_grid,
    build_demo_gaussian_path,
    build_symmetric_gaussian_mixture_path,
    resolve_device,
    run_notebook_demo_training,
    set_seed,
    train_demo_velocity_field,
)
from intro_fm_diffusion.paths.paths_base import ConditionalProbabilityPath
from intro_fm_diffusion.paths.paths_gaussian import GaussianConditionalProbabilityPath
from intro_fm_diffusion.training import ConditionalFlowMatchingTrainer, MLPVelocityField
from intro_fm_diffusion.viz import (
    imshow_log_density,
    plot_cfm_decomposition_by_time,
    plot_late_training_plateau,
    plot_learned_vs_marginal_velocity_grid,
    plot_posterior_mcmc_grid,
    plot_training_loss_linear_log,
)

__all__ = [
    "DEFAULT_HIDDEN_LAYERS",
    "ExperimentConfig",
    "TrainingRun",
    "ConditionalProbabilityPath",
    "GaussianConditionalProbabilityPath",
    "MLPVelocityField",
    "ConditionalFlowMatchingTrainer",
    "set_seed",
    "resolve_device",
    "build_symmetric_gaussian_mixture_path",
    "build_demo_gaussian_path",
    "train_demo_velocity_field",
    "run_notebook_demo_training",
    "analyze_velocity_fields_on_grid",
    "analyze_cfm_decomposition_by_time",
    "analyze_training_plateau_vs_global_cfm",
    "imshow_log_density",
    "plot_training_loss_linear_log",
    "plot_late_training_plateau",
    "plot_cfm_decomposition_by_time",
    "plot_learned_vs_marginal_velocity_grid",
    "plot_posterior_mcmc_grid",
]
