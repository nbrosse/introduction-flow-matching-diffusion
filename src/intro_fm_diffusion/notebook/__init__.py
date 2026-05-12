"""Notebook-first API facade."""

from intro_fm_diffusion.notebook.analysis import (
    analyze_cfm_decomposition_by_time,
    analyze_training_plateau_vs_global_cfm,
    analyze_velocity_fields_on_grid,
)
from intro_fm_diffusion.notebook.config import DEFAULT_HIDDEN_LAYERS, ExperimentConfig
from intro_fm_diffusion.notebook.paths_demo import (
    build_demo_gaussian_path,
    build_symmetric_gaussian_mixture_path,
)
from intro_fm_diffusion.notebook.training import (
    TrainingRun,
    resolve_device,
    run_notebook_demo_training,
    set_seed,
    train_demo_velocity_field,
)

__all__ = [
    "DEFAULT_HIDDEN_LAYERS",
    "ExperimentConfig",
    "TrainingRun",
    "set_seed",
    "resolve_device",
    "build_symmetric_gaussian_mixture_path",
    "build_demo_gaussian_path",
    "train_demo_velocity_field",
    "run_notebook_demo_training",
    "analyze_velocity_fields_on_grid",
    "analyze_cfm_decomposition_by_time",
    "analyze_training_plateau_vs_global_cfm",
]
