# Introduction to Flow Matching and Diffusion

This repository contains notebook-oriented code for small, interpretable experiments on conditional and marginal velocity fields in flow matching and diffusion models.

The accompanying blog post [Flow Matching as Posterior Averaging](https://nbrosse.github.io/posts/intro-fm-diffusion/intro-fm-diffusion.html) explains the theory and the code in more detail.

The current workflow focuses on:

- training a velocity model with conditional flow matching (CFM),
- analyzing marginal velocity fields and CFM loss decomposition terms,
- sampling and studying the posterior `p_t(z|x)` with MCMC and exact formulas (when available).

## What is in this repo

- `src/intro_fm_diffusion/`: core Python package (paths, training, diagnostics, plotting helpers)
- `notebooks/`: interactive experiment notebooks
- `figures/`: generated figures used in analyses/writeups

## Installation

Python `>=3.12` is required.

Install dependencies and set up the environment using [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

## Quick start (Python API)

The package exposes a notebook-friendly API through `intro_fm_diffusion`:

```python
from intro_fm_diffusion import (
    ExperimentConfig,
    run_notebook_demo_training,
    analyze_velocity_fields_on_grid,
)

config = ExperimentConfig(num_epochs=2000, batch_size=1000)
run = run_notebook_demo_training(config)
results = analyze_velocity_fields_on_grid(run.path, run.model)
```

For the full workflow and plots, start from:

- `notebooks/marginal_velocity_decomposition_and_posterior_mcmc.ipynb`

## Main components

- **Path modeling**: `GaussianConditionalProbabilityPath` for Gaussian/GMM-based conditional paths
- **Training**: `MLPVelocityField` with `ConditionalFlowMatchingTrainer`
- **Diagnostics**: marginal field estimation and decomposition utilities
- **Visualization**: plotting helpers for density, training curves, decomposition, velocity fields, and posterior sampling

## Notes

- The codebase is designed for research/teaching notebooks rather than production pipelines.
- Figures are typically produced during notebook runs and can be exported when needed.
