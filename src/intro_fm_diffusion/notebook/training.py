"""Notebook-oriented training orchestration."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from intro_fm_diffusion.notebook.config import ExperimentConfig
from intro_fm_diffusion.notebook.paths_demo import build_demo_gaussian_path
from intro_fm_diffusion.paths.paths_gaussian import GaussianConditionalProbabilityPath
from intro_fm_diffusion.training import ConditionalFlowMatchingTrainer, MLPVelocityField
from intro_fm_diffusion.utils import get_device


@dataclass(frozen=True)
class TrainingRun:
    config: ExperimentConfig | None
    path: GaussianConditionalProbabilityPath
    device: torch.device
    model: MLPVelocityField
    trainer: ConditionalFlowMatchingTrainer
    losses: list[float]


def set_seed(seed: int) -> None:
    """Set NumPy and PyTorch seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def resolve_device(
    device: torch.device | str | None = None,
    model: torch.nn.Module | None = None,
) -> torch.device:
    """Resolve target device with optional module fallback."""
    if device is not None:
        return get_device(device)
    if model is not None:
        try:
            return next(model.parameters()).device
        except StopIteration:
            pass
    return get_device(None)


def train_demo_velocity_field(
    path: GaussianConditionalProbabilityPath,
    model: MLPVelocityField | None = None,
    hidden_layers: Sequence[int] = (64, 64, 64, 64),
    num_epochs: int = 5000,
    learning_rate: float = 1e-3,
    batch_size: int = 1000,
    device: torch.device | str | None = None,
    seed: int | None = None,
) -> TrainingRun:
    """Train a velocity model and return a typed run bundle."""
    if seed is not None:
        set_seed(seed)

    resolved_device = resolve_device(device=device, model=model)
    path = path.to(resolved_device)

    if model is None:
        model = MLPVelocityField(dim=path.dim, hiddens=list(hidden_layers)).to(resolved_device)
    else:
        model = model.to(resolved_device)

    trainer = ConditionalFlowMatchingTrainer(path, model)
    losses = trainer.train(
        num_epochs=num_epochs,
        device=resolved_device,
        lr=learning_rate,
        batch_size=batch_size,
    )
    return TrainingRun(
        config=None,
        path=path,
        device=resolved_device,
        model=model,
        trainer=trainer,
        losses=losses,
    )


def run_notebook_demo_training(config: ExperimentConfig) -> TrainingRun:
    """Run the canonical notebook training with one config object."""
    path = build_demo_gaussian_path(
        nmodes=config.nmodes,
        target_std=config.target_std,
        target_scale=config.target_scale,
        device=config.device,
    )
    run = train_demo_velocity_field(
        path=path,
        hidden_layers=config.hidden_layers,
        num_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        device=config.device,
        seed=config.seed,
    )
    return TrainingRun(
        config=config,
        path=run.path,
        device=run.device,
        model=run.model,
        trainer=run.trainer,
        losses=run.losses,
    )
