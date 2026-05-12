"""Configuration types for notebook experiments."""

from dataclasses import dataclass

DEFAULT_HIDDEN_LAYERS = (64, 64, 64, 64)


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    device: str | None = "auto"
    nmodes: int = 5
    target_std: float = 1.0
    target_scale: float = 10.0
    hidden_layers: tuple[int, ...] = DEFAULT_HIDDEN_LAYERS
    num_epochs: int = 2000
    learning_rate: float = 1e-3
    batch_size: int = 1000
