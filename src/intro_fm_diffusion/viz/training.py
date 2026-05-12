"""Plots for training diagnostics."""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_training_loss_linear_log(
    training_losses: Sequence[float],
    figsize: tuple[int, int] = (10, 6),
    show: bool = True,
):
    """Plot training loss with linear and log scales."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].plot(training_losses, linewidth=2, color="#2E86AB")
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Training Loss (Linear Scale)", fontsize=14, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(training_losses, linewidth=2, color="#A23B72")
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Loss (log scale)", fontsize=12)
    axes[1].set_title("Training Loss (Log Scale)", fontsize=14, fontweight="bold")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes


def plot_late_training_plateau(
    training_losses: Sequence[float],
    plateau_analysis: dict,
    running_mean_window: int = 25,
    show: bool = True,
):
    """Plot late-training tail with plateau and global CFM estimate lines."""
    losses = np.asarray(training_losses, dtype=float)
    if losses.size == 0:
        raise ValueError("training_losses must not be empty.")

    plateau_window = int(plateau_analysis["plateau_window"])
    plateau_mean = float(plateau_analysis["plateau_mean"])
    tail_losses = losses[-plateau_window:]
    tail_epochs = np.arange(losses.size - plateau_window, losses.size)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(tail_epochs, tail_losses, color="#4c78a8", alpha=0.35, linewidth=1.5, label="Training loss")

    if running_mean_window > 1 and tail_losses.size >= running_mean_window:
        kernel = np.ones(running_mean_window, dtype=float) / running_mean_window
        running_mean = np.convolve(tail_losses, kernel, mode="valid")
        running_epochs = tail_epochs[running_mean_window - 1 :]
        ax.plot(
            running_epochs,
            running_mean,
            color="#1f4b7a",
            linewidth=2.5,
            label=f"{running_mean_window}-step running mean",
        )

    ax.axhline(plateau_mean, color="#e45756", linestyle="--", linewidth=2, label="Late-training mean")

    global_conditional_mse = plateau_analysis.get("global_conditional_mse")
    if global_conditional_mse is not None:
        ax.axhline(
            float(global_conditional_mse["mean_loss"]),
            color="#f58518",
            linestyle="-.",
            linewidth=2,
            label="Empirical global CFM loss",
        )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Late-Training Plateau", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax
