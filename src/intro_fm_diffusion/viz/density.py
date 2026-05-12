"""Density plotting utilities."""

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes

from intro_fm_diffusion.density import Density


def _infer_module_device(module: object) -> torch.device:
    if isinstance(module, torch.nn.Module):
        for parameter in module.parameters():
            return parameter.device
        for buffer in module.buffers():
            return buffer.device
    return torch.device("cpu")


def imshow_log_density(
    density: Density,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    bins: int,
    ax: Axes | None = None,
    x_offset: float = 0.0,
    **kwargs,
):
    """Render log-density values as an image."""
    if ax is None:
        ax = plt.gca()

    density_device = _infer_module_device(density)
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    x = torch.linspace(x_min, x_max, bins, device=density_device) + x_offset
    y = torch.linspace(y_min, y_max, bins, device=density_device)
    x_mesh, y_mesh = torch.meshgrid(x, y, indexing="ij")
    xy = torch.stack([x_mesh.reshape(-1), y_mesh.reshape(-1)], dim=-1)
    density_values = density.log_density(xy).reshape(bins, bins).T

    imshow_kwargs = dict(kwargs)
    if "vmin" in imshow_kwargs:
        vmax = imshow_kwargs.get("vmax", float(density_values.max().item()))
        imshow_kwargs["vmax"] = max(float(imshow_kwargs["vmin"]), float(vmax))

    ax.imshow(
        density_values.cpu(),
        extent=[x_min, x_max, y_min, y_max],
        origin="lower",
        **imshow_kwargs,
    )
