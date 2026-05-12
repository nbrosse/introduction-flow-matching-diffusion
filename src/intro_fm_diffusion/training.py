import torch
from abc import ABC, abstractmethod

from tqdm import tqdm

from intro_fm_diffusion.paths.paths_base import ConditionalProbabilityPath


def build_mlp(dims: list[int], activation: type[torch.nn.Module] = torch.nn.SiLU):
    mlp = []
    for idx in range(len(dims) - 1):
        mlp.append(torch.nn.Linear(dims[idx], dims[idx + 1]))
        if idx < len(dims) - 2:
            mlp.append(activation())
    return torch.nn.Sequential(*mlp)


class MLPVelocityField(torch.nn.Module):
    """
    MLP-parameterization of the learned velocity field u_t^theta(x)
    """
    def __init__(self, dim: int, hiddens: list[int]):
        super().__init__()
        self.dim = dim
        self.net = build_mlp([dim + 1] + hiddens + [dim])

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        Args:
        - x: (bs, dim)
        Returns:
        - u_t^theta(x): (bs, dim)
        """
        xt = torch.cat([x, t], dim=-1)
        return self.net(xt)


class Trainer(ABC):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        pass

    def get_optimizer(self, lr: float):
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, num_epochs: int, device: torch.device, lr: float = 1e-3, **kwargs) -> list[float]:
        # Start
        self.model.to(device)
        opt = self.get_optimizer(lr)
        self.model.train()

        # Train loop
        pbar = tqdm(range(num_epochs))
        losses = []
        for epoch in pbar:
            opt.zero_grad()
            loss = self.get_train_loss(**kwargs)
            loss.backward()
            opt.step()
            loss_value = loss.item()
            pbar.set_description(f"Epoch {epoch}, loss: {loss_value}")
            losses.append(loss_value)

        # Finish
        self.model.eval()
        return losses


class ConditionalFlowMatchingTrainer(Trainer):
    def __init__(self, path: ConditionalProbabilityPath, model: MLPVelocityField):
        super().__init__(model)
        self.path = path

    def train(self, num_epochs: int, device: torch.device, lr: float = 1e-3, **kwargs) -> list[float]:
        self.path = self.path.to(device)
        return super().train(num_epochs=num_epochs, device=device, lr=lr, **kwargs)

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        device = next(self.model.parameters()).device
        z = self.path.sample_conditioning_variable(batch_size).to(device)
        t = torch.rand(batch_size, 1, device=device)
        x = self.path.sample_conditional_path(z, t)
        u_ref = self.path.conditional_velocity_field(x, z, t)
        u_theta = self.model(x, t)
        return torch.mean((u_theta - u_ref) ** 2)
