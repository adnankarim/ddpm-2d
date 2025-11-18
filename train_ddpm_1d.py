import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class DDPMConfig:
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    batch_size: int = 64
    lr: float = 1e-3
    num_epochs: int = 4000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_path: str = "ddpm_1d.pt"


def make_beta_schedule(timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    """Linear beta schedule."""
    return torch.linspace(beta_start, beta_end, timesteps)


class SmallMLP(nn.Module):
    """Simple MLP that predicts noise epsilon for 1D data conditioned on timestep."""

    def __init__(self, hidden_dim: int = 64, time_embed_dim: int = 32):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
        )

        self.net = nn.Sequential(
            nn.Linear(1 + time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1]
        t: [B] integer timesteps
        """
        # Normalize timestep to [0, 1]
        t_norm = t.float().unsqueeze(-1) / 1000.0
        t_emb = self.time_embed(t_norm)
        inp = torch.cat([x, t_emb], dim=-1)
        return self.net(inp)


class DDPM1D:
    def __init__(self, config: DDPMConfig):
        self.cfg = config
        self.device = torch.device(self.cfg.device)

        betas = make_beta_schedule(self.cfg.timesteps, self.cfg.beta_start, self.cfg.beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas.to(self.device)
        self.alphas = alphas.to(self.device)
        self.alphas_cumprod = alphas_cumprod.to(self.device)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(self.device)

        self.model = SmallMLP().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """
        Diffusion forward process: q(x_t | x_0)
        x0: [B, 1]
        t: [B]
        """
        if noise is None:
            noise = torch.randn_like(x0)
        # Gather the corresponding alpha_cumprod for each t
        sqrt_ac = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        return sqrt_ac * x0 + sqrt_om * noise

    def p_sample(self, x_t: torch.Tensor, t: int) -> torch.Tensor:
        """
        One reverse diffusion step: p(x_{t-1} | x_t)
        x_t: [B, 1]
        t: scalar integer
        """
        b = x_t.shape[0]
        t_tensor = torch.full((b,), t, device=self.device, dtype=torch.long)
        beta_t = self.betas[t]
        sqrt_one_minus_ac = self.sqrt_one_minus_alphas_cumprod[t]
        alpha_t = self.alphas[t]
        alpha_cumprod_t = self.alphas_cumprod[t]
        sqrt_recip_alpha = torch.sqrt(1.0 / alpha_t)

        # Predict noise using the model
        eps_theta = self.model(x_t, t_tensor)

        # Equation (11) in DDPM paper
        mean = sqrt_recip_alpha * (x_t - (beta_t / sqrt_one_minus_ac) * eps_theta)

        if t == 0:
            return mean

        noise = torch.randn_like(x_t)
        sigma_t = torch.sqrt(beta_t)
        return mean + sigma_t * noise

    @torch.no_grad()
    def sample(self, num_samples: int = 100) -> torch.Tensor:
        """
        Generate samples from the model by starting from standard Gaussian noise.
        Returns: [num_samples, 1]
        """
        x_t = torch.randn(num_samples, 1, device=self.device)
        for t in reversed(range(self.cfg.timesteps)):
            x_t = self.p_sample(x_t, t)
        return x_t

    def train_step(self, x0_batch: torch.Tensor) -> float:
        """
        One training step using the simplified objective:
        E_{t, x0, eps} || eps - eps_theta(x_t, t) ||^2
        """
        self.model.train()
        b = x0_batch.shape[0]
        t = torch.randint(0, self.cfg.timesteps, (b,), device=self.device, dtype=torch.long)
        noise = torch.randn_like(x0_batch)

        x_t = self.q_sample(x0_batch, t, noise)
        eps_pred = self.model(x_t, t)

        loss = torch.mean((noise - eps_pred) ** 2)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())


def make_toy_dataset(n: int = 100, offset: float = 2.0) -> torch.Tensor:
    """
    Generate a simple 1D dataset: x = offset + U(0, 1)
    Shape: [n, 1]
    """
    return offset + torch.rand(n, 1)


def train():
    cfg = DDPMConfig()
    device = torch.device(cfg.device)
    print(f"Using device: {device}")

    # Dataset
    data = make_toy_dataset(100, offset=2.0)
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    ddpm = DDPM1D(cfg)

    global_step = 0
    for epoch in range(cfg.num_epochs):
        for (x_batch,) in loader:
            x_batch = x_batch.to(device)
            loss = ddpm.train_step(x_batch)
            global_step += 1

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{cfg.num_epochs}, loss={loss:.6f}")

    # Save model and config
    save_obj = {
        "model_state_dict": ddpm.model.state_dict(),
        "config": cfg.__dict__,
    }
    torch.save(save_obj, cfg.model_path)
    print(f"Saved trained DDPM model to {cfg.model_path}")


if __name__ == "__main__":
    train()


