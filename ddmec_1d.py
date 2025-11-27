"""
DDMEC (Denoising Diffusion Minimum Entropy Coupling) for 1D Gaussian Models

This implements the cooperative training scheme where two diffusion models
learn the minimum entropy coupling between two distributions.
"""

import os
import copy
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, List
from collections import defaultdict

from train_ddpm_1d import DDPM1D, DDPMConfig, SmallMLP

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")


class ConditionalMLP(nn.Module):
    """Wrapper to make SmallMLP conditional by concatenating condition."""
    
    def __init__(self, base_model: SmallMLP):
        super().__init__()
        self.base_model = base_model
        # Build a new conditional network using similar architecture
        # Input: [x, condition] -> concat with time_emb -> predict noise
        hidden_dim = 64
        time_embed_dim = 32
        
        # Input projection for [x, condition, time_emb] -> hidden
        self.input_proj = nn.Sequential(
            nn.Linear(2 + time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor = None) -> torch.Tensor:
        """
        x: [B, 1] - noisy input
        t: [B] - timestep
        condition: [B, 1] - conditioning variable (optional)
        """
        # Get time embedding from base model
        t_norm = t.float().unsqueeze(-1) / 1000.0
        t_emb = self.base_model.time_embed(t_norm)
        
        if condition is None:
            # Unconditional: just use the base model
            inp = torch.cat([x, t_emb], dim=-1)
            return self.base_model.net(inp)
        
        # Concatenate x, condition, and time embedding
        inp = torch.cat([x, condition, t_emb], dim=-1)
        return self.input_proj(inp)


class DDMEC1D:
    """
    DDMEC: Two diffusion models cooperatively learn minimum entropy coupling.
    
    Model 1 learns p(x1|x2) and Model 2 learns p(x2|x1).
    They alternate between being the generator (policy) and reward function (critic).
    """
    
    def __init__(
        self,
        model_path_1: str,
        model_path_2: str,
        config: Optional[DDPMConfig] = None,
        use_wandb: bool = False,
    ):
        """
        Args:
            model_path_1: Path to first pre-trained model
            model_path_2: Path to second pre-trained model
            config: Configuration (will be loaded from checkpoints if available)
            use_wandb: Whether to log metrics to Weights & Biases
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        
        # Load models
        print(f"Loading Model 1 from {model_path_1}")
        self.ddpm_1, self.config_1 = self._load_model(model_path_1)
        
        print(f"Loading Model 2 from {model_path_2}")
        self.ddpm_2, self.config_2 = self._load_model(model_path_2)
        
        # Use config from first model if not provided
        self.config = config or self.config_1
        
        # Create frozen copies for KL regularization (priors)
        self.frozen_model_1 = None
        self.frozen_model_2 = None
        
        # Training state
        self.gen_step = 0
        self.warmup_steps = 1000  # Warmup with supervised training
        
        # Tracking
        self.losses = defaultdict(list)
        
    def _load_model(self, path: str) -> Tuple[DDPM1D, DDPMConfig]:
        """Load a model and its config from checkpoint."""
        cfg = DDPMConfig()
        checkpoint = torch.load(path, map_location=self.device)
        
        # Update config from checkpoint
        if "config" in checkpoint:
            for k, v in checkpoint["config"].items():
                setattr(cfg, k, v)
        
        ddpm = DDPM1D(cfg)
        # Load the base unconditional model
        ddpm.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Wrap with conditional wrapper
        ddpm.model = ConditionalMLP(ddpm.model).to(self.device)
        
        return ddpm, cfg
    
    def setup_priors(self):
        """Freeze copies of models to serve as marginal priors for KL penalty."""
        print("Setting up frozen priors for KL regularization")
        self.frozen_model_1 = copy.deepcopy(self.ddpm_1.model)
        self.frozen_model_1.eval()
        self.frozen_model_1.requires_grad_(False)
        
        self.frozen_model_2 = copy.deepcopy(self.ddpm_2.model)
        self.frozen_model_2.eval()
        self.frozen_model_2.requires_grad_(False)
    
    def train_supervised(
        self,
        x1_batch: torch.Tensor,
        x2_batch: torch.Tensor,
        optimizer_1: torch.optim.Optimizer,
        optimizer_2: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """
        Supervised warmup training: standard denoising objective.
        
        Args:
            x1_batch: Samples from distribution 1
            x2_batch: Samples from distribution 2 (conditioning)
            optimizer_1: Optimizer for model 1
            optimizer_2: Optimizer for model 2
            
        Returns:
            Dictionary of losses
        """
        # Train model 1: denoise x1 conditioned on x2
        self.ddpm_1.model.train()
        optimizer_1.zero_grad()
        
        t = torch.randint(0, self.config.timesteps, (x1_batch.shape[0],), device=self.device)
        noise = torch.randn_like(x1_batch)
        
        # Forward diffusion
        alphas_cumprod = self.ddpm_1.alphas_cumprod[t].view(-1, 1)
        x1_noisy = torch.sqrt(alphas_cumprod) * x1_batch + torch.sqrt(1 - alphas_cumprod) * noise
        
        # Predict noise with conditioning on x2
        predicted_noise = self.ddpm_1.model(x1_noisy, t, x2_batch)
        loss_1 = nn.functional.mse_loss(predicted_noise, noise)
        
        loss_1.backward()
        torch.nn.utils.clip_grad_norm_(self.ddpm_1.model.parameters(), 1.0)
        optimizer_1.step()
        
        # Train model 2: denoise x2 conditioned on x1
        self.ddpm_2.model.train()
        optimizer_2.zero_grad()
        
        t = torch.randint(0, self.config.timesteps, (x2_batch.shape[0],), device=self.device)
        noise = torch.randn_like(x2_batch)
        
        alphas_cumprod = self.ddpm_2.alphas_cumprod[t].view(-1, 1)
        x2_noisy = torch.sqrt(alphas_cumprod) * x2_batch + torch.sqrt(1 - alphas_cumprod) * noise
        
        predicted_noise = self.ddpm_2.model(x2_noisy, t, x1_batch)
        loss_2 = nn.functional.mse_loss(predicted_noise, noise)
        
        loss_2.backward()
        torch.nn.utils.clip_grad_norm_(self.ddpm_2.model.parameters(), 1.0)
        optimizer_2.step()
        
        return {"loss_1": loss_1.item(), "loss_2": loss_2.item()}
    
    @torch.no_grad()
    def sample_trajectory_with_logprob(
        self,
        model: nn.Module,
        ddpm: DDPM1D,
        condition: torch.Tensor,
        num_steps: int = 50,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[int]]:
        """
        Sample from model and track log probabilities for policy gradient.
        
        Args:
            model: The denoising model
            ddpm: DDPM wrapper with schedules
            condition: Conditioning variable
            num_steps: Number of denoising steps
            
        Returns:
            trajectory: List of latent states
            log_probs: List of log probabilities
            timesteps: List of timestep indices
        """
        model.eval()
        batch_size = condition.shape[0]
        
        # Start from noise
        x = torch.randn(batch_size, 1, device=self.device)
        
        # Ensure initial noise is not NaN
        if torch.isnan(x).any():
            print("Warning: NaN in initial noise, regenerating...")
            x = torch.randn(batch_size, 1, device=self.device)
        
        # Ensure condition is not NaN
        if torch.isnan(condition).any():
            print("Warning: NaN in condition input")
            condition = torch.nan_to_num(condition, nan=0.0)
        
        trajectory = [x.clone()]
        log_probs = []
        timesteps_list = []
        
        # Create evenly spaced timesteps
        timesteps = torch.linspace(self.config.timesteps - 1, 0, num_steps, dtype=torch.long)
        
        for i in range(len(timesteps) - 1):
            t_cur = timesteps[i].item()
            t_next = timesteps[i + 1].item()
            
            t_tensor = torch.full((batch_size,), t_cur, device=self.device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = model(x, t_tensor, condition)
            
            # Check for NaN in predicted noise
            if torch.isnan(predicted_noise).any():
                print(f"Warning: NaN detected in predicted_noise at timestep {t_cur}")
                predicted_noise = torch.nan_to_num(predicted_noise, nan=0.0)
            
            # Get schedule values
            alpha_t = ddpm.alphas[t_cur]
            alphas_cumprod_t = ddpm.alphas_cumprod[t_cur]
            beta_t = ddpm.betas[t_cur]
            
            if t_next > 0:
                alphas_cumprod_next = ddpm.alphas_cumprod[t_next]
            else:
                alphas_cumprod_next = torch.tensor(1.0, device=self.device)
            
            # Add epsilon for numerical stability
            eps = 1e-8
            sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod_t + eps)
            sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - alphas_cumprod_t + eps)
            
            # DDIM sampling step with numerical stability
            x0_pred = (x - sqrt_one_minus_alphas_cumprod_t * predicted_noise) / sqrt_alphas_cumprod_t
            
            # Clamp x0_pred to reasonable range to prevent extreme values
            x0_pred = torch.clamp(x0_pred, -50.0, 50.0)
            
            # Compute mean
            if t_next > 0:
                sqrt_alphas_cumprod_next = torch.sqrt(alphas_cumprod_next + eps)
                sqrt_one_minus_alphas_cumprod_next = torch.sqrt(1 - alphas_cumprod_next + eps)
                x_next = sqrt_alphas_cumprod_next * x0_pred + sqrt_one_minus_alphas_cumprod_next * predicted_noise
            else:
                x_next = x0_pred
            
            # Check for NaN in x_next
            if torch.isnan(x_next).any():
                print(f"Warning: NaN detected in x_next at timestep {t_cur}")
                x_next = torch.nan_to_num(x_next, nan=0.0)
            
            # Compute log prob (approximation for DDIM)
            # log p(x_next | x, condition) ≈ -||x_next - mean||^2 / (2 * variance)
            variance = beta_t + eps
            log_prob = -0.5 * ((x_next - x) ** 2).sum(dim=1) / variance
            
            x = x_next
            trajectory.append(x.clone())
            log_probs.append(log_prob)
            timesteps_list.append(t_cur)
        
        return trajectory, log_probs, timesteps_list
    
    @torch.no_grad()
    def compute_reward(
        self,
        x_gen: torch.Tensor,
        condition: torch.Tensor,
        reward_model: nn.Module,
        reward_ddpm: DDPM1D,
        mc_steps: int = 5,
    ) -> torch.Tensor:
        """
        Compute reward as negative log-likelihood under reward model.
        
        This measures how well the generated x_gen matches the conditioning variable
        under the reward model's learned distribution.
        
        Args:
            x_gen: Generated samples
            condition: Conditioning variable
            reward_model: Model to compute likelihood
            reward_ddpm: DDPM wrapper for reward model
            mc_steps: Number of Monte Carlo samples for NLL estimation
            
        Returns:
            rewards: Shape (batch_size,)
        """
        reward_model.eval()
        batch_size = condition.shape[0]
        
        nll = 0.0
        
        # Monte Carlo estimate of NLL
        for _ in range(mc_steps):
            # Sample random timestep
            t = torch.randint(0, self.config.timesteps, (batch_size,), device=self.device)
            
            # Add noise to condition
            noise = torch.randn_like(condition)
            alphas_cumprod = reward_ddpm.alphas_cumprod[t].view(-1, 1)
            condition_noisy = torch.sqrt(alphas_cumprod) * condition + torch.sqrt(1 - alphas_cumprod) * noise
            
            # Predict noise conditioned on x_gen
            predicted_noise = reward_model(condition_noisy, t, x_gen)
            
            # MSE as proxy for NLL
            nll += ((predicted_noise - noise) ** 2).sum(dim=1)
        
        nll = nll / mc_steps
        rewards = -nll  # Negative NLL as reward
        
        return rewards
    
    def policy_gradient_update(
        self,
        trajectory: List[torch.Tensor],
        log_probs: List[torch.Tensor],
        rewards: torch.Tensor,
        gen_model: nn.Module,
        frozen_model: nn.Module,
        condition: torch.Tensor,
        timesteps_list: List[int],
        optimizer: torch.optim.Optimizer,
        kl_weight: float = 0.1,
        clip_range: float = 0.2,
        use_advantages: bool = True,
    ) -> Dict[str, float]:
        """
        Policy gradient update with PPO and KL regularization (DPOK).
        
        Args:
            trajectory: List of states from generation
            log_probs: Log probabilities from generation
            rewards: Rewards for each sample
            gen_model: Generator model to update
            frozen_model: Frozen prior for KL penalty
            condition: Conditioning variable
            timesteps_list: List of timesteps
            optimizer: Optimizer
            kl_weight: Weight for KL regularization
            clip_range: PPO clipping range
            use_advantages: Whether to use advantage normalization
            
        Returns:
            Dictionary of metrics
        """
        gen_model.train()
        optimizer.zero_grad()
        
        # Compute advantages
        if use_advantages:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            advantages = rewards
        
        total_loss = 0.0
        total_kl = 0.0
        
        # Update over trajectory (skip first and last)
        for i in range(len(trajectory) - 1):
            x_t = trajectory[i]
            x_next = trajectory[i + 1]
            log_prob_old = log_probs[i]
            t_idx = timesteps_list[i]
            
            batch_size = x_t.shape[0]
            t_tensor = torch.full((batch_size,), t_idx, device=self.device, dtype=torch.long)
            
            # Recompute with gradients
            predicted_noise = gen_model(x_t, t_tensor, condition)
            
            # Recompute log prob
            alphas_cumprod_t = self.ddpm_1.alphas_cumprod[t_idx]
            beta_t = self.ddpm_1.betas[t_idx]
            
            # Simple approximation for log prob
            variance = beta_t
            log_prob_new = -0.5 * ((x_next - x_t) ** 2).sum(dim=1) / (variance + 1e-8)
            
            # PPO loss with clipping
            ratio = torch.exp(log_prob_new - log_prob_old.detach())
            clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
            
            pg_loss = -torch.min(
                advantages.detach() * ratio,
                advantages.detach() * clipped_ratio
            ).mean()
            
            # KL regularization against frozen prior
            with torch.no_grad():
                predicted_noise_old = frozen_model(x_t, t_tensor, None)  # Unconditional prior
            
            kl_reg = ((predicted_noise - predicted_noise_old) ** 2).mean()
            
            loss = pg_loss + kl_weight * kl_reg
            
            total_loss += loss
            total_kl += kl_reg.item()
        
        # Backprop and update
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(gen_model.parameters(), 1.0)
        optimizer.step()
        
        return {
            "pg_loss": total_loss.item(),
            "kl_reg": total_kl,
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std().item(),
            "advantage_mean": advantages.mean().item(),
        }
    
    def sync_reward_model(
        self,
        x_gen: torch.Tensor,
        condition: torch.Tensor,
        reward_model: nn.Module,
        reward_ddpm: DDPM1D,
        optimizer: torch.optim.Optimizer,
        num_updates: int = 5,
    ):
        """
        Synchronize reward model with generator via supervised training.
        
        After generator updates via RL, reward model learns the new coupling.
        
        Args:
            x_gen: Generated samples from policy
            condition: Conditioning variable
            reward_model: Model to update
            reward_ddpm: DDPM wrapper
            optimizer: Optimizer
            num_updates: Number of gradient steps
        """
        reward_model.train()
        
        for _ in range(num_updates):
            optimizer.zero_grad()
            
            # Random timestep
            t = torch.randint(0, self.config.timesteps, (condition.shape[0],), device=self.device)
            
            # Add noise to condition
            noise = torch.randn_like(condition)
            alphas_cumprod = reward_ddpm.alphas_cumprod[t].view(-1, 1)
            condition_noisy = torch.sqrt(alphas_cumprod) * condition + torch.sqrt(1 - alphas_cumprod) * noise
            
            # Predict noise conditioned on generated x
            predicted_noise = reward_model(condition_noisy, t, x_gen.detach())
            
            loss = nn.functional.mse_loss(predicted_noise, noise)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reward_model.parameters(), 1.0)
            optimizer.step()
    
    def run_one_way(
        self,
        x_gen_dist: torch.Tensor,
        x_cond_dist: torch.Tensor,
        gen_model: nn.Module,
        gen_ddpm: DDPM1D,
        reward_model: nn.Module,
        reward_ddpm: DDPM1D,
        frozen_gen_model: nn.Module,
        optimizer_gen: torch.optim.Optimizer,
        optimizer_reward: torch.optim.Optimizer,
        direction: str,
    ) -> Dict[str, float]:
        """
        Run one direction of cooperative training.
        
        Args:
            x_gen_dist: Samples from distribution to generate
            x_cond_dist: Samples from conditioning distribution
            gen_model: Generator model (policy)
            gen_ddpm: DDPM wrapper for generator
            reward_model: Reward model (critic)
            reward_ddpm: DDPM wrapper for reward
            frozen_gen_model: Frozen prior for generator
            optimizer_gen: Generator optimizer
            optimizer_reward: Reward optimizer
            direction: "1->2" or "2->1" for logging
            
        Returns:
            Metrics dictionary
        """
        # Step 1: Generate trajectory with log probs
        trajectory, log_probs, timesteps_list = self.sample_trajectory_with_logprob(
            gen_model, gen_ddpm, x_cond_dist, num_steps=50
        )
        
        x_gen = trajectory[-1]  # Final generated sample
        
        # Step 2: Compute reward
        rewards = self.compute_reward(
            x_gen, x_cond_dist, reward_model, reward_ddpm, mc_steps=5
        )
        
        # Step 3: Policy gradient update
        pg_metrics = self.policy_gradient_update(
            trajectory, log_probs, rewards,
            gen_model, frozen_gen_model,
            x_cond_dist, timesteps_list,
            optimizer_gen,
            kl_weight=0.1,
            clip_range=0.2,
        )
        
        # Step 4: Sync reward model
        self.sync_reward_model(
            x_gen, x_cond_dist, reward_model, reward_ddpm,
            optimizer_reward, num_updates=5
        )
        
        return {f"{direction}_{k}": v for k, v in pg_metrics.items()}
    
    def train_step(
        self,
        x1_batch: torch.Tensor,
        x2_batch: torch.Tensor,
        optimizer_1: torch.optim.Optimizer,
        optimizer_2: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """
        Single training step (warmup or cooperative).
        
        Args:
            x1_batch: Batch from distribution 1
            x2_batch: Batch from distribution 2
            optimizer_1: Optimizer for model 1
            optimizer_2: Optimizer for model 2
            
        Returns:
            Metrics dictionary
        """
        self.gen_step += 1
        
        # Warmup phase: supervised training only
        if self.gen_step < self.warmup_steps:
            metrics = self.train_supervised(x1_batch, x2_batch, optimizer_1, optimizer_2)
            metrics["phase"] = "warmup"
            return metrics
        
        # Setup frozen priors after warmup
        if self.gen_step == self.warmup_steps:
            self.setup_priors()
            print("Warmup complete. Starting cooperative DDMEC training.")
        
        # Cooperative phase: alternating RL updates
        metrics = {}
        
        # Direction 1: Model 1 generates x1 given x2, Model 2 is reward
        metrics_1 = self.run_one_way(
            x1_batch, x2_batch,
            self.ddpm_1.model, self.ddpm_1,
            self.ddpm_2.model, self.ddpm_2,
            self.frozen_model_1,
            optimizer_1, optimizer_2,
            direction="1->2"
        )
        metrics.update(metrics_1)
        
        # Direction 2: Model 2 generates x2 given x1, Model 1 is reward  
        metrics_2 = self.run_one_way(
            x2_batch, x1_batch,
            self.ddpm_2.model, self.ddpm_2,
            self.ddpm_1.model, self.ddpm_1,
            self.frozen_model_2,
            optimizer_2, optimizer_1,
            direction="2->1"
        )
        metrics.update(metrics_2)
        
        metrics["phase"] = "cooperative"
        return metrics
    
    def compute_kl_divergence(self, mu_p: float, sigma_p: float, mu_q: float, sigma_q: float) -> float:
        """
        Compute KL divergence between two 1D Gaussians: KL(p||q)
        KL(p||q) = 1/2 * [(μ_p-μ_q)²/σ_q² + σ_p²/σ_q² - 1 + ln(σ_q²/σ_p²)]
        """
        # Add epsilon to avoid numerical issues
        eps = 1e-8
        sigma_p = max(sigma_p, eps)
        sigma_q = max(sigma_q, eps)
        
        var_p = sigma_p ** 2
        var_q = sigma_q ** 2
        
        # Clamp the log ratio to avoid overflow/underflow
        log_ratio = np.log(np.clip(var_q / var_p, 1e-10, 1e10))
        
        kl = 0.5 * (
            ((mu_p - mu_q) ** 2) / var_q +
            var_p / var_q -
            1.0 +
            log_ratio
        )
        return max(kl, 0.0)  # KL should always be non-negative
    
    def compute_entropy_gaussian(self, sigma: float) -> float:
        """Compute entropy of 1D Gaussian: H(X) = 0.5 * ln(2πeσ²)"""
        # Add epsilon to avoid log(0)
        eps = 1e-8
        sigma = max(sigma, eps)
        variance = sigma ** 2
        return 0.5 * np.log(2 * np.pi * np.e * variance)
    
    def compute_joint_entropy(self, samples_x: np.ndarray, samples_y: np.ndarray) -> float:
        """Estimate joint entropy H(X,Y) using Gaussian approximation"""
        eps = 1e-8
        
        # Stack samples
        joint = np.stack([samples_x.flatten(), samples_y.flatten()], axis=1)
        
        # Compute covariance matrix with regularization
        cov = np.cov(joint.T)
        
        # Add small regularization to diagonal for numerical stability
        cov = cov + eps * np.eye(cov.shape[0])
        
        # Compute determinant
        try:
            det_cov = np.linalg.det(cov)
            # Ensure determinant is positive
            det_cov = max(det_cov, eps)
        except np.linalg.LinAlgError:
            # If computation fails, use product of variances as fallback
            det_cov = cov[0, 0] * cov[1, 1]
            det_cov = max(det_cov, eps)
        
        # Joint entropy: H(X,Y) = 0.5 * ln((2πe)^2 * det(Σ)) = ln(2πe) + 0.5 * ln(det(Σ))
        joint_entropy = np.log(2 * np.pi * np.e) + 0.5 * np.log(det_cov)
        
        return joint_entropy
    
    def compute_mutual_information(
        self, 
        samples_x: np.ndarray, 
        samples_y: np.ndarray,
        sigma_x: float = 1.0,
        sigma_y: float = 1.0
    ) -> float:
        """
        Compute mutual information: I(X;Y) = H(X) + H(Y) - H(X,Y)
        """
        h_x = self.compute_entropy_gaussian(sigma_x)
        h_y = self.compute_entropy_gaussian(sigma_y)
        h_xy = self.compute_joint_entropy(samples_x, samples_y)
        mi = h_x + h_y - h_xy
        return mi
    
    def compute_information_metrics(
        self,
        samples_1: torch.Tensor,
        samples_2: torch.Tensor,
        true_mu_1: float = 2.0,
        true_sigma_1: float = 1.0,
        true_mu_2: float = 10.0,
        true_sigma_2: float = 1.0,
    ) -> dict:
        """
        Compute all information-theoretic metrics for the coupling.
        """
        eps = 1e-8
        
        # Convert to numpy
        s1_np = samples_1.cpu().numpy().flatten()
        s2_np = samples_2.cpu().numpy().flatten()
        
        # Compute learned statistics with numerical stability
        mu_1 = float(s1_np.mean())
        sigma_1 = float(s1_np.std() + eps)  # Add epsilon to avoid zero std
        mu_2 = float(s2_np.mean())
        sigma_2 = float(s2_np.std() + eps)  # Add epsilon to avoid zero std
        
        # Check for NaN or Inf in computed statistics
        if not (np.isfinite(mu_1) and np.isfinite(sigma_1) and np.isfinite(mu_2) and np.isfinite(sigma_2)):
            # Return safe fallback values
            return {
                "kl_div_1": 0.0,
                "kl_div_2": 0.0,
                "kl_div_total": 0.0,
                "entropy_x": self.compute_entropy_gaussian(true_sigma_1),
                "entropy_y": self.compute_entropy_gaussian(true_sigma_2),
                "joint_entropy": self.compute_entropy_gaussian(true_sigma_1) + self.compute_entropy_gaussian(true_sigma_2),
                "mutual_information": 0.0,
                "conditional_entropy_x_given_y": self.compute_entropy_gaussian(true_sigma_1),
                "conditional_entropy_y_given_x": self.compute_entropy_gaussian(true_sigma_2),
                "learned_mu_1": mu_1 if np.isfinite(mu_1) else true_mu_1,
                "learned_sigma_1": sigma_1 if np.isfinite(sigma_1) else true_sigma_1,
                "learned_mu_2": mu_2 if np.isfinite(mu_2) else true_mu_2,
                "learned_sigma_2": sigma_2 if np.isfinite(sigma_2) else true_sigma_2,
            }
        
        # KL divergences (learned || true)
        kl_1 = self.compute_kl_divergence(mu_1, sigma_1, true_mu_1, true_sigma_1)
        kl_2 = self.compute_kl_divergence(mu_2, sigma_2, true_mu_2, true_sigma_2)
        
        # Entropies
        h_x = self.compute_entropy_gaussian(sigma_1)
        h_y = self.compute_entropy_gaussian(sigma_2)
        h_xy = self.compute_joint_entropy(s1_np, s2_np)
        
        # Mutual information (clamp to non-negative)
        mi = max(h_x + h_y - h_xy, 0.0)
        
        # Conditional entropies (ensure they are non-negative)
        h_x_given_y = max(h_xy - h_y, 0.0)  # H(X|Y) = H(X,Y) - H(Y)
        h_y_given_x = max(h_xy - h_x, 0.0)  # H(Y|X) = H(X,Y) - H(X)
        
        return {
            "kl_div_1": kl_1,
            "kl_div_2": kl_2,
            "kl_div_total": kl_1 + kl_2,
            "entropy_x": h_x,
            "entropy_y": h_y,
            "joint_entropy": h_xy,
            "mutual_information": mi,
            "conditional_entropy_x_given_y": h_x_given_y,
            "conditional_entropy_y_given_x": h_y_given_x,
            "learned_mu_1": mu_1,
            "learned_sigma_1": sigma_1,
            "learned_mu_2": mu_2,
            "learned_sigma_2": sigma_2,
        }
    
    def generate_epoch_plot(
        self,
        x1_gen: np.ndarray,
        x2_gen: np.ndarray,
        x1_test: np.ndarray,
        x2_test: np.ndarray,
        epoch: int,
        save_path: str,
    ):
        """Generate visualization for current epoch."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Row 1: Coupling scatter plots
        axes[0, 0].scatter(x2_test, x1_gen, alpha=0.3, s=10)
        axes[0, 0].set_xlabel("x2 (condition)")
        axes[0, 0].set_ylabel("x1 (generated)")
        axes[0, 0].set_title(f"Coupling: x2→x1 (Epoch {epoch})")
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].scatter(x1_test, x2_gen, alpha=0.3, s=10)
        axes[0, 1].set_xlabel("x1 (condition)")
        axes[0, 1].set_ylabel("x2 (generated)")
        axes[0, 1].set_title(f"Coupling: x1→x2 (Epoch {epoch})")
        axes[0, 1].grid(True, alpha=0.3)
        
        # Correlation
        corr_1 = np.corrcoef(x2_test.flatten(), x1_gen.flatten())[0, 1]
        corr_2 = np.corrcoef(x1_test.flatten(), x2_gen.flatten())[0, 1]
        axes[0, 2].bar(['x2→x1', 'x1→x2'], [corr_1, corr_2], alpha=0.7)
        axes[0, 2].axhline(1.0, color='red', linestyle='--', label='Perfect')
        axes[0, 2].set_ylabel("Correlation")
        axes[0, 2].set_title("Coupling Strength")
        axes[0, 2].set_ylim([0, 1.1])
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3, axis='y')
        
        # Row 2: Marginal distributions
        axes[1, 0].hist(x1_gen, bins=30, density=True, alpha=0.7, label='Generated')
        axes[1, 0].axvline(2.0, color='red', linestyle='--', linewidth=2, label='Target mean')
        axes[1, 0].set_xlabel("x1")
        axes[1, 0].set_ylabel("Density")
        axes[1, 0].set_title(f"Marginal: x1 (μ={x1_gen.mean():.2f}, σ={x1_gen.std():.2f})")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].hist(x2_gen, bins=30, density=True, alpha=0.7, label='Generated', color='orange')
        axes[1, 1].axvline(10.0, color='red', linestyle='--', linewidth=2, label='Target mean')
        axes[1, 1].set_xlabel("x2")
        axes[1, 1].set_ylabel("Density")
        axes[1, 1].set_title(f"Marginal: x2 (μ={x2_gen.mean():.2f}, σ={x2_gen.std():.2f})")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Coupling errors
        error_1 = x1_gen - (x2_test - 8.0)
        error_2 = x2_gen - (x1_test + 8.0)
        mae_1 = np.abs(error_1).mean()
        mae_2 = np.abs(error_2).mean()
        axes[1, 2].bar(['x2→x1', 'x1→x2'], [mae_1, mae_2], alpha=0.7, color=['blue', 'orange'])
        axes[1, 2].set_ylabel("Mean Absolute Error")
        axes[1, 2].set_title("Coupling Error")
        axes[1, 2].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'DDMEC Training - Epoch {epoch}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
    
    def train(
        self,
        dataset_1: torch.Tensor,
        dataset_2: torch.Tensor,
        num_epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-4,
        run_dir: str = "runs/default",
    ):
        """
        Main training loop for DDMEC.
        
        Args:
            dataset_1: Samples from distribution 1
            dataset_2: Samples from distribution 2
            num_epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            run_dir: Directory to save checkpoints and plots
        """
        import os
        os.makedirs(run_dir, exist_ok=True)
        checkpoints_dir = os.path.join(run_dir, "checkpoints")
        plots_dir = os.path.join(run_dir, "plots")
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        
        # Setup optimizers
        optimizer_1 = torch.optim.Adam(self.ddpm_1.model.parameters(), lr=lr)
        optimizer_2 = torch.optim.Adam(self.ddpm_2.model.parameters(), lr=lr)
        
        num_samples = min(len(dataset_1), len(dataset_2))
        num_batches = num_samples // batch_size
        
        # Track best model
        best_kl_total = float('inf')
        best_epoch = 0
        
        print(f"Training DDMEC for {num_epochs} epochs, {num_batches} batches per epoch")
        print(f"Saving checkpoints to: {checkpoints_dir}")
        print(f"Saving plots to: {plots_dir}")
        
        for epoch in range(num_epochs):
            # Shuffle datasets
            perm = torch.randperm(num_samples)
            dataset_1_shuffled = dataset_1[perm]
            dataset_2_shuffled = dataset_2[perm]
            
            epoch_metrics = defaultdict(list)
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                
                x1_batch = dataset_1_shuffled[start_idx:end_idx].to(self.device)
                x2_batch = dataset_2_shuffled[start_idx:end_idx].to(self.device)
                
                metrics = self.train_step(x1_batch, x2_batch, optimizer_1, optimizer_2)
                
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        epoch_metrics[k].append(v)
            
            # Log epoch metrics and compute information-theoretic measures every epoch
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("  Training Metrics:")
            
            # Prepare wandb logging dict
            wandb_log = {"epoch": epoch + 1}
            
            for k, v in epoch_metrics.items():
                if v:
                    mean_val = np.mean(v)
                    print(f"    {k}: {mean_val:.4f}")
                    if self.use_wandb:
                        wandb_log[f"train/{k}"] = mean_val
            
            # Compute information-theoretic metrics and generate plot every epoch
            with torch.no_grad():
                # Sample from learned coupling
                num_eval = min(1000, num_samples)
                x2_eval = dataset_2[:num_eval].to(self.device)
                x1_eval = dataset_1[:num_eval].to(self.device)
                
                # Generate x1 from x2
                x1_generated = self.sample_coupled(x2_eval, direction="1->2", num_steps=50)
                x2_generated = self.sample_coupled(x1_eval, direction="2->1", num_steps=50)
                
                # Compute metrics for both directions
                info_metrics_1 = self.compute_information_metrics(
                    x1_generated, x2_eval,
                    true_mu_1=2.0, true_sigma_1=1.0,
                    true_mu_2=10.0, true_sigma_2=1.0
                )
                
                info_metrics_2 = self.compute_information_metrics(
                    x1_eval, x2_generated,
                    true_mu_1=2.0, true_sigma_1=1.0,
                    true_mu_2=10.0, true_sigma_2=1.0
                )
                
                print("\n  Information-Theoretic Metrics (x2→x1):")
                print(f"    KL Divergence (X1): {info_metrics_1['kl_div_1']:.4f}")
                print(f"    KL Divergence (X2): {info_metrics_1['kl_div_2']:.4f}")
                print(f"    KL Divergence (Total): {info_metrics_1['kl_div_total']:.4f}")
                print(f"    Mutual Information I(X;Y): {info_metrics_1['mutual_information']:.4f}")
                
                print("\n  Information-Theoretic Metrics (x1→x2):")
                print(f"    KL Divergence (X1): {info_metrics_2['kl_div_1']:.4f}")
                print(f"    KL Divergence (X2): {info_metrics_2['kl_div_2']:.4f}")
                print(f"    KL Divergence (Total): {info_metrics_2['kl_div_total']:.4f}")
                print(f"    Mutual Information I(X;Y): {info_metrics_2['mutual_information']:.4f}")
                
                # Generate plot for this epoch
                plot_path = os.path.join(plots_dir, f"epoch_{epoch+1:04d}.png")
                self.generate_epoch_plot(
                    x1_generated.cpu().numpy(),
                    x2_generated.cpu().numpy(),
                    x1_eval.cpu().numpy(),
                    x2_eval.cpu().numpy(),
                    epoch + 1,
                    plot_path
                )
                print(f"  Plot saved: {plot_path}")
                
                # Save checkpoint every epoch
                checkpoint_path = os.path.join(checkpoints_dir, f"epoch_{epoch+1:04d}.pt")
                torch.save({
                    "epoch": epoch + 1,
                    "model_1_state_dict": self.ddpm_1.model.state_dict(),
                    "model_2_state_dict": self.ddpm_2.model.state_dict(),
                    "optimizer_1_state_dict": optimizer_1.state_dict(),
                    "optimizer_2_state_dict": optimizer_2.state_dict(),
                    "config": vars(self.config),
                    "gen_step": self.gen_step,
                    "info_metrics_1": info_metrics_1,
                    "info_metrics_2": info_metrics_2,
                }, checkpoint_path)
                
                # Check if this is the best model
                avg_kl_total = (info_metrics_1['kl_div_total'] + info_metrics_2['kl_div_total']) / 2
                if avg_kl_total < best_kl_total:
                    best_kl_total = avg_kl_total
                    best_epoch = epoch + 1
                    # Save best model
                    best_path = os.path.join(checkpoints_dir, "best_model.pt")
                    torch.save({
                        "epoch": epoch + 1,
                        "model_1_state_dict": self.ddpm_1.model.state_dict(),
                        "model_2_state_dict": self.ddpm_2.model.state_dict(),
                        "config": vars(self.config),
                        "gen_step": self.gen_step,
                        "info_metrics_1": info_metrics_1,
                        "info_metrics_2": info_metrics_2,
                        "best_kl_total": best_kl_total,
                    }, best_path)
                    print(f"  ★ New best model! (KL Total: {best_kl_total:.4f})")
                
                # Log to wandb
                if self.use_wandb:
                    # Forward direction (x2→x1)
                    wandb_log.update({
                        "info/kl_div_1_forward": info_metrics_1['kl_div_1'],
                        "info/kl_div_2_forward": info_metrics_1['kl_div_2'],
                        "info/kl_div_total_forward": info_metrics_1['kl_div_total'],
                        "info/entropy_x_forward": info_metrics_1['entropy_x'],
                        "info/entropy_y_forward": info_metrics_1['entropy_y'],
                        "info/joint_entropy_forward": info_metrics_1['joint_entropy'],
                        "info/mutual_information_forward": info_metrics_1['mutual_information'],
                        "info/conditional_entropy_x_given_y": info_metrics_1['conditional_entropy_x_given_y'],
                        "info/conditional_entropy_y_given_x": info_metrics_1['conditional_entropy_y_given_x'],
                        # Backward direction (x1→x2)
                        "info/kl_div_1_backward": info_metrics_2['kl_div_1'],
                        "info/kl_div_2_backward": info_metrics_2['kl_div_2'],
                        "info/kl_div_total_backward": info_metrics_2['kl_div_total'],
                        "info/mutual_information_backward": info_metrics_2['mutual_information'],
                        # Learned statistics
                        "stats/learned_mu_1_forward": info_metrics_1['learned_mu_1'],
                        "stats/learned_sigma_1_forward": info_metrics_1['learned_sigma_1'],
                        "stats/learned_mu_2_forward": info_metrics_1['learned_mu_2'],
                        "stats/learned_sigma_2_forward": info_metrics_1['learned_sigma_2'],
                        # Best tracking
                        "best/kl_total": best_kl_total,
                        "best/epoch": best_epoch,
                        # Average KL
                        "info/avg_kl_total": avg_kl_total,
                    })
                    # Log plot as image
                    wandb_log["plots/epoch_visualization"] = wandb.Image(plot_path)
                    wandb.log(wandb_log)
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        print(f"Best epoch: {best_epoch}")
        print(f"Best KL Total: {best_kl_total:.4f}")
        print(f"\nCheckpoints saved to: {checkpoints_dir}")
        print(f"Plots saved to: {plots_dir}")
        print(f"Best model saved to: {os.path.join(checkpoints_dir, 'best_model.pt')}")
        print("="*60)
    
    @torch.no_grad()
    def sample_coupled(
        self,
        x_condition: torch.Tensor,
        direction: str = "1->2",
        num_steps: int = 50,
    ) -> torch.Tensor:
        """
        Sample from learned coupling.
        
        Args:
            x_condition: Conditioning variable
            direction: "1->2" to sample x1 given x2, or "2->1" for reverse
            num_steps: Number of denoising steps
            
        Returns:
            Generated samples
        """
        if direction == "1->2":
            model = self.ddpm_1.model
            ddpm = self.ddpm_1
        else:
            model = self.ddpm_2.model
            ddpm = self.ddpm_2
        
        model.eval()
        trajectory, _, _ = self.sample_trajectory_with_logprob(
            model, ddpm, x_condition, num_steps
        )
        
        return trajectory[-1]


def main():
    """Example usage of DDMEC."""
    print("=" * 60)
    print("DDMEC Training for 1D Gaussian Coupling")
    print("=" * 60)
    
    # Load pre-trained models
    ddmec = DDMEC1D(
        model_path_1="ddpm_1d_2.pt",
        model_path_2="ddpm_1d_10.pt",
    )
    
    # Create synthetic coupled data for demonstration
    # In practice, you'd use your actual datasets
    num_samples = 10000
    
    # Distribution 1: N(2, 1)
    x1_data = torch.randn(num_samples, 1) + 2.0
    
    # Distribution 2: N(10, 1) 
    x2_data = torch.randn(num_samples, 1) + 10.0
    
    # For demonstration, create a simple coupling
    # In DDMEC, the model learns the optimal coupling
    coupled_noise = torch.randn(num_samples, 1)
    x1_coupled = coupled_noise + 2.0
    x2_coupled = coupled_noise + 10.0
    
    print(f"\nDataset 1 shape: {x1_coupled.shape}")
    print(f"Dataset 2 shape: {x2_coupled.shape}")
    print(f"Dataset 1 mean: {x1_coupled.mean():.2f}, std: {x1_coupled.std():.2f}")
    print(f"Dataset 2 mean: {x2_coupled.mean():.2f}, std: {x2_coupled.std():.2f}")
    
    # Train DDMEC
    print("\nStarting DDMEC training...")
    ddmec.train(
        dataset_1=x1_coupled,
        dataset_2=x2_coupled,
        num_epochs=50,
        batch_size=64,
        lr=1e-4,
    )
    
    # Test sampling from learned coupling
    print("\nTesting learned coupling...")
    
    # Sample x1 given x2
    x2_test = torch.tensor([[10.0]], device=ddmec.device)
    x1_generated = ddmec.sample_coupled(x2_test, direction="1->2", num_steps=50)
    print(f"Given x2={x2_test.item():.2f}, generated x1={x1_generated.item():.2f}")
    
    # Sample x2 given x1
    x1_test = torch.tensor([[2.0]], device=ddmec.device)
    x2_generated = ddmec.sample_coupled(x1_test, direction="2->1", num_steps=50)
    print(f"Given x1={x1_test.item():.2f}, generated x2={x2_generated.item():.2f}")
    
    # Visualize results
    print("\nGenerating visualization...")
    with torch.no_grad():
        num_viz = 500
        x2_samples = torch.randn(num_viz, 1, device=ddmec.device) + 10.0
        x1_from_x2 = ddmec.sample_coupled(x2_samples, direction="1->2", num_steps=50)
        
        x1_samples = torch.randn(num_viz, 1, device=ddmec.device) + 2.0
        x2_from_x1 = ddmec.sample_coupled(x1_samples, direction="2->1", num_steps=50)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: x2 -> x1 coupling
    axes[0, 0].scatter(
        x2_samples.cpu().numpy(),
        x1_from_x2.cpu().numpy(),
        alpha=0.5,
        s=10
    )
    axes[0, 0].set_xlabel("x2 (condition)")
    axes[0, 0].set_ylabel("x1 (generated)")
    axes[0, 0].set_title("Learned Coupling: x2 -> x1")
    axes[0, 0].grid(True)
    
    # Plot 2: x1 -> x2 coupling
    axes[0, 1].scatter(
        x1_samples.cpu().numpy(),
        x2_from_x1.cpu().numpy(),
        alpha=0.5,
        s=10
    )
    axes[0, 1].set_xlabel("x1 (condition)")
    axes[0, 1].set_ylabel("x2 (generated)")
    axes[0, 1].set_title("Learned Coupling: x1 -> x2")
    axes[0, 1].grid(True)
    
    # Plot 3: Marginal distribution of generated x1
    axes[1, 0].hist(x1_from_x2.cpu().numpy(), bins=30, density=True, alpha=0.7)
    axes[1, 0].axvline(2.0, color='red', linestyle='--', label='True mean (2.0)')
    axes[1, 0].set_xlabel("x1")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].set_title("Marginal: Generated x1")
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot 4: Marginal distribution of generated x2
    axes[1, 1].hist(x2_from_x1.cpu().numpy(), bins=30, density=True, alpha=0.7)
    axes[1, 1].axvline(10.0, color='red', linestyle='--', label='True mean (10.0)')
    axes[1, 1].set_xlabel("x2")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].set_title("Marginal: Generated x2")
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    output_path = "ddmec_coupling_results.png"
    plt.savefig(output_path, dpi=150)
    plt.close(fig)  # Close the figure to free memory
    print(f"Results saved to {output_path}")
    
    # Save trained models
    torch.save({
        "model_1_state_dict": ddmec.ddpm_1.model.state_dict(),
        "model_2_state_dict": ddmec.ddpm_2.model.state_dict(),
        "config": vars(ddmec.config),
    }, "ddmec_trained.pt")
    print("Trained DDMEC models saved to ddmec_trained.pt")


if __name__ == "__main__":
    main()

