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

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: tensorboard not installed. Install with: pip install tensorboard")


class ConditionalMLP(nn.Module):
    """Wrapper to make SmallMLP conditional by concatenating condition."""
    
    def __init__(self, base_model: SmallMLP):
        super().__init__()
        self.base_model = base_model
        # New input layer that takes [x, condition, time_emb]
        self.input_proj = nn.Linear(2 + 32, 64)  # 2 inputs + 32 time_emb -> 64
        
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
            # Unconditional: use zeros
            condition = torch.zeros_like(x)
        
        # Concatenate x and condition, then with time embedding
        inp = torch.cat([x, condition, t_emb], dim=-1)
        h = self.input_proj(inp)
        h = torch.relu(h)
        
        # Use rest of base model network (skip first layer)
        # Base model net: Linear(33, 64), SiLU, Linear(64, 64), SiLU, Linear(64, 1)
        # We'll just use our own small network for simplicity
        h = self.base_model.net[2](h)  # Second Linear layer
        h = self.base_model.net[3](h)  # SiLU
        h = self.base_model.net[4](h)  # Output layer
        
        return h


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
        use_tensorboard: bool = True,  # TensorBoard enabled by default
        # RL hyperparameters (for ablation studies)
        kl_weight: float = 0.5,  # KL regularization weight (Adnan used 0.1, we use higher for stability)
        ppo_clip: float = 0.1,   # PPO clipping range (Adnan used 0.2, we use smaller for stability)
        warmup_epochs: int = 10,  # Number of warmup epochs before RL
        rl_lr_scale: float = 0.1,  # Scale factor for LR during RL phase (lower = more stable)
    ):
        """
        Args:
            model_path_1: Path to first pre-trained model
            model_path_2: Path to second pre-trained model
            config: Configuration (will be loaded from checkpoints if available)
            use_wandb: Whether to log metrics to Weights & Biases
            use_tensorboard: Whether to log metrics to TensorBoard
            kl_weight: KL regularization weight for policy gradient (higher = more conservative)
            ppo_clip: PPO clipping range (smaller = more stable updates)
            warmup_epochs: Number of warmup epochs with supervised training only
            rl_lr_scale: Learning rate scale factor during RL phase
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        self.tb_writer = None  # Will be initialized in train()
        
        # RL hyperparameters
        self.kl_weight = kl_weight
        self.ppo_clip = ppo_clip
        self.warmup_epochs = warmup_epochs
        self.rl_lr_scale = rl_lr_scale
        
        print(f"RL Hyperparameters:")
        print(f"  KL Weight: {self.kl_weight}")
        print(f"  PPO Clip: {self.ppo_clip}")
        print(f"  Warmup Epochs: {self.warmup_epochs}")
        print(f"  RL LR Scale: {self.rl_lr_scale}")
        
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
        self.warmup_steps = 1000  # Will be updated based on warmup_epochs
        
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
        
        # Validate inputs - fail fast if data is corrupt
        if not torch.isfinite(x).all():
            raise ValueError(
                f"Initial noise contains NaN/Inf. This should never happen.\n"
                f"Check random seed/GPU state. Device: {self.device}"
            )
        
        if not torch.isfinite(condition).all():
            raise ValueError(
                f"Condition input contains NaN/Inf. Check your data preprocessing:\n"
                f"  Mean: {condition[torch.isfinite(condition)].mean() if torch.isfinite(condition).any() else 'N/A'}\n"
                f"  Std: {condition[torch.isfinite(condition)].std() if torch.isfinite(condition).any() else 'N/A'}\n"
                f"  NaN count: {torch.isnan(condition).sum().item()}\n"
                f"  Inf count: {torch.isinf(condition).sum().item()}"
            )
        
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
            
            # Diagnostic check - fail fast on numerical issues
            if not torch.isfinite(predicted_noise).all():
                nan_count = torch.isnan(predicted_noise).sum().item()
                inf_count = torch.isinf(predicted_noise).sum().item()
                print(f"\n{'='*70}")
                print(f"NUMERICAL INSTABILITY DETECTED at timestep {t_cur}/{self.config.timesteps}")
                print(f"{'='*70}")
                print(f"Input x:           mean={x.mean():.6f}, std={x.std():.6f}, range=[{x.min():.6f}, {x.max():.6f}]")
                print(f"Condition:         mean={condition.mean():.6f}, std={condition.std():.6f}")
                print(f"Predicted noise:   NaN={nan_count}, Inf={inf_count}")
                
                # Check model parameters
                model_nan_count = sum(torch.isnan(p).sum().item() for p in model.parameters())
                model_inf_count = sum(torch.isinf(p).sum().item() for p in model.parameters())
                print(f"Model weights:     NaN={model_nan_count}, Inf={model_inf_count}")
                
                print(f"\nPossible causes:")
                print(f"  1. Learning rate too high (current step: {self.gen_step})")
                print(f"  2. Gradient explosion (check gradient norms)")
                print(f"  3. Poor weight initialization")
                print(f"  4. Numerical instability in schedule (check beta values)")
                print(f"{'='*70}\n")
                raise RuntimeError("NaN/Inf detected in forward pass. Training cannot continue safely.")
            
            # Get schedule values
            alpha_t = ddpm.alphas[t_cur]
            alphas_cumprod_t = ddpm.alphas_cumprod[t_cur]
            beta_t = ddpm.betas[t_cur]
            
            if t_next > 0:
                alphas_cumprod_next = ddpm.alphas_cumprod[t_next]
            else:
                alphas_cumprod_next = torch.tensor(1.0, device=self.device)
            
            # Add epsilon ONLY for sqrt numerical stability (legitimate use)
            eps = 1e-10
            sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod_t.clamp(min=eps))
            sqrt_one_minus_alphas_cumprod_t = torch.sqrt((1 - alphas_cumprod_t).clamp(min=eps))
            
            # DDIM sampling step
            x0_pred = (x - sqrt_one_minus_alphas_cumprod_t * predicted_noise) / sqrt_alphas_cumprod_t
            
            # Compute mean
            if t_next > 0:
                sqrt_alphas_cumprod_next = torch.sqrt(alphas_cumprod_next.clamp(min=eps))
                sqrt_one_minus_alphas_cumprod_next = torch.sqrt((1 - alphas_cumprod_next).clamp(min=eps))
                x_next = sqrt_alphas_cumprod_next * x0_pred + sqrt_one_minus_alphas_cumprod_next * predicted_noise
            else:
                x_next = x0_pred
            
            # Check for divergence BEFORE it propagates
            if not torch.isfinite(x_next).all():
                print(f"\n{'='*70}")
                print(f"SAMPLING DIVERGED at timestep {t_cur}")
                print(f"{'='*70}")
                print(f"x_next:     NaN={torch.isnan(x_next).sum().item()}, Inf={torch.isinf(x_next).sum().item()}")
                print(f"x_next:     range=[{x_next[torch.isfinite(x_next)].min() if torch.isfinite(x_next).any() else float('nan'):.6f}, "
                      f"{x_next[torch.isfinite(x_next)].max() if torch.isfinite(x_next).any() else float('nan'):.6f}]")
                print(f"x0_pred:    range=[{x0_pred.min():.6f}, {x0_pred.max():.6f}]")
                print(f"pred_noise: range=[{predicted_noise.min():.6f}, {predicted_noise.max():.6f}]")
                print(f"\nThe model has become unstable. Training should be restarted with:")
                print(f"  - Lower learning rate")
                print(f"  - Better initialization")
                print(f"  - Gradient clipping")
                print(f"{'='*70}\n")
                raise RuntimeError("Sampling produced NaN/Inf. Model is unstable.")
            
            # Sanity check for extreme values that indicate divergence
            if x_next.abs().max() > 1e4:
                print(f"\nWARNING: Extreme values at timestep {t_cur}")
                print(f"  x_next max magnitude: {x_next.abs().max():.2e}")
                print(f"  This suggests the model is diverging.")
            
            # Compute log prob (for policy gradient)
            # log p(x_next | x, condition) ≈ -||x_next - mean||^2 / (2 * variance)
            variance = beta_t.clamp(min=eps)  # Legitimate: prevent division by zero
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
        kl_weight: float = 0.5,  # Increased from 0.1 to prevent divergence
        clip_range: float = 0.1,  # Reduced from 0.2 for stability
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
        
        # Step 3: Policy gradient update (using instance hyperparameters)
        pg_metrics = self.policy_gradient_update(
            trajectory, log_probs, rewards,
            gen_model, frozen_gen_model,
            x_cond_dist, timesteps_list,
            optimizer_gen,
            kl_weight=self.kl_weight,
            clip_range=self.ppo_clip,
        )
        
        # Step 4: Sync reward model (reduced updates for stability)
        self.sync_reward_model(
            x_gen, x_cond_dist, reward_model, reward_ddpm,
            optimizer_reward, num_updates=2  # Reduced from 5 to prevent drift
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
            # Gradually increase learning rate during warmup
            if self.gen_step % 100 == 0:
                warmup_progress = self.gen_step / self.warmup_steps
                # Linear warmup from 10% to 100% of target LR
                target_lr = getattr(self, 'target_lr', 1e-4)
                current_lr = target_lr * (0.1 + 0.9 * warmup_progress)
                for param_group in optimizer_1.param_groups:
                    param_group['lr'] = current_lr
                for param_group in optimizer_2.param_groups:
                    param_group['lr'] = current_lr
            return metrics
        
        # Setup frozen priors after warmup
        if self.gen_step == self.warmup_steps:
            # REDUCE learning rate for RL phase to prevent divergence
            target_lr = getattr(self, 'target_lr', 1e-4)
            rl_lr = target_lr * self.rl_lr_scale  # Use smaller LR for RL phase
            for param_group in optimizer_1.param_groups:
                param_group['lr'] = rl_lr
            for param_group in optimizer_2.param_groups:
                param_group['lr'] = rl_lr
            self.setup_priors()
            print(f"\n{'='*60}")
            print(f"Warmup complete! Starting cooperative DDMEC training")
            print(f"{'='*60}")
            print(f"  RL Learning Rate: {rl_lr:.2e} (scaled by {self.rl_lr_scale})")
            print(f"  KL Weight: {self.kl_weight}")
            print(f"  PPO Clip: {self.ppo_clip}")
            print(f"{'='*60}\n")
        
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
        eps = 1e-10  # Only for preventing log(0) and division by zero
        
        # Ensure positive variance
        if sigma_p <= 0 or sigma_q <= 0:
            raise ValueError(f"Standard deviations must be positive: sigma_p={sigma_p}, sigma_q={sigma_q}")
        
        var_p = sigma_p ** 2
        var_q = sigma_q ** 2
        
        # Compute log ratio
        log_ratio = np.log(var_q / var_p)
        
        kl = 0.5 * (
            ((mu_p - mu_q) ** 2) / var_q +
            var_p / var_q -
            1.0 +
            log_ratio
        )
        
        # KL divergence should be non-negative by definition
        if kl < -eps:
            print(f"WARNING: Negative KL divergence ({kl:.6f}). This suggests a numerical issue.")
        
        return max(kl, 0.0)
    
    def compute_entropy_gaussian(self, sigma: float) -> float:
        """Compute entropy of 1D Gaussian: H(X) = 0.5 * ln(2πeσ²)"""
        eps = 1e-10  # Only for preventing log(0)
        
        if sigma <= 0:
            raise ValueError(f"Standard deviation must be positive: sigma={sigma}")
        
        variance = sigma ** 2
        return 0.5 * np.log(2 * np.pi * np.e * variance)
    
    def compute_joint_entropy(self, samples_x: np.ndarray, samples_y: np.ndarray) -> float:
        """Estimate joint entropy H(X,Y) using Gaussian approximation"""
        eps = 1e-10  # Only for legitimate numerical stability in determinant
        
        # Stack samples
        joint = np.stack([samples_x.flatten(), samples_y.flatten()], axis=1)
        
        # Compute covariance matrix
        cov = np.cov(joint.T)
        
        # Add small regularization to diagonal ONLY if matrix is singular
        # This is legitimate because we're dealing with finite samples
        cov_regularized = cov + eps * np.eye(cov.shape[0])
        
        # Compute determinant
        try:
            det_cov = np.linalg.det(cov_regularized)
            
            if det_cov <= 0:
                # This suggests perfectly correlated or anti-correlated samples
                print(f"WARNING: Covariance matrix has non-positive determinant ({det_cov:.2e})")
                print(f"  Covariance matrix:\n{cov}")
                print(f"  This may indicate numerical issues or degenerate distributions.")
                # Use absolute value as emergency fallback
                det_cov = abs(det_cov) + eps
                
        except np.linalg.LinAlgError as e:
            raise RuntimeError(
                f"Failed to compute covariance determinant: {e}\n"
                f"Covariance matrix:\n{cov}\n"
                f"This indicates serious numerical problems in the samples."
            )
        
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
        Raises errors on invalid data rather than returning fallbacks.
        """
        eps = 1e-10  # Only for legitimate numerical stability
        
        # Convert to numpy
        s1_np = samples_1.cpu().numpy().flatten()
        s2_np = samples_2.cpu().numpy().flatten()
        
        # Check input validity first
        if not np.isfinite(s1_np).all():
            raise ValueError(
                f"Sample 1 contains NaN/Inf values:\n"
                f"  NaN: {np.isnan(s1_np).sum()}, Inf: {np.isinf(s1_np).sum()}\n"
                f"  The model has diverged. Cannot compute metrics."
            )
        
        if not np.isfinite(s2_np).all():
            raise ValueError(
                f"Sample 2 contains NaN/Inf values:\n"
                f"  NaN: {np.isnan(s2_np).sum()}, Inf: {np.isinf(s2_np).sum()}\n"
                f"  The model has diverged. Cannot compute metrics."
            )
        
        # Compute learned statistics
        mu_1 = float(s1_np.mean())
        sigma_1 = float(s1_np.std())
        mu_2 = float(s2_np.mean())
        sigma_2 = float(s2_np.std())
        
        # Add epsilon to std only if it's legitimately zero (constant samples)
        if sigma_1 < eps:
            print(f"WARNING: Sample 1 has near-zero variance (std={sigma_1:.2e}). All samples are identical.")
            sigma_1 = eps
        if sigma_2 < eps:
            print(f"WARNING: Sample 2 has near-zero variance (std={sigma_2:.2e}). All samples are identical.")
            sigma_2 = eps
        
        # Final check - these should NEVER be non-finite after the above checks
        if not all(np.isfinite([mu_1, sigma_1, mu_2, sigma_2])):
            raise RuntimeError(
                f"Computed statistics are non-finite:\n"
                f"  mu_1={mu_1}, sigma_1={sigma_1}\n"
                f"  mu_2={mu_2}, sigma_2={sigma_2}\n"
                f"  This indicates a serious numerical problem."
            )
        
        # KL divergences (learned || true)
        kl_1 = self.compute_kl_divergence(mu_1, sigma_1, true_mu_1, true_sigma_1)
        kl_2 = self.compute_kl_divergence(mu_2, sigma_2, true_mu_2, true_sigma_2)
        
        # Entropies
        h_x = self.compute_entropy_gaussian(sigma_1)
        h_y = self.compute_entropy_gaussian(sigma_2)
        h_xy = self.compute_joint_entropy(s1_np, s2_np)
        
        # Mutual information
        mi = h_x + h_y - h_xy
        
        # Warn if MI is negative (theoretically impossible, indicates numerical issues)
        if mi < -eps:
            print(f"WARNING: Negative mutual information ({mi:.6f}). This suggests numerical instability.")
            print(f"  H(X)={h_x:.6f}, H(Y)={h_y:.6f}, H(X,Y)={h_xy:.6f}")
        
        # Conditional entropies
        h_x_given_y = h_xy - h_y  # H(X|Y) = H(X,Y) - H(Y)
        h_y_given_x = h_xy - h_x  # H(Y|X) = H(X,Y) - H(X)
        
        # Warn if conditional entropies are negative
        if h_x_given_y < -eps:
            print(f"WARNING: Negative H(X|Y) = {h_x_given_y:.6f}")
        if h_y_given_x < -eps:
            print(f"WARNING: Negative H(Y|X) = {h_y_given_x:.6f}")
        
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
    
    def _generate_metrics_grid_plot(self, epoch: int, save_dir: str):
        """
        Generate a 4x4 grid plot showing all metrics up to current epoch.
        Creates comprehensive visualizations for research papers.
        """
        if not hasattr(self, 'local_log') or not self.local_log.get('epochs'):
            return
        
        epochs_data = self.local_log['epochs']
        if len(epochs_data) == 0:
            return
        
        # Extract data from epochs
        epochs = [e['epoch'] for e in epochs_data]
        
        # Forward metrics
        kl_1_fwd = [e['info_metrics_forward']['kl_div_1'] for e in epochs_data]
        kl_2_fwd = [e['info_metrics_forward']['kl_div_2'] for e in epochs_data]
        kl_total_fwd = [e['info_metrics_forward']['kl_div_total'] for e in epochs_data]
        mi_fwd = [e['info_metrics_forward']['mutual_information'] for e in epochs_data]
        entropy_x = [e['info_metrics_forward']['entropy_x'] for e in epochs_data]
        entropy_y = [e['info_metrics_forward']['entropy_y'] for e in epochs_data]
        joint_entropy = [e['info_metrics_forward']['joint_entropy'] for e in epochs_data]
        cond_entropy_x_y = [e['info_metrics_forward']['conditional_entropy_x_given_y'] for e in epochs_data]
        cond_entropy_y_x = [e['info_metrics_forward']['conditional_entropy_y_given_x'] for e in epochs_data]
        mu_1_fwd = [e['info_metrics_forward']['learned_mu_1'] for e in epochs_data]
        sigma_1_fwd = [e['info_metrics_forward']['learned_sigma_1'] for e in epochs_data]
        mu_2_fwd = [e['info_metrics_forward']['learned_mu_2'] for e in epochs_data]
        sigma_2_fwd = [e['info_metrics_forward']['learned_sigma_2'] for e in epochs_data]
        
        # Backward metrics
        kl_total_bwd = [e['info_metrics_backward']['kl_div_total'] for e in epochs_data]
        mi_bwd = [e['info_metrics_backward']['mutual_information'] for e in epochs_data]
        
        # Coupling metrics
        corr_x2_x1 = [e['coupling_metrics']['corr_x2_to_x1'] for e in epochs_data]
        corr_x1_x2 = [e['coupling_metrics']['corr_x1_to_x2'] for e in epochs_data]
        mae_x2_x1 = [e['coupling_metrics']['mae_x2_to_x1'] for e in epochs_data]
        mae_x1_x2 = [e['coupling_metrics']['mae_x1_to_x2'] for e in epochs_data]
        
        # Best tracking
        best_kl = [e['best']['kl_total'] for e in epochs_data]
        
        # Training info
        learning_rates = [e['learning_rate'] for e in epochs_data]
        phases = [1 if e['phase'] == 'cooperative' else 0 for e in epochs_data]
        avg_kl = [e['avg_kl_total'] for e in epochs_data]
        
        # Create 4x4 grid
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        fig.suptitle(f'DDMEC Training Metrics - Epoch {epoch}', fontsize=16, fontweight='bold')
        
        # Row 1: KL Divergence metrics
        # Plot 1: KL Divergence Forward
        ax = axes[0, 0]
        ax.plot(epochs, kl_1_fwd, 'b-', label='KL₁ (X₁)', linewidth=2)
        ax.plot(epochs, kl_2_fwd, 'r-', label='KL₂ (X₂)', linewidth=2)
        ax.plot(epochs, kl_total_fwd, 'g--', label='KL Total', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('KL Divergence')
        ax.set_title('KL Divergence (Forward: x2→x1)')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([1, max(epochs)])
        
        # Plot 2: KL Divergence Comparison
        ax = axes[0, 1]
        ax.plot(epochs, kl_total_fwd, 'b-', label='Forward (x2→x1)', linewidth=2)
        ax.plot(epochs, kl_total_bwd, 'r-', label='Backward (x1→x2)', linewidth=2)
        ax.plot(epochs, avg_kl, 'g--', label='Average', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('KL Total')
        ax.set_title('KL Divergence Comparison')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([1, max(epochs)])
        
        # Plot 3: Best KL Tracking
        ax = axes[0, 2]
        ax.plot(epochs, best_kl, 'g-', linewidth=2)
        ax.fill_between(epochs, best_kl, alpha=0.3, color='green')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Best KL Total')
        ax.set_title('Best KL Total (Lower is Better)')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([1, max(epochs)])
        
        # Plot 4: Learning Rate & Phase
        ax = axes[0, 3]
        ax2 = ax.twinx()
        l1, = ax.plot(epochs, learning_rates, 'b-', label='Learning Rate', linewidth=2)
        l2, = ax2.plot(epochs, phases, 'r-', label='Phase', linewidth=2, alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate', color='blue')
        ax2.set_ylabel('Phase (0=Warmup, 1=RL)', color='red')
        ax.set_title('Training Schedule')
        ax.legend(handles=[l1, l2], loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([1, max(epochs)])
        
        # Row 2: Mutual Information & Entropy
        # Plot 5: Mutual Information
        ax = axes[1, 0]
        ax.plot(epochs, mi_fwd, 'b-', label='Forward', linewidth=2)
        ax.plot(epochs, mi_bwd, 'r-', label='Backward', linewidth=2)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mutual Information (nats)')
        ax.set_title('Mutual Information I(X;Y)')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([1, max(epochs)])
        
        # Plot 6: Entropy Metrics
        ax = axes[1, 1]
        ax.plot(epochs, entropy_x, 'b-', label='H(X)', linewidth=2)
        ax.plot(epochs, entropy_y, 'r-', label='H(Y)', linewidth=2)
        ax.plot(epochs, joint_entropy, 'g-', label='H(X,Y)', linewidth=2)
        ax.axhline(y=0.5*np.log(2*np.pi*np.e), color='gray', linestyle='--', alpha=0.5, label='Theoretical')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Entropy (nats)')
        ax.set_title('Entropy Metrics')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([1, max(epochs)])
        
        # Plot 7: Conditional Entropy
        ax = axes[1, 2]
        ax.plot(epochs, cond_entropy_x_y, 'b-', label='H(X|Y)', linewidth=2)
        ax.plot(epochs, cond_entropy_y_x, 'r-', label='H(Y|X)', linewidth=2)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Conditional Entropy (nats)')
        ax.set_title('Conditional Entropy')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([1, max(epochs)])
        
        # Plot 8: Information Decomposition
        ax = axes[1, 3]
        ax.stackplot(epochs, 
                     [max(0, m) for m in mi_fwd],  # MI (clamp to 0)
                     [max(0, c) for c in cond_entropy_x_y],  # H(X|Y)
                     labels=['I(X;Y)', 'H(X|Y)'],
                     alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Entropy (nats)')
        ax.set_title('Information Decomposition')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([1, max(epochs)])
        
        # Row 3: Coupling Metrics
        # Plot 9: Correlation
        ax = axes[2, 0]
        ax.plot(epochs, corr_x2_x1, 'b-', label='x2→x1', linewidth=2, marker='o', markersize=3)
        ax.plot(epochs, corr_x1_x2, 'r-', label='x1→x2', linewidth=2, marker='s', markersize=3)
        ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Perfect')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Correlation')
        ax.set_title('Coupling Correlation')
        ax.set_ylim([min(0, min(min(corr_x2_x1), min(corr_x1_x2)) - 0.1), 1.1])
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([1, max(epochs)])
        
        # Plot 10: MAE
        ax = axes[2, 1]
        ax.plot(epochs, mae_x2_x1, 'b-', label='x2→x1', linewidth=2, marker='o', markersize=3)
        ax.plot(epochs, mae_x1_x2, 'r-', label='x1→x2', linewidth=2, marker='s', markersize=3)
        ax.axhline(y=0, color='green', linestyle='--', alpha=0.7, label='Perfect')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title('Coupling Error (MAE)')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([1, max(epochs)])
        
        # Plot 11: Average Coupling Quality
        ax = axes[2, 2]
        avg_corr = [(c1 + c2) / 2 for c1, c2 in zip(corr_x2_x1, corr_x1_x2)]
        avg_mae = [(m1 + m2) / 2 for m1, m2 in zip(mae_x2_x1, mae_x1_x2)]
        ax.plot(epochs, avg_corr, 'g-', label='Avg Correlation', linewidth=2)
        ax2 = ax.twinx()
        ax2.plot(epochs, avg_mae, 'orange', label='Avg MAE', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Correlation', color='green')
        ax2.set_ylabel('MAE', color='orange')
        ax.set_title('Average Coupling Quality')
        ax.legend(loc='upper left', fontsize=8)
        ax2.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([1, max(epochs)])
        
        # Plot 12: Coupling Quality Score
        ax = axes[2, 3]
        # Quality score: high correlation, low MAE, low KL
        quality_scores = []
        for i in range(len(epochs)):
            corr_score = (corr_x2_x1[i] + corr_x1_x2[i]) / 2
            mae_penalty = min(1.0, (mae_x2_x1[i] + mae_x1_x2[i]) / 2)
            kl_penalty = min(1.0, avg_kl[i] / 2)
            score = max(0, corr_score - 0.3 * mae_penalty - 0.3 * kl_penalty)
            quality_scores.append(score)
        ax.plot(epochs, quality_scores, 'purple', linewidth=2)
        ax.fill_between(epochs, quality_scores, alpha=0.3, color='purple')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Quality Score')
        ax.set_title('Overall Coupling Quality')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3)
        ax.set_xlim([1, max(epochs)])
        
        # Row 4: Learned Statistics
        # Plot 13: Learned Mean (μ)
        ax = axes[3, 0]
        ax.plot(epochs, mu_1_fwd, 'b-', label='μ₁ (target=2.0)', linewidth=2)
        ax.axhline(y=2.0, color='blue', linestyle='--', alpha=0.5)
        ax.plot(epochs, mu_2_fwd, 'r-', label='μ₂ (target=10.0)', linewidth=2)
        ax.axhline(y=10.0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learned Mean (μ)')
        ax.set_title('Learned Distribution Mean')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([1, max(epochs)])
        
        # Plot 14: Learned Std (σ)
        ax = axes[3, 1]
        ax.plot(epochs, sigma_1_fwd, 'b-', label='σ₁ (target=1.0)', linewidth=2)
        ax.plot(epochs, sigma_2_fwd, 'r-', label='σ₂ (target=1.0)', linewidth=2)
        ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Target')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learned Std (σ)')
        ax.set_title('Learned Distribution Std')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([1, max(epochs)])
        
        # Plot 15: Mean Error
        ax = axes[3, 2]
        mu_1_error = [abs(m - 2.0) for m in mu_1_fwd]
        mu_2_error = [abs(m - 10.0) for m in mu_2_fwd]
        ax.plot(epochs, mu_1_error, 'b-', label='|μ₁ - 2.0|', linewidth=2)
        ax.plot(epochs, mu_2_error, 'r-', label='|μ₂ - 10.0|', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Absolute Error')
        ax.set_title('Mean Estimation Error')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([1, max(epochs)])
        
        # Plot 16: Std Error
        ax = axes[3, 3]
        sigma_1_error = [abs(s - 1.0) for s in sigma_1_fwd]
        sigma_2_error = [abs(s - 1.0) for s in sigma_2_fwd]
        ax.plot(epochs, sigma_1_error, 'b-', label='|σ₁ - 1.0|', linewidth=2)
        ax.plot(epochs, sigma_2_error, 'r-', label='|σ₂ - 1.0|', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Absolute Error')
        ax.set_title('Std Estimation Error')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([1, max(epochs)])
        
        plt.tight_layout()
        
        # Save to epoch folder
        epoch_plots_dir = os.path.join(save_dir, f"epoch_{epoch:04d}")
        os.makedirs(epoch_plots_dir, exist_ok=True)
        
        grid_path = os.path.join(epoch_plots_dir, "metrics_grid.png")
        plt.savefig(grid_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Also generate individual metric plots for detailed analysis
        self._generate_individual_metric_plots(epochs_data, epoch_plots_dir)
        
        return grid_path
    
    def _generate_individual_metric_plots(self, epochs_data: List[Dict], save_dir: str):
        """Generate individual plots for each metric category."""
        if len(epochs_data) < 2:
            return
        
        epochs = [e['epoch'] for e in epochs_data]
        
        # 1. KL Divergence Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        kl_1_fwd = [e['info_metrics_forward']['kl_div_1'] for e in epochs_data]
        kl_2_fwd = [e['info_metrics_forward']['kl_div_2'] for e in epochs_data]
        kl_total_fwd = [e['info_metrics_forward']['kl_div_total'] for e in epochs_data]
        kl_total_bwd = [e['info_metrics_backward']['kl_div_total'] for e in epochs_data]
        
        ax.plot(epochs, kl_1_fwd, 'b-', label='KL₁ Forward', linewidth=2)
        ax.plot(epochs, kl_2_fwd, 'r-', label='KL₂ Forward', linewidth=2)
        ax.plot(epochs, kl_total_fwd, 'g--', label='KL Total Forward', linewidth=2)
        ax.plot(epochs, kl_total_bwd, 'm--', label='KL Total Backward', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('KL Divergence', fontsize=12)
        ax.set_title('KL Divergence Over Training', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'kl_divergence.png'), dpi=150)
        plt.close(fig)
        
        # 2. Mutual Information Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        mi_fwd = [e['info_metrics_forward']['mutual_information'] for e in epochs_data]
        mi_bwd = [e['info_metrics_backward']['mutual_information'] for e in epochs_data]
        
        ax.plot(epochs, mi_fwd, 'b-', label='I(X;Y) Forward', linewidth=2, marker='o', markersize=4)
        ax.plot(epochs, mi_bwd, 'r-', label='I(X;Y) Backward', linewidth=2, marker='s', markersize=4)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Mutual Information (nats)', fontsize=12)
        ax.set_title('Mutual Information Over Training', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'mutual_information.png'), dpi=150)
        plt.close(fig)
        
        # 3. Entropy Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        entropy_x = [e['info_metrics_forward']['entropy_x'] for e in epochs_data]
        entropy_y = [e['info_metrics_forward']['entropy_y'] for e in epochs_data]
        joint_entropy = [e['info_metrics_forward']['joint_entropy'] for e in epochs_data]
        
        ax.plot(epochs, entropy_x, 'b-', label='H(X)', linewidth=2)
        ax.plot(epochs, entropy_y, 'r-', label='H(Y)', linewidth=2)
        ax.plot(epochs, joint_entropy, 'g-', label='H(X,Y)', linewidth=2)
        theoretical = 0.5 * np.log(2 * np.pi * np.e)
        ax.axhline(y=theoretical, color='gray', linestyle='--', alpha=0.7, label=f'Theoretical H(N(0,1))={theoretical:.2f}')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Entropy (nats)', fontsize=12)
        ax.set_title('Entropy Metrics Over Training', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'entropy.png'), dpi=150)
        plt.close(fig)
        
        # 4. Coupling Correlation Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        corr_x2_x1 = [e['coupling_metrics']['corr_x2_to_x1'] for e in epochs_data]
        corr_x1_x2 = [e['coupling_metrics']['corr_x1_to_x2'] for e in epochs_data]
        
        ax.plot(epochs, corr_x2_x1, 'b-', label='Corr(x2→x1)', linewidth=2, marker='o', markersize=4)
        ax.plot(epochs, corr_x1_x2, 'r-', label='Corr(x1→x2)', linewidth=2, marker='s', markersize=4)
        ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Perfect Correlation')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Correlation', fontsize=12)
        ax.set_title('Coupling Correlation Over Training', fontsize=14)
        ax.set_ylim([min(0, min(min(corr_x2_x1), min(corr_x1_x2)) - 0.1), 1.1])
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'correlation.png'), dpi=150)
        plt.close(fig)
        
        # 5. MAE Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        mae_x2_x1 = [e['coupling_metrics']['mae_x2_to_x1'] for e in epochs_data]
        mae_x1_x2 = [e['coupling_metrics']['mae_x1_to_x2'] for e in epochs_data]
        
        ax.plot(epochs, mae_x2_x1, 'b-', label='MAE(x2→x1)', linewidth=2, marker='o', markersize=4)
        ax.plot(epochs, mae_x1_x2, 'r-', label='MAE(x1→x2)', linewidth=2, marker='s', markersize=4)
        ax.axhline(y=0, color='green', linestyle='--', alpha=0.7, label='Perfect (MAE=0)')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Mean Absolute Error', fontsize=12)
        ax.set_title('Coupling Error (MAE) Over Training', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'mae.png'), dpi=150)
        plt.close(fig)
        
        # 6. Learned Statistics Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        mu_1 = [e['info_metrics_forward']['learned_mu_1'] for e in epochs_data]
        mu_2 = [e['info_metrics_forward']['learned_mu_2'] for e in epochs_data]
        sigma_1 = [e['info_metrics_forward']['learned_sigma_1'] for e in epochs_data]
        sigma_2 = [e['info_metrics_forward']['learned_sigma_2'] for e in epochs_data]
        
        axes[0].plot(epochs, mu_1, 'b-', label='μ₁ (target=2.0)', linewidth=2)
        axes[0].plot(epochs, mu_2, 'r-', label='μ₂ (target=10.0)', linewidth=2)
        axes[0].axhline(y=2.0, color='blue', linestyle='--', alpha=0.5)
        axes[0].axhline(y=10.0, color='red', linestyle='--', alpha=0.5)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Learned Mean', fontsize=12)
        axes[0].set_title('Learned Distribution Mean', fontsize=14)
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(epochs, sigma_1, 'b-', label='σ₁ (target=1.0)', linewidth=2)
        axes[1].plot(epochs, sigma_2, 'r-', label='σ₂ (target=1.0)', linewidth=2)
        axes[1].axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Target σ=1.0')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Learned Std', fontsize=12)
        axes[1].set_title('Learned Distribution Std', fontsize=14)
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'learned_statistics.png'), dpi=150)
        plt.close(fig)
    
    def _generate_latex_table(self, csv_path: str, latex_path: str):
        """Generate LaTeX table from CSV metrics for research paper."""
        import csv
        
        # Read CSV data
        rows = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        
        if not rows:
            print("No data to generate LaTeX table")
            return
        
        # Generate LaTeX table with key metrics
        latex = []
        latex.append("% Auto-generated LaTeX table from DDMEC training")
        latex.append("% Copy this into your research paper")
        latex.append("")
        latex.append("\\begin{table}[htbp]")
        latex.append("\\centering")
        latex.append("\\caption{DDMEC Training Progress: Information-Theoretic Metrics}")
        latex.append("\\label{tab:ddmec_training}")
        latex.append("\\begin{tabular}{@{}lcccccccc@{}}")
        latex.append("\\toprule")
        latex.append("Epoch & Phase & $D_{KL}^{fwd}$ & $D_{KL}^{bwd}$ & $I(X;Y)^{fwd}$ & $\\rho_{x2\\to x1}$ & $\\rho_{x1\\to x2}$ & MAE$_{x2\\to x1}$ & MAE$_{x1\\to x2}$ \\\\")
        latex.append("\\midrule")
        
        # Select key epochs (first, every 5, best, last)
        key_epochs = set([1])
        best_epoch = int(rows[-1]['best_epoch']) if rows else 1
        key_epochs.add(best_epoch)
        key_epochs.add(len(rows))
        for i in range(5, len(rows) + 1, 5):
            key_epochs.add(i)
        
        for row in rows:
            epoch_num = int(row['epoch'])
            if epoch_num in key_epochs:
                phase = row['phase'][:4]  # "warm" or "coop"
                kl_fwd = float(row['kl_div_total_fwd'])
                kl_bwd = float(row['kl_div_total_bwd'])
                mi_fwd = float(row['mutual_info_fwd'])
                corr_1 = float(row['corr_x2_to_x1'])
                corr_2 = float(row['corr_x1_to_x2'])
                mae_1 = float(row['mae_x2_to_x1'])
                mae_2 = float(row['mae_x1_to_x2'])
                
                # Highlight best epoch
                if epoch_num == best_epoch:
                    latex.append(f"\\textbf{{{epoch_num}}} & \\textbf{{{phase}}} & \\textbf{{{kl_fwd:.3f}}} & \\textbf{{{kl_bwd:.3f}}} & \\textbf{{{mi_fwd:.3f}}} & \\textbf{{{corr_1:.3f}}} & \\textbf{{{corr_2:.3f}}} & \\textbf{{{mae_1:.3f}}} & \\textbf{{{mae_2:.3f}}} \\\\")
                else:
                    latex.append(f"{epoch_num} & {phase} & {kl_fwd:.3f} & {kl_bwd:.3f} & {mi_fwd:.3f} & {corr_1:.3f} & {corr_2:.3f} & {mae_1:.3f} & {mae_2:.3f} \\\\")
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        latex.append("")
        
        # Add detailed table with all learned statistics
        latex.append("\\begin{table}[htbp]")
        latex.append("\\centering")
        latex.append("\\caption{DDMEC Learned Distribution Statistics}")
        latex.append("\\label{tab:ddmec_stats}")
        latex.append("\\begin{tabular}{@{}lccccccc@{}}")
        latex.append("\\toprule")
        latex.append("Epoch & $\\mu_1^{fwd}$ & $\\sigma_1^{fwd}$ & $\\mu_2^{fwd}$ & $\\sigma_2^{fwd}$ & $H(X)$ & $H(Y)$ & $H(X,Y)$ \\\\")
        latex.append("\\midrule")
        latex.append("Target & 2.00 & 1.00 & 10.00 & 1.00 & 1.42 & 1.42 & 1.42 \\\\")
        latex.append("\\midrule")
        
        for row in rows:
            epoch_num = int(row['epoch'])
            if epoch_num in key_epochs:
                mu1 = float(row['mu_1_fwd'])
                sig1 = float(row['sigma_1_fwd'])
                mu2 = float(row['mu_2_fwd'])
                sig2 = float(row['sigma_2_fwd'])
                hx = float(row['entropy_x_fwd'])
                hy = float(row['entropy_y_fwd'])
                hxy = float(row['joint_entropy_fwd'])
                
                if epoch_num == best_epoch:
                    latex.append(f"\\textbf{{{epoch_num}}} & \\textbf{{{mu1:.2f}}} & \\textbf{{{sig1:.2f}}} & \\textbf{{{mu2:.2f}}} & \\textbf{{{sig2:.2f}}} & \\textbf{{{hx:.2f}}} & \\textbf{{{hy:.2f}}} & \\textbf{{{hxy:.2f}}} \\\\")
                else:
                    latex.append(f"{epoch_num} & {mu1:.2f} & {sig1:.2f} & {mu2:.2f} & {sig2:.2f} & {hx:.2f} & {hy:.2f} & {hxy:.2f} \\\\")
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        
        # Write to file
        with open(latex_path, 'w') as f:
            f.write('\n'.join(latex))
        
        print(f"  LaTeX table generated: {latex_path}")
    
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
        import csv
        import json
        import datetime
        os.makedirs(run_dir, exist_ok=True)
        checkpoints_dir = os.path.join(run_dir, "checkpoints")
        plots_dir = os.path.join(run_dir, "plots")
        tables_dir = os.path.join(run_dir, "tables")
        tensorboard_dir = os.path.join(run_dir, "tensorboard")
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(tables_dir, exist_ok=True)
        
        # Initialize TensorBoard writer
        if self.use_tensorboard:
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=tensorboard_dir)
            print(f"TensorBoard logging to: {tensorboard_dir}")
            print(f"  Run: tensorboard --logdir={tensorboard_dir}")
        
        num_samples = min(len(dataset_1), len(dataset_2))
        num_batches = num_samples // batch_size
        
        # Calculate warmup steps based on warmup_epochs
        self.warmup_steps = self.warmup_epochs * num_batches
        print(f"Warmup: {self.warmup_epochs} epochs = {self.warmup_steps} steps")
        
        # Setup optimizers with smaller initial learning rate for stability
        # The conditional layers need gentler training initially
        self.target_lr = lr  # Save target LR for warmup
        warmup_lr = lr * 0.1  # Start with 10% of target learning rate
        optimizer_1 = torch.optim.Adam(self.ddpm_1.model.parameters(), lr=warmup_lr)
        optimizer_2 = torch.optim.Adam(self.ddpm_2.model.parameters(), lr=warmup_lr)
        
        # Track best model
        best_kl_total = float('inf')
        best_epoch = 0
        
        # Divergence tracking
        divergence_threshold = 5.0  # Max acceptable coupling error
        consecutive_divergences = 0
        max_consecutive_divergences = 3  # After this many, reduce LR
        
        # Initialize local JSON log for backup (in case W&B/TensorBoard fail)
        logs_dir = os.path.join(run_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        local_log_path = os.path.join(logs_dir, "training_log.json")
        
        # Initialize log structure
        self.local_log = {
            "experiment_name": os.path.basename(run_dir),
            "run_dir": run_dir,
            "config": {
                "kl_weight": self.kl_weight,
                "ppo_clip": self.ppo_clip,
                "warmup_epochs": self.warmup_epochs,
                "rl_lr_scale": self.rl_lr_scale,
                "learning_rate": lr,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
            },
            "epochs": [],
            "training_metrics": [],
        }
        
        # Initialize CSV file for epoch metrics (research paper table)
        metrics_csv_path = os.path.join(tables_dir, "epoch_metrics.csv")
        csv_headers = [
            "epoch", "phase",
            # Forward direction (x2→x1)
            "kl_div_1_fwd", "kl_div_2_fwd", "kl_div_total_fwd",
            "entropy_x_fwd", "entropy_y_fwd", "joint_entropy_fwd",
            "mutual_info_fwd", "cond_entropy_x_given_y", "cond_entropy_y_given_x",
            "mu_1_fwd", "sigma_1_fwd", "mu_2_fwd", "sigma_2_fwd",
            # Backward direction (x1→x2)
            "kl_div_1_bwd", "kl_div_2_bwd", "kl_div_total_bwd",
            "mutual_info_bwd",
            "mu_1_bwd", "sigma_1_bwd", "mu_2_bwd", "sigma_2_bwd",
            # Coupling metrics
            "corr_x2_to_x1", "corr_x1_to_x2",
            "mae_x2_to_x1", "mae_x1_to_x2",
            # Best tracking
            "best_kl_total", "best_epoch",
            # Training info
            "current_lr"
        ]
        
        with open(metrics_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)
        
        print(f"Starting with warmup LR: {warmup_lr:.2e}")
        print(f"After warmup: supervised LR={lr:.2e}, RL LR={lr * self.rl_lr_scale:.2e}")
        
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
            
            # Compute mean metrics
            epoch_train_metrics = {}
            for k, v in epoch_metrics.items():
                if v:
                    mean_val = np.mean(v)
                    print(f"    {k}: {mean_val:.4f}")
                    epoch_train_metrics[k] = float(mean_val)
                    if self.use_wandb:
                        wandb_log[f"train/{k}"] = mean_val
            
            # Add to local log
            self.local_log["training_metrics"].append({
                "epoch": epoch + 1,
                "metrics": epoch_train_metrics,
            })
            
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
                
                # Compute coupling metrics
                x1_gen_np = x1_generated.cpu().numpy().flatten()
                x2_gen_np = x2_generated.cpu().numpy().flatten()
                x1_eval_np = x1_eval.cpu().numpy().flatten()
                x2_eval_np = x2_eval.cpu().numpy().flatten()
                
                corr_x2_to_x1 = np.corrcoef(x2_eval_np, x1_gen_np)[0, 1]
                corr_x1_to_x2 = np.corrcoef(x1_eval_np, x2_gen_np)[0, 1]
                mae_x2_to_x1 = np.abs(x1_gen_np - (x2_eval_np - 8.0)).mean()
                mae_x1_to_x2 = np.abs(x2_gen_np - (x1_eval_np + 8.0)).mean()
                
                # Compute average KL total (needed for logging and best model tracking)
                avg_kl_total = (info_metrics_1['kl_div_total'] + info_metrics_2['kl_div_total']) / 2
                
                print(f"\n  Coupling Metrics:")
                print(f"    Correlation x2→x1: {corr_x2_to_x1:.4f}")
                print(f"    Correlation x1→x2: {corr_x1_to_x2:.4f}")
                print(f"    MAE x2→x1: {mae_x2_to_x1:.4f}")
                print(f"    MAE x1→x2: {mae_x1_to_x2:.4f}")
                print(f"    Avg KL Total: {avg_kl_total:.4f}")
                
                # Check for divergence during RL phase
                if self.gen_step >= self.warmup_steps:
                    max_mae = max(mae_x2_to_x1, mae_x1_to_x2)
                    if max_mae > divergence_threshold or not np.isfinite(max_mae):
                        consecutive_divergences += 1
                        print(f"\n  [WARNING] DIVERGENCE DETECTED (MAE={max_mae:.2f} > {divergence_threshold})")
                        print(f"      Consecutive divergences: {consecutive_divergences}/{max_consecutive_divergences}")
                        
                        if consecutive_divergences >= max_consecutive_divergences:
                            # Reduce learning rate
                            old_lr = optimizer_1.param_groups[0]['lr']
                            new_lr = old_lr * 0.5
                            for param_group in optimizer_1.param_groups:
                                param_group['lr'] = new_lr
                            for param_group in optimizer_2.param_groups:
                                param_group['lr'] = new_lr
                            
                            print(f"\n  [LR DECAY] REDUCING LEARNING RATE: {old_lr:.2e} -> {new_lr:.2e}")
                            consecutive_divergences = 0  # Reset counter
                    else:
                        consecutive_divergences = 0  # Reset on good epoch
                
                # Determine current phase
                current_phase = "warmup" if self.gen_step < self.warmup_steps else "cooperative"
                current_lr = optimizer_1.param_groups[0]['lr']
                
                # Create epoch folder for all plots
                epoch_plots_dir = os.path.join(plots_dir, f"epoch_{epoch+1:04d}")
                os.makedirs(epoch_plots_dir, exist_ok=True)
                
                # Generate main coupling plot for this epoch
                plot_path = os.path.join(epoch_plots_dir, "coupling_plot.png")
                self.generate_epoch_plot(
                    x1_gen_np,
                    x2_gen_np,
                    x1_eval_np,
                    x2_eval_np,
                    epoch + 1,
                    plot_path
                )
                print(f"  Coupling plot saved: {plot_path}")
                
                # Also save a copy at the top level for easy access
                top_level_plot = os.path.join(plots_dir, f"epoch_{epoch+1:04d}.png")
                import shutil
                shutil.copy(plot_path, top_level_plot)
                
                # Save epoch metrics to CSV (for research paper tables)
                csv_row = [
                    epoch + 1, current_phase,
                    # Forward direction
                    info_metrics_1['kl_div_1'], info_metrics_1['kl_div_2'], info_metrics_1['kl_div_total'],
                    info_metrics_1['entropy_x'], info_metrics_1['entropy_y'], info_metrics_1['joint_entropy'],
                    info_metrics_1['mutual_information'], 
                    info_metrics_1['conditional_entropy_x_given_y'], info_metrics_1['conditional_entropy_y_given_x'],
                    info_metrics_1['learned_mu_1'], info_metrics_1['learned_sigma_1'],
                    info_metrics_1['learned_mu_2'], info_metrics_1['learned_sigma_2'],
                    # Backward direction
                    info_metrics_2['kl_div_1'], info_metrics_2['kl_div_2'], info_metrics_2['kl_div_total'],
                    info_metrics_2['mutual_information'],
                    info_metrics_2['learned_mu_1'], info_metrics_2['learned_sigma_1'],
                    info_metrics_2['learned_mu_2'], info_metrics_2['learned_sigma_2'],
                    # Coupling metrics
                    corr_x2_to_x1, corr_x1_to_x2,
                    mae_x2_to_x1, mae_x1_to_x2,
                    # Best tracking
                    best_kl_total, best_epoch,
                    # Training info
                    current_lr
                ]
                
                with open(metrics_csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(csv_row)
                print(f"  Table saved: {metrics_csv_path}")
                
                # Save to local JSON log (backup for W&B/TensorBoard failures)
                epoch_log = {
                    "epoch": epoch + 1,
                    "phase": current_phase,
                    "learning_rate": current_lr,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "info_metrics_forward": {
                        "kl_div_1": info_metrics_1['kl_div_1'],
                        "kl_div_2": info_metrics_1['kl_div_2'],
                        "kl_div_total": info_metrics_1['kl_div_total'],
                        "entropy_x": info_metrics_1['entropy_x'],
                        "entropy_y": info_metrics_1['entropy_y'],
                        "joint_entropy": info_metrics_1['joint_entropy'],
                        "mutual_information": info_metrics_1['mutual_information'],
                        "conditional_entropy_x_given_y": info_metrics_1['conditional_entropy_x_given_y'],
                        "conditional_entropy_y_given_x": info_metrics_1['conditional_entropy_y_given_x'],
                        "learned_mu_1": info_metrics_1['learned_mu_1'],
                        "learned_sigma_1": info_metrics_1['learned_sigma_1'],
                        "learned_mu_2": info_metrics_1['learned_mu_2'],
                        "learned_sigma_2": info_metrics_1['learned_sigma_2'],
                    },
                    "info_metrics_backward": {
                        "kl_div_1": info_metrics_2['kl_div_1'],
                        "kl_div_2": info_metrics_2['kl_div_2'],
                        "kl_div_total": info_metrics_2['kl_div_total'],
                        "mutual_information": info_metrics_2['mutual_information'],
                        "learned_mu_1": info_metrics_2['learned_mu_1'],
                        "learned_sigma_1": info_metrics_2['learned_sigma_1'],
                        "learned_mu_2": info_metrics_2['learned_mu_2'],
                        "learned_sigma_2": info_metrics_2['learned_sigma_2'],
                    },
                    "coupling_metrics": {
                        "corr_x2_to_x1": float(corr_x2_to_x1),
                        "corr_x1_to_x2": float(corr_x1_to_x2),
                        "mae_x2_to_x1": float(mae_x2_to_x1),
                        "mae_x1_to_x2": float(mae_x1_to_x2),
                    },
                    "best": {
                        "kl_total": best_kl_total,
                        "epoch": best_epoch,
                    },
                    "avg_kl_total": avg_kl_total,
                }
                self.local_log["epochs"].append(epoch_log)
                
                # Write local log to disk (atomic write)
                temp_log_path = local_log_path + ".tmp"
                with open(temp_log_path, 'w') as f:
                    json.dump(self.local_log, f, indent=2)
                os.replace(temp_log_path, local_log_path)
                print(f"  Local log saved: {local_log_path}")
                
                # Generate 4x4 metrics grid plot for this epoch
                try:
                    grid_path = self._generate_metrics_grid_plot(epoch + 1, plots_dir)
                    if grid_path:
                        print(f"  Metrics grid saved: {grid_path}")
                        
                        # Log grid to TensorBoard
                        if self.use_tensorboard and self.tb_writer is not None:
                            try:
                                import PIL.Image
                                img = PIL.Image.open(grid_path)
                                img_array = np.array(img)
                                if len(img_array.shape) == 3:
                                    img_array = img_array.transpose(2, 0, 1)
                                self.tb_writer.add_image('plots/metrics_grid', img_array, epoch + 1)
                            except Exception:
                                pass
                        
                        # Log grid to W&B
                        if self.use_wandb:
                            try:
                                wandb.log({"plots/metrics_grid": wandb.Image(grid_path)}, step=epoch + 1)
                            except Exception:
                                pass
                except Exception as e:
                    print(f"  Warning: Failed to generate metrics grid: {e}")
                
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
                    print(f"  [BEST] New best model! (KL Total: {best_kl_total:.4f})")
                
                # Log to TensorBoard
                if self.use_tensorboard and self.tb_writer is not None:
                    # Information metrics - Forward
                    self.tb_writer.add_scalar('info/kl_div_1_forward', info_metrics_1['kl_div_1'], epoch + 1)
                    self.tb_writer.add_scalar('info/kl_div_2_forward', info_metrics_1['kl_div_2'], epoch + 1)
                    self.tb_writer.add_scalar('info/kl_div_total_forward', info_metrics_1['kl_div_total'], epoch + 1)
                    self.tb_writer.add_scalar('info/entropy_x_forward', info_metrics_1['entropy_x'], epoch + 1)
                    self.tb_writer.add_scalar('info/entropy_y_forward', info_metrics_1['entropy_y'], epoch + 1)
                    self.tb_writer.add_scalar('info/joint_entropy_forward', info_metrics_1['joint_entropy'], epoch + 1)
                    self.tb_writer.add_scalar('info/mutual_information_forward', info_metrics_1['mutual_information'], epoch + 1)
                    self.tb_writer.add_scalar('info/conditional_entropy_x_given_y', info_metrics_1['conditional_entropy_x_given_y'], epoch + 1)
                    self.tb_writer.add_scalar('info/conditional_entropy_y_given_x', info_metrics_1['conditional_entropy_y_given_x'], epoch + 1)
                    
                    # Information metrics - Backward
                    self.tb_writer.add_scalar('info/kl_div_1_backward', info_metrics_2['kl_div_1'], epoch + 1)
                    self.tb_writer.add_scalar('info/kl_div_2_backward', info_metrics_2['kl_div_2'], epoch + 1)
                    self.tb_writer.add_scalar('info/kl_div_total_backward', info_metrics_2['kl_div_total'], epoch + 1)
                    self.tb_writer.add_scalar('info/mutual_information_backward', info_metrics_2['mutual_information'], epoch + 1)
                    self.tb_writer.add_scalar('info/avg_kl_total', avg_kl_total, epoch + 1)
                    
                    # Learned statistics
                    self.tb_writer.add_scalar('stats/learned_mu_1_forward', info_metrics_1['learned_mu_1'], epoch + 1)
                    self.tb_writer.add_scalar('stats/learned_sigma_1_forward', info_metrics_1['learned_sigma_1'], epoch + 1)
                    self.tb_writer.add_scalar('stats/learned_mu_2_forward', info_metrics_1['learned_mu_2'], epoch + 1)
                    self.tb_writer.add_scalar('stats/learned_sigma_2_forward', info_metrics_1['learned_sigma_2'], epoch + 1)
                    
                    # Coupling metrics
                    self.tb_writer.add_scalar('coupling/corr_x2_to_x1', corr_x2_to_x1, epoch + 1)
                    self.tb_writer.add_scalar('coupling/corr_x1_to_x2', corr_x1_to_x2, epoch + 1)
                    self.tb_writer.add_scalar('coupling/mae_x2_to_x1', mae_x2_to_x1, epoch + 1)
                    self.tb_writer.add_scalar('coupling/mae_x1_to_x2', mae_x1_to_x2, epoch + 1)
                    
                    # Best tracking
                    self.tb_writer.add_scalar('best/kl_total', best_kl_total, epoch + 1)
                    self.tb_writer.add_scalar('best/epoch', best_epoch, epoch + 1)
                    
                    # Training info
                    self.tb_writer.add_scalar('train/phase', 0 if current_phase == "warmup" else 1, epoch + 1)
                    self.tb_writer.add_scalar('train/learning_rate', current_lr, epoch + 1)
                    
                    # Add plot as image to TensorBoard
                    try:
                        import PIL.Image
                        img = PIL.Image.open(plot_path)
                        img_array = np.array(img)
                        # Convert to CHW format for TensorBoard
                        if len(img_array.shape) == 3:
                            img_array = img_array.transpose(2, 0, 1)
                        self.tb_writer.add_image('plots/epoch_visualization', img_array, epoch + 1)
                    except Exception as e:
                        pass  # Silently skip if image logging fails
                    
                    self.tb_writer.flush()
                
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
                        # Coupling metrics
                        "coupling/corr_x2_to_x1": corr_x2_to_x1,
                        "coupling/corr_x1_to_x2": corr_x1_to_x2,
                        "coupling/mae_x2_to_x1": mae_x2_to_x1,
                        "coupling/mae_x1_to_x2": mae_x1_to_x2,
                        # Best tracking
                        "best/kl_total": best_kl_total,
                        "best/epoch": best_epoch,
                        # Average KL
                        "info/avg_kl_total": avg_kl_total,
                        # Training info
                        "train/phase": 0 if current_phase == "warmup" else 1,
                        "train/learning_rate": current_lr,
                    })
                    # Log plot as image
                    wandb_log["plots/epoch_visualization"] = wandb.Image(plot_path)
                    wandb.log(wandb_log)
        
        # Generate LaTeX table for research paper
        latex_path = os.path.join(tables_dir, "results_table.tex")
        self._generate_latex_table(metrics_csv_path, latex_path)
        
        # Close TensorBoard writer
        if self.use_tensorboard and self.tb_writer is not None:
            self.tb_writer.close()
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        print(f"Best epoch: {best_epoch}")
        print(f"Best KL Total: {best_kl_total:.4f}")
        print(f"\nOutput directories:")
        print(f"  Checkpoints: {checkpoints_dir}")
        print(f"  Plots: {plots_dir}")
        print(f"  Tables: {tables_dir}")
        print(f"    - CSV: {metrics_csv_path}")
        print(f"    - LaTeX: {latex_path}")
        print(f"  Local logs: {logs_dir}")
        print(f"    - JSON: {local_log_path}")
        if self.use_tensorboard:
            print(f"  TensorBoard: {tensorboard_dir}")
            print(f"    Run: tensorboard --logdir={tensorboard_dir}")
        print(f"\nBest model: {os.path.join(checkpoints_dir, 'best_model.pt')}")
        print(f"\nTo sync local logs to W&B/TensorBoard later:")
        print(f"  python sync_logs.py --run-dir {run_dir}")
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

