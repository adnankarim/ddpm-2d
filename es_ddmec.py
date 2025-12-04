
"""
ES-DDMEC: Evolution Strategies for Denoising Diffusion Minimum Entropy Coupling

This module implements Evolution Strategies (ES) as an alternative to PPO
for training DDMEC models. Based on the paper:
"Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning"
https://arxiv.org/abs/2509.24372
https://github.com/VsonicV/es-fine-tuning-paper

Key advantages of ES over PPO:
1. More stable training across runs
2. Better handling of long-horizon rewards (diffusion trajectories)
3. Less prone to reward hacking
4. No backpropagation needed during ES phase (memory efficient)
5. Naturally conservative updates (no KL penalty needed)

Author: ES-DDMEC Integration
"""

import os
import copy
import json
import csv
import datetime
import shutil
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, List, Callable
from collections import defaultdict
from dataclasses import dataclass

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


@dataclass
class ESConfig:
    """Configuration for Evolution Strategies optimizer."""
    population_size: int = 30        # Number of perturbed models per iteration
    sigma: float = 0.001             # Noise scale for perturbations
    learning_rate: float = 5e-4      # ES learning rate (digests 1/sigma term)
    use_rank_transform: bool = False # Whether to use rank-based fitness shaping
    use_mirror_sampling: bool = False # Whether to use antithetic sampling


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
        h = self.base_model.net[2](h)  # Second Linear layer
        h = self.base_model.net[3](h)  # SiLU
        h = self.base_model.net[4](h)  # Output layer
        
        return h


class ESOptimizer:
    """
    Evolution Strategies optimizer for neural networks.
    
    Based on: "Evolution Strategies at Scale: LLM Fine-Tuning Beyond RL"
    https://github.com/VsonicV/es-fine-tuning-paper
    
    Key implementation details from the paper:
    1. Noise retrieval with random seeds (memory efficient)
    2. Layer-level in-place perturbation and restoration
    3. Z-score reward normalization
    4. Greedy decoding for deterministic evaluation
    5. Decomposed parameter updates
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: ESConfig = None,
        device: torch.device = None,
    ):
        """
        Args:
            model: Neural network to optimize
            config: ES configuration
            device: Computation device
        """
        self.model = model
        self.config = config or ESConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Count parameters
        self.num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.param_shapes = [(name, p.shape) for name, p in model.named_parameters() if p.requires_grad]
        
        # Statistics tracking
        self.step_count = 0
        self.reward_history = []
        
        print(f"ES Optimizer initialized:")
        print(f"  Population size: {self.config.population_size}")
        print(f"  Sigma: {self.config.sigma}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Trainable parameters: {self.num_params:,}")
        print(f"  Use rank transform: {self.config.use_rank_transform}")
        print(f"  Use mirror sampling: {self.config.use_mirror_sampling}")
    
    def _get_flat_params(self) -> torch.Tensor:
        """Get flattened model parameters."""
        return torch.cat([p.data.view(-1) for p in self.model.parameters() if p.requires_grad])
    
    def _set_flat_params(self, flat_params: torch.Tensor) -> None:
        """Set model parameters from flattened tensor."""
        offset = 0
        for p in self.model.parameters():
            if p.requires_grad:
                numel = p.numel()
                p.data.copy_(flat_params[offset:offset + numel].view(p.shape))
                offset += numel
    
    def perturb_model_layerwise(self, seed: int) -> List[int]:
        """
        Perturb model parameters layer by layer using seed.
        Memory efficient: only stores seed per layer.
        
        Args:
            seed: Base random seed
            
        Returns:
            List of layer seeds for restoration
        """
        layer_seeds = []
        
        for idx, (name, param) in enumerate(self.model.named_parameters()):
            if param.requires_grad:
                # Generate unique seed for each layer
                layer_seed = seed + idx * 1000
                layer_seeds.append(layer_seed)
                
                # Set seed and generate noise
                torch.manual_seed(layer_seed)
                noise = torch.randn_like(param, device=self.device) * self.config.sigma
                
                # In-place perturbation
                param.data.add_(noise)
        
        return layer_seeds
    
    def restore_model_layerwise(self, layer_seeds: List[int]) -> None:
        """
        Restore model by subtracting the same perturbation.
        Uses same seeds to regenerate identical noise.
        
        Args:
            layer_seeds: Seeds used for perturbation
        """
        for idx, (name, param) in enumerate(self.model.named_parameters()):
            if param.requires_grad:
                # Regenerate same noise
                torch.manual_seed(layer_seeds[idx])
                noise = torch.randn_like(param, device=self.device) * self.config.sigma
                
                # In-place restoration
                param.data.sub_(noise)
    
    def update_model_layerwise(
        self,
        seeds_list: List[List[int]],
        normalized_rewards: np.ndarray,
    ) -> None:
        """
        Update model parameters using aggregated weighted noise.
        Decomposed update: layer by layer, seed by seed.
        
        Args:
            seeds_list: List of layer seeds for each population member
            normalized_rewards: Z-score normalized rewards
        """
        N = len(seeds_list)
        
        for idx, (name, param) in enumerate(self.model.named_parameters()):
            if param.requires_grad:
                # Aggregate updates for this layer
                update = torch.zeros_like(param, device=self.device)
                
                for n, layer_seeds in enumerate(seeds_list):
                    # Regenerate noise for this layer
                    torch.manual_seed(layer_seeds[idx])
                    noise = torch.randn_like(param, device=self.device)
                    
                    # Weighted accumulation
                    update.add_(noise * normalized_rewards[n])
                
                # Apply update: θ_t ← θ_{t-1} + α * (1/N) * Σ R_n * ε_n
                param.data.add_(self.config.learning_rate / N * update)
    
    def rank_transform(self, rewards: np.ndarray) -> np.ndarray:
        """
        Apply rank-based fitness shaping.
        Maps rewards to range [-0.5, 0.5] based on rank.
        
        Args:
            rewards: Raw reward values
            
        Returns:
            Rank-transformed fitness values
        """
        N = len(rewards)
        ranks = np.argsort(np.argsort(rewards))  # Get ranks
        centered_ranks = ranks - (N - 1) / 2.0
        normalized = centered_ranks / N
        return normalized
    
    def step(self, reward_fn: Callable[[], float]) -> Dict[str, float]:
        """
        Perform one ES update step.
        
        Algorithm from the paper:
        1. Sample N perturbations using random seeds
        2. Evaluate perturbed models (greedy decoding)
        3. Normalize rewards using z-score
        4. Update parameters as weighted sum of perturbations
        
        Args:
            reward_fn: Function that returns scalar reward for current model
            
        Returns:
            Metrics dictionary
        """
        self.step_count += 1
        
        N = self.config.population_size
        rewards = []
        seeds_list = []
        
        # Handle mirror sampling (antithetic)
        if self.config.use_mirror_sampling:
            assert N % 2 == 0, "Population size must be even for mirror sampling"
            actual_samples = N // 2
        else:
            actual_samples = N
        
        # Generate population and evaluate
        for n in range(actual_samples):
            # Generate base seed
            base_seed = np.random.randint(0, 2**31)
            
            # Positive perturbation
            layer_seeds = self.perturb_model_layerwise(base_seed)
            seeds_list.append(layer_seeds)
            
            with torch.no_grad():
                reward_pos = reward_fn()
            rewards.append(reward_pos)
            
            # Restore model
            self.restore_model_layerwise(layer_seeds)
            
            # Mirror sampling: negative perturbation
            if self.config.use_mirror_sampling:
                # Apply negative perturbation
                for idx, (name, param) in enumerate(self.model.named_parameters()):
                    if param.requires_grad:
                        torch.manual_seed(layer_seeds[idx])
                        noise = torch.randn_like(param, device=self.device) * self.config.sigma
                        param.data.sub_(noise)  # Subtract instead of add
                
                with torch.no_grad():
                    reward_neg = reward_fn()
                rewards.append(reward_neg)
                
                # Store negative seeds (same seeds, but will be subtracted)
                seeds_list.append(layer_seeds)  # Will handle sign in update
                
                # Restore from negative perturbation
                for idx, (name, param) in enumerate(self.model.named_parameters()):
                    if param.requires_grad:
                        torch.manual_seed(layer_seeds[idx])
                        noise = torch.randn_like(param, device=self.device) * self.config.sigma
                        param.data.add_(noise)  # Add back
        
        rewards = np.array(rewards)
        
        # Check for invalid rewards
        if not np.isfinite(rewards).all():
            nan_count = np.isnan(rewards).sum()
            inf_count = np.isinf(rewards).sum()
            print(f"WARNING: Invalid rewards detected - NaN: {nan_count}, Inf: {inf_count}")
            # Replace invalid values with mean of valid values
            valid_mask = np.isfinite(rewards)
            if valid_mask.any():
                rewards[~valid_mask] = rewards[valid_mask].mean()
            else:
                print("ERROR: All rewards are invalid!")
                return {"error": 1.0}
        
        # Normalize rewards
        if self.config.use_rank_transform:
            normalized_rewards = self.rank_transform(rewards)
        else:
            # Z-score normalization
            reward_mean = rewards.mean()
            reward_std = rewards.std()
            if reward_std < 1e-8:
                # All rewards are the same, no gradient signal
                normalized_rewards = np.zeros_like(rewards)
            else:
                normalized_rewards = (rewards - reward_mean) / reward_std
        
        # Handle mirror sampling in update
        if self.config.use_mirror_sampling:
            # For mirror sampling, odd indices are negative perturbations
            for i in range(1, len(normalized_rewards), 2):
                normalized_rewards[i] = -normalized_rewards[i]
        
        # Update model parameters
        self.update_model_layerwise(seeds_list, normalized_rewards)
        
        # Track statistics
        self.reward_history.append(rewards.mean())
        
        return {
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std()),
            "reward_max": float(rewards.max()),
            "reward_min": float(rewards.min()),
            "normalized_reward_mean": float(normalized_rewards.mean()),
            "normalized_reward_std": float(normalized_rewards.std()),
            "es_step": self.step_count,
        }


class ESDDMEC1D:
    """
    ES-DDMEC: Evolution Strategies for Minimum Entropy Coupling.
    
    This class implements DDMEC using Evolution Strategies instead of PPO
    for the cooperative training phase. The key differences are:
    
    1. No policy gradient computation needed
    2. Parameter-space exploration instead of action-space
    3. More stable training with consistent results across runs
    4. No KL regularization penalty needed (ES naturally conservative)
    5. Better handling of long-horizon rewards (diffusion trajectories)
    
    Training phases:
    1. Warmup: Supervised denoising training (same as PPO-DDMEC)
    2. Cooperative: ES-based parameter optimization
    """
    
    def __init__(
        self,
        model_path_1: str,
        model_path_2: str,
        config: Optional[DDPMConfig] = None,
        use_wandb: bool = False,
        use_tensorboard: bool = True,
        # ES hyperparameters
        population_size: int = 30,
        sigma: float = 0.001,
        es_lr: float = 5e-4,
        warmup_epochs: int = 10,
        use_rank_transform: bool = False,
        use_mirror_sampling: bool = False,
        # Reward computation
        mc_steps: int = 5,
        num_sampling_steps: int = 50,
    ):
        """
        Args:
            model_path_1: Path to first pre-trained model (e.g., N(2,1))
            model_path_2: Path to second pre-trained model (e.g., N(10,1))
            config: DDPM configuration
            use_wandb: Whether to log to Weights & Biases
            use_tensorboard: Whether to log to TensorBoard
            population_size: ES population size (paper used 30)
            sigma: Noise scale for parameter perturbations
            es_lr: ES learning rate
            warmup_epochs: Supervised warmup epochs before ES phase
            use_rank_transform: Whether to use rank-based fitness shaping
            use_mirror_sampling: Whether to use antithetic sampling
            mc_steps: Monte Carlo samples for reward estimation
            num_sampling_steps: Number of diffusion sampling steps
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        self.tb_writer = None
        
        # ES configuration
        self.es_config = ESConfig(
            population_size=population_size,
            sigma=sigma,
            learning_rate=es_lr,
            use_rank_transform=use_rank_transform,
            use_mirror_sampling=use_mirror_sampling,
        )
        
        self.warmup_epochs = warmup_epochs
        self.mc_steps = mc_steps
        self.num_sampling_steps = num_sampling_steps
        
        print(f"\n{'='*60}")
        print("ES-DDMEC: Evolution Strategies for Minimum Entropy Coupling")
        print("="*60)
        print(f"\nES Hyperparameters:")
        print(f"  Population Size: {population_size}")
        print(f"  Sigma: {sigma}")
        print(f"  ES Learning Rate: {es_lr}")
        print(f"  Warmup Epochs: {warmup_epochs}")
        print(f"  Rank Transform: {use_rank_transform}")
        print(f"  Mirror Sampling: {use_mirror_sampling}")
        print(f"  MC Steps: {mc_steps}")
        print(f"  Sampling Steps: {num_sampling_steps}")
        
        # Load models
        print(f"\nLoading Model 1 from {model_path_1}")
        self.ddpm_1, self.config_1 = self._load_model(model_path_1)
        
        print(f"Loading Model 2 from {model_path_2}")
        self.ddpm_2, self.config_2 = self._load_model(model_path_2)
        
        self.config = config or self.config_1
        
        # ES optimizers (created after warmup)
        self.es_optimizer_1 = None
        self.es_optimizer_2 = None
        
        # Training state
        self.gen_step = 0
        self.warmup_steps = 1000  # Will be updated based on warmup_epochs
        
        # Tracking
        self.losses = defaultdict(list)
        self.local_log = None
        
        print(f"\nDevice: {self.device}")
        print("="*60 + "\n")
    
    def _load_model(self, path: str) -> Tuple[DDPM1D, DDPMConfig]:
        """Load a model and its config from checkpoint."""
        cfg = DDPMConfig()
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
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
    
    def setup_es_optimizers(self):
        """Initialize ES optimizers after warmup phase."""
        print("\n" + "="*60)
        print("Setting up ES optimizers for cooperative training")
        print("="*60)
        
        self.es_optimizer_1 = ESOptimizer(
            model=self.ddpm_1.model,
            config=self.es_config,
            device=self.device,
        )
        
        print()  # Separator
        
        self.es_optimizer_2 = ESOptimizer(
            model=self.ddpm_2.model,
            config=self.es_config,
            device=self.device,
        )
        
        print("="*60 + "\n")
    
    def train_supervised(
        self,
        x1_batch: torch.Tensor,
        x2_batch: torch.Tensor,
        optimizer_1: torch.optim.Optimizer,
        optimizer_2: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """
        Supervised warmup training: standard denoising objective.
        Same as PPO-DDMEC warmup phase.
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
    def sample_trajectory(
        self,
        model: nn.Module,
        ddpm: DDPM1D,
        condition: torch.Tensor,
        num_steps: int = None,
    ) -> torch.Tensor:
        """
        Sample from model using DDIM (greedy, no action-space noise).
        
        For ES, we use greedy decoding so all performance differences
        come from parameter-space exploration.
        
        Args:
            model: The denoising model
            ddpm: DDPM wrapper with schedules
            condition: Conditioning variable
            num_steps: Number of denoising steps
            
        Returns:
            Generated samples
        """
        if num_steps is None:
            num_steps = self.num_sampling_steps
            
        model.eval()
        batch_size = condition.shape[0]
        
        # Start from noise
        x = torch.randn(batch_size, 1, device=self.device)
        
        # Create evenly spaced timesteps
        timesteps = torch.linspace(self.config.timesteps - 1, 0, num_steps, dtype=torch.long)
        
        for i in range(len(timesteps) - 1):
            t_cur = timesteps[i].item()
            t_next = timesteps[i + 1].item()
            
            t_tensor = torch.full((batch_size,), t_cur, device=self.device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = model(x, t_tensor, condition)
            
            # Check for numerical issues
            if not torch.isfinite(predicted_noise).all():
                print(f"WARNING: NaN/Inf in predicted noise at step {i}")
                predicted_noise = torch.nan_to_num(predicted_noise, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Get schedule values
            alphas_cumprod_t = ddpm.alphas_cumprod[int(t_cur)]
            
            eps = 1e-10
            sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod_t.clamp(min=eps))
            sqrt_one_minus_alphas_cumprod_t = torch.sqrt((1 - alphas_cumprod_t).clamp(min=eps))
            
            # DDIM sampling step
            x0_pred = (x - sqrt_one_minus_alphas_cumprod_t * predicted_noise) / sqrt_alphas_cumprod_t
            
            # Compute next x
            if t_next > 0:
                alphas_cumprod_next = ddpm.alphas_cumprod[int(t_next)]
                sqrt_alphas_cumprod_next = torch.sqrt(alphas_cumprod_next.clamp(min=eps))
                sqrt_one_minus_alphas_cumprod_next = torch.sqrt((1 - alphas_cumprod_next).clamp(min=eps))
                x = sqrt_alphas_cumprod_next * x0_pred + sqrt_one_minus_alphas_cumprod_next * predicted_noise
            else:
                x = x0_pred
            
            # Check for divergence
            if not torch.isfinite(x).all():
                print(f"WARNING: Sampling diverged at step {i}")
                x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
        
        return x
    
    @torch.no_grad()
    def compute_reward(
        self,
        x_gen: torch.Tensor,
        condition: torch.Tensor,
        reward_model: nn.Module,
        reward_ddpm: DDPM1D,
    ) -> float:
        """
        Compute reward as negative log-likelihood under reward model.
        
        Args:
            x_gen: Generated samples
            condition: Conditioning variable (target for reward model)
            reward_model: Model to compute likelihood
            reward_ddpm: DDPM wrapper for reward model
            
        Returns:
            Mean reward (scalar)
        """
        reward_model.eval()
        batch_size = condition.shape[0]
        
        nll = 0.0
        
        # Monte Carlo estimate of NLL
        for _ in range(self.mc_steps):
            # Sample random timestep
            t = torch.randint(0, self.config.timesteps, (batch_size,), device=self.device)
            
            # Add noise to condition
            noise = torch.randn_like(condition)
            alphas_cumprod = reward_ddpm.alphas_cumprod[t].view(-1, 1)
            condition_noisy = torch.sqrt(alphas_cumprod) * condition + torch.sqrt(1 - alphas_cumprod) * noise
            
            # Predict noise conditioned on x_gen
            predicted_noise = reward_model(condition_noisy, t, x_gen)
            
            # MSE as proxy for NLL
            nll += ((predicted_noise - noise) ** 2).sum(dim=1).mean()
        
        nll = nll / self.mc_steps
        reward = -nll.item()  # Negative NLL as reward
        
        return reward
    
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
        Same as PPO-DDMEC sync phase.
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
    
    def run_one_way_es(
        self,
        x_gen_batch: torch.Tensor,
        x_cond_batch: torch.Tensor,
        gen_model: nn.Module,
        gen_ddpm: DDPM1D,
        reward_model: nn.Module,
        reward_ddpm: DDPM1D,
        es_optimizer: ESOptimizer,
        sync_optimizer: torch.optim.Optimizer,
        direction: str,
    ) -> Dict[str, float]:
        """
        Run one direction of ES-based cooperative training.
        
        Algorithm:
        1. ES updates generator (parameter-space exploration)
        2. Generate samples with updated generator
        3. Sync reward model on new samples (supervised)
        
        Args:
            x_gen_batch: Target samples (what generator should produce)
            x_cond_batch: Conditioning samples
            gen_model: Generator model
            gen_ddpm: DDPM wrapper for generator
            reward_model: Reward model (critic)
            reward_ddpm: DDPM wrapper for reward model
            es_optimizer: ES optimizer for generator
            sync_optimizer: Optimizer for reward model sync
            direction: "1->2" or "2->1" for logging
            
        Returns:
            Metrics dictionary
        """
        
        def compute_reward_fn():
            """Reward function for ES evaluation."""
            # Sample from generator (greedy decoding)
            x_gen = self.sample_trajectory(
                gen_model, gen_ddpm, x_cond_batch,
                num_steps=self.num_sampling_steps
            )
            
            # Compute reward as negative NLL under reward model
            reward = self.compute_reward(
                x_gen, x_cond_batch, reward_model, reward_ddpm
            )
            
            return reward
        
        # Step 1: ES update for generator
        es_metrics = es_optimizer.step(compute_reward_fn)
        
        # Step 2: Generate samples with updated generator
        gen_model.eval()
        with torch.no_grad():
            x_gen = self.sample_trajectory(
                gen_model, gen_ddpm, x_cond_batch,
                num_steps=self.num_sampling_steps
            )
        
        # Step 3: Sync reward model (supervised learning)
        self.sync_reward_model(
            x_gen, x_cond_batch, reward_model, reward_ddpm,
            sync_optimizer, num_updates=2
        )
        
        return {f"{direction}_{k}": v for k, v in es_metrics.items()}
    
    def train_step(
        self,
        x1_batch: torch.Tensor,
        x2_batch: torch.Tensor,
        optimizer_1: torch.optim.Optimizer,
        optimizer_2: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """
        Single ES-DDMEC training step.
        
        Phases:
        1. Warmup (step < warmup_steps): Supervised training
        2. Cooperative (step >= warmup_steps): ES-based optimization
        """
        self.gen_step += 1
        
        # Warmup phase: supervised training only
        if self.gen_step < self.warmup_steps:
            metrics = self.train_supervised(x1_batch, x2_batch, optimizer_1, optimizer_2)
            metrics["phase"] = "warmup"
            return metrics
        
        # Setup ES optimizers after warmup (once)
        if self.gen_step == self.warmup_steps:
            self.setup_es_optimizers()
        
        # ES cooperative phase
        metrics = {}
        
        # Direction 1: Model 1 generates x1 given x2, Model 2 is reward
        metrics_1 = self.run_one_way_es(
            x1_batch, x2_batch,
            self.ddpm_1.model, self.ddpm_1,
            self.ddpm_2.model, self.ddpm_2,
            self.es_optimizer_1,
            optimizer_2,
            direction="1->2"
        )
        metrics.update(metrics_1)
        
        # Direction 2: Model 2 generates x2 given x1, Model 1 is reward
        metrics_2 = self.run_one_way_es(
            x2_batch, x1_batch,
            self.ddpm_2.model, self.ddpm_2,
            self.ddpm_1.model, self.ddpm_1,
            self.es_optimizer_2,
            optimizer_1,
            direction="2->1"
        )
        metrics.update(metrics_2)
        
        metrics["phase"] = "es_cooperative"
        return metrics
    
    def compute_kl_divergence(self, mu_p: float, sigma_p: float, mu_q: float, sigma_q: float) -> float:
        """Compute KL divergence between two 1D Gaussians."""
        eps = 1e-10
        if sigma_p <= 0 or sigma_q <= 0:
            return float('inf')
        
        var_p = sigma_p ** 2
        var_q = sigma_q ** 2
        log_ratio = np.log(var_q / var_p)
        
        kl = 0.5 * (((mu_p - mu_q) ** 2) / var_q + var_p / var_q - 1.0 + log_ratio)
        return max(kl, 0.0)
    
    def compute_information_metrics(
        self,
        samples_1: torch.Tensor,
        samples_2: torch.Tensor,
        true_mu_1: float = 2.0,
        true_sigma_1: float = 1.0,
        true_mu_2: float = 10.0,
        true_sigma_2: float = 1.0,
    ) -> dict:
        """Compute information-theoretic metrics for the coupling."""
        eps = 1e-10
        
        s1_np = samples_1.cpu().numpy().flatten()
        s2_np = samples_2.cpu().numpy().flatten()
        
        # Check validity
        if not np.isfinite(s1_np).all() or not np.isfinite(s2_np).all():
            return {
                "kl_div_1": float('inf'),
                "kl_div_2": float('inf'),
                "kl_div_total": float('inf'),
                "mutual_information": 0.0,
                "learned_mu_1": 0.0,
                "learned_sigma_1": 1.0,
                "learned_mu_2": 0.0,
                "learned_sigma_2": 1.0,
            }
        
        # Compute statistics
        mu_1, sigma_1 = float(s1_np.mean()), float(max(s1_np.std(), eps))
        mu_2, sigma_2 = float(s2_np.mean()), float(max(s2_np.std(), eps))
        
        # KL divergences
        kl_1 = self.compute_kl_divergence(mu_1, sigma_1, true_mu_1, true_sigma_1)
        kl_2 = self.compute_kl_divergence(mu_2, sigma_2, true_mu_2, true_sigma_2)
        
        # Entropies (Gaussian approximation)
        h_x = 0.5 * np.log(2 * np.pi * np.e * sigma_1**2)
        h_y = 0.5 * np.log(2 * np.pi * np.e * sigma_2**2)
        
        # Joint entropy
        joint = np.stack([s1_np, s2_np], axis=1)
        cov = np.cov(joint.T) + eps * np.eye(2)
        det_cov = np.linalg.det(cov)
        if det_cov <= 0:
            det_cov = eps
        h_xy = np.log(2 * np.pi * np.e) + 0.5 * np.log(det_cov)
        
        # Mutual information
        mi = h_x + h_y - h_xy
        
        return {
            "kl_div_1": kl_1,
            "kl_div_2": kl_2,
            "kl_div_total": kl_1 + kl_2,
            "entropy_x": h_x,
            "entropy_y": h_y,
            "joint_entropy": h_xy,
            "mutual_information": mi,
            "conditional_entropy_x_given_y": h_xy - h_y,
            "conditional_entropy_y_given_x": h_xy - h_x,
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
        axes[0, 0].scatter(x2_test, x1_gen, alpha=0.3, s=10, c='blue')
        axes[0, 0].set_xlabel("x2 (condition)")
        axes[0, 0].set_ylabel("x1 (generated)")
        axes[0, 0].set_title(f"ES-DDMEC Coupling: x2→x1 (Epoch {epoch})")
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].scatter(x1_test, x2_gen, alpha=0.3, s=10, c='orange')
        axes[0, 1].set_xlabel("x1 (condition)")
        axes[0, 1].set_ylabel("x2 (generated)")
        axes[0, 1].set_title(f"ES-DDMEC Coupling: x1→x2 (Epoch {epoch})")
        axes[0, 1].grid(True, alpha=0.3)
        
        # Correlation
        corr_1 = np.corrcoef(x2_test.flatten(), x1_gen.flatten())[0, 1]
        corr_2 = np.corrcoef(x1_test.flatten(), x2_gen.flatten())[0, 1]
        axes[0, 2].bar(['x2→x1', 'x1→x2'], [corr_1, corr_2], alpha=0.7, color=['blue', 'orange'])
        axes[0, 2].axhline(1.0, color='red', linestyle='--', label='Perfect')
        axes[0, 2].set_ylabel("Correlation")
        axes[0, 2].set_title("Coupling Strength")
        axes[0, 2].set_ylim([0, 1.1])
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3, axis='y')
        
        # Row 2: Marginal distributions
        axes[1, 0].hist(x1_gen, bins=30, density=True, alpha=0.7, color='blue', label='Generated')
        axes[1, 0].axvline(2.0, color='red', linestyle='--', linewidth=2, label='Target mean')
        axes[1, 0].set_xlabel("x1")
        axes[1, 0].set_ylabel("Density")
        axes[1, 0].set_title(f"Marginal: x1 (μ={x1_gen.mean():.2f}, σ={x1_gen.std():.2f})")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].hist(x2_gen, bins=30, density=True, alpha=0.7, color='orange', label='Generated')
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
        
        plt.suptitle(f'ES-DDMEC Training - Epoch {epoch}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
    
    @torch.no_grad()
    def sample_coupled(
        self,
        x_condition: torch.Tensor,
        direction: str = "1->2",
        num_steps: int = None,
    ) -> torch.Tensor:
        """Sample from learned coupling."""
        if num_steps is None:
            num_steps = self.num_sampling_steps
            
        if direction == "1->2":
            model = self.ddpm_1.model
            ddpm = self.ddpm_1
        else:
            model = self.ddpm_2.model
            ddpm = self.ddpm_2
        
        return self.sample_trajectory(model, ddpm, x_condition, num_steps)
    
    def train(
        self,
        dataset_1: torch.Tensor,
        dataset_2: torch.Tensor,
        num_epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-4,
        run_dir: str = "runs/es_ddmec",
    ):
        """
        Main training loop for ES-DDMEC.
        
        Args:
            dataset_1: Samples from distribution 1
            dataset_2: Samples from distribution 2
            num_epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate for supervised phases
            run_dir: Directory to save outputs
        """
        # Setup directories
        os.makedirs(run_dir, exist_ok=True)
        checkpoints_dir = os.path.join(run_dir, "checkpoints")
        plots_dir = os.path.join(run_dir, "plots")
        tables_dir = os.path.join(run_dir, "tables")
        logs_dir = os.path.join(run_dir, "logs")
        tensorboard_dir = os.path.join(run_dir, "tensorboard")
        
        for d in [checkpoints_dir, plots_dir, tables_dir, logs_dir]:
            os.makedirs(d, exist_ok=True)
        
        # Initialize TensorBoard
        if self.use_tensorboard:
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=tensorboard_dir)
            print(f"TensorBoard: {tensorboard_dir}")
        
        num_samples = min(len(dataset_1), len(dataset_2))
        num_batches = num_samples // batch_size
        
        # Calculate warmup steps
        self.warmup_steps = self.warmup_epochs * num_batches
        print(f"\nWarmup: {self.warmup_epochs} epochs = {self.warmup_steps} steps")
        
        # Setup optimizers (for supervised phases)
        optimizer_1 = torch.optim.Adam(self.ddpm_1.model.parameters(), lr=lr)
        optimizer_2 = torch.optim.Adam(self.ddpm_2.model.parameters(), lr=lr)
        
        # Track best model
        best_kl_total = float('inf')
        best_epoch = 0
        
        # Initialize logging
        local_log_path = os.path.join(logs_dir, "training_log.json")
        self.local_log = {
            "experiment": "ES-DDMEC",
            "run_dir": run_dir,
            "config": {
                "population_size": self.es_config.population_size,
                "sigma": self.es_config.sigma,
                "es_lr": self.es_config.learning_rate,
                "warmup_epochs": self.warmup_epochs,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "supervised_lr": lr,
            },
            "epochs": [],
        }
        
        # CSV for metrics
        metrics_csv_path = os.path.join(tables_dir, "epoch_metrics.csv")
        csv_headers = [
            "epoch", "phase",
            "kl_div_1_fwd", "kl_div_2_fwd", "kl_div_total_fwd",
            "mutual_info_fwd",
            "corr_x2_to_x1", "corr_x1_to_x2",
            "mae_x2_to_x1", "mae_x1_to_x2",
            "es_reward_mean_1", "es_reward_mean_2",
            "best_kl_total", "best_epoch",
        ]
        
        with open(metrics_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)
        
        print(f"\nTraining ES-DDMEC for {num_epochs} epochs")
        print(f"  Batches per epoch: {num_batches}")
        print(f"  Checkpoints: {checkpoints_dir}")
        print(f"  Plots: {plots_dir}")
        print("-" * 60)
        
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
            
            # Compute epoch averages
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            current_phase = epoch_metrics.get("phase", ["warmup"])[-1] if epoch_metrics.get("phase") else "warmup"
            print(f"  Phase: {current_phase}")
            
            # Log training metrics
            avg_metrics = {}
            for k, v in epoch_metrics.items():
                if v and k != "phase":
                    avg_metrics[k] = np.mean(v)
            
            if "loss_1" in avg_metrics:
                print(f"  Supervised Loss 1: {avg_metrics['loss_1']:.4f}")
                print(f"  Supervised Loss 2: {avg_metrics['loss_2']:.4f}")
            
            if "1->2_reward_mean" in avg_metrics:
                print(f"  ES Reward 1->2: {avg_metrics['1->2_reward_mean']:.4f}")
                print(f"  ES Reward 2->1: {avg_metrics['2->1_reward_mean']:.4f}")
            
            # Evaluate coupling
            with torch.no_grad():
                num_eval = min(1000, num_samples)
                x2_eval = dataset_2[:num_eval].to(self.device)
                x1_eval = dataset_1[:num_eval].to(self.device)
                
                # Generate samples
                x1_generated = self.sample_coupled(x2_eval, direction="1->2")
                x2_generated = self.sample_coupled(x1_eval, direction="2->1")
                
                # Information metrics
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
                
                # Coupling metrics
                x1_gen_np = x1_generated.cpu().numpy().flatten()
                x2_gen_np = x2_generated.cpu().numpy().flatten()
                x1_eval_np = x1_eval.cpu().numpy().flatten()
                x2_eval_np = x2_eval.cpu().numpy().flatten()
                
                corr_x2_to_x1 = np.corrcoef(x2_eval_np, x1_gen_np)[0, 1]
                corr_x1_to_x2 = np.corrcoef(x1_eval_np, x2_gen_np)[0, 1]
                mae_x2_to_x1 = np.abs(x1_gen_np - (x2_eval_np - 8.0)).mean()
                mae_x1_to_x2 = np.abs(x2_gen_np - (x1_eval_np + 8.0)).mean()
                
                avg_kl_total = (info_metrics_1['kl_div_total'] + info_metrics_2['kl_div_total']) / 2
                
                print(f"\n  Information Metrics:")
                print(f"    KL Total (x2→x1): {info_metrics_1['kl_div_total']:.4f}")
                print(f"    KL Total (x1→x2): {info_metrics_2['kl_div_total']:.4f}")
                print(f"    Mutual Info (x2→x1): {info_metrics_1['mutual_information']:.4f}")
                
                print(f"\n  Coupling Metrics:")
                print(f"    Correlation x2→x1: {corr_x2_to_x1:.4f}")
                print(f"    Correlation x1→x2: {corr_x1_to_x2:.4f}")
                print(f"    MAE x2→x1: {mae_x2_to_x1:.4f}")
                print(f"    MAE x1→x2: {mae_x1_to_x2:.4f}")
                
                # Generate plot
                epoch_plots_dir = os.path.join(plots_dir, f"epoch_{epoch+1:04d}")
                os.makedirs(epoch_plots_dir, exist_ok=True)
                plot_path = os.path.join(epoch_plots_dir, "coupling_plot.png")
                
                self.generate_epoch_plot(
                    x1_gen_np, x2_gen_np,
                    x1_eval_np, x2_eval_np,
                    epoch + 1, plot_path
                )
                
                # Also save at top level
                top_plot = os.path.join(plots_dir, f"epoch_{epoch+1:04d}.png")
                shutil.copy(plot_path, top_plot)
                
                # Save to CSV
                es_reward_1 = avg_metrics.get("1->2_reward_mean", 0.0)
                es_reward_2 = avg_metrics.get("2->1_reward_mean", 0.0)
                
                csv_row = [
                    epoch + 1, current_phase,
                    info_metrics_1['kl_div_1'], info_metrics_1['kl_div_2'], info_metrics_1['kl_div_total'],
                    info_metrics_1['mutual_information'],
                    corr_x2_to_x1, corr_x1_to_x2,
                    mae_x2_to_x1, mae_x1_to_x2,
                    es_reward_1, es_reward_2,
                    best_kl_total, best_epoch,
                ]
                
                with open(metrics_csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(csv_row)
                
                # Update local log
                epoch_log = {
                    "epoch": epoch + 1,
                    "phase": current_phase,
                    "info_metrics_forward": info_metrics_1,
                    "info_metrics_backward": info_metrics_2,
                    "coupling_metrics": {
                        "corr_x2_to_x1": float(corr_x2_to_x1),
                        "corr_x1_to_x2": float(corr_x1_to_x2),
                        "mae_x2_to_x1": float(mae_x2_to_x1),
                        "mae_x1_to_x2": float(mae_x1_to_x2),
                    },
                    "es_metrics": {
                        "reward_mean_1": es_reward_1,
                        "reward_mean_2": es_reward_2,
                    },
                    "best": {"kl_total": best_kl_total, "epoch": best_epoch},
                }
                self.local_log["epochs"].append(epoch_log)
                
                # Save log
                with open(local_log_path, 'w') as f:
                    json.dump(self.local_log, f, indent=2)
                
                # TensorBoard logging
                if self.use_tensorboard and self.tb_writer:
                    self.tb_writer.add_scalar('info/kl_total_forward', info_metrics_1['kl_div_total'], epoch + 1)
                    self.tb_writer.add_scalar('info/kl_total_backward', info_metrics_2['kl_div_total'], epoch + 1)
                    self.tb_writer.add_scalar('info/mutual_information', info_metrics_1['mutual_information'], epoch + 1)
                    self.tb_writer.add_scalar('coupling/corr_x2_to_x1', corr_x2_to_x1, epoch + 1)
                    self.tb_writer.add_scalar('coupling/corr_x1_to_x2', corr_x1_to_x2, epoch + 1)
                    self.tb_writer.add_scalar('coupling/mae_x2_to_x1', mae_x2_to_x1, epoch + 1)
                    self.tb_writer.add_scalar('coupling/mae_x1_to_x2', mae_x1_to_x2, epoch + 1)
                    
                    if "1->2_reward_mean" in avg_metrics:
                        self.tb_writer.add_scalar('es/reward_1_to_2', avg_metrics['1->2_reward_mean'], epoch + 1)
                        self.tb_writer.add_scalar('es/reward_2_to_1', avg_metrics['2->1_reward_mean'], epoch + 1)
                    
                    self.tb_writer.flush()
                
                # Save checkpoint
                checkpoint_path = os.path.join(checkpoints_dir, f"epoch_{epoch+1:04d}.pt")
                torch.save({
                    "epoch": epoch + 1,
                    "model_1_state_dict": self.ddpm_1.model.state_dict(),
                    "model_2_state_dict": self.ddpm_2.model.state_dict(),
                    "optimizer_1_state_dict": optimizer_1.state_dict(),
                    "optimizer_2_state_dict": optimizer_2.state_dict(),
                    "config": vars(self.config),
                    "es_config": {
                        "population_size": self.es_config.population_size,
                        "sigma": self.es_config.sigma,
                        "learning_rate": self.es_config.learning_rate,
                    },
                    "gen_step": self.gen_step,
                    "info_metrics_1": info_metrics_1,
                    "info_metrics_2": info_metrics_2,
                }, checkpoint_path)
                
                # Check for best model
                if avg_kl_total < best_kl_total:
                    best_kl_total = avg_kl_total
                    best_epoch = epoch + 1
                    
                    best_path = os.path.join(checkpoints_dir, "best_model.pt")
                    torch.save({
                        "epoch": epoch + 1,
                        "model_1_state_dict": self.ddpm_1.model.state_dict(),
                        "model_2_state_dict": self.ddpm_2.model.state_dict(),
                        "config": vars(self.config),
                        "es_config": {
                            "population_size": self.es_config.population_size,
                            "sigma": self.es_config.sigma,
                            "learning_rate": self.es_config.learning_rate,
                        },
                        "gen_step": self.gen_step,
                        "info_metrics_1": info_metrics_1,
                        "info_metrics_2": info_metrics_2,
                        "best_kl_total": best_kl_total,
                    }, best_path)
                    
                    print(f"\n  [BEST] New best model! (Avg KL Total: {best_kl_total:.4f})")
        
        # Close TensorBoard
        if self.use_tensorboard and self.tb_writer:
            self.tb_writer.close()
        
        print("\n" + "="*60)
        print("ES-DDMEC Training Complete!")
        print("="*60)
        print(f"Best epoch: {best_epoch}")
        print(f"Best Avg KL Total: {best_kl_total:.4f}")
        print(f"\nOutputs:")
        print(f"  Checkpoints: {checkpoints_dir}")
        print(f"  Plots: {plots_dir}")
        print(f"  Metrics CSV: {metrics_csv_path}")
        print(f"  Log: {local_log_path}")
        if self.use_tensorboard:
            print(f"  TensorBoard: tensorboard --logdir={tensorboard_dir}")
        print("="*60)


def main():
    """Example usage of ES-DDMEC."""
    print("="*60)
    print("ES-DDMEC Demo: Evolution Strategies for 1D Gaussian Coupling")
    print("="*60)
    
    # Check for pre-trained models
    if not os.path.exists("ddpm_1d_2.pt") or not os.path.exists("ddpm_1d_10.pt"):
        print("\nERROR: Pre-trained models not found!")
        print("Please run train_both_ddpm.py first.")
        return
    
    # Initialize ES-DDMEC
    es_ddmec = ESDDMEC1D(
        model_path_1="ddpm_1d_2.pt",
        model_path_2="ddpm_1d_10.pt",
        use_wandb=False,
        use_tensorboard=True,
        # ES hyperparameters from paper
        population_size=30,
        sigma=0.001,
        es_lr=5e-4,
        warmup_epochs=10,
    )
    
    # Create coupled data
    num_samples = 10000
    z = torch.randn(num_samples, 1)
    x1_train = z + 2.0   # N(2, 1)
    x2_train = z + 10.0  # N(10, 1)
    
    print(f"\nDataset:")
    print(f"  X1: mean={x1_train.mean():.2f}, std={x1_train.std():.2f}")
    print(f"  X2: mean={x2_train.mean():.2f}, std={x2_train.std():.2f}")
    
    # Train
    es_ddmec.train(
        dataset_1=x1_train,
        dataset_2=x2_train,
        num_epochs=30,
        batch_size=64,
        lr=1e-4,
        run_dir="runs/es_ddmec_demo",
    )


if __name__ == "__main__":
    main()
