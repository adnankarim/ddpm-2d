# DDMEC Algorithm: Implementation Guide

This document explains how the DDMEC algorithm is implemented in `ddmec_1d.py` and maps it to the reference implementation.

## Algorithm Overview

DDMEC learns minimum entropy coupling through **cooperative training** of two diffusion models:

```
Model 1: p_θ(x₁|x₂)  - generates x₁ given x₂
Model 2: p_φ(x₂|x₁)  - generates x₂ given x₁
```

### Training Procedure

```python
# Phase 1: Warmup (Steps 1 to T_warmup)
for step in range(1, warmup_steps):
    # Standard supervised denoising
    train_supervised(model_1, x1, x2)
    train_supervised(model_2, x2, x1)

# After warmup: freeze priors
frozen_model_1 = deepcopy(model_1).eval()
frozen_model_2 = deepcopy(model_2).eval()

# Phase 2: Cooperative Training (Steps T_warmup+1 to T_max)
for step in range(warmup_steps, max_steps):
    
    # === Direction 1: Model 1 generates, Model 2 rewards ===
    
    # Step 1: Generate trajectory with log probs (Policy)
    trajectory_1, log_probs_1 = sample_with_logprob(
        model=model_1, 
        condition=x2,
        num_steps=50
    )
    x1_generated = trajectory_1[-1]
    
    # Step 2: Compute reward (Critic)
    reward_1 = -negative_log_likelihood(
        model=model_2,
        x=x2,
        condition=x1_generated
    )
    
    # Step 3: Policy gradient update (DPOK)
    loss_pg = policy_gradient_loss(
        log_probs=log_probs_1,
        rewards=reward_1,
        model=model_1,
        frozen_model=frozen_model_1,  # KL regularization
        kl_weight=0.1
    )
    update(model_1, loss_pg)
    
    # Step 4: Sync reward model (Supervised)
    sync_loss = denoising_loss(
        model=model_2,
        x=x2,
        condition=x1_generated
    )
    update(model_2, sync_loss)
    
    # === Direction 2: Model 2 generates, Model 1 rewards ===
    
    # (Same process, roles reversed)
    trajectory_2, log_probs_2 = sample_with_logprob(model_2, x1)
    reward_2 = -negative_log_likelihood(model_1, x1, trajectory_2[-1])
    loss_pg_2 = policy_gradient_loss(log_probs_2, reward_2, model_2, frozen_model_2)
    update(model_2, loss_pg_2)
    sync_loss_2 = denoising_loss(model_1, x1, trajectory_2[-1])
    update(model_1, sync_loss_2)
```

## Key Components

### 1. Sampling with Log Probability

**Purpose**: Generate samples while tracking probabilities for policy gradient.

```python
def sample_trajectory_with_logprob(model, condition, num_steps=50):
    """
    DDIM sampling with log probability tracking.
    
    Returns:
        trajectory: [x_T, x_{T-1}, ..., x_1, x_0]
        log_probs: [log p(x_{t-1}|x_t), ...]
    """
    x = torch.randn_like(condition)  # Start from noise
    trajectory = [x]
    log_probs = []
    
    for t in reversed(range(num_steps)):
        # Predict noise
        eps_pred = model(x, t, condition)
        
        # DDIM step: x_{t-1} = f(x_t, eps_pred)
        x_next = ddim_step(x, eps_pred, t)
        
        # Compute log p(x_next | x_t, condition)
        log_prob = compute_log_prob(x, x_next, eps_pred, t)
        
        trajectory.append(x_next)
        log_probs.append(log_prob)
        x = x_next
    
    return trajectory, log_probs
```

**Reference mapping**:
```python
# Reference: src/diffusion/sde.py
x_traj, log_probs, timesteps, timesteps_prev = self.sde.sample_from_model_with_logprob(
    score_function=denoiser_gen,
    x0=torch.randn_like(x),
    c=y_cond,
    guidance=self.args.guidance,
    num_steps=self.args.num_steps_train
)
```

### 2. Reward Computation

**Purpose**: Evaluate quality of generated samples using partner model.

```python
def compute_reward(x_gen, condition, reward_model, mc_steps=5):
    """
    Reward = -NLL under reward model.
    
    Monte Carlo estimate of:
        R(x_gen, condition) = -E_t[||ε - ε_θ(√ᾱ_t·condition + √(1-ᾱ_t)·ε, t, x_gen)||²]
    """
    nll = 0.0
    
    for _ in range(mc_steps):
        # Sample random timestep
        t = random_timestep()
        
        # Add noise to condition
        condition_noisy = sqrt(alpha_bar_t) * condition + sqrt(1 - alpha_bar_t) * noise
        
        # Predict noise conditioned on generated x
        eps_pred = reward_model(condition_noisy, t, x_gen)
        
        # MSE is proxy for NLL
        nll += mse(eps_pred, noise)
    
    return -nll / mc_steps  # Negative NLL as reward
```

**Reference mapping**:
```python
# Reference: gen_trajectories method
nll = self.sde.nll_grad(
    y_cond,
    cond=x_gen,
    model=denoiser_reward,
    t=t,
    w=w,
    type="eps",
    one_step=self.args.one_step,
    reduce=self.args.reduce
).detach()
rewards = -nll
```

### 3. Policy Gradient Update (DPOK)

**Purpose**: Update generator to maximize reward with KL constraint.

```python
def policy_gradient_update(trajectory, log_probs, rewards, model, frozen_model, kl_weight=0.1):
    """
    PPO-style update with KL regularization.
    
    Loss = -E[A_t · log π_θ(x_{t-1}|x_t)] + λ_KL · D_KL(π_θ || π_frozen)
    
    Where:
        A_t = (R - R_mean) / R_std  (advantage)
        π_θ is current policy
        π_frozen is frozen prior
    """
    # Normalize rewards -> advantages
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    
    total_loss = 0
    
    for t, (x_t, x_next, log_prob_old) in enumerate(trajectory):
        # Recompute with gradients
        eps_pred = model(x_t, t, condition)
        log_prob_new = compute_log_prob(x_t, x_next, eps_pred, t)
        
        # PPO clipped loss
        ratio = exp(log_prob_new - log_prob_old)
        clipped_ratio = clip(ratio, 1 - clip_range, 1 + clip_range)
        pg_loss = -min(advantages * ratio, advantages * clipped_ratio)
        
        # KL regularization
        eps_frozen = frozen_model(x_t, t, condition=None)  # Unconditional
        kl_reg = mse(eps_pred, eps_frozen)
        
        loss = pg_loss + kl_weight * kl_reg
        total_loss += loss
    
    backward(total_loss)
    update_parameters(model)
```

**Reference mapping**:
```python
# Reference: policy_gradient_update_dpok method
if self.args.pg_importance_sampling == 1:
    ratio = torch.exp(log_prob - log_prob_old)
    clipped_loss = -self.args.reward_weight[mod] * advantages * torch.clamp(
        ratio,
        1.0 - self.args.clip_range_pg,
        1.0 + self.args.clip_range_pg,
    ).float()
    loss = clipped_loss
else:
    loss = -advantages * log_prob

# KL regularization
noise_pred_old = denoiser_gen(x_t, cond=None, t=t, old=True)
kl_reg = ((noise_pred - noise_pred_old) ** 2).mean()

loss = loss.mean() + self.args.kl_weight[mod] * kl_reg
```

### 4. Model Synchronization

**Purpose**: Align reward model with generator's new distribution.

```python
def sync_reward_model(x_gen, condition, reward_model, optimizer, num_updates=5):
    """
    Standard supervised denoising on new samples.
    
    After generator updates via RL, reward model learns:
        p_φ(condition | x_gen)
    
    This ensures both models represent the same joint:
        p_θ(x₁, x₂) ≈ p_φ(x₁, x₂)
    """
    for _ in range(num_updates):
        # Standard denoising objective
        t = random_timestep()
        condition_noisy = add_noise(condition, t)
        eps_pred = reward_model(condition_noisy, t, x_gen)
        
        loss = mse(eps_pred, true_noise)
        
        backward(loss)
        update_parameters(reward_model)
```

**Reference mapping**:
```python
# Reference: update_models method
def update_models(self, x, x_cond, denoiser, mod, opt, nb):
    for _ in range(nb):
        rand_idx = torch.randperm(x_cond.size(0))
        x_rand, x_cond_rand = x[rand_idx], x_cond[rand_idx]
        
        for x_batch, cond_batch in zip(batched_x, batched_cond):
            self.train_denoiser(
                x=x_batch.detach(),
                x_cond=cond_batch.detach(),
                denoiser=denoiser,
                mod=mod,
                opt=opt
            )
```

## Hyperparameters

### Critical Parameters

| Parameter | Default | Description | Tuning Advice |
|-----------|---------|-------------|---------------|
| `warmup_steps` | 1000 | Supervised pre-training steps | Increase if models are unstable |
| `kl_weight` | 0.1 | Weight for KL regularization | Higher = stay closer to marginals |
| `clip_range` | 0.2 | PPO clipping range | Standard PPO value |
| `mc_steps` | 5 | Monte Carlo samples for reward | More = better estimate, slower |
| `num_updates` | 5 | Sync updates per step | More = better alignment |
| `lr` | 1e-4 | Learning rate | Lower if training unstable |

### Reference Values

From the reference implementation:
```python
args.warmup = 1000              # Warmup steps
args.kl_weight = [0.1, 0.1]     # KL weights for both models
args.clip_range_pg = 0.2        # PPO clipping
args.mc_steps = 5               # MC samples for NLL
args.nb_update_reward = 5       # Sync updates
args.lr = [1e-4, 1e-5]         # Learning rates [controlnet, unet]
```

## Comparison with Reference

### Architecture Differences

| Aspect | Reference (Images) | This Implementation (1D) |
|--------|-------------------|--------------------------|
| Model | ControlNet + UNet | SimpleMLP |
| Conditioning | ControlNet fusion | Concatenation |
| Guidance | Classifier-free guidance | Simple guidance |
| Timesteps | 1000 | 1000 |
| Sampling | DDIM with eta | DDIM |

### Algorithm Similarities

✅ **Same core algorithm**:
- Two-phase training (warmup + cooperative)
- Frozen priors for KL regularization
- Alternating generator/reward roles
- PPO with clipping
- Model synchronization

✅ **Same loss formulation**:
- Policy gradient with advantages
- KL regularization vs frozen prior
- Supervised sync updates

✅ **Same reward**:
- Negative log-likelihood
- Monte Carlo estimation

## Why DDMEC Works

### Intuition

1. **Warmup phase**: Models learn good marginals
   - Model 1 learns p(x₁|x₂) 
   - Model 2 learns p(x₂|x₁)
   - But not necessarily minimum entropy coupling

2. **Cooperative phase**: Models learn optimal coupling
   - Generator (policy) proposes coupling
   - Reward (critic) evaluates how well it matches
   - PPO ensures stable updates
   - KL prevents violating marginals
   - Sync ensures consistency

3. **Minimum entropy emerges**:
   - Maximizing likelihood = minimizing entropy
   - Alternating directions ensures symmetry
   - Frozen priors preserve marginals
   - Result: minimum entropy coupling!

### Mathematical Foundation

The DDMEC objective is:

```
min H(X₁, X₂)  subject to  X₁ ~ p₁, X₂ ~ p₂
```

Which decomposes as:

```
H(X₁, X₂) = H(X₁) + H(X₂|X₁) = H(X₂) + H(X₁|X₂)
```

DDMEC:
- **Fixes marginals**: H(X₁), H(X₂) via frozen priors
- **Minimizes conditionals**: H(X₂|X₁), H(X₁|X₂) via reward maximization
- **Ensures consistency**: p_θ(x₁,x₂) = p_φ(x₁,x₂) via synchronization

## Debugging Guide

### Training is Unstable

**Symptoms**: Loss explodes, rewards collapse to zero, NaN values

**Solutions**:
1. Increase `warmup_steps` to 2000 or more
2. Reduce learning rate to 5e-5
3. Increase `kl_weight` to 0.5 or 1.0
4. Check gradient norms, clip if > 1.0

### Marginals Don't Preserve

**Symptoms**: Generated samples have wrong mean/std

**Solutions**:
1. Increase `kl_weight` significantly (try 1.0, 5.0)
2. Ensure frozen models are truly frozen (`.requires_grad_(False)`)
3. Use unconditional prior in KL term
4. Check that frozen models were saved after good warmup

### Coupling is Weak

**Symptoms**: Low correlation, high round-trip error

**Solutions**:
1. Increase `num_updates` for sync (try 10-20)
2. Increase `mc_steps` for better reward estimates (try 10)
3. Train for more epochs
4. Check that reward signal is informative (not constant)

### Memory Issues

**Solutions**:
1. Reduce `batch_size`
2. Reduce `num_steps` in sampling
3. Use gradient checkpointing
4. Clear trajectory after each update

## Next Steps

1. **Verify warmup**: Ensure models can denoise well before cooperative phase
2. **Monitor metrics**: Track rewards, KL reg, advantages
3. **Visualize coupling**: Plot scatter plots to see relationship
4. **Compare to baseline**: Independent sampling vs DDMEC coupling
5. **Scale up**: Try 2D, images, or other modalities

Good luck! 🚀

