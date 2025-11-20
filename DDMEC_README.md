# DDMEC Implementation for 1D Gaussian Models

This implementation provides **DDMEC (Denoising Diffusion Minimum Entropy Coupling)** for learning the minimum entropy coupling between two 1D Gaussian distributions.

## What is DDMEC?

DDMEC is a method that uses **two diffusion models** to learn the optimal coupling between two probability distributions. The key idea is:

1. **Model 1** learns `p(x1|x2)` - how to generate samples from distribution 1 given distribution 2
2. **Model 2** learns `p(x2|x1)` - how to generate samples from distribution 2 given distribution 1
3. They cooperatively train to minimize the **joint entropy** of the coupling

### Training Phases

#### Phase 1: Warmup (Supervised)
- Both models are trained with standard denoising objectives
- Model 1 learns to denoise x1 conditioned on x2
- Model 2 learns to denoise x2 conditioned on x1
- This provides good initialization before cooperative training

#### Phase 2: Cooperative (DDMEC)
The models alternate between two roles:

**Generator (Policy):** 
- Generates samples using DDPM sampling
- Tracks log probabilities for policy gradient

**Reward (Critic):**
- Evaluates the quality of generated samples
- Provides reward signal = negative log-likelihood

The training alternates:
1. **Model 1 generates x1 given x2, Model 2 provides reward**
   - Model 1 is updated via policy gradient (PPO with KL regularization)
   - Model 2 is synchronized via supervised learning on the new samples

2. **Model 2 generates x2 given x1, Model 1 provides reward**
   - Model 2 is updated via policy gradient (PPO with KL regularization)
   - Model 1 is synchronized via supervised learning on the new samples

## Key Components

### 1. Frozen Priors
```python
self.frozen_model_1 = deepcopy(self.ddpm_1.model)
self.frozen_model_2 = deepcopy(self.ddpm_2.model)
```
These serve as reference marginals to enforce KL regularization, preventing models from deviating too far from original distributions.

### 2. Trajectory Sampling with Log Probabilities
```python
trajectory, log_probs, timesteps = sample_trajectory_with_logprob(
    model, ddpm, condition, num_steps=50
)
```
Generates samples while tracking log probabilities needed for policy gradient.

### 3. Reward Computation
```python
rewards = compute_reward(x_gen, condition, reward_model, reward_ddpm)
```
Uses the partner model to evaluate likelihood, providing the "minimum entropy" signal.

### 4. Policy Gradient Update (DPOK)
```python
policy_gradient_update(
    trajectory, log_probs, rewards,
    gen_model, frozen_model, condition, timesteps,
    optimizer, kl_weight=0.1, clip_range=0.2
)
```
PPO-style update with:
- **Advantages:** Normalized rewards for variance reduction
- **KL penalty:** Prevents divergence from frozen prior
- **Clipping:** PPO clipping for stable training

### 5. Model Synchronization
```python
sync_reward_model(x_gen, condition, reward_model, optimizer, num_updates=5)
```
After RL update, reward model learns the new coupling via supervised training.

## Usage

### Basic Usage

```python
from ddmec_1d import DDMEC1D

# Initialize DDMEC with two pre-trained models
ddmec = DDMEC1D(
    model_path_1="ddpm_1d_2.pt",   # Model for N(2, 1)
    model_path_2="ddpm_1d_10.pt",  # Model for N(10, 1)
)

# Create datasets (should be coupled in reality)
x1_data = torch.randn(10000, 1) + 2.0   # Distribution 1
x2_data = torch.randn(10000, 1) + 10.0  # Distribution 2

# Train DDMEC
ddmec.train(
    dataset_1=x1_data,
    dataset_2=x2_data,
    num_epochs=50,
    batch_size=64,
    lr=1e-4,
)

# Sample from learned coupling
x2_test = torch.tensor([[10.0]])
x1_generated = ddmec.sample_coupled(x2_test, direction="1->2")
print(f"Given x2={x2_test.item():.2f}, generated x1={x1_generated.item():.2f}")
```

### Running the Example

```bash
python ddmec_1d.py
```

This will:
1. Load the two pre-trained models
2. Train DDMEC for 50 epochs
3. Generate visualizations showing:
   - Learned coupling x2 → x1
   - Learned coupling x1 → x2
   - Marginal distributions
4. Save results to `ddmec_coupling_results.png`
5. Save trained models to `ddmec_trained.pt`

## Expected Behavior

For your two Gaussian models:
- **Model 1**: Trained on N(2, 1)
- **Model 2**: Trained on N(10, 1)

After DDMEC training:
- When conditioned on x2 ≈ 10, model should generate x1 ≈ 2
- When conditioned on x1 ≈ 2, model should generate x2 ≈ 10
- The coupling should preserve the marginal distributions

## Key Parameters

### Training Parameters
- `warmup_steps`: Number of supervised training steps before cooperative phase (default: 1000)
- `num_epochs`: Total training epochs (default: 50)
- `batch_size`: Batch size for training (default: 64)
- `lr`: Learning rate (default: 1e-4)

### DDMEC Hyperparameters
- `kl_weight`: Weight for KL regularization (default: 0.1)
  - Higher values keep models closer to original distributions
  - Lower values allow more flexibility in learning coupling

- `clip_range`: PPO clipping range (default: 0.2)
  - Controls how much policy can change per update
  - Standard PPO value

- `mc_steps`: Monte Carlo steps for reward estimation (default: 5)
  - More steps = better reward estimate but slower
  
- `num_updates`: Sync updates for reward model (default: 5)
  - Number of gradient steps for synchronization

## Architecture Compatibility

The code works with your existing `SimpleMLP` models:
- Input: concatenation of `[x, condition]`
- Time embedding: sinusoidal encoding
- Network: 3-layer MLP with ReLU activations

## Differences from Reference Implementation

This 1D implementation is adapted from the full DDMEC paper but simplified:

1. **No ControlNet**: Uses simple conditioning via concatenation
2. **Simpler sampling**: DDIM-style sampling instead of complex guidance
3. **Direct log prob computation**: Simplified likelihood calculations
4. **No distributed training**: Single-GPU implementation
5. **1D specific**: Optimized for scalar data rather than images

## Debugging Tips

If training is unstable:
1. **Increase `warmup_steps`** - ensure good initialization
2. **Reduce `lr`** - slower but more stable training
3. **Increase `kl_weight`** - stronger regularization
4. **Check frozen models** - ensure they're properly frozen after warmup

Monitor these metrics:
- `reward_mean`: Should be stable and not collapse
- `kl_reg`: Should be non-zero but not too large
- `pg_loss`: Policy gradient loss, should decrease
- `advantage_mean`: Should be near zero after normalization

## Theoretical Background

DDMEC minimizes:
```
H(X, Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)
```

Where:
- `H(X)`, `H(Y)` are marginal entropies (fixed by frozen priors)
- `H(Y|X)`, `H(X|Y)` are conditional entropies (minimized by RL)

The minimum entropy coupling satisfies:
```
p*(x, y) ∝ exp(-c(x, y))
```

For optimal transport cost `c(x, y)`, which DDMEC approximates through cooperative training.

## References

1. **DDMEC Paper**: "Minimum Entropy Coupling with Diffusion Models"
2. **DPOK**: "Denoising Diffusion Policy Optimization" (policy gradient for diffusion)
3. **PPO**: "Proximal Policy Optimization" (stable RL algorithm)

## File Structure

```
ddmec_1d.py              # Main DDMEC implementation
train_ddpm_1d.py         # DDPM training code
sample_and_plot.py       # Sampling and visualization
ddpm_1d_2.pt            # Pre-trained model 1 (N(2,1))
ddpm_1d_10.pt           # Pre-trained model 2 (N(10,1))
DDMEC_README.md         # This file
```

## Next Steps

1. **Run the example**: `python ddmec_1d.py`
2. **Visualize results**: Check `ddmec_coupling_results.png`
3. **Experiment with hyperparameters**: Adjust `kl_weight`, `clip_range`, etc.
4. **Try different distributions**: Modify the data generation
5. **Extend to higher dimensions**: Adapt for 2D or image data

Good luck with your DDMEC experiments! 🚀

