# Weights & Biases Integration Guide

This guide explains how to use Weights & Biases (W&B) for tracking your DDMEC training.

## 🚀 Quick Start

### 1. Install wandb

```bash
pip install wandb
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

### 2. Login to W&B

First time only:
```bash
wandb login
```

You'll be prompted to paste your API key from https://wandb.ai/authorize

### 3. Run Training with W&B

```bash
# Basic usage (W&B enabled by default)
python run_ddmec.py

# Disable W&B
python run_ddmec.py --no-wandb

# Custom project name
python run_ddmec.py --wandb-project my-ddmec-project

# Custom run name
python run_ddmec.py --wandb-name "test-run-1"
```

---

## 📊 What Gets Logged

### **Training Metrics** (every 10 epochs)
- `train/loss_1`: Loss for model 1
- `train/loss_2`: Loss for model 2  
- `train/reward_1`: Reward for model 1
- `train/reward_2`: Reward for model 2
- `train/phase`: Training phase (warmup/cooperative)

### **Information-Theoretic Metrics** (every 10 epochs)

#### Forward Direction (x2 → x1):
- `info/kl_div_1_forward`: KL divergence for X₁
- `info/kl_div_2_forward`: KL divergence for X₂
- `info/kl_div_total_forward`: Total KL divergence
- `info/entropy_x_forward`: Entropy H(X₁)
- `info/entropy_y_forward`: Entropy H(X₂)
- `info/joint_entropy_forward`: Joint entropy H(X₁,X₂)
- `info/mutual_information_forward`: Mutual information I(X₁;X₂)
- `info/conditional_entropy_x_given_y`: Conditional entropy H(X₁|X₂)
- `info/conditional_entropy_y_given_x`: Conditional entropy H(X₂|X₁)

#### Backward Direction (x1 → x2):
- `info/kl_div_1_backward`: KL divergence for X₁
- `info/kl_div_2_backward`: KL divergence for X₂
- `info/kl_div_total_backward`: Total KL divergence
- `info/mutual_information_backward`: Mutual information I(X₁;X₂)

#### Learned Statistics:
- `stats/learned_mu_1_forward`: Learned mean for X₁
- `stats/learned_sigma_1_forward`: Learned std for X₁
- `stats/learned_mu_2_forward`: Learned mean for X₂
- `stats/learned_sigma_2_forward`: Learned std for X₂

### **Final Evaluation Metrics**
- `final/kl_div_1_forward`: Final KL divergence (forward)
- `final/mutual_information_forward`: Final MI (forward)
- `final/joint_entropy_forward`: Final joint entropy (forward)
- `final/conditional_entropy_x_given_y`: Final conditional entropy
- Plus backward direction metrics

### **Visualizations**
- `visualization`: Complete DDMEC results plot (9 subplots)

### **Model Artifacts**
- `ddmec-model`: Trained model checkpoint saved as W&B artifact

---

## 🎨 Dashboard Features

### What You'll See in W&B:

1. **Training Curves**
   - Real-time loss tracking for both models
   - Reward signals during RL phase
   - Phase transitions (warmup → cooperative)

2. **Information Theory Metrics**
   - KL divergence convergence (should → 0)
   - Mutual information growth (should → max ≈ 1.419)
   - Joint entropy minimization (should → min ≈ 1.419)
   - Conditional entropy reduction (should → 0)

3. **Quality Indicators**
   - Learned distribution statistics
   - Comparison to true N(2,1) and N(10,1)

4. **Final Results**
   - Comprehensive visualization with 9 subplots
   - Coupling quality assessment
   - Round-trip consistency metrics

---

## 📈 Interpreting the Dashboard

### **Good Training Signs:**
✅ **KL Divergence** curves trending toward 0
✅ **Mutual Information** increasing toward 1.419 nats
✅ **Joint Entropy** decreasing toward 1.419 nats
✅ **Conditional Entropy** decreasing toward 0
✅ Learned means: `μ₁ → 2.0`, `μ₂ → 10.0`
✅ Learned stds: `σ₁ → 1.0`, `σ₂ → 1.0`

### **Warning Signs:**
⚠️ KL divergence increasing or oscillating wildly
⚠️ Mutual information stuck near 0
⚠️ Joint entropy approaching 2.838 (independence)
⚠️ Conditional entropy staying high (≈ 1.419)

---

## 🔧 Configuration

### Training Config (automatically logged):
```python
{
    "architecture": "DDMEC",
    "dataset": "1D Gaussians",
    "dist_1": "N(2, 1)",
    "dist_2": "N(10, 1)",
    "num_train_samples": 10000,
    "batch_size": 64,
    "learning_rate": 1e-4,
    "num_epochs": 100,
    "warmup_epochs": 10,
    "num_timesteps": 1000,
}
```

### Custom Configuration:
You can modify the config in `run_ddmec.py`:
```python
wandb.init(
    project="my-project",
    name="experiment-name",
    config={
        # Your custom config here
        "batch_size": 128,
        "learning_rate": 5e-5,
        # ...
    }
)
```

---

## 🎯 Example Dashboard URL

After running:
```
✓ Weights & Biases initialized: ethereal-dawn-42
  Dashboard: https://wandb.ai/username/ddmec-1d/runs/abc123
```

Click the URL to view your live training dashboard!

---

## 💾 Model Artifacts

Your trained model is automatically saved as a W&B artifact:

### Download a trained model:
```python
import wandb

run = wandb.init(project="ddmec-1d")
artifact = run.use_artifact('username/ddmec-1d/ddmec-model:latest')
artifact_dir = artifact.download()

# Load the model
import torch
checkpoint = torch.load(f"{artifact_dir}/ddmec_trained.pt")
```

---

## 🔍 Comparing Runs

W&B makes it easy to compare multiple training runs:

1. Navigate to your project page
2. Select multiple runs
3. Compare:
   - Hyperparameters
   - Metric curves
   - Final performance
   - Visualizations

### Useful Comparisons:
- Different learning rates
- Different batch sizes  
- Different coupling data
- Warmup duration effects

---

## 🐛 Troubleshooting

### W&B Not Logging?

**Check installation:**
```bash
pip show wandb
```

**Check login:**
```bash
wandb login --relogin
```

**Disable if issues:**
```bash
python run_ddmec.py --no-wandb
```

### Slow Logging?

W&B logging is asynchronous and shouldn't slow training significantly. If you experience issues:
- Check internet connection
- Reduce logging frequency (modify `if (epoch + 1) % 10 == 0` in code)
- Use offline mode: `wandb.init(mode="offline")`

### API Key Issues?

Get your key from: https://wandb.ai/authorize

```bash
wandb login
# Paste your key when prompted
```

---

## 📚 Learn More

- **W&B Documentation**: https://docs.wandb.ai/
- **Guides**: https://docs.wandb.ai/guides
- **Examples**: https://github.com/wandb/examples

---

## 🎓 Best Practices

1. **Use descriptive run names:**
   ```bash
   python run_ddmec.py --wandb-name "high-lr-experiment"
   ```

2. **Group related experiments:**
   Use the same project name for related experiments

3. **Add notes to runs:**
   In the W&B UI, add notes describing what you're testing

4. **Tag important runs:**
   Mark successful runs with tags like "good-coupling", "production", etc.

5. **Compare systematically:**
   Change one hyperparameter at a time for clear comparisons

---

## 🎉 Happy Experimenting!

With W&B, you can:
- Track experiments automatically
- Compare different approaches
- Share results with collaborators
- Resume failed training
- Never lose a good model

Your DDMEC training is now fully instrumented! 🚀

