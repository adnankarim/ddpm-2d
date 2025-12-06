# ES-DDMEC: Evolution Strategies for Denoising Diffusion Minimum Entropy Coupling

This directory contains the Evolution Strategies (ES) implementation for DDMEC, a robust alternative to the PPO-based reinforcement learning approach.

## Overview

ES-DDMEC replaces the policy gradient updates in DDMEC with Evolution Strategies, offering:
- **Robustness**: Less sensitive to hyperparameters than RL
- **Sample Efficiency**: Achieves similar or better performance with fewer evaluations
- **Stability**: More stable training without variance issues from policy gradients
- **Simplicity**: No complex RL machinery (no advantage estimation, value functions, or clipping)

## Key Files

### Core Implementation
- **`es_ddmec.py`**: Main ES-DDMEC implementation
  - `ESConfig`: Configuration for ES hyperparameters
  - `ESOptimizer`: Layer-wise ES optimizer with memory-efficient perturbation/restoration
  - `ESDDMEC1D`: ES-based DDMEC trainer

### Scripts
- **`run_es_ddmec.py`**: Main training script with comprehensive logging
- **`run_es_ablation.py`**: Ablation study script for ES hyperparameters

## Algorithm Overview

### ES Update Formula

For each model parameter θ, ES performs the following update:

```
θ ← θ + α/(Nσ) * Σᵢ₌₁ᴺ Rᵢ * εᵢ
```

Where:
- `α`: Learning rate (ES-LR)
- `N`: Population size
- `σ`: Noise scale (sigma)
- `Rᵢ`: Normalized reward for perturbation i
- `εᵢ`: Gaussian noise vector for perturbation i

### Training Phases

1. **Warmup Phase (Supervised Learning)**
   - Models learn unconditional generation from marginal distributions
   - Standard denoising diffusion training
   - No coupling constraints yet

2. **Cooperative Phase (ES-based)**
   - Models learn conditional coupling via ES
   - Generator perturbed and evaluated with reward function
   - Reward model synchronized via supervised learning
   - Bidirectional coupling: X1 ↔ X2

## Running ES-DDMEC

### Quick Start

```bash
# Train with default ES hyperparameters
python run_es_ddmec.py

# Custom hyperparameters
python run_es_ddmec.py \
    --population-size 30 \
    --sigma 0.001 \
    --es-lr 5e-4 \
    --warmup-epochs 10 \
    --num-epochs 50 \
    --batch-size 64 \
    --num-samples 100000 \
    --mc-steps 5 \
    --sampling-steps 50 \
    --run-name "my_es_experiment"
```

### ES Hyperparameters

**Population Size** (`--population-size`): Number of perturbations per ES update
- Smaller (10-20): Faster but noisier
- Larger (40-50): Slower but more stable
- **Recommended**: 30

**Sigma** (`--sigma`): Noise scale for parameter perturbation
- Too small (<0.0005): Slow exploration
- Too large (>0.003): Unstable
- **Recommended**: 0.001

**ES Learning Rate** (`--es-lr`): Step size for parameter updates
- Too small (<1e-4): Slow convergence
- Too large (>1e-3): Unstable
- **Recommended**: 5e-4

**Optional Techniques**:
- `--use-mirror-sampling`: Use antithetic sampling (reduces variance)
- `--use-rank-transform`: Use rank-based fitness shaping (improves robustness)

### Training Options

- `--warmup-epochs`: Supervised pretraining epochs (default: 10)
- `--num-epochs`: Total training epochs (default: 50)
- `--batch-size`: Batch size (default: 64)
- `--num-samples`: Training dataset size (default: 100000)
- `--mc-steps`: Monte Carlo steps for reward estimation (default: 5)
- `--sampling-steps`: Diffusion sampling steps (default: 50)

### Logging Options

- `--use-wandb` / `--no-wandb`: Enable/disable Weights & Biases logging
- `--use-tensorboard` / `--no-tensorboard`: Enable/disable TensorBoard logging
- `--run-name`: Custom experiment name

## Output Structure

Each training run creates a directory: `runs/<run_name>_<timestamp>/`

```
runs/my_es_experiment_20251204_003108/
├── checkpoints/                    # Model checkpoints
│   ├── epoch_0001.pt
│   ├── epoch_0002.pt
│   └── ...
├── plots/                          # Per-epoch visualizations
│   ├── epoch_0001/
│   │   ├── coupling_plot.png       # Main 2x3 grid plot
│   │   ├── coupling_x2_to_x1.png   # Forward coupling scatter
│   │   ├── coupling_x1_to_x2.png   # Backward coupling scatter
│   │   ├── marginals.png           # Marginal distributions
│   │   ├── error_distributions.png # Coupling errors
│   │   ├── joint_distributions.png # 2D heatmaps
│   │   ├── information_metrics.png # Info-theoretic metrics bar chart
│   │   └── metrics_grid.png        # 3x4 comprehensive metrics grid
│   ├── epoch_0002/
│   │   └── ...
│   └── ...
├── tables/                         # CSV logs
│   └── epoch_metrics.csv
├── logs/                           # JSON logs
│   └── training_log.json
├── tensorboard/                    # TensorBoard logs (if enabled)
│   └── events.out.tfevents...
└── final_results.png               # Final coupling visualization
```

### Detailed Plots (Per Epoch)

Each epoch directory contains:

1. **`coupling_plot.png`**: Main 2×3 grid with coupling, marginals, and errors
2. **`coupling_x2_to_x1.png`**: Forward coupling scatter with ideal line
3. **`coupling_x1_to_x2.png`**: Backward coupling scatter with ideal line
4. **`marginals.png`**: Marginal distributions vs. target N(μ, 1)
5. **`error_distributions.png`**: Coupling error histograms
6. **`joint_distributions.png`**: 2D joint distribution heatmaps
7. **`information_metrics.png`**: Information-theoretic metrics bar chart
8. **`metrics_grid.png`**: 3×4 comprehensive metrics over all epochs

## Ablation Studies

Run systematic ablation studies over ES hyperparameters:

```bash
# Run all ablations
python run_es_ablation.py --ablation all --num-epochs 30

# Run specific ablation
python run_es_ablation.py --ablation population_size --num-epochs 30
python run_es_ablation.py --ablation sigma --num-epochs 30
python run_es_ablation.py --ablation es_lr --num-epochs 30
python run_es_ablation.py --ablation mirror_sampling --num-epochs 30
python run_es_ablation.py --ablation rank_transform --num-epochs 30
```

### Ablation Configurations

1. **Population Size**: [10, 20, 30, 40, 50]
2. **Sigma**: [0.0005, 0.001, 0.0015, 0.002, 0.003]
3. **ES Learning Rate**: [1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3]
4. **Mirror Sampling**: [False, True]
5. **Rank Transform**: [False, True]

### Ablation Outputs

Each ablation study creates a report directory: `runs/es_ablation_report_<type>_<timestamp>/`

```
runs/es_ablation_report_population_size_20251204_120000/
├── ablation_comparison.png      # Comparison plots for all metrics
├── ablation_results.json        # Detailed results in JSON
└── ablation_summary.txt         # Human-readable summary table
```

After all ablations complete, a consolidated report is generated:

```
runs/ALL_ES_ABLATION_RESULTS.txt  # All ablation results concatenated
```

This file contains:
- All individual ablation summaries
- Summary tables for each ablation
- Best configurations identified

### Ablation Options

```bash
python run_es_ablation.py \
    --ablation all \              # Type: all, population_size, sigma, es_lr, etc.
    --num-epochs 30 \             # Epochs per experiment
    --batch-size 64 \             # Batch size
    --num-samples 100000 \        # Dataset size
    --no-wandb \                  # Disable W&B
    --force \                     # Force re-run even if exists
    --min-epochs 10               # Min epochs to consider complete
```

## Key Features

### 1. Memory-Efficient Layer-wise Perturbation

Instead of storing full noise tensors for all parameters, we:
- Generate noise on-the-fly using seeded RNGs
- Perturb and restore parameters layer by layer
- Store only random seeds (negligible memory)

### 2. Reward Function

Reward = -NLL (Negative Log-Likelihood under reward model)

Higher reward → Generator produces samples more likely under reward model

### 3. Reward Normalization

Two options:
- **Z-score normalization** (default): `(R - mean(R)) / std(R)`
- **Rank transformation**: Map rewards to [-0.5, 0.5] based on rank

### 4. Mirror Sampling (Antithetic Sampling)

For each noise vector ε, also evaluate -ε
- Reduces variance in gradient estimation
- Doubles effective population size with minimal cost

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir runs/<run_name>/tensorboard
```

Tracks:
- KL divergence (forward & backward)
- Mutual information
- Coupling correlation
- Coupling error (MAE)
- ES rewards

### Weights & Biases

If W&B is enabled, metrics are logged to your W&B project:
- Real-time metric plots
- Hyperparameter tracking
- Experiment comparison

### JSON Logs

`logs/training_log.json` contains full training history:
```json
{
  "experiment": "ES-DDMEC",
  "config": { ... },
  "epochs": [
    {
      "epoch": 1,
      "phase": "warmup",
      "info_metrics_forward": { ... },
      "info_metrics_backward": { ... },
      "coupling_metrics": { ... },
      "es_metrics": { ... }
    },
    ...
  ]
}
```

## Evaluation Metrics

### Information-Theoretic Metrics

- **KL Divergence**: `KL(learned || target)` for each marginal
  - Measures how well learned marginals match target distributions
  - Lower is better (0 = perfect match)

- **Mutual Information**: `I(X; Y) = H(X) + H(Y) - H(X,Y)`
  - Measures statistical dependence between X and Y
  - Higher means stronger coupling

- **Entropies**: `H(X)`, `H(Y)`, `H(X,Y)`, `H(X|Y)`, `H(Y|X)`
  - Characterize uncertainty in distributions

### Coupling Metrics

- **Correlation**: Pearson correlation between generated and condition
  - Perfect coupling → correlation = 1.0
  - Measures linear relationship strength

- **MAE (Mean Absolute Error)**: `|x₁ - (x₂ - 8)|`
  - Measures coupling accuracy
  - Lower is better (0 = perfect coupling)

### ES-Specific Metrics

- **ES Reward**: Mean reward across population
  - Tracks generator improvement
  - Higher is better

## Comparison: ES vs PPO-DDMEC

| Aspect | ES-DDMEC | PPO-DDMEC |
|--------|----------|-----------|
| **Optimization** | Zeroth-order (parameter space) | First-order (gradient-based) |
| **Stability** | More stable | Can be unstable |
| **Hyperparameters** | Less sensitive | More sensitive |
| **Sample Efficiency** | Comparable or better | Good with tuning |
| **Implementation** | Simpler | More complex |
| **Variance** | Lower (especially with mirror sampling) | Higher (needs PPO tricks) |
| **Memory** | Layer-wise (efficient) | Full gradients |

## Troubleshooting

### Training instabilities

1. **Reward exploding/collapsing**
   - Reduce sigma (try 0.0005)
   - Reduce ES-LR (try 1e-4)
   - Enable rank transform: `--use-rank-transform`

2. **Slow convergence**
   - Increase population size (try 40-50)
   - Increase ES-LR (try 7.5e-4)
   - Increase warmup epochs

3. **Poor coupling quality**
   - Increase warmup epochs (try 15-20)
   - Increase MC steps (try 10)
   - Increase sampling steps (try 100)

### Out of memory

- Reduce population size
- Reduce batch size
- Reduce num_samples

## Citation

If you use ES-DDMEC in your research, please cite:

```bibtex
@article{ddmec2024,
  title={Minimum Entropy Coupling via Denoising Diffusion Models},
  author={...},
  year={2024}
}

@article{salimans2017evolution,
  title={Evolution strategies as a scalable alternative to reinforcement learning},
  author={Salimans, Tim and Ho, Jonathan and Chen, Xi and Sidor, Szymon and Sutskever, Ilya},
  journal={arXiv preprint arXiv:1703.03864},
  year={2017}
}
```

## License

See LICENSE file in the repository root.

## Contact

For questions or issues, please open a GitHub issue.

