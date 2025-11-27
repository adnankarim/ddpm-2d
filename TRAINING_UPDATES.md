# DDMEC Training Updates

## New Features Added ✅

### 1. **Trainable Parameters Logging**
The system now logs:
- DDPM Model 1 trainable/total parameters
- DDPM Model 2 trainable/total parameters  
- DDMEC overall trainable/total parameters
- Logged to console and W&B config

### 2. **Per-Epoch Visualization**
- **Generates plot every single epoch** (not just every 10)
- Plots saved to: `runs/{run_name}/plots/epoch_XXXX.png`
- Each plot contains:
  - Coupling scatter plots (x2→x1 and x1→x2)
  - Correlation strength
  - Marginal distributions with stats
  - Coupling errors (MAE)

### 3. **Comprehensive Checkpointing**
Every epoch saves:
- Model states (both DDPM models)
- Optimizer states
- Training config
- Current step counter
- Information-theoretic metrics
- Saved to: `runs/{run_name}/checkpoints/epoch_XXXX.pt`

### 4. **Best Model Tracking**
- Tracks best model based on **average KL divergence total**
- Automatically saves best model to: `runs/{run_name}/checkpoints/best_model.pt`
- Logs best epoch and best KL total to W&B
- Shows ★ indicator when new best is found

### 5. **Organized Run Structure**
Each run creates a timestamped directory:
```
runs/
└── ddmec_20251127_023045/
    ├── checkpoints/
    │   ├── epoch_0001.pt
    │   ├── epoch_0002.pt
    │   ├── ...
    │   ├── epoch_0050.pt
    │   └── best_model.pt
    └── plots/
        ├── epoch_0001.png
        ├── epoch_0002.png
        ├── ...
        └── epoch_0050.png
```

### 6. **Enhanced W&B Logging**
Every epoch logs:
- All training metrics
- Information-theoretic metrics (both directions)
- Learned distribution statistics
- **Visualization plots as images**
- Best model tracking
- Average KL divergence

## Usage

### Basic Usage (No W&B)
```bash
python run_ddmec.py --no-wandb
```

### With W&B
```bash
python run_ddmec.py
```

### With Custom Run Name
```bash
python run_ddmec.py --wandb-name "my_experiment"
```

### With Custom Project
```bash
python run_ddmec.py --wandb-project "my-project" --wandb-name "exp1"
```

## Output Example

```
============================================================
DDMEC: Minimum Entropy Coupling for 1D Gaussians
============================================================

Run directory: runs/ddmec_20251127_023045

✓ Weights & Biases initialized: ddmec_20251127_023045
  Dashboard: https://wandb.ai/...

============================================================
Model Architecture
============================================================

DDPM Model 1 (N(2,1)):
  Trainable parameters: 8,321
  Total parameters:     8,321

DDPM Model 2 (N(10,1)):
  Trainable parameters: 8,321
  Total parameters:     8,321

DDMEC Overall:
  Trainable parameters: 16,642
  Total parameters:     16,642
============================================================

Creating coupled training data...
X1: mean=2.00, std=1.00
X2: mean=10.00, std=1.00
Correlation: 1.000

Training DDMEC...
Phase 1: Warmup (supervised) - first 1000 steps
Phase 2: Cooperative (RL-based) - remaining steps
------------------------------------------------------------
Training DDMEC for 50 epochs, 1562 batches per epoch
Saving checkpoints to: runs/ddmec_20251127_023045/checkpoints
Saving plots to: runs/ddmec_20251127_023045/plots

Epoch 1/50
  Training Metrics:
    loss_1: 0.1234
    loss_2: 0.1456
    phase: warmup

  Information-Theoretic Metrics (x2→x1):
    KL Divergence (X1): 0.0123
    KL Divergence (X2): 0.0045
    KL Divergence (Total): 0.0168
    Mutual Information I(X;Y): 1.2345

  Information-Theoretic Metrics (x1→x2):
    KL Divergence (X1): 0.0145
    KL Divergence (X2): 0.0034
    KL Divergence (Total): 0.0179
    Mutual Information I(X;Y): 1.2567

  Plot saved: runs/ddmec_20251127_023045/plots/epoch_0001.png
  ★ New best model! (KL Total: 0.0174)

Epoch 2/50
  ...
```

## Key Improvements

### Before ❌
- No parameter logging
- Plots only every 10 epochs
- No per-epoch checkpoints
- No best model tracking
- No organized directory structure
- Plots not logged to W&B

### After ✅
- Complete parameter logging
- **Plot generated every epoch**
- **Checkpoint saved every epoch**
- **Automatic best model tracking**
- Organized run directories with timestamps
- **All plots logged to W&B as images**
- Easy to compare runs
- Easy to resume from any epoch
- Easy to identify best performing epoch

## Benefits

1. **Better Monitoring**: See progress every single epoch, not just every 10
2. **Better Debugging**: If something goes wrong, you have checkpoints and plots for every epoch
3. **Better Analysis**: Can load any epoch's checkpoint and analyze
4. **Better Comparison**: Each run is self-contained with all artifacts
5. **Better Reproducibility**: Full state saved including optimizers
6. **Better W&B Integration**: Visualizations automatically uploaded

## File Structure

### Modified Files
- ✅ `run_ddmec.py` - Added parameter counting, run directory setup
- ✅ `ddmec_1d.py` - Added plotting function, checkpointing, best model tracking

### New Directories Created (per run)
- `runs/{run_name}/checkpoints/` - Model checkpoints
- `runs/{run_name}/plots/` - Visualization plots

## W&B Dashboard Features

Now your W&B dashboard shows:
- **Plots tab**: All epoch visualizations as images
- **Metrics tab**: All training and info metrics
- **System tab**: Parameter counts
- **Config tab**: All hyperparameters and run info
- Easy comparison between runs

## Notes

- Training still focuses on DDMEC (not DDPM pretraining)
- Best model selection based on KL divergence (lower is better)
- Plots are lightweight (100 DPI) for faster generation
- All epochs logged, but you can change frequency if needed
- Run directories are timestamped to avoid conflicts

---

**All updates are production-ready and tested!** 🚀

