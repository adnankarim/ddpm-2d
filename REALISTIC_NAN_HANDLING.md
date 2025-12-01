# Realistic NaN Handling - Removed Band-Aids

## Overview
Removed all "band-aid" solutions that masked NaN/numerical issues with clamping or fallback values. Replaced with **proper error handling** that fails fast with diagnostic information.

## Changes Made

### 1. ✅ `sample_trajectory_with_logprob()` - Input Validation

**Before:**
```python
# Ensure initial noise is not NaN
if torch.isnan(x).any():
    print("Warning: NaN in initial noise, regenerating...")
    x = torch.randn(batch_size, 1, device=self.device)

# Ensure condition is not NaN
if torch.isnan(condition).any():
    print("Warning: NaN in condition input")
    condition = torch.nan_to_num(condition, nan=0.0)
```

**After:**
```python
# Validate inputs - fail fast if data is corrupt
if not torch.isfinite(x).all():
    raise ValueError(
        f"Initial noise contains NaN/Inf. This should never happen.\n"
        f"Check random seed/GPU state. Device: {self.device}"
    )

if not torch.isfinite(condition).all():
    raise ValueError(
        f"Condition input contains NaN/Inf. Check your data preprocessing:\n"
        f"  Mean: {condition[torch.isfinite(condition)].mean()}\n"
        f"  NaN count: {torch.isnan(condition).sum().item()}\n"
        f"  Inf count: {torch.isinf(condition).sum().item()}"
    )
```

**Why:** Input corruption should never happen. If it does, fail immediately with diagnostics instead of masking the problem.

---

### 2. ✅ `sample_trajectory_with_logprob()` - Forward Pass Checking

**Before:**
```python
# Predict noise
predicted_noise = model(x, t_tensor, condition)

# (no checking)
```

**After:**
```python
# Predict noise
predicted_noise = model(x, t_tensor, condition)

# Diagnostic check - fail fast on numerical issues
if not torch.isfinite(predicted_noise).all():
    print(f"\n{'='*70}")
    print(f"NUMERICAL INSTABILITY DETECTED at timestep {t_cur}")
    print(f"Input x: mean={x.mean():.6f}, std={x.std():.6f}")
    print(f"Predicted noise: NaN={nan_count}, Inf={inf_count}")
    print(f"Model weights: NaN={model_nan_count}")
    print(f"\nPossible causes:")
    print(f"  1. Learning rate too high")
    print(f"  2. Gradient explosion")
    print(f"  3. Poor weight initialization")
    raise RuntimeError("NaN/Inf detected in forward pass.")
```

**Why:** Model producing NaN means training has gone wrong. Stop immediately and tell user what to fix.

---

### 3. ✅ Removed x0_pred Clamping

**Before:**
```python
x0_pred = (x - sqrt_one_minus_alphas_cumprod_t * predicted_noise) / sqrt_alphas_cumprod_t

# Clamp x0_pred to reasonable range to prevent extreme values
x0_pred = torch.clamp(x0_pred, -50.0, 50.0)
```

**After:**
```python
x0_pred = (x - sqrt_one_minus_alphas_cumprod_t * predicted_noise) / sqrt_alphas_cumprod_t

# (no clamping - let natural values flow through)
```

**Why:** Clamping hides divergence. If values exceed reasonable bounds, the model is broken and should be retrained.

---

### 4. ✅ Added Divergence Detection

**Before:**
```python
x_next = sqrt_alphas_cumprod_next * x0_pred + ...

# (no checking)
```

**After:**
```python
x_next = sqrt_alphas_cumprod_next * x0_pred + ...

# Check for divergence BEFORE it propagates
if not torch.isfinite(x_next).all():
    print(f"SAMPLING DIVERGED at timestep {t_cur}")
    print(f"x_next: NaN={...}, range=[...]")
    print(f"The model has become unstable. Training should be restarted.")
    raise RuntimeError("Sampling produced NaN/Inf. Model is unstable.")

# Sanity check for extreme values
if x_next.abs().max() > 1e4:
    print(f"WARNING: Extreme values at timestep {t_cur}")
    print(f"  x_next max magnitude: {x_next.abs().max():.2e}")
```

**Why:** Detect divergence early. Give clear warning about what's happening.

---

### 5. ✅ Fixed `compute_information_metrics()` - Removed Fallbacks

**Before:**
```python
# Check for NaN or Inf in computed statistics
if not (np.isfinite(mu_1) and ...):
    # Return safe fallback values
    return {
        "kl_div_1": 0.0,
        "kl_div_2": 0.0,
        "mutual_information": 0.0,
        # ... all zeros ...
    }

# Mutual information (clamp to non-negative)
mi = max(h_x + h_y - h_xy, 0.0)

# Conditional entropies (ensure they are non-negative)
h_x_given_y = max(h_xy - h_y, 0.0)
```

**After:**
```python
# Check input validity first
if not np.isfinite(s1_np).all():
    raise ValueError(
        f"Sample 1 contains NaN/Inf values.\n"
        f"The model has diverged. Cannot compute metrics."
    )

# Compute statistics without fallbacks
mu_1 = float(s1_np.mean())
sigma_1 = float(s1_np.std())

# Only add epsilon if variance is legitimately zero
if sigma_1 < eps:
    print(f"WARNING: Sample 1 has near-zero variance. All samples identical.")
    sigma_1 = eps

# Compute MI without clamping - warn if negative
mi = h_x + h_y - h_xy
if mi < -eps:
    print(f"WARNING: Negative mutual information ({mi:.6f}). Numerical instability.")
```

**Why:** Returning zeros masks the problem. Fail properly and tell user model has diverged.

---

### 6. ✅ Fixed `compute_kl_divergence()` - Removed Clamping

**Before:**
```python
# Add epsilon to avoid numerical issues
eps = 1e-8
sigma_p = max(sigma_p, eps)
sigma_q = max(sigma_q, eps)

# Clamp the log ratio to avoid overflow/underflow
log_ratio = np.log(np.clip(var_q / var_p, 1e-10, 1e10))
```

**After:**
```python
eps = 1e-10  # Only for preventing log(0)

# Ensure positive variance
if sigma_p <= 0 or sigma_q <= 0:
    raise ValueError(f"Standard deviations must be positive: sigma_p={sigma_p}, sigma_q={sigma_q}")

# Compute log ratio naturally
log_ratio = np.log(var_q / var_p)

# Warn if KL is negative (impossible theoretically)
if kl < -eps:
    print(f"WARNING: Negative KL divergence ({kl:.6f}). Numerical issue.")
```

**Why:** Negative std is impossible. If it happens, fail with clear error. No silent clamping.

---

### 7. ✅ Fixed `compute_entropy_gaussian()` - Validation

**Before:**
```python
# Add epsilon to avoid log(0)
eps = 1e-8
sigma = max(sigma, eps)
variance = sigma ** 2
return 0.5 * np.log(2 * np.pi * np.e * variance)
```

**After:**
```python
eps = 1e-10  # Only for preventing log(0)

if sigma <= 0:
    raise ValueError(f"Standard deviation must be positive: sigma={sigma}")

variance = sigma ** 2
return 0.5 * np.log(2 * np.pi * np.e * variance)
```

**Why:** Zero/negative std is a bug. Fail with clear error message.

---

### 8. ✅ Fixed `compute_joint_entropy()` - Better Error Handling

**Before:**
```python
try:
    det_cov = np.linalg.det(cov)
    det_cov = max(det_cov, eps)
except np.linalg.LinAlgError:
    # If computation fails, use product of variances as fallback
    det_cov = cov[0, 0] * cov[1, 1]
    det_cov = max(det_cov, eps)
```

**After:**
```python
try:
    det_cov = np.linalg.det(cov_regularized)
    
    if det_cov <= 0:
        print(f"WARNING: Covariance matrix has non-positive determinant")
        print(f"  Covariance matrix:\n{cov}")
        print(f"  This may indicate degenerate distributions.")
        det_cov = abs(det_cov) + eps
        
except np.linalg.LinAlgError as e:
    raise RuntimeError(
        f"Failed to compute covariance determinant: {e}\n"
        f"This indicates serious numerical problems."
    )
```

**Why:** Singular covariance is a red flag. Print warning and only use fallback as last resort.

---

## Legitimate Numerical Stability (Kept)

These are **NOT band-aids** - they're mathematically sound:

### ✅ Epsilon in sqrt() to prevent sqrt(0)
```python
eps = 1e-10
sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod_t.clamp(min=eps))
```
**Why:** `sqrt(0)` is fine but its gradient is undefined. Small epsilon prevents NaN in backprop.

### ✅ Epsilon in variance for log-prob computation
```python
variance = beta_t.clamp(min=eps)
log_prob = -0.5 * ((x_next - x) ** 2).sum(dim=1) / variance
```
**Why:** Prevents division by zero. This is standard practice.

### ✅ PPO clipping
```python
clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
```
**Why:** This is part of the PPO algorithm, not a band-aid.

### ✅ Covariance regularization
```python
cov_regularized = cov + eps * np.eye(cov.shape[0])
```
**Why:** Finite sample covariances can be singular. Small regularization is standard.

---

## What This Achieves

### Before (Band-Aids):
- ❌ NaNs silently replaced with zeros or random values
- ❌ Extreme values clamped to arbitrary bounds
- ❌ Metrics return dummy values when model diverges
- ❌ Training continues with corrupted model
- ❌ User has no idea what's wrong

### After (Proper Handling):
- ✅ NaNs cause immediate failure with diagnostics
- ✅ Extreme values trigger warnings about divergence
- ✅ Metrics fail with clear error messages
- ✅ Training stops before model corruption spreads
- ✅ User knows exactly what went wrong and how to fix it

---

## Expected Behavior

### Good Training Run:
```
Starting with warmup LR: 1.00e-05
Epoch 1/50
  Training Metrics:
    loss_1: 0.1234
    loss_2: 0.1456
  Information-Theoretic Metrics:
    KL Divergence (Total): 0.0523
    Mutual Information: 0.9123
```

### Bad Training Run (Model Diverging):
```
Epoch 3/50
======================================================================
NUMERICAL INSTABILITY DETECTED at timestep 247
======================================================================
Input x:           mean=0.123456, std=1.234567
Predicted noise:   NaN=32, Inf=0
Model weights:     NaN=0, Inf=0

Possible causes:
  1. Learning rate too high (current step: 1523)
  2. Gradient explosion (check gradient norms)
  3. Poor weight initialization
  4. Numerical instability in schedule
======================================================================

RuntimeError: NaN/Inf detected in forward pass. Training cannot continue safely.
```

### User Action:
1. Reduce learning rate
2. Add gradient clipping
3. Check weight initialization
4. Restart training from scratch

---

## Summary

All **masking/hiding** solutions removed. Replaced with **fail-fast diagnostics**.

The code now tells you **WHEN** something goes wrong, **WHERE** it happened, and **WHAT** to do about it.

No more silent failures. No more mysterious zeros in metrics. No more corrupted models.

**The model either trains correctly, or fails loudly with actionable information.**

---

**Status:** ✅ COMPLETE
**Date:** 2025-11-28

