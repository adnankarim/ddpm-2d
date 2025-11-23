# Information-Theoretic Metrics in DDMEC

This document describes the information-theoretic metrics tracked during DDMEC training.

## Overview

DDMEC (Denoising Diffusion Minimum Entropy Coupling) learns the minimum entropy coupling between two distributions. To monitor training quality, we track several key information-theoretic quantities.

## Metrics Tracked

### 1. **KL Divergence (Kullback-Leibler Divergence)**

**Formula for 1D Gaussians:**
```
KL(p||q) = 1/2 * [(μ₁-μ₀)²/σ₁² + σ₀²/σ₁² - 1 + ln(σ₁²/σ₀²)]
```

**What it measures:** How different the learned distribution is from the true marginal distribution.

**Interpretation:**
- KL = 0: Perfect match
- KL > 0: Distributions differ
- Lower is better

**Tracked:**
- `kl_div_1`: KL divergence for distribution X₁ (learned vs. true N(2,1))
- `kl_div_2`: KL divergence for distribution X₂ (learned vs. true N(10,1))
- `kl_div_total`: Sum of both KL divergences

---

### 2. **Entropy H(X)**

**Formula for 1D Gaussian:**
```
H(X) = 1/2 * ln(2πeσ²)
```

**What it measures:** The uncertainty/randomness in a single distribution.

**Interpretation:**
- Higher entropy = more uncertainty
- For N(0,1): H(X) ≈ 1.419 nats
- Measured in nats (natural logarithm units)

**Tracked:**
- `entropy_x`: Entropy of X₁ distribution
- `entropy_y`: Entropy of X₂ distribution

**Theoretical values:**
- H(N(2,1)) = H(N(10,1)) ≈ 1.419 nats (variance = 1)

---

### 3. **Joint Entropy H(X,Y)**

**Formula for 2D Gaussian:**
```
H(X,Y) = 1/2 * ln((2πe)² * det(Σ))
```

where Σ is the covariance matrix:
```
Σ = [σₓ²    ρσₓσᵧ]
    [ρσₓσᵧ  σᵧ²  ]
```

**What it measures:** The total uncertainty in the joint distribution.

**Interpretation:**
- If X and Y are independent: H(X,Y) = H(X) + H(Y) ≈ 2.838 nats
- If X and Y are coupled: H(X,Y) < H(X) + H(Y)
- **Minimum entropy coupling minimizes H(X,Y)** while preserving marginals

**Tracked:**
- `joint_entropy`: H(X₁,X₂)

---

### 4. **Mutual Information I(X;Y)**

**Formula:**
```
I(X;Y) = H(X) + H(Y) - H(X,Y)
```

**What it measures:** How much information X and Y share; reduction in uncertainty about X when Y is known.

**Interpretation:**
- I(X;Y) = 0: X and Y are independent
- I(X;Y) = min(H(X), H(Y)): X and Y are perfectly dependent
- **Minimum entropy coupling maximizes I(X;Y)**
- Higher is better for optimal coupling

**Tracked:**
- `mutual_information`: I(X₁;X₂)

**Theoretical bounds:**
- Independent: I(X₁;X₂) = 0
- Perfect coupling: I(X₁;X₂) ≈ 1.419 nats (= H(X) when σ=1)

---

### 5. **Conditional Entropy H(X|Y)**

**Formula:**
```
H(X|Y) = H(X,Y) - H(Y)
```

**What it measures:** Remaining uncertainty in X after observing Y.

**Interpretation:**
- H(X|Y) = H(X): X and Y are independent
- H(X|Y) = 0: Y completely determines X
- Lower is better for strong coupling

**Tracked:**
- `conditional_entropy_x_given_y`: H(X₁|X₂)
- `conditional_entropy_y_given_x`: H(X₂|X₁)

**Theoretical bounds:**
- Independent: H(X|Y) = H(X) ≈ 1.419 nats
- Perfect coupling: H(X|Y) = 0 nats

---

## Training Output Example

Every 10 epochs, you'll see:

```
Epoch 50/100
  Training Metrics:
    loss_1: 0.2345
    loss_2: 0.2134
    ...

  Information-Theoretic Metrics (x2→x1):
    KL Divergence (X1): 0.0123
    KL Divergence (X2): 0.0098
    KL Divergence (Total): 0.0221
    Entropy H(X): 1.4156 nats
    Entropy H(Y): 1.4189 nats
    Joint Entropy H(X,Y): 1.8234 nats
    Mutual Information I(X;Y): 1.0111 nats
    Conditional Entropy H(X|Y): 0.4045 nats
    Conditional Entropy H(Y|X): 0.4078 nats

  Information-Theoretic Metrics (x1→x2):
    KL Divergence (X1): 0.0156
    KL Divergence (X2): 0.0187
    KL Divergence (Total): 0.0343
    Mutual Information I(X;Y): 0.9876 nats
```

---

## Quality Indicators

### Good Training Signs:
1. ✅ **KL Divergence** → 0 (both distributions match target marginals)
2. ✅ **Joint Entropy** → minimum (close to 1.419 for perfect coupling)
3. ✅ **Mutual Information** → maximum (close to 1.419 for σ=1)
4. ✅ **Conditional Entropy** → 0 (strong deterministic coupling)

### Poor Training Signs:
1. ❌ High KL divergence (> 0.5)
2. ❌ Joint entropy close to 2.838 (independent distributions)
3. ❌ Mutual information close to 0
4. ❌ Conditional entropy close to H(X) ≈ 1.419

---

## Mathematical Relationships

These metrics satisfy several key identities:

```
I(X;Y) = H(X) - H(X|Y)           # Mutual info from conditional
I(X;Y) = H(Y) - H(Y|X)           # Symmetric
I(X;Y) = H(X) + H(Y) - H(X,Y)   # From joint entropy
H(X|Y) + H(Y|X) = H(X,Y)        # Conditional entropies sum
```

---

## Conversion to Bits

Metrics are reported in **nats** (natural logarithm). To convert to **bits**:

```
bits = nats / ln(2) ≈ nats / 0.693
```

Example:
- H(X) = 1.419 nats ≈ 2.047 bits

---

## References

1. **Minimum Entropy Coupling**: The goal is to find the joint distribution p(x,y) that:
   - Minimizes H(X,Y)
   - Preserves marginals: p(x) = p₁(x), p(y) = p₂(y)
   - Maximizes mutual information I(X;Y)

2. **DDMEC Paper**: These metrics directly measure how well DDMEC achieves minimum entropy coupling.

3. **Expected Values** (for our setup):
   - True marginals: X₁ ~ N(2, 1), X₂ ~ N(10, 1)
   - Perfect coupling: x₁ = x₂ - 8 (deterministic)
   - Expected I(X;Y) ≈ 1.419 nats (maximal)
   - Expected H(X,Y) ≈ 1.419 nats (minimal)

