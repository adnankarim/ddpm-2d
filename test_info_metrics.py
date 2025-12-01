"""
Quick test to verify information-theoretic metrics computation.
"""

import torch
import numpy as np
from ddmec_1d import DDMEC1D


def test_metrics():
    """Test information-theoretic metrics with known distributions."""
    print("="*60)
    print("Testing Information-Theoretic Metrics")
    print("="*60)
    
    # Initialize DDMEC
    ddmec = DDMEC1D(
        model_path_1="ddpm_1d_2.pt",
        model_path_2="ddpm_1d_10.pt",
    )
    
    # Test 1: Independent distributions (should have MI ≈ 0)
    print("\nTest 1: Independent Distributions")
    print("-" * 60)
    x1_indep = torch.randn(1000, 1) + 2.0  # N(2, 1)
    x2_indep = torch.randn(1000, 1) + 10.0  # N(10, 1)
    
    metrics_indep = ddmec.compute_information_metrics(x1_indep, x2_indep)
    
    print(f"H(X₁): {metrics_indep['entropy_x']:.4f} nats")
    print(f"H(X₂): {metrics_indep['entropy_y']:.4f} nats")
    print(f"H(X₁,X₂): {metrics_indep['joint_entropy']:.4f} nats")
    print(f"I(X₁;X₂): {metrics_indep['mutual_information']:.4f} nats")
    print(f"Expected: I(X₁;X₂) ≈ 0 for independent")
    print(f"H(X₁|X₂): {metrics_indep['conditional_entropy_x_given_y']:.4f} nats")
    
    # Test 2: Perfect coupling (should have high MI)
    print("\nTest 2: Perfect Deterministic Coupling")
    print("-" * 60)
    noise = torch.randn(1000, 1)
    x1_coupled = noise + 2.0   # N(2, 1)
    x2_coupled = noise + 10.0  # N(10, 1), perfectly coupled to x1
    
    metrics_coupled = ddmec.compute_information_metrics(x1_coupled, x2_coupled)
    
    print(f"H(X₁): {metrics_coupled['entropy_x']:.4f} nats")
    print(f"H(X₂): {metrics_coupled['entropy_y']:.4f} nats")
    print(f"H(X₁,X₂): {metrics_coupled['joint_entropy']:.4f} nats")
    print(f"I(X₁;X₂): {metrics_coupled['mutual_information']:.4f} nats")
    print(f"Expected: I(X₁;X₂) ≈ H(X) ≈ 1.419 for perfect coupling")
    print(f"H(X₁|X₂): {metrics_coupled['conditional_entropy_x_given_y']:.4f} nats")
    print(f"Expected: H(X₁|X₂) ≈ 0 for deterministic coupling")
    
    # Test 3: KL divergence sanity check
    print("\nTest 3: KL Divergence")
    print("-" * 60)
    print(f"KL(learned||true) for X₁: {metrics_coupled['kl_div_1']:.4f}")
    print(f"KL(learned||true) for X₂: {metrics_coupled['kl_div_2']:.4f}")
    print(f"Expected: ≈ 0 since we're sampling from true distributions")
    
    # Theoretical reference
    print("\n" + "="*60)
    print("Theoretical Reference Values")
    print("="*60)
    sigma = 1.0
    h_gaussian = 0.5 * np.log(2 * np.pi * np.e * sigma**2)
    print(f"H(X) for N(μ, 1): {h_gaussian:.4f} nats")
    print(f"H(X,Y) for independent: {2 * h_gaussian:.4f} nats")
    print(f"H(X,Y) for perfect coupling: {h_gaussian:.4f} nats")
    print(f"I(X;Y) for independent: 0.0000 nats")
    print(f"I(X;Y) for perfect coupling: {h_gaussian:.4f} nats")
    
    print("\n✓ Metrics test complete!")


if __name__ == "__main__":
    test_metrics()
