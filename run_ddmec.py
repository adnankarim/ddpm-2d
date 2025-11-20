"""
Simple script to run DDMEC on the two pre-trained 1D Gaussian models.

This script:
1. Loads the two models (N(2,1) and N(10,1))
2. Trains DDMEC to learn minimum entropy coupling
3. Evaluates and visualizes the learned coupling
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from ddmec_1d import DDMEC1D


def create_coupled_data(num_samples: int = 10000):
    """
    Create coupled data where x1 and x2 share common structure.
    
    For minimum entropy coupling, we want:
    - x1 ~ N(2, 1)
    - x2 ~ N(10, 1)
    - But coupled through shared latent variable
    """
    # Shared latent variable
    z = torch.randn(num_samples, 1)
    
    # Create coupling: x1 and x2 are offset versions of z
    x1 = z + 2.0
    x2 = z + 10.0
    
    # Verify marginals
    print(f"X1: mean={x1.mean():.2f}, std={x1.std():.2f}")
    print(f"X2: mean={x2.mean():.2f}, std={x2.std():.2f}")
    print(f"Correlation: {np.corrcoef(x1.numpy().flatten(), x2.numpy().flatten())[0, 1]:.3f}")
    
    return x1, x2


def evaluate_coupling(ddmec: DDMEC1D, num_samples: int = 500):
    """Evaluate the learned coupling quantitatively."""
    print("\n" + "="*60)
    print("Evaluating Learned Coupling")
    print("="*60)
    
    with torch.no_grad():
        # Test 1: Given x2, generate x1
        x2_test = torch.randn(num_samples, 1, device=ddmec.device) + 10.0
        x1_gen = ddmec.sample_coupled(x2_test, direction="1->2", num_steps=50)
        
        # Expected: x1_gen ≈ x2_test - 8 (since x1 ~ N(2,1) and x2 ~ N(10,1))
        diff_1 = (x1_gen - (x2_test - 8.0)).abs().mean().item()
        
        print(f"\nDirection x2 -> x1:")
        print(f"  Input x2: mean={x2_test.mean():.2f}, std={x2_test.std():.2f}")
        print(f"  Generated x1: mean={x1_gen.mean():.2f}, std={x1_gen.std():.2f}")
        print(f"  Expected x1: mean=2.00, std=1.00")
        print(f"  Mean absolute offset error: {diff_1:.3f}")
        
        # Test 2: Given x1, generate x2
        x1_test = torch.randn(num_samples, 1, device=ddmec.device) + 2.0
        x2_gen = ddmec.sample_coupled(x1_test, direction="2->1", num_steps=50)
        
        diff_2 = (x2_gen - (x1_test + 8.0)).abs().mean().item()
        
        print(f"\nDirection x1 -> x2:")
        print(f"  Input x1: mean={x1_test.mean():.2f}, std={x1_test.std():.2f}")
        print(f"  Generated x2: mean={x2_gen.mean():.2f}, std={x2_gen.std():.2f}")
        print(f"  Expected x2: mean=10.00, std=1.00")
        print(f"  Mean absolute offset error: {diff_2:.3f}")
        
        # Test 3: Check coupling consistency
        x2_back = ddmec.sample_coupled(x1_gen, direction="2->1", num_steps=50)
        reconstruction_error = (x2_back - x2_test).abs().mean().item()
        
        print(f"\nRound-trip consistency (x2 -> x1 -> x2):")
        print(f"  Reconstruction error: {reconstruction_error:.3f}")
        
        return {
            "x1_gen": x1_gen.cpu().numpy(),
            "x2_gen": x2_gen.cpu().numpy(),
            "x1_test": x1_test.cpu().numpy(),
            "x2_test": x2_test.cpu().numpy(),
        }


def visualize_results(results: dict, save_path: str = "ddmec_results.png"):
    """Create comprehensive visualization of DDMEC results."""
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Row 1: Coupling scatter plots
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(results["x2_test"], results["x1_gen"], alpha=0.3, s=10)
    ax1.plot([8, 12], [-6, -2], 'r--', label='Ideal coupling (slope=1)')
    ax1.set_xlabel("x2 (condition)")
    ax1.set_ylabel("x1 (generated)")
    ax1.set_title("Learned Coupling: x2 → x1")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(results["x1_test"], results["x2_gen"], alpha=0.3, s=10)
    ax2.plot([0, 4], [8, 12], 'r--', label='Ideal coupling (slope=1)')
    ax2.set_xlabel("x1 (condition)")
    ax2.set_ylabel("x2 (generated)")
    ax2.set_title("Learned Coupling: x1 → x2")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Joint distribution
    ax3 = fig.add_subplot(gs[0, 2])
    h = ax3.hist2d(results["x2_test"].flatten(), results["x1_gen"].flatten(), 
                   bins=30, cmap='Blues')
    ax3.set_xlabel("x2")
    ax3.set_ylabel("x1")
    ax3.set_title("Joint Distribution p(x1, x2)")
    plt.colorbar(h[3], ax=ax3, label='Density')
    
    # Row 2: Marginal distributions
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(results["x1_gen"], bins=40, density=True, alpha=0.7, label='Generated x1')
    ax4.axvline(2.0, color='red', linestyle='--', linewidth=2, label='True mean (2.0)')
    ax4.set_xlabel("x1")
    ax4.set_ylabel("Density")
    ax4.set_title("Marginal Distribution: x1")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(results["x2_gen"], bins=40, density=True, alpha=0.7, label='Generated x2', color='orange')
    ax5.axvline(10.0, color='red', linestyle='--', linewidth=2, label='True mean (10.0)')
    ax5.set_xlabel("x2")
    ax5.set_ylabel("Density")
    ax5.set_title("Marginal Distribution: x2")
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Compare both marginals
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(results["x1_gen"], bins=30, density=True, alpha=0.5, label='x1 (centered at 2)')
    ax6.hist(results["x2_gen"], bins=30, density=True, alpha=0.5, label='x2 (centered at 10)')
    ax6.set_xlabel("Value")
    ax6.set_ylabel("Density")
    ax6.set_title("Both Marginals")
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Row 3: Coupling quality metrics
    ax7 = fig.add_subplot(gs[2, 0])
    offset_1 = results["x1_gen"] - (results["x2_test"] - 8.0)
    ax7.hist(offset_1, bins=30, density=True, alpha=0.7)
    ax7.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect coupling')
    ax7.set_xlabel("x1_gen - (x2 - 8)")
    ax7.set_ylabel("Density")
    ax7.set_title("Coupling Error: x2 → x1")
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    ax8 = fig.add_subplot(gs[2, 1])
    offset_2 = results["x2_gen"] - (results["x1_test"] + 8.0)
    ax8.hist(offset_2, bins=30, density=True, alpha=0.7, color='orange')
    ax8.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect coupling')
    ax8.set_xlabel("x2_gen - (x1 + 8)")
    ax8.set_ylabel("Density")
    ax8.set_title("Coupling Error: x1 → x2")
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # Correlation analysis
    ax9 = fig.add_subplot(gs[2, 2])
    corr_forward = np.corrcoef(results["x2_test"].flatten(), results["x1_gen"].flatten())[0, 1]
    corr_backward = np.corrcoef(results["x1_test"].flatten(), results["x2_gen"].flatten())[0, 1]
    
    ax9.bar(['x2→x1', 'x1→x2'], [corr_forward, corr_backward], alpha=0.7)
    ax9.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Perfect correlation')
    ax9.set_ylabel("Correlation")
    ax9.set_title("Coupling Correlation")
    ax9.set_ylim([0, 1.1])
    ax9.legend()
    ax9.grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate([corr_forward, corr_backward]):
        ax9.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    plt.suptitle('DDMEC: Minimum Entropy Coupling Results', fontsize=16, fontweight='bold')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory and prevent hanging
    print(f"\nVisualization saved to {save_path}")


def main():
    print("="*60)
    print("DDMEC: Minimum Entropy Coupling for 1D Gaussians")
    print("="*60)
    
    # Check if models exist
    import os
    if not os.path.exists("ddpm_1d_2.pt") or not os.path.exists("ddpm_1d_10.pt"):
        print("\nERROR: Pre-trained models not found!")
        print("Please ensure ddpm_1d_2.pt and ddpm_1d_10.pt are in the current directory.")
        print("You can train them using train_ddpm_1d.py")
        return
    
    # Initialize DDMEC
    print("\nInitializing DDMEC...")
    ddmec = DDMEC1D(
        model_path_1="ddpm_1d_2.pt",
        model_path_2="ddpm_1d_10.pt",
    )
    
    # Create coupled training data
    print("\nCreating coupled training data...")
    num_train = 10000
    x1_train, x2_train = create_coupled_data(num_train)
    
    # Train DDMEC
    print("\nTraining DDMEC...")
    print("Phase 1: Warmup (supervised) - first 1000 steps")
    print("Phase 2: Cooperative (RL-based) - remaining steps")
    print("-" * 60)
    
    ddmec.train(
        dataset_1=x1_train,
        dataset_2=x2_train,
        num_epochs=50,
        batch_size=64,
        lr=1e-4,
    )
    
    # Evaluate
    results = evaluate_coupling(ddmec, num_samples=1000)
    
    # Visualize
    print("\nGenerating visualizations...")
    visualize_results(results)
    
    # Save trained model
    save_path = "ddmec_trained.pt"
    torch.save({
        "model_1_state_dict": ddmec.ddpm_1.model.state_dict(),
        "model_2_state_dict": ddmec.ddpm_2.model.state_dict(),
        "frozen_model_1_state_dict": ddmec.frozen_model_1.state_dict() if ddmec.frozen_model_1 else None,
        "frozen_model_2_state_dict": ddmec.frozen_model_2.state_dict() if ddmec.frozen_model_2 else None,
        "config": vars(ddmec.config),
        "gen_step": ddmec.gen_step,
    }, save_path)
    print(f"Trained models saved to {save_path}")
    
    print("\n" + "="*60)
    print("DDMEC Training Complete!")
    print("="*60)
    print("\nCheck ddmec_results.png for visualizations.")
    print("\nInterpretation:")
    print("  - Scatter plots show the learned coupling relationship")
    print("  - Marginals should match N(2,1) and N(10,1)")
    print("  - Coupling errors should be centered at 0")
    print("  - Correlation should be close to 1.0 for good coupling")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        import traceback
        traceback.print_exc()

