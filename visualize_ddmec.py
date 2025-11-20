"""
Script to visualize DDMEC trained models.
Shows forward and backward coupling results.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from ddmec_1d import DDMEC1D, ConditionalMLP
from train_ddpm_1d import SmallMLP


def visualize_coupling(ddmec, num_samples=1000, save_path="ddmec_visualization.png"):
    """
    Visualize both forward and backward coupling directions.
    
    Args:
        ddmec: Trained DDMEC model
        num_samples: Number of samples to generate
        save_path: Where to save the visualization
    """
    print(f"Generating {num_samples} samples for visualization...")
    
    with torch.no_grad():
        # Forward: x2 -> x1
        x2_samples = torch.randn(num_samples, 1, device=ddmec.device) + 10.0
        x1_from_x2 = ddmec.sample_coupled(x2_samples, direction="1->2", num_steps=50)
        
        # Backward: x1 -> x2
        x1_samples = torch.randn(num_samples, 1, device=ddmec.device) + 2.0
        x2_from_x1 = ddmec.sample_coupled(x1_samples, direction="2->1", num_steps=50)
    
    # Convert to numpy
    x1_from_x2_np = x1_from_x2.cpu().numpy().flatten()
    x2_from_x1_np = x2_from_x1.cpu().numpy().flatten()
    x1_samples_np = x1_samples.cpu().numpy().flatten()
    x2_samples_np = x2_samples.cpu().numpy().flatten()
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Forward direction (x2 -> x1)
    # Scatter plot
    axes[0, 0].scatter(x2_samples_np, x1_from_x2_np, alpha=0.3, s=10)
    axes[0, 0].plot([9, 11], [1, 3], 'r--', linewidth=2, label='Ideal (offset=8)')
    axes[0, 0].set_xlabel("x2 (condition)", fontsize=12)
    axes[0, 0].set_ylabel("x1 (generated)", fontsize=12)
    axes[0, 0].set_title("Forward: x2 → x1", fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Marginal histogram
    axes[0, 1].hist(x1_from_x2_np, bins=40, density=True, alpha=0.7, color='blue')
    axes[0, 1].axvline(2.0, color='red', linestyle='--', linewidth=2, label='True mean (2.0)')
    axes[0, 1].set_xlabel("x1 value", fontsize=12)
    axes[0, 1].set_ylabel("Density", fontsize=12)
    axes[0, 1].set_title("Generated x1 Distribution", fontsize=14)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Error distribution
    error_forward = x1_from_x2_np - (x2_samples_np - 8.0)
    axes[0, 2].hist(error_forward, bins=40, density=True, alpha=0.7, color='green')
    axes[0, 2].axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect')
    axes[0, 2].set_xlabel("Coupling Error", fontsize=12)
    axes[0, 2].set_ylabel("Density", fontsize=12)
    axes[0, 2].set_title(f"Forward Error (MAE={np.abs(error_forward).mean():.3f})", fontsize=14)
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Row 2: Backward direction (x1 -> x2)
    # Scatter plot
    axes[1, 0].scatter(x1_samples_np, x2_from_x1_np, alpha=0.3, s=10, color='orange')
    axes[1, 0].plot([1, 3], [9, 11], 'r--', linewidth=2, label='Ideal (offset=8)')
    axes[1, 0].set_xlabel("x1 (condition)", fontsize=12)
    axes[1, 0].set_ylabel("x2 (generated)", fontsize=12)
    axes[1, 0].set_title("Backward: x1 → x2", fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Marginal histogram
    axes[1, 1].hist(x2_from_x1_np, bins=40, density=True, alpha=0.7, color='orange')
    axes[1, 1].axvline(10.0, color='red', linestyle='--', linewidth=2, label='True mean (10.0)')
    axes[1, 1].set_xlabel("x2 value", fontsize=12)
    axes[1, 1].set_ylabel("Density", fontsize=12)
    axes[1, 1].set_title("Generated x2 Distribution", fontsize=14)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Error distribution
    error_backward = x2_from_x1_np - (x1_samples_np + 8.0)
    axes[1, 2].hist(error_backward, bins=40, density=True, alpha=0.7, color='purple')
    axes[1, 2].axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect')
    axes[1, 2].set_xlabel("Coupling Error", fontsize=12)
    axes[1, 2].set_ylabel("Density", fontsize=12)
    axes[1, 2].set_title(f"Backward Error (MAE={np.abs(error_backward).mean():.3f})", fontsize=14)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle('DDMEC Coupling Visualization: Forward & Backward', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n✓ Visualization saved to: {save_path}")
    
    # Print statistics
    print("\n" + "="*60)
    print("Coupling Statistics")
    print("="*60)
    print(f"\nForward (x2 → x1):")
    print(f"  Input x2:  mean={x2_samples_np.mean():.2f}, std={x2_samples_np.std():.2f}")
    print(f"  Output x1: mean={x1_from_x2_np.mean():.2f}, std={x1_from_x2_np.std():.2f}")
    print(f"  Expected:  mean=2.00, std=1.00")
    print(f"  MAE:       {np.abs(error_forward).mean():.3f}")
    print(f"  Correlation: {np.corrcoef(x2_samples_np, x1_from_x2_np)[0,1]:.3f}")
    
    print(f"\nBackward (x1 → x2):")
    print(f"  Input x1:  mean={x1_samples_np.mean():.2f}, std={x1_samples_np.std():.2f}")
    print(f"  Output x2: mean={x2_from_x1_np.mean():.2f}, std={x2_from_x1_np.std():.2f}")
    print(f"  Expected:  mean=10.00, std=1.00")
    print(f"  MAE:       {np.abs(error_backward).mean():.3f}")
    print(f"  Correlation: {np.corrcoef(x1_samples_np, x2_from_x1_np)[0,1]:.3f}")
    
    # Round-trip test
    with torch.no_grad():
        x1_roundtrip = ddmec.sample_coupled(
            torch.tensor(x2_from_x1_np[:100].reshape(-1, 1), device=ddmec.device),
            direction="1->2", num_steps=50
        )
    roundtrip_error = (x1_roundtrip.cpu().numpy().flatten() - x1_samples_np[:100]).std()
    print(f"\nRound-trip consistency (x1 → x2 → x1):")
    print(f"  STD error: {roundtrip_error:.3f}")


def load_and_visualize(model_path="ddmec_trained.pt", num_samples=1000):
    """
    Load a saved DDMEC model and visualize it.
    
    Args:
        model_path: Path to saved model
        num_samples: Number of samples for visualization
    """
    print("="*60)
    print("DDMEC Visualization Tool")
    print("="*60)
    
    try:
        # Load checkpoint
        print(f"\nLoading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize DDMEC with base models
        print("Initializing DDMEC structure...")
        ddmec = DDMEC1D(
            model_path_1="ddpm_1d_2.pt",
            model_path_2="ddpm_1d_10.pt",
        )
        
        # Load trained weights
        print("Loading trained weights...")
        ddmec.ddpm_1.model.load_state_dict(checkpoint["model_1_state_dict"])
        ddmec.ddpm_2.model.load_state_dict(checkpoint["model_2_state_dict"])
        
        # Frozen models are optional (only exist if DDMEC training completed warmup)
        if checkpoint.get("frozen_model_1_state_dict"):
            # Create a new ConditionalMLP instance for frozen model
            base_model_1 = SmallMLP()
            ddmec.frozen_model_1 = ConditionalMLP(base_model_1)
            ddmec.frozen_model_1.load_state_dict(checkpoint["frozen_model_1_state_dict"])
            ddmec.frozen_model_1.eval()
            ddmec.frozen_model_1.requires_grad_(False)
        
        if checkpoint.get("frozen_model_2_state_dict"):
            base_model_2 = SmallMLP()
            ddmec.frozen_model_2 = ConditionalMLP(base_model_2)
            ddmec.frozen_model_2.load_state_dict(checkpoint["frozen_model_2_state_dict"])
            ddmec.frozen_model_2.eval()
            ddmec.frozen_model_2.requires_grad_(False)
        
        print(f"✓ Model loaded (training step: {checkpoint.get('gen_step', 'unknown')})")
        
        # Visualize
        visualize_coupling(ddmec, num_samples=num_samples)
        
    except FileNotFoundError:
        print(f"\n✗ Model file not found: {model_path}")
        print("\nPlease train DDMEC first using: python run_ddmec.py")
    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()


def visualize_current_models(num_samples=1000):
    """
    Visualize the current state of the models (after training in current session).
    
    Args:
        num_samples: Number of samples for visualization
    """
    print("="*60)
    print("DDMEC Current Session Visualization")
    print("="*60)
    
    print("\nInitializing DDMEC with current models...")
    ddmec = DDMEC1D(
        model_path_1="ddpm_1d_2.pt",
        model_path_2="ddpm_1d_10.pt",
    )
    
    # Note: This will visualize the pre-trained models if DDMEC wasn't trained yet
    print("✓ Models loaded")
    
    visualize_coupling(ddmec, num_samples=num_samples, save_path="ddmec_current_state.png")


if __name__ == "__main__":
    import sys
    import os
    
    if len(sys.argv) > 1:
        # User provided a model path
        model_path = sys.argv[1]
        num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
        load_and_visualize(model_path, num_samples)
    else:
        # Check if trained model exists
        if os.path.exists("ddmec_trained.pt"):
            print("Found trained model: ddmec_trained.pt")
            load_and_visualize("ddmec_trained.pt")
        else:
            print("No trained model found. Visualizing current base models...")
            print("(Note: Train DDMEC first with 'python run_ddmec.py' for better results)")
            visualize_current_models()

