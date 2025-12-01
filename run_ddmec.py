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
import wandb
import os


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
        
        # Compute information-theoretic metrics
        print("\n" + "="*60)
        print("Information-Theoretic Metrics")
        print("="*60)
        
        info_metrics_1 = ddmec.compute_information_metrics(
            x1_gen, x2_test,
            true_mu_1=2.0, true_sigma_1=1.0,
            true_mu_2=10.0, true_sigma_2=1.0
        )
        
        info_metrics_2 = ddmec.compute_information_metrics(
            x1_test, x2_gen,
            true_mu_1=2.0, true_sigma_1=1.0,
            true_mu_2=10.0, true_sigma_2=1.0
        )
        
        print("\nDirection x2 → x1:")
        print(f"  KL Divergence D(p₁||q₁): {info_metrics_1['kl_div_1']:.4f}")
        print(f"  KL Divergence D(p₂||q₂): {info_metrics_1['kl_div_2']:.4f}")
        print(f"  Total KL Divergence: {info_metrics_1['kl_div_total']:.4f}")
        print(f"  Entropy H(X₁): {info_metrics_1['entropy_x']:.4f} nats")
        print(f"  Entropy H(X₂): {info_metrics_1['entropy_y']:.4f} nats")
        print(f"  Joint Entropy H(X₁,X₂): {info_metrics_1['joint_entropy']:.4f} nats")
        print(f"  Mutual Information I(X₁;X₂): {info_metrics_1['mutual_information']:.4f} nats")
        print(f"  Conditional Entropy H(X₁|X₂): {info_metrics_1['conditional_entropy_x_given_y']:.4f} nats")
        print(f"  Conditional Entropy H(X₂|X₁): {info_metrics_1['conditional_entropy_y_given_x']:.4f} nats")
        
        print("\nDirection x1 → x2:")
        print(f"  KL Divergence D(p₁||q₁): {info_metrics_2['kl_div_1']:.4f}")
        print(f"  KL Divergence D(p₂||q₂): {info_metrics_2['kl_div_2']:.4f}")
        print(f"  Total KL Divergence: {info_metrics_2['kl_div_total']:.4f}")
        print(f"  Mutual Information I(X₁;X₂): {info_metrics_2['mutual_information']:.4f} nats")
        
        print("\nTheoretical Reference (Perfect Independent Coupling):")
        print(f"  H(X₁) ≈ {0.5 * np.log(2 * np.pi * np.e):.4f} nats")
        print(f"  H(X₂) ≈ {0.5 * np.log(2 * np.pi * np.e):.4f} nats")
        print(f"  H(X₁,X₂) ≈ {np.log(2 * np.pi * np.e):.4f} nats (if independent)")
        print(f"  I(X₁;X₂) ≈ 0.0 nats (if independent)")
        print(f"  Note: Minimum entropy coupling has I(X₁;X₂) → max")
        
        return {
            "x1_gen": x1_gen.cpu().numpy(),
            "x2_gen": x2_gen.cpu().numpy(),
            "x1_test": x1_test.cpu().numpy(),
            "x2_test": x2_test.cpu().numpy(),
            "info_metrics_1": info_metrics_1,
            "info_metrics_2": info_metrics_2,
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


def count_parameters(model):
    """Count trainable and total parameters in a model."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def main(use_wandb=True, use_tensorboard=True, wandb_project="ddmec-1d", wandb_name=None,
         kl_weight=0.5, ppo_clip=0.1, lr=1e-4, warmup_epochs=10, rl_lr_scale=0.1,
         num_epochs=50, batch_size=64):
    print("="*60)
    print("DDMEC: Minimum Entropy Coupling for 1D Gaussians")
    print("="*60)
    
    # Create run directory with hyperparams in name for ablation studies
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if wandb_name:
        run_name = f"{wandb_name}_{timestamp}"
    else:
        run_name = f"ddmec_kl{kl_weight}_clip{ppo_clip}_lr{lr:.0e}_{timestamp}"
    
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"\nRun directory: {run_dir}")
    
    # Print hyperparameters
    print("\n" + "-"*60)
    print("Hyperparameters:")
    print("-"*60)
    print(f"  KL Weight:      {kl_weight}")
    print(f"  PPO Clip:       {ppo_clip}")
    print(f"  Learning Rate:  {lr}")
    print(f"  Warmup Epochs:  {warmup_epochs}")
    print(f"  RL LR Scale:    {rl_lr_scale}")
    print(f"  Num Epochs:     {num_epochs}")
    print(f"  Batch Size:     {batch_size}")
    print("-"*60)
    
    # Initialize Weights & Biases
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=run_name,
            config={
                "architecture": "DDMEC",
                "dataset": "1D Gaussians",
                "dist_1": "N(2, 1)",
                "dist_2": "N(10, 1)",
                "num_train_samples": 100000,
                "batch_size": batch_size,
                "learning_rate": lr,
                "num_epochs": num_epochs,
                "warmup_epochs": warmup_epochs,
                "num_timesteps": 1000,
                "run_dir": run_dir,
                # RL hyperparameters
                "kl_weight": kl_weight,
                "ppo_clip": ppo_clip,
                "rl_lr_scale": rl_lr_scale,
            }
        )
        print(f"\n[OK] Weights & Biases initialized: {wandb.run.name}")
        print(f"  Dashboard: {wandb.run.get_url()}")
    
    # Check if models exist
    if not os.path.exists("ddpm_1d_2.pt") or not os.path.exists("ddpm_1d_10.pt"):
        print("\nERROR: Pre-trained models not found!")
        print("Please ensure ddpm_1d_2.pt and ddpm_1d_10.pt are in the current directory.")
        print("You can train them using train_both_ddpm.py")
        if use_wandb:
            wandb.finish()
        return
    
    # Initialize DDMEC with hyperparameters
    print("\nInitializing DDMEC...")
    ddmec = DDMEC1D(
        model_path_1="ddpm_1d_2.pt",
        model_path_2="ddpm_1d_10.pt",
        use_wandb=use_wandb,
        use_tensorboard=use_tensorboard,
        kl_weight=kl_weight,
        ppo_clip=ppo_clip,
        warmup_epochs=warmup_epochs,
        rl_lr_scale=rl_lr_scale,
    )
    
    # Log parameter counts
    print("\n" + "="*60)
    print("Model Architecture")
    print("="*60)
    
    trainable_1, total_1 = count_parameters(ddmec.ddpm_1.model)
    trainable_2, total_2 = count_parameters(ddmec.ddpm_2.model)
    
    print(f"\nDDPM Model 1 (N(2,1)):")
    print(f"  Trainable parameters: {trainable_1:,}")
    print(f"  Total parameters:     {total_1:,}")
    
    print(f"\nDDPM Model 2 (N(10,1)):")
    print(f"  Trainable parameters: {trainable_2:,}")
    print(f"  Total parameters:     {total_2:,}")
    
    print(f"\nDDMEC Overall:")
    print(f"  Trainable parameters: {trainable_1 + trainable_2:,}")
    print(f"  Total parameters:     {total_1 + total_2:,}")
    print("="*60)
    
    if use_wandb:
        wandb.config.update({
            "model_1_trainable_params": trainable_1,
            "model_1_total_params": total_1,
            "model_2_trainable_params": trainable_2,
            "model_2_total_params": total_2,
            "ddmec_trainable_params": trainable_1 + trainable_2,
            "ddmec_total_params": total_1 + total_2,
        })
    
    # Create coupled training data
    print("\nCreating coupled training data...")
    num_train = 100000
    x1_train, x2_train = create_coupled_data(num_train)
    
    # Train DDMEC
    print("\nTraining DDMEC...")
    print(f"Phase 1: Warmup (supervised) - first {warmup_epochs} epochs")
    print("Phase 2: Cooperative (RL-based) - remaining epochs")
    print("-" * 60)
    
    ddmec.train(
        dataset_1=x1_train,
        dataset_2=x2_train,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        run_dir=run_dir,
    )
    
    # Evaluate
    results = evaluate_coupling(ddmec, num_samples=10000)
    
    # Visualize
    print("\nGenerating visualizations...")
    visualize_results(results)
    
    # Log final metrics and visualization to wandb
    if use_wandb:
        print("\nLogging results to Weights & Biases...")
        
        # Log final information-theoretic metrics
        if "info_metrics_1" in results:
            wandb.log({
                "final/kl_div_1_forward": results["info_metrics_1"]["kl_div_1"],
                "final/kl_div_2_forward": results["info_metrics_1"]["kl_div_2"],
                "final/kl_div_total_forward": results["info_metrics_1"]["kl_div_total"],
                "final/mutual_information_forward": results["info_metrics_1"]["mutual_information"],
                "final/joint_entropy_forward": results["info_metrics_1"]["joint_entropy"],
                "final/conditional_entropy_x_given_y": results["info_metrics_1"]["conditional_entropy_x_given_y"],
            })
        
        if "info_metrics_2" in results:
            wandb.log({
                "final/kl_div_1_backward": results["info_metrics_2"]["kl_div_1"],
                "final/kl_div_2_backward": results["info_metrics_2"]["kl_div_2"],
                "final/kl_div_total_backward": results["info_metrics_2"]["kl_div_total"],
                "final/mutual_information_backward": results["info_metrics_2"]["mutual_information"],
            })
        
        # Log visualization
        if os.path.exists("ddmec_results.png"):
            try:
                wandb.log({"visualization": wandb.Image("ddmec_results.png")})
            except Exception as e:
                print(f"Warning: Failed to log visualization to W&B: {e}")
    
    # Save trained model FIRST (before artifact logging which may fail)
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
    
    # Try to log artifact to wandb (may fail due to network issues)
    if use_wandb:
        try:
            artifact = wandb.Artifact("ddmec-model", type="model")
            artifact.add_file(save_path)
            wandb.log_artifact(artifact)
            print(f"[OK] Model artifact logged to W&B")
        except Exception as e:
            print(f"Warning: Failed to log model artifact to W&B: {e}")
            print("  (Model was saved locally - this is just a W&B connectivity issue)")
        
        try:
            print(f"[OK] Results logged to W&B: {wandb.run.get_url()}")
            wandb.finish()
        except Exception as e:
            print(f"Warning: Error finishing W&B run: {e}")
    
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
    import argparse
    
    parser = argparse.ArgumentParser(description="Train DDMEC on 1D Gaussian coupling")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard logging")
    parser.add_argument("--wandb-project", type=str, default="ddmec-1d", help="W&B project name")
    parser.add_argument("--wandb-name", type=str, default=None, help="W&B run name")
    
    # RL Hyperparameters for ablation studies
    parser.add_argument("--kl-weight", type=float, default=0.5,
                        help="KL regularization weight (Adnan used 0.1, we recommend 0.5 for stability)")
    parser.add_argument("--ppo-clip", type=float, default=0.1,
                        help="PPO clipping range (Adnan used 0.2, we recommend 0.1 for stability)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--warmup-epochs", type=int, default=10,
                        help="Number of warmup epochs with supervised training only")
    parser.add_argument("--rl-lr-scale", type=float, default=0.1,
                        help="Learning rate scale factor during RL phase (lower = more stable)")
    parser.add_argument("--num-epochs", type=int, default=50,
                        help="Total number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for training")
    
    args = parser.parse_args()
    
    try:
        main(
            use_wandb=not args.no_wandb,
            use_tensorboard=not args.no_tensorboard,
            wandb_project=args.wandb_project,
            wandb_name=args.wandb_name,
            kl_weight=args.kl_weight,
            ppo_clip=args.ppo_clip,
            lr=args.lr,
            warmup_epochs=args.warmup_epochs,
            rl_lr_scale=args.rl_lr_scale,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        if wandb.run is not None:
            wandb.finish()
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        if wandb.run is not None:
            wandb.finish()

