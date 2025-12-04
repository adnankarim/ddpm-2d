"""
Run ES-DDMEC: Evolution Strategies for Minimum Entropy Coupling

This script provides a complete interface for training ES-DDMEC on
1D Gaussian distributions, with support for ablation studies.

Based on the paper:
"Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning"
https://arxiv.org/abs/2509.24372

Usage:
    # Basic run with default parameters
    python run_es_ddmec.py

    # Run with custom ES parameters
    python run_es_ddmec.py --population-size 50 --sigma 0.0005 --es-lr 1e-4

    # Run ablation study
    python run_es_ddmec.py --ablation

    # Compare ES vs PPO
    python run_es_ddmec.py --compare-ppo
"""

import os
import sys
import argparse
import datetime
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from es_ddmec import ESDDMEC1D, ESConfig

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def create_coupled_data(num_samples: int = 10000, seed: int = 42):
    """
    Create coupled Gaussian data for minimum entropy coupling.
    
    The coupling is perfect: x1 = z + 2, x2 = z + 10
    where z ~ N(0, 1)
    
    This means:
    - x1 ~ N(2, 1)
    - x2 ~ N(10, 1)
    - x1 = x2 - 8 (perfect coupling)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Shared latent variable
    z = torch.randn(num_samples, 1)
    
    # Create coupling
    x1 = z + 2.0
    x2 = z + 10.0
    
    print(f"Dataset created (n={num_samples}, seed={seed}):")
    print(f"  X1: mean={x1.mean():.2f}, std={x1.std():.2f} (target: N(2,1))")
    print(f"  X2: mean={x2.mean():.2f}, std={x2.std():.2f} (target: N(10,1))")
    print(f"  Correlation: {np.corrcoef(x1.numpy().flatten(), x2.numpy().flatten())[0, 1]:.4f}")
    
    return x1, x2


def evaluate_coupling(es_ddmec: ESDDMEC1D, num_samples: int = 1000):
    """
    Evaluate the learned coupling quality.
    
    Returns metrics dictionary.
    """
    print("\n" + "="*60)
    print("Evaluating Learned Coupling")
    print("="*60)
    
    with torch.no_grad():
        # Direction 1: x2 -> x1
        x2_test = torch.randn(num_samples, 1, device=es_ddmec.device) + 10.0
        x1_gen = es_ddmec.sample_coupled(x2_test, direction="1->2")
        
        # Expected: x1 ≈ x2 - 8
        diff_1 = (x1_gen - (x2_test - 8.0)).abs().mean().item()
        corr_1 = np.corrcoef(
            x2_test.cpu().numpy().flatten(),
            x1_gen.cpu().numpy().flatten()
        )[0, 1]
        
        print(f"\nDirection x2 → x1:")
        print(f"  Input x2: mean={x2_test.mean():.2f}, std={x2_test.std():.2f}")
        print(f"  Generated x1: mean={x1_gen.mean():.2f}, std={x1_gen.std():.2f}")
        print(f"  Expected x1: mean=2.00, std=1.00")
        print(f"  Correlation: {corr_1:.4f}")
        print(f"  MAE from ideal coupling: {diff_1:.4f}")
        
        # Direction 2: x1 -> x2
        x1_test = torch.randn(num_samples, 1, device=es_ddmec.device) + 2.0
        x2_gen = es_ddmec.sample_coupled(x1_test, direction="2->1")
        
        diff_2 = (x2_gen - (x1_test + 8.0)).abs().mean().item()
        corr_2 = np.corrcoef(
            x1_test.cpu().numpy().flatten(),
            x2_gen.cpu().numpy().flatten()
        )[0, 1]
        
        print(f"\nDirection x1 → x2:")
        print(f"  Input x1: mean={x1_test.mean():.2f}, std={x1_test.std():.2f}")
        print(f"  Generated x2: mean={x2_gen.mean():.2f}, std={x2_gen.std():.2f}")
        print(f"  Expected x2: mean=10.00, std=1.00")
        print(f"  Correlation: {corr_2:.4f}")
        print(f"  MAE from ideal coupling: {diff_2:.4f}")
        
        # Round-trip consistency
        x2_back = es_ddmec.sample_coupled(x1_gen, direction="2->1")
        round_trip_error = (x2_back - x2_test).abs().mean().item()
        
        print(f"\nRound-trip (x2 → x1 → x2):")
        print(f"  Reconstruction error: {round_trip_error:.4f}")
        
        # Information metrics
        info_metrics = es_ddmec.compute_information_metrics(
            x1_gen, x2_test,
            true_mu_1=2.0, true_sigma_1=1.0,
            true_mu_2=10.0, true_sigma_2=1.0
        )
        
        print(f"\nInformation-Theoretic Metrics:")
        print(f"  KL Divergence (X1): {info_metrics['kl_div_1']:.4f}")
        print(f"  KL Divergence (X2): {info_metrics['kl_div_2']:.4f}")
        print(f"  Total KL: {info_metrics['kl_div_total']:.4f}")
        print(f"  Mutual Information I(X1;X2): {info_metrics['mutual_information']:.4f}")
        
        return {
            "x1_gen": x1_gen.cpu().numpy(),
            "x2_gen": x2_gen.cpu().numpy(),
            "x1_test": x1_test.cpu().numpy(),
            "x2_test": x2_test.cpu().numpy(),
            "corr_x2_to_x1": corr_1,
            "corr_x1_to_x2": corr_2,
            "mae_x2_to_x1": diff_1,
            "mae_x1_to_x2": diff_2,
            "round_trip_error": round_trip_error,
            "info_metrics": info_metrics,
        }


def visualize_results(results: dict, save_path: str = "es_ddmec_results.png"):
    """Create comprehensive visualization of results."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Row 1: Coupling scatter plots
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(results["x2_test"], results["x1_gen"], alpha=0.3, s=10, c='blue')
    ax1.plot([8, 12], [0, 4], 'r--', linewidth=2, label='Ideal: x1=x2-8')
    ax1.set_xlabel("x2 (condition)")
    ax1.set_ylabel("x1 (generated)")
    ax1.set_title("ES-DDMEC Coupling: x2 → x1")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(results["x1_test"], results["x2_gen"], alpha=0.3, s=10, c='orange')
    ax2.plot([0, 4], [8, 12], 'r--', linewidth=2, label='Ideal: x2=x1+8')
    ax2.set_xlabel("x1 (condition)")
    ax2.set_ylabel("x2 (generated)")
    ax2.set_title("ES-DDMEC Coupling: x1 → x2")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Joint distribution
    ax3 = fig.add_subplot(gs[0, 2])
    h = ax3.hist2d(
        results["x2_test"].flatten(),
        results["x1_gen"].flatten(),
        bins=30, cmap='Blues'
    )
    ax3.set_xlabel("x2")
    ax3.set_ylabel("x1")
    ax3.set_title("Joint Distribution p(x1, x2)")
    plt.colorbar(h[3], ax=ax3, label='Count')
    
    # Row 2: Marginal distributions
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(results["x1_gen"], bins=40, density=True, alpha=0.7, color='blue', label='Generated')
    ax4.axvline(2.0, color='red', linestyle='--', linewidth=2, label='Target μ=2')
    x_range = np.linspace(-2, 6, 100)
    ax4.plot(x_range, 1/np.sqrt(2*np.pi) * np.exp(-(x_range-2)**2/2), 'g-', linewidth=2, label='N(2,1)')
    ax4.set_xlabel("x1")
    ax4.set_ylabel("Density")
    ax4.set_title(f"Marginal X1 (μ={results['x1_gen'].mean():.2f}, σ={results['x1_gen'].std():.2f})")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(results["x2_gen"], bins=40, density=True, alpha=0.7, color='orange', label='Generated')
    ax5.axvline(10.0, color='red', linestyle='--', linewidth=2, label='Target μ=10')
    x_range = np.linspace(6, 14, 100)
    ax5.plot(x_range, 1/np.sqrt(2*np.pi) * np.exp(-(x_range-10)**2/2), 'g-', linewidth=2, label='N(10,1)')
    ax5.set_xlabel("x2")
    ax5.set_ylabel("Density")
    ax5.set_title(f"Marginal X2 (μ={results['x2_gen'].mean():.2f}, σ={results['x2_gen'].std():.2f})")
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Quality metrics bar chart
    ax6 = fig.add_subplot(gs[1, 2])
    metrics = ['Corr x2→x1', 'Corr x1→x2']
    values = [results['corr_x2_to_x1'], results['corr_x1_to_x2']]
    colors = ['blue', 'orange']
    bars = ax6.bar(metrics, values, color=colors, alpha=0.7)
    ax6.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Perfect')
    ax6.set_ylabel("Correlation")
    ax6.set_title("Coupling Correlation")
    ax6.set_ylim([0, 1.1])
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, values):
        ax6.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}',
                ha='center', va='bottom', fontweight='bold')
    
    # Row 3: Coupling errors
    ax7 = fig.add_subplot(gs[2, 0])
    errors_1 = results["x1_gen"].flatten() - (results["x2_test"].flatten() - 8.0)
    ax7.hist(errors_1, bins=40, density=True, alpha=0.7, color='blue')
    ax7.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect (0)')
    ax7.set_xlabel("Error: x1_gen - (x2 - 8)")
    ax7.set_ylabel("Density")
    ax7.set_title(f"Coupling Error x2→x1 (MAE={results['mae_x2_to_x1']:.3f})")
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    ax8 = fig.add_subplot(gs[2, 1])
    errors_2 = results["x2_gen"].flatten() - (results["x1_test"].flatten() + 8.0)
    ax8.hist(errors_2, bins=40, density=True, alpha=0.7, color='orange')
    ax8.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect (0)')
    ax8.set_xlabel("Error: x2_gen - (x1 + 8)")
    ax8.set_ylabel("Density")
    ax8.set_title(f"Coupling Error x1→x2 (MAE={results['mae_x1_to_x2']:.3f})")
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # Summary metrics
    ax9 = fig.add_subplot(gs[2, 2])
    info = results['info_metrics']
    metrics_text = (
        f"Information-Theoretic Metrics\n"
        f"{'='*30}\n\n"
        f"KL(X1 || Target): {info['kl_div_1']:.4f}\n"
        f"KL(X2 || Target): {info['kl_div_2']:.4f}\n"
        f"Total KL: {info['kl_div_total']:.4f}\n\n"
        f"H(X1): {info['entropy_x']:.4f} nats\n"
        f"H(X2): {info['entropy_y']:.4f} nats\n"
        f"H(X1,X2): {info['joint_entropy']:.4f} nats\n\n"
        f"I(X1;X2): {info['mutual_information']:.4f} nats\n\n"
        f"Round-trip Error: {results['round_trip_error']:.4f}"
    )
    ax9.text(0.1, 0.5, metrics_text, transform=ax9.transAxes, fontsize=11,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax9.axis('off')
    ax9.set_title("Summary")
    
    plt.suptitle('ES-DDMEC: Evolution Strategies for Minimum Entropy Coupling',
                 fontsize=16, fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nVisualization saved to {save_path}")


def count_parameters(model):
    """Count trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def run_ablation_study(args):
    """
    Run ablation study over ES hyperparameters.
    
    Tests different combinations of:
    - Population size: [10, 30, 50]
    - Sigma: [0.0005, 0.001, 0.0015]
    - ES learning rate: [1e-4, 5e-4, 1e-3]
    """
    print("\n" + "="*60)
    print("ES-DDMEC Ablation Study")
    print("="*60)
    
    # Define ablation configurations
    configs = [
        # Baseline (paper defaults)
        {"population_size": 30, "sigma": 0.001, "es_lr": 5e-4, "name": "baseline"},
        # Population size variations
        {"population_size": 10, "sigma": 0.001, "es_lr": 5e-4, "name": "pop10"},
        {"population_size": 50, "sigma": 0.001, "es_lr": 5e-4, "name": "pop50"},
        # Sigma variations
        {"population_size": 30, "sigma": 0.0005, "es_lr": 5e-4, "name": "sigma_low"},
        {"population_size": 30, "sigma": 0.0015, "es_lr": 5e-4, "name": "sigma_high"},
        # Learning rate variations
        {"population_size": 30, "sigma": 0.001, "es_lr": 1e-4, "name": "lr_low"},
        {"population_size": 30, "sigma": 0.001, "es_lr": 1e-3, "name": "lr_high"},
    ]
    
    results = []
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ablation_dir = os.path.join("runs", f"es_ablation_{timestamp}")
    os.makedirs(ablation_dir, exist_ok=True)
    
    # Create shared dataset
    x1_train, x2_train = create_coupled_data(num_samples=args.num_samples, seed=args.seed)
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Running configuration: {config['name']}")
        print(f"  Population: {config['population_size']}")
        print(f"  Sigma: {config['sigma']}")
        print(f"  ES LR: {config['es_lr']}")
        print("="*60)
        
        run_dir = os.path.join(ablation_dir, config['name'])
        
        try:
            es_ddmec = ESDDMEC1D(
                model_path_1=args.model_path_1,
                model_path_2=args.model_path_2,
                use_wandb=args.use_wandb,
                use_tensorboard=True,
                population_size=config['population_size'],
                sigma=config['sigma'],
                es_lr=config['es_lr'],
                warmup_epochs=args.warmup_epochs,
            )
            
            es_ddmec.train(
                dataset_1=x1_train,
                dataset_2=x2_train,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                run_dir=run_dir,
            )
            
            # Evaluate
            eval_results = evaluate_coupling(es_ddmec, num_samples=1000)
            
            results.append({
                "config": config,
                "corr_x2_to_x1": eval_results["corr_x2_to_x1"],
                "corr_x1_to_x2": eval_results["corr_x1_to_x2"],
                "mae_x2_to_x1": eval_results["mae_x2_to_x1"],
                "mae_x1_to_x2": eval_results["mae_x1_to_x2"],
                "kl_total": eval_results["info_metrics"]["kl_div_total"],
                "mutual_info": eval_results["info_metrics"]["mutual_information"],
            })
            
        except Exception as e:
            print(f"ERROR in configuration {config['name']}: {e}")
            results.append({
                "config": config,
                "error": str(e),
            })
    
    # Save ablation results
    results_path = os.path.join(ablation_dir, "ablation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("Ablation Study Summary")
    print("="*60)
    print(f"\n{'Config':<15} {'Corr x2→x1':<12} {'Corr x1→x2':<12} {'MAE x2→x1':<12} {'KL Total':<12}")
    print("-"*60)
    
    for r in results:
        if "error" not in r:
            print(f"{r['config']['name']:<15} {r['corr_x2_to_x1']:<12.4f} "
                  f"{r['corr_x1_to_x2']:<12.4f} {r['mae_x2_to_x1']:<12.4f} "
                  f"{r['kl_total']:<12.4f}")
        else:
            print(f"{r['config']['name']:<15} ERROR: {r['error'][:30]}")
    
    print(f"\nResults saved to: {results_path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="ES-DDMEC: Evolution Strategies for Minimum Entropy Coupling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model paths
    parser.add_argument("--model-path-1", type=str, default="ddpm_1d_2.pt",
                        help="Path to first pre-trained model (N(2,1))")
    parser.add_argument("--model-path-2", type=str, default="ddpm_1d_10.pt",
                        help="Path to second pre-trained model (N(10,1))")
    
    # ES hyperparameters
    parser.add_argument("--population-size", type=int, default=30,
                        help="ES population size (paper recommends 30)")
    parser.add_argument("--sigma", type=float, default=0.001,
                        help="Noise scale for parameter perturbations")
    parser.add_argument("--es-lr", type=float, default=5e-4,
                        help="ES learning rate")
    parser.add_argument("--use-rank-transform", action="store_true",
                        help="Use rank-based fitness shaping")
    parser.add_argument("--use-mirror-sampling", action="store_true",
                        help="Use antithetic (mirror) sampling")
    
    # Training parameters
    parser.add_argument("--warmup-epochs", type=int, default=10,
                        help="Number of supervised warmup epochs")
    parser.add_argument("--num-epochs", type=int, default=50,
                        help="Total number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for supervised phases")
    parser.add_argument("--num-samples", type=int, default=100000,
                        help="Number of training samples")
    
    # Sampling parameters
    parser.add_argument("--mc-steps", type=int, default=5,
                        help="Monte Carlo steps for reward estimation")
    parser.add_argument("--sampling-steps", type=int, default=50,
                        help="Number of diffusion sampling steps")
    
    # Logging
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable Weights & Biases logging")
    parser.add_argument("--no-tensorboard", action="store_true",
                        help="Disable TensorBoard logging")
    parser.add_argument("--wandb-project", type=str, default="es-ddmec-1d",
                        help="W&B project name")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Custom run name")
    
    # Other
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--ablation", action="store_true",
                        help="Run ablation study")
    parser.add_argument("--compare-ppo", action="store_true",
                        help="Compare ES with PPO-DDMEC")
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Check for pre-trained models
    if not os.path.exists(args.model_path_1) or not os.path.exists(args.model_path_2):
        print("\nERROR: Pre-trained models not found!")
        print(f"  Model 1: {args.model_path_1} - {'Found' if os.path.exists(args.model_path_1) else 'NOT FOUND'}")
        print(f"  Model 2: {args.model_path_2} - {'Found' if os.path.exists(args.model_path_2) else 'NOT FOUND'}")
        print("\nPlease run train_both_ddpm.py first to create the base models.")
        return
    
    # Handle ablation study
    if args.ablation:
        run_ablation_study(args)
        return
    
    # Create run directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_name:
        run_name = f"{args.run_name}_{timestamp}"
    else:
        run_name = f"es_ddmec_pop{args.population_size}_sig{args.sigma}_lr{args.es_lr:.0e}_{timestamp}"
    
    run_dir = os.path.join("runs", run_name)
    
    print("="*60)
    print("ES-DDMEC: Evolution Strategies for Minimum Entropy Coupling")
    print("="*60)
    print(f"\nRun: {run_name}")
    print(f"Directory: {run_dir}")
    
    # Print configuration
    print("\n" + "-"*60)
    print("Configuration:")
    print("-"*60)
    print(f"  Population Size: {args.population_size}")
    print(f"  Sigma: {args.sigma}")
    print(f"  ES Learning Rate: {args.es_lr}")
    print(f"  Warmup Epochs: {args.warmup_epochs}")
    print(f"  Total Epochs: {args.num_epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Training Samples: {args.num_samples}")
    print(f"  Rank Transform: {args.use_rank_transform}")
    print(f"  Mirror Sampling: {args.use_mirror_sampling}")
    print("-"*60)
    
    # Initialize W&B
    args.use_wandb = not args.no_wandb and WANDB_AVAILABLE
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "method": "ES-DDMEC",
                "population_size": args.population_size,
                "sigma": args.sigma,
                "es_lr": args.es_lr,
                "warmup_epochs": args.warmup_epochs,
                "num_epochs": args.num_epochs,
                "batch_size": args.batch_size,
                "num_samples": args.num_samples,
                "supervised_lr": args.lr,
                "mc_steps": args.mc_steps,
                "sampling_steps": args.sampling_steps,
                "use_rank_transform": args.use_rank_transform,
                "use_mirror_sampling": args.use_mirror_sampling,
                "seed": args.seed,
            }
        )
        print(f"\n[OK] W&B initialized: {wandb.run.get_url()}")
    
    # Create data
    print("\nCreating training data...")
    x1_train, x2_train = create_coupled_data(num_samples=args.num_samples, seed=args.seed)
    
    # Initialize ES-DDMEC
    print("\nInitializing ES-DDMEC...")
    es_ddmec = ESDDMEC1D(
        model_path_1=args.model_path_1,
        model_path_2=args.model_path_2,
        use_wandb=args.use_wandb,
        use_tensorboard=not args.no_tensorboard,
        population_size=args.population_size,
        sigma=args.sigma,
        es_lr=args.es_lr,
        warmup_epochs=args.warmup_epochs,
        use_rank_transform=args.use_rank_transform,
        use_mirror_sampling=args.use_mirror_sampling,
        mc_steps=args.mc_steps,
        num_sampling_steps=args.sampling_steps,
    )
    
    # Log model info
    trainable_1, total_1 = count_parameters(es_ddmec.ddpm_1.model)
    trainable_2, total_2 = count_parameters(es_ddmec.ddpm_2.model)
    
    print("\nModel Architecture:")
    print(f"  Model 1: {trainable_1:,} trainable / {total_1:,} total parameters")
    print(f"  Model 2: {trainable_2:,} trainable / {total_2:,} total parameters")
    print(f"  Total: {trainable_1 + trainable_2:,} trainable parameters")
    
    if args.use_wandb:
        wandb.config.update({
            "model_1_params": trainable_1,
            "model_2_params": trainable_2,
            "total_params": trainable_1 + trainable_2,
        })
    
    # Train
    print("\nStarting training...")
    print("="*60)
    
    es_ddmec.train(
        dataset_1=x1_train,
        dataset_2=x2_train,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        run_dir=run_dir,
    )
    
    # Evaluate final model
    results = evaluate_coupling(es_ddmec, num_samples=10000)
    
    # Visualize
    print("\nGenerating final visualization...")
    vis_path = os.path.join(run_dir, "es_ddmec_final_results.png")
    visualize_results(results, save_path=vis_path)
    
    # Also save to current directory
    visualize_results(results, save_path="es_ddmec_results.png")
    
    # Log final metrics to W&B
    if args.use_wandb:
        wandb.log({
            "final/corr_x2_to_x1": results["corr_x2_to_x1"],
            "final/corr_x1_to_x2": results["corr_x1_to_x2"],
            "final/mae_x2_to_x1": results["mae_x2_to_x1"],
            "final/mae_x1_to_x2": results["mae_x1_to_x2"],
            "final/round_trip_error": results["round_trip_error"],
            "final/kl_total": results["info_metrics"]["kl_div_total"],
            "final/mutual_info": results["info_metrics"]["mutual_information"],
        })
        
        # Log visualization
        if os.path.exists(vis_path):
            wandb.log({"final/visualization": wandb.Image(vis_path)})
        
        wandb.finish()
    
    # Save final model
    save_path = os.path.join(run_dir, "es_ddmec_trained.pt")
    torch.save({
        "model_1_state_dict": es_ddmec.ddpm_1.model.state_dict(),
        "model_2_state_dict": es_ddmec.ddpm_2.model.state_dict(),
        "config": vars(es_ddmec.config),
        "es_config": {
            "population_size": args.population_size,
            "sigma": args.sigma,
            "learning_rate": args.es_lr,
        },
        "gen_step": es_ddmec.gen_step,
        "final_metrics": {
            "corr_x2_to_x1": results["corr_x2_to_x1"],
            "corr_x1_to_x2": results["corr_x1_to_x2"],
            "mae_x2_to_x1": results["mae_x2_to_x1"],
            "mae_x1_to_x2": results["mae_x1_to_x2"],
            "kl_total": results["info_metrics"]["kl_div_total"],
        },
    }, save_path)
    print(f"\nFinal model saved to: {save_path}")
    
    # Also save to current directory
    torch.save({
        "model_1_state_dict": es_ddmec.ddpm_1.model.state_dict(),
        "model_2_state_dict": es_ddmec.ddpm_2.model.state_dict(),
        "config": vars(es_ddmec.config),
        "es_config": {
            "population_size": args.population_size,
            "sigma": args.sigma,
            "learning_rate": args.es_lr,
        },
    }, "es_ddmec_trained.pt")
    
    print("\n" + "="*60)
    print("ES-DDMEC Training Complete!")
    print("="*60)
    print(f"\nFinal Results:")
    print(f"  Correlation x2→x1: {results['corr_x2_to_x1']:.4f}")
    print(f"  Correlation x1→x2: {results['corr_x1_to_x2']:.4f}")
    print(f"  MAE x2→x1: {results['mae_x2_to_x1']:.4f}")
    print(f"  MAE x1→x2: {results['mae_x1_to_x2']:.4f}")
    print(f"  Total KL: {results['info_metrics']['kl_div_total']:.4f}")
    print(f"  Mutual Information: {results['info_metrics']['mutual_information']:.4f}")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.finish()
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.finish()

