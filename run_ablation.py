"""
Ablation Study Script for DDMEC

This script runs multiple experiments with different hyperparameters:
- KL weight: 0.1, 0.3, 0.5, 1.0
- PPO clip: 0.1, 0.2, 0.3
- Learning rate: 1e-5, 5e-5, 1e-4, 5e-4

Usage:
    python run_ablation.py --ablation kl_weight     # Ablate KL weight only
    python run_ablation.py --ablation ppo_clip      # Ablate PPO clip only
    python run_ablation.py --ablation lr            # Ablate learning rate only
    python run_ablation.py --ablation all           # Run all ablations
    python run_ablation.py --quick                  # Quick ablation with fewer epochs
"""

import subprocess
import os
import sys
import datetime
import json
from pathlib import Path
from typing import Optional, List, Dict
from itertools import product


def find_existing_runs(runs_dir: str = "runs") -> List[Dict]:
    """Find all existing runs with their configs."""
    runs = []
    runs_path = Path(runs_dir)
    
    if not runs_path.exists():
        return runs
    
    for run_dir in runs_path.iterdir():
        if not run_dir.is_dir():
            continue
        
        # Check for local log
        log_file = run_dir / "logs" / "training_log.json"
        if log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
                
                runs.append({
                    "name": run_dir.name,
                    "path": str(run_dir),
                    "num_epochs": len(log_data.get("epochs", [])),
                    "config": log_data.get("config", {}),
                })
            except (json.JSONDecodeError, KeyError):
                pass
    
    return runs


def check_experiment_exists(kl_weight: float, ppo_clip: float, lr: float, 
                            runs_dir: str = "runs", min_epochs: int = 5) -> Optional[str]:
    """
    Check if an experiment with the same hyperparameters already exists and is complete.
    
    Returns path to existing run if found, None otherwise.
    """
    runs = find_existing_runs(runs_dir)
    
    for run in runs:
        config = run.get("config", {})
        run_kl = config.get("kl_weight")
        run_clip = config.get("ppo_clip")
        run_lr = config.get("learning_rate")
        num_epochs = run.get("num_epochs", 0)
        
        # Check if hyperparameters match (with floating point tolerance)
        if (run_kl is not None and abs(run_kl - kl_weight) < 1e-6 and
            run_clip is not None and abs(run_clip - ppo_clip) < 1e-6 and
            run_lr is not None and abs(run_lr - lr) < 1e-8):
            
            # Check if run has enough epochs to be considered complete
            if num_epochs >= min_epochs:
                return run["path"]
    
    return None


# Define hyperparameter grids
ABLATION_CONFIGS = {
    "kl_weight": {
        "values": [0.1, 0.3, 0.5, 1.0],
        "default_others": {"ppo_clip": 0.1, "lr": 1e-4}
    },
    "ppo_clip": {
        "values": [0.05, 0.1, 0.2, 0.3],
        "default_others": {"kl_weight": 0.5, "lr": 1e-4}
    },
    "lr": {
        "values": [1e-5, 5e-5, 1e-4, 5e-4],
        "default_others": {"kl_weight": 0.5, "ppo_clip": 0.1}
    }
}

# Best found hyperparameters from initial experiments
BEST_HYPERPARAMS = {
    "kl_weight": 0.5,
    "ppo_clip": 0.1,
    "lr": 1e-4,
    "warmup_epochs": 10,
    "rl_lr_scale": 0.1,
}


def run_experiment(kl_weight, ppo_clip, lr, num_epochs=50, warmup_epochs=10, 
                   rl_lr_scale=0.1, wandb_project="ddmec-ablation", no_wandb=False,
                   no_tensorboard=False, skip_existing=True, min_epochs_for_complete=5):
    """Run a single experiment with given hyperparameters.
    
    Args:
        skip_existing: If True, skip experiments that already exist with same hyperparameters
        min_epochs_for_complete: Minimum epochs for a run to be considered complete
    """
    
    # Check if this experiment already exists
    if skip_existing:
        existing_path = check_experiment_exists(
            kl_weight, ppo_clip, lr, 
            min_epochs=min_epochs_for_complete
        )
        if existing_path:
            print("\n" + "="*80)
            print(f"[SKIP] Experiment already exists!")
            print("="*80)
            print(f"  KL Weight:      {kl_weight}")
            print(f"  PPO Clip:       {ppo_clip}")
            print(f"  Learning Rate:  {lr}")
            print(f"  Existing run:   {existing_path}")
            print("="*80 + "\n")
            
            return {
                "run_name": os.path.basename(existing_path),
                "kl_weight": kl_weight,
                "ppo_clip": ppo_clip,
                "lr": lr,
                "return_code": 0,  # Consider existing as success
                "skipped": True,
                "existing_path": existing_path,
            }
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ablation_kl{kl_weight}_clip{ppo_clip}_lr{lr:.0e}_{timestamp}"
    
    cmd = [
        sys.executable, "run_ddmec.py",
        "--kl-weight", str(kl_weight),
        "--ppo-clip", str(ppo_clip),
        "--lr", str(lr),
        "--warmup-epochs", str(warmup_epochs),
        "--rl-lr-scale", str(rl_lr_scale),
        "--num-epochs", str(num_epochs),
        "--wandb-project", wandb_project,
        "--wandb-name", run_name,
    ]
    
    if no_wandb:
        cmd.append("--no-wandb")
    
    if no_tensorboard:
        cmd.append("--no-tensorboard")
    
    print("\n" + "="*80)
    print(f"[NEW] Running experiment: {run_name}")
    print("="*80)
    print(f"  KL Weight:      {kl_weight}")
    print(f"  PPO Clip:       {ppo_clip}")
    print(f"  Learning Rate:  {lr}")
    print(f"  Warmup Epochs:  {warmup_epochs}")
    print(f"  RL LR Scale:    {rl_lr_scale}")
    print(f"  Num Epochs:     {num_epochs}")
    print("="*80 + "\n")
    
    # Run the experiment
    result = subprocess.run(cmd)
    
    return {
        "run_name": run_name,
        "kl_weight": kl_weight,
        "ppo_clip": ppo_clip,
        "lr": lr,
        "return_code": result.returncode,
        "skipped": False,
    }


def run_kl_weight_ablation(num_epochs=50, no_wandb=False, no_tensorboard=False, 
                            skip_existing=True, min_epochs=5):
    """Ablation study on KL weight."""
    print("\n" + "#"*80)
    print("# KL WEIGHT ABLATION STUDY")
    print("#"*80)
    
    config = ABLATION_CONFIGS["kl_weight"]
    results = []
    
    for kl_weight in config["values"]:
        result = run_experiment(
            kl_weight=kl_weight,
            ppo_clip=config["default_others"]["ppo_clip"],
            lr=config["default_others"]["lr"],
            num_epochs=num_epochs,
            wandb_project="ddmec-ablation-kl",
            no_wandb=no_wandb,
            no_tensorboard=no_tensorboard,
            skip_existing=skip_existing,
            min_epochs_for_complete=min_epochs,
        )
        results.append(result)
    
    return results


def run_ppo_clip_ablation(num_epochs=50, no_wandb=False, no_tensorboard=False,
                          skip_existing=True, min_epochs=5):
    """Ablation study on PPO clip range."""
    print("\n" + "#"*80)
    print("# PPO CLIP ABLATION STUDY")
    print("#"*80)
    
    config = ABLATION_CONFIGS["ppo_clip"]
    results = []
    
    for ppo_clip in config["values"]:
        result = run_experiment(
            kl_weight=config["default_others"]["kl_weight"],
            ppo_clip=ppo_clip,
            lr=config["default_others"]["lr"],
            num_epochs=num_epochs,
            wandb_project="ddmec-ablation-clip",
            no_wandb=no_wandb,
            no_tensorboard=no_tensorboard,
            skip_existing=skip_existing,
            min_epochs_for_complete=min_epochs,
        )
        results.append(result)
    
    return results


def run_lr_ablation(num_epochs=50, no_wandb=False, no_tensorboard=False,
                    skip_existing=True, min_epochs=5):
    """Ablation study on learning rate."""
    print("\n" + "#"*80)
    print("# LEARNING RATE ABLATION STUDY")
    print("#"*80)
    
    config = ABLATION_CONFIGS["lr"]
    results = []
    
    for lr in config["values"]:
        result = run_experiment(
            kl_weight=config["default_others"]["kl_weight"],
            ppo_clip=config["default_others"]["ppo_clip"],
            lr=lr,
            num_epochs=num_epochs,
            wandb_project="ddmec-ablation-lr",
            no_wandb=no_wandb,
            no_tensorboard=no_tensorboard,
            skip_existing=skip_existing,
            min_epochs_for_complete=min_epochs,
        )
        results.append(result)
    
    return results


def run_all_ablations(num_epochs=50, no_wandb=False, no_tensorboard=False,
                      skip_existing=True, min_epochs=5):
    """Run all ablation studies."""
    all_results = {}
    
    all_results["kl_weight"] = run_kl_weight_ablation(num_epochs, no_wandb, no_tensorboard, skip_existing, min_epochs)
    all_results["ppo_clip"] = run_ppo_clip_ablation(num_epochs, no_wandb, no_tensorboard, skip_existing, min_epochs)
    all_results["lr"] = run_lr_ablation(num_epochs, no_wandb, no_tensorboard, skip_existing, min_epochs)
    
    return all_results


def generate_ablation_report(results, output_dir="runs/ablation_report"):
    """Generate a summary report of ablation results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw results
    results_path = os.path.join(output_dir, "ablation_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Generate summary (use ASCII-safe characters for Windows compatibility)
    summary_path = os.path.join(output_dir, "ablation_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("DDMEC ABLATION STUDY SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        for ablation_name, ablation_results in results.items():
            f.write(f"\n{ablation_name.upper()} ABLATION:\n")
            f.write("-"*40 + "\n")
            for r in ablation_results:
                if r.get("skipped", False):
                    status = "[SKIP]"
                elif r["return_code"] == 0:
                    status = "[OK]"
                else:
                    status = "[FAIL]"
                f.write(f"  {status} {r['run_name']}\n")
                f.write(f"      KL={r['kl_weight']}, Clip={r['ppo_clip']}, LR={r['lr']}\n")
                if r.get("skipped"):
                    f.write(f"      (existing: {r.get('existing_path', 'unknown')})\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Check W&B dashboard or runs/*/tables/epoch_metrics.csv for detailed metrics.\n")
        f.write("="*80 + "\n")
    
    print(f"\nAblation report saved to: {output_dir}")
    print(f"  - Results: {results_path}")
    print(f"  - Summary: {summary_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run DDMEC ablation studies")
    parser.add_argument("--ablation", type=str, default="all",
                        choices=["kl_weight", "ppo_clip", "lr", "all"],
                        help="Which ablation to run")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode with fewer epochs (20 instead of 50)")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable W&B logging")
    parser.add_argument("--no-tensorboard", action="store_true",
                        help="Disable TensorBoard logging")
    parser.add_argument("--num-epochs", type=int, default=None,
                        help="Number of epochs (default: 50, or 20 if --quick)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run even if experiment already exists")
    parser.add_argument("--min-epochs", type=int, default=5,
                        help="Minimum epochs for a run to be considered complete (default: 5)")
    
    args = parser.parse_args()
    
    # Determine number of epochs
    if args.num_epochs is not None:
        num_epochs = args.num_epochs
    elif args.quick:
        num_epochs = 20
    else:
        num_epochs = 50
    
    # Determine if we should skip existing
    skip_existing = not args.force
    
    print("="*80)
    print("DDMEC ABLATION STUDY")
    print("="*80)
    print(f"Ablation type: {args.ablation}")
    print(f"Number of epochs: {num_epochs}")
    print(f"W&B logging: {'disabled' if args.no_wandb else 'enabled'}")
    print(f"TensorBoard logging: {'disabled' if args.no_tensorboard else 'enabled'}")
    print(f"Skip existing: {'no (--force)' if args.force else 'yes'}")
    print(f"Min epochs for complete: {args.min_epochs}")
    print("="*80)
    
    # Run ablations
    if args.ablation == "kl_weight":
        results = {"kl_weight": run_kl_weight_ablation(num_epochs, args.no_wandb, args.no_tensorboard, skip_existing, args.min_epochs)}
    elif args.ablation == "ppo_clip":
        results = {"ppo_clip": run_ppo_clip_ablation(num_epochs, args.no_wandb, args.no_tensorboard, skip_existing, args.min_epochs)}
    elif args.ablation == "lr":
        results = {"lr": run_lr_ablation(num_epochs, args.no_wandb, args.no_tensorboard, skip_existing, args.min_epochs)}
    else:  # all
        results = run_all_ablations(num_epochs, args.no_wandb, args.no_tensorboard, skip_existing, args.min_epochs)
    
    # Generate report
    generate_ablation_report(results)
    
    print("\n" + "="*80)
    print("ABLATION STUDY COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()

