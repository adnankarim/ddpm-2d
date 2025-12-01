"""
Sync Local Logs to W&B and TensorBoard

This script syncs local JSON logs to W&B and/or TensorBoard.
Useful when network issues prevented logging during training.

Usage:
    # Sync a single run to both W&B and TensorBoard
    python sync_logs.py --run-dir runs/ddmec_kl0.5_clip0.1_lr1e-04_20251128_123456
    
    # Sync to W&B only
    python sync_logs.py --run-dir runs/my_run --wandb-only
    
    # Sync to TensorBoard only
    python sync_logs.py --run-dir runs/my_run --tensorboard-only
    
    # Sync all runs in a directory
    python sync_logs.py --runs-dir runs --all
    
    # List all available runs
    python sync_logs.py --list
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional, List, Dict

# Check for optional dependencies
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: tensorboard not installed. Install with: pip install tensorboard")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


def find_runs(runs_dir: str = "runs") -> List[Dict]:
    """Find all runs with local logs."""
    runs = []
    runs_path = Path(runs_dir)
    
    if not runs_path.exists():
        return runs
    
    for run_dir in runs_path.iterdir():
        if not run_dir.is_dir():
            continue
        
        log_file = run_dir / "logs" / "training_log.json"
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    log_data = json.load(f)
                
                runs.append({
                    "name": run_dir.name,
                    "path": str(run_dir),
                    "log_file": str(log_file),
                    "num_epochs": len(log_data.get("epochs", [])),
                    "config": log_data.get("config", {}),
                })
            except (json.JSONDecodeError, KeyError):
                pass
    
    return sorted(runs, key=lambda x: x["name"], reverse=True)


def list_runs(runs_dir: str = "runs"):
    """List all available runs with local logs."""
    runs = find_runs(runs_dir)
    
    if not runs:
        print(f"No runs found in {runs_dir}/")
        return
    
    print("=" * 80)
    print("Available Runs with Local Logs")
    print("=" * 80)
    print(f"{'Run Name':<50} {'Epochs':<10} {'KL Weight':<10}")
    print("-" * 80)
    
    for run in runs:
        kl_weight = run["config"].get("kl_weight", "N/A")
        print(f"{run['name']:<50} {run['num_epochs']:<10} {kl_weight:<10}")
    
    print("-" * 80)
    print(f"Total: {len(runs)} runs")
    print("=" * 80)


def load_local_log(run_dir: str) -> Optional[Dict]:
    """Load local JSON log from a run directory."""
    log_file = Path(run_dir) / "logs" / "training_log.json"
    
    if not log_file.exists():
        print(f"Error: Log file not found: {log_file}")
        return None
    
    try:
        with open(log_file, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse log file: {e}")
        return None


def sync_to_tensorboard(log_data: Dict, run_dir: str) -> bool:
    """Sync local log to TensorBoard."""
    if not TENSORBOARD_AVAILABLE:
        print("Error: TensorBoard not available")
        return False
    
    tensorboard_dir = Path(run_dir) / "tensorboard"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Syncing to TensorBoard: {tensorboard_dir}")
    
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    
    for epoch_data in log_data.get("epochs", []):
        epoch = epoch_data["epoch"]
        
        # Info metrics - Forward
        fwd = epoch_data.get("info_metrics_forward", {})
        writer.add_scalar('info/kl_div_1_forward', fwd.get('kl_div_1', 0), epoch)
        writer.add_scalar('info/kl_div_2_forward', fwd.get('kl_div_2', 0), epoch)
        writer.add_scalar('info/kl_div_total_forward', fwd.get('kl_div_total', 0), epoch)
        writer.add_scalar('info/entropy_x_forward', fwd.get('entropy_x', 0), epoch)
        writer.add_scalar('info/entropy_y_forward', fwd.get('entropy_y', 0), epoch)
        writer.add_scalar('info/joint_entropy_forward', fwd.get('joint_entropy', 0), epoch)
        writer.add_scalar('info/mutual_information_forward', fwd.get('mutual_information', 0), epoch)
        writer.add_scalar('info/conditional_entropy_x_given_y', fwd.get('conditional_entropy_x_given_y', 0), epoch)
        writer.add_scalar('info/conditional_entropy_y_given_x', fwd.get('conditional_entropy_y_given_x', 0), epoch)
        
        # Info metrics - Backward
        bwd = epoch_data.get("info_metrics_backward", {})
        writer.add_scalar('info/kl_div_1_backward', bwd.get('kl_div_1', 0), epoch)
        writer.add_scalar('info/kl_div_2_backward', bwd.get('kl_div_2', 0), epoch)
        writer.add_scalar('info/kl_div_total_backward', bwd.get('kl_div_total', 0), epoch)
        writer.add_scalar('info/mutual_information_backward', bwd.get('mutual_information', 0), epoch)
        writer.add_scalar('info/avg_kl_total', epoch_data.get('avg_kl_total', 0), epoch)
        
        # Learned statistics
        writer.add_scalar('stats/learned_mu_1_forward', fwd.get('learned_mu_1', 0), epoch)
        writer.add_scalar('stats/learned_sigma_1_forward', fwd.get('learned_sigma_1', 0), epoch)
        writer.add_scalar('stats/learned_mu_2_forward', fwd.get('learned_mu_2', 0), epoch)
        writer.add_scalar('stats/learned_sigma_2_forward', fwd.get('learned_sigma_2', 0), epoch)
        
        # Coupling metrics
        coupling = epoch_data.get("coupling_metrics", {})
        writer.add_scalar('coupling/corr_x2_to_x1', coupling.get('corr_x2_to_x1', 0), epoch)
        writer.add_scalar('coupling/corr_x1_to_x2', coupling.get('corr_x1_to_x2', 0), epoch)
        writer.add_scalar('coupling/mae_x2_to_x1', coupling.get('mae_x2_to_x1', 0), epoch)
        writer.add_scalar('coupling/mae_x1_to_x2', coupling.get('mae_x1_to_x2', 0), epoch)
        
        # Best tracking
        best = epoch_data.get("best", {})
        writer.add_scalar('best/kl_total', best.get('kl_total', float('inf')), epoch)
        writer.add_scalar('best/epoch', best.get('epoch', 0), epoch)
        
        # Training info
        phase = 0 if epoch_data.get("phase") == "warmup" else 1
        writer.add_scalar('train/phase', phase, epoch)
        writer.add_scalar('train/learning_rate', epoch_data.get('learning_rate', 0), epoch)
    
    writer.close()
    print(f"  [OK] Synced {len(log_data.get('epochs', []))} epochs to TensorBoard")
    return True


def sync_to_wandb(log_data: Dict, run_dir: str, project: str = "ddmec-sync") -> bool:
    """Sync local log to W&B."""
    if not WANDB_AVAILABLE:
        print("Error: W&B not available")
        return False
    
    experiment_name = log_data.get("experiment_name", os.path.basename(run_dir))
    config = log_data.get("config", {})
    
    print(f"Syncing to W&B project '{project}' as '{experiment_name}'")
    
    try:
        run = wandb.init(
            project=project,
            name=f"sync_{experiment_name}",
            config=config,
            reinit=True,
        )
        
        for epoch_data in log_data.get("epochs", []):
            epoch = epoch_data["epoch"]
            
            # Build log dict
            log_dict = {"epoch": epoch}
            
            # Info metrics - Forward
            fwd = epoch_data.get("info_metrics_forward", {})
            log_dict.update({
                "info/kl_div_1_forward": fwd.get('kl_div_1', 0),
                "info/kl_div_2_forward": fwd.get('kl_div_2', 0),
                "info/kl_div_total_forward": fwd.get('kl_div_total', 0),
                "info/entropy_x_forward": fwd.get('entropy_x', 0),
                "info/entropy_y_forward": fwd.get('entropy_y', 0),
                "info/joint_entropy_forward": fwd.get('joint_entropy', 0),
                "info/mutual_information_forward": fwd.get('mutual_information', 0),
                "info/conditional_entropy_x_given_y": fwd.get('conditional_entropy_x_given_y', 0),
                "info/conditional_entropy_y_given_x": fwd.get('conditional_entropy_y_given_x', 0),
            })
            
            # Info metrics - Backward
            bwd = epoch_data.get("info_metrics_backward", {})
            log_dict.update({
                "info/kl_div_1_backward": bwd.get('kl_div_1', 0),
                "info/kl_div_2_backward": bwd.get('kl_div_2', 0),
                "info/kl_div_total_backward": bwd.get('kl_div_total', 0),
                "info/mutual_information_backward": bwd.get('mutual_information', 0),
                "info/avg_kl_total": epoch_data.get('avg_kl_total', 0),
            })
            
            # Learned statistics
            log_dict.update({
                "stats/learned_mu_1_forward": fwd.get('learned_mu_1', 0),
                "stats/learned_sigma_1_forward": fwd.get('learned_sigma_1', 0),
                "stats/learned_mu_2_forward": fwd.get('learned_mu_2', 0),
                "stats/learned_sigma_2_forward": fwd.get('learned_sigma_2', 0),
            })
            
            # Coupling metrics
            coupling = epoch_data.get("coupling_metrics", {})
            log_dict.update({
                "coupling/corr_x2_to_x1": coupling.get('corr_x2_to_x1', 0),
                "coupling/corr_x1_to_x2": coupling.get('corr_x1_to_x2', 0),
                "coupling/mae_x2_to_x1": coupling.get('mae_x2_to_x1', 0),
                "coupling/mae_x1_to_x2": coupling.get('mae_x1_to_x2', 0),
            })
            
            # Best tracking
            best = epoch_data.get("best", {})
            log_dict.update({
                "best/kl_total": best.get('kl_total', float('inf')),
                "best/epoch": best.get('epoch', 0),
            })
            
            # Training info
            phase = 0 if epoch_data.get("phase") == "warmup" else 1
            log_dict.update({
                "train/phase": phase,
                "train/learning_rate": epoch_data.get('learning_rate', 0),
            })
            
            # Log epoch plot if exists
            plot_path = Path(run_dir) / "plots" / f"epoch_{epoch:04d}.png"
            if plot_path.exists():
                log_dict["plots/epoch_visualization"] = wandb.Image(str(plot_path))
            
            wandb.log(log_dict)
        
        wandb.finish()
        print(f"  [OK] Synced {len(log_data.get('epochs', []))} epochs to W&B")
        return True
        
    except Exception as e:
        print(f"  [ERROR] Error syncing to W&B: {e}")
        return False


def sync_run(run_dir: str, to_wandb: bool = True, to_tensorboard: bool = True,
             wandb_project: str = "ddmec-sync") -> bool:
    """Sync a single run to W&B and/or TensorBoard."""
    print("\n" + "=" * 60)
    print(f"Syncing: {run_dir}")
    print("=" * 60)
    
    log_data = load_local_log(run_dir)
    if log_data is None:
        return False
    
    print(f"Experiment: {log_data.get('experiment_name', 'unknown')}")
    print(f"Epochs: {len(log_data.get('epochs', []))}")
    print(f"Config: {log_data.get('config', {})}")
    
    success = True
    
    if to_tensorboard:
        if not sync_to_tensorboard(log_data, run_dir):
            success = False
    
    if to_wandb:
        if not sync_to_wandb(log_data, run_dir, wandb_project):
            success = False
    
    return success


def sync_all_runs(runs_dir: str = "runs", to_wandb: bool = True, 
                  to_tensorboard: bool = True, wandb_project: str = "ddmec-sync"):
    """Sync all runs in a directory."""
    runs = find_runs(runs_dir)
    
    if not runs:
        print(f"No runs found in {runs_dir}/")
        return
    
    print(f"\nFound {len(runs)} runs to sync")
    
    results = {"success": 0, "failed": 0}
    
    for run in runs:
        if sync_run(run["path"], to_wandb, to_tensorboard, wandb_project):
            results["success"] += 1
        else:
            results["failed"] += 1
    
    print("\n" + "=" * 60)
    print("Sync Complete!")
    print("=" * 60)
    print(f"Successful: {results['success']}")
    print(f"Failed: {results['failed']}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Sync local DDMEC logs to W&B and TensorBoard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sync a single run
  python sync_logs.py --run-dir runs/ddmec_kl0.5_clip0.1_lr1e-04_20251128_123456
  
  # Sync to W&B only
  python sync_logs.py --run-dir runs/my_run --wandb-only
  
  # Sync to TensorBoard only
  python sync_logs.py --run-dir runs/my_run --tensorboard-only
  
  # Sync all runs
  python sync_logs.py --all
  
  # List available runs
  python sync_logs.py --list
        """
    )
    
    parser.add_argument("--run-dir", type=str, help="Path to a single run directory to sync")
    parser.add_argument("--runs-dir", type=str, default="runs", help="Directory containing runs (default: runs)")
    parser.add_argument("--all", action="store_true", help="Sync all runs in runs-dir")
    parser.add_argument("--list", action="store_true", help="List available runs")
    parser.add_argument("--wandb-only", action="store_true", help="Only sync to W&B")
    parser.add_argument("--tensorboard-only", action="store_true", help="Only sync to TensorBoard")
    parser.add_argument("--wandb-project", type=str, default="ddmec-sync", help="W&B project name for synced runs")
    
    args = parser.parse_args()
    
    # Determine what to sync to
    to_wandb = not args.tensorboard_only
    to_tensorboard = not args.wandb_only
    
    if args.list:
        list_runs(args.runs_dir)
    elif args.all:
        sync_all_runs(args.runs_dir, to_wandb, to_tensorboard, args.wandb_project)
    elif args.run_dir:
        sync_run(args.run_dir, to_wandb, to_tensorboard, args.wandb_project)
    else:
        parser.print_help()
        print("\nError: Please specify --run-dir, --all, or --list")


if __name__ == "__main__":
    main()

