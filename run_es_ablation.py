"""
ES-DDMEC Ablation Study Script

Runs systematic ablation studies over ES hyperparameters:
- Population size
- Sigma (noise scale)
- ES learning rate
- Mirror sampling
- Rank transform

Generates comprehensive reports and visualizations at the end.
"""

import os
import sys
import json
import datetime
import subprocess
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any

import torch

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# Ablation configurations
ABLATION_CONFIGS = {
    "population_size": {
        "name": "Population Size Ablation",
        "base": {"sigma": 0.001, "es_lr": 5e-4, "warmup_epochs": 10},
        "sweep": "population_size",
        "values": [10, 20, 30, 40, 50],
    },
    "sigma": {
        "name": "Sigma (Noise Scale) Ablation",
        "base": {"population_size": 30, "es_lr": 5e-4, "warmup_epochs": 10},
        "sweep": "sigma",
        "values": [0.0005, 0.001, 0.0015, 0.002, 0.003],
    },
    "es_lr": {
        "name": "ES Learning Rate Ablation",
        "base": {"population_size": 30, "sigma": 0.001, "warmup_epochs": 10},
        "sweep": "es_lr",
        "values": [1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3],
    },
    "mirror_sampling": {
        "name": "Mirror Sampling Ablation",
        "base": {"population_size": 30, "sigma": 0.001, "es_lr": 5e-4, "warmup_epochs": 10},
        "sweep": "use_mirror_sampling",
        "values": [False, True],
    },
    "rank_transform": {
        "name": "Rank Transform Ablation",
        "base": {"population_size": 30, "sigma": 0.001, "es_lr": 5e-4, "warmup_epochs": 10},
        "sweep": "use_rank_transform",
        "values": [False, True],
    },
}


def check_experiment_exists(run_dir: str, min_epochs: int = 10) -> bool:
    """Check if experiment already ran successfully."""
    log_path = os.path.join(run_dir, "logs", "training_log.json")
    if not os.path.exists(log_path):
        return False
    
    try:
        with open(log_path, 'r') as f:
            log = json.load(f)
        
        epochs = len(log.get("epochs", []))
        return epochs >= min_epochs
    except:
        return False


def run_single_experiment(
    config: Dict[str, Any],
    num_epochs: int,
    batch_size: int,
    num_samples: int,
    base_run_dir: str,
    use_wandb: bool = False,
    force: bool = False,
    min_epochs: int = 10,
) -> Dict[str, Any]:
    """Run a single ES-DDMEC experiment."""
    
    # Create run name (without timestamp - run_es_ddmec.py will add it)
    param_str = "_".join([f"{k}{v}" for k, v in sorted(config.items())])
    run_name = f"es_ablation_{param_str}"
    
    # Check if already exists
    if not force:
        if os.path.exists(base_run_dir):
            existing_dirs = [d for d in os.listdir(base_run_dir) if d.startswith(run_name)]
            for existing in existing_dirs:
                existing_path = os.path.join(base_run_dir, existing)
                if check_experiment_exists(existing_path, min_epochs):
                    print("\n" + "="*80)
                    print(f"[SKIP] Experiment already exists!")
                    print("="*80)
                    print(f"  Config: {config}")
                    print(f"  Existing run: {existing_path}")
                    print("="*80 + "\n")
                    
                    # Load existing results
                    log_path = os.path.join(existing_path, "logs", "training_log.json")
                    with open(log_path, 'r') as f:
                        log = json.load(f)
                    
                    last_epoch = log["epochs"][-1]
                    return {
                        "config": config,
                        "run_dir": existing_path,
                        "skipped": True,
                        "final_metrics": {
                            "kl_total_fwd": last_epoch["info_metrics_forward"]["kl_div_total"],
                            "kl_total_bwd": last_epoch["info_metrics_backward"]["kl_div_total"],
                            "mutual_info": last_epoch["info_metrics_forward"]["mutual_information"],
                            "corr_x2_to_x1": last_epoch["coupling_metrics"]["corr_x2_to_x1"],
                            "corr_x1_to_x2": last_epoch["coupling_metrics"]["corr_x1_to_x2"],
                            "mae_x2_to_x1": last_epoch["coupling_metrics"]["mae_x2_to_x1"],
                            "mae_x1_to_x2": last_epoch["coupling_metrics"]["mae_x1_to_x2"],
                        }
                    }
    
    print("\n" + "="*80)
    print(f"[NEW] Running experiment: {run_name}")
    print("="*80)
    print(f"  Config: {config}")
    print(f"  Num Epochs: {num_epochs}")
    print("="*80 + "\n")
    
    # Build command
    cmd = [
        sys.executable, "run_es_ddmec.py",
        "--num-epochs", str(num_epochs),
        "--batch-size", str(batch_size),
        "--num-samples", str(num_samples),
        "--run-name", run_name,
    ]
    
    # Add config parameters
    for key, value in config.items():
        if key == "use_mirror_sampling":
            if value:
                cmd.append("--use-mirror-sampling")
        elif key == "use_rank_transform":
            if value:
                cmd.append("--use-rank-transform")
        else:
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    
    # Add wandb flag
    if not use_wandb:
        cmd.append("--no-wandb")
    
    # Run experiment
    try:
        result = subprocess.run(cmd, check=True)
        
        # Find the actual created directory (run_es_ddmec.py adds timestamp)
        actual_run_dir = None
        if os.path.exists(base_run_dir):
            matching_dirs = [d for d in os.listdir(base_run_dir) if d.startswith(run_name)]
            if matching_dirs:
                # Get the most recently created one
                matching_dirs.sort(key=lambda x: os.path.getctime(os.path.join(base_run_dir, x)), reverse=True)
                actual_run_dir = os.path.join(base_run_dir, matching_dirs[0])
        
        if not actual_run_dir or not os.path.exists(actual_run_dir):
            raise FileNotFoundError(f"Could not find created run directory matching {run_name}")
        
        # Load results from actual directory
        log_path = os.path.join(actual_run_dir, "logs", "training_log.json")
        with open(log_path, 'r') as f:
            log = json.load(f)
        
        last_epoch = log["epochs"][-1]
        
        return {
            "config": config,
            "run_dir": actual_run_dir,
            "success": True,
            "final_metrics": {
                "kl_total_fwd": last_epoch["info_metrics_forward"]["kl_div_total"],
                "kl_total_bwd": last_epoch["info_metrics_backward"]["kl_div_total"],
                "mutual_info": last_epoch["info_metrics_forward"]["mutual_information"],
                "corr_x2_to_x1": last_epoch["coupling_metrics"]["corr_x2_to_x1"],
                "corr_x1_to_x2": last_epoch["coupling_metrics"]["corr_x1_to_x2"],
                "mae_x2_to_x1": last_epoch["coupling_metrics"]["mae_x2_to_x1"],
                "mae_x1_to_x2": last_epoch["coupling_metrics"]["mae_x1_to_x2"],
            }
        }
    
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Experiment failed: {e}")
        return {
            "config": config,
            "run_dir": run_name,
            "success": False,
            "error": str(e),
        }
    except FileNotFoundError as e:
        print(f"\n[ERROR] Could not find results: {e}")
        return {
            "config": config,
            "run_dir": run_name,
            "success": False,
            "error": str(e),
        }


def run_ablation_study(
    ablation_type: str,
    num_epochs: int,
    batch_size: int,
    num_samples: int,
    use_wandb: bool,
    force: bool,
    min_epochs: int,
) -> List[Dict[str, Any]]:
    """Run a full ablation study."""
    
    if ablation_type not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown ablation type: {ablation_type}")
    
    config = ABLATION_CONFIGS[ablation_type]
    
    print("\n" + "#"*80)
    print(f"# {config['name']}")
    print("#"*80)
    
    base_run_dir = "runs"
    os.makedirs(base_run_dir, exist_ok=True)
    
    results = []
    
    for value in config["values"]:
        # Create experiment config
        exp_config = config["base"].copy()
        exp_config[config["sweep"]] = value
        
        # Run experiment
        result = run_single_experiment(
            config=exp_config,
            num_epochs=num_epochs,
            batch_size=batch_size,
            num_samples=num_samples,
            base_run_dir=base_run_dir,
            use_wandb=use_wandb,
            force=force,
            min_epochs=min_epochs,
        )
        
        results.append(result)
    
    return results


def generate_comparison_plots(results: List[Dict[str, Any]], output_dir: str, ablation_name: str):
    """Generate comparison plots for ablation results."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    configs = [r["config"] for r in results if "final_metrics" in r]
    if not configs:
        print("No successful results to plot")
        return
    
    # Determine sweep parameter
    sweep_param = None
    for key in configs[0].keys():
        values = [c[key] for c in configs]
        if len(set(map(str, values))) > 1:
            sweep_param = key
            break
    
    if sweep_param is None:
        print("Could not determine sweep parameter")
        return
    
    x_values = [c[sweep_param] for c in configs]
    x_labels = [str(x) for x in x_values]
    
    # Metrics to plot
    metrics = {
        "kl_total_fwd": "KL Divergence (Forward)",
        "kl_total_bwd": "KL Divergence (Backward)",
        "mutual_info": "Mutual Information",
        "corr_x2_to_x1": "Correlation x2→x1",
        "corr_x1_to_x2": "Correlation x1→x2",
        "mae_x2_to_x1": "MAE x2→x1",
        "mae_x1_to_x2": "MAE x1→x2",
    }
    
    # Create 2x4 subplot
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, (metric_key, metric_name) in enumerate(metrics.items()):
        ax = axes[idx]
        
        y_values = [r["final_metrics"][metric_key] for r in results if "final_metrics" in r]
        
        ax.plot(range(len(x_values)), y_values, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel(sweep_param.replace('_', ' ').title(), fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(x_values)))
        ax.set_xticklabels(x_labels, rotation=45 if len(x_labels) > 5 else 0)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (x, y) in enumerate(zip(range(len(x_values)), y_values)):
            ax.text(x, y, f'{y:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Remove extra subplot
    fig.delaxes(axes[7])
    
    plt.suptitle(f'ES-DDMEC Ablation: {ablation_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "ablation_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n[PLOT] Comparison plot saved: {plot_path}")


def save_ablation_report(
    results: List[Dict[str, Any]], 
    output_dir: str, 
    ablation_name: str,
    ablation_type: str,
):
    """Save comprehensive ablation report."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # JSON report
    json_path = os.path.join(output_dir, "ablation_results.json")
    with open(json_path, 'w') as f:
        json.dump({
            "ablation_type": ablation_type,
            "ablation_name": ablation_name,
            "num_experiments": len(results),
            "timestamp": datetime.datetime.now().isoformat(),
            "results": results,
        }, f, indent=2)
    
    print(f"\n[JSON] Results saved: {json_path}")
    
    # Text report
    txt_path = os.path.join(output_dir, "ablation_summary.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"ES-DDMEC ABLATION STUDY: {ablation_name}\n")
        f.write("="*80 + "\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of experiments: {len(results)}\n")
        f.write("="*80 + "\n\n")
        
        for idx, result in enumerate(results, 1):
            f.write(f"Experiment {idx}/{len(results)}\n")
            f.write("-"*80 + "\n")
            f.write(f"Config: {result['config']}\n")
            f.write(f"Run dir: {result.get('run_dir', 'N/A')}\n")
            
            if "final_metrics" in result:
                f.write("\nFinal Metrics:\n")
                for key, value in result["final_metrics"].items():
                    f.write(f"  {key}: {value:.6f}\n")
            elif "error" in result:
                f.write(f"\nERROR: {result['error']}\n")
            
            f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("SUMMARY TABLE\n")
        f.write("="*80 + "\n")
        
        # Create summary table
        successful = [r for r in results if "final_metrics" in r]
        if successful:
            # Determine sweep parameter
            configs = [r["config"] for r in successful]
            sweep_param = None
            for key in configs[0].keys():
                values = [c[key] for c in configs]
                if len(set(map(str, values))) > 1:
                    sweep_param = key
                    break
            
            if sweep_param:
                f.write(f"\n{sweep_param:<15} {'KL Fwd':<10} {'KL Bwd':<10} {'MI':<10} {'Corr→':<10} {'Corr←':<10} {'MAE→':<10} {'MAE←':<10}\n")
                f.write("-"*80 + "\n")
                
                for r in successful:
                    param_val = str(r["config"][sweep_param])
                    m = r["final_metrics"]
                    f.write(f"{param_val:<15} "
                           f"{m['kl_total_fwd']:<10.4f} "
                           f"{m['kl_total_bwd']:<10.4f} "
                           f"{m['mutual_info']:<10.4f} "
                           f"{m['corr_x2_to_x1']:<10.4f} "
                           f"{m['corr_x1_to_x2']:<10.4f} "
                           f"{m['mae_x2_to_x1']:<10.4f} "
                           f"{m['mae_x1_to_x2']:<10.4f}\n")
                
                # Find best configuration
                best_idx = min(range(len(successful)), 
                              key=lambda i: (successful[i]["final_metrics"]["kl_total_fwd"] + 
                                           successful[i]["final_metrics"]["kl_total_bwd"]) / 2)
                
                f.write("\n" + "="*80 + "\n")
                f.write("BEST CONFIGURATION\n")
                f.write("="*80 + "\n")
                f.write(f"Config: {successful[best_idx]['config']}\n")
                f.write(f"Avg KL Total: {(successful[best_idx]['final_metrics']['kl_total_fwd'] + successful[best_idx]['final_metrics']['kl_total_bwd'])/2:.6f}\n")
                f.write(f"Run dir: {successful[best_idx]['run_dir']}\n")
    
    print(f"[TXT] Summary saved: {txt_path}")
    
    return txt_path


def concatenate_all_ablation_results(base_dir: str = "runs"):
    """Concatenate all ablation results into a single file."""
    
    output_file = os.path.join(base_dir, "ALL_ES_ABLATION_RESULTS.txt")
    
    # Find all ablation result directories
    ablation_dirs = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and item.startswith("es_ablation_report_"):
            ablation_dirs.append(item_path)
    
    if not ablation_dirs:
        print("No ablation results found to concatenate")
        return None
    
    with open(output_file, 'w', encoding='utf-8') as outf:
        outf.write("="*100 + "\n")
        outf.write("ES-DDMEC COMPLETE ABLATION STUDY RESULTS\n")
        outf.write("="*100 + "\n")
        outf.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        outf.write(f"Total ablation studies: {len(ablation_dirs)}\n")
        outf.write("="*100 + "\n\n\n")
        
        for idx, ablation_dir in enumerate(sorted(ablation_dirs), 1):
            summary_file = os.path.join(ablation_dir, "ablation_summary.txt")
            
            if os.path.exists(summary_file):
                outf.write("\n" + "#"*100 + "\n")
                outf.write(f"# ABLATION STUDY {idx}/{len(ablation_dirs)}\n")
                outf.write(f"# Directory: {ablation_dir}\n")
                outf.write("#"*100 + "\n\n")
                
                with open(summary_file, 'r', encoding='utf-8') as inf:
                    outf.write(inf.read())
                
                outf.write("\n\n")
        
        outf.write("\n" + "="*100 + "\n")
        outf.write("END OF CONCATENATED RESULTS\n")
        outf.write("="*100 + "\n")
    
    print(f"\n{'='*80}")
    print(f"[CONCATENATED] All results saved to: {output_file}")
    print(f"{'='*80}\n")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="ES-DDMEC Ablation Study",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--ablation", type=str, 
                       choices=list(ABLATION_CONFIGS.keys()) + ["all"],
                       default="all",
                       help="Type of ablation study to run")
    parser.add_argument("--num-epochs", type=int, default=30,
                       help="Number of training epochs per experiment")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size")
    parser.add_argument("--num-samples", type=int, default=100000,
                       help="Number of training samples")
    parser.add_argument("--no-wandb", action="store_true",
                       help="Disable W&B logging")
    parser.add_argument("--force", action="store_true",
                       help="Force re-run even if experiment exists")
    parser.add_argument("--min-epochs", type=int, default=10,
                       help="Minimum epochs for experiment to be considered complete")
    
    args = parser.parse_args()
    
    print("="*80)
    print("ES-DDMEC ABLATION STUDY")
    print("="*80)
    print(f"Ablation type: {args.ablation}")
    print(f"Number of epochs: {args.num_epochs}")
    print(f"W&B logging: {'disabled' if args.no_wandb else 'enabled'}")
    print(f"Skip existing: {'no (--force)' if args.force else 'yes'}")
    print(f"Min epochs for complete: {args.min_epochs}")
    print("="*80)
    
    # Check for pre-trained models
    if not os.path.exists("ddpm_1d_2.pt") or not os.path.exists("ddpm_1d_10.pt"):
        print("\nERROR: Pre-trained models not found!")
        print("Please run: python train_both_ddpm.py")
        return
    
    # Run ablations
    ablation_types = list(ABLATION_CONFIGS.keys()) if args.ablation == "all" else [args.ablation]
    
    all_reports = []
    
    for ablation_type in ablation_types:
        results = run_ablation_study(
            ablation_type=ablation_type,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            use_wandb=not args.no_wandb,
            force=args.force,
            min_epochs=args.min_epochs,
        )
        
        # Create report directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join("runs", f"es_ablation_report_{ablation_type}_{timestamp}")
        os.makedirs(report_dir, exist_ok=True)
        
        # Generate plots and reports
        generate_comparison_plots(results, report_dir, ABLATION_CONFIGS[ablation_type]["name"])
        report_file = save_ablation_report(results, report_dir, 
                                          ABLATION_CONFIGS[ablation_type]["name"],
                                          ablation_type)
        
        all_reports.append(report_file)
    
    # Concatenate all results
    if len(ablation_types) > 1 or args.ablation == "all":
        concat_file = concatenate_all_ablation_results()
        
        if concat_file:
            print(f"\n{'='*80}")
            print("ALL ABLATION STUDIES COMPLETE!")
            print(f"{'='*80}")
            print(f"\nConcatenated results: {concat_file}")
            print(f"Individual reports: runs/es_ablation_report_*/")
            print(f"{'='*80}\n")
    
    print("\n" + "="*80)
    print("ABLATION STUDY COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()

