"""
Train both DDPM models (for N(2,1) and N(10,1)) with 1 million samples each.
Run this script before running run_ddmec.py
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from train_ddpm_1d import DDPM1D, DDPMConfig, make_toy_dataset


def train_ddpm(offset: float, model_path: str, num_samples: int = 1000000):
    """
    Train a DDPM model on 1D Gaussian data.
    
    Args:
        offset: Mean of the Gaussian (e.g., 2.0 or 10.0)
        model_path: Path to save the trained model
        num_samples: Number of training samples (default: 1 million)
    """
    cfg = DDPMConfig()
    cfg.model_path = model_path
    cfg.num_epochs = 100  # Reduced since we have 1M samples
    cfg.batch_size = 128  # Larger batch size for efficiency
    
    device = torch.device(cfg.device)
    print(f"\n{'='*60}")
    print(f"Training DDPM for N({offset}, 1) distribution")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Training samples: {num_samples:,}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Epochs: {cfg.num_epochs}")
    print(f"Model will be saved to: {model_path}")
    
    # Generate dataset
    print("\nGenerating dataset...")
    data = make_toy_dataset(num_samples, offset=offset)
    print(f"Dataset stats: mean={data.mean():.4f}, std={data.std():.4f}")
    
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    
    # Initialize DDPM
    ddpm = DDPM1D(cfg)
    
    # Training loop
    print("\nStarting training...")
    global_step = 0
    for epoch in range(cfg.num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for (x_batch,) in loader:
            x_batch = x_batch.to(device)
            loss = ddpm.train_step(x_batch)
            epoch_loss += loss
            num_batches += 1
            global_step += 1
        
        avg_loss = epoch_loss / num_batches
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{cfg.num_epochs}, loss={avg_loss:.6f}")
            
            # Test sampling
            if (epoch + 1) % 20 == 0:
                with torch.no_grad():
                    samples = ddpm.sample(num_samples=1000)
                    sample_mean = samples.mean().item()
                    sample_std = samples.std().item()
                    print(f"  Generated samples: mean={sample_mean:.4f}, std={sample_std:.4f} (target: {offset:.1f}, 1.0)")
    
    # Save model
    save_obj = {
        "model_state_dict": ddpm.model.state_dict(),
        "config": cfg.__dict__,
    }
    torch.save(save_obj, model_path)
    print(f"\n✓ Saved trained DDPM model to {model_path}")
    
    # Final evaluation
    print("\nFinal evaluation:")
    with torch.no_grad():
        samples = ddpm.sample(num_samples=5000)
        sample_mean = samples.mean().item()
        sample_std = samples.std().item()
        print(f"  Generated samples (n=5000):")
        print(f"    Mean: {sample_mean:.4f} (target: {offset:.1f})")
        print(f"    Std:  {sample_std:.4f} (target: 1.0)")
        print(f"    Mean error: {abs(sample_mean - offset):.4f}")
        print(f"    Std error:  {abs(sample_std - 1.0):.4f}")
    
    return ddpm


def main():
    print("="*60)
    print("Training Both DDPM Models with 1 Million Samples")
    print("="*60)
    
    # Train model 1: N(2, 1)
    ddpm_1 = train_ddpm(offset=2.0, model_path="ddpm_1d_2.pt", num_samples=1000000)
    
    # Train model 2: N(10, 1)
    ddpm_2 = train_ddpm(offset=10.0, model_path="ddpm_1d_10.pt", num_samples=1000000)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print("\nBoth models have been trained and saved:")
    print("  1. ddpm_1d_2.pt  - Models N(2, 1) distribution")
    print("  2. ddpm_1d_10.pt - Models N(10, 1) distribution")
    print("\nYou can now run: python run_ddmec.py")
    print("="*60)


if __name__ == "__main__":
    main()

