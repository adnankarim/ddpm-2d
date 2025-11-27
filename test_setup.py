"""
Quick test to verify the setup is working
"""
import torch
import os

def test_setup():
    print("="*60)
    print("Testing DDMEC Setup")
    print("="*60)
    
    # Check if models exist
    model1_exists = os.path.exists("ddpm_1d_2.pt")
    model2_exists = os.path.exists("ddpm_1d_10.pt")
    
    print(f"\n✓ Checking pre-trained models:")
    print(f"  ddpm_1d_2.pt:  {'✓ Found' if model1_exists else '✗ Missing'}")
    print(f"  ddpm_1d_10.pt: {'✓ Found' if model2_exists else '✗ Missing'}")
    
    if not model1_exists or not model2_exists:
        print("\n⚠ Models not found! Run: python train_both_ddpm.py")
        return False
    
    # Test imports
    print(f"\n✓ Testing imports:")
    try:
        from ddmec_1d import DDMEC1D
        print("  ddmec_1d: ✓")
    except Exception as e:
        print(f"  ddmec_1d: ✗ {e}")
        return False
    
    try:
        from run_ddmec import create_coupled_data, count_parameters
        print("  run_ddmec: ✓")
    except Exception as e:
        print(f"  run_ddmec: ✗ {e}")
        return False
    
    # Test model loading
    print(f"\n✓ Testing model loading:")
    try:
        ddmec = DDMEC1D(
            model_path_1="ddpm_1d_2.pt",
            model_path_2="ddpm_1d_10.pt",
            use_wandb=False,
        )
        print("  Models loaded: ✓")
        
        # Count parameters
        trainable_1, total_1 = count_parameters(ddmec.ddpm_1.model)
        trainable_2, total_2 = count_parameters(ddmec.ddpm_2.model)
        
        print(f"\n  Model 1 parameters: {trainable_1:,}")
        print(f"  Model 2 parameters: {trainable_2:,}")
        print(f"  Total DDMEC parameters: {trainable_1 + trainable_2:,}")
        
    except Exception as e:
        print(f"  Model loading: ✗ {e}")
        return False
    
    # Test data creation
    print(f"\n✓ Testing data creation:")
    try:
        x1, x2 = create_coupled_data(100)
        print(f"  Created coupled data: ✓")
        print(f"  X1 shape: {x1.shape}, mean={x1.mean():.2f}, std={x1.std():.2f}")
        print(f"  X2 shape: {x2.shape}, mean={x2.mean():.2f}, std={x2.std():.2f}")
    except Exception as e:
        print(f"  Data creation: ✗ {e}")
        return False
    
    # Test sampling
    print(f"\n✓ Testing sampling:")
    try:
        with torch.no_grad():
            x2_test = torch.randn(10, 1).to(ddmec.device) + 10.0
            x1_gen = ddmec.sample_coupled(x2_test, direction="1->2", num_steps=10)
            print(f"  Sampling x1 from x2: ✓")
            print(f"  Generated x1: mean={x1_gen.mean():.2f}, std={x1_gen.std():.2f}")
            
            if torch.isnan(x1_gen).any():
                print("  ⚠ Warning: NaN detected in generated samples")
    except Exception as e:
        print(f"  Sampling: ✗ {e}")
        return False
    
    # Test run directory creation
    print(f"\n✓ Testing directory structure:")
    try:
        os.makedirs("runs/test_run/checkpoints", exist_ok=True)
        os.makedirs("runs/test_run/plots", exist_ok=True)
        print("  Directory creation: ✓")
        print("  runs/test_run/checkpoints: ✓")
        print("  runs/test_run/plots: ✓")
    except Exception as e:
        print(f"  Directory creation: ✗ {e}")
        return False
    
    print("\n" + "="*60)
    print("✓ All tests passed! Setup is ready.")
    print("="*60)
    print("\nYou can now run:")
    print("  python run_ddmec.py --no-wandb")
    print("or")
    print("  python run_ddmec.py")
    print("="*60)
    
    return True

if __name__ == "__main__":
    success = test_setup()
    exit(0 if success else 1)

