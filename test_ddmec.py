"""
Quick test script to verify DDMEC implementation.

This performs a minimal test to check if all components work.
"""

import torch
import numpy as np
from ddmec_1d import DDMEC1D


def test_loading():
    """Test that models load correctly."""
    print("Test 1: Loading models...")
    try:
        ddmec = DDMEC1D(
            model_path_1="ddpm_1d_2.pt",
            model_path_2="ddpm_1d_10.pt",
        )
        print("  ✓ Models loaded successfully")
        print(f"  ✓ Device: {ddmec.device}")
        print(f"  ✓ Config timesteps: {ddmec.config.timesteps}")
        return ddmec
    except Exception as e:
        print(f"  ✗ Failed to load models: {e}")
        return None


def test_sampling(ddmec):
    """Test trajectory sampling with log probabilities."""
    print("\nTest 2: Sampling with log probabilities...")
    try:
        condition = torch.randn(4, 1, device=ddmec.device) + 10.0
        
        trajectory, log_probs, timesteps = ddmec.sample_trajectory_with_logprob(
            model=ddmec.ddpm_1.model,
            ddpm=ddmec.ddpm_1,
            condition=condition,
            num_steps=20,
        )
        
        print(f"  ✓ Generated trajectory with {len(trajectory)} states")
        print(f"  ✓ Log probs shape: {len(log_probs)}")
        print(f"  ✓ Final sample mean: {trajectory[-1].mean():.2f}")
        return True
    except Exception as e:
        print(f"  ✗ Sampling failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reward(ddmec):
    """Test reward computation."""
    print("\nTest 3: Computing rewards...")
    try:
        x_gen = torch.randn(4, 1, device=ddmec.device) + 2.0
        condition = torch.randn(4, 1, device=ddmec.device) + 10.0
        
        rewards = ddmec.compute_reward(
            x_gen=x_gen,
            condition=condition,
            reward_model=ddmec.ddpm_2.model,
            reward_ddpm=ddmec.ddpm_2,
            mc_steps=3,
        )
        
        print(f"  ✓ Rewards computed: shape={rewards.shape}")
        print(f"  ✓ Reward mean: {rewards.mean():.2f}")
        print(f"  ✓ Reward std: {rewards.std():.2f}")
        return True
    except Exception as e:
        print(f"  ✗ Reward computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_supervised_training(ddmec):
    """Test supervised training step."""
    print("\nTest 4: Supervised training step...")
    try:
        x1 = torch.randn(8, 1, device=ddmec.device) + 2.0
        x2 = torch.randn(8, 1, device=ddmec.device) + 10.0
        
        opt1 = torch.optim.Adam(ddmec.ddpm_1.model.parameters(), lr=1e-4)
        opt2 = torch.optim.Adam(ddmec.ddpm_2.model.parameters(), lr=1e-4)
        
        losses = ddmec.train_supervised(x1, x2, opt1, opt2)
        
        print(f"  ✓ Training step completed")
        print(f"  ✓ Loss 1: {losses['loss_1']:.4f}")
        print(f"  ✓ Loss 2: {losses['loss_2']:.4f}")
        return True
    except Exception as e:
        print(f"  ✗ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_frozen_priors(ddmec):
    """Test frozen prior setup."""
    print("\nTest 5: Setting up frozen priors...")
    try:
        ddmec.setup_priors()
        
        # Verify frozen
        assert ddmec.frozen_model_1 is not None
        assert ddmec.frozen_model_2 is not None
        
        # Verify no gradients
        for param in ddmec.frozen_model_1.parameters():
            assert not param.requires_grad
        
        for param in ddmec.frozen_model_2.parameters():
            assert not param.requires_grad
        
        print("  ✓ Frozen priors created")
        print("  ✓ Gradients disabled")
        return True
    except Exception as e:
        print(f"  ✗ Frozen prior setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_policy_gradient(ddmec):
    """Test policy gradient update."""
    print("\nTest 6: Policy gradient update...")
    try:
        # Setup frozen priors first
        if ddmec.frozen_model_1 is None:
            ddmec.setup_priors()
        
        # Generate trajectory
        condition = torch.randn(4, 1, device=ddmec.device) + 10.0
        trajectory, log_probs, timesteps = ddmec.sample_trajectory_with_logprob(
            model=ddmec.ddpm_1.model,
            ddpm=ddmec.ddpm_1,
            condition=condition,
            num_steps=10,
        )
        
        # Compute rewards
        x_gen = trajectory[-1]
        rewards = torch.randn(4, device=ddmec.device)  # Dummy rewards
        
        # PG update
        opt = torch.optim.Adam(ddmec.ddpm_1.model.parameters(), lr=1e-4)
        metrics = ddmec.policy_gradient_update(
            trajectory=trajectory,
            log_probs=log_probs,
            rewards=rewards,
            gen_model=ddmec.ddpm_1.model,
            frozen_model=ddmec.frozen_model_1,
            condition=condition,
            timesteps_list=timesteps,
            optimizer=opt,
            kl_weight=0.1,
        )
        
        print(f"  ✓ PG update completed")
        print(f"  ✓ PG loss: {metrics['pg_loss']:.4f}")
        print(f"  ✓ KL reg: {metrics['kl_reg']:.4f}")
        return True
    except Exception as e:
        print(f"  ✗ PG update failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_sync(ddmec):
    """Test model synchronization."""
    print("\nTest 7: Model synchronization...")
    try:
        x_gen = torch.randn(8, 1, device=ddmec.device) + 2.0
        condition = torch.randn(8, 1, device=ddmec.device) + 10.0
        
        opt = torch.optim.Adam(ddmec.ddpm_2.model.parameters(), lr=1e-4)
        
        ddmec.sync_reward_model(
            x_gen=x_gen,
            condition=condition,
            reward_model=ddmec.ddpm_2.model,
            reward_ddpm=ddmec.ddpm_2,
            optimizer=opt,
            num_updates=3,
        )
        
        print("  ✓ Model sync completed")
        return True
    except Exception as e:
        print(f"  ✗ Model sync failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_training_step(ddmec):
    """Test complete training step."""
    print("\nTest 8: Full training step...")
    try:
        x1 = torch.randn(16, 1, device=ddmec.device) + 2.0
        x2 = torch.randn(16, 1, device=ddmec.device) + 10.0
        
        opt1 = torch.optim.Adam(ddmec.ddpm_1.model.parameters(), lr=1e-4)
        opt2 = torch.optim.Adam(ddmec.ddpm_2.model.parameters(), lr=1e-4)
        
        # Warmup step
        metrics = ddmec.train_step(x1, x2, opt1, opt2)
        print(f"  ✓ Warmup step completed")
        print(f"    Phase: {metrics['phase']}")
        
        # Advance to cooperative phase
        ddmec.gen_step = ddmec.warmup_steps
        
        # Cooperative step
        metrics = ddmec.train_step(x1, x2, opt1, opt2)
        print(f"  ✓ Cooperative step completed")
        print(f"    Phase: {metrics['phase']}")
        print(f"    Metrics: {len(metrics)} values logged")
        
        return True
    except Exception as e:
        print(f"  ✗ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sampling_coupled(ddmec):
    """Test coupled sampling."""
    print("\nTest 9: Coupled sampling...")
    try:
        # Sample x1 given x2
        x2 = torch.tensor([[10.0]], device=ddmec.device)
        x1_gen = ddmec.sample_coupled(x2, direction="1->2", num_steps=20)
        print(f"  ✓ Generated x1={x1_gen.item():.2f} given x2=10.0")
        
        # Sample x2 given x1
        x1 = torch.tensor([[2.0]], device=ddmec.device)
        x2_gen = ddmec.sample_coupled(x1, direction="2->1", num_steps=20)
        print(f"  ✓ Generated x2={x2_gen.item():.2f} given x1=2.0")
        
        return True
    except Exception as e:
        print(f"  ✗ Coupled sampling failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("DDMEC Implementation Test Suite")
    print("="*60)
    
    results = {}
    
    # Test 1: Loading
    ddmec = test_loading()
    results["loading"] = ddmec is not None
    
    if ddmec is None:
        print("\n✗ Cannot proceed without models. Please ensure:")
        print("  - ddpm_1d_2.pt exists")
        print("  - ddpm_1d_10.pt exists")
        return
    
    # Test 2-9: All components
    results["sampling"] = test_sampling(ddmec)
    results["reward"] = test_reward(ddmec)
    results["supervised"] = test_supervised_training(ddmec)
    results["frozen"] = test_frozen_priors(ddmec)
    results["policy_grad"] = test_policy_gradient(ddmec)
    results["sync"] = test_model_sync(ddmec)
    results["train_step"] = test_full_training_step(ddmec)
    results["coupled_sample"] = test_sampling_coupled(ddmec)
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print("-"*60)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! DDMEC is ready to use.")
        print("\nRun 'python run_ddmec.py' to train DDMEC.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please debug before training.")
    
    return passed == total


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)

