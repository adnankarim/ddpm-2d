"""
Quick script to test DDMEC coupling in forward/backward directions.
"""

import torch
from ddmec_1d import DDMEC1D


def test_forward(ddmec, x2_value=10.0):
    """Test forward direction: x2 -> x1"""
    x2 = torch.tensor([[x2_value]], device=ddmec.device)
    x1 = ddmec.sample_coupled(x2, direction="1->2", num_steps=50)
    return x1.item()


def test_backward(ddmec, x1_value=2.0):
    """Test backward direction: x1 -> x2"""
    x1 = torch.tensor([[x1_value]], device=ddmec.device)
    x2 = ddmec.sample_coupled(x1, direction="2->1", num_steps=50)
    return x2.item()


def main():
    print("="*60)
    print("DDMEC Coupling Test")
    print("="*60)
    
    # Load model
    print("\nLoading DDMEC models...")
    try:
        ddmec = DDMEC1D(
            model_path_1="ddpm_1d_2.pt",
            model_path_2="ddpm_1d_10.pt",
        )
        print("✓ Models loaded successfully")
        
        # Try to load trained model if exists
        try:
            checkpoint = torch.load("ddmec_trained.pt", map_location=ddmec.device)
            ddmec.ddpm_1.model.load_state_dict(checkpoint["model_1_state_dict"])
            ddmec.ddpm_2.model.load_state_dict(checkpoint["model_2_state_dict"])
            print("✓ Loaded trained DDMEC weights")
        except FileNotFoundError:
            print("⚠ No trained model found, using base models")
        
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        return
    
    # Test forward direction
    print("\n" + "-"*60)
    print("Testing FORWARD direction (x2 → x1)")
    print("-"*60)
    print("Expected: x2 ≈ 10 should generate x1 ≈ 2 (offset of 8)")
    
    test_values_x2 = [9.0, 10.0, 11.0, 12.0]
    for x2_val in test_values_x2:
        x1_generated = test_forward(ddmec, x2_val)
        expected_x1 = x2_val - 8.0
        error = abs(x1_generated - expected_x1)
        print(f"  x2 = {x2_val:5.2f} → x1 = {x1_generated:5.2f} (expected {expected_x1:5.2f}, error: {error:.3f})")
    
    # Test backward direction
    print("\n" + "-"*60)
    print("Testing BACKWARD direction (x1 → x2)")
    print("-"*60)
    print("Expected: x1 ≈ 2 should generate x2 ≈ 10 (offset of 8)")
    
    test_values_x1 = [1.0, 2.0, 3.0, 4.0]
    for x1_val in test_values_x1:
        x2_generated = test_backward(ddmec, x1_val)
        expected_x2 = x1_val + 8.0
        error = abs(x2_generated - expected_x2)
        print(f"  x1 = {x1_val:5.2f} → x2 = {x2_generated:5.2f} (expected {expected_x2:5.2f}, error: {error:.3f})")
    
    # Round-trip test
    print("\n" + "-"*60)
    print("Testing ROUND-TRIP consistency (x2 → x1 → x2)")
    print("-"*60)
    
    x2_start = 10.0
    x1_intermediate = test_forward(ddmec, x2_start)
    x2_final = test_backward(ddmec, x1_intermediate)
    roundtrip_error = abs(x2_final - x2_start)
    
    print(f"  Start:        x2 = {x2_start:.2f}")
    print(f"  Forward:      x2 → x1 = {x1_intermediate:.2f}")
    print(f"  Backward:     x1 → x2 = {x2_final:.2f}")
    print(f"  Round-trip error: {roundtrip_error:.3f}")
    
    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)
    
    # Quality assessment
    print("\nQuality Assessment:")
    if roundtrip_error < 0.5:
        print("  ✓ EXCELLENT - Very good coupling consistency")
    elif roundtrip_error < 1.0:
        print("  ✓ GOOD - Reasonable coupling quality")
    elif roundtrip_error < 2.0:
        print("  ⚠ FAIR - Coupling needs improvement")
    else:
        print("  ✗ POOR - Model needs more training")


if __name__ == "__main__":
    main()

