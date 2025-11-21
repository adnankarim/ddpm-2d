import os
import sys
import subprocess
import traceback

import matplotlib

# Use non-interactive backend so script won't hang on environments without a GUI
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch

from train_ddpm_1d import DDPM1D, DDPMConfig, compute_kl_divergence, make_toy_dataset


def load_model(model_path: str) -> DDPM1D:
    print(f"[DEBUG] Loading model from {model_path}")
    cfg = DDPMConfig()
    checkpoint = torch.load(model_path, map_location=cfg.device)

    # If config was saved, update defaults from file
    if "config" in checkpoint:
        for k, v in checkpoint["config"].items():
            setattr(cfg, k, v)

    ddpm = DDPM1D(cfg)
    ddpm.model.load_state_dict(checkpoint["model_state_dict"])
    ddpm.model.eval()
    print("[DEBUG] Model loaded and set to eval()")
    return ddpm


def _open_image(path: str) -> None:
    """Try to open an image file with the default viewer."""
    try:
        print(f"[DEBUG] Attempting to open image: {path}")
        if os.name == "nt":  # Windows
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":  # macOS
            subprocess.run(["open", path], check=False)
        else:  # Linux and others
            subprocess.run(["xdg-open", path], check=False)
    except Exception as exc:
        print(f"Could not auto-open image viewer: {exc}")


def main(model_path: str = "ddpm_1d.pt", num_samples: int = 200):
    print("[DEBUG] sample_and_plot.py main() starting")
    ddpm = load_model(model_path)

    with torch.no_grad():
        print(f"[DEBUG] Sampling {num_samples} points from DDPM")
        samples = ddpm.sample(num_samples=num_samples)
        samples_np = samples.cpu().numpy().flatten()
        
        # Generate true data for comparison
        true_data = make_toy_dataset(num_samples, offset=10.0)
        
        # Compute KL divergence
        kl_div = compute_kl_divergence(samples, true_data)
        print(f"[INFO] KL Divergence: {kl_div:.6f}")
        print(f"[INFO] Generated samples - Mean: {samples.mean().item():.4f}, Std: {samples.std().item():.4f}")
        print(f"[INFO] True data - Mean: {true_data.mean().item():.4f}, Std: {true_data.std().item():.4f}")

    print("[DEBUG] Creating plot")
    # Plot histogram of generated samples
    plt.figure(figsize=(8, 5))
    plt.hist(samples_np, bins=30, density=True, alpha=0.7, label="Generated")
    plt.hist(true_data.numpy().flatten(), bins=30, density=True, alpha=0.5, label="True Data")
    # Dataset is 10 + U(0, 1), so center around 10.5 roughly
    plt.axvline(10.0, color="red", linestyle="--", label="Offset (10.0)")
    plt.title(f"DDPM 1D Samples (KL Div: {kl_div:.4f})")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

    # Save to file so you always get an output, even in non-GUI environments
    out_path = os.path.abspath("ddpm_samples.png")
    plt.savefig(out_path)
    print(f"[DEBUG] Saved plot to {out_path}")

    # Try to open with system image viewer (helpful on Windows)
    _open_image(out_path)

    print("[DEBUG] Done")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("[ERROR] Exception while running sample_and_plot.py:")
        traceback.print_exc()


