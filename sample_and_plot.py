import os
import sys
import subprocess
import traceback

import matplotlib

# Use non-interactive backend so script won't hang on environments without a GUI
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch

from train_ddpm_1d import DDPM1D, DDPMConfig


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
        samples = ddpm.sample(num_samples=num_samples).cpu().numpy().flatten()

    print("[DEBUG] Creating plot")
    # Plot histogram of generated samples
    plt.figure(figsize=(6, 4))
    plt.hist(samples, bins=30, density=True, alpha=0.7, label="Generated")
    # Dataset is 10 + U(0, 1), so center around 10.5 roughly
    plt.axvline(10.0, color="red", linestyle="--", label="Offset (10.0)")
    plt.title("DDPM 1D Samples")
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

