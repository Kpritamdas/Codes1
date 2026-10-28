# -*- coding: utf-8 -*-
"""
evaluate_single_image.py
---------------------------
Runs the trained SRNet on ONE image and shows/saves:
  1. Original image (input, cropped to a multiple of `SCALE`)
  2. Bicubic baseline (naive upscale — the "no model" reference)
  3. Super-resolved output (the model's prediction)

PSNR/SSIM are computed the same way as in ablation evaluation: on the
Y (luma) channel, against a bicubic-upsampled version of the true Y
channel — consistent with how SRDataset builds targets during training
and evaluation.

The model itself only predicts the Y (luma) channel (see SRNet.head =
Conv2d(1, ...)), so the Cb/Cr chroma channels are bicubic-upsampled and
recombined with the model's Y output to produce a full-color image for
display and saving.

Works with ANY checkpoint produced by the ablation scripts:
  - base_model_fixed_ratios.pth        (Arm A — use FIXED_RATIOS)
  - results from Arm B (GA, no fine-tune) — set GA_RATIOS manually
    to whatever run_ga() printed, since Arm B doesn't save a checkpoint
    with new weights (only the base weights + a results json)
  - ga_finetuned_model.pth             (Arm C — ga_ratios are stored
    INSIDE the checkpoint and loaded automatically)
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from sr_common import (
    SRNet, set_model_ga_ratios, get_device,
    load_and_upscale_color, y_tensor_to_color_image,
)

# ----------------------------- CONFIG -----------------------------
IMAGE_PATH = r"F:\DAS D DRIVE\verydeepsisr\test\YumeiroCooking.png"   # <-- set your image
CKPT_PATH = r"F:\DAS D DRIVE\verydeepsisr\Ga_Models\improved_ema_s4.pth"
SCALE = 4

# Used only if the checkpoint does NOT already contain 'ga_ratios'
# (the base/Arm-A checkpoint doesn't; the Arm-C fine-tuned one does).
FIXED_RATIOS = [0.34, 0.33, 0.33]
#FIXED_RATIOS = [0.4063292168051492, 0.2452731728245415, 0.3483976103703092]
# If evaluating Arm B (GA ratios, no fine-tune), paste the best_ratios
# that run_ga() printed for your run here, and it will override the
# checkpoint / FIXED_RATIOS default:
GA_RATIOS_OVERRIDE = None   # e.g. [0.41, 0.35, 0.24]
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "single_image_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
#OUTPUT_DIR = "single_image_results"
# --------------------------------------------------------------------


def main():
    device = get_device()
    model = SRNet(scale=SCALE).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    #Ratio precedence: explicit override > ratios stored in checkpoint > fixed default
    if GA_RATIOS_OVERRIDE is not None:
        ratios_used = GA_RATIOS_OVERRIDE
        print(f"Using GA_RATIOS_OVERRIDE: {ratios_used}")
    elif 'ga_ratios' in ckpt:
        ratios_used = ckpt['ga_ratios']
        print(f"Using ga_ratios stored in checkpoint: {ratios_used}")
    else:
        ratios_used = FIXED_RATIOS
        print(f"No ga_ratios in checkpoint — using FIXED_RATIOS: {ratios_used}")
    set_model_ga_ratios(model, ratios_used)

    # --- Build input/reference from the image ---
    data = load_and_upscale_color(IMAGE_PATH, SCALE)
    lr_up_tensor = data['lr'].to(device)
    hr_y_up_tensor = data['hr'].to(device)

    # --- Run the model ---
    with torch.no_grad():
        sr_y_tensor = model(lr_up_tensor)

    
# --- Metrics: Y channel, original HR resolution ---
    sr_y_np = sr_y_tensor.squeeze().detach().cpu().numpy()
    hr_y_np = hr_y_up_tensor.squeeze().detach().cpu().numpy()

    bicubic_y_np = data["bicubic_y"].squeeze().cpu().numpy()

    # Verify dimensions before calculating metrics
    print("HR shape      :", hr_y_np.shape)
    print("SR shape      :", sr_y_np.shape)
    print("Bicubic shape :", bicubic_y_np.shape)

    assert hr_y_np.shape == sr_y_np.shape, \
    "HR and SR dimensions do not match."

    assert hr_y_np.shape == bicubic_y_np.shape, \
    "HR and bicubic dimensions do not match."

    sr_psnr = psnr(hr_y_np, sr_y_np, data_range=1.0)
    sr_ssim = ssim(hr_y_np, sr_y_np, data_range=1.0)

    bicubic_psnr = psnr(hr_y_np, bicubic_y_np, data_range=1.0)
    bicubic_ssim = ssim(hr_y_np, bicubic_y_np, data_range=1.0)


    print(f"Bicubic baseline : PSNR={bicubic_psnr:.2f} dB, SSIM={bicubic_ssim:.4f}")
    print(f"Model (SR)       : PSNR={sr_psnr:.2f} dB, SSIM={sr_ssim:.4f}")

    # --- Recombine model's Y output with chroma for a full-color image ---
    sr_color = y_tensor_to_color_image(sr_y_tensor, data['cb_up'], data['cr_up'])

    # --- Save individual images ---
    orig_path = OUTPUT_DIR / "original.png"
    bicubic_path = OUTPUT_DIR / "bicubic_baseline.png"
    sr_path = OUTPUT_DIR / "super_resolved.png"

    print("Output directory:", OUTPUT_DIR)
    print("Super-resolved output path:", sr_path)

    data["original_rgb"].save(str(orig_path), format="PNG")
    data["bicubic_baseline_rgb"].save(str(bicubic_path), format="PNG")

    # Verify that the super-resolved image is a valid PIL image
    print("SR image type:", type(sr_color))
    print("SR image mode:", sr_color.mode)
    print("SR image size:", sr_color.size)

    sr_color.save(str(sr_path), format="PNG")

    print("All images saved successfully:")
    print(orig_path)
    print(bicubic_path)
    print(sr_path)
    # --- Side-by-side comparison figure ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    axes[0].imshow(data['original_rgb'])
    axes[0].set_title(f"Original\n{data['original_rgb'].size[0]}x{data['original_rgb'].size[1]}")
    axes[0].axis('off')

    axes[1].imshow(data['bicubic_baseline_rgb'])
    axes[1].set_title(f"Bicubic baseline\nPSNR={bicubic_psnr:.2f} dB, SSIM={bicubic_ssim:.4f}")
    axes[1].axis('off')

    axes[2].imshow(sr_color)
    axes[2].set_title(f"Super-Resolved (model)\nPSNR={sr_psnr:.2f} dB, SSIM={sr_ssim:.4f}")
    axes[2].axis('off')

    fig.suptitle(f"Single-image evaluation — {os.path.basename(IMAGE_PATH)} (x{SCALE})",
                 fontsize=13, weight='bold')
    plt.tight_layout()
    comparison_path = os.path.join(OUTPUT_DIR, "comparison.png")
    plt.savefig(comparison_path, dpi=200, bbox_inches='tight')
    print(f"Saved comparison figure -> {comparison_path}")
    plt.show()


if __name__ == "__main__":
    main()
