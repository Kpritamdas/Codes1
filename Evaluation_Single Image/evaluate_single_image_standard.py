# -*- coding: utf-8 -*-
"""
evaluate_single_image.py
---------------------------
Runs a trained model on ONE image and saves/shows:
  1. Original (HR ground truth, cropped to a multiple of SCALE)
  2. Bicubic baseline (LR image bicubically upsampled back to HR size)
  3. Super-resolved output (model prediction)

Protocol (matches the full-image evaluation used for the paper's tables):
  HR -> bicubic downsample -> LR -> model (tiled inference) -> SR at HR size.
PSNR/SSIM are computed on the Y (luma) channel, all at the HR resolution.

The model predicts luma only. For display, Cb/Cr come from the ORIGINAL
image (same chroma for baseline and SR), so the comparison is about luma
detail; say so in the figure caption.

Set USE_GA_MODEL = True to evaluate the GA-capable SRNet (sr_common) with
the ratios stored in its checkpoint; False evaluates SRNetStandard.
"""

from pathlib import Path
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from sr_standard import (
    SRNetStandard, get_device,
    load_and_upscale_color, y_tensor_to_color_image,
)

# ----------------------------- CONFIG -----------------------------
IMAGE_PATH = r"F:\DAS D DRIVE\verydeepsisr\test\img_073_SRF_4_HR.png"
CKPT_PATH = r"F:\DAS D DRIVE\verydeepsisr\Ga_Models\improved_ema_S4_standard.pth"
SCALE = 4
USE_GA_MODEL = False                 # True -> SRNet + ga_ratios from checkpoint

# Tiled inference: tile size must equal the LR size used in training
# (training patch_size // scale), otherwise the channel-attention pooling
# sees a different spatial extent than it was trained on.
USE_TILING = True
PATCH_SIZE = 256                     # training patch size of this checkpoint
TILE_SIZE = PATCH_SIZE // SCALE      # 64
TILE_OVERLAP = TILE_SIZE // 4        # 16

# SSIM settings: use the SAME settings as the evaluation scripts / tables.
SSIM_KWARGS = dict(gaussian_weights=True, sigma=1.5,
                   use_sample_covariance=False, win_size=11)

OUTPUT_DIR = Path(__file__).resolve().parent / "single_image_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# --------------------------------------------------------------------


def forward_tiled(model, lr, tile_size, tile_overlap, scale):
    """Overlapping-tile inference with linear-ramp blending (fp32).
    lr: [1,1,h,w] on the model's device. Returns [1,1,h*scale,w*scale]."""
    device = lr.device
    _, C, H, W = lr.shape
    canvas = torch.zeros(1, C, H * scale, W * scale, device=device)
    weight = torch.zeros(1, 1, H * scale, W * scale, device=device)
    stride = max(tile_size - tile_overlap, 1)

    def starts(total):
        if total <= tile_size:
            return [0]
        s = list(range(0, total - tile_size + 1, stride))
        if s[-1] != total - tile_size:
            s.append(total - tile_size)
        return sorted(set(s))

    ramp = torch.linspace(0, 1, tile_overlap, device=device) if tile_overlap > 0 else None
    with torch.no_grad():
        for y0 in starts(H):
            for x0 in starts(W):
                h, w = min(tile_size, H - y0), min(tile_size, W - x0)
                out = model(lr[:, :, y0:y0 + h, x0:x0 + w]).float()
                wy, wx = torch.ones(h, device=device), torch.ones(w, device=device)
                if ramp is not None:
                    if y0 > 0:        wy[:tile_overlap] = ramp
                    if y0 + h < H:    wy[-tile_overlap:] = torch.flip(ramp, [0])
                    if x0 > 0:        wx[:tile_overlap] = ramp
                    if x0 + w < W:    wx[-tile_overlap:] = torch.flip(ramp, [0])
                win = torch.outer(wy, wx).repeat_interleave(scale, 0).repeat_interleave(scale, 1)
                oy, ox = y0 * scale, x0 * scale
                canvas[:, :, oy:oy + h * scale, ox:ox + w * scale] += out * win
                weight[:, :, oy:oy + h * scale, ox:ox + w * scale] += win
    return torch.clamp(canvas / weight.clamp_min(1e-8), 0.0, 1.0)


def main():
    device = get_device()

    ckpt = torch.load(CKPT_PATH, map_location=device)
    if USE_GA_MODEL:
        from sr_common import SRNet, set_model_ga_ratios
        model = SRNet(scale=SCALE).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        ratios = ckpt.get('ga_ratios') or [0.34, 0.33, 0.33]
        set_model_ga_ratios(model, ratios)
        print(f"GA model, ratios: {ratios}")
    else:
        model = SRNetStandard(scale=SCALE).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # --- Inputs / references (all Y-channel tensors; see load_and_upscale_color) ---
    data = load_and_upscale_color(IMAGE_PATH, SCALE)
    lr = data['lr'].to(device)

    with torch.no_grad():
        if USE_TILING:
            sr_y_tensor = forward_tiled(model, lr, TILE_SIZE, TILE_OVERLAP, SCALE)
        else:
            sr_y_tensor = model(lr)

    sr_y_np = sr_y_tensor.squeeze().detach().cpu().numpy()
    hr_y_np = data['hr'].squeeze().numpy()
    bicubic_y_np = data['bicubic_y'].squeeze().numpy()

    print("HR / SR / bicubic shapes:", hr_y_np.shape, sr_y_np.shape, bicubic_y_np.shape)
    assert hr_y_np.shape == sr_y_np.shape == bicubic_y_np.shape, "size mismatch"

    sr_psnr = psnr(hr_y_np, sr_y_np, data_range=1.0)
    sr_ssim = ssim(hr_y_np, sr_y_np, data_range=1.0, **SSIM_KWARGS)
    bic_psnr = psnr(hr_y_np, bicubic_y_np, data_range=1.0)
    bic_ssim = ssim(hr_y_np, bicubic_y_np, data_range=1.0, **SSIM_KWARGS)
    print(f"Bicubic baseline : PSNR={bic_psnr:.2f} dB, SSIM={bic_ssim:.4f}")
    print(f"Model (SR)       : PSNR={sr_psnr:.2f} dB, SSIM={sr_ssim:.4f}")

    sr_color = y_tensor_to_color_image(sr_y_tensor, data['cb_up'], data['cr_up'])

    # --- Save images ---
    data['original_rgb'].save(OUTPUT_DIR / "original.png")
    data['bicubic_baseline_rgb'].save(OUTPUT_DIR / "bicubic_baseline.png")
    sr_color.save(OUTPUT_DIR / "super_resolved.png")
    print("Saved images to", OUTPUT_DIR)

    # --- Side-by-side figure ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    panels = [
        (data['original_rgb'], f"Original (HR)\n{data['original_rgb'].size[0]}x{data['original_rgb'].size[1]}"),
        (data['bicubic_baseline_rgb'], f"Bicubic\nPSNR={bic_psnr:.2f} dB, SSIM={bic_ssim:.4f}"),
        (sr_color, f"Super-resolved\nPSNR={sr_psnr:.2f} dB, SSIM={sr_ssim:.4f}"),
    ]
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img); ax.set_title(title); ax.axis('off')
    fig.suptitle(f"{os.path.basename(IMAGE_PATH)} (x{SCALE})", fontsize=13, weight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "comparison.png", dpi=200, bbox_inches='tight')
    print("Saved comparison figure ->", OUTPUT_DIR / "comparison.png")
    plt.show()


if __name__ == "__main__":
    main()
