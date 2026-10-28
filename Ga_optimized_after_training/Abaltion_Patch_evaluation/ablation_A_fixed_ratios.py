# -*- coding: utf-8 -*-
"""
ablation_A_fixed_ratios.py
---------------------------
Arm A: FIXED channel-selection ratios.

Loads the shared base checkpoint, leaves ga_ratios at the default
uniform split (0.34 freq / 0.33 edge / 0.33 other), and evaluates
PSNR/SSIM directly. No GA search, no fine-tuning.

This is the control arm the other two are measured against.
"""

import torch

from sr_common import SRNet, SRDatasetTrueScale, set_model_ga_ratios, evaluate_model, save_results, get_device

# ----------------------------- CONFIG -----------------------------
CKPT_PATH = r"F:\DAS D DRIVE\verydeepsisr\Ga_Models\improved_best_val_psnr_s4.pth"
#CKPT_PATH = r"F:\DAS D DRIVE\verydeepsisr\Abalation_trained\base_model_fixed_ratios_256_s4.pth"
TEST_DIR = r"F:\DAS D DRIVE\verydeepsisr\MANGA109"
SCALE = 4
PATCH_SIZE = 64
RESULTS_PATH = "results_A_fixed.json"

FIXED_RATIOS = [0.34, 0.33, 0.33]   # explicit, matches the buffer default
# --------------------------------------------------------------------


def main():
    device = get_device()

    model = SRNet(scale=SCALE).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    set_model_ga_ratios(model, FIXED_RATIOS)

    dataset_eval = SRDatasetTrueScale(TEST_DIR, scale=SCALE, patch_size=PATCH_SIZE)
    print(f"Eval images: {len(dataset_eval)}")

    avg_psnr, avg_ssim = evaluate_model(model, dataset_eval, device)

    save_results(
        RESULTS_PATH, arm_name="A_fixed_ratios",
        ratios=FIXED_RATIOS, psnr_val=avg_psnr, ssim_val=avg_ssim,
    )


if __name__ == "__main__":
    main()
