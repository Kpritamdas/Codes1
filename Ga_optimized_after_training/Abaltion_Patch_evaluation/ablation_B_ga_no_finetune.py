# -*- coding: utf-8 -*-
"""
ablation_B_ga_no_finetune.py
------------------------------
Arm B: GA-OPTIMIZED ratios, NO fine-tuning afterward.

Loads the same base checkpoint as Arm A, runs the genetic algorithm
on the validation set to search for better (freq, edge, other)
channel-selection ratios, applies the best ratios found, and
evaluates PSNR/SSIM immediately — the backbone weights are never
updated after this point.

This isolates the effect of GA-searched ratios alone, decoupled
from any extra training.
"""

import torch

from sr_common import (
    SRNet, SRDatasetTrueScale, set_model_ga_ratios, run_ga, evaluate_model, save_results, get_device
)

# ----------------------------- CONFIG -----------------------------
CKPT_PATH = r"F:\DAS D DRIVE\verydeepsisr\Ga_Models\improved_best_val_psnr_S4.pth"
#CKPT_PATH = r"F:\DAS D DRIVE\verydeepsisr\Abalation_trained\base_model_fixed_ratios_256_s4.pth"
TEST_DIR = r"F:\DAS D DRIVE\verydeepsisr\DIV2K_valid_HR"
TEST_DIRS = {
    "Set5": r"F:\DAS D DRIVE\verydeepsisr\Set 5_HR",
    "Set14": r"F:\DAS D DRIVE\verydeepsisr\Set14",
    "BSD100": r"F:\DAS D DRIVE\verydeepsisr\BSD_100_s4",
    "Urban100": r"F:\DAS D DRIVE\verydeepsisr\Urban_100",
    "Manga109": r"F:\DAS D DRIVE\verydeepsisr\MANGA109",
}
SCALE = 4
PATCH_SIZE = 64
RESULTS_PATH = "results_B_ga_no_finetune.json"

GA_POP_SIZE = 12
GA_GENERATIONS = 15
GA_MAX_EVAL = 20     # images used per fitness evaluation (speed vs. accuracy)
# GA cost = pop_size * generations * max_eval forward passes.
# For a quick laptop-GPU run, try GA_POP_SIZE=8, GA_GENERATIONS=8, GA_MAX_EVAL=10
# (192 forward passes instead of 3600) — coarser search, but much faster.
# --------------------------------------------------------------------


def main():
    device = get_device()

    model = SRNet(scale=SCALE).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    dataset_eval = SRDatasetTrueScale(TEST_DIR, scale=SCALE, patch_size=PATCH_SIZE)
    print(f"Eval images: {len(dataset_eval)}")

    # --- GA search for best ratios (uses model in eval mode internally) ---
    best_ratios, best_fitness = run_ga(
        model, dataset_eval, device,
        pop_size=GA_POP_SIZE, generations=GA_GENERATIONS, max_eval=GA_MAX_EVAL,
    )
    set_model_ga_ratios(model, best_ratios)
    
    results = {}
    for name, path in TEST_DIRS.items():
        dataset_eval = SRDatasetTrueScale(path, scale=SCALE, patch_size=PATCH_SIZE)
        avg_psnr, avg_ssim = evaluate_model(model, dataset_eval, device)
        results[name] = (avg_psnr, avg_ssim)
        print(f"{name:<12}{avg_psnr:<20.2f}{avg_ssim:.4f}")

    # --- Evaluate immediately, no fine-tuning ---
    # avg_psnr, avg_ssim = evaluate_model(model, dataset_eval, device)

    save_results(
        RESULTS_PATH, arm_name="B_ga_no_finetune",
        ratios=best_ratios, psnr_val=avg_psnr, ssim_val=avg_ssim,
        extra={"ga_fitness": best_fitness},
    )


if __name__ == "__main__":
    main()
