# -*- coding: utf-8 -*-
"""
ablation_C_ga_finetune.py
---------------------------
Arm C: GA-OPTIMIZED ratios + FINE-TUNING.

Loads the same base checkpoint as Arms A/B, runs the same GA search
to find (freq, edge, other) ratios, fixes those ratios on the
ChannelAttention buffers, then fine-tunes the backbone weights for
a number of extra epochs with those ratios held constant, and
finally evaluates PSNR/SSIM.

This isolates the added benefit of letting the backbone adapt to
the GA-selected channel routing, on top of Arm B.
"""

import os
import torch
import torch.optim as optim

from sr_common import (
    SRNet, SRDatasetTrueScale, AblationHybridLoss, set_model_ga_ratios, 
    run_ga, train_loop, evaluate_model, save_results, get_device
)

# ----------------------------- CONFIG -----------------------------
CKPT_PATH = r"F:\DAS D DRIVE\verydeepsisr\Ga_Models\improved_best_val_psnr_s4.pth"
#CKPT_PATH = r"F:\DAS D DRIVE\verydeepsisr\Abalation_trained\base_model_fixed_ratios_256_s4.pth"
TRAIN_DIR = r"F:\DAS D DRIVE\verydeepsisr\DIV2K_train_HR"
TEST_DIR = r"F:\DAS D DRIVE\verydeepsisr\MANGA109"
FT_CKPT_PATH = r"F:\DAS D DRIVE\verydeepsisr\Fine_tuned_models\ga_finetuned_mode_Urban_100_best_val.pth"

SCALE = 4
PATCH_SIZE = 64

BATCH_SIZE = 4
FINETUNE_EPOCHS = 20   # extra epochs, on top of the base training
FINETUNE_LR = 1e-5       # smaller LR than base training, typical for fine-tuning
RESULTS_PATH = "results_C_ga_finetune.json"

GA_POP_SIZE = 15
GA_GENERATIONS = 10
GA_MAX_EVAL = 20
# Same GA cost note as Arm B: try (8, 8, 10) for a faster, coarser search.
NUM_WORKERS = 4          # parallel data loading; set to 0 on Windows if you hit issues
# --------------------------------------------------------------------


def main():
    device = get_device()

    model = SRNet(scale=SCALE).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    dataset_eval = SRDatasetTrueScale(TRAIN_DIR, scale=SCALE, patch_size=PATCH_SIZE)
    print(f"Eval images: {len(dataset_eval)}")

    # --- GA search for best ratios (identical procedure to Arm B) ---
    best_ratios, best_fitness = run_ga(
        model, dataset_eval, device,
        pop_size=GA_POP_SIZE, generations=GA_GENERATIONS, max_eval=GA_MAX_EVAL,
    )
    set_model_ga_ratios(model, best_ratios)

    # --- Fine-tune the backbone with ratios held fixed ---
    for p in model.parameters():
        p.requires_grad = True

    dataset_train = SRDatasetTrueScale(TRAIN_DIR, scale=SCALE, patch_size=PATCH_SIZE)
    dataloader = torch.utils.data.DataLoader(
        dataset_train, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0))

    criterion = AblationHybridLoss(
        use_pixel=True, use_percep=True, use_spectral=True, use_edge=True,
        lambda_pixel=1.0, lambda_percep=0.08, lambda_spectral=0.06, lambda_edge=0.04,
        vgg_layer='relu3_3', device=device,
    )
    optimizer = optim.Adam(model.parameters(), lr=FINETUNE_LR)

    # ga_ratios is a buffer, not touched by backprop — still holds best_ratios
    # throughout. Checkpoint every 5 epochs so progress isn't lost on a long run.
    model = train_loop(
        model, dataloader, criterion, optimizer, device, num_epochs=FINETUNE_EPOCHS,
        checkpoint_path=FT_CKPT_PATH, checkpoint_every=5,
        extra_state={'ga_ratios': best_ratios},
    )
    print(f"Fine-tuned model saved to {FT_CKPT_PATH}")

    # --- Evaluate after fine-tuning ---
    dataset_eval_test = SRDatasetTrueScale(TEST_DIR, scale=SCALE, patch_size=PATCH_SIZE)
    avg_psnr, avg_ssim = evaluate_model(model, dataset_eval_test, device)

    save_results(
        RESULTS_PATH, arm_name="C_ga_finetune",
        ratios=best_ratios, psnr_val=avg_psnr, ssim_val=avg_ssim,
        extra={"ga_fitness": best_fitness, "finetune_epochs": FINETUNE_EPOCHS},
    )


if __name__ == "__main__":
    main()
