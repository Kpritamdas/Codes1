# -*- coding: utf-8 -*-
"""
train_improved.py
--------------------
UPDATED: channel-selection ratios are now searched by GA DURING
training, using the TRAINING LOSS itself as the fitness function --
NOT via a separate post-hoc PSNR/SSIM search after training finishes
(that was the old ablation_B_ga_no_finetune.py design).

Every GA_EVERY_N_EPOCHS epochs, training pauses briefly, runs a GA
search over the three ratios (fitness = negative training loss on a
handful of freshly-sampled batches), applies the best ratios found,
and resumes weight training. Ratios and backbone weights are therefore
co-adapted across ONE training run, rather than trained separately.

BUG FIX from the previous version of this file: it previously imported
SRDatasetTrueScale, AblationHybridLoss, train_loop, evaluate_model, etc.
from `sr_standard` -- those all actually live in `sr_common`; sr_standard
only defines SRNetStandard and its building blocks (no ga_ratios support
at all, so in-training GA search has nothing to optimize on that class).
MODEL_FACTORY now defaults to the GA-capable SRNet (from sr_common).
Use SRNetStandard only for the separate "Standard, no masking" baseline
arm, which is not compatible with this GA-search training loop.

IMPORTANT for fair comparison: apply the SAME recipe (augmentation,
oversampling, schedule, grad clipping, EMA) to every arm you compare,
exactly as before -- only the ratio-search mechanism has changed here.
"""

import os
import torch
import torch.optim as optim

from sr_common import (
    SRNet, SRDatasetTrueScale, AblationHybridLoss, train_loop_with_ga,
    evaluate_model_tiled, evaluate_model_self_ensemble_tiled,
    get_device, ModelEMA, save_results,
)

# ----------------------------- CONFIG -----------------------------
TRAIN_DIR = r"F:\DAS D DRIVE\verydeepsisr\DIV2K_train_HR"
VAL_DIR = r"F:\DAS D DRIVE\verydeepsisr\Set5"           # small, fast validation set
TEST_DIR = r"F:\DAS D DRIVE\verydeepsisr\Urban_100"     # final held-out test set
CKPT_DIR = r"F:\DAS D DRIVE\verydeepsisr\Ga_Models"
LAST_CKPT_PATH = os.path.join(CKPT_DIR, "improved_last_S3_inloop_ga.pth")
BEST_CKPT_PATH = os.path.join(CKPT_DIR, "improved_best_val_psnr_S3_inloop_ga.pth")
EMA_CKPT_PATH = os.path.join(CKPT_DIR, "improved_ema_S3_inloop_ga.pth")

SCALE = 3
PATCH_SIZE = 252
BATCH_SIZE = 4
NUM_EPOCHS = 500
LR = 1e-4
NUM_WORKERS = 4
SAMPLES_PER_EPOCH = 4000    # oversampling: virtual epoch size (>> number of training images)
GRAD_CLIP_NORM = 1.0        # set to None to disable
EMA_DECAY = 0.999           # set to None to disable EMA
VAL_EVERY = 5               # check validation PSNR every N epochs
VAL_MAX_EVAL = 5            # Set5 has 5 images — evaluate all of them

# --- In-training GA ratio-search settings ---
GA_EVERY_N_EPOCHS = 5   # run a GA ratio search every N epochs
GA_POP_SIZE = 8         # kept small since this runs many times during training
GA_GENERATIONS = 8
GA_BATCHES_N = 4        # number of freshly-sampled batches used as the fitness set each search
# --------------------------------------------------------------------

MODEL_FACTORY = lambda: SRNet(scale=SCALE)   # GA-capable architecture — required for in-training GA


def main():
    device = get_device()
    os.makedirs(CKPT_DIR, exist_ok=True)

    # --- Augmentation + oversampling on TRAIN only; val/test stay deterministic ---
    train_ds = SRDatasetTrueScale(
        TRAIN_DIR, scale=SCALE, patch_size=PATCH_SIZE,
        augment=True, samples_per_epoch=SAMPLES_PER_EPOCH,
    )
    val_ds = SRDatasetTrueScale(VAL_DIR, scale=SCALE, patch_size=PATCH_SIZE)
    test_ds = SRDatasetTrueScale(TEST_DIR, scale=SCALE, patch_size=PATCH_SIZE)
    print(f"Training images on disk: {len(train_ds.image_files)}, "
          f"virtual samples/epoch: {len(train_ds)}, validation images: {len(val_ds)}")

    dataloader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0))

    model = MODEL_FACTORY().to(device)

    criterion = AblationHybridLoss(
        use_pixel=True, use_percep=True, use_spectral=True, use_edge=True,
        lambda_pixel=1.0, lambda_percep=0.08, lambda_spectral=0.06, lambda_edge=0.04,
        vgg_layer='relu3_3', device=device,
    )
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Cosine annealing: LR smoothly decays from LR to ~0 over training
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    ema = ModelEMA(model, decay=EMA_DECAY) if EMA_DECAY is not None else None

    model, final_ratios = train_loop_with_ga(
        model, dataloader, criterion, optimizer, device, num_epochs=NUM_EPOCHS,
        ga_every_n_epochs=GA_EVERY_N_EPOCHS, ga_pop_size=GA_POP_SIZE,
        ga_generations=GA_GENERATIONS, ga_batches_n=GA_BATCHES_N,
        checkpoint_path=LAST_CKPT_PATH, checkpoint_every=10,
        scheduler=scheduler, grad_clip_norm=GRAD_CLIP_NORM,
        val_dataset=val_ds, val_every=VAL_EVERY,
        best_checkpoint_path=BEST_CKPT_PATH, val_max_eval=VAL_MAX_EVAL,
        ema=ema,
    )
    # model = train_loop(
    #     model, dataloader, criterion, optimizer, device, num_epochs=NUM_EPOCHS,
    #     checkpoint_path=LAST_CKPT_PATH, checkpoint_every=10,
    #     scheduler=scheduler, grad_clip_norm=GRAD_CLIP_NORM,
    #     val_dataset=val_ds, val_every=VAL_EVERY,
    #     best_checkpoint_path=BEST_CKPT_PATH, val_max_eval=VAL_MAX_EVAL,
    #     ema=ema,
    # )
    print(f"Training complete.")
    print(f"  Last-epoch checkpoint           : {LAST_CKPT_PATH}")
    print(f"  Best-validation-PSNR checkpoint : {BEST_CKPT_PATH}")

# Save EMA checkpoint IMMEDIATELY — don't let it depend on evaluation succeeding
    if ema is not None:
        torch.save({'model_state_dict': ema.state_dict()}, EMA_CKPT_PATH)
        print(f"EMA checkpoint saved -> {EMA_CKPT_PATH}")

# --- Final comparison on the held-out TEST set: raw vs EMA vs self-ensemble ---
    print("\n" + "=" * 60)
    print("Final test-set comparison (Urban100) — same model, three ways to evaluate it")
    print("=" * 60)

    raw_psnr, raw_ssim = evaluate_model_tiled(model, test_ds, device, tile_size=64, tile_overlap=16,
                              scale=SCALE,max_eval=None)
    print(f"Raw final weights        : PSNR={raw_psnr:.2f} dB, SSIM={raw_ssim:.4f}")

    results = {"raw_psnr": raw_psnr, "raw_ssim": raw_ssim}

    if ema is not None:
        ema_model = MODEL_FACTORY().to(device)
        ema.copy_to(ema_model)
        ema_psnr, ema_ssim = evaluate_model_tiled(ema_model, test_ds, device, tile_size=64, tile_overlap=16,
                                  scale=SCALE,max_eval=None)
        print(f"EMA weights               : PSNR={ema_psnr:.2f} dB, SSIM={ema_ssim:.4f}")
        results.update({"ema_psnr": ema_psnr, "ema_ssim": ema_ssim})

        se_psnr, se_ssim = evaluate_model_self_ensemble_tiled(ema_model, test_ds, device, tile_size=64, tile_overlap=16,
                                  scale=SCALE,max_eval=None)
        print(f"EMA + self-ensemble (x8)  : PSNR={se_psnr:.2f} dB, SSIM={se_ssim:.4f}")
        results.update({"self_ensemble_psnr": se_psnr, "self_ensemble_ssim": se_ssim})
        final_psnr, final_ssim = se_psnr, se_ssim
    else:
        se_psnr, se_ssim = evaluate_model_self_ensemble_tiled(model, test_ds, device, tile_size=64, tile_overlap=16,
                                  scale=SCALE,max_eval=None)
        print(f"Raw + self-ensemble (x8)  : PSNR={se_psnr:.2f} dB, SSIM={se_ssim:.4f}")
        results.update({"self_ensemble_psnr": se_psnr, "self_ensemble_ssim": se_ssim})
        final_psnr, final_ssim = se_psnr, se_ssim

    save_results(
        "results_improved_recipe_inloop_ga.json", arm_name="Improved_Recipe_InLoop_GA",
        ratios=final_ratios, psnr_val=final_psnr, ssim_val=final_ssim,
        extra={
            **results,
            "recipe": {
                "augmentation": True, "samples_per_epoch": SAMPLES_PER_EPOCH,
                "lr_schedule": "cosine", "grad_clip_norm": GRAD_CLIP_NORM,
                "ema_decay": EMA_DECAY, "self_ensemble_eval": True,
                "ga_every_n_epochs": GA_EVERY_N_EPOCHS, "ga_pop_size": GA_POP_SIZE,
                "ga_generations": GA_GENERATIONS, "ga_batches_n": GA_BATCHES_N,
                "ga_fitness": "negative training loss (in-loop)",
            },
        },
    )


if __name__ == "__main__":
    main()
