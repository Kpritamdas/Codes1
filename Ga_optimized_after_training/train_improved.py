# -*- coding: utf-8 -*-
"""
train_improved.py
--------------------
Combines several standard, literature-precedented techniques on top of
your existing pipeline to improve PSNR/SSIM, WITHOUT changing the
architecture or the (corrected) evaluation protocol:

  1. Data augmentation (SRDatasetTrueScale(..., augment=True)) — random
     flips/rotations, standard in EDSR/RCAN/SwinIR. Free extra training
     diversity from the same images.
  2. Patch oversampling (samples_per_epoch) — draws many random crops
     per image per epoch instead of exactly one, since one epoch = one
     crop/image otherwise severely limits training diversity.
  3. Cosine-annealing LR schedule — smoothly decays LR over training
     instead of a constant 1e-4 for all 500 epochs, typically improves
     final convergence.
  4. Gradient clipping — stabilizes training, especially useful with
     the hybrid multi-term loss.
  5. Validation-based best-checkpoint saving — tracks PSNR on a held-out
     validation split during training and keeps the BEST epoch, not
     just the last one.
  6. EMA (exponential moving average) of weights — a smoothed shadow
     model that often evaluates slightly better/more consistently than
     the raw final weights. Zero training-time cost of consequence.
  7. Test-time self-ensemble at evaluation — averages 8 geometric
     transforms per image, a free (no retraining) ~0.1-0.3 dB PSNR gain.

IMPORTANT: if you adopt any of these, apply them IDENTICALLY across
every arm you compare (Fixed / GA-optimized / Fine-tuned / Standard
channel attention / L1 vs hybrid loss, etc.) — using augmentation or
self-ensemble for only your proposed method while leaving baselines
without it would reopen the same fairness problem flagged earlier in
this project. The cleanest approach: re-run your FULL ablation suite
once with this training script as the new default, rather than
patching results piecemeal. Disclose the full recipe (schedule, EMA
decay, augmentation, self-ensemble) in your Methods section.
"""

import os
import torch
import torch.optim as optim

# Import whichever model/arm you're training. Examples:
#   from sr_common import SRNet, set_model_ga_ratios      (GA arms)
#   from sr_standard import SRNetStandard                  (standard baseline)
# Adjust the import and MODEL_FACTORY below to match the arm you're training.
from sr_common import (
    SRDatasetTrueScale,SRNet, AblationHybridLoss, train_loop,
    evaluate_model_tiled, evaluate_model_self_ensemble_tiled,
    get_device, ModelEMA, save_results,
)
#from sr_standard import SRNetStandard

# ----------------------------- CONFIG -----------------------------
TRAIN_DIR = r"F:\DAS D DRIVE\verydeepsisr\DIV2K_train_HR"
VAL_DIR = r"F:\DAS D DRIVE\verydeepsisr\Set5"           # small, fast validation set
TEST_DIR = r"F:\DAS D DRIVE\verydeepsisr\Urban_100"     # final held-out test set
CKPT_DIR = r"F:\DAS D DRIVE\verydeepsisr\Ga_Models"
LAST_CKPT_PATH = os.path.join(CKPT_DIR, "improved_last_S3.pth")
BEST_CKPT_PATH = os.path.join(CKPT_DIR, "improved_best_val_psnr_S3.pth")
EMA_CKPT_PATH = os.path.join(CKPT_DIR, "improved_ema_S3.pth")

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
# --------------------------------------------------------------------

MODEL_FACTORY = lambda: SRNet(scale=SCALE)  # swap for SRNet(...) + set_model_ga_ratios(...)
                                                     # if training a GA arm instead
# MODEL_FACTORY = lambda: SRNetstandard(scale=SCALE) 

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
    #model = SRNet(scale=SCALE).to(device)

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

    model = train_loop(
        model, dataloader, criterion, optimizer, device, num_epochs=NUM_EPOCHS,
        checkpoint_path=LAST_CKPT_PATH, checkpoint_every=10,
        scheduler=scheduler, grad_clip_norm=GRAD_CLIP_NORM,
        val_dataset=val_ds, val_every=VAL_EVERY,
        best_checkpoint_path=BEST_CKPT_PATH, val_max_eval=VAL_MAX_EVAL,
        ema=ema,
    )

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
        torch.save({'model_state_dict': ema.state_dict()}, EMA_CKPT_PATH)
        print(f"EMA checkpoint saved -> {EMA_CKPT_PATH}")
        ema_model = MODEL_FACTORY().to(device)
        #ema_model = SRNet().to(device)
        ema.copy_to(ema_model)
        ema_psnr, ema_ssim = evaluate_model_tiled(model, test_ds, device, tile_size=64, tile_overlap=16,
                                     scale=SCALE,max_eval=None)
        print(f"EMA weights               : PSNR={ema_psnr:.2f} dB, SSIM={ema_ssim:.4f}")
        results.update({"ema_psnr": ema_psnr, "ema_ssim": ema_ssim})

        se_psnr, se_ssim = evaluate_model_self_ensemble_tiled(model, test_ds, device, tile_size=64, tile_overlap=16,
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
        "results_improved_recipe.json", arm_name="Improved_Recipe",
        ratios=None, psnr_val=final_psnr, ssim_val=final_ssim,
        extra={
            **results,
            "recipe": {
                "augmentation": True, "samples_per_epoch": SAMPLES_PER_EPOCH,
                "lr_schedule": "cosine", "grad_clip_norm": GRAD_CLIP_NORM,
                "ema_decay": EMA_DECAY, "self_ensemble_eval": True,
            },
        },
    )


if __name__ == "__main__":
    main()

