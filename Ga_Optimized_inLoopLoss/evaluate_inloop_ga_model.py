# -*- coding: utf-8 -*-
"""
evaluate_inloop_ga_model.py
------------------------------
Evaluates a checkpoint produced by train_improved.py's in-loop GA
training (ratios co-adapted with backbone weights via loss-based GA
search during training) — using the CORRECT, standard-comparable
protocol established earlier in this project: full test images
(SRDatasetFullImage), tiled inference (forward_tiled) to avoid the
global-average-pooling train/test resolution mismatch, across all
five benchmark datasets.

Reports three variants of the SAME checkpoint, matching what
train_improved.py itself compares at the end of training:
  1. Raw final weights
  2. EMA weights (if an EMA checkpoint is provided)
  3. EMA + self-ensemble (8-way geometric test-time augmentation)

The ga_ratios used are read directly from the checkpoint (saved
automatically by train_loop_with_ga during training) — no manual
ratio entry needed, unlike the old post-hoc Arm B script.
"""

import torch

from sr_common import (
    SRNet, SRDatasetTrueScale, set_model_ga_ratios, get_device,evaluate_model_self_ensemble_tiled,
    evaluate_model_tiled, save_results,
)

# ----------------------------- CONFIG -----------------------------
# Point these at whichever checkpoints train_improved.py produced.
# Set EMA_CKPT_PATH to None to skip the EMA/self-ensemble rows.
LAST_CKPT_PATH = r"F:\DAS D DRIVE\verydeepsisr\Ga_Models\improved_last_S3_inloop_ga.pth"
EMA_CKPT_PATH = r"F:\DAS D DRIVE\verydeepsisr\Ga_Models\improved_ema_S3_inloop_ga.pth"

TEST_DIRS = {
    "Set5": r"F:\DAS D DRIVE\verydeepsisr\Set 5_HR",
    "Set14": r"F:\DAS D DRIVE\verydeepsisr\Set14",
    "BSD100": r"F:\DAS D DRIVE\verydeepsisr\BSD_100_s4",
    "Urban100": r"F:\DAS D DRIVE\verydeepsisr\Urban_100",
    "Manga109": r"F:\DAS D DRIVE\verydeepsisr\MANGA109",
}


# Must match the TRAINING LR resolution: (training patch_size // scale).
SCALE = 3
GA_VAL_PATCH_SIZE = 63 # for scale =3  # patch size used only for the (fast) GA fitness search
#GA_VAL_PATCH_SIZE = 64 
# Must match the TRAINING LR resolution: (training patch_size // scale).
PATCH_SIZE=252
#TILE_SIZE = 84
#TILE_OVERLAP = 4
TILE_SIZE = PATCH_SIZE // SCALE          # 252 // 3 = 84
TILE_OVERLAP = TILE_SIZE // 4            # 21


RESULTS_PATH = "results_inloop_ga_tiled_fullimage.json"
# --------------------------------------------------------------------


def load_model_with_ratios(ckpt_path, device):
    model = SRNet(scale=SCALE).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    if 'ga_ratios' in ckpt and ckpt['ga_ratios'] is not None:
        ratios = ckpt['ga_ratios']
        set_model_ga_ratios(model, ratios)
        print(f"  ga_ratios from checkpoint: {ratios}")
    else:
        ratios = [0.34, 0.33, 0.33]
        print(f"  WARNING: no ga_ratios in checkpoint — falling back to uniform {ratios}")
    return model, ratios


def evaluate_all_datasets(model, use_self_ensemble):
    per_dataset = {}
    for name, path in TEST_DIRS.items():
        #dataset = SRDatasetTrueScale(path, scale=SCALE, patch_size=256)
        dataset = SRDatasetTrueScale(path, scale=SCALE, patch_size=PATCH_SIZE)
        if use_self_ensemble:
            avg_psnr, avg_ssim = evaluate_model_self_ensemble_tiled(
                model, dataset, device, tile_size=TILE_SIZE,
                tile_overlap=TILE_OVERLAP, scale=SCALE,
            )
            # avg_psnr, avg_ssim = evaluate_model_self_ensemble_tiled(
            #     model, dataset, device,
            # )
        else:
            # 
            avg_psnr, avg_ssim = evaluate_model_tiled(
                model, dataset, device, tile_size=TILE_SIZE,
                tile_overlap=TILE_OVERLAP, scale=SCALE,
            )
            
            # avg_psnr, avg_ssim = evaluate_model_self_ensemble_tiled(
            #     model, dataset, device,
            # )
        per_dataset[name] = {"psnr": avg_psnr, "ssim": avg_ssim}
        print(f"    {name:<12}{avg_psnr:<10.3f}{avg_ssim:.4f}")
    mean_psnr = sum(v["psnr"] for v in per_dataset.values()) / len(per_dataset)
    mean_ssim = sum(v["ssim"] for v in per_dataset.values()) / len(per_dataset)
    return per_dataset, mean_psnr, mean_ssim


device = get_device()
all_results = {}

print(f"\n{'='*60}\nRaw final weights — tiled full-image evaluation\n{'='*60}")
model, ratios_raw = load_model_with_ratios(LAST_CKPT_PATH, device)
per_ds_raw, mean_psnr_raw, mean_ssim_raw = evaluate_all_datasets(model, use_self_ensemble=False)
print(f"    {'Mean':<12}{mean_psnr_raw:<10.3f}{mean_ssim_raw:.4f}")
all_results["raw"] = {"per_dataset": per_ds_raw, "mean_psnr": mean_psnr_raw, "mean_ssim": mean_ssim_raw}

if EMA_CKPT_PATH:
    print(f"\n{'='*60}\nEMA weights — tiled full-image evaluation\n{'='*60}")
    ema_model, ratios_ema = load_model_with_ratios(EMA_CKPT_PATH, device)
    per_ds_ema, mean_psnr_ema, mean_ssim_ema = evaluate_all_datasets(ema_model, use_self_ensemble=False)
    print(f"    {'Mean':<12}{mean_psnr_ema:<10.3f}{mean_ssim_ema:.4f}")
    all_results["ema"] = {"per_dataset": per_ds_ema, "mean_psnr": mean_psnr_ema, "mean_ssim": mean_ssim_ema}

    print(f"\n{'='*60}\nEMA + self-ensemble (x8) — tiled full-image evaluation\n{'='*60}")
    per_ds_se, mean_psnr_se, mean_ssim_se = evaluate_all_datasets(ema_model, use_self_ensemble=True)
    print(f"    {'Mean':<12}{mean_psnr_se:<10.3f}{mean_ssim_se:.4f}")
    all_results["ema_self_ensemble"] = {"per_dataset": per_ds_se, "mean_psnr": mean_psnr_se, "mean_ssim": mean_ssim_se}

    final_psnr, final_ssim, final_ratios = mean_psnr_se, mean_ssim_se, ratios_ema
else:
    final_psnr, final_ssim, final_ratios = mean_psnr_raw, mean_ssim_raw, ratios_raw

save_results(
    RESULTS_PATH, arm_name="InLoop_GA_tiled_fullimage",
    ratios=final_ratios, psnr_val=final_psnr, ssim_val=final_ssim,
    extra={**all_results, "tile_size": TILE_SIZE, "tile_overlap": TILE_OVERLAP},
)
