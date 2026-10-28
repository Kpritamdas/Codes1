# -*- coding: utf-8 -*-
"""
ablation_B_ga_tiled_fullimage.py
------------------------------------
Arm B: GA-OPTIMIZED ratios, NO fine-tuning — evaluated via TILED,
FULL-IMAGE inference.

Loads the same base checkpoint as Arm A, runs the genetic algorithm to
search for (freq, edge, other) ratios — the GA SEARCH itself still uses
fast, patch-based fitness evaluation (SRDatasetTrueScale on a small
validation subset), exactly as before, since running full tiled
inference for every one of pop_size*generations*max_eval candidate
chromosomes would be far too slow. Only the FINAL reported PSNR/SSIM,
once the best ratios are fixed, uses tiled full-image evaluation.

Backbone weights are never updated — this isolates the effect of
GA-searched ratios alone.
"""

import torch

from sr_common import (
    SRNet, SRDatasetFullImage, set_model_ga_ratios, get_device,evaluate_model_self_ensemble_tiled,
    evaluate_model_tiled, save_results,SRDatasetTrueScale, run_ga,
)

# ----------------------------- CONFIG -----------------------------
#CKPT_PATH = r"F:\DAS D DRIVE\verydeepsisr\Ga_Models\improved_best_val_psnr_s4.pth"
LAST_CKPT_PATH = r"F:\DAS D DRIVE\verydeepsisr\Ga_Models\improved_last_s3.pth"
EMA_CKPT_PATH = r"F:\DAS D DRIVE\verydeepsisr\Ga_Models\improved_ema_S3.pth"
VAL_DIR = r"F:\DAS D DRIVE\verydeepsisr\DIV2K_train_HR"   # small, fast set for the GA fitness search
TEST_DIRS = {
    "Set5": r"F:\DAS D DRIVE\verydeepsisr\Set 5_HR",
    "Set14": r"F:\DAS D DRIVE\verydeepsisr\Set14",
    "BSD100": r"F:\DAS D DRIVE\verydeepsisr\BSD_100_s4",
    "Urban100": r"F:\DAS D DRIVE\verydeepsisr\Urban_100",
    "Manga109": r"F:\DAS D DRIVE\verydeepsisr\MANGA109",
}
SCALE = 3
GA_VAL_PATCH_SIZE = 63 # for scale =3  # patch size used only for the (fast) GA fitness search
#GA_VAL_PATCH_SIZE = 64 
# Must match the TRAINING LR resolution: (training patch_size // scale).
PATCH_SIZE=252
#TILE_SIZE = 84
#TILE_OVERLAP = 4
TILE_SIZE = PATCH_SIZE // SCALE          # 252 // 3 = 84
TILE_OVERLAP = TILE_SIZE // 4            # 21


GA_POP_SIZE = 12
GA_GENERATIONS = 15
GA_MAX_EVAL = 20
RESULTS_PATH = "results_B_ga_tiled_fullimage.json"
# --------------------------------------------------------------------

def load_model_with_ratios_last(ckpt_path, device):
    model1 = SRNet(scale=SCALE).to(device)
    last_ckpt = torch.load(ckpt_path, map_location=device)
    model1.load_state_dict(last_ckpt['model_state_dict'])
    ga_val_dataset = SRDatasetTrueScale(VAL_DIR, scale=SCALE, patch_size=GA_VAL_PATCH_SIZE)
    best_ratios1, best_fitness1 = run_ga(
        model1, ga_val_dataset, device,
        pop_size=GA_POP_SIZE, generations=GA_GENERATIONS, max_eval=GA_MAX_EVAL,
    )
    set_model_ga_ratios(model1, best_ratios1)
    return model1, best_ratios1
    print(f"\nGA search complete. Best ratios: {best_ratios1}, fitness: {best_fitness1:.3f}")
#load_model_with_ratios_last(LAST_CKPT_PATH)  

def load_model_with_ratios_ema(ckpt_path, device):
    model2 = SRNet(scale=SCALE).to(device)
    ema_ckpt = torch.load(ckpt_path, map_location=device)
    model2.load_state_dict(ema_ckpt['model_state_dict'])
    ga_val_dataset = SRDatasetTrueScale(VAL_DIR, scale=SCALE, patch_size=GA_VAL_PATCH_SIZE)
    best_ratios2, best_fitness2 = run_ga(
        model2, ga_val_dataset, device,
        pop_size=GA_POP_SIZE, generations=GA_GENERATIONS, max_eval=GA_MAX_EVAL,
    )
    set_model_ga_ratios(model2, best_ratios2)
    return model2, best_ratios2
    
   
    print(f"\nGA search complete. Best ratios: {best_ratios2}, fitness: {best_fitness2:.3f}")

    # --- Final reported numbers: tiled, full-image evaluation ---
    print(f"\nArm B (GA-optimized) — tile_size={TILE_SIZE}, overlap={TILE_OVERLAP}\n")
    print(f"{'Dataset':<12}{'PSNR':<10}{'SSIM'}")
    print("-" * 35)
#load_model_with_ratios_ema(EMA_CKPT_PATH)
def evaluate_all_datasets(model, use_self_ensemble):
        per_dataset = {}
        for name, path in TEST_DIRS.items():
            dataset = SRDatasetFullImage(path, scale=SCALE)
            if use_self_ensemble:
                avg_psnr, avg_ssim = evaluate_model_self_ensemble_tiled(
                    model, dataset, device, tile_size=TILE_SIZE,
                    tile_overlap=TILE_OVERLAP, scale=SCALE,
                )
            else:
                avg_psnr, avg_ssim = evaluate_model_tiled(
                model, dataset, device, tile_size=TILE_SIZE,
                tile_overlap=TILE_OVERLAP, scale=SCALE,
                )
            per_dataset[name] = {"psnr": avg_psnr, "ssim": avg_ssim}
            print(f"    {name:<12}{avg_psnr:<10.3f}{avg_ssim:.4f}")                                      
        mean_psnr = sum(v["psnr"] for v in per_dataset.values()) / len(per_dataset)
        mean_ssim = sum(v["ssim"] for v in per_dataset.values()) / len(per_dataset)
        return per_dataset, mean_psnr, mean_ssim


device = get_device()
all_results = {}

print(f"\n{'='*60}\nRaw final weights — tiled full-image evaluation\n{'='*60}")
model, ratios_raw = load_model_with_ratios_last(LAST_CKPT_PATH, device)
per_ds_raw, mean_psnr_raw, mean_ssim_raw = evaluate_all_datasets(model, use_self_ensemble=False)
print(f"    {'Mean':<12}{mean_psnr_raw:<10.3f}{mean_ssim_raw:.4f}")
all_results["raw"] = {"per_dataset": per_ds_raw, "mean_psnr": mean_psnr_raw, "mean_ssim": mean_ssim_raw}

if EMA_CKPT_PATH:
        print(f"\n{'='*60}\nEMA weights — tiled full-image evaluation\n{'='*60}")
        ema_model, ratios_ema = load_model_with_ratios_ema(EMA_CKPT_PATH, device)
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
        RESULTS_PATH, arm_name="Ga_Fitness_validation",
        ratios=final_ratios, psnr_val=final_psnr, ssim_val=final_ssim,
        extra={**all_results, "tile_size": TILE_SIZE, "tile_overlap": TILE_OVERLAP},
)


