# -*- coding: utf-8 -*-
"""
sr_common.py
------------
Shared building blocks for the GA-controlled RCAB ablation study:
  - ChannelAttention (with GA-controlled freq/edge/other channel ratios)
  - RCAB / SRNet
  - SRDataset
  - AblationHybridLoss (pixel + perceptual + spectral + edge)
  - GA search utilities (random_chromosome, ga_fitness, run_ga)
  - train_loop / evaluate_model helpers

All three ablation scripts (A/B/C) import from this file so that the
backbone architecture, dataset pipeline and loss are IDENTICAL across
arms. Only how the channel-selection ratios are set (and whether the
backbone is fine-tuned afterwards) differs between arms — that is what
makes the PSNR/SSIM comparison a valid ablation.
"""

import os
import random
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


# ----------------------------------------------------------------------
# Device / GPU reporting
# ----------------------------------------------------------------------
def count_parameters(model, verbose=True):
    """Returns (total_params, trainable_params). Buffers (e.g. ga_ratios,
    registered via register_buffer) are NOT included — only nn.Parameter
    tensors are counted, which matches how model size is conventionally
    reported in the SR literature (RCAN, EDSR, etc. report parameter
    counts, not buffer counts). This means the GA channel-selection
    mechanism (ga_ratios) adds ZERO parameters to the model — it only
    changes which channels are masked at inference time, not the number
    of learnable weights."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"Total parameters:     {total:,}  ({total/1e6:.3f} M)")
        print(f"Trainable parameters: {trainable:,}  ({trainable/1e6:.3f} M)")
    return total, trainable


def get_device(verbose=True):
    """Picks CUDA if available and prints what it found. Also flips on
    cudnn.benchmark, which speeds up training when input sizes are
    fixed (true here: patch_size is constant)."""
    cuda_ok = torch.cuda.is_available()
    device = torch.device('cuda' if cuda_ok else 'cpu')

    if verbose:
        print("=" * 60)
        if cuda_ok:
            idx = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            total_mem = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
            print(f"CUDA available: YES")
            print(f"GPU: {name}  ({total_mem:.1f} GB total memory)")
            print(f"PyTorch CUDA build: {torch.version.cuda}")
        else:
            print("CUDA available: NO — running on CPU.")
            print("If you have an NVIDIA GPU, check that (1) drivers are installed,")
            print("(2) your PyTorch build is the CUDA version, not CPU-only:")
            print("    https://pytorch.org/get-started/locally/")
        print("=" * 60)

    if cuda_ok:
        torch.backends.cudnn.benchmark = True  # fixed input size -> faster conv algo selection

    return device


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, n_feats, reduction=16):
        super().__init__()
        self.n_feats = n_feats

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(n_feats, n_feats // reduction, 1)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(n_feats // reduction, n_feats, 1)
        self.sigmoid = nn.Sigmoid()

        # GA-controlled ratios (initialized uniform == "fixed ratios" arm)
        self.register_buffer("ga_ratios", torch.tensor([0.34, 0.33, 0.33]))

    def set_ga_ratios(self, ratios):
        """Called by GA (freq, edge, other) or to reset to fixed/uniform."""
        self.ga_ratios[:] = torch.tensor(ratios, device=self.ga_ratios.device)

    def forward(self, x):
        B, C, H, W = x.shape

        # --- Standard channel attention ---
        y = self.pool(x)
        y = self.conv2(self.relu(self.conv1(y)))
        attn = self.sigmoid(y)
        x_attn = x * attn

        # --- Channel selection driven by self.ga_ratios ---
        freq_r, edge_r, other_r = self.ga_ratios.tolist()

        kf = int(freq_r * C)
        ke = int(edge_r * C)
        ko = C - kf - ke

        # Force this FFT to run in fp32 regardless of the outer autocast
        # context: cuFFT's half-precision path only supports power-of-two
        # signal sizes, which real images/tiles are almost never (this
        # breaks even when a configured tile_size IS a power of two,
        # since boundary tiles get clamped to whatever remains at an
        # image edge). Disabling autocast just for this op keeps AMP's
        # speed benefit everywhere else in the network.
        with torch.cuda.amp.autocast(enabled=False):
            fft = torch.fft.fft2(x_attn.float(), norm='ortho')
        freq_score = fft.abs().mean(dim=(2, 3))

        gx = x_attn[:, :, :, 1:] - x_attn[:, :, :, :-1]
        gy = x_attn[:, :, 1:, :] - x_attn[:, :, :-1, :]
        edge_score = gx.abs().mean((2, 3)) + gy.abs().mean((2, 3))

        other_score = torch.ones_like(freq_score)

        idx_f = torch.topk(freq_score, kf, dim=1).indices
        idx_e = torch.topk(edge_score, ke, dim=1).indices
        idx_o = torch.topk(other_score, ko, dim=1).indices

        mask = torch.zeros_like(freq_score)
        mask.scatter_(1, idx_f, 1)
        mask.scatter_(1, idx_e, 1)
        mask.scatter_(1, idx_o, 1)

        mask = mask.unsqueeze(-1).unsqueeze(-1)
        return x_attn * mask


class RCAB(nn.Module):
    def __init__(self, n_feats, res_scale=0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            ChannelAttention(n_feats),
        )
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x) * self.res_scale
        return x + res


class SRNet(nn.Module):
    def __init__(self, scale=8, n_feats=64, n_blocks=16, res_scale=0.1, global_residual=True):
        super().__init__()
        self.scale = scale
        self.global_residual = global_residual
        self.head = nn.Conv2d(1, n_feats, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[RCAB(n_feats, res_scale) for _ in range(n_blocks)])
        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(n_feats, 1, kernel_size=3, padding=1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        feat = self.head(x)
        feat = self.body(feat)
        out = self.tail(feat)
        if self.global_residual:
            up = F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)
            out = out + up
        return torch.clamp(out, 0.0, 1.0)


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
class SRDataset(Dataset):
    def __init__(self, image_dir, scale=4, patch_size=256):
        self.image_dir = image_dir
        self.image_files = [
            os.path.join(image_dir, x) for x in os.listdir(image_dir)
            if x.lower().endswith(('png', 'jpg', 'jpeg'))
        ]
        self.scale = scale
        #self.patch_size = patch_size
        self.patch_size = patch_size - (patch_size % scale)
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert('YCbCr')
        y, _, _ = img.split()  # only Y channel
        w, h = y.size
        new_w, new_h = w - (w % self.scale), h - (h % self.scale)
        y = y.resize((new_w, new_h), Image.BICUBIC)

        if new_w > self.patch_size and new_h > self.patch_size:
            x0 = random.randint(0, new_w - self.patch_size)
            y0 = random.randint(0, new_h - self.patch_size)
            hr = y.crop((x0, y0, x0 + self.patch_size, y0 + self.patch_size))
        else:
            hr = y.resize((self.patch_size, self.patch_size), Image.BICUBIC)

        lr = hr.resize(
            (self.patch_size // self.scale, self.patch_size // self.scale),
            Image.BICUBIC)
        lr_up = lr.resize((self.patch_size, self.patch_size), Image.BICUBIC)
        hr_up = hr.resize(
            (self.patch_size * self.scale, self.patch_size * self.scale),
            Image.BICUBIC)

        return self.to_tensor(lr_up), self.to_tensor(hr_up)


class SRDatasetTrueScale(Dataset):
    """CORRECTED dataset pipeline for genuine end-to-end super-resolution.

    Unlike SRDataset above, this does NOT pre-upsample the LR image back
    to patch size before feeding it to the model. The model receives the
    true low-resolution image directly, and its own PixelShuffle(scale)
    performs the one and only upsampling step, producing output at
    exactly `patch_size` resolution — matching `hr` (the real, cropped
    ground-truth patch) directly. No `hr_up` bicubic stretch is used as
    a target, since `hr` itself already IS the target resolution.

    Concretely, at patch_size=256, scale=4:
        hr  (real ground truth)        : 256 x 256
        lr  (real low-resolution input): 64  x 64   <- fed to model AS-IS
        model(lr) output                : 256 x 256  (via PixelShuffle x4)
                                           -> compared directly to hr

    This makes "scale" mean what it conventionally means in the SR
    literature (a true `scale`x enlargement of real low-resolution
    content), and makes PSNR/SSIM comparable to standard benchmark
    protocols (Set5/Set14/BSD100/Urban100/Manga109 results reported
    elsewhere in the literature).
    """
    def __init__(self, image_dir, scale, patch_size, augment=False, samples_per_epoch=None):
        self.image_dir = image_dir
        self.image_files = [
            os.path.join(image_dir, x) for x in os.listdir(image_dir)
            if x.lower().endswith(('png', 'jpg', 'jpeg'))
        ]
        self.scale = scale
        self.patch_size = patch_size
        #self.patch_size = patch_size - (patch_size % scale)   # 256 -> 255 for scale 3; unchanged for 2, 4, 8
        self.augment = augment  # random flips/90-deg rotations (train only — leave
                                 # False for eval/test datasets so results stay reproducible)
        # Oversampling: your current dataset draws exactly ONE random crop
        # per image per __getitem__ call, so one epoch = one crop per image
        # — very little patch diversity if you have few training images.
        # Setting samples_per_epoch > len(image_files) makes __len__ report
        # a larger virtual epoch size; __getitem__ picks a RANDOM image each
        # call (not index-based), so the same image contributes multiple
        # different random crops per epoch. This is standard practice in
        # EDSR/RCAN-style training (they define an epoch as N iterations,
        # not N images). Leave as None to keep the original 1-crop-per-
        # image-per-epoch behavior.
        self.samples_per_epoch = samples_per_epoch
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return self.samples_per_epoch if self.samples_per_epoch is not None else len(self.image_files)

    def __getitem__(self, idx):
        # When oversampling, ignore the passed idx and pick a random image —
        # this is what actually produces different crops across the larger
        # virtual epoch, rather than just repeating the same len(files) crops.
        file_idx = random.randrange(len(self.image_files)) if self.samples_per_epoch is not None else idx
        img = Image.open(self.image_files[file_idx]).convert('YCbCr')
        y, _, _ = img.split()  # only Y channel
        w, h = y.size
        new_w, new_h = w - (w % self.scale), h - (h % self.scale)
        y = y.resize((new_w, new_h), Image.BICUBIC)

        if new_w > self.patch_size and new_h > self.patch_size:
            x0 = random.randint(0, new_w - self.patch_size)
            y0 = random.randint(0, new_h - self.patch_size)
            hr = y.crop((x0, y0, x0 + self.patch_size, y0 + self.patch_size))
        else:
            hr = y.resize((self.patch_size, self.patch_size), Image.BICUBIC)

        if self.augment:
            # 8 geometric variants: identity, 3 rotations, and their mirror —
            # standard SR augmentation (used in EDSR/RCAN). Applied to `hr`
            # BEFORE downsampling, so lr/hr stay perfectly aligned.
            if random.random() < 0.5:
                hr = hr.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                hr = hr.transpose(Image.FLIP_TOP_BOTTOM)
            k = random.choice([0, 90, 180, 270])
            if k:
                hr = hr.rotate(k, expand=True)
                # rotate(90/270) swaps width/height — re-crop/pad back to a
                # square patch_size x patch_size so shapes stay consistent
                if hr.size != (self.patch_size, self.patch_size):
                    hr = hr.resize((self.patch_size, self.patch_size), Image.BICUBIC)

        # True low-resolution image — this is what the model receives,
        # with NO upsampling back to patch_size.
        lr = hr.resize(
            (self.patch_size // self.scale, self.patch_size // self.scale),
            Image.BICUBIC)

        # Ground truth is the real HR patch itself — no bicubic stretch.
        return self.to_tensor(lr), self.to_tensor(hr)


# ----------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------
class VGGFeatureExtractor(nn.Module):
    def __init__(self, layer_name='relu3_3', use_bn=False, device='cpu'):
        super().__init__()
        from torchvision import models
        vgg = models.vgg19_bn(pretrained=True) if use_bn else models.vgg19(pretrained=True)
        vgg_features = vgg.features
        layer_map = {'relu1_2': 3, 'relu2_2': 8, 'relu3_3': 15, 'relu4_3': 24}
        assert layer_name in layer_map
        self.features = nn.Sequential(*list(vgg_features.children())[:layer_map[layer_name] + 1])
        for p in self.parameters():
            p.requires_grad = False
        self.to(device).eval()

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        return self.features(x)


class AblationHybridLoss(nn.Module):
    def __init__(self, use_pixel=True, use_percep=False, use_spectral=False, use_edge=False,
                 lambda_pixel=1.0, lambda_percep=0.08, lambda_spectral=0.06, lambda_edge=0.04,
                 vgg_layer='relu3_3', device='cpu'):
        super().__init__()
        self.use_pixel = use_pixel
        self.use_percep = use_percep
        self.use_spectral = use_spectral
        self.use_edge = use_edge

        self.pixel_loss = nn.L1Loss()
        self.vgg = VGGFeatureExtractor(layer_name=vgg_layer, device=device) if use_percep else None

        self.lambda_pixel = float(lambda_pixel)
        self.lambda_percep = float(lambda_percep)
        self.lambda_spectral = float(lambda_spectral)
        self.lambda_edge = float(lambda_edge)

    def fft_loss(self, pred, gt):
        Fp = torch.fft.fft2(pred, norm='ortho')
        Fg = torch.fft.fft2(gt, norm='ortho')
        return torch.mean(torch.abs(torch.abs(Fp) - torch.abs(Fg)))

    def edge_map(self, x):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
        ex = F.conv2d(x, sobel_x.expand(x.size(1), -1, -1, -1), padding=1, groups=x.size(1))
        ey = F.conv2d(x, sobel_y.expand(x.size(1), -1, -1, -1), padding=1, groups=x.size(1))
        mag = torch.sqrt(ex ** 2 + ey ** 2 + 1e-6)
        return mag.mean(dim=1, keepdim=True)

    def forward(self, pred, gt):
        total = 0.0
        terms = {}

        if self.use_pixel:
            pix = self.pixel_loss(pred, gt)
            total = total + self.lambda_pixel * pix
            terms['pixel'] = float(pix.detach().cpu().item())

        if self.use_percep and self.vgg is not None:
            f_pred = self.vgg(pred)
            f_gt = self.vgg(gt)
            percep = F.l1_loss(f_pred, f_gt)
            total = total + self.lambda_percep * percep
            terms['percep'] = float(percep.detach().cpu().item())

        if self.use_spectral:
            spec = self.fft_loss(pred, gt)
            total = total + self.lambda_spectral * spec
            terms['spectral'] = float(spec.detach().cpu().item())

        if self.use_edge:
            e_pred = self.edge_map(pred)
            e_gt = self.edge_map(gt)
            edge = F.l1_loss(e_pred, e_gt)
            total = total + self.lambda_edge * edge
            terms['edge'] = float(edge.detach().cpu().item())

        terms['total'] = float(total.detach().cpu().item())
        return total, terms


# ----------------------------------------------------------------------
# GA channel-ratio search
# ----------------------------------------------------------------------
def set_model_ga_ratios(model, ratios):
    for m in model.modules():
        if isinstance(m, ChannelAttention):
            m.set_ga_ratios(ratios)


def random_chromosome():
    r = np.random.rand(3)
    return (r / r.sum()).tolist()


def ga_fitness(chromosome, model, dataset_eval, device, max_eval=20, use_amp=None):
    if use_amp is None:
        use_amp = (device.type == 'cuda')
    set_model_ga_ratios(model, chromosome)
    model.eval()

    total_psnr, total_ssim = 0.0, 0.0
    n = min(len(dataset_eval), max_eval)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
        for i in range(n):
            lr, hr = dataset_eval[i]
            lr = lr.unsqueeze(0).to(device)
            hr = hr.unsqueeze(0).to(device)
            sr = model(lr).float()

            hr_np = hr.squeeze().cpu().numpy()
            sr_np = sr.squeeze().cpu().numpy()

            total_psnr += psnr(hr_np, sr_np, data_range=1.0)
            total_ssim += ssim(hr_np, sr_np, data_range=1.0)

    avg_psnr = total_psnr / n
    avg_ssim = total_ssim / n
    return float(avg_psnr + 10 * avg_ssim)


def run_ga(model, dataset_eval, device, pop_size=12, generations=15, max_eval=20):
    population = [random_chromosome() for _ in range(pop_size)]
    best_global, best_fitness = None, -1e9

    for gen in range(generations):
        fitness = [ga_fitness(c, model, dataset_eval, device, max_eval=max_eval) for c in population]

        gen_best_idx = int(np.argmax(fitness))
        if fitness[gen_best_idx] > best_fitness:
            best_fitness = fitness[gen_best_idx]
            best_global = population[gen_best_idx]

        elite_idx = np.argsort(fitness)[-4:]
        elites = [population[i] for i in elite_idx]

        new_pop = elites.copy()
        while len(new_pop) < pop_size:
            p1, p2 = random.sample(elites, 2)
            child = [(a + b) / 2 for a, b in zip(p1, p2)]
            if random.random() < 0.3:
                i = random.randint(0, 2)
                child[i] += np.random.randn() * 0.1
            child = np.clip(child, 0.01, 1.0)
            child = (child / np.sum(child)).tolist()
            new_pop.append(child)

        population = new_pop
        print(f"GA Gen {gen + 1}/{generations} best fitness = {best_fitness:.4f}")

    print("Best GA ratios:", best_global)
    return best_global, best_fitness


def ga_fitness_loss(chromosome, model, criterion, batches, device, use_amp=None):
    """Fitness for a candidate ratio chromosome, measured as NEGATIVE
    average TRAINING LOSS (the same criterion used for backbone weight
    updates) over a fixed set of pre-fetched (lr, hr) batches — not
    PSNR/SSIM on a held-out evaluation set.

    Sign convention: run_ga_loss (like run_ga) selects the chromosome
    with the HIGHEST fitness each generation. Since lower loss = better
    ratios, fitness is returned as -avg_loss, so "highest fitness" still
    means "best (lowest-loss) ratios" — this keeps the exact same
    generational selection code as run_ga, just fed a different signal.

    No backward pass here — ratios are searched by evolution, not
    gradient descent (the top-k channel mask is non-differentiable
    w.r.t. which channels are selected), so this only needs forward
    passes under torch.no_grad()."""
    if use_amp is None:
        use_amp = (device.type == 'cuda')
    set_model_ga_ratios(model, chromosome)
    model.eval()

    total_loss, n = 0.0, 0
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
        for lr, hr in batches:
            lr = lr.to(device)
            hr = hr.to(device)
            pred = model(lr).float()
            loss, _ = criterion(pred, hr)
            total_loss += float(loss.detach().cpu().item())
            n += 1

    avg_loss = total_loss / max(n, 1)
    return -avg_loss


def run_ga_loss(model, criterion, batches, device, pop_size=12, generations=10, use_amp=None):
    """Same generational GA procedure as run_ga (random init -> elite
    selection -> crossover -> mutation), but fitness = negative training
    loss (ga_fitness_loss) instead of PSNR+SSIM. Intended to be called
    PERIODICALLY DURING TRAINING (see train_loop_with_ga below), using a
    small, FIXED set of freshly-sampled training batches so every
    candidate in this search is compared on identical data.

    Returns (best_ratios, best_loss) — best_loss is reported as a
    positive loss value (lower is better), not the internal negated
    fitness, for interpretability in logs/checkpoints."""
    population = [random_chromosome() for _ in range(pop_size)]
    best_global, best_fitness = None, -1e9   # fitness = -loss, so -1e9 is "infinitely bad"

    for gen in range(generations):
        fitness = [ga_fitness_loss(c, model, criterion, batches, device, use_amp=use_amp)
                   for c in population]

        gen_best_idx = int(np.argmax(fitness))
        if fitness[gen_best_idx] > best_fitness:
            best_fitness = fitness[gen_best_idx]
            best_global = population[gen_best_idx]

        elite_idx = np.argsort(fitness)[-4:]
        elites = [population[i] for i in elite_idx]

        new_pop = elites.copy()
        while len(new_pop) < pop_size:
            p1, p2 = random.sample(elites, 2)
            child = [(a + b) / 2 for a, b in zip(p1, p2)]
            if random.random() < 0.3:
                i = random.randint(0, 2)
                child[i] += np.random.randn() * 0.1
            child = np.clip(child, 0.01, 1.0)
            child = (child / np.sum(child)).tolist()
            new_pop.append(child)

        population = new_pop

    best_loss = -best_fitness
    return best_global, best_loss


# ----------------------------------------------------------------------
# Train / evaluate helpers
# ----------------------------------------------------------------------
def train_loop(model, dataloader, criterion, optimizer, device, num_epochs, log_every=200,
                checkpoint_path=None, checkpoint_every=1, extra_state=None, use_amp=None,
                scheduler=None, grad_clip_norm=None,
                val_dataset=None, val_every=5, best_checkpoint_path=None, val_max_eval=20,
                ema=None):
    """Generic training loop. GA ratios (buffers) stay whatever they were
    set to before calling this — they are NOT touched here, so callers
    control which ratios are in effect during training/fine-tuning.

    If checkpoint_path is given, saves model/optimizer state to that path
    every `checkpoint_every` epochs (and after the final epoch), so a
    long run is recoverable/inspectable before it fully finishes.
    extra_state: optional dict of extra fields to include in the checkpoint
    (e.g. {'ga_ratios': [...]}).

    use_amp: enables mixed-precision training (faster + less GPU memory
    on NVIDIA GPUs). Defaults to True when running on CUDA, False on CPU
    (AMP has no benefit on CPU here).

    scheduler: optional torch.optim.lr_scheduler instance; .step() is
    called once per epoch (after the epoch's batches), AFTER unscaling
    (safe to use with AMP).

    grad_clip_norm: if set (e.g. 1.0), clips gradient norm before each
    optimizer step — helps stabilize training, especially with a hybrid
    multi-term loss or a scheduler with a higher initial LR.

    val_dataset / val_every / best_checkpoint_path: if val_dataset is
    given, evaluates PSNR every `val_every` epochs and saves a SEPARATE
    checkpoint to best_checkpoint_path whenever validation PSNR improves
    — so you keep the best-generalizing epoch, not just the last one.
    val_max_eval limits how many validation images are scored per check
    (for speed on a large validation set).

    ema: optional ModelEMA instance; .update(model) is called after every
    optimizer step. Does not affect training itself — purely a smoothed
    shadow copy you can evaluate/save separately at the end via
    ema.copy_to(some_other_model_instance)."""
    if use_amp is None:
        use_amp = (device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_psnr = -1e9
    model.train()
    epoch_times = []
    for epoch in range(num_epochs):
        epoch_start = time.time()
        running_loss, steps = 0.0, 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
        for i, (inputs, targets) in enumerate(loop):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(inputs)
                loss, terms = criterion(outputs, targets)

            scaler.scale(loss).backward()

            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()

            if ema is not None:
                ema.update(model)

            running_loss += float(loss.detach().cpu().item())
            steps += 1

            if (i + 1) % log_every == 0 or (i + 1) == len(dataloader):
                print(f"Epoch {epoch + 1} [{i + 1}/{len(dataloader)}] "
                      f"avg_loss={running_loss / steps:.4f}")

        if scheduler is not None:
            scheduler.step()
            print(f"Epoch {epoch + 1}: LR now {optimizer.param_groups[0]['lr']:.2e}")

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining = (num_epochs - (epoch + 1)) * avg_epoch_time
        print(f"Epoch {epoch + 1}/{num_epochs} done in {epoch_time:.1f}s "
              f"(avg {avg_epoch_time:.1f}s/epoch, ~{remaining/60:.1f} min remaining)")

        is_last_epoch = (epoch + 1) == num_epochs
        if checkpoint_path and ((epoch + 1) % checkpoint_every == 0 or is_last_epoch):
            os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
            state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            if extra_state:
                state.update(extra_state)
            torch.save(state, checkpoint_path)
            print(f"Checkpoint saved (epoch {epoch + 1}/{num_epochs}) -> {checkpoint_path}")

        if val_dataset is not None and ((epoch + 1) % val_every == 0 or is_last_epoch):
            val_psnr, val_ssim = evaluate_model(model, val_dataset, device, max_eval=val_max_eval)
            print(f"Epoch {epoch + 1}: validation PSNR={val_psnr:.2f} dB, SSIM={val_ssim:.4f}")
            if val_psnr > best_val_psnr and best_checkpoint_path:
                best_val_psnr = val_psnr
                os.makedirs(os.path.dirname(best_checkpoint_path) or ".", exist_ok=True)
                state = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'val_psnr': val_psnr, 'val_ssim': val_ssim,
                }
                if extra_state:
                    state.update(extra_state)
                torch.save(state, best_checkpoint_path)
                print(f"New best validation PSNR ({val_psnr:.2f} dB) -> saved to {best_checkpoint_path}")
            model.train()  # evaluate_model leaves the model in eval() mode

    return model


def train_loop_with_ga(model, dataloader, criterion, optimizer, device, num_epochs,
                        ga_every_n_epochs=5, ga_pop_size=8, ga_generations=8, ga_batches_n=4,
                        log_every=200, checkpoint_path=None, checkpoint_every=1,
                        extra_state=None, use_amp=None, scheduler=None, grad_clip_norm=None,
                        val_dataset=None, val_every=5, best_checkpoint_path=None, val_max_eval=20,
                        ema=None):
    """Same training loop as train_loop, but every `ga_every_n_epochs`
    epochs, PAUSES weight training and runs an in-training GA search
    (run_ga_loss) over the three channel-selection ratios, using
    `ga_batches_n` freshly-drawn batches from `dataloader` and the SAME
    `criterion` used for weight training as the fitness function (lower
    loss = better ratios). The best ratios found are applied via
    set_model_ga_ratios and training resumes for the next
    ga_every_n_epochs epochs with those ratios held fixed, until the
    next GA update.

    This makes the ratios and the backbone weights CO-ADAPT across a
    single training run: weights are updated every step via gradient
    descent (loss-based), ratios are updated every ga_every_n_epochs via
    evolutionary search (also loss-based, since the top-k channel mask
    the ratios control is non-differentiable and cannot itself receive
    a gradient). This replaces the earlier two-phase design (train with
    fixed ratios -> separately search ratios via PSNR/SSIM on a
    validation set -> optionally fine-tune) with a single alternating-
    optimization training run.

    GA search cost per invocation = ga_pop_size * ga_generations *
    ga_batches_n forward passes (no backward pass, so cheaper per-pass
    than a training step, but this still adds real wall-clock time --
    keep ga_pop_size/ga_generations/ga_batches_n small, e.g. the
    defaults, for frequent (small ga_every_n_epochs) updates; use larger
    values only if you space GA updates further apart.

    All other arguments and behavior (checkpointing, scheduler, grad
    clipping, validation-based best-checkpoint saving, EMA) are
    identical to train_loop -- see its docstring."""
    if use_amp is None:
        use_amp = (device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_psnr = -1e9
    current_ratios = None
    model.train()
    epoch_times = []
    for epoch in range(num_epochs):
        epoch_start = time.time()
        running_loss, steps = 0.0, 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
        for i, (inputs, targets) in enumerate(loop):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(inputs)
                loss, terms = criterion(outputs, targets)

            scaler.scale(loss).backward()

            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()

            if ema is not None:
                ema.update(model)

            running_loss += float(loss.detach().cpu().item())
            steps += 1

            if (i + 1) % log_every == 0 or (i + 1) == len(dataloader):
                print(f"Epoch {epoch + 1} [{i + 1}/{len(dataloader)}] "
                      f"avg_loss={running_loss / steps:.4f}")

        if scheduler is not None:
            scheduler.step()
            print(f"Epoch {epoch + 1}: LR now {optimizer.param_groups[0]['lr']:.2e}")

        # --- Periodic in-training GA search over channel-selection ratios ---
        is_last_epoch = (epoch + 1) == num_epochs
        if (epoch + 1) % ga_every_n_epochs == 0 or is_last_epoch:
            ga_batches = []
            for lr_batch, hr_batch in dataloader:
                ga_batches.append((lr_batch, hr_batch))
                if len(ga_batches) >= ga_batches_n:
                    break

            print(f"Epoch {epoch + 1}: running in-training GA ratio search "
                  f"({ga_pop_size} pop x {ga_generations} gen x {len(ga_batches)} batches)...")
            best_ratios, best_loss = run_ga_loss(
                model, criterion, ga_batches, device,
                pop_size=ga_pop_size, generations=ga_generations, use_amp=use_amp,
            )
            set_model_ga_ratios(model, best_ratios)
            current_ratios = best_ratios
            print(f"Epoch {epoch + 1}: GA search complete. ratios={best_ratios}, "
                  f"fitness_loss={best_loss:.4f}")
            model.train()  # run_ga_loss leaves the model in eval() mode

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining = (num_epochs - (epoch + 1)) * avg_epoch_time
        print(f"Epoch {epoch + 1}/{num_epochs} done in {epoch_time:.1f}s "
              f"(avg {avg_epoch_time:.1f}s/epoch, ~{remaining/60:.1f} min remaining)")

        if checkpoint_path and ((epoch + 1) % checkpoint_every == 0 or is_last_epoch):
            os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
            state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ga_ratios': current_ratios,
            }
            if extra_state:
                state.update(extra_state)
            torch.save(state, checkpoint_path)
            print(f"Checkpoint saved (epoch {epoch + 1}/{num_epochs}) -> {checkpoint_path}")

        if val_dataset is not None and ((epoch + 1) % val_every == 0 or is_last_epoch):
            val_psnr, val_ssim = evaluate_model(model, val_dataset, device, max_eval=val_max_eval)
            print(f"Epoch {epoch + 1}: validation PSNR={val_psnr:.2f} dB, SSIM={val_ssim:.4f}")
            if val_psnr > best_val_psnr and best_checkpoint_path:
                best_val_psnr = val_psnr
                os.makedirs(os.path.dirname(best_checkpoint_path) or ".", exist_ok=True)
                state = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'val_psnr': val_psnr, 'val_ssim': val_ssim,
                    'ga_ratios': current_ratios,
                }
                if extra_state:
                    state.update(extra_state)
                torch.save(state, best_checkpoint_path)
                print(f"New best validation PSNR ({val_psnr:.2f} dB) -> saved to {best_checkpoint_path}")
            model.train()

    return model, current_ratios


class ModelEMA:
    """Exponential moving average of model weights. Keeps a shadow copy
    of the model's parameters, updated after every optimizer step as
    ema_param = decay * ema_param + (1 - decay) * current_param.

    Why this helps: SGD/Adam weights bounce around the loss surface from
    step to step, especially with a multi-term hybrid loss; the raw final
    checkpoint can land in a slightly noisy point. The EMA weights trace
    a smoothed trajectory, which frequently generalizes slightly better
    and evaluates more consistently run-to-run. Used in EDSR-style
    training and very common in modern vision training recipes generally.
    Adds negligible compute cost (one weighted-average op per step) and
    no retraining risk — if it doesn't help, just evaluate the raw model
    instead; the EMA copy never affects the actual training gradients.
    """
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()  # e.g. buffers like ga_ratios / num_batches_tracked

    def copy_to(self, model):
        """Loads the EMA weights into `model` in-place (e.g. for evaluation)."""
        model.load_state_dict(self.shadow)

    def state_dict(self):
        return self.shadow


def load_and_upscale_color(image_path, scale):
    """Loads a single image and builds the same LR -> LR-upsampled input
    pipeline used by SRDataset/training (Y channel only), so single-image
    inference is consistent with how the model was trained. Also prepares
    the Cb/Cr chroma channels (bicubic-upsampled to the model's output
    resolution) so the model's Y-only output can be recombined into a
    full-color image, plus a bicubic full-color baseline for comparison.

    Returns a dict:
      original_rgb        - PIL image, input cropped to a multiple of scale
      lr_up_tensor         - [1,1,H,W] tensor, the model's input
      hr_y_up_tensor        - [1,1,H*scale,W*scale] tensor, Y-channel pseudo
                              ground truth (bicubic upscale of the true Y),
                              used for PSNR/SSIM exactly as in evaluate_model
      cb_up, cr_up          - PIL 'L' images, chroma upsampled to output size
      bicubic_baseline_rgb  - PIL image, naive bicubic upscale in full color
    """
    img = Image.open(image_path).convert('RGB')
    w, h = img.size
    new_w, new_h = w - (w % scale), h - (h % scale)
    img = img.resize((new_w, new_h), Image.BICUBIC)

    ycbcr = img.convert('YCbCr')
    y, cb, cr = ycbcr.split()

    lr = y.resize((new_w // scale, new_h // scale), Image.BICUBIC)
    lr_up = lr.resize((new_w, new_h), Image.BICUBIC)

    out_w, out_h = new_w * scale, new_h * scale
    cb_up = cb.resize((out_w, out_h), Image.BICUBIC)
    cr_up = cr.resize((out_w, out_h), Image.BICUBIC)
    hr_y_up = y.resize((out_w, out_h), Image.BICUBIC)

    bicubic_baseline_rgb = img.resize((out_w, out_h), Image.BICUBIC)

    to_tensor = transforms.ToTensor()
    return {
        'original_rgb': img,
        'lr_up_tensor': to_tensor(lr_up).unsqueeze(0),
        'hr_y_up_tensor': to_tensor(hr_y_up).unsqueeze(0),
        'cb_up': cb_up,
        'cr_up': cr_up,
        'bicubic_baseline_rgb': bicubic_baseline_rgb,
    }


def y_tensor_to_color_image(y_tensor, cb_img, cr_img):
    """Combines a single-channel Y tensor (model output, values in [0,1])
    with PIL Cb/Cr chroma images of matching size into a full-color RGB
    PIL image."""
    y_np = (y_tensor.squeeze().clamp(0, 1).detach().cpu().numpy() * 255.0).astype(np.uint8)
    y_img = Image.fromarray(y_np, mode='L')
    if y_img.size != cb_img.size:
        cb_img = cb_img.resize(y_img.size, Image.BICUBIC)
        cr_img = cr_img.resize(y_img.size, Image.BICUBIC)
    merged = Image.merge('YCbCr', (y_img, cb_img, cr_img))
    return merged.convert('RGB')


class SRDatasetFullImage(Dataset):
    """Evaluation-only dataset that uses the ENTIRE test image, not a
    patch_size crop. This is what standard-protocol PSNR/SSIM numbers
    (RCAN, EDSR, and every other paper you'd compare against) are
    actually computed on — full images (cropped only to a multiple of
    `scale`, never sub-cropped to a fixed patch size). Never use this
    for training — full images vary in size, so batch_size must stay 1,
    and there is no random cropping (deterministic, one sample per
    image, no augmentation)."""
    def __init__(self, image_dir, scale=4):
        self.image_dir = image_dir
        self.image_files = [
            os.path.join(image_dir, x) for x in os.listdir(image_dir)
            if x.lower().endswith(('png', 'jpg', 'jpeg'))
        ]
        self.scale = scale
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert('YCbCr')
        y, _, _ = img.split()
        w, h = y.size
        new_w, new_h = w - (w % self.scale), h - (h % self.scale)
        hr = y.resize((new_w, new_h), Image.BICUBIC)  # crop only to a multiple of scale — NO patch crop
        lr = hr.resize((new_w // self.scale, new_h // self.scale), Image.BICUBIC)
        return self.to_tensor(lr), self.to_tensor(hr)


def forward_tiled(model, lr, device, tile_size=64, tile_overlap=8, scale=4, use_amp=None):
    """
    Runs the model on a full low-resolution image via OVERLAPPING TILES
    instead of one single forward pass over the whole image.

    Why: ChannelAttention uses global average pooling (AdaptiveAvgPool2d(1)),
    which computes statistics over the ENTIRE spatial extent of whatever
    is fed in. Trained on tile_size x tile_size patches (LR resolution =
    training patch_size // scale), the pooled "global context" vector is
    calibrated to that scale. Fed a full test image directly (often 4-10x
    larger), the pooled statistics shift to a regime the attention module
    never saw during training. Tiling at ~training resolution keeps each
    forward pass's global-pool statistics close to what the model was
    trained on.

    lr: [1, C, H, W] tensor, the TRUE low-resolution input.
    tile_size: tile size in LR pixels — set to (training patch_size //
        scale) so each tile matches the spatial extent the model was
        actually trained on.
    tile_overlap: overlap between adjacent tiles, in LR pixels. Blended
        with a linear ramp so tile seams do not appear in the output.
    Returns: [1, C, H*scale, W*scale] SR tensor, reconstructed from tiles.
    """
    if use_amp is None:
        use_amp = (device.type == 'cuda')

    _, C, H, W = lr.shape
    out_H, out_W = H * scale, W * scale
    canvas = torch.zeros(1, C, out_H, out_W, device=device)
    weight = torch.zeros(1, 1, out_H, out_W, device=device)

    stride = max(tile_size - tile_overlap, 1)

    def get_starts(total, tile, stride):
        if total <= tile:
            return [0]
        starts = list(range(0, total - tile + 1, stride))
        if starts[-1] != total - tile:
            starts.append(total - tile)
        return sorted(set(starts))

    ys = get_starts(H, tile_size, stride)
    xs = get_starts(W, tile_size, stride)

    ramp = torch.linspace(0, 1, tile_overlap, device=device) if tile_overlap > 0 else torch.tensor([], device=device)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
        for y0 in ys:
            for x0 in xs:
                h = min(tile_size, H - y0)
                w = min(tile_size, W - x0)
                tile_in = lr[:, :, y0:y0 + h, x0:x0 + w]
                sr_tile = model(tile_in).float()  # [1, C, h*scale, w*scale]

                wy = torch.ones(h, device=device)
                wx = torch.ones(w, device=device)
                if tile_overlap > 0:
                    if y0 > 0:
                        wy[:tile_overlap] = ramp
                    if y0 + h < H:
                        wy[-tile_overlap:] = torch.flip(ramp, dims=[0])
                    if x0 > 0:
                        wx[:tile_overlap] = ramp
                    if x0 + w < W:
                        wx[-tile_overlap:] = torch.flip(ramp, dims=[0])
                win = torch.outer(wy, wx)  # [h, w], values in [0,1]
                win_sr = win.repeat_interleave(scale, dim=0).repeat_interleave(scale, dim=1)  # [h*scale, w*scale]

                oy0, ox0 = y0 * scale, x0 * scale
                oh, ow = h * scale, w * scale
                canvas[:, :, oy0:oy0 + oh, ox0:ox0 + ow] += sr_tile * win_sr
                weight[:, :, oy0:oy0 + oh, ox0:ox0 + ow] += win_sr.unsqueeze(0).unsqueeze(0)

    return torch.clamp(canvas / weight.clamp_min(1e-8), 0.0, 1.0)


def evaluate_model_tiled(model, dataset_eval, device, tile_size=64, tile_overlap=8,
                          scale=4, max_eval=None):
    """Same metric computation as evaluate_model, but runs inference via
    forward_tiled instead of a single whole-image forward pass. Use this
    for full-image evaluation (SRDatasetFullImage) to close the train/test
    resolution gap in the channel-attention module."""
    model.eval()
    n = len(dataset_eval) if max_eval is None else min(len(dataset_eval), max_eval)
    total_psnr, total_ssim = 0.0, 0.0

    for i in tqdm(range(n), desc="Evaluating (tiled)"):
        lr, hr = dataset_eval[i]
        lr = lr.unsqueeze(0).to(device)
        sr = forward_tiled(model, lr, device, tile_size=tile_size,
                            tile_overlap=tile_overlap, scale=scale).squeeze(0).cpu()
        hr = hr.cpu()

        sr_img = sr.permute(1, 2, 0).numpy()
        hr_img = hr.permute(1, 2, 0).numpy()

        total_psnr += psnr(hr_img, sr_img, data_range=1.0)
        #total_ssim += ssim(hr_img, sr_img, channel_axis=2, data_range=1.0)
        total_ssim += ssim(hr_img, sr_img, channel_axis=2, data_range=1.0,
        gaussian_weights=True, sigma=1.5, use_sample_covariance=False, win_size=11)

    return total_psnr / n, total_ssim / n


def evaluate_model(model, dataset_eval, device, max_eval=None, use_amp=None):
    """Returns (avg_psnr, avg_ssim) over dataset_eval (or first max_eval items)."""
    if use_amp is None:
        use_amp = (device.type == 'cuda')
    model.eval()
    n = len(dataset_eval) if max_eval is None else min(len(dataset_eval), max_eval)
    total_psnr, total_ssim = 0.0, 0.0

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
        for i in tqdm(range(n), desc="Evaluating"):
            lr, hr = dataset_eval[i]
            lr = lr.unsqueeze(0).to(device)
            sr = model(lr).squeeze(0).float().cpu()
            hr = hr.cpu()

            sr_img = sr.permute(1, 2, 0).numpy()
            hr_img = hr.permute(1, 2, 0).numpy()

            total_psnr += psnr(hr_img, sr_img, data_range=1.0)
            total_ssim += ssim(hr_img, sr_img, channel_axis=2, data_range=1.0)

    return float(total_psnr / n), float(total_ssim / n)


def _self_ensemble_forward(model, lr_tensor):
    """Runs the model on 8 geometric variants of lr_tensor (identity, 3
    rotations, and their horizontal mirrors), undoes each transform on
    the output, and averages. Standard "self-ensemble" trick (used in
    EDSR) — costs 8x inference compute at TEST time only, no retraining,
    no architecture change, and is a legitimate, literature-precedented
    way to gain ~0.1-0.3 dB PSNR. Apply it identically to every arm you
    compare, or don't use it at all — using it for only one arm would
    reintroduce the same unfair-comparison problem flagged earlier."""
    outputs = []
    for hflip in (False, True):
        for k in (0, 1, 2, 3):  # number of 90-degree rotations
            x = lr_tensor
            if hflip:
                x = torch.flip(x, dims=[-1])
            if k:
                x = torch.rot90(x, k=k, dims=[-2, -1])

            with torch.no_grad():
                out = model(x)

            if k:
                out = torch.rot90(out, k=-k, dims=[-2, -1])
            if hflip:
                out = torch.flip(out, dims=[-1])
            outputs.append(out)

    return torch.stack(outputs, dim=0).mean(dim=0)


def _self_ensemble_forward_tiled(model, lr_tensor, device, tile_size, tile_overlap, scale):
    """Same 8-transform self-ensemble as _self_ensemble_forward, but each
    transform is run through forward_tiled instead of a single whole-
    image model(x) call. Use this (not the untiled version) for full-
    image evaluation — a single whole-image forward pass reintroduces
    the global-average-pooling train/test resolution mismatch that
    tiling exists to fix, even inside a self-ensemble loop."""
    outputs = []
    for hflip in (False, True):
        for k in (0, 1, 2, 3):
            x = lr_tensor
            if hflip:
                x = torch.flip(x, dims=[-1])
            if k:
                x = torch.rot90(x, k=k, dims=[-2, -1])

            out = forward_tiled(model, x, device, tile_size=tile_size,
                                 tile_overlap=tile_overlap, scale=scale)

            if k:
                out = torch.rot90(out, k=-k, dims=[-2, -1])
            if hflip:
                out = torch.flip(out, dims=[-1])
            outputs.append(out)

    return torch.stack(outputs, dim=0).mean(dim=0)


def evaluate_model_self_ensemble_tiled(model, dataset_eval, device, tile_size, tile_overlap,
                                        scale, max_eval=None):
    """Tiled counterpart to evaluate_model_self_ensemble — use this for
    full-image (SRDatasetFullImage) evaluation with self-ensembling."""
    model.eval()
    n = len(dataset_eval) if max_eval is None else min(len(dataset_eval), max_eval)
    total_psnr, total_ssim = 0.0, 0.0

    for i in tqdm(range(n), desc="Evaluating (tiled self-ensemble)"):
        lr, hr = dataset_eval[i]
        lr = lr.unsqueeze(0).to(device)
        sr = _self_ensemble_forward_tiled(
            model, lr, device, tile_size=tile_size, tile_overlap=tile_overlap, scale=scale
        ).squeeze(0).float().cpu()
        hr = hr.cpu()

        sr_img = sr.permute(1, 2, 0).numpy()
        hr_img = hr.permute(1, 2, 0).numpy()

        total_psnr += psnr(hr_img, sr_img, data_range=1.0)
        #total_ssim += ssim(hr_img, sr_img, channel_axis=2, data_range=1.0)
        total_ssim += ssim(hr_img, sr_img, channel_axis=2, data_range=1.0,
        gaussian_weights=True, sigma=1.5, use_sample_covariance=False, win_size=11)

    return float(total_psnr / n), float(total_ssim / n)


def evaluate_model_self_ensemble(model, dataset_eval, device, max_eval=None):
    """Same as evaluate_model, but averages over 8 geometric test-time
    transforms per image (see _self_ensemble_forward). No use_amp param —
    self-ensemble does 8 forward passes per image already, so it is run
    in full precision here for numerical stability of the averaging."""
    model.eval()
    n = len(dataset_eval) if max_eval is None else min(len(dataset_eval), max_eval)
    total_psnr, total_ssim = 0.0, 0.0

    for i in tqdm(range(n), desc="Evaluating (self-ensemble)"):
        lr, hr = dataset_eval[i]
        lr = lr.unsqueeze(0).to(device)
        sr = _self_ensemble_forward(model, lr).squeeze(0).float().cpu()
        hr = hr.cpu()

        sr_img = sr.permute(1, 2, 0).numpy()
        hr_img = hr.permute(1, 2, 0).numpy()

        total_psnr += psnr(hr_img, sr_img, data_range=1.0)
        total_ssim += ssim(hr_img, sr_img, channel_axis=2, data_range=1.0)

    return float(total_psnr / n), float(total_ssim / n)


def _to_jsonable(obj):
    """Recursively convert numpy scalars/arrays (and anything else json
    chokes on, like torch tensors) into plain Python types."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def save_results(path, arm_name, ratios, psnr_val, ssim_val, extra=None):
    result = {
        "arm": arm_name,
        "ga_ratios_used": ratios,
        "psnr": psnr_val,
        "ssim": ssim_val,
    }
    if extra:
        result.update(extra)
    result = _to_jsonable(result)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved results -> {path}")
    print(json.dumps(result, indent=2))
