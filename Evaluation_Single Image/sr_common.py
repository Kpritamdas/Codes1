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

        fft = torch.fft.fft2(x_attn, norm='ortho')
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
    def __init__(self, scale=4, n_feats=64, n_blocks=16, res_scale=0.1, global_residual=True):
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
        self.patch_size = patch_size
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


# ----------------------------------------------------------------------
# Train / evaluate helpers
# ----------------------------------------------------------------------
def train_loop(model, dataloader, criterion, optimizer, device, num_epochs, log_every=200,
                checkpoint_path=None, checkpoint_every=1, extra_state=None, use_amp=None):
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
    (AMP has no benefit on CPU here)."""
    if use_amp is None:
        use_amp = (device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

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
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.detach().cpu().item())
            steps += 1

            if (i + 1) % log_every == 0 or (i + 1) == len(dataloader):
                print(f"Epoch {epoch + 1} [{i + 1}/{len(dataloader)}] "
                      f"avg_loss={running_loss / steps:.4f}")

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

    return model


# def load_and_upscale_color(image_path, scale):
#     """Loads a single image and builds the same LR -> LR-upsampled input
#     pipeline used by SRDataset/training (Y channel only), so single-image
#     inference is consistent with how the model was trained. Also prepares
#     the Cb/Cr chroma channels (bicubic-upsampled to the model's output
#     resolution) so the model's Y-only output can be recombined into a
#     full-color image, plus a bicubic full-color baseline for comparison.

#     Returns a dict:
#       original_rgb        - PIL image, input cropped to a multiple of scale
#       lr_up_tensor         - [1,1,H,W] tensor, the model's input
#       hr_y_up_tensor        - [1,1,H*scale,W*scale] tensor, Y-channel pseudo
#                               ground truth (bicubic upscale of the true Y),
#                               used for PSNR/SSIM exactly as in evaluate_model
#       cb_up, cr_up          - PIL 'L' images, chroma upsampled to output size
#       bicubic_baseline_rgb  - PIL image, naive bicubic upscale in full color
#     """
#     img = Image.open(image_path).convert('RGB')
#     w, h = img.size
#     new_w, new_h = w - (w % scale), h - (h % scale)
#     img = img.resize((new_w, new_h), Image.BICUBIC)

#     ycbcr = img.convert('YCbCr')
#     y, cb, cr = ycbcr.split()

#     lr = y.resize((new_w // scale, new_h // scale), Image.BICUBIC)
#     #lr_up = lr.resize((new_w, new_h), Image.BICUBIC)
    

#     out_w, out_h = new_w * scale, new_h * scale
#     cb_up = cb.resize((out_w, out_h), Image.BICUBIC)
#     cr_up = cr.resize((out_w, out_h), Image.BICUBIC)
#     #hr_y_up = y.resize((out_w, out_h), Image.BICUBIC)

#     bicubic_baseline_rgb = img.resize((out_w, out_h), Image.BICUBIC)

#     to_tensor = transforms.ToTensor()
#     return {
#         'original_rgb': img,
#         #'lr_up_tensor': to_tensor(lr_up).unsqueeze(0),
#         'lr': to_tensor(lr).unsqueeze(0),
#         #'hr_y_up_tensor': to_tensor(hr_y_up).unsqueeze(0),
#         'hr': to_tensor(y).unsqueeze(0),
#         'cb_up': cb_up,
#         'cr_up': cr_up,
#         'bicubic_baseline_rgb': bicubic_baseline_rgb,
    # }


def load_and_upscale_color(image_path, scale):
    """
    Standard paired single-image SR evaluation:
    HR original -> bicubic downsampling -> LR input
    LR -> SRNet -> reconstructed HR

    All metric images have the original HR dimensions.
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    # Crop to dimensions divisible by scale (avoid resizing HR)
    new_w = w - (w % scale)
    new_h = h - (h % scale)
    img = img.crop((0, 0, new_w, new_h))

    y, cb, cr = img.convert("YCbCr").split()

    # Generate low-resolution input
    lr_y = y.resize(
        (new_w // scale, new_h // scale),
        Image.Resampling.BICUBIC
    )

    # Bicubic reconstruction at the ORIGINAL HR resolution
    bicubic_y = lr_y.resize(
        (new_w, new_h),
        Image.Resampling.BICUBIC
    )

    # Both model output and chroma must be HR-sized
    cb_up = cb
    cr_up = cr

    bicubic_baseline_rgb = Image.merge(
        "YCbCr", (bicubic_y, cb_up, cr_up)
    ).convert("RGB")

    to_tensor = transforms.ToTensor()

    return {
        "original_rgb": img,
        "lr": to_tensor(lr_y).unsqueeze(0),
        "hr": to_tensor(y).unsqueeze(0),
        "bicubic_y": to_tensor(bicubic_y).unsqueeze(0),
        "cb_up": cb_up,
        "cr_up": cr_up,
        "bicubic_baseline_rgb": bicubic_baseline_rgb,
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
