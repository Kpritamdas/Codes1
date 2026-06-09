# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 07:55:15 2026

@author: Dr.Prithamdas
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 06:59:36 2025

@author: das19
"""

# Define the VDSR Model
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim

# class ChannelAttention(nn.Module):
#     def __init__(self, n_feats, reduction=16):
#         super().__init__()
#         self.n_feats = n_feats

#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.conv1 = nn.Conv2d(n_feats, n_feats // reduction, 1)
#         self.relu = nn.ReLU(True)
#         self.conv2 = nn.Conv2d(n_feats // reduction, n_feats, 1)
#         self.sigmoid = nn.Sigmoid()

#         # GA-controlled ratios (initialized uniform)
#         self.register_buffer("ga_ratios", torch.tensor([0.34, 0.33, 0.33]))

#     def set_ga_ratios(self, ratios):
#         """Called by GA (freq, edge, other)"""
#         self.ga_ratios[:] = torch.tensor(ratios, device=self.ga_ratios.device)

#     def forward(self, x):
#         B, C, H, W = x.shape

#         # --- Standard channel attention ---
#         y = self.pool(x)
#         y = self.conv2(self.relu(self.conv1(y)))
#         attn = self.sigmoid(y)

#         x_attn = x * attn

#         # --- GA channel selection ---
#         freq_r, edge_r, other_r = self.ga_ratios.tolist()

#         kf = int(freq_r * C)
#         ke = int(edge_r * C)
#         ko = C - kf - ke

#         # Frequency score (FFT energy)
#         fft = torch.fft.fft2(x_attn, norm='ortho')
#         freq_score = fft.abs().mean(dim=(2,3))

#         # Edge score
#         gx = x_attn[:, :, :, 1:] - x_attn[:, :, :, :-1]
#         gy = x_attn[:, :, 1:, :] - x_attn[:, :, :-1, :]
#         edge_score = gx.abs().mean((2,3)) + gy.abs().mean((2,3))

#         # Other score (neutral)
#         other_score = torch.ones_like(freq_score)

#         idx_f = torch.topk(freq_score, kf, dim=1).indices
#         idx_e = torch.topk(edge_score, ke, dim=1).indices
#         idx_o = torch.topk(other_score, ko, dim=1).indices

#         mask = torch.zeros_like(freq_score)
#         mask.scatter_(1, idx_f, 1)
#         mask.scatter_(1, idx_e, 1)
#         mask.scatter_(1, idx_o, 1)

#         mask = mask.unsqueeze(-1).unsqueeze(-1)
#         return x_attn * mask
    
class RCAB(nn.Module):
        def __init__(self, n_feats, res_scale=0.1):
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(n_feats, n_feats, 3, 1, 1),
                nn.ReLU(True),
                nn.Conv2d(n_feats, n_feats, 3, 1, 1),
                #ChannelAttention(n_feats)
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
                nn.Conv2d(n_feats, n_feats * (scale**2), kernel_size=3, padding=1),
                nn.PixelShuffle(scale),
                nn.Conv2d(n_feats, 1, kernel_size=3, padding=1)
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
        
        ## Inspect Model architecture
from torchsummary import summary
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SRNet().to(device)
    #summary(model, input_size=(3, 128, 128))
summary(model, input_size=(1, 128, 128))


    ## Export & Visualize Model

    # Export to ONNX
    #dummy_input = torch.randn(1, 3, 128, 128).to(device)
# dummy_input = torch.randn(1, 1, 128, 128).to(device)
# onxx_path = "VDSR_EDSR_spaandch_Attention_Y_pixelShu.onnx"
# torch.onnx.export(model, dummy_input, onxx_path,
#                       input_names=["input"], output_names=["output"],
#                       dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
#                       opset_version=11)
# print(f"✅ Model exported to {onxx_path}")

    # Visualize using Netron
# import netron
# netron.start("VDSR_EDSR_spaandch_Attention_Y_pixelShu.onnx")

    # Visualize as PNG using torchviz
    # from torchviz import make_dot
    # out = model(dummy_input)
    # make_dot(out, params=dict(list(model.named_parameters()))).render("vdsr_graph", format="png")

from torchviz import make_dot
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SRNet().to(device)
    #dummy_input = torch.randn(1, 3, 128, 128).to(device)
dummy_input = torch.randn(1, 1, 128, 128).to(device)
output = model(dummy_input)

    # Generate and visualize computation graph
viz = make_dot(output, params=dict(model.named_parameters()))
viz.format = "png"
viz.render("SRNet_architecture", cleanup=True)

    # Display in Spyder's IPython Console (optional)
from IPython.display import Image
Image(filename="SRNet_architecture.png")


        
        # Dataset (DIV2K or Custom Folder)
        
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np

class SRDataset(Dataset):
        def __init__(self, image_dir, scale=4, patch_size=128, apply_fft_filter=False):
            self.image_dir = image_dir
            self.image_files = [os.path.join(image_dir, x) for x in os.listdir(image_dir) if x.endswith(('png', 'jpg'))]
            self.scale = scale
            self.patch_size = patch_size
            self.apply_fft_filter = apply_fft_filter
            self.to_tensor = transforms.ToTensor()
            
        def __len__(self):
           return len(self.image_files)
            
            ## Random cropping support has been added to your SRDataset class using a 
            #patch_size parameter. This helps train the model on diverse patches, improving 
            #generalization while preserving the HR–LR structure.
    # Frequency-aware preprocessing using FFT-based high-pass filtering has been added to the SRDataset. 
    #You can now pass apply_fft_filter=True to emphasize high-frequency content during training.
        def fft_filter(self, img_pil):
        # """
        # Apply FFT-based high-pass filtering on the Y channel of YCbCr color space.
        # Input: PIL image (RGB)
        # Output: PIL image with filtered Y channel, converted back to RGB
        # """
        # Convert to YCbCr and split channels
            ycbcr = img_pil.convert('YCbCr')
            y, cb, cr = ycbcr.split()
            y_np = np.array(y).astype(np.float32)

        # FFT on Y channel
            fft = np.fft.fft2(y_np)
            fft_shift = np.fft.fftshift(fft)

            h, w = y_np.shape
            mask = np.ones((h, w), np.float32)
            center = (h // 2, w // 2)
            r = min(h, w) // 6  # keep only high frequency components
            mask[center[0]-r:center[0]+r, center[1]-r:center[1]+r] = 0

            fft_shift *= mask
            fft_ishift = np.fft.ifftshift(fft_shift)
            img_back = np.fft.ifft2(fft_ishift)
            img_filtered = np.abs(img_back)
            img_filtered = np.clip(img_filtered, 0, 255).astype(np.uint8)

        # Reconstruct YCbCr image with filtered Y and original Cb, Cr
        
            return Image.fromarray(img_filtered)  # Return grayscale image

        # def __len__(self):
        #     return len(self.image_files)

    ## Updated __getitem__ using Y-channel FFT filter:
        def __getitem__(self, idx):
            img = Image.open(self.image_files[idx]).convert('YCbCr')
            y, _, _ = img.split()  # only Y channel
            w, h = y.size
            new_w, new_h = w - (w % self.scale), h - (h % self.scale)
            y = y.resize((new_w, new_h), Image.BICUBIC)

        # Random crop HR patch
            if new_w > self.patch_size and new_h > self.patch_size:
                x0 = random.randint(0, new_w - self.patch_size)
                y0 = random.randint(0, new_h - self.patch_size)
                hr = y.crop((x0, y0, x0 + self.patch_size, y0 + self.patch_size))
            else:
                hr = y.resize((self.patch_size, self.patch_size), Image.BICUBIC)

        # Generate LR patch
            lr = hr.resize(
            (self.patch_size // self.scale, self.patch_size // self.scale),
            Image.BICUBIC)

        # Upsample LR back to HR size (for model input)
            lr_up = lr.resize((self.patch_size, self.patch_size), Image.BICUBIC)

            if self.apply_fft_filter:
                lr_up = self.fft_filter(lr_up)
                hr = self.fft_filter(hr)

        # **Upscale HR to match model output size**
            hr_up = hr.resize(
            (self.patch_size * self.scale, self.patch_size * self.scale),
            Image.BICUBIC)

            return self.to_tensor(lr_up), self.to_tensor(hr_up)
        
        
       # # Load Dataset
     
dataset = SRDataset("F:/DAS D DRIVE/verydeepsisr/DIV2K_train_HR", scale=4, patch_size=256, apply_fft_filter=True)
# dataset = SRDataset("D:/verydeepsisr/HIGH x4 URban100", scale=4, patch_size=256, apply_fft_filter=True)

print(len(dataset))  # Should print the number of images
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, pin_memory=True)
sample = dataset[0] 
print(sample[0].shape)          # Should be [1, H, W]
    # Train VDSR (with automatic GPU memory check)
       
import torch.optim as optim
from tqdm import tqdm

if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
        print(f"✅ CUDA available: Using GPU - {torch.cuda.get_device_name(0)} with {total_memory} MB memory")
        torch.backends.cudnn.benchmark = True
        batch_size = 2 if total_memory < 4000 else 4
        device = torch.device('cuda')
else:
        print("❌ CUDA not available. Using CPU instead.")
        batch_size = 2
        device = torch.device('cpu')
      # -----------------------------
    # Ablation-ready Hybrid Loss
    # -----------------------------
class VGGFeatureExtractor(nn.Module):
        def __init__(self, layer_name='relu3_3', use_bn=False, device='cpu'):
            super().__init__()
            from torchvision import models
            vgg = models.vgg19_bn(pretrained=True) if use_bn else models.vgg19(pretrained=True)
            vgg_features = vgg.features
            layer_map = {'relu1_2':3, 'relu2_2':8, 'relu3_3':15, 'relu4_3':24}
            assert layer_name in layer_map
            self.features = nn.Sequential(*list(vgg_features.children())[:layer_map[layer_name]+1])
            for p in self.parameters():
                p.requires_grad = False
            self.to(device).eval()

        def forward(self, x):
            # ensure 3-ch input for VGG
            if x.shape[1] == 1:
                x = x.repeat(1,3,1,1)
            mean = torch.tensor([0.485,0.456,0.406], device=x.device).view(1,3,1,1)
            std  = torch.tensor([0.229,0.224,0.225], device=x.device).view(1,3,1,1)
            x = (x - mean) / std
            return self.features(x)

class AblationHybridLoss(nn.Module):
        def __init__(self, use_pixel=True, use_percep=True, use_spectral=True, use_edge=True,
                     lambda_pixel=1.0, lambda_percep=0.1, lambda_spectral=0.1, lambda_edge=0.05,
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
            # Use absolute magnitude difference (robust)
            return torch.mean(torch.abs(torch.abs(Fp) - torch.abs(Fg)))

        def edge_map(self, x):
            sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device=x.device).view(1,1,3,3)
            sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32, device=x.device).view(1,1,3,3)
            # apply per-channel, then average channels
            ex = F.conv2d(x, sobel_x.expand(x.size(1),-1,-1,-1), padding=1, groups=x.size(1))
            ey = F.conv2d(x, sobel_y.expand(x.size(1),-1,-1,-1), padding=1, groups=x.size(1))
            mag = torch.sqrt(ex**2 + ey**2 + 1e-6)
            return mag.mean(dim=1, keepdim=True)  # return single-channel edge map

        def forward(self, pred, gt):
            """
            pred, gt: tensors with shape [B, C, H, W] (C can be 1 or 3)
            returns: (total_loss_tensor, terms_dict)
            """
            total = 0.0
            terms = {}

            if self.use_pixel:
                pix = self.pixel_loss(pred, gt)
                total = total + self.lambda_pixel * pix
                terms['pixel'] = float(pix.detach().cpu().item())

            if self.use_percep and self.vgg is not None:
                f_pred = self.vgg(pred)
                f_gt   = self.vgg(gt)
                percep = F.l1_loss(f_pred, f_gt)
                total = total + self.lambda_percep * percep
                terms['percep'] = float(percep.detach().cpu().item())   

            if self.use_spectral:
                spec = self.fft_loss(pred, gt)
                total = total + self.lambda_spectral * spec
                terms['spectral'] = float(spec.detach().cpu().item())

            if self.use_edge:
                e_pred = self.edge_map(pred)
                e_gt   = self.edge_map(gt)
                edge = F.l1_loss(e_pred, e_gt)
                total = total + self.lambda_edge * edge
                terms['edge'] = float(edge.detach().cpu().item())

            terms['total'] = float(total.detach().cpu().item())
            return total, terms  
        
        
        
model = SRNet().to(device)

    # Use AblationHybridLoss (toggle components here as needed)
criterion = AblationHybridLoss(
        use_pixel=True,
        use_percep=True,
        use_spectral=True,
        use_edge=True,
        lambda_pixel=1.0,
        lambda_percep=0.08,
        lambda_spectral=0.06,
        lambda_edge=0.04,
        vgg_layer='relu3_3',
        device=device
    )

optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # testing part
model.eval()
with torch.no_grad():
        dummy_lr = torch.rand(1,1,128,128, device=device)
        dummy_hr = torch.rand(1,1,512,512, device=device)  # if scale=2, hr size must be 2x
        out = model(dummy_lr)
        l, t = criterion(out, dummy_hr)
        print("Loss ok:", float(l.item()), "terms:", t)
model.train()
        
        
    #num_epochs = 5
for epoch in range(1):
        model.train()
        running_loss = 0.0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/1", leave=False)
        for i, (inputs, targets) in enumerate(loop):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss, terms = criterion(outputs, targets)   # loss is a tensor, terms is a dict

            # Backprop (regular, not using scaler in this snippet)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.detach().cpu().item())

                # accumulate term-wise values for logging
            running_terms = {}
            steps = 0
            for k, v in terms.items():
                running_terms[k] = running_terms.get(k, 0.0) + v

                steps += 1
                if (i+1) % 1 == 0 or (i+1) == len(dataloader):
                    avg_terms = {k: running_terms[k] / steps for k in running_terms}
                    print(f"Epoch {epoch+1} [{i+1}/{len(dataloader)}] loss={running_loss/steps:.4f} terms={avg_terms}")
        
        # Saving the Model After Training
        # After training (e.g., after the loop)
        save_path = "F:DAS D DRIVE/verydeepsisr/optimization_based_super reso/models/sr_GA_model_psnr_ssim_withFFT_16_RCAB_epoch_1.pth"  # Create 'models' folder if needed
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure directory exists

        torch.save({
            'epoch': 1,  # Optional: Save epoch number
            'model_state_dict': model.state_dict(),  # Save weights
            'optimizer_state_dict': optimizer.state_dict(),  # Optional: Save optimizer state for resuming training
            #'loss': final_loss,  # Optional: Save final loss
            }, save_path)

        print(f"✅ Model saved to {save_path}")
        
        # Helper: Set GA ratios globally in model

        # def set_model_ga_ratios(model, ratios):
        #     for m in model.modules():
        #         if isinstance(m, ChannelAttention):
        #             m.set_ga_ratios(ratios)
                    
#dataset_Eva = SRDataset("D:/verydeepsisr/Set 5/Set 5 x4", scale=4, patch_size=256, apply_fft_filter=True)
#dataset_Eva = SRDataset("F:/DAS D DRIVE/verydeepsisr/Set 5_HR", scale=4, patch_size=256, apply_fft_filter=True)   
dataset_Eva = SRDataset("F:/DAS D DRIVE/verydeepsisr/HIGH x4 URban100", scale=4, patch_size=256, apply_fft_filter=True) 

def random_chromosome():
    r = np.random.rand(3)
    return (r / r.sum()).tolist()

#Fitness function (uses your validation dataset)
from skimage.metrics import peak_signal_noise_ratio as psnr

# def ga_fitness(chromosome, model, dataset_Eva, device, max_eval=10):
#     set_model_ga_ratios(model, chromosome)
#     model.eval()

#     total_psnr = 0.0
#     total_ssim = 0.0

#     with torch.no_grad():
#         for i in range(min(len(dataset_Eva), max_eval)):
#             lr, hr = dataset_Eva[i]

#             lr = lr.unsqueeze(0).to(device)
#             hr = hr.unsqueeze(0).to(device)

#             sr = model(lr)

#             hr_np = hr.squeeze().cpu().numpy()
#             sr_np = sr.squeeze().cpu().numpy()

#             total_psnr += psnr(hr_np, sr_np, data_range=1.0)
#             total_ssim += ssim(hr_np, sr_np, data_range=1.0)

#     avg_psnr = total_psnr / max_eval
#     avg_ssim = total_ssim / max_eval

#     # Combined fitness
#     fitness = avg_psnr + 10 * avg_ssim

#     return fitness


# GA optimization loop (outer loop)
# Run this after initial training:
# def run_ga(model, dataset_Eva, device, pop_size=12, generations=10):

#     population = [random_chromosome() for _ in range(pop_size)]

#     for gen in range(generations):

#         fitness = [ga_fitness(c, model, dataset_Eva, device) for c in population]

#         elite_idx = np.argsort(fitness)[-4:]
#         elites = [population[i] for i in elite_idx]

#         new_pop = elites.copy()

#         while len(new_pop) < pop_size:

#             p1, p2 = random.sample(elites, 2)

#             child = [(a+b)/2 for a, b in zip(p1, p2)]

#             if random.random() < 0.3:
#                 i = random.randint(0,2)
#                 child[i] += np.random.randn()*0.05

#             child = np.clip(child, 0.01, 1.0)
#             child = (child / np.sum(child)).tolist()

#             new_pop.append(child)

#         population = new_pop

#         print(f"🧬 GA Gen {gen+1} best fitness = {max(fitness):.2f}")

#     best = population[np.argmax(fitness)]
#     print("✅ Best GA ratios:", best)

#     return best

# def run_ga(model, dataset_Eva, device, pop_size=12, generations=10):

#     population = [random_chromosome() for _ in range(pop_size)]

#     best_global = None
#     best_fitness = -1e9

#     for gen in range(generations):

#         fitness = [ga_fitness(c, model, dataset_Eva, device) for c in population]

#         # Track global best
#         gen_best_idx = np.argmax(fitness)
#         if fitness[gen_best_idx] > best_fitness:
#             best_fitness = fitness[gen_best_idx]
#             best_global = population[gen_best_idx]

#         elite_idx = np.argsort(fitness)[-4:]
#         elites = [population[i] for i in elite_idx]

#         new_pop = elites.copy()

#         while len(new_pop) < pop_size:

#             p1, p2 = random.sample(elites, 2)

#             child = [(a+b)/2 for a, b in zip(p1, p2)]

#             if random.random() < 0.3:
#                 i = random.randint(0,2)
#                 #child[i] += np.random.randn()*0.05
#                 child[i] += np.random.randn()*0.1

#             child = np.clip(child, 0.01, 1.0)
#             child = (child / np.sum(child)).tolist()

#             new_pop.append(child)

#         population = new_pop

#         print(f"🧬 GA Gen {gen+1} best fitness = {best_fitness:.3f}")

#     print("✅ Best GA ratios:", best_global)

#     return best_global

# Run GA on validation set
# dataset_Eva = SRDataset("D:/verydeepsisr/Set 5", scale=2, patch_size=256, apply_fft_filter=True)
# best_ratios = run_ga(model, dataset_Eva, device)
# set_model_ga_ratios(model, best_ratios)

# Fine-tune with fixed GA ratios
for param in model.parameters():
    param.requires_grad = True
        
        # Evaluation and Visualization
from tqdm import tqdm
    #dataset_Eva = SRDataset("D:/HIGH X2 Urban", scale=2, patch_size=256, apply_fft_filter=True)
# dataset_Eva = SRDataset("D:/verydeepsisr/Set 5", scale=2, patch_size=256, apply_fft_filter=True)
        
#import matplotlib.pyplot as plt
#import torchvision.transforms.functional as TF
# from skimage.metrics import peak_signal_noise_ratio as psnr
#from skimage.metrics import structural_similarity as ssim
from PIL import Image


    # Evaluate on full dataset

model.eval()
total_psnr = 0.0
total_ssim = 0.0
num_samples = len(dataset_Eva)
def tensor_to_image(tensor):
        # tensor: [1, H, W] or [H, W]
        if tensor.dim() == 3:
            return tensor.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
        elif tensor.dim() == 2:
            return tensor.cpu().numpy()  # [H, W]
        else:
            raise ValueError(f"Unexpected tensor shape: {tensor.shape}")

with torch.no_grad():
        for lr, hr in tqdm(dataset_Eva, desc="Evaluating"):
            lr = lr.unsqueeze(0).to(device)
            sr = model(lr).squeeze(0).cpu()

            sr_img = sr.permute(1, 2, 0).numpy()
            hr_img = hr.permute(1, 2, 0).numpy()

            total_psnr += psnr(hr_img, sr_img, data_range=1.0)
            total_ssim += ssim(hr_img, sr_img, channel_axis=2, data_range=1.0)

        avg_psnr = total_psnr / num_samples
        avg_ssim = total_ssim / num_samples

        print(f"📊 Average PSNR: {avg_psnr:.2f} dB")
        print(f"📊 Average SSIM: {avg_ssim:.4f}")

    
# from tqdm import tqdm
# dataset_Eva_ex = SRDataset("D:/verydeepsisr/test", scale=4, patch_size=256, apply_fft_filter=True)
        
# from skimage.metrics import peak_signal_noise_ratio as psnr
# from skimage.metrics import structural_similarity as ssim
# from PIL import Image


        
# def show_images(lr, sr, hr):
#     fig, axs = plt.subplots(1, 3, figsize=(12, 4))
#     axs[0].imshow(lr.squeeze(0).permute(1, 2, 0).cpu())
#     axs[0].set_title("Low-Res Input")
#     axs[1].imshow(sr.permute(1, 2, 0).numpy(), cmap='gray')
#     axs[1].set_title("Super-Res Output")
#     axs[2].imshow(hr.permute(1, 2, 0).numpy(), cmap='gray')
#     axs[2].set_title("High-Res Ground Truth")
#     for ax in axs:
#         ax.axis('off')
#     plt.show()

#     # Evaluate on a sample image


# with torch.no_grad():
#         for lr_sample, hr_sample in tqdm(dataset_Eva, desc="Evaluating"):
#             lr_sample = lr_sample.unsqueeze(0).to(device)
#             sr_sample = model(lr_sample).squeeze(0).cpu()

#             sr_img_sample = sr_sample.permute(1, 2, 0).numpy()
#             hr_img_sample = hr_sample.permute(1, 2, 0).numpy()

#             total_psnr += psnr(hr_img_sample, sr_img_sample, data_range=1.0)
#             total_ssim += ssim(hr_img_sample, sr_img_sample, channel_axis=2, data_range=1.0)

#         Exter_psnr = total_psnr / num_samples
#         Exter_ssim = total_ssim / num_samples

#         print(f"📊 External Image PSNR: {Exter_psnr:.2f} dB")
#         print(f"📊 External Image SSIM: {Exter_ssim:.4f}")

#         show_images(lr_sample, sr_sample, hr_sample)

    
        


