"""
train.py
────────
Run:  python train.py

Trains for 50 epochs on synthetic data.
Saves:  best_generator.pth   (best validation PSNR)
        loss_curve.png
"""

import os, json
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr

from dataset  import get_loaders, from_tensor
from generator     import UNet
from discriminator import PatchGAN

# ── Config ──────────────────────────────────────────────────────────────────
IMAGE_SIZE  = 256
N_IMAGES    = 300
BATCH_SIZE  = 4       # keep low for CPU training
N_EPOCHS    = 50
LR          = 2e-4
LAMBDA_L1   = 100.0   # weight of pixel loss vs adversarial loss
NOISE_STD   = 0.05
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH   = "best_generator.pth"

# ── Setup ───────────────────────────────────────────────────────────────────
print(f"Device: {DEVICE}")
train_loader, val_loader, all_images = get_loaders(
    n_images=N_IMAGES, image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE, noise_std=NOISE_STD
)

G = UNet(base=32).to(DEVICE)
D = PatchGAN(base=32).to(DEVICE)

opt_G = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

bce = nn.BCEWithLogitsLoss()
l1  = nn.L1Loss()

# ── Helpers ──────────────────────────────────────────────────────────────────

def real_labels(pred):
    return torch.full_like(pred, 0.9)   # label smoothing

def fake_labels(pred):
    return torch.zeros_like(pred)


@torch.no_grad()
def validate():
    G.eval()
    psnr_vals = []
    for noisy, clean in val_loader:
        noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
        denoised = G(noisy)
        for i in range(denoised.size(0)):
            d = from_tensor(denoised[i])
            c = from_tensor(clean[i])
            psnr_vals.append(psnr(c, d, data_range=1.0))
    return float(np.mean(psnr_vals))

# ── Training loop ────────────────────────────────────────────────────────────

g_losses, d_losses, psnr_log = [], [], []
best_psnr = 0.0

print(f"\nTraining for {N_EPOCHS} epochs...\n")

for epoch in range(1, N_EPOCHS + 1):
    G.train(); D.train()
    ep_g, ep_d = [], []

    for noisy, clean in tqdm(train_loader, desc=f"Epoch {epoch:02d}", leave=False):
        noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)

        # ── Train Discriminator ──────────────────────────────────
        opt_D.zero_grad()
        denoised     = G(noisy).detach()           # no grad to G
        loss_D_real  = bce(D(noisy, clean),    real_labels(D(noisy, clean)))
        loss_D_fake  = bce(D(noisy, denoised), fake_labels(D(noisy, denoised)))
        loss_D       = 0.5 * (loss_D_real + loss_D_fake)
        loss_D.backward()
        opt_D.step()

        # ── Train Generator ──────────────────────────────────────
        opt_G.zero_grad()
        denoised     = G(noisy)                    # fresh forward with grad
        loss_G_adv   = bce(D(noisy, denoised), real_labels(D(noisy, denoised)))
        loss_G_l1    = l1(denoised, clean)
        loss_G       = loss_G_adv + LAMBDA_L1 * loss_G_l1
        loss_G.backward()
        opt_G.step()

        ep_g.append(loss_G.item())
        ep_d.append(loss_D.item())

    avg_g   = np.mean(ep_g)
    avg_d   = np.mean(ep_d)
    val_psnr = validate()

    g_losses.append(round(float(avg_g), 4))
    d_losses.append(round(float(avg_d), 4))
    psnr_log.append(round(val_psnr, 3))

    print(f"Epoch {epoch:02d}/{N_EPOCHS}  G={avg_g:.4f}  D={avg_d:.4f}  PSNR={val_psnr:.2f} dB")

    if val_psnr > best_psnr:
        best_psnr = val_psnr
        torch.save(G.state_dict(), SAVE_PATH)
        print(f"  ★ New best PSNR {val_psnr:.2f} dB → saved {SAVE_PATH}")

# ── Save loss curve ───────────────────────────────────────────────────────────

epochs = range(1, N_EPOCHS + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(epochs, g_losses, label="Generator",     color="royalblue", linewidth=2)
ax1.plot(epochs, d_losses, label="Discriminator", color="tomato",    linewidth=2)
ax1.set_title("GAN Loss"); ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(alpha=0.3)

ax2.plot(epochs, psnr_log, color="seagreen", linewidth=2)
ax2.set_title("Validation PSNR (dB)"); ax2.set_xlabel("Epoch"); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150)
print(f"\nDone. Best PSNR: {best_psnr:.2f} dB")
print("Saved: best_generator.pth  |  loss_curve.png")
