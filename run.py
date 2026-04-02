"""
      python run.py --image my_scan.png    (your own image)
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from dataset    import generate_images, from_tensor, to_tensor
from generator  import UNet
from noise      import add_mixed_noise

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
CKPT      = "best_generator.pth"
IMG_SIZE  = 256
NOISE_STD = 0.05

# ── Load model ───────────────────────────────────────────────────────────────

G = UNet(base=32).to(DEVICE)
G.load_state_dict(torch.load(CKPT, map_location=DEVICE))
G.eval()
print(f"Loaded {CKPT}")



parser = argparse.ArgumentParser()
parser.add_argument("--image", default=None, help="Path to your own image (optional)")
args = parser.parse_args()

if args.image:
    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    clean = img
else:
   
    test_images = generate_images(n=10, size=IMG_SIZE, seed=999)
    clean = test_images[0]

noisy = add_mixed_noise(clean, std=NOISE_STD)



with torch.no_grad():
    tensor   = to_tensor(noisy).unsqueeze(0).to(DEVICE)
    output   = G(tensor)
    denoised = from_tensor(output.squeeze(0))



psnr_noisy    = psnr(clean, noisy,    data_range=1.0)
psnr_denoised = psnr(clean, denoised, data_range=1.0)
ssim_noisy    = ssim(clean, noisy,    data_range=1.0)
ssim_denoised = ssim(clean, denoised, data_range=1.0)

print(f"\nNoisy    → PSNR: {psnr_noisy:.2f} dB   SSIM: {ssim_noisy:.4f}")
print(f"Denoised → PSNR: {psnr_denoised:.2f} dB   SSIM: {ssim_denoised:.4f}")
print(f"Improvement: +{psnr_denoised - psnr_noisy:.2f} dB PSNR")



fig, axes = plt.subplots(1, 3, figsize=(13, 4))
data = [
    (noisy,    f"Noisy\nPSNR={psnr_noisy:.1f} dB"),
    (denoised, f"Denoised (GAN)\nPSNR={psnr_denoised:.1f} dB"),
    (clean,    "Ground Truth"),
]
for ax, (img, title) in zip(axes, data):
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")

plt.suptitle("MedDenoiser Results", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("result.png", dpi=400)
plt.show()
print("Saved result.png")
