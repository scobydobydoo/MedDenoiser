import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.ndimage import gaussian_filter
from noise import add_mixed_noise



def make_phantom(size=256, kind=0):
    """Create a simple medical-looking phantom image."""
    img = np.zeros((size, size), dtype=np.float32)
    cx, cy = size // 2, size // 2

    def ellipse(img, x, y, rx, ry, val):
        Y, X = np.ogrid[:size, :size]
        mask = ((X - x) / rx) ** 2 + ((Y - y) / ry) ** 2 <= 1
        img[mask] = val

    if kind == 0:  # brain
        ellipse(img, cx, cy, cx - 10, cy - 20, 0.85)  # skull
        ellipse(img, cx, cy, cx - 30, cy - 36, 0.55)  # tissue
        ellipse(img, cx, cy - 12, 22, 11, 0.12)         # ventricle
        ellipse(img, cx - 40, cy, 26, 22, 0.72)          # white matter L
        ellipse(img, cx + 40, cy, 26, 22, 0.72)          # white matter R
    elif kind == 1:  # CT
        ellipse(img, cx, cy, cx - 5, cy - 15, 0.35)     # body
        ellipse(img, cx, cy, 20, 15, 0.95)               # spine
        ellipse(img, cx, cy, 8, 8, 0.1)                  # canal
        ellipse(img, cx - 60, cy, 40, 55, 0.08)          # lung L
        ellipse(img, cx + 60, cy, 40, 55, 0.08)          # lung R
    else:  # xray
        ellipse(img, cx, cy, cx - 5, cy - 10, 0.7)      # chest
        ellipse(img, cx - 20, cy + 20, 50, 60, 0.85)    # heart
        ellipse(img, cx - 70, cy, 55, 80, 0.25)          # lung L
        ellipse(img, cx + 70, cy, 55, 80, 0.25)          # lung R

    img = gaussian_filter(img, sigma=2.5)
    return np.clip(img, 0, 1).astype(np.float32)


def generate_images(n=300, size=256, seed=42):
    np.random.seed(seed)
    images = []
    for i in range(n):
        img = make_phantom(size=size, kind=i % 3)
        
        img = np.clip(img * np.random.uniform(0.85, 1.15), 0, 1)
        if np.random.rand() > 0.5:
            img = img[:, ::-1].copy()   
        images.append(img)
    return images




def to_tensor(img):
    """float32 [0,1] → tensor [-1,1] with channel dim."""
    return torch.from_numpy((img * 2.0) - 1.0).unsqueeze(0)

def from_tensor(t):
    """tensor [-1,1] → float32 [0,1] numpy."""
    return np.clip((t.squeeze(0).cpu().numpy() + 1.0) / 2.0, 0, 1)


class MedDataset(Dataset):
    def __init__(self, images, noise_std=0.05):
        self.images = images
        self.noise_std = noise_std

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        clean = self.images[idx]
        noisy = add_mixed_noise(clean, std=self.noise_std)
        return to_tensor(noisy), to_tensor(clean)


def get_loaders(n_images=300, image_size=256, batch_size=8, noise_std=0.05):
    images = generate_images(n=n_images, size=image_size)
    dataset = MedDataset(images, noise_std=noise_std)

    n_val   = max(1, int(len(dataset) * 0.15))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)

    print(f"Dataset: {n_train} train  |  {n_val} val")
    return train_loader, val_loader, images  