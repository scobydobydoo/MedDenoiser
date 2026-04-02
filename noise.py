import numpy as np

def add_gaussian_noise(image, std=0.05):
    noise = np.random.normal(0, std, image.shape).astype(np.float32)
    return np.clip(image + noise, 0, 1)

def add_mixed_noise(image, std=0.05):
    # Gaussian (additive) + Speckle (multiplicative)
    gauss   = np.random.normal(0, std, image.shape).astype(np.float32)
    speckle = np.random.normal(0, std * 0.6, image.shape).astype(np.float32)
    return np.clip(image + gauss + image * speckle, 0, 1)
