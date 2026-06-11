"""
losses.py  —  MedDenoiser
==========================
Defines all generator losses used during training.

Fixed issues vs original
-------------------------
* Removed circular self-import  (`from losses import ...` inside losses.py)
* Removed stray imports of UNet / PatchGAN / dataset helpers (not needed here)
* CharbonnierLoss and edge_loss are unchanged in behaviour
* Added SSIMLoss wrapper (uses pytorch_msssim)
* Added CombinedGeneratorLoss for one-call convenience in train.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pytorch_msssim import ssim as _ssim_fn
    _SSIM_AVAILABLE = True
except ImportError:
    _SSIM_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Charbonnier Loss  (robust L1 — avoids gradient blow-up near zero)
# ─────────────────────────────────────────────────────────────────────────────
class CharbonnierLoss(nn.Module):
    """
    L_charb(pred, target) = mean( sqrt( (pred-target)^2 + eps^2 ) )

    Smoother than plain L1 at zero; sharper than L2 everywhere else.
    eps=1e-3 is a safe default for normalised [-1, 1] tensors.
    """

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Edge Loss  (Sobel-based gradient matching)
# ─────────────────────────────────────────────────────────────────────────────
def edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes L1 loss between Sobel-filtered pred and target.
    Penalises blurring of anatomical edges explicitly.

    Both tensors: (B, 1, H, W), range [-1, 1].
    """
    sobel_x = torch.tensor(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        device=pred.device, dtype=pred.dtype
    ).view(1, 1, 3, 3)

    sobel_y = torch.tensor(
        [[-1, -2, -1],
         [ 0,  0,  0],
         [ 1,  2,  1]],
        device=pred.device, dtype=pred.dtype
    ).view(1, 1, 3, 3)

    pred_gx   = F.conv2d(pred,   sobel_x, padding=1)
    pred_gy   = F.conv2d(pred,   sobel_y, padding=1)
    target_gx = F.conv2d(target, sobel_x, padding=1)
    target_gy = F.conv2d(target, sobel_y, padding=1)

    return F.l1_loss(pred_gx, target_gx) + F.l1_loss(pred_gy, target_gy)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SSIM Loss  (structural similarity — punishes perceptual degradation)
# ─────────────────────────────────────────────────────────────────────────────
class SSIMLoss(nn.Module):
    """
    Loss = 1 - SSIM(pred, target).
    Requires pytorch_msssim (`pip install pytorch-msssim`).
    Falls back to Charbonnier if package is missing.

    Inputs: (B, 1, H, W) in [-1, 1].
    data_range is set to 2.0 because our tensors span [-1, 1].
    """

    def __init__(self, data_range: float = 2.0, size_average: bool = True):
        super().__init__()
        self.data_range   = data_range
        self.size_average = size_average
        self._fallback    = CharbonnierLoss()
        if not _SSIM_AVAILABLE:
            import warnings
            warnings.warn(
                "pytorch_msssim not found — SSIMLoss will fall back to CharbonnierLoss. "
                "Install with: pip install pytorch-msssim",
                stacklevel=2,
            )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not _SSIM_AVAILABLE:
            return self._fallback(pred, target)
        ssim_val = _ssim_fn(
            pred, target,
            data_range=self.data_range,
            size_average=self.size_average,
        )
        return 1.0 - ssim_val


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Combined Generator Loss  (convenience wrapper used in train.py)
# ─────────────────────────────────────────────────────────────────────────────
class CombinedGeneratorLoss(nn.Module):
    """
    total = λ_charb * Charbonnier
          + λ_edge  * EdgeLoss
          + λ_ssim  * SSIMLoss

    Default weights are tuned for the MedDenoiser pix2pix setup.
    Adversarial loss is kept separate in train.py so the discriminator
    call stays outside this module.

    Weight rationale
    ----------------
    λ_charb = 10   replaces the original L1×100; Charbonnier is already
                   a sharper prior so a lower weight suffices.
    λ_edge  =  5   forces the generator to preserve sulci / tissue edges.
    λ_ssim  =  2   structural penalty discourages global brightness shift.
    """

    def __init__(
        self,
        lambda_charb: float = 10.0,
        lambda_edge:  float =  5.0,
        lambda_ssim:  float =  2.0,
    ):
        super().__init__()
        self.lambda_charb = lambda_charb
        self.lambda_edge  = lambda_edge
        self.lambda_ssim  = lambda_ssim

        self.charb = CharbonnierLoss(eps=1e-3)
        self.ssim  = SSIMLoss(data_range=2.0)

    def forward(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns (total_loss, component_dict).
        component_dict lets train.py log individual terms.
        """
        l_charb = self.charb(pred, target)
        l_edge  = edge_loss(pred, target)
        l_ssim  = self.ssim(pred, target)

        total = (
            self.lambda_charb * l_charb
            + self.lambda_edge  * l_edge
            + self.lambda_ssim  * l_ssim
        )

        return total, {
            "charb": l_charb.item(),
            "edge":  l_edge.item(),
            "ssim":  l_ssim.item(),
        }