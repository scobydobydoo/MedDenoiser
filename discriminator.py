import torch
import torch.nn as nn


class PatchGAN(nn.Module):
    """
    PatchGAN Discriminator.
    Input: cat(noisy, denoised_or_clean) → shape (B, 2, H, W)
    Output: patch-level logits (B, 1, H', W')
    """

    def __init__(self, base=32):
        super().__init__()

        def block(ic, oc, stride=2, norm=True):
            layers = [nn.Conv2d(ic, oc, 4, stride=stride, padding=1, bias=not norm)]
            if norm:
                layers.append(nn.InstanceNorm2d(oc, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        b = base
        self.model = nn.Sequential(
            block(2,    b,   norm=False),   # 2 input channels (noisy + target)
            block(b,    b*2),
            block(b*2,  b*4),
            block(b*4,  b*8, stride=1),     # stride 1 — begin patch scoring
            nn.Conv2d(b*8, 1, 4, stride=1, padding=1),  # patch logits
        )

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, noisy, target):
        x = torch.cat([noisy, target], dim=1)
        return self.model(x)


if __name__ == "__main__":
    d = PatchGAN(base=32)
    noisy  = torch.randn(2, 1, 256, 256)
    target = torch.randn(2, 1, 256, 256)
    out = d(noisy, target)
    print(f"OK: input {noisy.shape} -> patch logits {out.shape}")
