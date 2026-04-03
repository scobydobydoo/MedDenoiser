import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):


    def __init__(self, base=32):
        super().__init__()

        def enc_block(ic, oc, norm=True):
            layers = [nn.Conv2d(ic, oc, 4, stride=2, padding=1, bias=not norm)]
            if norm:
                layers.append(nn.InstanceNorm2d(oc, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        def dec_block(ic, oc, dropout=False):
            layers = [
                nn.ConvTranspose2d(ic, oc, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(oc, affine=True),
                nn.ReLU(inplace=True),
            ]
            if dropout:
                layers.append(nn.Dropout(0.3))
            return nn.Sequential(*layers)

        b = base
        self.e1 = enc_block(1,    b,    norm=False)   # 1   -> b
        self.e2 = enc_block(b,    b*2)               # b   -> b*2
        self.e3 = enc_block(b*2,  b*4)               # b*2 -> b*4
        self.e4 = enc_block(b*4,  b*8)               # b*4 -> b*8

        self.bn = nn.Sequential(
            nn.Conv2d(b*8, b*8, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.d1 = dec_block(b*8 + b*8, b*8, dropout=True)   # btn+e4 -> b*8
        self.d2 = dec_block(b*8 + b*4, b*4)                 # d1+e3  -> b*4
        self.d3 = dec_block(b*4 + b*2, b*2)                 # d2+e2  -> b*2
        self.d4 = dec_block(b*2 + b,   b)                   # d3+e1  -> b

       
        self.out = nn.Sequential(
            nn.Conv2d(b + 1, 1, 3, padding=1),   
            nn.Tanh(),
        )

    @staticmethod
    def _pad(h, ref):
        """Align h to ref's H×W (handles ±1 px from odd dims)."""
        if h.shape[2:] != ref.shape[2:]:
            h = F.interpolate(h, size=ref.shape[2:], mode='bilinear', align_corners=False)
        return h

    def forward(self, x):
       
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)

        
        bn = self.bn(e4)

        
        d1 = self.d1(torch.cat([self._pad(bn, e4), e4], dim=1))
        d2 = self.d2(torch.cat([self._pad(d1, e3), e3], dim=1))
        d3 = self.d3(torch.cat([self._pad(d2, e2), e2], dim=1))
        d4 = self.d4(torch.cat([self._pad(d3, e1), e1], dim=1))

        
        d4 = self._pad(d4, x)
        return self.out(torch.cat([d4, x], dim=1))


if __name__ == "__main__":
    m = UNet(base=32)
    t = torch.randn(2, 1, 256, 256)
    o = m(t)
    assert o.shape == t.shape, f"Shape mismatch: {o.shape}"
    print(f"OK: {t.shape} -> {o.shape}  params={sum(p.numel() for p in m.parameters()):,}")
