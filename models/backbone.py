"""
Lightweight CSPDarknet-style backbone (YOLOv8-flavored).
Produces multi-scale feature maps at strides 8, 16, 32 (P3, P4, P5),
which feed into the transformer neck.
"""
import torch
import torch.nn as nn


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class ConvBNAct(nn.Module):
    """Conv2d -> BatchNorm -> SiLU, the basic building block."""
    def __init__(self, c_in, c_out, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard residual bottleneck used inside C2f blocks."""
    def __init__(self, c_in, c_out, shortcut=True, e=0.5):
        super().__init__()
        c_hidden = int(c_out * e)
        self.cv1 = ConvBNAct(c_in, c_hidden, 3, 1)
        self.cv2 = ConvBNAct(c_hidden, c_out, 3, 1)
        self.add = shortcut and c_in == c_out

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f(nn.Module):
    """CSP-style block with 'n' bottlenecks and a cross-stage split,
    the core building block of YOLOv8's backbone/neck."""
    def __init__(self, c_in, c_out, n=1, shortcut=True, e=0.5):
        super().__init__()
        self.c_hidden = int(c_out * e)
        self.cv1 = ConvBNAct(c_in, 2 * self.c_hidden, 1, 1)
        self.cv2 = ConvBNAct((2 + n) * self.c_hidden, c_out, 1, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c_hidden, self.c_hidden, shortcut, e=1.0) for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast: cheap multi-scale receptive field."""
    def __init__(self, c_in, c_out, k=5):
        super().__init__()
        c_hidden = c_in // 2
        self.cv1 = ConvBNAct(c_in, c_hidden, 1, 1)
        self.cv2 = ConvBNAct(c_hidden * 4, c_out, 1, 1)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))


class CSPDarknetBackbone(nn.Module):
    """
    Returns a dict of feature maps at three scales:
      P3 (stride 8)  -> small objects  (distant pedestrians, signs)
      P4 (stride 16) -> medium objects (cars, cyclists)
      P5 (stride 32) -> large objects  (trucks/buses close-up), also
                         has the largest receptive field, good input
                         for the transformer neck's global reasoning.
    width_mult / depth_mult let you scale it like YOLOv8 n/s/m/l/x.
    """
    def __init__(self, in_channels=3, width_mult=0.5, depth_mult=0.33):
        super().__init__()

        def w(c):
            return max(8, int(c * width_mult + 4) // 8 * 8)

        def d(n):
            return max(1, round(n * depth_mult))

        self.stem = ConvBNAct(in_channels, w(64), 3, 2)                  # /2

        self.stage1 = nn.Sequential(
            ConvBNAct(w(64), w(128), 3, 2),                              # /4
            C2f(w(128), w(128), n=d(3), shortcut=True),
        )
        self.stage2 = nn.Sequential(
            ConvBNAct(w(128), w(256), 3, 2),                             # /8  -> P3
            C2f(w(256), w(256), n=d(6), shortcut=True),
        )
        self.stage3 = nn.Sequential(
            ConvBNAct(w(256), w(512), 3, 2),                             # /16 -> P4
            C2f(w(512), w(512), n=d(6), shortcut=True),
        )
        self.stage4 = nn.Sequential(
            ConvBNAct(w(512), w(1024), 3, 2),                            # /32 -> P5
            C2f(w(1024), w(1024), n=d(3), shortcut=True),
            SPPF(w(1024), w(1024)),
        )

        self.out_channels = {"P3": w(256), "P4": w(512), "P5": w(1024)}

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        return {"P3": p3, "P4": p4, "P5": p5}


if __name__ == "__main__":
    m = CSPDarknetBackbone()
    x = torch.randn(1, 3, 640, 640)
    out = m(x)
    for k, v in out.items():
        print(k, v.shape)
