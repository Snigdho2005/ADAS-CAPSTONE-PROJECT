"""
Detection head (anchor-free, decoupled — YOLOv8 style) + auxiliary
heads for the "Scene Understanding" stretch goal:
  - DetectionHead: boxes + class scores at P3/P4/P5
  - DrivableAreaHead: pixel-wise road/sidewalk/off-road segmentation
  - WeatherHead: frame-level weather/timeofday classification

Kept as separate small heads so you can drop DrivableAreaHead /
WeatherHead entirely for the MVP and re-add them for the stretch goal
without touching the backbone/neck/detection-head code.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import ConvBNAct


class DFL(nn.Module):
    """Distribution Focal Loss integral module — turns a discrete
    probability distribution over `reg_max` bins into a continuous
    box-edge distance, the standard YOLOv8 anchor-free box encoding."""
    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max
        self.conv = nn.Conv2d(reg_max, 1, 1, bias=False)
        self.conv.weight.data[:] = torch.arange(reg_max, dtype=torch.float).view(1, reg_max, 1, 1)
        self.conv.weight.requires_grad_(False)

    def forward(self, x):
        # x: (B, 4*reg_max, N) -> (B, 4, N)
        B, C, N = x.shape
        x = x.view(B, 4, self.reg_max, N).transpose(2, 1).softmax(1)
        return self.conv(x).view(B, 4, N)


class DecoupledHeadBlock(nn.Module):
    """Per-scale decoupled cls/box branches, sharing nothing —
    YOLOv8 found this outperforms a shared-then-split head."""
    def __init__(self, c_in, num_classes, reg_max=16, c_mid=None):
        super().__init__()
        c_mid = c_mid or max(c_in // 2, 64)
        self.cls_branch = nn.Sequential(
            ConvBNAct(c_in, c_mid, 3, 1), ConvBNAct(c_mid, c_mid, 3, 1),
            nn.Conv2d(c_mid, num_classes, 1),
        )
        self.box_branch = nn.Sequential(
            ConvBNAct(c_in, c_mid, 3, 1), ConvBNAct(c_mid, c_mid, 3, 1),
            nn.Conv2d(c_mid, 4 * reg_max, 1),
        )

    def forward(self, x):
        return self.cls_branch(x), self.box_branch(x)


class DetectionHead(nn.Module):
    def __init__(self, channels, num_classes=10, reg_max=16, strides=(8, 16, 32)):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.strides = strides
        self.blocks = nn.ModuleDict({
            k: DecoupledHeadBlock(channels[k], num_classes, reg_max)
            for k in ["P3", "P4", "P5"]
        })
        self.dfl = DFL(reg_max)

    def _make_anchors(self, shapes, device, dtype):
        anchors, stride_tensor = [], []
        for (h, w), s in zip(shapes, self.strides):
            sy, sx = torch.meshgrid(
                torch.arange(h, device=device, dtype=dtype) + 0.5,
                torch.arange(w, device=device, dtype=dtype) + 0.5,
                indexing="ij",
            )
            anchors.append(torch.stack([sx, sy], -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), s, device=device, dtype=dtype))
        return torch.cat(anchors), torch.cat(stride_tensor)

    def forward(self, feats):
        cls_outs, box_outs, shapes = [], [], []
        for k in ["P3", "P4", "P5"]:
            cls_o, box_o = self.blocks[k](feats[k])
            B, _, H, W = cls_o.shape
            shapes.append((H, W))
            cls_outs.append(cls_o.view(B, self.num_classes, -1))
            box_outs.append(box_o.view(B, 4 * self.reg_max, -1))

        cls_cat = torch.cat(cls_outs, -1)                 # (B, num_classes, N)
        box_cat = torch.cat(box_outs, -1)                 # (B, 4*reg_max, N)
        dist = self.dfl(box_cat)                          # (B, 4, N) = l,t,r,b distances

        anchors, stride_tensor = self._make_anchors(shapes, cls_cat.device, cls_cat.dtype)
        anchors = anchors.unsqueeze(0)                     # (1, N, 2)
        stride_tensor = stride_tensor.unsqueeze(0)          # (1, N, 1)

        lt, rb = dist[:, :2, :], dist[:, 2:, :]
        lt = lt.transpose(1, 2)  # (B, N, 2)
        rb = rb.transpose(1, 2)
        x1y1 = anchors - lt
        x2y2 = anchors + rb
        boxes = torch.cat([x1y1, x2y2], -1) * stride_tensor  # (B, N, 4) in pixel coords, xyxy

        return {
            "cls_logits": cls_cat.transpose(1, 2),  # (B, N, num_classes) raw logits, apply sigmoid for scores
            "boxes": boxes,                          # (B, N, 4) xyxy, pixel space
            "anchors": anchors,
            "strides": stride_tensor,
        }


class DrivableAreaHead(nn.Module):
    """Lightweight segmentation head off P3 (highest-res feature map),
    3 classes: background / drivable / alternative-drivable (BDD100K
    convention)."""
    def __init__(self, c_p3, num_classes=3):
        super().__init__()
        self.up = nn.Sequential(
            ConvBNAct(c_p3, c_p3 // 2, 3, 1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNAct(c_p3 // 2, c_p3 // 4, 3, 1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNAct(c_p3 // 4, c_p3 // 4, 3, 1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c_p3 // 4, num_classes, 1),
        )

    def forward(self, p3_feat):
        return self.up(p3_feat)  # (B, num_classes, H, W) at input resolution (P3 is stride 8, 3x up = stride 1)


class WeatherHead(nn.Module):
    """Frame-level auxiliary classifier: weather x timeofday, pooled
    from P5 (most semantic / globally-contextualized feature map)."""
    def __init__(self, c_p5, num_weather=4, num_timeofday=3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_weather = nn.Linear(c_p5, num_weather)
        self.fc_time = nn.Linear(c_p5, num_timeofday)

    def forward(self, p5_feat):
        x = self.pool(p5_feat).flatten(1)
        return self.fc_weather(x), self.fc_time(x)
