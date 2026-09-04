r"""
Full Hybrid CNN-Transformer ADAS detector: assembles
  CSPDarknetBackbone -> TransformerNeck -> DetectionHead
                                        \-> DrivableAreaHead (optional)
                                        \-> WeatherHead (optional)

`num_transformer_layers=0` gives you the pure-CNN baseline (for the
ablation study / fair comparison against plain YOLOv8) using the exact
same backbone/head code, which is important for a clean comparison —
only the neck changes.
"""
import torch
import torch.nn as nn

from models.backbone import CSPDarknetBackbone
from models.transformer_neck import TransformerNeck
from models.head import DetectionHead, DrivableAreaHead, WeatherHead


class HybridADASModel(nn.Module):
    def __init__(
        self,
        num_classes=10,
        width_mult=0.5,
        depth_mult=0.33,
        num_transformer_layers=2,
        window_size=7,
        num_heads=4,
        use_drivable_head=True,
        use_weather_head=True,
        reg_max=16,
    ):
        super().__init__()
        self.backbone = CSPDarknetBackbone(width_mult=width_mult, depth_mult=depth_mult)
        self.neck = TransformerNeck(
            self.backbone.out_channels,
            num_transformer_layers=num_transformer_layers,
            window_size=window_size,
            num_heads=num_heads,
        )
        self.det_head = DetectionHead(self.backbone.out_channels, num_classes=num_classes, reg_max=reg_max)

        self.use_drivable_head = use_drivable_head
        if use_drivable_head:
            self.drivable_head = DrivableAreaHead(self.backbone.out_channels["P3"])

        self.use_weather_head = use_weather_head
        if use_weather_head:
            self.weather_head = WeatherHead(self.backbone.out_channels["P5"])

    def forward(self, x):
        feats = self.backbone(x)
        neck_feats = self.neck(feats)

        out = self.det_head(neck_feats)

        if self.use_drivable_head:
            out["drivable_logits"] = self.drivable_head(neck_feats["P3"])
        if self.use_weather_head:
            weather_logits, time_logits = self.weather_head(neck_feats["P5"])
            out["weather_logits"] = weather_logits
            out["timeofday_logits"] = time_logits

        return out

    @torch.no_grad()
    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def build_model_variant(name="hybrid_s", num_classes=10, **kwargs):
    """Convenience presets for the ablation study."""
    presets = {
        # pure CNN baseline: 0 transformer layers, same backbone/head as hybrid
        "cnn_baseline":     dict(width_mult=0.50, depth_mult=0.33, num_transformer_layers=0),
        "hybrid_n":         dict(width_mult=0.25, depth_mult=0.33, num_transformer_layers=1),
        "hybrid_s":         dict(width_mult=0.50, depth_mult=0.33, num_transformer_layers=2),
        "hybrid_m":         dict(width_mult=0.75, depth_mult=0.67, num_transformer_layers=4),
    }
    cfg = presets[name].copy()
    cfg.update(kwargs)
    return HybridADASModel(num_classes=num_classes, **cfg)


if __name__ == "__main__":
    for name in ["cnn_baseline", "hybrid_n", "hybrid_s"]:
        m = build_model_variant(name, num_classes=10)
        x = torch.randn(1, 3, 640, 640)
        out = m(x)
        print(f"{name}: {m.count_params()/1e6:.2f}M params | "
              f"boxes {out['boxes'].shape} | cls {out['cls_logits'].shape}")
