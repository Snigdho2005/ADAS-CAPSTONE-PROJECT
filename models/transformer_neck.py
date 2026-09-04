"""
Lightweight windowed-attention Transformer neck — the core contribution
of the project. Replaces / augments a standard FPN+PANet with a few
Swin-style self-attention blocks so feature maps get *global* context
before detection, while keeping compute roughly linear in resolution
(windowed attention, not full quadratic attention like plain ViT/DETR).

Design choices, and why (useful for the ablation study + your writeup):
  - Operates on P5 (stride 32, smallest spatial size) primarily, where
    quadratic-cost full attention is still affordable, then fuses back
    down through a standard FPN/PANet path -> keeps the expensive part
    small and cheap-to-ablate (num_transformer_layers is a single knob).
  - Windowed (not global) attention at P4 for extra context at
    manageable cost; P3 stays pure-CNN (too large spatially for
    attention to be worth the FLOPs at this scale).
  - Shifted windows (Swin-style) let information flow across window
    boundaries every other block, avoiding the "blind" seams that
    plain fixed-window attention would produce.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import ConvBNAct, C2f


def window_partition(x, window_size):
    """x: (B, H, W, C) -> (num_windows*B, window_size, window_size, C)"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

        # relative position bias, standard Swin trick, cheap and helps a lot
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        # x: (num_windows*B, N, C)  where N = window_size*window_size
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.to(q.dtype)

        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(N, N, -1).permute(2, 0, 1).contiguous()
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0).to(q.dtype)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)


class SwinBlock(nn.Module):
    """One windowed (optionally shifted) self-attention block + MLP,
    pre-norm, residual — a single lightweight Transformer layer."""
    def __init__(self, dim, num_heads=4, window_size=7, shift=False, mlp_ratio=2.0):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2 if shift else 0
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def _attn_mask(self, H, W, device):
        if self.shift_size == 0:
            return None
        img_mask = torch.zeros((1, H, W, 1), device=device)
        slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size),
                  slice(-self.shift_size, None))
        cnt = 0
        for h in slices:
            for w_ in slices:
                img_mask[:, h, w_, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size).view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        # x: (B, C, H, W) -> internally (B, H, W, C)
        B, C, H, W = x.shape
        ws = self.window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x = x.permute(0, 2, 3, 1)  # B H W C
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w

        shortcut = x
        x = self.norm1(x)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        windows = window_partition(x, ws).view(-1, ws * ws, C)
        mask = self._attn_mask(Hp, Wp, x.device)
        attn_out = self.attn(windows, mask=mask)
        attn_out = attn_out.view(-1, ws, ws, C)
        x = window_reverse(attn_out, ws, Hp, Wp)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        if pad_h or pad_w:
            x = x[:, :H, :W, :]
        return x.permute(0, 3, 1, 2)  # back to B C H W


class TransformerNeck(nn.Module):
    """
    Fuses backbone features {P3, P4, P5} with:
      - `num_transformer_layers` Swin blocks applied at P5 (and
        optionally P4) for global context
      - a standard top-down + bottom-up FPN/PANet path to spread that
        context back out to all three scales

    Set num_transformer_layers=0 to get a pure-CNN FPN/PANet neck —
    this is exactly the knob your ablation study sweeps.
    """
    def __init__(self, channels, num_transformer_layers=2, window_size=7,
                 num_heads=4, apply_attn_to_p4=True):
        super().__init__()
        c3, c4, c5 = channels["P3"], channels["P4"], channels["P5"]
        self.num_transformer_layers = num_transformer_layers

        # transformer blocks on P5 (and optionally P4), alternating shift
        self.p5_blocks = nn.ModuleList([
            SwinBlock(c5, num_heads=num_heads, window_size=window_size, shift=(i % 2 == 1))
            for i in range(num_transformer_layers)
        ])
        self.apply_attn_to_p4 = apply_attn_to_p4
        if apply_attn_to_p4 and num_transformer_layers > 0:
            self.p4_blocks = nn.ModuleList([
                SwinBlock(c4, num_heads=num_heads, window_size=window_size, shift=(i % 2 == 1))
                for i in range(max(1, num_transformer_layers // 2))
            ])
        else:
            self.p4_blocks = nn.ModuleList()

        # standard top-down path
        self.reduce_p5 = ConvBNAct(c5, c4, 1, 1)
        self.td_c2f_p4 = C2f(c4 * 2, c4, n=2, shortcut=False)
        self.reduce_p4 = ConvBNAct(c4, c3, 1, 1)
        self.td_c2f_p3 = C2f(c3 * 2, c3, n=2, shortcut=False)

        # bottom-up path
        self.down_p3 = ConvBNAct(c3, c3, 3, 2)
        self.bu_c2f_p4 = C2f(c3 + c4, c4, n=2, shortcut=False)
        self.down_p4 = ConvBNAct(c4, c4, 3, 2)
        self.bu_c2f_p5 = C2f(c4 + c5, c5, n=2, shortcut=False)

    def forward(self, feats):
        p3, p4, p5 = feats["P3"], feats["P4"], feats["P5"]

        for blk in self.p5_blocks:
            p5 = blk(p5)
        for blk in self.p4_blocks:
            p4 = blk(p4)

        # top-down
        p5_up = F.interpolate(self.reduce_p5(p5), scale_factor=2, mode="nearest")
        p4_td = self.td_c2f_p4(torch.cat([p5_up, p4], 1))
        p4_up = F.interpolate(self.reduce_p4(p4_td), scale_factor=2, mode="nearest")
        p3_out = self.td_c2f_p3(torch.cat([p4_up, p3], 1))

        # bottom-up
        p3_down = self.down_p3(p3_out)
        p4_out = self.bu_c2f_p4(torch.cat([p3_down, p4_td], 1))
        p4_down = self.down_p4(p4_out)
        p5_out = self.bu_c2f_p5(torch.cat([p4_down, p5], 1))

        return {"P3": p3_out, "P4": p4_out, "P5": p5_out}


if __name__ == "__main__":
    from models.backbone import CSPDarknetBackbone
    bb = CSPDarknetBackbone()
    neck = TransformerNeck(bb.out_channels, num_transformer_layers=2)
    x = torch.randn(1, 3, 640, 640)
    feats = bb(x)
    out = neck(feats)
    for k, v in out.items():
        print(k, v.shape)
