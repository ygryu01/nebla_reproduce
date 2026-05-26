import torch
import torch.nn as nn
import torch.nn.functional as F


class NeRF(nn.Module):
    def __init__(
        self,
        D: int = 8,
        W: int = 128,
        input_ch: int = 42,
        image_ch: int = 128,
        output_ch: int = 1,
        skips=(4,),
    ):
        super().__init__()
        self.D = int(D)
        self.W = int(W)
        self.input_ch = int(input_ch)
        self.image_ch = int(image_ch)
        self.skips = set(int(s) for s in skips)

        self.first_pts_linear = nn.Linear(self.input_ch, self.W)
        self.image_linear = nn.Linear(self.image_ch, self.W)
        self.pts_linears = nn.ModuleList(
            [
                nn.Linear(self.W * 2, self.W) if i in self.skips else nn.Linear(self.W, self.W)
                for i in range(self.D - 1)
            ]
        )
        self.output_linear = nn.Linear(self.W, output_ch)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_ch:
            raise ValueError(f"Expected point feature dim {self.input_ch}, got {x.shape[-1]}")
        if c.shape[-1] != self.image_ch:
            raise ValueError(f"Expected image feature dim {self.image_ch}, got {c.shape[-1]}")

        base = self.first_pts_linear(x) + self.image_linear(c)
        h = base

        for i, layer in enumerate(self.pts_linears):
            if i in self.skips:
                h = torch.cat([base, h], dim=-1)
            h = F.relu(layer(h), inplace=True)

        return torch.sigmoid(self.output_linear(h))


def gather_image_features(feature_map: torch.Tensor, z_idx: torch.Tensor, r_idx: torch.Tensor) -> torch.Tensor:
    """
    Args:
        feature_map: [B, C, H, W]
        z_idx: [B, N]
        r_idx: [B, N]

    Returns:
        features: [B, N, C]
    """
    B, C, H, W = feature_map.shape

    z_idx = z_idx.long().clamp(0, H - 1)
    r_idx = r_idx.long().clamp(0, W - 1)

    fmap = feature_map.permute(0, 2, 3, 1).contiguous()
    b_idx = torch.arange(B, device=feature_map.device)[:, None]
    return fmap[b_idx, z_idx, r_idx]
