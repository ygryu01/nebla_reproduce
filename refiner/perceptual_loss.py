from __future__ import annotations

from typing import Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights


class VGGPerceptualLoss2D(nn.Module):
    """
    2D VGG perceptual loss for normalized grayscale images in [0, 1].

    Input:
        x, y: [B, 1, H, W]
    """

    def __init__(
        self,
        layers: Sequence[int] = (3, 8, 15, 22),
        layer_weights: Sequence[float] | None = None,
        resize_to: Tuple[int, int] | None = None,
    ):
        super().__init__()

        weights = VGG16_Weights.IMAGENET1K_V1
        features = vgg16(weights=weights).features.eval()

        self.features = features
        for p in self.features.parameters():
            p.requires_grad_(False)

        self.layers = tuple(int(x) for x in layers)
        if layer_weights is None:
            layer_weights = [1.0] * len(self.layers)
        self.layer_weights = tuple(float(x) for x in layer_weights)
        self.resize_to = resize_to

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().clamp(0.0, 1.0)

        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"Expected [B,1,H,W], got {tuple(x.shape)}")

        if self.resize_to is not None:
            x = F.interpolate(x, size=self.resize_to, mode="bilinear", align_corners=False)

        x = x.repeat(1, 3, 1, 1)
        x = (x - self.mean) / self.std
        return x

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = self._prep(pred)
        target = self._prep(target)

        loss = pred.new_tensor(0.0)

        for i, layer in enumerate(self.features):
            pred = layer(pred)
            target = layer(target)

            if i in self.layers:
                j = self.layers.index(i)
                loss = loss + self.layer_weights[j] * F.l1_loss(pred, target)

            if i >= max(self.layers):
                break

        return loss


class VGGPerceptualLoss3DMIP(nn.Module):
    """
    Apply VGG perceptual loss to axial/coronal/sagittal MIP images.

    Input:
        pred_volume:   [B,1,D,H,W]
        target_volume: [B,D,H,W] or [B,1,D,H,W]
    """

    def __init__(self, resize_to: Tuple[int, int] | None = None):
        super().__init__()
        self.loss2d = VGGPerceptualLoss2D(resize_to=resize_to)

    @staticmethod
    def _ensure_5d(volume: torch.Tensor) -> torch.Tensor:
        if volume.ndim == 4:
            volume = volume[:, None, ...]
        if volume.ndim != 5 or volume.shape[1] != 1:
            raise ValueError(f"Expected [B,D,H,W] or [B,1,D,H,W], got {tuple(volume.shape)}")
        return volume

    @staticmethod
    def _mips(volume: torch.Tensor):
        # volume: [B,1,D,H,W]
        axial = volume.amax(dim=2)      # [B,1,H,W]
        coronal = volume.amax(dim=3)    # [B,1,D,W]
        sagittal = volume.amax(dim=4)   # [B,1,D,H]
        return axial, coronal, sagittal

    def forward(self, pred_volume: torch.Tensor, target_volume: torch.Tensor) -> torch.Tensor:
        pred_volume = self._ensure_5d(pred_volume)
        target_volume = self._ensure_5d(target_volume)

        pred_mips = self._mips(pred_volume)
        target_mips = self._mips(target_volume)

        loss = pred_volume.new_tensor(0.0)
        for pred_mip, target_mip in zip(pred_mips, target_mips):
            loss = loss + self.loss2d(pred_mip, target_mip)

        return loss / 3.0