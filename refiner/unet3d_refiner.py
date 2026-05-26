#!/usr/bin/env python3
"""
unet3d_refiner.py

3D U-Net refinement module for NeBLa-style reconstruction.

Design target:
    rho(I, x) sparse/crude volume  ->  3D U-Net  ->  sigma(I, x) refined volume

This is a compact, project-local implementation following the same structural
choices used by wolny/pytorch-3dunet's standard UNet3D:
    - encoder/decoder U-Net topology
    - DoubleConv blocks
    - GroupNorm + Conv3d + ReLU by default
    - MaxPool3d downsampling
    - nearest-neighbor interpolation upsampling
    - skip concatenation in the decoder

It is configured by default for the NeBLa paper's refinement module:
    f_maps = (64, 128, 256, 512)
    in_channels = 1
    out_channels = 1

Input shape:
    [B, 1, D, H, W] or [B, D, H, W]
Output shape:
    [B, 1, D, H, W]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn
import torch.nn.functional as F


def _as_tuple_of_ints(values: int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, int):
        return (values,)
    return tuple(int(v) for v in values)


def _valid_num_groups(num_channels: int, requested_groups: int) -> int:
    """Choose a GroupNorm group count that divides num_channels."""
    requested_groups = max(int(requested_groups), 1)
    if num_channels < requested_groups:
        return 1
    for g in range(requested_groups, 0, -1):
        if num_channels % g == 0:
            return g
    return 1


class SingleConv3D(nn.Sequential):
    """
    Single convolutional unit.

    layer_order follows the convention used by pytorch-3dunet:
        c: Conv3d
        g: GroupNorm
        b: BatchNorm3d
        r: ReLU
        l: LeakyReLU
        e: ELU
        d: Dropout3d

    Examples:
        "gcr" = GroupNorm -> Conv3d -> ReLU
        "crg" = Conv3d -> ReLU -> GroupNorm
        "cbr" = Conv3d -> BatchNorm3d -> ReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int] = 3,
        padding: int | tuple[int, int, int] = 1,
        layer_order: str = "gcr",
        num_groups: int = 8,
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        if "c" not in layer_order:
            raise ValueError(f"layer_order must contain 'c' for Conv3d, got {layer_order!r}")
        if layer_order[0] in "rle":
            raise ValueError(f"Non-linearity cannot be first in layer_order, got {layer_order!r}")

        for i, char in enumerate(layer_order):
            if char == "c":
                bias = not ("g" in layer_order or "b" in layer_order)
                self.add_module(
                    "conv",
                    nn.Conv3d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        bias=bias,
                    ),
                )
            elif char == "g":
                before_conv = i < layer_order.index("c")
                channels = in_channels if before_conv else out_channels
                groups = _valid_num_groups(channels, num_groups)
                self.add_module("groupnorm", nn.GroupNorm(groups, channels))
            elif char == "b":
                before_conv = i < layer_order.index("c")
                channels = in_channels if before_conv else out_channels
                self.add_module("batchnorm", nn.BatchNorm3d(channels))
            elif char == "r":
                self.add_module("relu", nn.ReLU(inplace=True))
            elif char == "l":
                self.add_module("leaky_relu", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            elif char == "e":
                self.add_module("elu", nn.ELU(inplace=True))
            elif char == "d":
                self.add_module("dropout", nn.Dropout3d(p=float(dropout_prob)))
            else:
                raise ValueError(f"Unsupported layer_order character {char!r} in {layer_order!r}")


class DoubleConv3D(nn.Sequential):
    """Two consecutive SingleConv3D blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        encoder: bool,
        kernel_size: int | tuple[int, int, int] = 3,
        padding: int | tuple[int, int, int] = 1,
        layer_order: str = "gcr",
        num_groups: int = 8,
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()

        if encoder:
            # Same channel policy as wolny/pytorch-3dunet DoubleConv:
            # first conv can map to out_channels//2, but not below in_channels.
            mid_channels = max(in_channels, out_channels // 2)
        else:
            mid_channels = out_channels

        self.add_module(
            "conv1",
            SingleConv3D(
                in_channels,
                mid_channels,
                kernel_size=kernel_size,
                padding=padding,
                layer_order=layer_order,
                num_groups=num_groups,
                dropout_prob=dropout_prob,
            ),
        )
        self.add_module(
            "conv2",
            SingleConv3D(
                mid_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                layer_order=layer_order,
                num_groups=num_groups,
                dropout_prob=dropout_prob,
            ),
        )


class Encoder3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        apply_pooling: bool,
        pool_kernel_size: int | tuple[int, int, int] = 2,
        **conv_kwargs,
    ) -> None:
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=pool_kernel_size) if apply_pooling else nn.Identity()
        self.block = DoubleConv3D(in_channels, out_channels, encoder=True, **conv_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.pool(x))


class Decoder3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample_mode: str = "nearest",
        **conv_kwargs,
    ) -> None:
        super().__init__()
        self.upsample_mode = str(upsample_mode)
        self.block = DoubleConv3D(in_channels, out_channels, encoder=False, **conv_kwargs)

    def forward(self, encoder_features: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=encoder_features.shape[2:], mode=self.upsample_mode)
        x = torch.cat([encoder_features, x], dim=1)
        return self.block(x)


@dataclass(frozen=True)
class NeBLa3DUNetConfig:
    in_channels: int = 1
    out_channels: int = 1
    f_maps: tuple[int, ...] = (64, 128, 256, 512)
    layer_order: str = "gcr"
    num_groups: int = 8
    dropout_prob: float = 0.0
    final_activation: str = "sigmoid"  # "sigmoid", "none"


class NeBLa3DUNetRefiner(nn.Module):
    """
    3D U-Net refinement module for NeBLa-style density volumes.

    The default feature hierarchy (64, 128, 256, 512) matches the NeBLa paper's
    reported 3D U-Net hidden dimensions. The module is fully convolutional, so it
    can process [128, 256, 256] as in the paper, or project-specific shapes such
    as [60, 200, 350], subject to GPU memory.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        f_maps: Sequence[int] = (64, 128, 256, 512),
        layer_order: str = "gcr",
        num_groups: int = 8,
        dropout_prob: float = 0.0,
        final_activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        f_maps = _as_tuple_of_ints(f_maps)
        if len(f_maps) < 2:
            raise ValueError(f"f_maps must contain at least two levels, got {f_maps}")
        if final_activation not in {"sigmoid", "none", "identity", ""}:
            raise ValueError("final_activation must be one of: sigmoid, none, identity, ''.")

        conv_kwargs = dict(
            kernel_size=3,
            padding=1,
            layer_order=layer_order,
            num_groups=num_groups,
            dropout_prob=dropout_prob,
        )

        encoders: list[nn.Module] = []
        for i, out_ch in enumerate(f_maps):
            enc_in = in_channels if i == 0 else f_maps[i - 1]
            encoders.append(
                Encoder3D(
                    enc_in,
                    out_ch,
                    apply_pooling=(i != 0),
                    pool_kernel_size=2,
                    **conv_kwargs,
                )
            )
        self.encoders = nn.ModuleList(encoders)

        decoders: list[nn.Module] = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            # concat(current decoder tensor, matching encoder skip)
            in_ch = reversed_f_maps[i] + reversed_f_maps[i + 1]
            out_ch = reversed_f_maps[i + 1]
            decoders.append(Decoder3D(in_ch, out_ch, upsample_mode="nearest", **conv_kwargs))
        self.decoders = nn.ModuleList(decoders)

        self.final_conv = nn.Conv3d(f_maps[0], out_channels, kernel_size=1)
        self.final_activation = final_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x.unsqueeze(1)
        if x.ndim != 5:
            raise ValueError(f"Expected [B,D,H,W] or [B,C,D,H,W], got shape {tuple(x.shape)}")

        encoder_features: list[torch.Tensor] = []
        for encoder in self.encoders:
            x = encoder(x)
            encoder_features.insert(0, x)

        # Remove bottleneck from skip list.
        skips = encoder_features[1:]
        for decoder, skip in zip(self.decoders, skips, strict=False):
            x = decoder(skip, x)

        x = self.final_conv(x)
        if self.final_activation == "sigmoid":
            x = torch.sigmoid(x)
        return x


def build_nebla_3d_unet_refiner(
    in_channels: int = 1,
    out_channels: int = 1,
    f_maps: Sequence[int] = (64, 128, 256, 512),
    final_activation: str = "sigmoid",
) -> NeBLa3DUNetRefiner:
    return NeBLa3DUNetRefiner(
        in_channels=in_channels,
        out_channels=out_channels,
        f_maps=f_maps,
        layer_order="gcr",
        num_groups=8,
        dropout_prob=0.0,
        final_activation=final_activation,
    )


if __name__ == "__main__":
    # Small smoke test. Use a smaller volume than 128x256x256 to avoid CPU/GPU memory pressure.
    model = build_nebla_3d_unet_refiner(f_maps=(8, 16, 32, 64))
    x = torch.randn(1, 1, 16, 32, 32)
    y = model(x)
    print("input:", tuple(x.shape), "output:", tuple(y.shape))
