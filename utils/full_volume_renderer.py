#!/usr/bin/env python3
"""
Full-volume inference utilities for NeBLa Stage 1.

The key function, render_full_rho_no_grad(), evaluates the trained image_encoder + MLP
on every SimPX pixel and every ray sample index, then scatters the predicted point values
back to a Cartesian CBCT grid.

Output convention:
    rho:   [B, 1, D, H, W], averaged MLP predictions per voxel
    mask:  [B, 1, D, H, W], 1 where at least one ray sample hit the voxel
    count: [B, 1, D, H, W], number of ray samples accumulated per voxel

Coordinate convention:
    points_xyz[..., 0] = x, points_xyz[..., 1] = y, points_xyz[..., 2] = z
    volume indexing is [z, y, x] = [D, H, W]
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torch.cuda.amp import autocast

from models.mlp import gather_image_features
from models.point_embbeder import (
    encode_points_from_indices,
    sample_indexed_points_batched,
)


@torch.no_grad()
def render_full_rho_no_grad(
    image_encoder: nn.Module,
    mlp: nn.Module,
    simpx: torch.Tensor,
    start_xyz: torch.Tensor,
    end_xyz: torch.Tensor,
    volume_shape_zyx: Tuple[int, int, int],
    n_samples: int,
    chunk_points: int = 131072,
    k_stride: int = 1,
    amp: bool = False,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Render a full intermediate density volume rho from a trained Stage 1 model.

    Args:
        image_encoder: trained 2D UNet image encoder.
        mlp: trained NeRF-style MLP.
        simpx: [B,1,Z,R] normalized SimPX tensor.
        start_xyz: [B,Z,R,3] or broadcast-compatible ray start tensor.
        end_xyz: [B,Z,R,3] or broadcast-compatible ray end tensor.
        volume_shape_zyx: (D,H,W).
        n_samples: number of samples along each ray, matching Stage 1 training.
        chunk_points: maximum approximate number of point samples per MLP chunk.
        k_stride: stride along the ray sample index. 1 uses every sample.
        amp: use autocast for image_encoder/MLP forward on CUDA.
        device: torch device. If None, inferred from simpx.

    Returns:
        rho:   [B,1,D,H,W]
        mask:  [B,1,D,H,W]
        count: [B,1,D,H,W]
    """
    if device is None:
        device = simpx.device

    image_encoder.eval()
    mlp.eval()

    simpx = simpx.to(device, non_blocking=True).float()
    start_xyz = start_xyz.to(device, non_blocking=True).float()
    end_xyz = end_xyz.to(device, non_blocking=True).float()

    B, _, Z, R = simpx.shape
    D, H, W = map(int, volume_shape_zyx)

    k_stride = max(int(k_stride), 1)
    k_values = torch.arange(0, int(n_samples), k_stride, device=device, dtype=torch.long)
    if k_values.numel() == 0:
        k_values = torch.zeros(1, device=device, dtype=torch.long)
    K = int(k_values.numel())

    volume_numel = int(D) * int(H) * int(W)
    pred_sum = torch.zeros(B, volume_numel, device=device, dtype=torch.float32)
    pred_count = torch.zeros(B, volume_numel, device=device, dtype=torch.float32)

    with autocast(enabled=bool(amp) and device.type == "cuda"):
        feature_map = image_encoder(simpx)

    chunk_points = max(int(chunk_points), K)
    zr_chunk = max(chunk_points // K, 1)

    for zr0 in range(0, Z * R, zr_chunk):
        zr1 = min(zr0 + zr_chunk, Z * R)
        M = zr1 - zr0

        zr_lin = torch.arange(zr0, zr1, device=device, dtype=torch.long)
        z_base = torch.div(zr_lin, R, rounding_mode="floor")
        r_base = zr_lin % R

        z_idx_1 = z_base.repeat_interleave(K).view(1, -1)
        r_idx_1 = r_base.repeat_interleave(K).view(1, -1)
        k_idx_1 = k_values.repeat(M).view(1, -1)

        # expand() is enough because index tensors are read-only.
        z_idx = z_idx_1.expand(B, -1)
        r_idx = r_idx_1.expand(B, -1)
        k_idx = k_idx_1.expand(B, -1)

        with autocast(enabled=bool(amp) and device.type == "cuda"):
            gamma_x = encode_points_from_indices(
                start_xyz=start_xyz,
                end_xyz=end_xyz,
                z_idx=z_idx,
                r_idx=r_idx,
                k_idx=k_idx,
                n_samples=n_samples,
                volume_shape_zyx=volume_shape_zyx,
                multires=7,
            )

            image_feat = gather_image_features(feature_map, z_idx, r_idx)

            pred = mlp(
                gamma_x.reshape(-1, 42),
                image_feat.reshape(-1, 128),
            ).reshape(B, -1)

            points_xyz = sample_indexed_points_batched(
                start_xyz=start_xyz,
                end_xyz=end_xyz,
                z_idx=z_idx,
                r_idx=r_idx,
                k_idx=k_idx,
                n_samples=n_samples,
                volume_shape_zyx=volume_shape_zyx,
                normalize=False,
            )

        x = points_xyz[..., 0].round().long().clamp(0, W - 1)
        y = points_xyz[..., 1].round().long().clamp(0, H - 1)
        z = points_xyz[..., 2].round().long().clamp(0, D - 1)
        flat = z * (H * W) + y * W + x

        pred_sum.scatter_add_(1, flat, pred.float())
        pred_count.scatter_add_(1, flat, torch.ones_like(pred, dtype=torch.float32))

    rho = pred_sum / pred_count.clamp_min(1.0)
    mask = (pred_count > 0).to(torch.float32)

    return (
        rho.view(B, 1, D, H, W),
        mask.view(B, 1, D, H, W),
        pred_count.view(B, 1, D, H, W),
    )
