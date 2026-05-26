#!/usr/bin/env python3
"""
refinement_ops.py

Differentiable scatter/gather and projection losses for adding the NeBLa-style
3D U-Net refinement module to the current ray-conditioned MLP training path.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def ensure_volume_5d(volume: torch.Tensor) -> torch.Tensor:
    """Convert [B,D,H,W] to [B,1,D,H,W]; leave [B,C,D,H,W] unchanged."""
    if volume.ndim == 4:
        return volume.unsqueeze(1)
    if volume.ndim == 5:
        return volume
    raise ValueError(f"Expected volume [B,D,H,W] or [B,C,D,H,W], got {tuple(volume.shape)}")


def scatter_points_to_volume(
    values: torch.Tensor,
    points_xyz: torch.Tensor,
    volume_shape_zyx: Tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Scatter point-wise predictions into a dense Cartesian 3D grid.

    Args:
        values:
            [B, N] or [B, N, 1] predicted density/intensity values.
        points_xyz:
            [B, N, 3] continuous voxel coordinates in [x, y, z] order.
        volume_shape_zyx:
            (D, H, W), where volume indexing is [z, y, x].

    Returns:
        rho:
            [B, 1, D, H, W] mean value per voxel. Empty voxels are zero.
        count:
            [B, 1, D, H, W] number of samples accumulated per voxel.

    This implements the NeBLa-style crude density aggregation:
        rho(I, x) = average of F(x, e(I,p)) over rays/pixels p crossing x.
    """
    if values.ndim == 3:
        if values.shape[-1] != 1:
            raise ValueError(f"values last dim must be 1, got {tuple(values.shape)}")
        values = values[..., 0]
    if values.ndim != 2:
        raise ValueError(f"Expected values [B,N] or [B,N,1], got {tuple(values.shape)}")
    if points_xyz.ndim != 3 or points_xyz.shape[-1] != 3:
        raise ValueError(f"Expected points_xyz [B,N,3], got {tuple(points_xyz.shape)}")
    if points_xyz.shape[:2] != values.shape[:2]:
        raise ValueError(f"values and points shape mismatch: {tuple(values.shape)} vs {tuple(points_xyz.shape)}")

    B, N = values.shape
    D, H, W = map(int, volume_shape_zyx)
    device = values.device
    dtype = values.dtype
    volume_numel = D * H * W

    x = points_xyz[..., 0].round().long().clamp(0, W - 1)
    y = points_xyz[..., 1].round().long().clamp(0, H - 1)
    z = points_xyz[..., 2].round().long().clamp(0, D - 1)

    flat = z * (H * W) + y * W + x
    batch_offsets = (torch.arange(B, device=device, dtype=torch.long) * volume_numel).view(B, 1)
    flat = (flat + batch_offsets).reshape(-1)

    values_flat = values.reshape(-1).to(dtype=torch.float32)
    sum_flat = torch.zeros(B * volume_numel, device=device, dtype=torch.float32)
    count_flat = torch.zeros(B * volume_numel, device=device, dtype=torch.float32)

    sum_flat.scatter_add_(0, flat, values_flat)
    count_flat.scatter_add_(0, flat, torch.ones_like(values_flat))

    rho = sum_flat / count_flat.clamp_min(1.0)
    rho = rho.view(B, 1, D, H, W).to(dtype=dtype)
    count = count_flat.view(B, 1, D, H, W).to(dtype=dtype)
    return rho, count


def sample_volume_at_points(volume: torch.Tensor, points_xyz: torch.Tensor) -> torch.Tensor:
    """
    Trilinear sample a dense volume at continuous point coordinates.

    Args:
        volume: [B,D,H,W] or [B,1,D,H,W]
        points_xyz: [B,N,3] in voxel-index coordinates [x,y,z]

    Returns:
        sampled: [B,N,1]
    """
    volume = ensure_volume_5d(volume)
    if volume.shape[1] != 1:
        raise ValueError(f"Expected single-channel volume, got {tuple(volume.shape)}")
    B, _, D, H, W = volume.shape

    x = points_xyz[..., 0]
    y = points_xyz[..., 1]
    z = points_xyz[..., 2]

    gx = 2.0 * x / max(W - 1, 1) - 1.0
    gy = 2.0 * y / max(H - 1, 1) - 1.0
    gz = 2.0 * z / max(D - 1, 1) - 1.0
    grid = torch.stack([gx, gy, gz], dim=-1).view(B, -1, 1, 1, 3)

    sampled = F.grid_sample(
        volume,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled[:, 0, :, 0, 0].unsqueeze(-1)


def mip_projection_loss(pred_volume: torch.Tensor, target_volume: torch.Tensor) -> torch.Tensor:
    """
    MSE between axial, coronal/front, and sagittal maximum intensity projections.

    pred_volume / target_volume shape:
        [B,D,H,W] or [B,1,D,H,W]

    For [B,1,D,H,W]:
        axial   MIP: max over D -> [B,1,H,W]
        coronal MIP: max over H -> [B,1,D,W]
        sagittal MIP: max over W -> [B,1,D,H]
    """
    pred_volume = ensure_volume_5d(pred_volume)
    target_volume = ensure_volume_5d(target_volume).to(dtype=pred_volume.dtype, device=pred_volume.device)

    losses = []
    # D-axis / axial projection.
    losses.append(F.mse_loss(pred_volume.amax(dim=2), target_volume.amax(dim=2)))
    # H-axis / coronal-front projection.
    losses.append(F.mse_loss(pred_volume.amax(dim=3), target_volume.amax(dim=3)))
    # W-axis / sagittal projection.
    losses.append(F.mse_loss(pred_volume.amax(dim=4), target_volume.amax(dim=4)))
    return sum(losses) / len(losses)


def nebla_refinement_loss(
    refined_volume,
    target_volume,
    refined_points=None,
    target_points=None,
    point_weight: float = 0.0,
    volume_weight: float = 1.0,
    proj_weight: float = 10.0,
    perc_weight: float = 1.0,
    perceptual_loss_fn=None,
):
    if target_volume.ndim == 4:
        target_volume_5d = target_volume[:, None, ...]
    else:
        target_volume_5d = target_volume

    volume_mse = F.mse_loss(refined_volume, target_volume_5d)

    pred_axial = refined_volume.amax(dim=2)
    gt_axial = target_volume_5d.amax(dim=2)

    pred_coronal = refined_volume.amax(dim=3)
    gt_coronal = target_volume_5d.amax(dim=3)

    pred_sagittal = refined_volume.amax(dim=4)
    gt_sagittal = target_volume_5d.amax(dim=4)

    mip_proj = (
        F.mse_loss(pred_axial, gt_axial)
        + F.mse_loss(pred_coronal, gt_coronal)
        + F.mse_loss(pred_sagittal, gt_sagittal)
    ) / 3.0

    if refined_points is not None and target_points is not None:
        point_mse = F.mse_loss(refined_points, target_points)
    else:
        point_mse = refined_volume.new_tensor(0.0)

    if perceptual_loss_fn is not None and perc_weight > 0.0:
        perc = perceptual_loss_fn(refined_volume.float(), target_volume_5d.float())
    else:
        perc = refined_volume.new_tensor(0.0)

    loss = (
        volume_weight * volume_mse
        + proj_weight * mip_proj
        + point_weight * point_mse
        + perc_weight * perc
    )

    loss_terms = {
        "volume_mse": volume_mse.detach(),
        "mip_proj": mip_proj.detach(),
        "point_mse": point_mse.detach(),
        "perc": perc.detach(),
    }

    return loss, loss_terms


if __name__ == "__main__":
    B, D, H, W, N = 2, 8, 16, 16, 1024
    pred = torch.rand(B, N, 1)
    pts = torch.stack(
        [
            torch.rand(B, N) * (W - 1),
            torch.rand(B, N) * (H - 1),
            torch.rand(B, N) * (D - 1),
        ],
        dim=-1,
    )
    rho, count = scatter_points_to_volume(pred, pts, (D, H, W))
    sampled = sample_volume_at_points(rho, pts)
    print("rho:", tuple(rho.shape), "count:", tuple(count.shape), "sampled:", tuple(sampled.shape))
