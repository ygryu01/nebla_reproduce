#!/usr/bin/env python3
"""
Validation MIP rendering and visualization utilities for NEBLA training.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from models.mlp import gather_image_features
from models.point_embbeder import (
    encode_points_from_indices,
    sample_indexed_points_batched,
)

def _unit_to_uint8(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0 + 0.5).astype(np.uint8)


def save_mip_panel(
    pred_mip: np.ndarray,
    gt_mip: np.ndarray,
    err_mip: np.ndarray,
    out_prefix: Path,
    view_name: str = "MIP",
) -> None:
    """
    Save predicted MIP, GT MIP, absolute-error MIP, and a side-by-side panel.
    Values are assumed to be normalized to [0, 1].
    """
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    pred_u8 = _unit_to_uint8(pred_mip)
    gt_u8 = _unit_to_uint8(gt_mip)
    err_u8 = _unit_to_uint8(err_mip)

    Image.fromarray(pred_u8).save(str(out_prefix) + "_pred_mip.png")
    Image.fromarray(gt_u8).save(str(out_prefix) + "_gt_mip.png")
    Image.fromarray(err_u8).save(str(out_prefix) + "_abs_err_mip.png")

    h, w = pred_u8.shape
    label_h = 20
    gap = 6
    panel = Image.new("L", (w * 3 + gap * 2, h + label_h), color=0)
    panel.paste(Image.fromarray(gt_u8), (0, label_h))
    panel.paste(Image.fromarray(pred_u8), (w + gap, label_h))
    panel.paste(Image.fromarray(err_u8), (2 * (w + gap), label_h))

    try:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(panel)
        draw.text((4, 3), f"GT {view_name}", fill=255)
        draw.text((w + gap + 4, 3), f"Pred {view_name}", fill=255)
        draw.text((2 * (w + gap) + 4, 3), "Abs Error", fill=255)
    except Exception:
        pass

    panel.save(str(out_prefix) + "_panel.png")


@torch.no_grad()
def render_scatter_volume_mip_for_item(
    image_encoder: nn.Module,
    mlp: nn.Module,
    refiner: Union[nn.Module, None],
    simpx: torch.Tensor,
    volume: torch.Tensor,
    start_xyz: torch.Tensor,
    end_xyz: torch.Tensor,
    volume_shape: Tuple[int, int, int],
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Render validation MIP through a sparse 3D voxel volume.

    The model is still queried at ray samples (z_idx, r_idx, k_idx), but each
    predicted value is placed back into a Cartesian CBCT grid using its voxel
    coordinate point_xyz = [x, y, z]. The final MIP is computed from
    pred_volume[D, H, W], not directly from the ray-coordinate tensor [Z, R, K].

    Scatter rule:
        pred_sum[z, y, x]   += pred(x, y, z)
        pred_count[z, y, x] += 1
        pred_volume          = pred_sum / pred_count

    Empty voxels remain zero. Since the network output is sigmoid-normalized to
    [0, 1], empty zeros do not create artificial bright structures in the MIP.
    """
    image_encoder.eval()
    mlp.eval()
    if refiner is not None:
        refiner.eval()

    simpx = simpx.to(device, non_blocking=True).float()
    volume = volume.to(device, non_blocking=True).float()
    start_xyz = start_xyz.to(device, non_blocking=True).float()
    end_xyz = end_xyz.to(device, non_blocking=True).float()

    D, H, W = map(int, volume_shape)
    _, _, Z, R = simpx.shape

    k_stride = max(int(args.val_mip_k_stride), 1)
    k_values = torch.arange(0, int(args.n_samples), k_stride, device=device, dtype=torch.long)
    if k_values.numel() == 0:
        k_values = torch.zeros(1, device=device, dtype=torch.long)
    K_vis = int(k_values.numel())

    volume_numel = D * H * W
    pred_sum = torch.zeros(volume_numel, device=device, dtype=torch.float32)
    pred_count = torch.zeros(volume_numel, device=device, dtype=torch.float32)

    with autocast(enabled=args.amp and device.type == "cuda"):
        feature_map = image_encoder(simpx)

    chunk_points = max(int(args.val_image_chunk_points), K_vis)
    zr_chunk = max(chunk_points // K_vis, 1)

    for zr0 in range(0, Z * R, zr_chunk):
        zr1 = min(zr0 + zr_chunk, Z * R)
        M = zr1 - zr0

        zr_lin = torch.arange(zr0, zr1, device=device, dtype=torch.long)
        z_base = torch.div(zr_lin, R, rounding_mode="floor")
        r_base = zr_lin % R

        z_idx = z_base.repeat_interleave(K_vis).view(1, -1)
        r_idx = r_base.repeat_interleave(K_vis).view(1, -1)
        k_idx = k_values.repeat(M).view(1, -1)

        with autocast(enabled=args.amp and device.type == "cuda"):
            gamma_x = encode_points_from_indices(
                start_xyz=start_xyz,
                end_xyz=end_xyz,
                z_idx=z_idx,
                r_idx=r_idx,
                k_idx=k_idx,
                n_samples=args.n_samples,
                volume_shape_zyx=volume_shape,
                multires=7,
            )

            image_feat = gather_image_features(feature_map, z_idx, r_idx)

            pred = mlp(
                gamma_x.reshape(-1, 42),
                image_feat.reshape(-1, 128),
            ).reshape(-1)

            points_xyz = sample_indexed_points_batched(
                start_xyz=start_xyz,
                end_xyz=end_xyz,
                z_idx=z_idx,
                r_idx=r_idx,
                k_idx=k_idx,
                n_samples=args.n_samples,
                volume_shape_zyx=volume_shape,
                normalize=False,
            ).reshape(-1, 3)

        # point_xyz is [x, y, z], while volume indexing is [z, y, x].
        x = points_xyz[:, 0].round().long().clamp(0, W - 1)
        y = points_xyz[:, 1].round().long().clamp(0, H - 1)
        z = points_xyz[:, 2].round().long().clamp(0, D - 1)
        flat_idx = z * (H * W) + y * W + x

        pred_f = pred.float()
        pred_sum.scatter_add_(0, flat_idx, pred_f)
        pred_count.scatter_add_(0, flat_idx, torch.ones_like(pred_f))

    pred_volume = (pred_sum / pred_count.clamp_min(1.0)).view(D, H, W)

    if refiner is not None:
        with autocast(enabled=args.amp and device.type == "cuda"):
            pred_volume = refiner(pred_volume.view(1, 1, D, H, W))[0, 0].float()

    # Axial MIP: max over depth/z, resulting shape [H, W].
    pred_mip = pred_volume.amax(dim=0).detach().cpu().numpy()
    gt_mip = volume[0].float().amax(dim=0).detach().cpu().numpy()
    err_mip = np.abs(pred_mip - gt_mip)

    # Front/coronal MIP: max over y, resulting shape [D, W].
    # With volume convention [D, H, W] = [z, y, x], this is the front-facing view.
    pred_front_mip = pred_volume.amax(dim=1).detach().cpu().numpy()
    gt_front_mip = volume[0].float().amax(dim=1).detach().cpu().numpy()
    err_front_mip = np.abs(pred_front_mip - gt_front_mip)

    return pred_mip, gt_mip, err_mip, pred_front_mip, gt_front_mip, err_front_mip


@torch.no_grad()
def save_validation_mip_images(
    image_encoder: nn.Module,
    mlp: nn.Module,
    refiner: Union[nn.Module, None],
    loader: Union[DataLoader, None],
    args: argparse.Namespace,
    device: torch.device,
    global_step: int,
) -> None:
    """
    Save intermediate validation MIP images.
    Output layout:
        {out_dir}/val_images/step_XXXXXX/{sid}_panel.png
        {out_dir}/val_images/step_XXXXXX/{sid}_pred_mip.png
        {out_dir}/val_images/step_XXXXXX/{sid}_gt_mip.png
        {out_dir}/val_images/step_XXXXXX/{sid}_abs_err_mip.png
    """
    if loader is None or int(args.val_image_max_items) <= 0:
        return

    out_root = Path(args.out_dir) / "val_images" / f"step_{global_step:06d}"
    saved = 0

    image_encoder.eval()
    mlp.eval()
    if refiner is not None:
        refiner.eval()

    for batch in loader:
        B = len(batch["sid"])

        for bi in range(B):
            if saved >= int(args.val_image_max_items):
                print(f"[val image] saved {saved} item(s) to {out_root}")
                return

            sid = str(batch["sid"][bi])
            simpx = batch["simpx"][bi:bi + 1]
            volume = batch["volume"][bi:bi + 1]
            start_xyz = batch["start_xyz"][bi:bi + 1]
            end_xyz = batch["end_xyz"][bi:bi + 1]
            volume_shape = tuple(map(int, batch["volume_shape"][bi].tolist()))

            (
                pred_mip,
                gt_mip,
                err_mip,
                pred_front_mip,
                gt_front_mip,
                err_front_mip,
            ) = render_scatter_volume_mip_for_item(
                image_encoder=image_encoder,
                mlp=mlp,
                refiner=refiner,
                simpx=simpx,
                volume=volume,
                start_xyz=start_xyz,
                end_xyz=end_xyz,
                volume_shape=volume_shape,
                args=args,
                device=device,
            )

            # Existing axial MIP outputs are preserved:
            #   {sid}_pred_mip.png, {sid}_gt_mip.png, {sid}_abs_err_mip.png, {sid}_panel.png
            out_prefix = out_root / f"{sid}"
            save_mip_panel(pred_mip, gt_mip, err_mip, out_prefix, view_name="Axial MIP")

            # Additional front/coronal MIP outputs:
            #   {sid}_front_pred_mip.png, {sid}_front_gt_mip.png,
            #   {sid}_front_abs_err_mip.png, {sid}_front_panel.png
            front_out_prefix = out_root / f"{sid}_front"
            save_mip_panel(
                pred_front_mip,
                gt_front_mip,
                err_front_mip,
                front_out_prefix,
                view_name="Front MIP",
            )
            saved += 1

    print(f"[val image] saved {saved} item(s) to {out_root}")
