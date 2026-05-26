#!/usr/bin/env python3
"""
Training/evaluation helper functions for NEBLA.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from dataset.dataset import SimPXCBCTDataset, collate_same_shape
from models.mlp import gather_image_features
from models.point_embbeder import (
    encode_points_from_indices,
    make_random_indices,
    sample_indexed_points_batched,
)
from refiner.refinement_ops import (
    scatter_points_to_volume,
    sample_volume_at_points,
    nebla_refinement_loss,
)

def sample_gt_volume(volume: torch.Tensor, points_xyz: torch.Tensor) -> torch.Tensor:
    """
    Trilinear sample GT CBCT at continuous point coordinates.

    Args:
        volume: [B, D, H, W]
        points_xyz: [B, N, 3], voxel-index coordinates [x, y, z]

    Returns:
        target: [B, N, 1]
    """
    B, D, H, W = volume.shape
    x = points_xyz[..., 0]
    y = points_xyz[..., 1]
    z = points_xyz[..., 2]

    gx = 2.0 * x / max(W - 1, 1) - 1.0
    gy = 2.0 * y / max(H - 1, 1) - 1.0
    gz = 2.0 * z / max(D - 1, 1) - 1.0

    grid = torch.stack([gx, gy, gz], dim=-1).view(B, -1, 1, 1, 3)

    sampled = F.grid_sample(
        volume[:, None, ...],
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )

    return sampled[:, 0, :, 0, 0].unsqueeze(-1)


def psnr_from_mse(mse: torch.Tensor) -> torch.Tensor:
    return -10.0 * torch.log10(torch.clamp(mse, min=1e-10))


def make_dataset(
    ids: Sequence[str],
    args: argparse.Namespace,
) -> SimPXCBCTDataset:
    return SimPXCBCTDataset(
        ids=ids,
        simpx_root=args.simpx_root,
        cbct_root=args.cbct_root,
        geom_root=args.geom_root,
        n_samples=args.n_samples,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
        already_normalized=args.already_normalized,
        geom_xy_transform=args.geom_xy_transform,
    )


def make_loader(
    dataset: SimPXCBCTDataset,
    args: argparse.Namespace,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_same_shape,
        drop_last=False,
    )


@torch.no_grad()
def evaluate(
    image_encoder: nn.Module,
    mlp: nn.Module,
    refiner: Union[nn.Module, None],
    loader: Union[DataLoader, None],
    args: argparse.Namespace,
    device: torch.device,
    split_name: str,
    max_batches: int = 0,
    perceptual_loss_fn=None,
) -> Dict[str, float]:
    """
    Evaluate on a split using random sampled points.

    max_batches <= 0 means evaluate the full split.
    """
    if loader is None:
        return {}

    image_encoder.eval()
    mlp.eval()
    if refiner is not None:
        refiner.eval()

    total_loss = 0.0
    total_items = 0
    num_batches = 0

    for batch in loader:
        if max_batches > 0 and num_batches >= max_batches:
            break

        simpx = batch["simpx"].to(device, non_blocking=True).float()
        volume = batch["volume"].to(device, non_blocking=True).float()
        start_xyz = batch["start_xyz"].to(device, non_blocking=True).float()
        end_xyz = batch["end_xyz"].to(device, non_blocking=True).float()

        B, _, Z, R = simpx.shape
        vol_shapes = batch["volume_shape"]
        if not torch.all(vol_shapes == vol_shapes[0]):
            raise ValueError("All items in a batch must have the same volume shape. Use --batch_size 1.")
        volume_shape = tuple(map(int, vol_shapes[0].tolist()))

        z_idx, r_idx, k_idx = make_random_indices(
            batch_size=B,
            z_count=Z,
            ray_count=R,
            n_samples=args.n_samples,
            n_points=args.n_points,
            device=device,
        )

        with autocast(enabled=args.amp and device.type == "cuda"):
            feature_map = image_encoder(simpx)

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
                gamma_x.reshape(B * args.n_points, 42),
                image_feat.reshape(B * args.n_points, 128),
            ).reshape(B, args.n_points, 1)

            points_xyz = sample_indexed_points_batched(
                start_xyz=start_xyz,
                end_xyz=end_xyz,
                z_idx=z_idx,
                r_idx=r_idx,
                k_idx=k_idx,
                n_samples=args.n_samples,
                volume_shape_zyx=volume_shape,
                normalize=False,
            )

            target = sample_gt_volume(volume, points_xyz)

            if refiner is None:
                loss = F.mse_loss(pred, target)
            else:
                rho, _ = scatter_points_to_volume(
                    values=pred,
                    points_xyz=points_xyz,
                    volume_shape_zyx=volume_shape,
                )
                refined_volume = refiner(rho)
                refined_points = sample_volume_at_points(refined_volume, points_xyz)
                loss, _ = nebla_refinement_loss(
                    refined_volume=refined_volume,
                    target_volume=volume,
                    refined_points=refined_points,
                    target_points=target,
                    point_weight=args.refiner_point_weight,
                    volume_weight=args.refiner_volume_weight,
                    proj_weight=args.refiner_proj_weight,
                    perc_weight=args.refiner_perc_weight,
                    perceptual_loss_fn=perceptual_loss_fn,
                )

        total_loss += float(loss.detach().item()) * B
        total_items += int(B)
        num_batches += 1

    if total_items == 0:
        return {}

    mean_loss = total_loss / max(total_items, 1)
    psnr = float(-10.0 * np.log10(max(mean_loss, 1e-10)))

    print(f"[{split_name}] loss={mean_loss:.6f} psnr={psnr:.2f} batches={num_batches}")

    return {
        "loss": mean_loss,
        "psnr": psnr,
        "batches": float(num_batches),
        "items": float(total_items),
    }


def build_lr_scheduler(args: argparse.Namespace, optimizer: torch.optim.Optimizer):
    """
    Build an iteration-based LR scheduler.

    Supported schedules:
      - none
      - cosine: optional warmup, then cosine decay to --lr_min
      - linear: optional warmup, then linear decay to --lr_min
      - step: StepLR with --lr_step_size and --lr_gamma
    """
    if args.lr_scheduler == "none":
        return None

    if args.lr_scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(int(args.lr_step_size), 1),
            gamma=float(args.lr_gamma),
        )

    base_lr = float(args.lr)
    min_lr = float(args.lr_min)
    min_ratio = min_lr / max(base_lr, 1e-12)
    warmup_iters = max(int(args.warmup_iters), 0)
    decay_iters = int(args.lr_decay_iters) if int(args.lr_decay_iters) > 0 else int(args.iters)
    decay_iters = max(decay_iters, warmup_iters + 1)

    def lr_lambda(step: int) -> float:
        step = int(step)

        if warmup_iters > 0 and step < warmup_iters:
            return max(float(step + 1) / float(warmup_iters), 1e-8)

        progress = (step - warmup_iters) / max(decay_iters - warmup_iters, 1)
        progress = min(max(progress, 0.0), 1.0)

        if args.lr_scheduler == "cosine":
            return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))

        if args.lr_scheduler == "linear":
            return min_ratio + (1.0 - min_ratio) * (1.0 - progress)

        raise ValueError(f"Unsupported lr_scheduler: {args.lr_scheduler}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def save_checkpoint(
    path: Path,
    global_step: int,
    image_encoder: nn.Module,
    mlp: nn.Module,
    refiner: Optional[nn.Module],
    optimizer: torch.optim.Optimizer,
    scheduler,
    args: argparse.Namespace,
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    test_ids: Sequence[str],
    metrics: Optional[Dict[str, float]] = None,
) -> None:
    torch.save(
        {
            "global_step": global_step,
            "image_encoder": image_encoder.state_dict(),
            "mlp": mlp.state_dict(),
            "refiner": refiner.state_dict() if refiner is not None else None,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "args": vars(args),
            "train_ids": list(train_ids),
            "val_ids": list(val_ids),
            "test_ids": list(test_ids),
            "metrics": metrics or {},
        },
        path,
    )
