#!/usr/bin/env python3
"""
Stage 2 training script.

Train only the 3D U-Net refiner from precomputed Stage 1 full-volume inference outputs:
    rho + mask -> 3D U-Net -> refined CBCT volume

This script does not call the Stage 1 image_encoder/MLP during training.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from utils.utils import parse_int_list
from refiner.unet3d_refiner import build_nebla_3d_unet_refiner
from dataset.rho_dataset import PrecomputedRhoDataset, collate_rho_batch
from vis.stage2_visualization import save_stage2_debug_images

try:
    from refiner.perceptual_loss import VGGPerceptualLoss3DMIP
except Exception:
    VGGPerceptualLoss3DMIP = None


def psnr_from_mse_value(mse: float) -> float:
    return -10.0 * math.log10(max(float(mse), 1e-10))


def mip_projection_loss(pred_5d: torch.Tensor, target_5d: torch.Tensor) -> torch.Tensor:
    """pred_5d/target_5d: [B,1,D,H,W]."""
    pred_axial = pred_5d.amax(dim=2)     # [B,1,H,W]
    gt_axial = target_5d.amax(dim=2)
    pred_coronal = pred_5d.amax(dim=3)   # [B,1,D,W]
    gt_coronal = target_5d.amax(dim=3)
    pred_sagittal = pred_5d.amax(dim=4)  # [B,1,D,H]
    gt_sagittal = target_5d.amax(dim=4)
    return (
        F.mse_loss(pred_axial, gt_axial)
        + F.mse_loss(pred_coronal, gt_coronal)
        + F.mse_loss(pred_sagittal, gt_sagittal)
    ) / 3.0


def stage2_loss(
    pred_5d: torch.Tensor,
    target_4d: torch.Tensor,
    volume_weight: float,
    proj_weight: float,
    perc_weight: float,
    perceptual_loss_fn: nn.Module | None,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    target_5d = target_4d[:, None, ...].float()
    pred_5d = pred_5d.float()

    volume_mse = F.mse_loss(pred_5d, target_5d)
    mip_proj = mip_projection_loss(pred_5d, target_5d)

    if perceptual_loss_fn is not None and perc_weight > 0.0:
        perc = perceptual_loss_fn(pred_5d.clamp(0.0, 1.0), target_5d.clamp(0.0, 1.0))
    else:
        perc = pred_5d.new_tensor(0.0)

    loss = volume_weight * volume_mse + proj_weight * mip_proj + perc_weight * perc
    return loss, {
        "volume_mse": volume_mse.detach(),
        "mip_proj": mip_proj.detach(),
        "perc": perc.detach(),
    }


def make_loader(root: str, split: str, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader | None:
    split_dir = Path(root) / split
    if not split_dir.is_dir() or len(list(split_dir.glob("*.pt"))) == 0:
        return None
    dataset = PrecomputedRhoDataset(root=root, split=split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_rho_batch,
        drop_last=False,
    )


@torch.no_grad()
def evaluate(refiner: nn.Module, loader: DataLoader | None, args: argparse.Namespace, device: torch.device, perceptual_loss_fn=None):
    if loader is None:
        return {}

    refiner.eval()
    total_loss = 0.0
    total_vol_mse = 0.0
    total_items = 0

    for batch in loader:
        rho = batch["rho"].to(device, non_blocking=True).float()
        mask = batch["mask"].to(device, non_blocking=True).float()
        target = batch["target"].to(device, non_blocking=True).float()

        if args.use_mask_channel:
            x = torch.cat([rho, mask], dim=1)
        else:
            x = rho

        with autocast(enabled=args.amp and device.type == "cuda"):
            pred = refiner(x)
            loss, terms = stage2_loss(
                pred_5d=pred,
                target_4d=target,
                volume_weight=args.volume_weight,
                proj_weight=args.proj_weight,
                perc_weight=args.perc_weight,
                perceptual_loss_fn=perceptual_loss_fn,
            )

        B = rho.shape[0]
        total_loss += float(loss.detach().item()) * B
        total_vol_mse += float(terms["volume_mse"].item()) * B
        total_items += B

    if total_items == 0:
        return {}
    mean_loss = total_loss / total_items
    mean_vol_mse = total_vol_mse / total_items
    return {"loss": mean_loss, "volume_mse": mean_vol_mse, "psnr": psnr_from_mse_value(mean_vol_mse)}


def save_stage2_checkpoint(path: Path, refiner: nn.Module, optimizer, scheduler, scaler, args, step: int, metrics=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "global_step": int(step),
            "refiner": refiner.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "args": vars(args),
            "metrics": metrics or {},
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--rho_root", type=str, required=True, help="Root produced by infer_stage1_full_volume.py.")
    parser.add_argument("--out_dir", type=str, default="./logs/nebla_stage2_refiner")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")

    parser.add_argument("--use_mask_channel", action="store_true", default=True, help="Use concat(rho, mask) as 3D U-Net input.")
    parser.add_argument("--refiner_f_maps", type=str, default="32,64,128,256")
    parser.add_argument("--refiner_final_activation", type=str, default="sigmoid", choices=["sigmoid", "none", "identity", ""])

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--iters", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--volume_weight", type=float, default=1.0)
    parser.add_argument("--proj_weight", type=float, default=10.0)
    parser.add_argument("--perc_weight", type=float, default=1.0)
    parser.add_argument("--perc_resize", type=int, default=0, help="If >0, resize MIPs to perc_resize x perc_resize before VGG perceptual loss.")

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--debug_image_every", type=int, default=1000, help="Save Stage 2 MIP debug images every N steps. Set <=0 to disable.")
    parser.add_argument("--debug_image_max_items", type=int, default=2, help="Maximum subjects to visualize per debug save.")
    parser.add_argument("--debug_image_split", type=str, default="val", choices=["train", "val"], help="Split used for debug image saving.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader = make_loader(args.rho_root, args.train_split, args.batch_size, args.num_workers, shuffle=True)
    val_loader = make_loader(args.rho_root, args.val_split, args.batch_size, args.num_workers, shuffle=False)
    if train_loader is None:
        raise FileNotFoundError(f"No training rho files found in {Path(args.rho_root) / args.train_split}")

    f_maps = parse_int_list(args.refiner_f_maps)
    in_channels = 2 if args.use_mask_channel else 1
    refiner = build_nebla_3d_unet_refiner(
        in_channels=in_channels,
        out_channels=1,
        f_maps=f_maps,
        final_activation=args.refiner_final_activation,
    ).to(device)

    perceptual_loss_fn = None
    if args.perc_weight > 0.0:
        if VGGPerceptualLoss3DMIP is None:
            raise ImportError("refiner.perceptual_loss.VGGPerceptualLoss3DMIP is unavailable, but --perc_weight > 0.")
        resize_to = None if args.perc_resize <= 0 else (args.perc_resize, args.perc_resize)
        perceptual_loss_fn = VGGPerceptualLoss3DMIP(resize_to=resize_to).to(device)
        perceptual_loss_fn.eval()

    optimizer = torch.optim.Adam(refiner.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")

    print(
        f"[stage2] device={device} rho_root={args.rho_root} in_channels={in_channels} "
        f"f_maps={f_maps} batch_size={args.batch_size} amp={args.amp} "
        f"loss=(volume={args.volume_weight}, proj={args.proj_weight}, perc={args.perc_weight})"
    )

    global_step = 0
    best_val_loss = float("inf")
    refiner.train()

    while global_step < args.iters:
        for batch in train_loader:
            if global_step >= args.iters:
                break

            rho = batch["rho"].to(device, non_blocking=True).float()
            mask = batch["mask"].to(device, non_blocking=True).float()
            target = batch["target"].to(device, non_blocking=True).float()

            if args.use_mask_channel:
                x = torch.cat([rho, mask], dim=1)
            else:
                x = rho

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=args.amp and device.type == "cuda"):
                pred = refiner(x)
                loss, terms = stage2_loss(
                    pred_5d=pred,
                    target_4d=target,
                    volume_weight=args.volume_weight,
                    proj_weight=args.proj_weight,
                    perc_weight=args.perc_weight,
                    perceptual_loss_fn=perceptual_loss_fn,
                )

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(refiner.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            if global_step % args.log_every == 0:
                vol_mse = float(terms["volume_mse"].item())
                print(
                    f"[step {global_step:06d}] loss={loss.item():.6f} "
                    f"volume_mse={vol_mse:.6f} psnr={psnr_from_mse_value(vol_mse):.2f} "
                    f"mip_proj={terms['mip_proj'].item():.6f} perc={terms['perc'].item():.6f} "
                    f"x={tuple(x.shape)} pred={tuple(pred.shape)} target={tuple(target.shape)}"
                )

            if val_loader is not None and args.eval_every > 0 and global_step > 0 and global_step % args.eval_every == 0:
                metrics = evaluate(refiner, val_loader, args, device, perceptual_loss_fn)
                print(
                    f"[val@{global_step:06d}] loss={metrics.get('loss', float('nan')):.6f} "
                    f"volume_mse={metrics.get('volume_mse', float('nan')):.6f} "
                    f"psnr={metrics.get('psnr', float('nan')):.2f}"
                )
                if metrics and metrics["loss"] < best_val_loss:
                    best_val_loss = metrics["loss"]
                    save_stage2_checkpoint(out_dir / "ckpt_best_val.pt", refiner, optimizer, scheduler, scaler, args, global_step, metrics)
                    print(f"[save best val] {out_dir / 'ckpt_best_val.pt'}")
                refiner.train()

            if args.debug_image_every > 0 and global_step > 0 and global_step % args.debug_image_every == 0:
                debug_loader = val_loader if args.debug_image_split == "val" and val_loader is not None else train_loader
                save_stage2_debug_images(
                    refiner=refiner,
                    loader=debug_loader,
                    args=args,
                    device=device,
                    global_step=global_step,
                    split_name=args.debug_image_split if debug_loader is val_loader else "train",
                )
                refiner.train()

            if global_step > 0 and global_step % args.save_every == 0:
                save_stage2_checkpoint(out_dir / f"ckpt_{global_step:06d}.pt", refiner, optimizer, scheduler, scaler, args, global_step)
                print(f"[save] {out_dir / f'ckpt_{global_step:06d}.pt'}")

            global_step += 1

    final_metrics = evaluate(refiner, val_loader, args, device, perceptual_loss_fn) if val_loader is not None else {}
    if args.debug_image_every > 0:
        debug_loader = val_loader if args.debug_image_split == "val" and val_loader is not None else train_loader
        save_stage2_debug_images(
            refiner=refiner,
            loader=debug_loader,
            args=args,
            device=device,
            global_step=global_step,
            split_name=args.debug_image_split if debug_loader is val_loader else "train",
        )
        refiner.train()
    save_stage2_checkpoint(out_dir / "ckpt_final.pt", refiner, optimizer, scheduler, scaler, args, global_step, final_metrics)
    print(f"[save] {out_dir / 'ckpt_final.pt'}")


if __name__ == "__main__":
    main()
