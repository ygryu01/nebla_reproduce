#!/usr/bin/env python3
"""
Visualization utilities for Stage 2 refiner training and inference.

Saves MIP panels for:
    - refined prediction vs target vs absolute error
    - Stage 1 rho input
    - observation mask

Tensor conventions:
    pred_5d:   [B,1,D,H,W]
    target_4d: [B,D,H,W]
    rho_5d:    [B,1,D,H,W]
    mask_5d:   [B,1,D,H,W]
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.cuda.amp import autocast


def _unit_to_uint8(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0 + 0.5).astype(np.uint8)


def _save_single_gray(img: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_unit_to_uint8(img)).save(path)


def _save_panel(images: Dict[str, np.ndarray], path: Path) -> None:
    """Save horizontal grayscale panel with labels."""
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(images.keys())
    arrays = [_unit_to_uint8(images[k]) for k in labels]

    max_h = max(a.shape[0] for a in arrays)
    max_w = max(a.shape[1] for a in arrays)
    label_h = 22
    gap = 6

    panel = Image.new("L", (len(arrays) * max_w + (len(arrays) - 1) * gap, max_h + label_h), color=0)
    draw = ImageDraw.Draw(panel)

    x0 = 0
    for label, arr in zip(labels, arrays):
        im = Image.fromarray(arr)
        panel.paste(im, (x0, label_h))
        draw.text((x0 + 4, 4), label, fill=255)
        x0 += max_w + gap

    panel.save(path)


def volume_mips(volume_3d: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Args:
        volume_3d: [D,H,W]
    Returns:
        axial:    [H,W], max over z/depth
        coronal:  [D,W], max over y
        sagittal: [D,H], max over x
    """
    if volume_3d.ndim != 3:
        raise ValueError(f"Expected [D,H,W], got {volume_3d.shape}")
    return {
        "axial": volume_3d.max(axis=0),
        "coronal": volume_3d.max(axis=1),
        "sagittal": volume_3d.max(axis=2),
    }


def save_stage2_mip_debug_for_item(
    sid: str,
    rho_3d: np.ndarray,
    mask_3d: np.ndarray,
    pred_3d: np.ndarray,
    target_3d: np.ndarray,
    out_dir: str | Path,
) -> None:
    """
    Save Stage 2 debug MIPs for one subject.

    Output examples:
        {out_dir}/{sid}_axial_panel.png
        {out_dir}/{sid}_rho_axial.png
        {out_dir}/{sid}_mask_axial.png
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rho_mips = volume_mips(rho_3d)
    mask_mips = volume_mips(mask_3d)
    pred_mips = volume_mips(pred_3d)
    target_mips = volume_mips(target_3d)

    for view in ("axial", "coronal", "sagittal"):
        pred = pred_mips[view]
        target = target_mips[view]
        err = np.abs(pred - target)

        _save_panel(
            {
                "rho": rho_mips[view],
                "mask": mask_mips[view],
                "pred": pred,
                "gt": target,
                "abs_err": err,
            },
            out_dir / f"{sid}_{view}_panel.png",
        )

        _save_single_gray(rho_mips[view], out_dir / f"{sid}_rho_{view}.png")
        _save_single_gray(mask_mips[view], out_dir / f"{sid}_mask_{view}.png")
        _save_single_gray(pred, out_dir / f"{sid}_pred_{view}.png")
        _save_single_gray(target, out_dir / f"{sid}_gt_{view}.png")
        _save_single_gray(err, out_dir / f"{sid}_abs_err_{view}.png")


@torch.no_grad()
def save_stage2_debug_images(
    refiner,
    loader,
    args,
    device: torch.device,
    global_step: int,
    split_name: str = "val",
) -> int:
    """
    Run a few samples through the Stage 2 refiner and save MIP debug images.

    Returns:
        number of saved items
    """
    if loader is None or int(args.debug_image_max_items) <= 0:
        return 0

    out_dir = Path(args.out_dir) / "debug_images" / f"step_{int(global_step):06d}" / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    refiner.eval()
    saved = 0

    for batch in loader:
        rho = batch["rho"].to(device, non_blocking=True).float()
        mask = batch["mask"].to(device, non_blocking=True).float()
        target = batch["target"].to(device, non_blocking=True).float()

        if args.use_mask_channel:
            x = torch.cat([rho, mask], dim=1)
        else:
            x = rho

        with autocast(enabled=args.amp and device.type == "cuda"):
            pred = refiner(x).float().clamp(0.0, 1.0)

        B = rho.shape[0]
        for bi in range(B):
            if saved >= int(args.debug_image_max_items):
                print(f"[stage2 debug image] saved {saved} item(s) to {out_dir}")
                return saved

            sid = str(batch["sid"][bi])
            save_stage2_mip_debug_for_item(
                sid=sid,
                rho_3d=rho[bi, 0].detach().cpu().numpy(),
                mask_3d=mask[bi, 0].detach().cpu().numpy(),
                pred_3d=pred[bi, 0].detach().cpu().numpy(),
                target_3d=target[bi].detach().cpu().numpy(),
                out_dir=out_dir,
            )
            saved += 1

    print(f"[stage2 debug image] saved {saved} item(s) to {out_dir}")
    return saved
