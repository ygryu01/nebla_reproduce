#!/usr/bin/env python3
"""
Stage 2 inference script.

Input:
    Precomputed Stage 1 rho files from infer_stage1_full_volume.py:
        {rho_root}/{split}/{sid}.pt

Primary output:
    Refined 3D volumes from the trained Stage 2 3D U-Net refiner:
        {out_root}/{split}/{sid}_pred.nii.gz

Optional outputs:
    {out_root}/{split}/{sid}_target.nii.gz   if --save_target is enabled
    {out_root}/{split}/{sid}_rho.nii.gz      if --save_input is enabled
    {out_root}/{split}/{sid}_mask.nii.gz     if --save_input is enabled

Optional MIP debug PNGs are saved under:
    {out_root}/{split}/mips/

Notes:
    - Volumes are saved in array order [D, H, W] as stored in the dataset.
    - Since the precomputed rho dataset does not provide a physical affine,
      an identity affine is used for NIfTI export.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from utils.utils import parse_int_list
from refiner.unet3d_refiner import build_nebla_3d_unet_refiner
from dataset.rho_dataset import PrecomputedRhoDataset, collate_rho_batch
from vis.stage2_visualization import save_stage2_mip_debug_for_item


IDENTITY_AFFINE = np.eye(4, dtype=np.float32)


def _ckpt_args(ckpt: dict) -> dict:
    args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    return args if isinstance(args, dict) else {}


def _get_arg(ckpt_args: dict, runtime_args: argparse.Namespace, name: str):
    if name in ckpt_args:
        return ckpt_args[name]
    return getattr(runtime_args, name)


def make_loader(root: str, split: str, batch_size: int, num_workers: int) -> DataLoader | None:
    split_dir = Path(root) / split
    if not split_dir.is_dir() or len(list(split_dir.glob("*.pt"))) == 0:
        return None
    dataset = PrecomputedRhoDataset(root=root, split=split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_rho_batch,
        drop_last=False,
    )


def build_refiner_from_ckpt(args: argparse.Namespace, ckpt: dict, device: torch.device):
    ckpt_args = _ckpt_args(ckpt)

    refiner_f_maps = parse_int_list(_get_arg(ckpt_args, args, "refiner_f_maps"))
    final_activation = _get_arg(ckpt_args, args, "refiner_final_activation")
    use_mask_channel = bool(_get_arg(ckpt_args, args, "use_mask_channel"))
    in_channels = 2 if use_mask_channel else 1

    refiner = build_nebla_3d_unet_refiner(
        in_channels=in_channels,
        out_channels=1,
        f_maps=refiner_f_maps,
        final_activation=final_activation,
    ).to(device)

    state = ckpt.get("refiner", ckpt)
    refiner.load_state_dict(state, strict=True)
    refiner.eval()

    meta = {
        "refiner_f_maps": refiner_f_maps,
        "refiner_final_activation": final_activation,
        "use_mask_channel": use_mask_channel,
        "stage2_ckpt": str(args.stage2_ckpt),
    }
    return refiner, meta


def _to_numpy_volume(x: torch.Tensor, dtype_name: str) -> np.ndarray:
    """
    Convert a torch tensor volume to a NumPy array suitable for NIfTI saving.

    Accepts shapes [1,D,H,W] or [D,H,W]. Returns [D,H,W].
    """
    x = x.detach().cpu()
    if x.ndim == 4 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 3:
        raise ValueError(f"Expected [1,D,H,W] or [D,H,W], got {tuple(x.shape)}")

    if dtype_name == "float16":
        return x.to(torch.float16).numpy().astype(np.float16, copy=False)
    if dtype_name == "float32":
        return x.to(torch.float32).numpy().astype(np.float32, copy=False)
    if dtype_name == "uint8":
        return x.to(torch.uint8).numpy().astype(np.uint8, copy=False)
    raise ValueError(f"Unsupported save dtype: {dtype_name}")


def _save_nifti(volume: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nii = nib.Nifti1Image(volume, IDENTITY_AFFINE)
    nib.save(nii, str(out_path))


def _save_sidecar_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


@torch.no_grad()
def infer_split(split_name: str, loader: DataLoader, refiner, args, device, model_meta: dict) -> None:
    out_dir = Path(args.out_root) / split_name
    out_dir.mkdir(parents=True, exist_ok=True)
    mip_dir = out_dir / "mips"
    meta_dir = out_dir / "meta"

    for batch_idx, batch in enumerate(loader):
        rho = batch["rho"].to(device, non_blocking=True).float()
        mask = batch["mask"].to(device, non_blocking=True).float()
        target = batch["target"].to(device, non_blocking=True).float()

        use_mask_channel = bool(model_meta["use_mask_channel"])
        x = torch.cat([rho, mask], dim=1) if use_mask_channel else rho

        with autocast(enabled=args.amp and device.type == "cuda"):
            pred = refiner(x).float().clamp(0.0, 1.0)

        B = pred.shape[0]
        for bi in range(B):
            sid = str(batch["sid"][bi])
            pred_path = out_dir / f"{sid}_pred.nii.gz"
            if pred_path.exists() and not args.overwrite:
                print(f"[stage2 infer:{split_name}] skip existing {pred_path}")
                continue

            pred_np = _to_numpy_volume(pred[bi], args.save_dtype)
            _save_nifti(pred_np, pred_path)

            if args.save_target:
                target_np = _to_numpy_volume(target[bi], args.save_dtype)
                _save_nifti(target_np, out_dir / f"{sid}_target.nii.gz")

            if args.save_input:
                rho_np = _to_numpy_volume(rho[bi], args.save_dtype)
                mask_np = _to_numpy_volume(mask[bi], "uint8")
                _save_nifti(rho_np, out_dir / f"{sid}_rho.nii.gz")
                _save_nifti(mask_np, out_dir / f"{sid}_mask.nii.gz")

            if args.save_meta:
                meta_obj = {
                    "sid": sid,
                    "split": split_name,
                    "source_rho_path": batch["path"][bi],
                    "pred_path": str(pred_path),
                    **model_meta,
                }
                _save_sidecar_json(meta_dir / f"{sid}.json", meta_obj)

            if args.save_mip_images:
                save_stage2_mip_debug_for_item(
                    sid=sid,
                    rho_3d=rho[bi, 0].detach().cpu().numpy(),
                    mask_3d=mask[bi, 0].detach().cpu().numpy(),
                    pred_3d=pred[bi, 0].detach().cpu().numpy(),
                    target_3d=target[bi].detach().cpu().numpy(),
                    out_dir=mip_dir,
                )

            print(
                f"[stage2 infer:{split_name}] {batch_idx + 1}/{len(loader)} sid={sid} "
                f"pred={tuple(pred[bi].shape)} saved={pred_path}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--stage2_ckpt", type=str, required=True, help="Path to Stage 2 checkpoint, e.g. ckpt_final.pt.")
    parser.add_argument("--rho_root", type=str, required=True, help="Root produced by infer_stage1_full_volume.py.")
    parser.add_argument("--out_root", type=str, required=True, help="Output root for refined volumes.")
    parser.add_argument("--split", type=str, default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--overwrite", action="store_true")

    # Fallbacks used only if the checkpoint does not contain args.
    parser.add_argument("--use_mask_channel", action="store_true", default=True)
    parser.add_argument("--refiner_f_maps", type=str, default="32,64,128,256")
    parser.add_argument("--refiner_final_activation", type=str, default="sigmoid", choices=["sigmoid", "none", "identity", ""])

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--save_dtype", type=str, default="float32", choices=["float16", "float32"])
    parser.add_argument("--save_target", action="store_true", help="Also save target volume as NIfTI.")
    parser.add_argument("--save_input", action="store_true", help="Also save rho/mask as NIfTI.")
    parser.add_argument("--save_meta", action="store_true", help="Save a JSON sidecar per subject.")
    parser.add_argument("--save_mip_images", action="store_true", help="Save rho/mask/pred/gt/error MIP panels.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    ckpt = torch.load(args.stage2_ckpt, map_location=device)
    refiner, model_meta = build_refiner_from_ckpt(args, ckpt, device)

    selected = ["train", "val", "test"] if args.split == "all" else [args.split]
    print(
        f"[stage2 infer] device={device} split={args.split} rho_root={args.rho_root} "
        f"out_root={args.out_root} batch_size={args.batch_size} amp={args.amp} "
        f"model={model_meta}"
    )

    for split_name in selected:
        loader = make_loader(args.rho_root, split_name, args.batch_size, args.num_workers)
        if loader is None:
            print(f"[stage2 infer:{split_name}] skipped missing/empty split")
            continue
        infer_split(split_name, loader, refiner, args, device, model_meta)


if __name__ == "__main__":
    main()
