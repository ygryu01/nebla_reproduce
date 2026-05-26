#!/usr/bin/env python3
"""
Stage 1 full-volume inference script.

Given a trained Stage 1 checkpoint, this script renders an intermediate 3D volume:
    SimPX -> image_encoder + MLP over all ray samples -> scatter/average -> rho volume

Saved file per subject:
    {out_root}/{split}/{sid}.pt

Each .pt contains:
    sid: str
    rho:    [1,D,H,W] tensor, usually float16 or float32
    mask:   [1,D,H,W] tensor, uint8 or float32-like; 1 means observed by at least one ray sample
    target: [D,H,W] tensor, normalized GT CBCT volume from the dataset
    count:  optional [1,D,H,W] tensor if --save_count is set
    meta:   dict
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import torch
from torch.utils.data import DataLoader

from models.image_encoder import UNET
from models.mlp import NeRF
from utils.utils import parse_int_list, resolve_id_splits
from helpers.helpers import make_dataset
from utils.full_volume_renderer import render_full_rho_no_grad


def _get_ckpt_arg(ckpt_args: dict, key: str, fallback):
    if isinstance(ckpt_args, dict) and key in ckpt_args:
        return ckpt_args[key]
    return fallback


def build_stage1_models(args: argparse.Namespace, ckpt: dict, device: torch.device):
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}

    nerf_depth = int(_get_ckpt_arg(ckpt_args, "nerf_depth", args.nerf_depth))
    nerf_width = int(_get_ckpt_arg(ckpt_args, "nerf_width", args.nerf_width))
    nerf_skips = parse_int_list(_get_ckpt_arg(ckpt_args, "nerf_skips", args.nerf_skips))

    image_encoder = UNET(in_channels=1, out_channels=128, features=[64, 128, 256, 512]).to(device)
    mlp = NeRF(
        D=nerf_depth,
        W=nerf_width,
        input_ch=42,
        output_ch=1,
        skips=nerf_skips,
    ).to(device)

    image_encoder.load_state_dict(ckpt["image_encoder"], strict=True)
    mlp.load_state_dict(ckpt["mlp"], strict=True)

    image_encoder.eval()
    mlp.eval()
    return image_encoder, mlp, {"nerf_depth": nerf_depth, "nerf_width": nerf_width, "nerf_skips": nerf_skips}


def _save_tensor_dtype(x: torch.Tensor, dtype_name: str) -> torch.Tensor:
    x = x.detach().cpu()
    if dtype_name == "float16":
        return x.to(torch.float16)
    if dtype_name == "float32":
        return x.to(torch.float32)
    raise ValueError(f"Unsupported save dtype: {dtype_name}")


def make_single_item_loader(ids: Sequence[str], args: argparse.Namespace) -> DataLoader:
    dataset = make_dataset(ids, args)
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=None if False else None,
    )


def _default_collate_same_shape(batch):
    # Local copy to avoid depending on helper internals. Dataset returns tensors and sid strings.
    return {
        "sid": [b["sid"] for b in batch],
        "simpx": torch.stack([b["simpx"] for b in batch], dim=0),
        "volume": torch.stack([b["volume"] for b in batch], dim=0),
        "start_xyz": torch.stack([b["start_xyz"] for b in batch], dim=0),
        "end_xyz": torch.stack([b["end_xyz"] for b in batch], dim=0),
        "volume_shape": torch.stack([b["volume_shape"] for b in batch], dim=0),
    }


def make_loader_for_ids(ids: Sequence[str], args: argparse.Namespace) -> DataLoader:
    dataset = make_dataset(ids, args)
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=_default_collate_same_shape,
        drop_last=False,
    )


def infer_split(split_name: str, ids: Sequence[str], image_encoder, mlp, args, device, model_meta: dict) -> None:
    if len(ids) == 0:
        print(f"[infer:{split_name}] skipped empty split")
        return

    out_dir = Path(args.rho_out_root) / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = make_loader_for_ids(ids, args)

    for batch_idx, batch in enumerate(loader):
        sid = str(batch["sid"][0])
        out_path = out_dir / f"{sid}.pt"
        if out_path.exists() and not args.overwrite:
            print(f"[infer:{split_name}] skip existing {out_path}")
            continue

        simpx = batch["simpx"].to(device, non_blocking=True).float()
        start_xyz = batch["start_xyz"].to(device, non_blocking=True).float()
        end_xyz = batch["end_xyz"].to(device, non_blocking=True).float()
        target = batch["volume"][0].detach().cpu()

        volume_shape = tuple(map(int, batch["volume_shape"][0].tolist()))

        rho, mask, count = render_full_rho_no_grad(
            image_encoder=image_encoder,
            mlp=mlp,
            simpx=simpx,
            start_xyz=start_xyz,
            end_xyz=end_xyz,
            volume_shape_zyx=volume_shape,
            n_samples=args.n_samples,
            chunk_points=args.dense_chunk_points,
            k_stride=args.dense_k_stride,
            amp=args.amp,
            device=device,
        )

        mask_to_save = mask.detach().cpu().to(torch.uint8)
        obj = {
            "sid": sid,
            "rho": _save_tensor_dtype(rho[0], args.save_dtype),
            "mask": mask_to_save[0],
            "target": _save_tensor_dtype(target, args.save_dtype),
            "meta": {
                "split": split_name,
                "volume_shape_zyx": volume_shape,
                "n_samples": int(args.n_samples),
                "dense_k_stride": int(args.dense_k_stride),
                "dense_chunk_points": int(args.dense_chunk_points),
                "geom_xy_transform": args.geom_xy_transform,
                "stage1_ckpt": str(args.stage1_ckpt),
                **model_meta,
            },
        }
        if args.save_count:
            # count can be large; use float16/float32 according to save dtype.
            obj["count"] = _save_tensor_dtype(count[0], args.save_dtype)

        torch.save(obj, out_path)
        observed = float(mask.float().mean().item())
        print(
            f"[infer:{split_name}] {batch_idx + 1}/{len(loader)} sid={sid} "
            f"rho={tuple(obj['rho'].shape)} observed_ratio={observed:.6f} saved={out_path}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--stage1_ckpt", type=str, default="/home/alphayoung8/log1/nebla/logs/nebla_stage1_mlp/ckpt_final.pt", help="Path to Stage 1 checkpoint, e.g. ckpt_final.pt.")
    parser.add_argument("--rho_out_root", type=str, default="stage1_infer", help="Output root for inferred rho volumes.")
    parser.add_argument("--split", type=str, default="all", choices=["train", "val", "test", "all"], help="Which resolved split to infer.")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--ids", type=str, default="327")
    parser.add_argument("--ids_file", type=str, default="")
    parser.add_argument("--train_ids_file", type=str, default="/home/alphayoung8/log1/nebla/ids_file/train_ids.txt")
    parser.add_argument("--val_ids_file", type=str, default="/home/alphayoung8/log1/nebla/ids_file/val_ids.txt")
    parser.add_argument("--test_ids_file", type=str, default="/home/alphayoung8/log1/nebla/ids_file/test_ids.txt")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--no_split_shuffle", action="store_true")

    parser.add_argument("--simpx_root", type=str, default="/home/alphayoung8/log1/nebla/simpx_renderer/simpx_result_noresize")
    parser.add_argument("--cbct_root", type=str, default="/home/alphayoung8/log1/oral3d_reproduce/data/cbct_npy")
    parser.add_argument("--geom_root", type=str, default="/home/alphayoung8/log1/nebla/simpx_renderer/center_geometries")
    parser.add_argument("--geom_xy_transform", type=str, default="rot90_cw", choices=["none", "rot90_cw", "rot90_ccw", "swap_xy", "flip_x", "flip_y"])

    parser.add_argument("--nerf_depth", type=int, default=8)
    parser.add_argument("--nerf_width", type=int, default=256)
    parser.add_argument("--nerf_skips", type=str, default="4")

    parser.add_argument("--n_samples", type=int, default=400)
    parser.add_argument("--dense_chunk_points", type=int, default=131072)
    parser.add_argument("--dense_k_stride", type=int, default=1, help="Use 1 for every ray sample. Larger values are faster but less dense.")

    parser.add_argument("--clip_min", type=float, default=-1000.0)
    parser.add_argument("--clip_max", type=float, default=4000.0)
    parser.add_argument("--already_normalized", default=True, action="store_true")

    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--save_dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--save_count", action="store_true")

    # Attributes expected by make_dataset but not functionally used here.
    parser.add_argument("--batch_size", type=int, default=1)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.batch_size = 1

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ckpt = torch.load(args.stage1_ckpt, map_location=device)
    image_encoder, mlp, model_meta = build_stage1_models(args, ckpt, device)

    train_ids, val_ids, test_ids = resolve_id_splits(args)
    split_map = {"train": train_ids, "val": val_ids, "test": test_ids}

    if args.split == "all":
        selected = ["train", "val", "test"]
    else:
        selected = [args.split]

    print(
        f"[infer] device={device} split={args.split} "
        f"dense_k_stride={args.dense_k_stride} dense_chunk_points={args.dense_chunk_points} "
        f"save_dtype={args.save_dtype}"
    )
    for split_name in selected:
        infer_split(split_name, split_map[split_name], image_encoder, mlp, args, device, model_meta)


if __name__ == "__main__":
    main()
