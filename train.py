#!/usr/bin/env python3
"""
train_nebla.py

Training loop for:
    SimPX image -> UNET feature map -> point-conditioned NeRF/MLP -> CBCT intensity

Utility functions/classes have been moved into:
    utils.py
    dataset.py
    helpers.py
    visualization.py

Optional NeBLa-style 3D U-Net refinement can be enabled with --use_3d_refiner.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast, GradScaler

from models.image_encoder import UNET
from models.mlp import NeRF, gather_image_features
from models.point_embbeder import (
    encode_points_from_indices,
    make_random_indices,
    sample_indexed_points_batched,
)
from helpers.helpers import (
    build_lr_scheduler,
    evaluate,
    get_current_lr,
    make_dataset,
    make_loader,
    psnr_from_mse,
    sample_gt_volume,
    save_checkpoint,
)
from utils.utils import (
    parse_int_list,
    resolve_id_splits,
    write_split_files,
)
from vis.stage1_visualization import save_validation_mip_images
from refiner.unet3d_refiner import build_nebla_3d_unet_refiner
from refiner.refinement_ops import (
    scatter_points_to_volume,
    sample_volume_at_points,
    nebla_refinement_loss,
)

from refiner.perceptual_loss import VGGPerceptualLoss3DMIP


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ids, val_ids, test_ids = resolve_id_splits(args)
    write_split_files(out_dir, train_ids, val_ids, test_ids)

    print(
        f"[split] train={len(train_ids)} val={len(val_ids)} test={len(test_ids)} "
        f"seed={args.split_seed} val_ratio={args.val_ratio} test_ratio={args.test_ratio}"
    )
    print(f"[split] saved to {out_dir / 'ids_train.txt'}, {out_dir / 'ids_val.txt'}, {out_dir / 'ids_test.txt'}")
    print(f"[geometry] xy_transform={args.geom_xy_transform}")

    train_dataset = make_dataset(train_ids, args)
    val_dataset = make_dataset(val_ids, args) if len(val_ids) > 0 else None
    test_dataset = make_dataset(test_ids, args) if len(test_ids) > 0 else None

    train_loader = make_loader(train_dataset, args, shuffle=True)
    val_loader = make_loader(val_dataset, args, shuffle=False) if val_dataset is not None else None
    test_loader = make_loader(test_dataset, args, shuffle=False) if test_dataset is not None else None

    nerf_skips = parse_int_list(args.nerf_skips)

    image_encoder = UNET(in_channels=1, out_channels=128, features=[64, 128, 256, 512]).to(device)
    mlp = NeRF(
        D=args.nerf_depth,
        W=args.nerf_width,
        input_ch=42,
        output_ch=1,
        skips=nerf_skips,
    ).to(device)

    refiner = None
    refiner_f_maps = parse_int_list(args.refiner_f_maps)
    if args.use_3d_refiner:
        refiner = build_nebla_3d_unet_refiner(
            in_channels=1,
            out_channels=1,
            f_maps=refiner_f_maps,
            final_activation=args.refiner_final_activation,
        ).to(device)

    perceptual_loss_fn = None
    if refiner is not None and args.refiner_perc_weight > 0.0:
        perceptual_loss_fn = VGGPerceptualLoss3DMIP(
            resize_to=None,
        ).to(device)
        perceptual_loss_fn.eval()

    print(
        f"[model] UNET out_channels=128 features=[64, 128, 256, 512] "
        f"NeRF depth={args.nerf_depth} width={args.nerf_width} skips={nerf_skips}"
    )
    if refiner is not None:
        print(
            f"loss_weights=(point={args.refiner_point_weight}, "
            f"volume={args.refiner_volume_weight}, "
            f"proj={args.refiner_proj_weight}, "
            f"perc={args.refiner_perc_weight})"
        )

    params = list(image_encoder.parameters()) + list(mlp.parameters())
    if refiner is not None:
        params += list(refiner.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_lr_scheduler(args, optimizer)
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")

    print(
        f"[lr] scheduler={args.lr_scheduler} initial_lr={get_current_lr(optimizer):.6e} "
        f"base_lr={args.lr:.6e} min_lr={args.lr_min:.6e} warmup_iters={args.warmup_iters}"
    )

    global_step = 0
    best_val_loss = float("inf")
    image_encoder.train()
    mlp.train()
    if refiner is not None:
        refiner.train()

    while global_step < args.iters:
        for batch in train_loader:
            if global_step >= args.iters:
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

            optimizer.zero_grad(set_to_none=True)

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
                    loss_terms = {"point_mse": loss.detach()}
                    refined_volume = None
                else:
                    rho, scatter_count = scatter_points_to_volume(
                        values=pred,
                        points_xyz=points_xyz,
                        volume_shape_zyx=volume_shape,
                    )
                    refined_volume = refiner(rho)
                    refined_points = sample_volume_at_points(refined_volume, points_xyz)
                    loss, loss_terms = nebla_refinement_loss(
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
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(params, args.grad_clip)

            scale_before = scaler.get_scale() if scaler.is_enabled() else 1.0
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale() if scaler.is_enabled() else 1.0
            if scheduler is not None and scale_after >= scale_before:
                scheduler.step()

            if global_step % args.log_every == 0:
                with torch.no_grad():
                    psnr = psnr_from_mse(loss.detach())
                extra = ""
                if refiner is not None:
                    extra = (
                        f" volume_mse={loss_terms.get('volume_mse', torch.tensor(float('nan'), device=device)).detach().item():.6f}"
                        f" mip_proj={loss_terms.get('mip_proj', torch.tensor(float('nan'), device=device)).detach().item():.6f}"
                        f" perc={loss_terms.get('perc', torch.tensor(float('nan'), device=device)).detach().item():.6f}"
                        f" point_mse={loss_terms.get('point_mse', torch.tensor(float('nan'), device=device)).detach().item():.6f}"
                    )
                print(
                    f"[step {global_step:06d}] "
                    f"loss={loss.item():.6f} psnr={psnr.item():.2f} "
                    f"lr={get_current_lr(optimizer):.6e} "
                    f"B={B} N={args.n_points} simpx={tuple(simpx.shape)} "
                    f"pred={tuple(pred.shape)} target={tuple(target.shape)}"
                    f"{extra}"
                )

            if val_loader is not None and args.eval_every > 0 and global_step > 0 and global_step % args.eval_every == 0:
                val_metrics = evaluate(
                    image_encoder=image_encoder,
                    mlp=mlp,
                    refiner=refiner,
                    perceptual_loss_fn=perceptual_loss_fn,
                    loader=val_loader,
                    args=args,
                    device=device,
                    split_name=f"val@{global_step:06d}",
                    max_batches=args.eval_batches,
                )
                if val_metrics and val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    ckpt_path = out_dir / "ckpt_best_val.pt"
                    save_checkpoint(
                        path=ckpt_path,
                        global_step=global_step,
                        image_encoder=image_encoder,
                        mlp=mlp,
                        refiner=refiner,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        args=args,
                        train_ids=train_ids,
                        val_ids=val_ids,
                        test_ids=test_ids,
                        metrics={"best_val_loss": best_val_loss, **val_metrics},
                    )
                    print(f"[save best val] {ckpt_path}")

                image_encoder.train()
                mlp.train()
                if refiner is not None:
                    refiner.train()

            if val_loader is not None and args.val_image_every > 0 and global_step > 0 and global_step % args.val_image_every == 0:
                save_validation_mip_images(
                    image_encoder=image_encoder,
                    mlp=mlp,
                    refiner=refiner,
                    loader=val_loader,
                    args=args,
                    device=device,
                    global_step=global_step,
                )
                image_encoder.train()
                mlp.train()
                if refiner is not None:
                    refiner.train()

            if global_step > 0 and global_step % args.save_every == 0:
                ckpt_path = out_dir / f"ckpt_{global_step:06d}.pt"
                save_checkpoint(
                    path=ckpt_path,
                    global_step=global_step,
                    image_encoder=image_encoder,
                    mlp=mlp,
                    refiner=refiner,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    args=args,
                    train_ids=train_ids,
                    val_ids=val_ids,
                    test_ids=test_ids,
                )
                print(f"[save] {ckpt_path}")

            global_step += 1

    final_metrics: Dict[str, float] = {}

    if val_loader is not None:
        val_metrics = evaluate(
            image_encoder=image_encoder,
            mlp=mlp,
            refiner=refiner,
            perceptual_loss_fn=perceptual_loss_fn,
            loader=val_loader,
            args=args,
            device=device,
            split_name="val_final",
            max_batches=0,
        )
        final_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
        if args.val_image_every > 0:
            save_validation_mip_images(
                image_encoder=image_encoder,
                mlp=mlp,
                refiner=refiner,
                loader=val_loader,
                args=args,
                device=device,
                global_step=global_step,
            )
        image_encoder.train()
        mlp.train()
        if refiner is not None:
            refiner.train()

    if test_loader is not None:
        test_metrics = evaluate(
            image_encoder=image_encoder,
            mlp=mlp,
            refiner=refiner,
            perceptual_loss_fn=perceptual_loss_fn,
            loader=test_loader,
            args=args,
            device=device,
            split_name="test_final",
            max_batches=0,
        )
        final_metrics.update({f"test_{k}": v for k, v in test_metrics.items()})
        image_encoder.train()
        mlp.train()
        if refiner is not None:
            refiner.train()

    final_path = out_dir / "ckpt_final.pt"
    save_checkpoint(
        path=final_path,
        global_step=global_step,
        image_encoder=image_encoder,
        mlp=mlp,
        refiner=refiner,
        optimizer=optimizer,
        scheduler=scheduler,
        args=args,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        metrics=final_metrics,
    )
    print(f"[save] {final_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--ids", type=str, default="327", help="Comma-separated subject ids. Used when --ids_file is not given.")
    parser.add_argument("--ids_file", type=str, default="", help="Text file with one subject id per line. Automatically split unless explicit split files are given.")

    parser.add_argument("--train_ids_file", type=str, default="/home/alphayoung8/log1/nebla/ids_file/train_ids.txt", help="Explicit train ids file. If given, automatic split is disabled.")
    parser.add_argument("--val_ids_file", type=str, default="/home/alphayoung8/log1/nebla/ids_file/val_ids.txt", help="Explicit validation ids file. Requires --train_ids_file.")
    parser.add_argument("--test_ids_file", type=str, default="/home/alphayoung8/log1/nebla/ids_file/test_ids.txt", help="Explicit test ids file. Requires --train_ids_file.")

    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio for automatic subject-level split.")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test ratio for automatic subject-level split.")
    parser.add_argument("--split_seed", type=int, default=0, help="Random seed for automatic split.")
    parser.add_argument("--no_split_shuffle", action="store_true", help="Do not shuffle ids before automatic split.")

    parser.add_argument("--simpx_root", type=str, default="/home/alphayoung8/log1/nebla/simpx_renderer/simpx_result_noresize")
    parser.add_argument("--cbct_root", type=str, default="/home/alphayoung8/log1/oral3d_reproduce/data/cbct_npy")
    parser.add_argument("--geom_root", type=str, default="/home/alphayoung8/log1/nebla/simpx_renderer/center_geometries")
    parser.add_argument(
        "--geom_xy_transform",
        type=str,
        default="rot90_cw",
        choices=["none", "rot90_cw", "rot90_ccw", "swap_xy", "flip_x", "flip_y"],
        help=(
            "Transform ray geometry XY coordinates into CBCT voxel XY coordinates. "
            "Use rot90_cw when Pred MIP must be rotated clockwise by 90 degrees to match GT MIP."
        ),
    )
    parser.add_argument("--out_dir", type=str, default="./logs/nebla_train_unet")

    parser.add_argument("--nerf_depth", type=int, default=8, help="Number of NeRF MLP layers. Larger values stack more MLP layers.")
    parser.add_argument("--nerf_width", type=int, default=256, help="Hidden width of the NeRF MLP.")
    parser.add_argument("--nerf_skips", type=str, default="4", help="Comma-separated skip-layer indices for the NeRF MLP.")

    parser.add_argument("--use_3d_refiner", action="store_true", default=True, help="Enable NeBLa-style 3D U-Net refinement after MLP scatter.")
    parser.add_argument("--refiner_f_maps", type=str, default="64,128,256,512", help="Comma-separated 3D U-Net feature dimensions.")
    parser.add_argument("--refiner_final_activation", type=str, default="sigmoid", choices=["sigmoid", "none", "identity", ""], help="Final activation for the 3D refiner.")
    parser.add_argument("--refiner_point_weight", type=float, default=0.0, help="Weight for sampled-point MSE after refinement.")
    parser.add_argument("--refiner_volume_weight", type=float, default=1.0, help="Weight for dense volume MSE after refinement.")
    parser.add_argument("--refiner_proj_weight", type=float, default=10.0, help="Weight for axial/coronal/sagittal MIP projection loss.")
    parser.add_argument("--refiner_perc_weight", type=float, default=1.0, help="Weight for VGG perceptual loss on axial/coronal/sagittal MIP images.")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--iters", type=int, default=100000)
    parser.add_argument("--n_points", type=int, default=32768)
    parser.add_argument("--n_samples", type=int, default=400)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["none", "cosine", "linear", "step"])
    parser.add_argument("--lr_min", type=float, default=1e-6, help="Final/minimum LR for cosine or linear schedule.")
    parser.add_argument("--warmup_iters", type=int, default=0, help="Linear warmup iterations before decay.")
    parser.add_argument("--lr_decay_iters", type=int, default=0, help="Decay length. Set <=0 to use --iters.")
    parser.add_argument("--lr_step_size", type=int, default=30000, help="StepLR period when --lr_scheduler step.")
    parser.add_argument("--lr_gamma", type=float, default=0.5, help="StepLR gamma when --lr_scheduler step.")

    parser.add_argument("--clip_min", type=float, default=-1000.0)
    parser.add_argument("--clip_max", type=float, default=4000.0)
    parser.add_argument("--already_normalized", action="store_true")

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--eval_every", type=int, default=1000, help="Run validation every N training steps. Set <=0 to disable periodic validation.")
    parser.add_argument("--eval_batches", type=int, default=10, help="Number of validation batches for periodic eval. Set <=0 for full validation split.")

    parser.add_argument("--val_image_every", type=int, default=1000, help="Save validation MIP images every N steps. Set <=0 to disable.")
    parser.add_argument("--val_image_max_items", type=int, default=2, help="Maximum validation subjects to visualize per save event.")
    parser.add_argument("--val_image_chunk_points", type=int, default=131072, help="Chunk size for rendering validation MIP images.")
    parser.add_argument("--val_mip_k_stride", type=int, default=1, help="Ray-depth stride for validation scatter-volume MIP rendering. Use 1 for full n_samples.")

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
