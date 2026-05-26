#!/usr/bin/env python3
"""
train_nebla.py

Training loop for:
    SimPX image -> UNET feature map -> point-conditioned NeRF/MLP -> CBCT intensity

Expected project files in the same directory:
    image_encoder.py
    mlp.py
    point_embbeder.py

Default data assumption:c
    simpx_root/{sid}.png or simpx_root/{sid}.npy
        PNG shape [Z, R], or NPY shape [Z, R] / [1, Z, R]

    cbct_root/{sid}.npy
        shape [D, H, W]
        numeric array or pickled dict containing one volume array

    geom_root/{sid}/rotation_geometry_{sid}.npz
        contains "ray_segments": [R, 4] = [x_start, y_start, x_end, y_end]

Target values are sampled from the GT CBCT volume by trilinear interpolation.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader

from models.image_encoder import UNET
from models.mlp import NeRF, gather_image_features
from models.point_embbeder import (
    build_3d_endpoints_from_npz,
    make_random_indices,
    encode_points_from_indices,
    sample_indexed_points_batched,
)


def load_npy_array(path: Union[str, Path], candidate_keys: Sequence[str] = ("volume", "vol", "ct", "cbct", "image", "data", "arr", "array")) -> np.ndarray:
    """
    Load a numeric .npy or a pickled object .npy.
    """
    path = str(path)

    try:
        arr = np.load(path, mmap_mode="r")
        if arr.dtype != object:
            return np.asarray(arr)
    except ValueError:
        pass

    obj = np.load(path, allow_pickle=True)

    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
        obj = obj.item()

    if isinstance(obj, dict):
        for key in candidate_keys:
            if key in obj:
                return np.asarray(obj[key])
        raise KeyError(f"No volume key found in {path}. Available keys: {list(obj.keys())}")

    return np.asarray(obj)




def load_image_array(path: Union[str, Path]) -> np.ndarray:
    """
    Load a SimPX image file, typically .png.

    Returns:
        img: [H, W] float32 array. H corresponds to z/row, W corresponds to ray index.
    """
    path = Path(path)
    img = Image.open(path)
    arr = np.asarray(img)

    # RGB/RGBA -> grayscale. SimPX should be scalar intensity.
    if arr.ndim == 3:
        arr = arr[..., :3].astype(np.float32).mean(axis=-1)
    elif arr.ndim != 2:
        raise ValueError(f"Expected 2D SimPX image or RGB image, got shape {arr.shape} from {path}")

    return arr.astype(np.float32)


def load_simpx_array(path: Union[str, Path]) -> np.ndarray:
    """
    Load SimPX from .npy or image file.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return load_npy_array(path)
    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        return load_image_array(path)
    raise ValueError(f"Unsupported SimPX file extension: {path}")

def normalize_volume(vol: np.ndarray, clip_min: float = -1000.0, clip_max: float = 4000.0) -> np.ndarray:
    """
    Convert GT CBCT to [0, 1], because the current NeRF MLP returns sigmoid(output).
    """
    vol = vol.astype(np.float32)
    vol = np.clip(vol, clip_min, clip_max)
    vol = (vol - clip_min) / max(clip_max - clip_min, 1e-6)
    return vol.astype(np.float32)


def normalize_simpx(img: np.ndarray) -> np.ndarray:
    """
    Basic SimPX normalization. Supports [Z, R] or [1, Z, R].
    """
    img = img.astype(np.float32)

    if img.ndim == 2:
        img = img[None, ...]
    elif img.ndim == 3:
        if img.shape[0] != 1:
            raise ValueError(f"Expected SimPX shape [Z,R] or [1,Z,R], got {img.shape}")
    else:
        raise ValueError(f"Expected SimPX shape [Z,R] or [1,Z,R], got {img.shape}")

    vmin = float(np.nanmin(img))
    vmax = float(np.nanmax(img))

    if vmax > 1.5:
        img = (img - vmin) / max(vmax - vmin, 1e-6)
    else:
        img = np.clip(img, 0.0, 1.0)

    return img.astype(np.float32)



def apply_geom_xy_transform(
    pts_xyz: np.ndarray,
    volume_shape_zyx: Tuple[int, int, int],
    mode: str,
) -> np.ndarray:
    """
    Convert geometry XY coordinates to CBCT voxel XY coordinates.

    The model/volume convention is:
        volume[z, y, x] with shape [D, H, W]
        point_xyz = [x, y, z]

    Some geometry npz files are generated in an axial-image display coordinate
    system that is rotated relative to the raw CBCT array.  If the validation
    Pred MIP has to be rotated clockwise by 90 degrees to match GT MIP, use
    mode="rot90_cw".  This applies the coordinate transform before training,
    validation sampling, and scatter-volume visualization.
    """
    mode = str(mode).lower()
    if mode in {"", "none", "identity"}:
        return pts_xyz

    D, H, W = map(int, volume_shape_zyx)
    out = pts_xyz.copy()
    x = pts_xyz[..., 0].astype(np.float32)
    y = pts_xyz[..., 1].astype(np.float32)

    # Scale factors let the transform work even when H != W.
    sx = (W - 1.0) / max(H - 1.0, 1.0)
    sy = (H - 1.0) / max(W - 1.0, 1.0)

    if mode == "rot90_cw":
        # Equivalent to np.rot90(image, k=-1) in MIP/image space:
        # new_x = H - 1 - old_y, new_y = old_x.
        out[..., 0] = (H - 1.0 - y) * sx
        out[..., 1] = x * sy
    elif mode == "rot90_ccw":
        # Equivalent to np.rot90(image, k=1).
        out[..., 0] = y * sx
        out[..., 1] = (W - 1.0 - x) * sy
    elif mode == "swap_xy":
        out[..., 0] = y * sx
        out[..., 1] = x * sy
    elif mode == "flip_x":
        out[..., 0] = W - 1.0 - x
    elif mode == "flip_y":
        out[..., 1] = H - 1.0 - y
    else:
        raise ValueError(
            f"Unsupported --geom_xy_transform={mode}. "
            "Choose from none, rot90_cw, rot90_ccw, swap_xy, flip_x, flip_y."
        )

    out[..., 0] = np.clip(out[..., 0], 0.0, max(W - 1.0, 0.0))
    out[..., 1] = np.clip(out[..., 1], 0.0, max(H - 1.0, 0.0))
    return out.astype(np.float32, copy=False)


class SimPXCBCTDataset(Dataset):
    def __init__(
        self,
        ids: Sequence[str],
        simpx_root: Union[str, Path],
        cbct_root: Union[str, Path],
        geom_root: Union[str, Path],
        n_samples: int = 200,
        clip_min: float = -1000.0,
        clip_max: float = 4000.0,
        already_normalized: bool = False,
        geom_xy_transform: str = "none",
    ):
        self.ids = [str(x).strip() for x in ids if str(x).strip()]
        self.simpx_root = Path(simpx_root)
        self.cbct_root = Path(cbct_root)
        self.geom_root = Path(geom_root)
        self.n_samples = int(n_samples)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)
        self.already_normalized = bool(already_normalized)
        self.geom_xy_transform = str(geom_xy_transform).lower()

        if len(self.ids) == 0:
            raise ValueError("No subject ids were provided.")

    def _simpx_path(self, sid: str) -> Path:
        image_exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]

        candidates = [
            self.simpx_root / f"{sid}.npy",
            self.simpx_root / sid / f"{sid}.npy",
            self.simpx_root / sid / "simpx.npy",
            self.simpx_root / sid / "pano.npy",
        ]

        for ext in image_exts:
            candidates.extend([
                self.simpx_root / f"{sid}{ext}",
                self.simpx_root / sid / f"{sid}{ext}",
                self.simpx_root / sid / f"simpx{ext}",
                self.simpx_root / sid / f"pano{ext}",
                self.simpx_root / sid / f"pano_final{ext}",
            ])

        for p in candidates:
            if p.exists():
                return p

        # Fallback: choose the first image under simpx_root/sid or simpx_root containing sid.
        glob_patterns = []
        for ext in image_exts:
            glob_patterns.extend([
                str(self.simpx_root / sid / f"*{ext}"),
                str(self.simpx_root / f"*{sid}*{ext}"),
            ])
        import glob
        for pattern in glob_patterns:
            matches = sorted(glob.glob(pattern))
            if matches:
                return Path(matches[0])

        raise FileNotFoundError(f"Could not find SimPX image/npy for sid={sid}. Tried: {candidates}")

    def _cbct_path(self, sid: str) -> Path:
        candidates = [
            self.cbct_root / f"{sid}.npy",
            self.cbct_root / sid / f"{sid}.npy",
            self.cbct_root / sid / "volume.npy",
        ]
        for p in candidates:
            if p.exists():
                return p
        raise FileNotFoundError(f"Could not find CBCT .npy for sid={sid}. Tried: {candidates}")

    def _geom_path(self, sid: str) -> Path:
        candidates = [
            self.geom_root / sid / f"rotation_geometry_{sid}.npz",
            self.geom_root / f"rotation_geometry_{sid}.npz",
            self.geom_root / sid / "rotation_geometry.npz",
        ]
        for p in candidates:
            if p.exists():
                return p
        raise FileNotFoundError(f"Could not find geometry npz for sid={sid}. Tried: {candidates}")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sid = self.ids[index]

        simpx = normalize_simpx(load_simpx_array(self._simpx_path(sid)))

        volume = load_npy_array(self._cbct_path(sid))
        if volume.ndim != 3:
            raise ValueError(f"Expected CBCT volume [D,H,W], got {volume.shape} for sid={sid}")

        volume = volume.astype(np.float32)
        if not self.already_normalized:
            volume = normalize_volume(volume, self.clip_min, self.clip_max)
        else:
            volume = np.clip(volume, 0.0, 1.0).astype(np.float32)

        volume_shape = tuple(map(int, volume.shape))
        simpx_height = int(simpx.shape[-2])

        start_np, end_np = build_3d_endpoints_from_npz(
            geometry_npz_path=str(self._geom_path(sid)),
            volume_shape_zyx=volume_shape,
            simpx_height=simpx_height,
        )

        start_np = apply_geom_xy_transform(start_np, volume_shape, self.geom_xy_transform)
        end_np = apply_geom_xy_transform(end_np, volume_shape, self.geom_xy_transform)

        if simpx.shape[-1] != start_np.shape[1]:
            raise ValueError(
                f"Ray count mismatch for sid={sid}: "
                f"simpx R={simpx.shape[-1]}, geometry R={start_np.shape[1]}"
            )

        return {
            "sid": sid,
            "simpx": torch.from_numpy(simpx),
            "volume": torch.from_numpy(volume),
            "start_xyz": torch.from_numpy(start_np),
            "end_xyz": torch.from_numpy(end_np),
            "volume_shape": torch.tensor(volume_shape).long(),
        }


def collate_same_shape(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "sid": [b["sid"] for b in batch],
        "simpx": torch.stack([b["simpx"] for b in batch], dim=0),
        "volume": torch.stack([b["volume"] for b in batch], dim=0),
        "start_xyz": torch.stack([b["start_xyz"] for b in batch], dim=0),
        "end_xyz": torch.stack([b["end_xyz"] for b in batch], dim=0),
        "volume_shape": torch.stack([b["volume_shape"] for b in batch], dim=0),
    }


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



def read_ids_file(path: Union[str, Path]) -> List[str]:
    """
    Read one subject id per line.

    Blank lines and lines starting with '#' are ignored.
    Inline comments are also supported:
        327  # subject 327
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"ids file not found: {path}")

    ids: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                ids.append(line)
    return ids


def parse_ids_string(ids: str) -> List[str]:
    """
    Parse comma-separated subject ids.
    """
    return [x.strip() for x in str(ids).split(",") if x.strip()]


def parse_int_list(value: Union[str, Sequence[int]]) -> List[int]:
    """
    Parse comma-separated integers such as "4,8" or "64,128,256,512".
    """
    if isinstance(value, str):
        out = [int(x.strip()) for x in value.split(",") if x.strip()]
    else:
        out = [int(x) for x in value]

    if len(out) == 0:
        raise ValueError(f"Expected at least one integer, got {value!r}")

    return out


def deduplicate_preserve_order(ids: Sequence[str]) -> List[str]:
    """
    Remove duplicated ids while preserving order.
    """
    out: List[str] = []
    seen = set()
    for sid in ids:
        sid = str(sid).strip()
        if not sid or sid in seen:
            continue
        out.append(sid)
        seen.add(sid)
    return out


def split_ids(
    ids: Sequence[str],
    val_ratio: float,
    test_ratio: float,
    seed: int,
    shuffle: bool = True,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split ids into train/val/test.

    The split is subject-level. For very small datasets, val/test can become empty
    so that train is not empty.
    """
    ids = deduplicate_preserve_order(ids)
    n = len(ids)
    if n == 0:
        raise ValueError("No subject ids were provided for splitting.")

    val_ratio = float(val_ratio)
    test_ratio = float(test_ratio)
    if val_ratio < 0.0 or test_ratio < 0.0:
        raise ValueError(f"val_ratio and test_ratio must be non-negative, got {val_ratio}, {test_ratio}")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError(
            f"val_ratio + test_ratio must be < 1.0 so that train is non-empty, "
            f"got {val_ratio + test_ratio}"
        )

    order = list(ids)
    if shuffle:
        rng = np.random.default_rng(int(seed))
        perm = rng.permutation(n).tolist()
        order = [order[i] for i in perm]

    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))

    # Keep at least one item for a requested split when the dataset is large enough.
    if test_ratio > 0.0 and n >= 3:
        n_test = max(1, n_test)
    if val_ratio > 0.0 and (n - n_test) >= 2:
        n_val = max(1, n_val)

    # Guarantee at least one train subject.
    while n_val + n_test >= n:
        if n_test >= n_val and n_test > 0:
            n_test -= 1
        elif n_val > 0:
            n_val -= 1
        else:
            break

    test_ids = order[:n_test]
    val_ids = order[n_test:n_test + n_val]
    train_ids = order[n_test + n_val:]

    if len(train_ids) == 0:
        raise ValueError("Split produced an empty train set. Reduce --val_ratio or --test_ratio.")

    return train_ids, val_ids, test_ids


def write_split_files(out_dir: Path, train_ids: Sequence[str], val_ids: Sequence[str], test_ids: Sequence[str]) -> None:
    """
    Save the resolved split for reproducibility.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    split_map = {
        "ids_train.txt": train_ids,
        "ids_val.txt": val_ids,
        "ids_test.txt": test_ids,
    }
    for filename, ids in split_map.items():
        (out_dir / filename).write_text("\n".join(map(str, ids)) + ("\n" if ids else ""), encoding="utf-8")


def resolve_id_splits(args: argparse.Namespace) -> Tuple[List[str], List[str], List[str]]:
    """
    Resolve train/val/test subject ids.

    Priority:
      1) If --train_ids_file is given, use explicit split files.
         --val_ids_file and --test_ids_file are optional.
      2) Otherwise, read --ids_file if given, else --ids.
         Then automatically split by --val_ratio and --test_ratio.
    """
    if args.train_ids_file:
        train_ids = read_ids_file(args.train_ids_file)
        val_ids = read_ids_file(args.val_ids_file) if args.val_ids_file else []
        test_ids = read_ids_file(args.test_ids_file) if args.test_ids_file else []

        train_ids = deduplicate_preserve_order(train_ids)
        val_ids = deduplicate_preserve_order(val_ids)
        test_ids = deduplicate_preserve_order(test_ids)

        if len(train_ids) == 0:
            raise ValueError("--train_ids_file produced an empty train id list.")
        return train_ids, val_ids, test_ids

    if args.val_ids_file or args.test_ids_file:
        raise ValueError("--val_ids_file/--test_ids_file require --train_ids_file.")

    if args.ids_file:
        ids = read_ids_file(args.ids_file)
    else:
        ids = parse_ids_string(args.ids)

    return split_ids(
        ids=ids,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.split_seed,
        shuffle=not args.no_split_shuffle,
    )


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
    loader: Union[DataLoader, None],
    args: argparse.Namespace,
    device: torch.device,
    split_name: str,
    max_batches: int = 0,
) -> Dict[str, float]:
    """
    Evaluate on a split using random sampled points.

    max_batches <= 0 means evaluate the full split.
    """
    if loader is None:
        return {}

    image_encoder.eval()
    mlp.eval()

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
            loss = F.mse_loss(pred, target)

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


def save_checkpoint(
    path: Path,
    global_step: int,
    image_encoder: nn.Module,
    mlp: nn.Module,
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

    print(
        f"[model] UNET out_channels=128 features=[64, 128, 256, 512] "
        f"NeRF depth={args.nerf_depth} width={args.nerf_width} skips={nerf_skips}"
    )

    params = list(image_encoder.parameters()) + list(mlp.parameters())
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
                loss = F.mse_loss(pred, target)

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
                print(
                    f"[step {global_step:06d}] "
                    f"loss={loss.item():.6f} psnr={psnr.item():.2f} "
                    f"lr={get_current_lr(optimizer):.6e} "
                    f"B={B} N={args.n_points} simpx={tuple(simpx.shape)} "
                    f"pred={tuple(pred.shape)} target={tuple(target.shape)}"
                )

            if val_loader is not None and args.eval_every > 0 and global_step > 0 and global_step % args.eval_every == 0:
                val_metrics = evaluate(
                    image_encoder=image_encoder,
                    mlp=mlp,
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

            if val_loader is not None and args.val_image_every > 0 and global_step > 0 and global_step % args.val_image_every == 0:
                save_validation_mip_images(
                    image_encoder=image_encoder,
                    mlp=mlp,
                    loader=val_loader,
                    args=args,
                    device=device,
                    global_step=global_step,
                )
                image_encoder.train()
                mlp.train()

            if global_step > 0 and global_step % args.save_every == 0:
                ckpt_path = out_dir / f"ckpt_{global_step:06d}.pt"
                save_checkpoint(
                    path=ckpt_path,
                    global_step=global_step,
                    image_encoder=image_encoder,
                    mlp=mlp,
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
                loader=val_loader,
                args=args,
                device=device,
                global_step=global_step,
            )
        image_encoder.train()
        mlp.train()

    if test_loader is not None:
        test_metrics = evaluate(
            image_encoder=image_encoder,
            mlp=mlp,
            loader=test_loader,
            args=args,
            device=device,
            split_name="test_final",
            max_batches=0,
        )
        final_metrics.update({f"test_{k}": v for k, v in test_metrics.items()})
        image_encoder.train()
        mlp.train()

    final_path = out_dir / "ckpt_final.pt"
    save_checkpoint(
        path=final_path,
        global_step=global_step,
        image_encoder=image_encoder,
        mlp=mlp,
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
    parser.add_argument("--out_dir", type=str, default="./logs/nebla_train2")

    parser.add_argument("--nerf_depth", type=int, default=8, help="Number of NeRF MLP layers. Larger values stack more MLP layers.")
    parser.add_argument("--nerf_width", type=int, default=256, help="Hidden width of the NeRF MLP.")
    parser.add_argument("--nerf_skips", type=str, default="4", help="Comma-separated skip-layer indices for the NeRF MLP.")

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
    parser.add_argument("--already_normalized", default=True, action="store_true")

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
