#!/usr/bin/env python3
"""
Utility functions for NEBLA training.

This file contains data I/O, normalization, geometry-coordinate transforms,
and subject-id split utilities extracted from train.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import numpy as np
from PIL import Image

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
