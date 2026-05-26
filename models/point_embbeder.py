"""
point_embedder.py

NeBLa-style point sampling and positional encoding utilities.

Coordinate convention
---------------------
CBCT volume array:
    volume[z, y, x] with shape [D, H, W]

Network point coordinate:
    point_xyz = [x, y, z]

Geometry file assumption
------------------------
rotation_geometry_*.npz contains:
    ray_segments: [R, 4] = [x_start, y_start, x_end, y_end]

The 2D axial ray segment is lifted to 3D by attaching z:
    start_xyz[z, r] = [x_start_r, y_start_r, z]
    end_xyz[z, r]   = [x_end_r,   y_end_r,   z]

For memory efficiency, this file is designed to keep only endpoints
[start_xyz, end_xyz] and generate sampled points / gamma(x) only for
the requested indices.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch


ArrayLikeShape = Union[Tuple[int, int, int], Sequence[int]]


def build_3d_endpoints_from_npz(
    geometry_npz_path: str,
    volume_shape_zyx: ArrayLikeShape,
    simpx_height: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build 3D ray endpoints from 2D axial ray segments.

    Args:
        geometry_npz_path:
            Path to rotation_geometry_*.npz containing "ray_segments".
        volume_shape_zyx:
            CBCT shape [D, H, W].
        simpx_height:
            Number of z rows to create.
            If None, simpx_height = D.

    Returns:
        start_xyz:
            [Z, R, 3], float32
        end_xyz:
            [Z, R, 3], float32

    Notes:
        If simpx_height == D, z_values are exactly [0, 1, ..., D-1].
        If simpx_height != D, z_values are linearly spaced in [0, D-1].
    """
    D, H, W = map(int, volume_shape_zyx)
    Z = D if simpx_height is None else int(simpx_height)

    geom = np.load(geometry_npz_path, allow_pickle=True)
    if "ray_segments" not in geom:
        raise KeyError(
            f"{geometry_npz_path} does not contain key 'ray_segments'. "
            f"Available keys: {list(geom.keys())}"
        )

    ray_segments = geom["ray_segments"].astype(np.float32)  # [R, 4]
    if ray_segments.ndim != 2 or ray_segments.shape[1] != 4:
        raise ValueError(
            f"ray_segments must have shape [R, 4], got {ray_segments.shape}"
        )

    # ray_segments = [x_start, y_start, x_end, y_end]
    x0 = ray_segments[:, 0]
    y0 = ray_segments[:, 1]
    x1 = ray_segments[:, 2]
    y1 = ray_segments[:, 3]

    R = ray_segments.shape[0]

    if Z == D:
        z_values = np.arange(D, dtype=np.float32)
    else:
        z_values = np.linspace(0.0, float(D - 1), Z, dtype=np.float32)

    start_xyz = np.empty((Z, R, 3), dtype=np.float32)
    end_xyz = np.empty((Z, R, 3), dtype=np.float32)

    start_xyz[..., 0] = x0[None, :]
    start_xyz[..., 1] = y0[None, :]
    start_xyz[..., 2] = z_values[:, None]

    end_xyz[..., 0] = x1[None, :]
    end_xyz[..., 1] = y1[None, :]
    end_xyz[..., 2] = z_values[:, None]

    return start_xyz, end_xyz


def build_3d_endpoints_from_npy_and_npz(
    npy_path: str,
    geometry_npz_path: str,
    simpx_height: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]:
    """
    Convenience wrapper that reads volume shape from a .npy file.

    Args:
        npy_path:
            Path to CBCT volume .npy with shape [D, H, W].
            Loaded with mmap_mode="r" to avoid reading the full array.
        geometry_npz_path:
            Path to rotation_geometry_*.npz.
        simpx_height:
            Optional z-row count.

    Returns:
        start_xyz:
            [Z, R, 3]
        end_xyz:
            [Z, R, 3]
        volume_shape_zyx:
            (D, H, W)
    """
    volume = np.load(npy_path, mmap_mode="r")
    if volume.ndim != 3:
        raise ValueError(f"Expected volume shape [D, H, W], got {volume.shape}")

    volume_shape_zyx = tuple(map(int, volume.shape))
    start_xyz, end_xyz = build_3d_endpoints_from_npz(
        geometry_npz_path=geometry_npz_path,
        volume_shape_zyx=volume_shape_zyx,
        simpx_height=simpx_height,
    )
    return start_xyz, end_xyz, volume_shape_zyx


def endpoints_to_torch(
    start_xyz: np.ndarray,
    end_xyz: np.ndarray,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert endpoint arrays to torch tensors.

    Args:
        start_xyz:
            [Z, R, 3]
        end_xyz:
            [Z, R, 3]

    Returns:
        start_t:
            [Z, R, 3]
        end_t:
            [Z, R, 3]
    """
    start_t = torch.as_tensor(start_xyz, dtype=dtype, device=device)
    end_t = torch.as_tensor(end_xyz, dtype=dtype, device=device)

    if start_t.shape != end_t.shape:
        raise ValueError(f"start/end shape mismatch: {start_t.shape} vs {end_t.shape}")
    if start_t.ndim != 3 or start_t.shape[-1] != 3:
        raise ValueError(f"Expected endpoint shape [Z, R, 3], got {start_t.shape}")

    return start_t, end_t


def sample_indexed_points(
    start_xyz: torch.Tensor,
    end_xyz: torch.Tensor,
    z_idx: torch.Tensor,
    r_idx: torch.Tensor,
    k_idx: torch.Tensor,
    n_samples: int,
    volume_shape_zyx: ArrayLikeShape,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Sample 3D points from selected endpoint indices.

    This is the memory-efficient version:
    it does NOT create [D, R, K, 3] or [D, R, K, 42].

    Args:
        start_xyz:
            [Z, R, 3]
        end_xyz:
            [Z, R, 3]
        z_idx:
            [N], z-row index into start_xyz/end_xyz.
        r_idx:
            [N], ray index.
        k_idx:
            [N], sample index along the ray, 0 <= k < n_samples.
        n_samples:
            K, number of uniform samples along a ray.
        volume_shape_zyx:
            Original CBCT shape [D, H, W], used for coordinate normalization.
        normalize:
            If True, return normalized coordinates:
                [x/(W-1), y/(H-1), z/(D-1)]
            If False, return voxel-index coordinates [x, y, z].

    Returns:
        points:
            [N, 3]
    """
    if start_xyz.shape != end_xyz.shape:
        raise ValueError(f"start/end shape mismatch: {start_xyz.shape} vs {end_xyz.shape}")
    if start_xyz.ndim != 3 or start_xyz.shape[-1] != 3:
        raise ValueError(f"Expected endpoint shape [Z, R, 3], got {start_xyz.shape}")

    z_idx = z_idx.long()
    r_idx = r_idx.long()
    k_idx = k_idx.long()

    if not (z_idx.shape == r_idx.shape == k_idx.shape):
        raise ValueError(
            f"Index shapes must match, got z={z_idx.shape}, r={r_idx.shape}, k={k_idx.shape}"
        )

    if n_samples <= 1:
        t = torch.zeros_like(k_idx, dtype=start_xyz.dtype)
    else:
        t = k_idx.to(dtype=start_xyz.dtype) / float(n_samples - 1)

    start = start_xyz[z_idx, r_idx]  # [N, 3]
    end = end_xyz[z_idx, r_idx]      # [N, 3]

    points = (1.0 - t[:, None]) * start + t[:, None] * end  # [N, 3]

    if not normalize:
        return points

    D, H, W = map(int, volume_shape_zyx)
    scale = torch.tensor(
        [W - 1.0, H - 1.0, D - 1.0],
        dtype=points.dtype,
        device=points.device,
    ).clamp_min(1.0)

    return points / scale[None, :]


def sample_indexed_points_batched(
    start_xyz: torch.Tensor,
    end_xyz: torch.Tensor,
    z_idx: torch.Tensor,
    r_idx: torch.Tensor,
    k_idx: torch.Tensor,
    n_samples: int,
    volume_shape_zyx: ArrayLikeShape,
    batch_idx: Optional[torch.Tensor] = None,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Batched version of sample_indexed_points.

    Supports either shared geometry:
        start_xyz/end_xyz: [Z, R, 3]
    or per-sample geometry:
        start_xyz/end_xyz: [B, Z, R, 3]

    Args:
        z_idx, r_idx, k_idx:
            Either [B, N] or [N].
        batch_idx:
            Required only when start_xyz is [B, Z, R, 3] and indices are flattened [N].
            Shape [N], with values in [0, B-1].
            If z_idx/r_idx/k_idx are [B, N], batch_idx is inferred.

    Returns:
        points:
            [B, N, 3] if indices are [B, N]
            [N, 3] if indices are [N]
    """
    if start_xyz.shape != end_xyz.shape:
        raise ValueError(f"start/end shape mismatch: {start_xyz.shape} vs {end_xyz.shape}")

    original_index_shape = z_idx.shape

    if start_xyz.ndim == 3:
        # Shared geometry: [Z, R, 3]
        if z_idx.ndim == 1:
            return sample_indexed_points(
                start_xyz, end_xyz, z_idx, r_idx, k_idx,
                n_samples=n_samples,
                volume_shape_zyx=volume_shape_zyx,
                normalize=normalize,
            )

        if z_idx.ndim == 2:
            B, N = z_idx.shape
            points = sample_indexed_points(
                start_xyz,
                end_xyz,
                z_idx.reshape(-1),
                r_idx.reshape(-1),
                k_idx.reshape(-1),
                n_samples=n_samples,
                volume_shape_zyx=volume_shape_zyx,
                normalize=normalize,
            )
            return points.reshape(B, N, 3)

        raise ValueError(f"Unsupported index shape: {original_index_shape}")

    if start_xyz.ndim != 4 or start_xyz.shape[-1] != 3:
        raise ValueError(
            f"Expected start_xyz shape [Z,R,3] or [B,Z,R,3], got {start_xyz.shape}"
        )

    # Per-sample geometry: [B, Z, R, 3]
    B = start_xyz.shape[0]

    if z_idx.ndim == 2:
        b_idx = torch.arange(B, device=start_xyz.device)[:, None].expand_as(z_idx)
        flat = True
        B_idx, N = z_idx.shape
        if B_idx != B:
            raise ValueError(f"Index batch {B_idx} does not match geometry batch {B}")
        b_flat = b_idx.reshape(-1)
        z_flat = z_idx.reshape(-1).long()
        r_flat = r_idx.reshape(-1).long()
        k_flat = k_idx.reshape(-1).long()
    elif z_idx.ndim == 1:
        if batch_idx is None:
            raise ValueError(
                "batch_idx is required when using per-sample geometry [B,Z,R,3] "
                "with flattened indices [N]."
            )
        flat = False
        b_flat = batch_idx.long()
        z_flat = z_idx.long()
        r_flat = r_idx.long()
        k_flat = k_idx.long()
    else:
        raise ValueError(f"Unsupported index shape: {original_index_shape}")

    if n_samples <= 1:
        t = torch.zeros_like(k_flat, dtype=start_xyz.dtype)
    else:
        t = k_flat.to(dtype=start_xyz.dtype) / float(n_samples - 1)

    start = start_xyz[b_flat, z_flat, r_flat]  # [M, 3]
    end = end_xyz[b_flat, z_flat, r_flat]      # [M, 3]
    points = (1.0 - t[:, None]) * start + t[:, None] * end

    if normalize:
        D, H, W = map(int, volume_shape_zyx)
        scale = torch.tensor(
            [W - 1.0, H - 1.0, D - 1.0],
            dtype=points.dtype,
            device=points.device,
        ).clamp_min(1.0)
        points = points / scale[None, :]

    if z_idx.ndim == 2:
        return points.reshape(B, N, 3)

    return points


def positional_encoding_nebla(
    points_xyz_norm: torch.Tensor,
    multires: int = 7,
    include_input: bool = False,
) -> torch.Tensor:
    """
    NeBLa/NeRF-style sinusoidal positional encoding.

    Args:
        points_xyz_norm:
            [..., 3], usually normalized [x, y, z].
        multires:
            Number of frequency bands. NeBLa uses 7.
        include_input:
            If False, output dim = 3 * 2 * multires.
            For multires=7, output dim = 42.
            The public NeBLa point_embedder uses include_input=False.

    Returns:
        encoded:
            [..., 42] when multires=7 and include_input=False.
    """
    if points_xyz_norm.shape[-1] != 3:
        raise ValueError(f"Expected last dim 3, got {points_xyz_norm.shape}")

    freq_bands = 2.0 ** torch.linspace(
        0.0,
        multires - 1,
        steps=multires,
        device=points_xyz_norm.device,
        dtype=points_xyz_norm.dtype,
    )

    encoded = []
    if include_input:
        encoded.append(points_xyz_norm)

    # Order matches the common Embedder loop:
    # for freq in freq_bands:
    #   for p_fn in [sin, cos]:
    #       p_fn(points * freq)
    for freq in freq_bands:
        encoded.append(torch.sin(points_xyz_norm * freq))
        encoded.append(torch.cos(points_xyz_norm * freq))

    return torch.cat(encoded, dim=-1)


def make_random_indices(
    batch_size: int,
    z_count: int,
    ray_count: int,
    n_samples: int,
    n_points: int,
    device: Union[str, torch.device],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Utility for random point sampling during training.

    Returns:
        z_idx, r_idx, k_idx:
            [B, n_points]
    """
    z_idx = torch.randint(0, z_count, (batch_size, n_points), device=device)
    r_idx = torch.randint(0, ray_count, (batch_size, n_points), device=device)
    k_idx = torch.randint(0, n_samples, (batch_size, n_points), device=device)
    return z_idx, r_idx, k_idx


def make_sequential_indices(
    z_count: int,
    ray_count: int,
    n_samples: int,
    start: int,
    length: int,
    device: Union[str, torch.device],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert flattened point indices to (z_idx, r_idx, k_idx).

    The flatten order is:
        flat = ((z * R) + r) * K + k

    Useful for chunked full-volume inference/scatter.

    Returns:
        z_idx, r_idx, k_idx:
            [length]
    """
    total = z_count * ray_count * n_samples
    if start < 0 or start >= total:
        raise ValueError(f"start must be in [0, {total}), got {start}")

    end = min(start + length, total)
    flat = torch.arange(start, end, device=device, dtype=torch.long)

    k_idx = flat % n_samples
    tmp = flat // n_samples
    r_idx = tmp % ray_count
    z_idx = tmp // ray_count

    return z_idx, r_idx, k_idx


def encode_points_from_indices(
    start_xyz: torch.Tensor,
    end_xyz: torch.Tensor,
    z_idx: torch.Tensor,
    r_idx: torch.Tensor,
    k_idx: torch.Tensor,
    n_samples: int,
    volume_shape_zyx: ArrayLikeShape,
    multires: int = 7,
) -> torch.Tensor:
    """
    Convenience function:
        indices -> normalized points -> gamma(x)

    Supports shared geometry [Z,R,3] or batched geometry [B,Z,R,3].

    Returns:
        gamma:
            [N, 42] or [B, N, 42]
    """
    points_norm = sample_indexed_points_batched(
        start_xyz=start_xyz,
        end_xyz=end_xyz,
        z_idx=z_idx,
        r_idx=r_idx,
        k_idx=k_idx,
        n_samples=n_samples,
        volume_shape_zyx=volume_shape_zyx,
        normalize=True,
    )
    return positional_encoding_nebla(points_norm, multires=multires)


if __name__ == "__main__":
    # Minimal shape test without external files.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    D, H, W = 200, 320, 320
    R = 311
    K = 200
    B = 2
    N = 4096

    # Dummy endpoints with shape [D, R, 3]
    start = torch.zeros(D, R, 3, device=device)
    end = torch.zeros(D, R, 3, device=device)
    start[..., 0] = 0.0
    start[..., 1] = 319.0
    start[..., 2] = torch.arange(D, device=device)[:, None]
    end[..., 0] = 319.0
    end[..., 1] = 0.0
    end[..., 2] = torch.arange(D, device=device)[:, None]

    z_idx, r_idx, k_idx = make_random_indices(
        batch_size=B,
        z_count=D,
        ray_count=R,
        n_samples=K,
        n_points=N,
        device=device,
    )

    gamma = encode_points_from_indices(
        start_xyz=start,
        end_xyz=end,
        z_idx=z_idx,
        r_idx=r_idx,
        k_idx=k_idx,
        n_samples=K,
        volume_shape_zyx=(D, H, W),
        multires=7,
    )

    print("gamma:", tuple(gamma.shape))  # [B, N, 42]
