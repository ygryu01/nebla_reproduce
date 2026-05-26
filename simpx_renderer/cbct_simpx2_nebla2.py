import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
torch.set_printoptions(profile='full')
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from scipy.interpolate import BSpline
from PIL import Image

# =========================================================
# Config
# =========================================================

device = torch.device("cuda")
torch.set_default_dtype(torch.float32)

NUM_X = 700
NUM_SAMPLES = 240
# Z_HALF = 150

VOXEL_MM = 0.25
SLICE_DEPTH = 40.0 / VOXEL_MM  # mm / voxel_mm => voxels

# -----------------------------
# SimPX-style pixel formula
# (원하신 "계산식/샘플링 방식"과 동일 계열)
# -----------------------------
CLIP_LOW   = -500.0
CLIP_HIGH  = 2500.0
BONE_THR   = 500.0
SIMPX_BETA = 1e-8
S_MAX      = 20.0

import json


def load_rotation_geometry_for_volume(npz_path, H, W, device=None):
    """
    저장된 rotation_geometry_{subject}.npz를 불러와서
    input image 좌표계(x=col, y=row)를 cbct_simpx2.py의 voxel 좌표계로 변환합니다.

    기존 load_centerline()은 image를 arr.T 한 뒤 scale하므로,
    image coordinate (x_img=col, y_img=row)는 volume coordinate에서

        x_vox = y_img * (H - 1) / (img_h - 1)
        y_vox = x_img * (W - 1) / (img_w - 1)

    로 들어갑니다.

    Returns
    -------
    ray_origins_vox : torch.Tensor, shape (R, 2)
    ray_dirs_vox    : torch.Tensor, shape (R, 2)
    centers_vox     : torch.Tensor, shape (21, 2)
    meta            : dict
    """
    if device is None:
        device = globals().get("device", torch.device("cuda"))

    data = np.load(npz_path, allow_pickle=False)

    ray_origins_img = data["ray_origins"].astype(np.float32)  # (R, 2), x_img,y_img
    ray_dirs_img = data["ray_dirs"].astype(np.float32)        # (R, 2), dx_img,dy_img
    centers_img = data["centers"].astype(np.float32)          # (21, 2)

    meta = json.loads(str(data["meta_json"]))
    img_h, img_w = meta["image_shape_H_W"]

    sx = (H - 1.0) / (img_h - 1.0) if img_h > 1 else 1.0
    sy = (W - 1.0) / (img_w - 1.0) if img_w > 1 else 1.0

    # image xy -> volume xy
    # x_vox = y_img * sx
    # y_vox = x_img * sy
    ray_origins_vox_np = np.empty_like(ray_origins_img, dtype=np.float32)
    ray_origins_vox_np[:, 0] = ray_origins_img[:, 1] * sx
    ray_origins_vox_np[:, 1] = ray_origins_img[:, 0] * sy
    

    centers_vox_np = np.empty_like(centers_img, dtype=np.float32)
    centers_vox_np[:, 0] = centers_img[:, 1] * sx
    centers_vox_np[:, 1] = centers_img[:, 0] * sy

    # direction도 같은 축 변환 후 normalize
    ray_dirs_vox_np = np.empty_like(ray_dirs_img, dtype=np.float32)
    ray_dirs_vox_np[:, 0] = ray_dirs_img[:, 1] * sx
    ray_dirs_vox_np[:, 1] = ray_dirs_img[:, 0] * sy

    norm = np.linalg.norm(ray_dirs_vox_np, axis=1, keepdims=True)
    ray_dirs_vox_np = ray_dirs_vox_np / (norm + 1e-12)

    ray_origins_vox = torch.from_numpy(ray_origins_vox_np).to(
        device=device,
        dtype=torch.float32,
    )
    ray_dirs_vox = torch.from_numpy(ray_dirs_vox_np).to(
        device=device,
        dtype=torch.float32,
    )
    centers_vox = torch.from_numpy(centers_vox_np).to(
        device=device,
        dtype=torch.float32,
    )

    return ray_origins_vox, ray_dirs_vox, centers_vox, meta



def get_default_circular_fov(H, W, margin=1.0):
    """
    원형 FOV를 axial plane 중심 원으로 둡니다.

    좌표계:
      - center_xy[0] = x coordinate on H-axis
      - center_xy[1] = y coordinate on W-axis
    """
    cx = (float(H) - 1.0) / 2.0
    cy = (float(W) - 1.0) / 2.0
    radius = min(float(H), float(W)) / 2.0 - float(margin)
    if radius <= 0.0:
        raise ValueError(f"Invalid circular FOV radius: H={H}, W={W}, margin={margin}")
    return (cx, cy), radius


def compute_ray_circle_t_range_2d(ray_origins, ray_dirs, center_xy, radius):
    """
    원형 FOV와 ray의 교차 구간 [t_enter, t_exit]를 계산합니다.

    ray_origins:
        torch.Tensor, shape (R, 2), [x_H_axis, y_W_axis]
    ray_dirs:
        torch.Tensor, shape (R, 2), unit direction, [dx_H_axis, dy_W_axis]
    center_xy:
        tuple/list/tensor, (cx, cy), same voxel xy coordinate
    radius:
        float

    Returns
    -------
    t_enter:
        torch.Tensor, shape (R,)
    t_exit:
        torch.Tensor, shape (R,)
    valid:
        torch.Tensor, shape (R,), bool
    """
    c = torch.as_tensor(center_xy, device=ray_origins.device, dtype=ray_origins.dtype)
    r = torch.as_tensor(float(radius), device=ray_origins.device, dtype=ray_origins.dtype)

    # 방향 벡터를 다시 normalize해서 수치적으로 안전하게 둡니다.
    ray_dirs = F.normalize(ray_dirs, dim=1)

    oc = ray_origins - c[None, :]  # (R, 2)

    # p(t) = o + t d
    # ||p(t) - c||^2 = r^2
    # d가 unit이면 a = 1
    b = 2.0 * (ray_dirs * oc).sum(dim=1)
    c0 = (oc * oc).sum(dim=1) - r * r

    disc = b * b - 4.0 * c0
    valid = disc >= 0.0

    sqrt_disc = torch.zeros_like(disc)
    sqrt_disc[valid] = torch.sqrt(torch.clamp(disc[valid], min=0.0))

    t1 = (-b - sqrt_disc) / 2.0
    t2 = (-b + sqrt_disc) / 2.0

    t_enter = torch.minimum(t1, t2)
    t_exit = torch.maximum(t1, t2)

    valid = valid & (t_exit > t_enter)
    return t_enter, t_exit, valid


def render_pano_from_rotation_rays(
    vol,
    ray_origins,
    ray_dirs,
    batch_z=32,
):
    """
    rotation_geometry npz에서 불러온 ray_origins, ray_dirs를 이용해
    원형 FOV 내부 구간 [t_enter, t_exit]에서만 sampling하여 pano를 생성합니다.

    Inputs
    ------
    vol:
        torch.Tensor, shape (H, W, Z)
    ray_origins:
        torch.Tensor, shape (R, 2), voxel coordinate [x(H-axis), y(W-axis)]
    ray_dirs:
        torch.Tensor, shape (R, 2), unit direction in voxel coordinate
    batch_z:
        z slice batch size

    Output
    ------
    pano:
        numpy float32, shape (ZW, R)
    """
    H, W, Z = vol.shape

    R = ray_origins.shape[0]
    z_mid = Z // 2
    Z_HALF = Z // 2

    ray_origins = ray_origins.to(device=vol.device, dtype=torch.float32)
    ray_dirs = ray_dirs.to(device=vol.device, dtype=torch.float32)
    ray_dirs = F.normalize(ray_dirs, dim=1)

    # -----------------------------------------------------
    # circular FOV clipping
    #   각 ray마다 원형 FOV와 만나는 [t_enter, t_exit] 구간만 샘플링합니다.
    #   SLICE_DEPTH는 rotation-ray rendering에서는 더 이상 샘플링 구간을 정하지 않습니다.
    # -----------------------------------------------------
    fov_center, fov_radius = get_default_circular_fov(H, W, margin=1.0)

    t_enter, t_exit, valid_ray = compute_ray_circle_t_range_2d(
        ray_origins,
        ray_dirs,
        center_xy=fov_center,
        radius=fov_radius,
    )

    # -----------------------------------------------------
    # ray별 t-sampling
    # -----------------------------------------------------
    alpha = torch.linspace(
        0.0,
        1.0,
        NUM_SAMPLES,
        device=vol.device,
        dtype=torch.float32,
    )  # (S,)

    t = t_enter[:, None] + (t_exit - t_enter)[:, None] * alpha[None, :]  # (R,S)
    t = torch.where(valid_ray[:, None], t, torch.zeros_like(t))

    # 선적분용 ray별 step size
    ds_ray = ((t_exit - t_enter) / max(NUM_SAMPLES - 1, 1)).clamp_min(0.0)  # (R,)
    ds_ray = torch.where(valid_ray, ds_ray, torch.zeros_like(ds_ray))
    ds_t0 = ds_ray.view(1, R, 1)  # (1,R,1) -> (B,R,S-1) broadcast

    # -----------------------------------------------------
    # sample points on circular-FOV-clipped rays
    # -----------------------------------------------------
    pts = ray_origins[:, None, :] + ray_dirs[:, None, :] * t[:, :, None]
    x_vox = pts[..., 0]  # H-axis
    y_vox = pts[..., 1]  # W-axis

    # -----------------------------------------------------
    # grid_sample 좌표계 변환
    # -----------------------------------------------------
    grid_x = (2.0 * y_vox / (W - 1.0)) - 1.0
    grid_y = (2.0 * x_vox / (H - 1.0)) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).to(torch.float32)  # (R,S,2)

    # -----------------------------------------------------
    # z window
    # -----------------------------------------------------
    z_ids = torch.arange(
        z_mid - Z_HALF,
        z_mid + Z_HALF,
        device=vol.device,
    ).clamp(0, Z - 1).long()

    ZW = int(z_ids.numel())

    z_rel_all = torch.arange(
        -Z_HALF,
        Z_HALF,
        device=vol.device,
        dtype=torch.float32,
    )

    out = torch.empty((ZW, R), device=vol.device, dtype=torch.float32)

    with torch.no_grad():
        for z0 in range(0, ZW, batch_z):
            z1 = min(ZW, z0 + batch_z)
            B = z1 - z0

            z_batch = z_ids[z0:z1]

            # (H,W,B) -> (B,H,W) -> (B,1,H,W)
            vol_b = vol[:, :, z_batch].permute(2, 0, 1).contiguous().unsqueeze(1)

            # (B,R,S,2)
            grid_b = grid.unsqueeze(0).expand(B, -1, -1, -1)

            # (B,1,R,S) -> (B,R,S)
            sigma = F.grid_sample(
                vol_b,
                grid_b,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            ).squeeze(1)

            # 기존 inferior boost 그대로 유지
            z_rel = z_rel_all[z0:z1]
            inferior_boost = 1.0 + 0.35 * torch.sigmoid(
                (z_rel + 0.15 * Z_HALF) / (0.25 * Z_HALF)
            )

            # -------------------------------------------------
            # NeBLa-style Beer-Lambert pixel formula
            #
            # Paper:
            #   T = exp(- sum_i beta * sigma_i * delta_i)
            #   pixel = 1 - T
            #
            # Here, CBCT gray value sigma is converted to a nonnegative
            # scaled density using the current clipping window.
            # -------------------------------------------------
            sigma = torch.clamp(sigma, CLIP_LOW, CLIP_HIGH)

            sigma_density = torch.clamp(sigma - CLIP_LOW, min=0.0)
            sigma_density = sigma_density / (CLIP_HIGH - CLIP_LOW + 1e-6)

            tau = (sigma_density[:, :, :-1] * ds_t0).sum(dim=-1)
            tau = tau * valid_ray.to(tau.dtype)[None, :]

            # T = exp(- beta * integral), SimPX opacity = 1 - T
            line = 1.0 - torch.exp(-SIMPX_BETA * tau)

            # Keep the existing inferior boost unchanged.
            # Remove the multiplier below if strict Eq. (4)-only rendering is desired.
            out[z0:z1] = line * inferior_boost[:, None]

    pano = out.flip(0).detach().cpu().numpy()
    pano = gaussian_filter(pano, (0.5, 0.9))

    return pano.astype(np.float32)

# =========================================================
# Centerline from skeleton image
# =========================================================
def _bfs_farthest(adj, start):
    n = len(adj)
    dist = -np.ones(n, dtype=np.int32)
    parent = -np.ones(n, dtype=np.int32)
    q = [int(start)]
    dist[int(start)] = 0
    head = 0
    while head < len(q):
        u = q[head]
        head += 1
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                parent[v] = u
                q.append(v)
    far = int(np.argmax(dist))
    return far, parent, dist


def _reconstruct_path(parent, start, end):
    path = [int(end)]
    cur = int(end)
    while cur != int(start) and cur >= 0:
        cur = int(parent[cur])
        if cur < 0:
            break
        path.append(cur)
    path.reverse()
    return path


def _largest_component(pts, adj):
    n = len(adj)
    seen = np.zeros(n, dtype=bool)
    best = []
    for s in range(n):
        if seen[s]:
            continue
        stack = [s]
        comp = []
        seen[s] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        if len(comp) > len(best):
            best = comp
    if len(best) == n:
        return pts, adj

    keep = np.array(best, dtype=np.int32)
    pts2 = pts[keep]
    old_to_new = {int(old): i for i, old in enumerate(keep.tolist())}
    keep_set = set(keep.tolist())
    adj2 = [[] for _ in range(len(keep))]
    for old in keep.tolist():
        new = old_to_new[int(old)]
        adj2[new] = [old_to_new[int(v)] for v in adj[int(old)] if int(v) in keep_set]
    return pts2, adj2


def _trace(mask):
    """
    Extract an ordered polyline (x,y) from a 1px skeleton mask.

    Robust choice for branched skeletons:
      - build 8-neighborhood graph
      - keep only largest connected component
      - take graph diameter (longest shortest path) as main centerline
    """
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        raise ValueError("Empty skeleton mask (no foreground pixels).")

    # nodes in (x=col, y=row)
    pts = np.stack([xs.astype(np.int32), ys.astype(np.int32)], axis=1)
    idx = {(int(x), int(y)): i for i, (x, y) in enumerate(pts.tolist())}

    adj = [[] for _ in range(len(pts))]
    for i, (x, y) in enumerate(pts.tolist()):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                j = idx.get((x + dx, y + dy))
                if j is not None:
                    adj[i].append(j)

    # largest component (guard against speckles)
    pts, adj = _largest_component(pts, adj)

    if len(adj) == 1:
        return pts.astype(np.float32)

    # diameter path
    u, _, _ = _bfs_farthest(adj, 0)
    v, parent, _ = _bfs_farthest(adj, u)
    order = _reconstruct_path(parent, u, v)

    poly = pts[np.array(order, dtype=np.int32)].astype(np.float32)
    return poly




def _resample(pts, n):
    if pts.shape[0] < 2:
        return np.repeat(pts[:1], n, axis=0)

    d = np.diff(pts, axis=0)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(d, axis=1))])
    if float(s[-1]) < 1e-6:
        return np.repeat(pts[:1], n, axis=0)

    su = np.linspace(0.0, float(s[-1]), n, dtype=np.float32)
    x = np.interp(su, s, pts[:, 0]).astype(np.float32)
    y = np.interp(su, s, pts[:, 1]).astype(np.float32)
    return np.stack([x, y], axis=1)


def _arc_length_t(pts):
    if pts.shape[0] < 2:
        return np.array([0.0], dtype=np.float64)
    d = np.diff(pts.astype(np.float64), axis=0)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(d, axis=1))])
    if float(s[-1]) < 1e-8:
        return np.linspace(0.0, 1.0, pts.shape[0], dtype=np.float64)
    return (s / float(s[-1])).astype(np.float64)


def _make_clamped_knots(n_ctrl, degree):
    k = int(degree)
    n = int(n_ctrl)
    internal_count = n - k - 1
    if internal_count < 0:
        raise ValueError("n_ctrl must be >= degree+1")
    if internal_count == 0:
        internal = np.array([], dtype=np.float64)
    else:
        internal = np.linspace(0.0, 1.0, internal_count + 2, dtype=np.float64)[1:-1]
    return np.concatenate([np.zeros(k + 1), internal, np.ones(k + 1)]).astype(np.float64)


def _bspline_basis_matrix(t, knots, degree, n_ctrl):
    A = np.empty((t.size, n_ctrl), dtype=np.float64)
    for i in range(n_ctrl):
        c = np.zeros(n_ctrl, dtype=np.float64)
        c[i] = 1.0
        A[:, i] = BSpline(knots, c, degree, extrapolate=False)(t)
    A[np.isnan(A)] = 0.0
    return A


def _second_diff_matrix(n):
    if n < 3:
        return np.zeros((0, n), dtype=np.float64)
    D = np.zeros((n - 2, n), dtype=np.float64)
    for i in range(n - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    return D


def _fit_bspline_endpoints(pts, n_out, n_ctrl=54, degree=3, lam=1e-2):
    """
    Differentiable curve fit: constrained cubic B-spline with exact endpoint preservation.
    Input pts: (M,2) polyline in voxel coords (x,y), float32/float64 OK.
    Output: (n_out,2) fitted curve samples, float32.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape[0] < 2:
        return np.repeat(pts[:1].astype(np.float32), n_out, axis=0)

    # parameter t by arc-length
    t = _arc_length_t(pts)
    t = np.clip(t, 0.0, 1.0)

    # knots & basis
    n_ctrl = int(n_ctrl)
    degree = int(degree)
    if n_ctrl < degree + 1:
        n_ctrl = degree + 1

    knots = _make_clamped_knots(n_ctrl, degree)
    A = _bspline_basis_matrix(t, knots, degree, n_ctrl)

    # smoothing penalty
    D2 = _second_diff_matrix(n_ctrl)

    # fix endpoints exactly (clamped spline => c0, c_{n-1})
    x0, y0 = float(pts[0, 0]), float(pts[0, 1])
    x1, y1 = float(pts[-1, 0]), float(pts[-1, 1])

    # free coeffs: 1..n_ctrl-2
    if n_ctrl <= 2:
        out = np.linspace(0.0, 1.0, n_out, dtype=np.float64)[:, None]
        return ((1.0 - out) * pts[0:1] + out * pts[-1:]).astype(np.float32)

    A_f = A[:, 1:-1]
    A_c = A[:, [0, -1]]

    c_fix_x = np.array([x0, x1], dtype=np.float64)
    c_fix_y = np.array([y0, y1], dtype=np.float64)

    # split D2
    D_f = D2[:, 1:-1] if D2.size else D2
    D_c = D2[:, [0, -1]] if D2.size else D2

    # normal matrix (shared by x,y)
    M = A_f.T @ A_f
    if D2.size and lam > 0:
        M = M + lam * (D_f.T @ D_f)

    # rhs for x
    bx = pts[:, 0] - A_c @ c_fix_x
    rhs_x = A_f.T @ bx
    if D2.size and lam > 0:
        rhs_x = rhs_x - lam * (D_f.T @ (D_c @ c_fix_x))

    # rhs for y
    by = pts[:, 1] - A_c @ c_fix_y
    rhs_y = A_f.T @ by
    if D2.size and lam > 0:
        rhs_y = rhs_y - lam * (D_f.T @ (D_c @ c_fix_y))

    # solve (add tiny ridge only if needed)
    try:
        c_free_x = np.linalg.solve(M, rhs_x)
        c_free_y = np.linalg.solve(M, rhs_y)
    except np.linalg.LinAlgError:
        eps = 1e-6
        M2 = M + eps * np.eye(M.shape[0], dtype=np.float64)
        c_free_x = np.linalg.solve(M2, rhs_x)
        c_free_y = np.linalg.solve(M2, rhs_y)

    cx = np.concatenate([[x0], c_free_x, [x1]]).astype(np.float64)
    cy = np.concatenate([[y0], c_free_y, [y1]]).astype(np.float64)

    # sample fitted curve
    t_out = np.linspace(0.0, 1.0, int(n_out), dtype=np.float64)
    sx = BSpline(knots, cx, degree, extrapolate=False)(t_out)
    sy = BSpline(knots, cy, degree, extrapolate=False)(t_out)
    out = np.stack([sx, sy], axis=1).astype(np.float32)
    return out




def load_centerline(path, H, W):
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.uint8)
    arr = arr.T
    mask = arr > 0
    pts = _trace(mask)  # (M,2) in image pixel coords (x=col, y=row)

    ih, iw = arr.shape
    sx = (H - 1.0) / (iw - 1.0) if iw > 1 else 1.0
    sy = (W - 1.0) / (ih - 1.0) if ih > 1 else 1.0

    pts[:, 0] *= sx
    pts[:, 1] *= sy
    
    pts = _fit_bspline_endpoints(pts, NUM_X, n_ctrl=54, degree=3, lam=1e-2).astype(np.float32)  # (NUM_X,2) fitted in voxel coords (x(H), y(W))
    return torch.from_numpy(pts).to(device=device, dtype=torch.float64)

# =========================================================
# Geometry
# =========================================================
# =========================================================
def compute_normals(centerline):
    tangent = -F.normalize(torch.gradient(centerline, dim=0)[0], dim=1)
    return F.normalize(
        torch.stack([
            -tangent[:, 1],
             tangent[:, 0],
             torch.zeros_like(tangent[:, 0])
        ], 1),
        dim=1
    )

# =========================================================
# Panoramic projection (grid_sample version: same style as make_simpx_volume_torch)
# =========================================================
def render_pano(vol, centerline, normals, batch_z=32):
    """
    Option A:
      - grid_sample로 법선 방향 샘플링 (make_simpx_volume_torch와 동일 스타일)
      - agg_mode="integral": stat = sum(mu[:-1] * ds)
      - exp(Beer–Lambert) 포화는 쓰지 않고, line = beta * stat 을 그대로 출력
      - 이후 tone_map에서 대비/윈도잉을 잡도록 설계

    Inputs
      vol:        (H, W, Z) torch.float32 on CUDA
      centerline: (N, 2)    [x(H-axis), y(W-axis)] float32 CUDA
      normals:    (N, 3)    float32 CUDA
      z_mid:      int
      batch_z:    int

    Output
      pano_raw: numpy float32 (2*Z_HALF, N)  (gaussian smoothing 적용)
    """
    H, W, Z = vol.shape
    N = centerline.shape[0]
    z_mid = Z // 2
    Z_HALF = Z // 2
    # -----------------------------------------------------
    # posterior-only normal adjustment (기존 유지)
    # -----------------------------------------------------
    u = torch.linspace(0, 1, N, device=device)
    ramus_mask = torch.abs(u - 0.5) > 0.30

    normals_adj = normals.clone()
    normals_adj[ramus_mask] = F.normalize(
        0.85 * normals[ramus_mask] +
        0.15 * torch.stack([
            torch.zeros_like(normals[ramus_mask, 0]),
            torch.sign(normals[ramus_mask, 1]),
            torch.zeros_like(normals[ramus_mask, 0])
        ], dim=1),
        dim=1
    )

    # -----------------------------------------------------
    # t-sampling (법선 방향 좌우) + ds (integral용)
    # -----------------------------------------------------
    t = torch.linspace(
        -SLICE_DEPTH / 2, SLICE_DEPTH / 2,
        NUM_SAMPLES,
        device=device,
        dtype=torch.float32
    )  # (S,)

    # integral aggregation용 ds (SimPX의 ds_t0 역할)
    dt = (t[1] - t[0]).abs()                 # scalar
    ds = dt.expand(NUM_SAMPLES - 1)          # (S-1,)
    ds_t0 = ds.view(1, 1, -1)                # (1,1,S-1) -> (B,N,S-1) broadcast

    # (N,S,2) sample points in voxel coord
    pts = centerline[:, None, :] + normals_adj[:, None, :2] * t[None, :, None]
    x_vox = pts[..., 0]  # H축
    y_vox = pts[..., 1]  # W축

    # -----------------------------------------------------
    # grid_sample 좌표계 변환
    #   vol[x,y,z]에서 slice는 (H,W)
    #   grid_sample은 (N,S,2)에서 [...,0]=x(width=W축), [...,1]=y(height=H축)
    #   => grid[...,0] = y_vox를 W로 normalize
    #      grid[...,1] = x_vox를 H로 normalize
    # -----------------------------------------------------
    grid_x = (2.0 * y_vox / (W - 1.0)) - 1.0
    grid_y = (2.0 * x_vox / (H - 1.0)) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).to(torch.float32)  # (N,S,2)

    # -----------------------------------------------------
    # z window
    # -----------------------------------------------------
    z_ids = torch.arange(z_mid - Z_HALF, z_mid + Z_HALF, device=device).clamp(0, Z - 1).long()
    ZW = int(z_ids.numel())

    # inferior boost용 z_rel (기존과 동일)
    z_rel_all = torch.arange(-Z_HALF, Z_HALF, device=device, dtype=torch.float32)  # (ZW,)

    out = torch.empty((ZW, N), device=device, dtype=torch.float32)

    with torch.no_grad():
        for z0 in range(0, ZW, batch_z):
            z1 = min(ZW, z0 + batch_z)
            B = z1 - z0

            z_batch = z_ids[z0:z1]  # (B,)

            # (H,W,B) -> (B,H,W) -> (B,1,H,W)
            vol_b = vol[:, :, z_batch].permute(2, 0, 1).contiguous().unsqueeze(1)

            # (B,N,S,2)
            grid_b = grid.unsqueeze(0).expand(B, -1, -1, -1)

            # (B,1,N,S) -> (B,N,S)
            sigma = F.grid_sample(
                vol_b, grid_b,
                mode="bilinear",
                padding_mode="border",
                align_corners=True
            ).squeeze(1)  # (B,N,S)

            # (항상 먼저) inferior_boost 정의
            z_rel = z_rel_all[z0:z1]  # (B,)
            inferior_boost = 1.0 + 0.35 * torch.sigmoid(
                (z_rel + 0.15 * Z_HALF) / (0.25 * Z_HALF)
            )  # (B,)

            # -------------------------------------------------
            # Option A pixel formula (integral + no exp saturation)
            # -------------------------------------------------
            sigma = torch.clamp(sigma, CLIP_LOW, CLIP_HIGH)     # (B,N,S)
            mu = torch.relu(sigma - BONE_THR)                   # (B,N,S)

            # agg_mode="integral" (SimPX 형태)
            stat = (mu[:, :, :-1] * ds_t0).sum(dim=-1)          # (B,N)

            # exp 포화 제거: 선적분 결과를 beta로만 스케일
            line = SIMPX_BETA * stat                            # (B,N)

            out[z0:z1] = line * inferior_boost[:, None]


    pano = out.flip(0).detach().cpu().numpy()
    pano = gaussian_filter(pano, (0.5, 0.9))
    return pano


# =========================================================
# Save PNG / Tone mapping
# =========================================================

def propose_beta_from_pano(pano_raw, beta_now, target_clip=0.08, q=99.5):
    """
    pano_raw는 현재 beta_now로 계산된 float 결과.
    target_clip(고정 윈도우 상한)을 기준으로 q-percentile이 target_clip 근처로 오도록
    beta를 제안한다.
    """
    qv = float(np.percentile(pano_raw, q))
    if qv <= 1e-12:
        return beta_now, qv
    beta_new = beta_now * (target_clip / qv)
    return float(beta_new), qv


def pano_raw_to_uint8_rgba(pano_raw, clip_max=0.08, invert=False):
    """
    pano_raw(float)를 고정 window [0, clip_max]로 자르고 0~255 uint8로 변환.
    train_0.png가 RGBA(alpha=255)이므로 동일하게 RGBA로 맞춰 저장한다.
    """
    x = np.asarray(pano_raw, dtype=np.float32)

    # fixed window
    x = np.clip(x, 0.0, float(clip_max))
    x = x / (float(clip_max) + 1e-6)  # 0..1

    if invert:
        x = 1.0 - x

    x_u8 = (x * 255.0 + 0.5).astype(np.uint8)  # 0..255

    # grayscale -> RGBA(alpha=255)
    rgba = np.stack([x_u8, x_u8, x_u8, np.full_like(x_u8, 255)], axis=-1)
    return rgba

# =========================================================
# Debug visualization (기존 유지)
# =========================================================
def save_centerline_debug(vol, centerline, normals, z_mid, path):

    mip = vol[:, :, max(0, z_mid-20):min(vol.shape[2], z_mid+20)].max(dim=2).values.detach().cpu().numpy()
    cl = centerline.detach().cpu().numpy()
    nrm = normals.detach().cpu().numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(mip.T, cmap="gray", origin="lower")
    plt.plot(cl[:, 0], cl[:, 1], 'r-', linewidth=1.5)

    for i in range(0, len(cl), 32):
        x, y = cl[i]
        nx, ny = nrm[i, 0], nrm[i, 1]
        plt.arrow(x, y, nx * 20, ny * 20,
                  color="lime", width=0.5, head_width=3)

    plt.axis("off")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

# =========================================================
# Main
# =========================================================
def main():
    global SIMPX_BETA
    subject_code = 1
    vol_np = nib.load(f"MMDental/{subject_code}/{subject_code}.nii.gz").get_fdata(dtype=np.float32)
    vol = torch.from_numpy(vol_np).to(device=device, dtype=torch.float32)

    z_mid = vol.shape[2] // 2

    sk = None
    for ext in ("png", "jpg", "jpeg"):
        p = f"extension_result/extended_{subject_code}.{ext}"
        if os.path.isfile(p):
            sk = p
            break
    if sk is None:
        raise FileNotFoundError(f"skeleton image not found: skeleton_{subject_code}.(png|jpg|jpeg)")
    centerline = load_centerline(sk, vol.shape[0], vol.shape[1])

    normals = compute_normals(centerline)
    save_centerline_debug(
        vol, centerline, normals, z_mid,
        f"{OUTDIR}/debug_centerline_normals.png"
    )

    # 1) 1차 pano_raw
    pano_raw_1 = render_pano(vol, centerline, normals, batch_z=32)

    # 2) beta 제안 (고정 clip_max에 q-percentile 맞추기)
    CLIP_MAX = 0.08  # 네가 원하는 “고정 윈도우 상한” (데이터셋 전체에서 고정 추천)
    beta_new, qv = propose_beta_from_pano(pano_raw_1, SIMPX_BETA, target_clip=CLIP_MAX, q=99.5)
    print(f"[beta suggest] current={SIMPX_BETA:.3e}, p99.5={qv:.6g}, suggested={beta_new:.3e} (clip_max={CLIP_MAX})")

    # 3) 제안 beta를 실제로 적용하려면 전역 값을 갱신하고 한 번 더 계산
    SIMPX_BETA = beta_new
    pano_raw = render_pano(vol, centerline, normals, batch_z=32)

    # 4) train_0.png과 동일 dtype/range(uint8 0..255, RGBA)로 변환 후 PIL 저장
    pano_rgba = pano_raw_to_uint8_rgba(pano_raw, clip_max=CLIP_MAX, invert=False)
    Image.fromarray(pano_rgba, mode="RGBA").save(f"{OUTDIR}/pano_final.png")

    # (선택) raw float도 보관 (나중에 재가공/검증용)
    np.save(f"{OUTDIR}/pano_raw_float32.npy", pano_raw.astype(np.float32))

    print("DONE (saved uint8 RGBA 0..255, beta is meaningful under fixed window)")

if __name__ == "__main__":
    main()
