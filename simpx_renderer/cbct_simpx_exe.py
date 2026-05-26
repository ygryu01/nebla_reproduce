# cbct_simpx_exe_rotation.py
import os
import re
import argparse
import traceback
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


def list_subject_codes(nii_root: str):
    nii_root_p = Path(nii_root)
    if not nii_root_p.is_dir():
        raise FileNotFoundError(f"Not found directory: {nii_root}")

    codes = []
    pattern = re.compile(r"^(.+)\.nii\.gz$", re.IGNORECASE)
    for p in nii_root_p.glob("*.nii.gz"):
        m = pattern.match(p.name)
        if m is None:
            continue
        code_str = m.group(1)
        if code_str.isdigit():
            codes.append(int(code_str))
    codes.sort()
    return codes


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def resize_pano_raw_np(pano_raw: np.ndarray, target_h: int = 200, target_w: int = 350) -> np.ndarray:
    arr = np.asarray(pano_raw, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"pano_raw must be 2D, but got shape={arr.shape}")
    if arr.shape == (target_h, target_w):
        return arr.astype(np.float32, copy=False)

    with torch.no_grad():
        x = torch.from_numpy(arr)[None, None].float()
        y = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return y[0, 0].cpu().numpy().astype(np.float32)


def tensor_to_numpy_xy(x) -> np.ndarray:
    """
    torch.Tensor 또는 numpy array를 CPU numpy float32 array로 변환합니다.
    좌표 convention은 load_rotation_geometry_for_volume()이 반환하는 voxel xy입니다.
      - x[:, 0] = volume H-axis coordinate
      - x[:, 1] = volume W-axis coordinate
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def save_centers_ray_origins_txt(
    centers,
    ray_origins,
    save_path: Path,
):
    """
    centers와 ray_origins를 같은 subject 출력 폴더에 txt로 저장합니다.
    ray_origins는 ray별 origin이므로 centers를 ray 개수만큼 펼친 배열입니다.
    """
    centers_np = tensor_to_numpy_xy(centers)
    ray_origins_np = tensor_to_numpy_xy(ray_origins)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    nearest_ids = None
    nearest_dists = None
    if centers_np.ndim == 2 and ray_origins_np.ndim == 2 and centers_np.shape[1] == 2 and ray_origins_np.shape[1] == 2:
        diff = ray_origins_np[:, None, :] - centers_np[None, :, :]
        dists = np.linalg.norm(diff, axis=2)
        nearest_ids = np.argmin(dists, axis=1)
        nearest_dists = dists[np.arange(ray_origins_np.shape[0]), nearest_ids]

    with save_path.open("w", encoding="utf-8") as f:
        f.write("# coordinate_system: voxel_xy_after_load_rotation_geometry_for_volume\n")
        f.write("# x = volume H-axis coordinate, y = volume W-axis coordinate\n")
        f.write(f"centers_shape: {tuple(centers_np.shape)}\n")
        f.write(f"ray_origins_shape: {tuple(ray_origins_np.shape)}\n\n")

        f.write("[centers]\n")
        f.write("idx\tx_H_axis\ty_W_axis\n")
        for i, (x, y) in enumerate(centers_np):
            f.write(f"{i}\t{x:.6f}\t{y:.6f}\n")

        f.write("\n[ray_origins]\n")
        if nearest_ids is None:
            f.write("idx\tx_H_axis\ty_W_axis\n")
            for i, (x, y) in enumerate(ray_origins_np):
                f.write(f"{i}\t{x:.6f}\t{y:.6f}\n")
        else:
            f.write("idx\tx_H_axis\ty_W_axis\tnearest_center_id\tnearest_center_dist\n")
            for i, (x, y) in enumerate(ray_origins_np):
                f.write(
                    f"{i}\t{x:.6f}\t{y:.6f}\t"
                    f"{int(nearest_ids[i])}\t{float(nearest_dists[i]):.8f}\n"
                )


def save_axial_mip_with_centers(
    vol_np: np.ndarray,
    centers,
    save_path: Path,
):
    """
    CBCT axial MIP 위에 centers를 overlay해서 저장합니다.
    vol_np shape은 (H, W, Z)이고 centers는 voxel xy 좌표입니다.
    PIL drawing 좌표는 (col, row)이므로 (y, x)로 찍습니다.
    """
    vol_np = np.asarray(vol_np, dtype=np.float32)
    if vol_np.ndim != 3:
        raise ValueError(f"vol_np must be 3D, but got shape={vol_np.shape}")

    centers_np = tensor_to_numpy_xy(centers)

    mip = np.max(vol_np, axis=2)  # (H, W)
    finite = mip[np.isfinite(mip)]
    if finite.size == 0:
        mip_u8 = np.zeros(mip.shape, dtype=np.uint8)
    else:
        lo, hi = np.percentile(finite, [1.0, 99.5])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.min(finite))
            hi = float(np.max(finite))
        if hi <= lo:
            mip_u8 = np.zeros(mip.shape, dtype=np.uint8)
        else:
            x = np.clip((mip - lo) / (hi - lo), 0.0, 1.0)
            mip_u8 = (x * 255.0 + 0.5).astype(np.uint8)

    rgb = np.stack([mip_u8, mip_u8, mip_u8], axis=-1)
    img = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(img)

    H, W = mip.shape

    # centerline 연결선
    line_pts = []
    for x, y in centers_np:
        row = int(round(float(x)))
        col = int(round(float(y)))
        if 0 <= row < H and 0 <= col < W:
            line_pts.append((col, row))
    if len(line_pts) >= 2:
        draw.line(line_pts, fill=(255, 255, 0), width=2)

    # centers 표시
    for i, (x, y) in enumerate(centers_np):
        row = int(round(float(x)))
        col = int(round(float(y)))
        if not (0 <= row < H and 0 <= col < W):
            continue

        if i == 0:
            color = (255, 0, 0)
        elif i == 10:
            color = (0, 255, 0)
        elif i == 20:
            color = (0, 128, 255)
        else:
            color = (255, 255, 0)

        radius = 5 if i in (0, 10, 20) else 3
        draw.ellipse(
            (col - radius, row - radius, col + radius, row + radius),
            fill=color,
            outline=(0, 0, 0),
        )
        draw.text((col + radius + 2, row - radius - 2), f"C{i}", fill=color)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(save_path))



def select_representative_ray_indices(geom_path: Path, num_rays: int, target_center_ids):
    """
    디버그 표시용 대표 ray 여러 개를 고릅니다.
    각 target center id에 대해 우선순위:
      1) 해당 center의 final ray
      2) 해당 center의 ray들 중 중앙 인덱스
      3) 전체 ray의 중앙 인덱스
    """
    target_center_ids = [int(x) for x in target_center_ids]
    fallback = int(num_rays // 2)
    selected = []

    try:
        data = np.load(str(geom_path), allow_pickle=False)
        center_ids = np.asarray(data["ray_center_ids"])
        is_final = np.asarray(data["ray_is_final"]).astype(bool)

        for cid in target_center_ids:
            cand = np.where((center_ids == cid) & is_final)[0]
            if cand.size > 0:
                selected.append(int(cand[0]))
                continue

            cand = np.where(center_ids == cid)[0]
            if cand.size > 0:
                selected.append(int(cand[len(cand) // 2]))
                continue

            selected.append(fallback)

    except Exception:
        selected = [fallback for _ in target_center_ids]

    return selected


def save_axial_mip_with_sampled_ray_depth(
    vol_np: np.ndarray,
    ray_origin,
    ray_dir,
    save_path: Path,
    sample_depth_vox: float = None,
    centers=None,
    ray_index=None,
    center_id=None,
    t_enter=None,
    t_exit=None,
):
    """
    CBCT axial MIP 위에 하나의 ray와 실제 sampling 구간을 표시합니다.

    t_enter/t_exit가 주어지면 원형 FOV intersection으로 계산된
    실제 sampling 구간 [t_enter, t_exit]만 빨간색으로 표시합니다.

    좌표계:
      - ray_origin = (x_H_axis, y_W_axis)
      - ray_dir    = (dx_H_axis, dy_W_axis)
      - PIL drawing 좌표 = (col, row) = (y_W_axis, x_H_axis)
    """
    vol_np = np.asarray(vol_np, dtype=np.float32)
    if vol_np.ndim != 3:
        raise ValueError(f"vol_np must be 3D, but got shape={vol_np.shape}")

    ray_origin_np = tensor_to_numpy_xy(ray_origin).reshape(2)
    ray_dir_np = tensor_to_numpy_xy(ray_dir).reshape(2)

    norm = float(np.linalg.norm(ray_dir_np))
    if norm < 1e-12:
        raise ValueError("ray_dir norm is too small")
    ray_dir_np = ray_dir_np / norm

    mip = np.max(vol_np, axis=2)  # (H, W)
    finite = mip[np.isfinite(mip)]
    if finite.size == 0:
        mip_u8 = np.zeros(mip.shape, dtype=np.uint8)
    else:
        lo, hi = np.percentile(finite, [1.0, 99.5])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.min(finite))
            hi = float(np.max(finite))
        if hi <= lo:
            mip_u8 = np.zeros(mip.shape, dtype=np.uint8)
        else:
            x = np.clip((mip - lo) / (hi - lo), 0.0, 1.0)
            mip_u8 = (x * 255.0 + 0.5).astype(np.uint8)

    rgb = np.stack([mip_u8, mip_u8, mip_u8], axis=-1)
    img = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(img)

    H, W = mip.shape

    # reference용 centers 표시
    if centers is not None:
        centers_np = tensor_to_numpy_xy(centers)
        for i, (x, y) in enumerate(centers_np):
            row = int(round(float(x)))
            col = int(round(float(y)))
            if 0 <= row < H and 0 <= col < W:
                radius = 3 if i in (6, 14) else 2
                color = (255, 255, 0) if i not in (6, 14) else (0, 255, 0)
                draw.ellipse(
                    (col - radius, row - radius, col + radius, row + radius),
                    fill=color,
                    outline=(0, 0, 0),
                )
                if i in (6, 14):
                    draw.text((col + radius + 2, row - radius - 2), f"C{i}", fill=color)

    if t_enter is not None and t_exit is not None:
        t0 = float(t_enter)
        t1 = float(t_exit)
        sample_len = max(0.0, t1 - t0)
        p0 = ray_origin_np + ray_dir_np * t0
        p1 = ray_origin_np + ray_dir_np * t1
    else:
        if sample_depth_vox is None:
            raise ValueError("Either (t_enter, t_exit) or sample_depth_vox must be provided.")
        half = float(sample_depth_vox) / 2.0
        t0 = -half
        t1 = half
        sample_len = float(sample_depth_vox)
        p0 = ray_origin_np - ray_dir_np * half
        p1 = ray_origin_np + ray_dir_np * half

    def clamp_point(pt):
        x = float(np.clip(pt[0], 0.0, H - 1.0))
        y = float(np.clip(pt[1], 0.0, W - 1.0))
        return x, y

    p0 = clamp_point(p0)
    p1 = clamp_point(p1)
    origin = clamp_point(ray_origin_np)

    p0_draw = (int(round(p0[1])), int(round(p0[0])))
    p1_draw = (int(round(p1[1])), int(round(p1[0])))
    origin_draw = (int(round(origin[1])), int(round(origin[0])))

    # sampled thickness segment
    draw.line([p0_draw, p1_draw], fill=(255, 0, 0), width=3)

    for pt_draw, color, radius in [
        (p0_draw, (0, 255, 255), 4),
        (p1_draw, (0, 255, 255), 4),
        (origin_draw, (0, 255, 0), 5),
    ]:
        col, row = pt_draw
        draw.ellipse(
            (col - radius, row - radius, col + radius, row + radius),
            fill=color,
            outline=(0, 0, 0),
        )

    label_parts = []
    if center_id is not None:
        label_parts.append(f"center C{int(center_id)}")
    if ray_index is not None:
        label_parts.append(f"ray {int(ray_index)}")
    label_parts.append(f"sample length = {sample_len:.1f} vox")
    label_parts.append(f"t=[{t0:.1f},{t1:.1f}]")
    label = ", ".join(label_parts)

    draw.text((origin_draw[0] + 8, origin_draw[1] - 14), label, fill=(255, 0, 0))
    draw.text((p0_draw[0] + 6, p0_draw[1] + 4), "start", fill=(0, 255, 255))
    draw.text((p1_draw[0] + 6, p1_draw[1] + 4), "end", fill=(0, 255, 255))
    draw.text((origin_draw[0] + 6, origin_draw[1] + 12), "origin", fill=(0, 255, 0))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(save_path))


def process_one(
    simpx,
    subject_code: int,
    nii_root: str,
    geometry_root: str,
    out_root: str,
    clip_max: float,
    q: float,
    batch_z: int,
    invert: bool,
    pano_h: int,
    pano_w: int,
    save_raw: bool,
):
    nii_path = Path(nii_root) / f"{subject_code}.nii.gz"
    if not nii_path.is_file():
        raise FileNotFoundError(f"NII not found: {nii_path}")

    geom_path = Path(geometry_root) / str(subject_code) / f"rotation_geometry_{subject_code}.npz"
    if not geom_path.is_file():
        raise FileNotFoundError(f"Rotation geometry not found: {geom_path}")

    out_dir = Path(out_root) / str(subject_code)
    ensure_dir(out_dir)

    vol_np = nib.load(str(nii_path)).get_fdata(dtype=np.float32)  # (H, W, Z)
    vol = torch.from_numpy(vol_np).to(device=simpx.device, dtype=torch.float32)

    # 핵심: shifted_curve를 trace하지 않고, 저장된 rotation_geometry npz를 직접 사용합니다.
    ray_origins, ray_dirs, centers, geom_meta = simpx.load_rotation_geometry_for_volume(
        str(geom_path),
        H=vol.shape[0],
        W=vol.shape[1],
        device=simpx.device,
    )

    # 디버깅 파일 저장
    save_centers_ray_origins_txt(
        centers=centers,
        ray_origins=ray_origins,
        save_path=out_dir / "debug_centers_ray_origins.txt",
    )
    save_axial_mip_with_centers(
        vol_np=vol_np,
        centers=centers,
        save_path=out_dir / "debug_axial_mip_centers.png",
    )

    # C6, C14 기준 representative ray의 sampling depth를 각각 저장합니다.
    # 파일명에는 center id를 넣지 않습니다.
    rep_ray_indices = select_representative_ray_indices(
        geom_path=geom_path,
        num_rays=int(ray_origins.shape[0]),
        target_center_ids=[6,14],
    )
    debug_depth_filenames = [
        "debug_axial_mip_sample_depth_1.png",
        "debug_axial_mip_sample_depth_2.png",
    ]
    debug_center_ids = [6, 14]

    # debug_axial_mip_sample_depth_{1,2}.png의 빨간 선도
    # 실제 rendering과 동일하게 원형 FOV intersection [t_enter, t_exit]를 사용합니다.
    fov_center, fov_radius = simpx.get_default_circular_fov(
        H=vol.shape[0],
        W=vol.shape[1],
        margin=1.0,
    )

    rep_ray_idx_tensor = torch.as_tensor(rep_ray_indices, device=simpx.device, dtype=torch.long)
    dbg_ray_origins = ray_origins.index_select(0, rep_ray_idx_tensor).to(device=simpx.device, dtype=torch.float32)
    dbg_ray_dirs = ray_dirs.index_select(0, rep_ray_idx_tensor).to(device=simpx.device, dtype=torch.float32)

    dbg_t_enter, dbg_t_exit, dbg_valid = simpx.compute_ray_circle_t_range_2d(
        dbg_ray_origins,
        dbg_ray_dirs,
        center_xy=fov_center,
        radius=fov_radius,
    )

    dbg_t_enter_np = dbg_t_enter.detach().cpu().numpy()
    dbg_t_exit_np = dbg_t_exit.detach().cpu().numpy()
    dbg_valid_np = dbg_valid.detach().cpu().numpy()

    for k, (rep_ray_idx, dbg_name, dbg_cid) in enumerate(zip(rep_ray_indices, debug_depth_filenames, debug_center_ids)):
        if not bool(dbg_valid_np[k]):
            raise RuntimeError(
                f"Representative debug ray does not intersect circular FOV: "
                f"subject={subject_code}, center_id={dbg_cid}, ray_index={rep_ray_idx}"
            )

        save_axial_mip_with_sampled_ray_depth(
            vol_np=vol_np,
            ray_origin=ray_origins[rep_ray_idx],
            ray_dir=ray_dirs[rep_ray_idx],
            centers=centers,
            save_path=out_dir / dbg_name,
            sample_depth_vox=None,
            ray_index=rep_ray_idx,
            center_id=dbg_cid,
            t_enter=float(dbg_t_enter_np[k]),
            t_exit=float(dbg_t_exit_np[k]),
        )

    base_beta = float(simpx.SIMPX_BETA)

    # 1st pass: rotation-ray SimPX
    pano_raw_1 = simpx.render_pano_from_rotation_rays(
        vol,
        ray_origins,
        ray_dirs,
        batch_z=batch_z,
    )
    raw_shape_1 = tuple(pano_raw_1.shape)

    pano_raw_1_resized = resize_pano_raw_np(pano_raw_1, target_h=pano_h, target_w=pano_w)

    beta_new, qv = simpx.propose_beta_from_pano(
        pano_raw_1_resized,
        base_beta,
        target_clip=float(clip_max),
        q=float(q),
    )

    # 2nd pass with suggested beta
    simpx.SIMPX_BETA = float(beta_new)
    pano_raw = simpx.render_pano_from_rotation_rays(
        vol,
        ray_origins,
        ray_dirs,
        batch_z=batch_z,
    )
    raw_shape_2 = tuple(pano_raw.shape)

    pano_raw = resize_pano_raw_np(pano_raw, target_h=pano_h, target_w=pano_w)

    pano_rgba = simpx.pano_raw_to_uint8_rgba(
        pano_raw,
        clip_max=float(clip_max),
        invert=bool(invert),
    )

    out_png = out_dir / "pano_final.png"
    img = Image.fromarray(pano_rgba, mode="RGBA")
    img.save(str(out_png))

    saved_size = img.size
    if saved_size != (pano_w, pano_h):
        raise RuntimeError(
            f"Saved pano size mismatch: got PIL size={saved_size}, expected={(pano_w, pano_h)}"
        )

    if save_raw:
        np.save(str(out_dir / "pano_raw_float32.npy"), pano_raw.astype(np.float32))

    meta = (
        f"subject_code: {subject_code}\n"
        f"nii: {nii_path}\n"
        f"rotation_geometry: {geom_path}\n"
        f"volume_shape_H_W_Z: {tuple(vol.shape)}\n"
        f"rotation_centers_shape: {tuple(centers.shape)}\n"
        f"ray_origins_shape: {tuple(ray_origins.shape)}\n"
        f"ray_dirs_shape: {tuple(ray_dirs.shape)}\n"
        f"raw_pano_shape_pass1_before_resize_H_W: {raw_shape_1}\n"
        f"raw_pano_shape_pass2_before_resize_H_W: {raw_shape_2}\n"
        f"final_pano_shape_H_W: {tuple(pano_raw.shape)}\n"
        f"final_png_size_W_H: {saved_size}\n"
        f"clip_max: {clip_max}\n"
        f"q_percentile: {q}\n"
        f"beta_base: {base_beta:.8e}\n"
        f"p{q}: {qv:.8g}\n"
        f"beta_used: {beta_new:.8e}\n"
        f"batch_z: {batch_z}\n"
        f"invert: {invert}\n"
    )
    (out_dir / "meta.txt").write_text(meta, encoding="utf-8")

    simpx.SIMPX_BETA = float(base_beta)
    del vol
    torch.cuda.empty_cache()

    print(
        f"[OK] subject={subject_code} | rays={ray_origins.shape[0]} | "
        f"raw_before_resize={raw_shape_2} -> final_HxW={tuple(pano_raw.shape)} | {out_png}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nii_root", default="/home/alphayoung8/log1/oral3d_reproduce/data/cbct_nii")
    ap.add_argument("--geometry_root", default="/home/alphayoung8/log1/nebla/center_geometries")
    ap.add_argument("--out_dir", default="/home/alphayoung8/log1/nebla/simpx_result_200x350")
    ap.add_argument("--cuda", default="7")
    ap.add_argument("--clip_max", type=float, default=0.08)
    ap.add_argument("--q", type=float, default=99.5)
    ap.add_argument("--batch_z", type=int, default=32)
    ap.add_argument("--invert", action="store_true")
    ap.add_argument("--pano_h", type=int, default=200)
    ap.add_argument("--pano_w", type=int, default=350)
    ap.add_argument("--save_raw", action="store_true")
    ap.add_argument("--only", type=str, default=None, help='e.g. "26" or "26,27,28"')
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    args = ap.parse_args()

    if args.cuda is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    import cbct_simpx2_nebla as simpx
    # import cbct_simpx2_nebla2 as simpx

    out_root = Path(args.out_dir)
    ensure_dir(out_root)

    if args.only:
        codes = [int(x.strip()) for x in args.only.split(",") if x.strip()]
        codes.sort()
    else:
        codes = list_subject_codes(args.nii_root)

    if args.start is not None:
        codes = [c for c in codes if c >= args.start]
    if args.end is not None:
        codes = [c for c in codes if c <= args.end]

    log_path = out_root / "run_log.txt"
    ok, fail = 0, 0

    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"nii_root={args.nii_root}\n")
        f.write(f"geometry_root={args.geometry_root}\n")
        f.write(f"out_dir={args.out_dir}\n")
        f.write(f"target_pano_H_W=({args.pano_h}, {args.pano_w})\n")
        f.write(f"num_subjects={len(codes)}\n\n")

        for i, code in enumerate(codes, 1):
            try:
                print(f"[{i}/{len(codes)}] subject={code}")
                process_one(
                    simpx=simpx,
                    subject_code=code,
                    nii_root=args.nii_root,
                    geometry_root=args.geometry_root,
                    out_root=args.out_dir,
                    clip_max=args.clip_max,
                    q=args.q,
                    batch_z=args.batch_z,
                    invert=args.invert,
                    pano_h=args.pano_h,
                    pano_w=args.pano_w,
                    save_raw=args.save_raw,
                )
                ok += 1
                f.write(f"OK  {code}\n")
                f.flush()
            except Exception as e:
                fail += 1
                print(f"[FAIL] subject={code}: {e}")
                f.write(f"FAIL {code}: {e}\n")
                f.write(traceback.format_exc() + "\n")
                f.flush()

    print(f"DONE. ok={ok}, fail={fail}. log={log_path}")


if __name__ == "__main__":
    main()
