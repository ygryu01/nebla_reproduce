import os
import cv2
import math
import json
import numpy as np


# ============================================================
# Utils
# ============================================================

def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def normalize_line_angle_deg(angle_deg):
    """
    직선 orientation은 180도 주기.
    결과 범위: [-90, 90)
    """
    return ((angle_deg + 90.0) % 180.0) - 90.0


def shortest_line_angle_diff_deg(a_from, a_to):
    """
    180도 주기 직선 orientation에서 signed shortest diff.
    C10에서 방향이 뒤집히는 문제 방지.
    """
    return ((a_to - a_from + 90.0) % 180.0) - 90.0


def line_orientation_between_points(p0, p1):
    x0, y0 = p0
    x1, y1 = p1
    angle = math.degrees(math.atan2(y1 - y0, x1 - x0))
    return normalize_line_angle_deg(angle)


def angle_to_dir(angle_deg):
    """
    line orientation angle -> unit direction.
    t를 [-a, a]로 symmetric하게 sampling하면 부호는 크게 중요하지 않음.
    """
    rad = math.radians(angle_deg)
    return np.array([math.cos(rad), math.sin(rad)], dtype=np.float32)


# ============================================================
# 1. Paper-like C0~Cmid~C20 curve
# ============================================================

def make_paper_like_rotation_curve(
    C0,
    Cmid,
    C20,
    n_total=21,
    mid_index=10,
    curvature_alpha=0.95,
):
    C0 = np.asarray(C0, dtype=np.float64)
    Cmid = np.asarray(Cmid, dtype=np.float64)
    C20 = np.asarray(C20, dtype=np.float64)

    B = 0.5 * (C0 + C20)

    u_vec = C20 - C0
    u_norm = np.linalg.norm(u_vec)
    if u_norm < 1e-8:
        raise ValueError("C0 and C20 are too close.")

    e_u = u_vec / u_norm
    w = u_norm / 2.0

    v_vec = Cmid - B
    H = np.linalg.norm(v_vec)
    if H < 1e-8:
        raise ValueError("Cmid is too close to midpoint of C0 and C20.")

    e_v = v_vec / H

    def local_to_image(u, v):
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        return B + u[..., None] * e_u + v[..., None] * e_v

    def v_from_u(u):
        q = 1.0 - np.abs(u) / w
        q = np.clip(q, 0.0, 1.0)
        return H * ((1.0 - curvature_alpha) * q + curvature_alpha * q * q)

    u_left = np.linspace(-w, 0.0, mid_index + 1)
    u_right = np.linspace(0.0, w, n_total - mid_index)

    u_all = np.concatenate([u_left, u_right[1:]])
    v_all = v_from_u(u_all)

    centers = local_to_image(u_all, v_all)

    u_dense = np.linspace(-w, w, 1200)
    v_dense = v_from_u(u_dense)
    dense_curve = local_to_image(u_dense, v_dense)

    info = {
        "B": B,
        "e_u": e_u,
        "e_v": e_v,
        "w": w,
        "H": H,
        "curvature_alpha": curvature_alpha,
        "mid_index": mid_index,
    }

    return centers.astype(np.float32), dense_curve.astype(np.float32), info


# ============================================================
# 2. Paper theta rule
# ============================================================

def get_theta_deg(i: int, include_c20: bool = True) -> float:
    """
    논문 규칙:
        i = 0,1,18,19 -> 0.5 deg
        i = 10        -> 1.5 deg
        otherwise     -> 0.6 deg

    C20 extension은 논문 원문에는 없지만, 대칭 extension으로 0.5 deg 사용.
    """
    if i in [0, 1, 18, 19]:
        return 0.5

    if include_c20 and i == 20:
        return 0.5

    if i == 10:
        return 1.5

    return 0.6


def generate_rotating_angles(start_angle, final_angle, step_deg, include_start=True):
    start_angle = normalize_line_angle_deg(start_angle)
    final_angle = normalize_line_angle_deg(final_angle)

    diff = shortest_line_angle_diff_deg(start_angle, final_angle)

    if abs(diff) < 1e-9:
        return [final_angle]

    sign = 1.0 if diff > 0 else -1.0
    diff_abs = abs(diff)

    angles = []
    if include_start:
        angles.append(start_angle)

    k = 1
    while True:
        moved = k * step_deg
        if moved >= diff_abs:
            break
        angles.append(normalize_line_angle_deg(start_angle + sign * moved))
        k += 1

    if len(angles) == 0 or abs(shortest_line_angle_diff_deg(angles[-1], final_angle)) > 1e-9:
        angles.append(final_angle)

    return angles


def clip_infinite_line_to_image(center, angle_deg, width, height):
    cx, cy = float(center[0]), float(center[1])

    rad = math.radians(angle_deg)
    dx = math.cos(rad)
    dy = math.sin(rad)

    eps = 1e-12
    candidates = []

    if abs(dx) > eps:
        for x in [0.0, float(width - 1)]:
            t = (x - cx) / dx
            y = cy + t * dy
            if 0.0 <= y <= height - 1:
                candidates.append((x, y, t))

    if abs(dy) > eps:
        for y in [0.0, float(height - 1)]:
            t = (y - cy) / dy
            x = cx + t * dx
            if 0.0 <= x <= width - 1:
                candidates.append((x, y, t))

    unique = []
    for x, y, t in candidates:
        duplicate = False
        for ux, uy, _ in unique:
            if abs(x - ux) < 1e-6 and abs(y - uy) < 1e-6:
                duplicate = True
                break
        if not duplicate:
            unique.append((x, y, t))

    if len(unique) < 2:
        return None

    unique = sorted(unique, key=lambda z: z[2])

    return (
        (unique[0][0], unique[0][1]),
        (unique[-1][0], unique[-1][1]),
    )


# ============================================================
# 3. Rotation rays 생성
# ============================================================

def generate_rotation_ray_geometry(
    centers,
    image_shape,
    include_c20=True,
    include_start_angle=True,
):
    """
    centers: (21, 2), image coordinate 기준.

    Returns
    -------
    geometry dict:
        centers          : (21, 2)
        ray_origins      : (R, 2)
        ray_dirs         : (R, 2)
        ray_angles_deg   : (R,)
        ray_center_ids   : (R,)
        ray_segments     : (R, 4) = x1,y1,x2,y2
        ray_is_final     : (R,)
        ray_is_terminal  : (R,)
    """
    centers = np.asarray(centers, dtype=np.float32)
    height, width = image_shape[:2]
    n = len(centers)

    if n != 21:
        raise ValueError(f"Expected 21 centers C0~C20, got {n}.")

    seg_angles = []
    for i in range(n - 1):
        seg_angles.append(line_orientation_between_points(centers[i], centers[i + 1]))

    ray_origins = []
    ray_dirs = []
    ray_angles = []
    ray_center_ids = []
    ray_segments = []
    ray_is_final = []
    ray_is_terminal = []

    # C0 ~ C19
    for i in range(n - 1):
        theta_i = get_theta_deg(i, include_c20=include_c20)
        final_angle = seg_angles[i]

        if i == 0:
            if len(seg_angles) >= 2:
                turn = shortest_line_angle_diff_deg(seg_angles[0], seg_angles[1])
                start_angle = normalize_line_angle_deg(seg_angles[0] - turn)
            else:
                start_angle = seg_angles[0]
        else:
            start_angle = seg_angles[i - 1]

        angles = generate_rotating_angles(
            start_angle=start_angle,
            final_angle=final_angle,
            step_deg=theta_i,
            include_start=include_start_angle,
        )

        for j, angle in enumerate(angles):
            segment = clip_infinite_line_to_image(
                center=centers[i],
                angle_deg=angle,
                width=width,
                height=height,
            )
            if segment is None:
                continue

            (x1, y1), (x2, y2) = segment

            ray_origins.append(centers[i])
            ray_dirs.append(angle_to_dir(angle))
            ray_angles.append(angle)
            ray_center_ids.append(i)
            ray_segments.append([x1, y1, x2, y2])
            ray_is_final.append(j == len(angles) - 1)
            ray_is_terminal.append(False)

    # C20 terminal extension
    if include_c20:
        i = n - 1

        angle_prevprev = seg_angles[-2]  # C18 -> C19
        angle_prev = seg_angles[-1]      # C19 -> C20

        terminal_turn = shortest_line_angle_diff_deg(angle_prevprev, angle_prev)

        start_angle = angle_prev
        final_angle = normalize_line_angle_deg(angle_prev + terminal_turn)
        theta_i = get_theta_deg(i, include_c20=True)

        angles = generate_rotating_angles(
            start_angle=start_angle,
            final_angle=final_angle,
            step_deg=theta_i,
            include_start=include_start_angle,
        )

        for j, angle in enumerate(angles):
            segment = clip_infinite_line_to_image(
                center=centers[i],
                angle_deg=angle,
                width=width,
                height=height,
            )
            if segment is None:
                continue

            (x1, y1), (x2, y2) = segment

            ray_origins.append(centers[i])
            ray_dirs.append(angle_to_dir(angle))
            ray_angles.append(angle)
            ray_center_ids.append(i)
            ray_segments.append([x1, y1, x2, y2])
            ray_is_final.append(j == len(angles) - 1)
            ray_is_terminal.append(True)

    geometry = {
        "centers": centers.astype(np.float32),
        "ray_origins": np.asarray(ray_origins, dtype=np.float32),
        "ray_dirs": np.asarray(ray_dirs, dtype=np.float32),
        "ray_angles_deg": np.asarray(ray_angles, dtype=np.float32),
        "ray_center_ids": np.asarray(ray_center_ids, dtype=np.int32),
        "ray_segments": np.asarray(ray_segments, dtype=np.float32),
        "ray_is_final": np.asarray(ray_is_final, dtype=np.bool_),
        "ray_is_terminal": np.asarray(ray_is_terminal, dtype=np.bool_),
    }

    return geometry


# ============================================================
# 4. Save / Load
# ============================================================

def save_rotation_geometry_npz(
    save_path,
    geometry,
    dense_curve,
    image_path,
    image_shape,
    C0,
    Cmid,
    C20,
    curvature_alpha,
    mid_index,
    include_c20,
):
    ensure_dir(os.path.dirname(save_path))

    meta = {
        "image_path": image_path,
        "image_shape_H_W": [int(image_shape[0]), int(image_shape[1])],
        "C0": [int(C0[0]), int(C0[1])],
        "Cmid": [int(Cmid[0]), int(Cmid[1])],
        "C20": [int(C20[0]), int(C20[1])],
        "curvature_alpha": float(curvature_alpha),
        "mid_index": int(mid_index),
        "include_c20": bool(include_c20),
        "coordinate_system": "input_image_xy",
        "note": "x is image column coordinate, y is image row coordinate.",
    }

    np.savez_compressed(
        save_path,
        centers=geometry["centers"],
        dense_curve=np.asarray(dense_curve, dtype=np.float32),
        ray_origins=geometry["ray_origins"],
        ray_dirs=geometry["ray_dirs"],
        ray_angles_deg=geometry["ray_angles_deg"],
        ray_center_ids=geometry["ray_center_ids"],
        ray_segments=geometry["ray_segments"],
        ray_is_final=geometry["ray_is_final"],
        ray_is_terminal=geometry["ray_is_terminal"],
        meta_json=np.asarray(json.dumps(meta, ensure_ascii=False)),
    )


def load_rotation_geometry_npz(npz_path):
    data = np.load(npz_path, allow_pickle=False)

    meta_json = str(data["meta_json"])
    meta = json.loads(meta_json)

    return {
        "centers": data["centers"].astype(np.float32),
        "dense_curve": data["dense_curve"].astype(np.float32),
        "ray_origins": data["ray_origins"].astype(np.float32),
        "ray_dirs": data["ray_dirs"].astype(np.float32),
        "ray_angles_deg": data["ray_angles_deg"].astype(np.float32),
        "ray_center_ids": data["ray_center_ids"].astype(np.int32),
        "ray_segments": data["ray_segments"].astype(np.float32),
        "ray_is_final": data["ray_is_final"].astype(bool),
        "ray_is_terminal": data["ray_is_terminal"].astype(bool),
        "meta": meta,
    }


def save_centers_txt(centers, save_path):
    ensure_dir(os.path.dirname(save_path))

    centers_int = np.round(centers).astype(int)

    with open(save_path, "w", encoding="utf-8") as f:
        for i, (x, y) in enumerate(centers_int):
            f.write(f"C{i}: ({x}, {y})\n")


# ============================================================
# 5. Debug visualization
# ============================================================

def draw_geometry_debug(
    image_path,
    geometry,
    dense_curve,
    C0,
    Cmid,
    C20,
    save_path,
):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)

    vis = img.copy()

    centers = geometry["centers"]
    ray_segments = geometry["ray_segments"]
    ray_center_ids = geometry["ray_center_ids"]
    ray_is_final = geometry["ray_is_final"]
    ray_is_terminal = geometry["ray_is_terminal"]

    # dense curve
    curve_int = np.round(dense_curve).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(
        vis,
        [curve_int],
        isClosed=False,
        color=(255, 255, 255),
        thickness=1,
        lineType=cv2.LINE_AA,
    )

    # rays
    for k, (x1, y1, x2, y2) in enumerate(ray_segments):
        cid = int(ray_center_ids[k])

        if ray_is_terminal[k]:
            color = (255, 0, 255)      # C20 terminal extension
        elif cid == 10:
            color = (0, 255, 0)        # C10
        elif ray_is_final[k]:
            color = (0, 165, 255)      # final ray for each center
        else:
            color = (0, 255, 255)      # normal ray

        thickness = 2 if ray_is_final[k] else 1

        cv2.line(
            vis,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )

    # centers
    centers_int = np.round(centers).astype(np.int32)
    for i, (x, y) in enumerate(centers_int):
        if i == 0:
            color = (0, 0, 255)
            r = 5
        elif i == 10:
            color = (0, 255, 0)
            r = 5
        elif i == 20:
            color = (255, 0, 0)
            r = 5
        else:
            color = (255, 255, 0)
            r = 3

        cv2.circle(vis, (int(x), int(y)), r, color, -1)
        cv2.putText(
            vis,
            f"C{i}",
            (int(x) + 3, int(y) - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            color,
            1,
            cv2.LINE_AA,
        )

    # key labels
    for name, p, color in [
        ("C0", C0, (0, 0, 255)),
        ("Cmid", Cmid, (0, 255, 0)),
        ("C20", C20, (255, 0, 0)),
    ]:
        x, y = map(int, p)
        cv2.putText(
            vis,
            f"{name} ({x},{y})",
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    ensure_dir(os.path.dirname(save_path))
    cv2.imwrite(save_path, vis)


def draw_per_center_ray_debugs(
    image_path,
    geometry,
    dense_curve,
    save_dir,
):
    """
    C0~C20 각각에 대해 해당 center에서 생성된 ray만 따로 시각화해서 저장.

    저장 파일 예:
        save_dir/c00_rays_debug.png
        save_dir/c01_rays_debug.png
        ...
        save_dir/c20_rays_debug.png
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)

    ensure_dir(save_dir)

    centers = geometry["centers"]
    ray_segments = geometry["ray_segments"]
    ray_center_ids = geometry["ray_center_ids"]
    ray_angles_deg = geometry["ray_angles_deg"]
    ray_is_final = geometry["ray_is_final"]
    ray_is_terminal = geometry["ray_is_terminal"]

    centers_int = np.round(centers).astype(np.int32)
    curve_int = np.round(dense_curve).astype(np.int32).reshape(-1, 1, 2)

    save_paths = []

    for target_cid in range(len(centers)):
        vis = img.copy()

        # dense curve
        cv2.polylines(
            vis,
            [curve_int],
            isClosed=False,
            color=(180, 180, 180),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

        # all centers: context용으로 작게 표시
        for i, (x, y) in enumerate(centers_int):
            if i == target_cid:
                continue
            cv2.circle(vis, (int(x), int(y)), 2, (120, 120, 120), -1)
            cv2.putText(
                vis,
                f"C{i}",
                (int(x) + 3, int(y) - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.28,
                (120, 120, 120),
                1,
                cv2.LINE_AA,
            )

        # target center의 ray만 표시
        idxs = np.where(ray_center_ids == target_cid)[0]
        for k in idxs:
            x1, y1, x2, y2 = ray_segments[k]

            if ray_is_terminal[k]:
                color = (255, 0, 255)      # C20 terminal extension
            elif ray_is_final[k]:
                color = (0, 165, 255)      # final ray
            else:
                color = (0, 255, 255)      # intermediate ray

            thickness = 2 if ray_is_final[k] else 1

            cv2.line(
                vis,
                (int(round(x1)), int(round(y1))),
                (int(round(x2)), int(round(y2))),
                color,
                thickness,
                lineType=cv2.LINE_AA,
            )

        # target center 강조
        x, y = centers_int[target_cid]
        cv2.circle(vis, (int(x), int(y)), 6, (0, 0, 255), -1)
        cv2.putText(
            vis,
            f"C{target_cid}",
            (int(x) + 6, int(y) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

        # ray 개수 및 angle 범위 표시
        if len(idxs) > 0:
            angles = ray_angles_deg[idxs]
            info_text = (
                f"C{target_cid}: rays={len(idxs)}, "
                f"angle=[{angles.min():.2f}, {angles.max():.2f}] deg"
            )
        else:
            info_text = f"C{target_cid}: rays=0"

        cv2.putText(
            vis,
            info_text,
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

        save_path = os.path.join(save_dir, f"c{target_cid:02d}_rays_debug.png")
        cv2.imwrite(save_path, vis)
        save_paths.append(save_path)

    return save_paths


# ============================================================
# 6. Optional: input image coordinate -> volume coordinate scaling
# ============================================================

def scale_geometry_xy(geometry, sx, sy):
    """
    나중에 input image 좌표계를 volume 좌표계로 변환할 때 사용.
    x에는 sx, y에는 sy를 곱함.

    주의:
    cbct_simpx2.py의 load_centerline()은 arr.T를 사용하므로,
    그 좌표계와 맞추려면 동일한 방식으로 sx, sy를 정해야 함.
    """
    g = {}
    for k, v in geometry.items():
        if isinstance(v, np.ndarray):
            g[k] = v.copy()
        else:
            g[k] = v

    for key in ["centers", "dense_curve", "ray_origins"]:
        g[key][:, 0] *= sx
        g[key][:, 1] *= sy

    # segment: x1,y1,x2,y2
    g["ray_segments"][:, [0, 2]] *= sx
    g["ray_segments"][:, [1, 3]] *= sy

    # ray_dirs는 방향 벡터라서 단순 scale 후 renormalize 필요
    dirs = g["ray_dirs"].copy()
    dirs[:, 0] *= sx
    dirs[:, 1] *= sy
    norms = np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    g["ray_dirs"] = (dirs / norms).astype(np.float32)

    return g


# ============================================================
# 7. Main
# ============================================================

if __name__ == "__main__":
    image_path = "/home/alphayoung8/log1/nebla/curve_smooth_320.png"

    C0 = (114, 66)
    Cmid = (266, 166)
    C20 = (114, 265)

    out_dir = "templete_center"
    ensure_dir(out_dir)

    curvature_alpha = 0.99
    mid_index = 10
    include_c20 = True

    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(image_path)

    image_shape = img_gray.shape  # (H, W)

    centers, dense_curve, info = make_paper_like_rotation_curve(
        C0=C0,
        Cmid=Cmid,
        C20=C20,
        n_total=21,
        mid_index=mid_index,
        curvature_alpha=curvature_alpha,
    )

    geometry = generate_rotation_ray_geometry(
        centers=centers,
        image_shape=image_shape,
        include_c20=include_c20,
        include_start_angle=True,
    )

    npz_path = os.path.join(out_dir, "rotation_geometry_input_image.npz")
    png_path = os.path.join(out_dir, "rotation_geometry_debug.png")
    txt_path = os.path.join(out_dir, "rotation_centers.txt")
    per_center_debug_dir = os.path.join(out_dir, "debug_rays_by_center")

    save_rotation_geometry_npz(
        save_path=npz_path,
        geometry=geometry,
        dense_curve=dense_curve,
        image_path=image_path,
        image_shape=image_shape,
        C0=C0,
        Cmid=Cmid,
        C20=C20,
        curvature_alpha=curvature_alpha,
        mid_index=mid_index,
        include_c20=include_c20,
    )

    draw_geometry_debug(
        image_path=image_path,
        geometry=geometry,
        dense_curve=dense_curve,
        C0=C0,
        Cmid=Cmid,
        C20=C20,
        save_path=png_path,
    )

    per_center_debug_paths = draw_per_center_ray_debugs(
        image_path=image_path,
        geometry=geometry,
        dense_curve=dense_curve,
        save_dir=per_center_debug_dir,
    )

    save_centers_txt(
        centers=geometry["centers"],
        save_path=txt_path,
    )

    print("Saved:")
    print("  ", npz_path)
    print("  ", png_path)
    print("  ", txt_path)
    print("  ", per_center_debug_dir)

    print("\nGeometry:")
    print("  centers       :", geometry["centers"].shape)
    print("  ray_origins   :", geometry["ray_origins"].shape)
    print("  ray_dirs      :", geometry["ray_dirs"].shape)
    print("  ray_segments  :", geometry["ray_segments"].shape)
    print("  ray count     :", len(geometry["ray_origins"]))
    print("  per-center png:", len(per_center_debug_paths))

    print("\nCenters:")
    for i, p in enumerate(np.round(geometry["centers"]).astype(int)):
        print(f"C{i}: ({p[0]}, {p[1]})")