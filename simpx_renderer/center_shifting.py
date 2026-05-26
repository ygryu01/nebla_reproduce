# save_shifted_rotation_geometry.py

import os
import math
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


# =========================================================
# Settings
# =========================================================

CSV_PATH = Path("/home/alphayoung8/log1/oral3d_reproduce/extension_result/rightmost_points_ij.csv")
TEMPLATE_CURVE = Path("/home/alphayoung8/log1/nebla/simpx_renderer/curve_smooth_320.png")

OUT_SHIFTED_CURVE_DIR = Path("/home/alphayoung8/log1/nebla/simpx_renderer/shifted_curves")
OUT_GEOM_DIR = Path("/home/alphayoung8/log1/nebla/simpx_renderer/center_geometries")

OUT_SHIFTED_CURVE_DIR.mkdir(parents=True, exist_ok=True)
OUT_GEOM_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 127
TIE = "median"

# template image coordinate 기준: (x, y) = (col, row)
TEMPLATE_C0 = (114, 66)
TEMPLATE_CMID = (266, 166)
TEMPLATE_C20 = (114, 265)

N_CENTERS = 21
MID_INDEX = 10
CURVATURE_ALPHA = 0.99

INCLUDE_C20_EXTENSION = True
INCLUDE_START_ANGLE = True


# =========================================================
# Basic image / shift utils
# =========================================================

def load_binary_png(path: Path, threshold: int = 127) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.asarray(img)
    return arr > threshold


def save_binary_png(mask: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = (mask > 0).astype(np.uint8) * 255
    Image.fromarray(out, mode="L").save(out_path)


def rightmost_point(binary_img: np.ndarray, tie: str = "median"):
    """
    return:
        (i, j) = (row, col) = (y, x)
    """
    fg = binary_img > 0
    ys, xs = np.nonzero(fg)

    if len(xs) == 0:
        return None

    j_max = int(xs.max())
    i_candidates = ys[xs == j_max]

    if tie == "top":
        i = int(i_candidates.min())
    elif tie == "bottom":
        i = int(i_candidates.max())
    elif tie == "median":
        i = int(np.median(i_candidates))
    else:
        raise ValueError(f"Unknown tie option: {tie}")

    return i, j_max


def shift_mask(mask: np.ndarray, di: int, dj: int) -> np.ndarray:
    """
    savecurve.py와 동일한 shift 방식.

    di > 0: 아래로 이동
    di < 0: 위로 이동
    dj > 0: 오른쪽으로 이동
    dj < 0: 왼쪽으로 이동
    """
    H, W = mask.shape
    out = np.zeros_like(mask, dtype=bool)

    src_i0 = max(0, -di)
    src_i1 = min(H, H - di)
    dst_i0 = max(0, di)
    dst_i1 = min(H, H + di)

    src_j0 = max(0, -dj)
    src_j1 = min(W, W - dj)
    dst_j0 = max(0, dj)
    dst_j1 = min(W, W + dj)

    if src_i0 >= src_i1 or src_j0 >= src_j1:
        return out

    out[dst_i0:dst_i1, dst_j0:dst_j1] = mask[src_i0:src_i1, src_j0:src_j1]

    return out


def normalize_subject_code(x):
    if isinstance(x, float) and x.is_integer():
        return str(int(x))

    s = str(x)

    if s.endswith(".0"):
        return s[:-2]

    return s


def shift_xy_point(p_xy, di: int, dj: int):
    """
    p_xy = (x, y) = (col, row)
    di = row shift
    dj = col shift
    """
    x, y = p_xy
    return (int(round(x + dj)), int(round(y + di)))


# =========================================================
# Paper-like rotation center curve
# =========================================================

def make_paper_like_rotation_curve(
    C0,
    Cmid,
    C20,
    n_total=21,
    mid_index=10,
    curvature_alpha=0.95,
):
    """
    C0, Cmid, C20을 지나는 paper-like curve 생성.

    local coordinate:
        C0   = (-w, 0)
        Cmid = ( 0, H)
        C20  = ( w, 0)

    curve:
        q = 1 - |u| / w
        v = H * ((1-alpha) * q + alpha * q^2)
    """
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

    return centers.astype(np.float32), dense_curve.astype(np.float32)


# =========================================================
# Rotation ray geometry
# =========================================================

def get_theta_deg(i: int, include_c20: bool = True) -> float:
    """
    논문 규칙:
        i = 0,1,18,19 -> 0.5 deg
        i = 10        -> 1.5 deg
        otherwise     -> 0.6 deg

    C20 extension은 논문 원문에는 없지만, 대칭 extension으로 0.5 deg 사용.
    """
    if i in [0, 1, 18, 19]:
        return 0.3

    if include_c20 and i == 20:
        return 0.3

    if 6 < i < 13 and i !=10:
        return 0.8
    
    if i == 10:
        return 3.0

    return 0.25


def normalize_line_angle_deg(angle_deg: float) -> float:
    """
    직선 orientation은 180도 주기.
    결과 범위: [-90, 90)
    """
    return ((angle_deg + 90.0) % 180.0) - 90.0


def shortest_line_angle_diff_deg(a_from: float, a_to: float) -> float:
    """
    180도 주기의 직선 orientation에서 signed shortest difference.
    C10 근처에서 방향이 반대로 튀는 문제를 막기 위해 사용.
    """
    return ((a_to - a_from + 90.0) % 180.0) - 90.0


def line_orientation_between_points(p0, p1) -> float:
    x0, y0 = p0
    x1, y1 = p1

    angle = math.degrees(math.atan2(y1 - y0, x1 - x0))
    return normalize_line_angle_deg(angle)


def angle_to_unit_dir(angle_deg: float) -> np.ndarray:
    rad = math.radians(angle_deg)
    return np.array([math.cos(rad), math.sin(rad)], dtype=np.float32)


def generate_rotating_angles(
    start_angle: float,
    final_angle: float,
    step_deg: float,
    include_start: bool = True,
):
    """
    start_angle에서 final_angle까지 step_deg 간격으로 회전.
    마지막 angle은 반드시 final_angle을 포함.
    """
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

        angle = normalize_line_angle_deg(start_angle + sign * moved)
        angles.append(angle)

        k += 1

    if len(angles) == 0 or abs(shortest_line_angle_diff_deg(angles[-1], final_angle)) > 1e-9:
        angles.append(final_angle)

    return angles


def clip_infinite_line_to_image(center, angle_deg, width, height):
    """
    center를 지나고 angle_deg orientation을 갖는 무한 직선을
    이미지 boundary 내부 segment로 자름.

    return:
        ((x1, y1), (x2, y2)) or None
    """
    cx, cy = float(center[0]), float(center[1])

    rad = math.radians(angle_deg)
    dx = math.cos(rad)
    dy = math.sin(rad)

    eps = 1e-12
    candidates = []

    # x = 0, x = W-1
    if abs(dx) > eps:
        for x in [0.0, float(width - 1)]:
            t = (x - cx) / dx
            y = cy + t * dy
            if 0.0 <= y <= height - 1:
                candidates.append((x, y, t))

    # y = 0, y = H-1
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

def get_default_circular_fov_xy(width, height, margin=1.0):
    """
    image coordinate 기준:
        x = col axis, [0, W-1]
        y = row axis, [0, H-1]

    return:
        center_xy = (cx, cy)
        radius
    """
    cx = 0.5 * (width - 1)
    cy = 0.5 * (height - 1)
    radius = 0.5 * min(width, height) - float(margin)

    if radius <= 0:
        raise ValueError(f"Invalid circular FOV radius: {radius}")

    return np.array([cx, cy], dtype=np.float32), float(radius)


def clip_infinite_line_to_circle(center, angle_deg, circle_center_xy, radius):
    """
    center를 지나고 angle_deg orientation을 갖는 무한 직선을
    원형 FOV 내부 segment로 자름.

    coordinate:
        center = (x, y)
        circle_center_xy = (cx, cy)

    return:
        ((x1, y1), (x2, y2)) or None
    """
    p = np.asarray(center, dtype=np.float64)

    rad = math.radians(angle_deg)
    d = np.array([math.cos(rad), math.sin(rad)], dtype=np.float64)

    c = np.asarray(circle_center_xy, dtype=np.float64)
    r = float(radius)

    # line: p(t) = p + t d
    # circle: ||p(t) - c||^2 = r^2
    oc = p - c

    b = float(np.dot(oc, d))
    cc = float(np.dot(oc, oc) - r * r)

    disc = b * b - cc
    if disc < 0:
        return None

    sqrt_disc = math.sqrt(max(0.0, disc))

    t1 = -b - sqrt_disc
    t2 = -b + sqrt_disc

    q1 = p + t1 * d
    q2 = p + t2 * d

    return (
        (float(q1[0]), float(q1[1])),
        (float(q2[0]), float(q2[1])),
    )


def generate_rotation_ray_geometry(
    centers,
    image_shape,
    include_c20=True,
    include_start_angle=True,
):
    """
    centers: (21, 2), image coordinate, (x, y)

    저장되는 geometry:
        centers         : (21, 2)
        ray_origins     : (R, 2)
        ray_dirs        : (R, 2)
        ray_angles_deg  : (R,)
        ray_center_ids  : (R,)
        ray_segments    : (R, 4) = x1,y1,x2,y2
        ray_is_final    : (R,)
        ray_is_terminal : (R,)
    """
    centers = np.asarray(centers, dtype=np.float32)

    height, width = image_shape[:2]
    
    fov_center_xy, fov_radius = get_default_circular_fov_xy(
    width=width,
    height=height,
    margin=1.0,
    )

    n = len(centers)
    if n != 21:
        raise ValueError(f"Expected 21 centers, got {n}")

    seg_angles = []
    for i in range(n - 1):
        seg_angles.append(line_orientation_between_points(centers[i], centers[i + 1]))

    ray_origins = []
    ray_dirs = []
    ray_angles_deg = []
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
            segment = clip_infinite_line_to_circle(
                center=centers[i],
                angle_deg=angle,
                circle_center_xy=fov_center_xy,
                radius=fov_radius,
            )
            if segment is None:
                continue

            (x1, y1), (x2, y2) = segment

            ray_origins.append(centers[i])
            ray_dirs.append(angle_to_unit_dir(angle))
            ray_angles_deg.append(angle)
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
            segment = clip_infinite_line_to_circle(
                center=centers[i],
                angle_deg=angle,
                circle_center_xy=fov_center_xy,
                radius=fov_radius,
            )

            if segment is None:
                continue

            (x1, y1), (x2, y2) = segment

            ray_origins.append(centers[i])
            ray_dirs.append(angle_to_unit_dir(angle))
            ray_angles_deg.append(angle)
            ray_center_ids.append(i)
            ray_segments.append([x1, y1, x2, y2])
            ray_is_final.append(j == len(angles) - 1)
            ray_is_terminal.append(True)

    geometry = {
        "centers": centers.astype(np.float32),
        "ray_origins": np.asarray(ray_origins, dtype=np.float32),
        "ray_dirs": np.asarray(ray_dirs, dtype=np.float32),
        "ray_angles_deg": np.asarray(ray_angles_deg, dtype=np.float32),
        "ray_center_ids": np.asarray(ray_center_ids, dtype=np.int32),
        "ray_segments": np.asarray(ray_segments, dtype=np.float32),
        "ray_is_final": np.asarray(ray_is_final, dtype=np.bool_),
        "ray_is_terminal": np.asarray(ray_is_terminal, dtype=np.bool_),
    }

    return geometry


# =========================================================
# Save / debug
# =========================================================

def save_geometry_npz(
    save_path: Path,
    geometry: dict,
    dense_curve: np.ndarray,
    meta: dict,
):
    save_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        save_path,
        centers=geometry["centers"],
        dense_curve=dense_curve.astype(np.float32),
        ray_origins=geometry["ray_origins"],
        ray_dirs=geometry["ray_dirs"],
        ray_angles_deg=geometry["ray_angles_deg"],
        ray_center_ids=geometry["ray_center_ids"],
        ray_segments=geometry["ray_segments"],
        ray_is_final=geometry["ray_is_final"],
        ray_is_terminal=geometry["ray_is_terminal"],
        meta_json=np.asarray(json.dumps(meta, ensure_ascii=False)),
    )


def save_centers_txt(centers, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)

    centers_int = np.round(centers).astype(int)

    with save_path.open("w", encoding="utf-8") as f:
        for i, (x, y) in enumerate(centers_int):
            f.write(f"C{i}: ({x}, {y})\n")


def draw_geometry_debug(
    base_mask: np.ndarray,
    geometry: dict,
    dense_curve: np.ndarray,
    keypoints: dict,
    save_path: Path,
):
    """
    base_mask 위에 centers와 rays를 overlay.
    """
    base = ((base_mask > 0).astype(np.uint8) * 255)
    vis = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

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
            color = (255, 0, 255)      # C20 extension
        elif cid == 10:
            color = (0, 255, 0)        # C10
        elif ray_is_final[k]:
            color = (0, 165, 255)      # final ray
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
            radius = 5
        elif i == 10:
            color = (0, 255, 0)
            radius = 5
        elif i == 20:
            color = (255, 0, 0)
            radius = 5
        else:
            color = (255, 255, 0)
            radius = 3

        cv2.circle(vis, (int(x), int(y)), radius, color, -1)
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

    # keypoint labels
    for name, p, color in [
        ("C0", keypoints["C0"], (0, 0, 255)),
        ("Cmid", keypoints["Cmid"], (0, 255, 0)),
        ("C20", keypoints["C20"], (255, 0, 0)),
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

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), vis)


# =========================================================
# Main
# =========================================================

def main():
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    if not TEMPLATE_CURVE.is_file():
        raise FileNotFoundError(f"Template curve not found: {TEMPLATE_CURVE}")

    df = pd.read_csv(CSV_PATH)

    required_cols = {"subject_code", "i", "j"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    template_mask = load_binary_png(TEMPLATE_CURVE, threshold=THRESHOLD)
    template_rightmost = rightmost_point(template_mask, tie=TIE)

    if template_rightmost is None:
        raise ValueError("Template curve has no foreground pixels.")

    temp_i, temp_j = template_rightmost

    H, W = template_mask.shape

    print(f"Template curve: {TEMPLATE_CURVE}")
    print(f"Template shape: H={H}, W={W}")
    print(f"Template rightmost: i={temp_i}, j={temp_j}")
    print(f"CSV: {CSV_PATH}")
    print(f"Subjects: {len(df)}")

    rows = []
    errors = []

    for _, r in tqdm(df.iterrows(), total=len(df), desc="saving shifted rotation geometry"):
        subject_code = normalize_subject_code(r["subject_code"])

        try:
            target_i = int(r["i"])
            target_j = int(r["j"])

            # savecurve.py와 동일한 shift
            di = target_i - temp_i
            dj = target_j - temp_j

            shifted_mask = shift_mask(template_mask, di=di, dj=dj)

            shifted_curve_path = OUT_SHIFTED_CURVE_DIR / f"shifted_curve_{subject_code}.png"
            save_binary_png(shifted_mask, shifted_curve_path)

            # keypoints도 같은 shift 적용
            C0 = shift_xy_point(TEMPLATE_C0, di=di, dj=dj)
            Cmid = shift_xy_point(TEMPLATE_CMID, di=di, dj=dj)
            C20 = shift_xy_point(TEMPLATE_C20, di=di, dj=dj)

            centers, dense_curve = make_paper_like_rotation_curve(
                C0=C0,
                Cmid=Cmid,
                C20=C20,
                n_total=N_CENTERS,
                mid_index=MID_INDEX,
                curvature_alpha=CURVATURE_ALPHA,
            )

            geometry = generate_rotation_ray_geometry(
                centers=centers,
                image_shape=shifted_mask.shape,
                include_c20=INCLUDE_C20_EXTENSION,
                include_start_angle=INCLUDE_START_ANGLE,
            )

            subject_geom_dir = OUT_GEOM_DIR / str(subject_code)
            subject_geom_dir.mkdir(parents=True, exist_ok=True)

            npz_path = subject_geom_dir / f"rotation_geometry_{subject_code}.npz"
            debug_path = subject_geom_dir / f"rotation_debug_{subject_code}.png"
            centers_txt_path = subject_geom_dir / f"rotation_centers_{subject_code}.txt"

            meta = {
                "subject_code": subject_code,
                "coordinate_system": "input_image_xy",
                "image_shape_H_W": [int(H), int(W)],
                "template_curve": str(TEMPLATE_CURVE),
                "shifted_curve": str(shifted_curve_path),
                "template_rightmost_i_j": [int(temp_i), int(temp_j)],
                "target_rightmost_i_j": [int(target_i), int(target_j)],
                "di": int(di),
                "dj": int(dj),
                "template_C0_xy": list(map(int, TEMPLATE_C0)),
                "template_Cmid_xy": list(map(int, TEMPLATE_CMID)),
                "template_C20_xy": list(map(int, TEMPLATE_C20)),
                "shifted_C0_xy": list(map(int, C0)),
                "shifted_Cmid_xy": list(map(int, Cmid)),
                "shifted_C20_xy": list(map(int, C20)),
                "n_centers": int(N_CENTERS),
                "mid_index": int(MID_INDEX),
                "curvature_alpha": float(CURVATURE_ALPHA),
                "include_c20_extension": bool(INCLUDE_C20_EXTENSION),
                "include_start_angle": bool(INCLUDE_START_ANGLE),
                "ray_count": int(len(geometry["ray_origins"])),
            }

            save_geometry_npz(
                save_path=npz_path,
                geometry=geometry,
                dense_curve=dense_curve,
                meta=meta,
            )

            draw_geometry_debug(
                base_mask=shifted_mask,
                geometry=geometry,
                dense_curve=dense_curve,
                keypoints={
                    "C0": C0,
                    "Cmid": Cmid,
                    "C20": C20,
                },
                save_path=debug_path,
            )

            save_centers_txt(
                centers=geometry["centers"],
                save_path=centers_txt_path,
            )

            rows.append({
                "subject_code": subject_code,
                "target_i": target_i,
                "target_j": target_j,
                "template_i": temp_i,
                "template_j": temp_j,
                "di": di,
                "dj": dj,
                "C0_x": C0[0],
                "C0_y": C0[1],
                "Cmid_x": Cmid[0],
                "Cmid_y": Cmid[1],
                "C20_x": C20[0],
                "C20_y": C20[1],
                "ray_count": len(geometry["ray_origins"]),
                "shifted_curve": str(shifted_curve_path),
                "geometry_npz": str(npz_path),
                "debug_png": str(debug_path),
                "centers_txt": str(centers_txt_path),
            })

        except Exception as e:
            errors.append({
                "subject_code": subject_code,
                "error": str(e),
            })

    log_csv = OUT_GEOM_DIR / "rotation_geometry_log.csv"
    pd.DataFrame(rows).to_csv(log_csv, index=False)

    print("\n[Done]")
    print(f"Saved shifted curves: {OUT_SHIFTED_CURVE_DIR}")
    print(f"Saved geometries: {OUT_GEOM_DIR}")
    print(f"Saved log: {log_csv}")
    print(f"success={len(rows)}, errors={len(errors)}, total={len(df)}")

    if len(errors) > 0:
        error_csv = OUT_GEOM_DIR / "rotation_geometry_errors.csv"
        pd.DataFrame(errors).to_csv(error_csv, index=False)
        print(f"Saved error log: {error_csv}")


if __name__ == "__main__":
    main()