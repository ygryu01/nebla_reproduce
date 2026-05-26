import os
import cv2
import numpy as np


def save_binary(path, mask):
    """
    bool mask를 0/255 이미지로 저장.
    """
    cv2.imwrite(path, (mask.astype(np.uint8) * 255))


def keep_largest_component(mask: np.ndarray):
    """
    가장 큰 흰색 connected component만 남깁니다.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8
    )

    if num_labels <= 1:
        raise ValueError("No white component found.")

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest = labels == largest_label

    return largest, labels, largest_label


def find_endpoint_candidates(mask: np.ndarray):
    """
    8-neighborhood degree가 1인 점들을 endpoint candidate로 검출합니다.
    선이 anti-aliasing되어 두꺼우면 실패할 수 있으므로,
    최종 선택에서는 left-band fallback도 같이 사용합니다.
    """
    h, w = mask.shape
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
    degree = np.zeros((h, w), dtype=np.uint8)

    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue

            degree += padded[
                1 + dy : 1 + dy + h,
                1 + dx : 1 + dx + w
            ]

    ys, xs = np.where(mask & (degree == 1))

    if len(xs) == 0:
        return np.empty((0, 2), dtype=int)

    return np.stack([xs, ys], axis=1)


def find_two_left_points(mask: np.ndarray, band_width: int = 4):
    """
    가장 왼쪽 흰 점 두 개를 찾습니다.
    방법:
    1. 전체 curve에서 min_x를 찾음
    2. x <= min_x + band_width 영역만 사용
    3. 그 안에서 connected component를 잡음
    4. 위쪽 component centroid를 C0, 아래쪽 component centroid를 C20으로 둠
    """
    ys, xs = np.where(mask)

    if len(xs) == 0:
        raise ValueError("Empty mask.")

    min_x = xs.min()

    yy, xx = np.indices(mask.shape)
    left_band = mask & (xx <= min_x + band_width)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        left_band.astype(np.uint8),
        connectivity=8
    )

    comps = []
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area > 0:
            cx, cy = centroids[label]
            comps.append({
                "label": label,
                "area": area,
                "centroid": (cx, cy)
            })

    if len(comps) < 2:
        raise ValueError(
            "Could not find two left components. "
            "Try increasing band_width or lowering threshold."
        )

    # 면적이 큰 두 component 선택
    comps = sorted(comps, key=lambda d: d["area"], reverse=True)[:2]

    pts = np.array(
        [[round(c["centroid"][0]), round(c["centroid"][1])] for c in comps],
        dtype=int
    )

    # 위쪽 점이 C0, 아래쪽 점이 C20
    pts = pts[np.argsort(pts[:, 1])]

    return pts[0], pts[1], left_band


def find_rightmost_point(mask: np.ndarray):
    """
    가장 오른쪽 흰 점을 C11로 잡습니다.
    같은 max_x에 여러 픽셀이 있으면 y 평균을 사용합니다.
    """
    ys, xs = np.where(mask)

    max_x = xs.max()
    right_ys = ys[xs == max_x]

    c11 = np.array([
        int(max_x),
        int(round(right_ys.mean()))
    ])

    return c11


def draw_points(gray, points_dict):
    """
    최종 keypoint를 원본 이미지 위에 표시합니다.
    """
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    colors = {
        "C0": (0, 0, 255),
        "C11": (0, 255, 0),
        "C20": (255, 0, 0),
    }

    for name, p in points_dict.items():
        x, y = int(p[0]), int(p[1])
        color = colors.get(name, (0, 0, 255))

        cv2.circle(vis, (x, y), 5, color, -1)
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

    return vis


def draw_endpoint_candidates(gray, endpoints):
    """
    endpoint candidate들을 노란색으로 표시합니다.
    """
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for x, y in endpoints:
        cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 255), -1)

    return vis


def draw_left_band(gray, left_band, c0=None, c20=None):
    """
    left-band 영역을 시각화합니다.
    """
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # left_band 픽셀을 빨간색으로 표시
    vis[left_band] = (0, 0, 255)

    if c0 is not None:
        cv2.circle(vis, tuple(map(int, c0)), 5, (0, 255, 0), -1)
        cv2.putText(
            vis,
            "C0",
            (int(c0[0]) + 8, int(c0[1]) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    if c20 is not None:
        cv2.circle(vis, tuple(map(int, c20)), 5, (255, 0, 0), -1)
        cv2.putText(
            vis,
            "C20",
            (int(c20[0]) + 8, int(c20[1]) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return vis


def find_rotation_center_keypoints(
    image_path: str,
    threshold: int = 127,
    left_band_width: int = 4,
    debug_dir: str | None = None,
):
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if gray is None:
        raise FileNotFoundError(image_path)

    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "00_original.png"), gray)

    # 1. thresholding
    raw_mask = gray > threshold

    if debug_dir is not None:
        save_binary(os.path.join(debug_dir, "01_threshold_mask.png"), raw_mask)

    # 2. largest connected component
    mask, labels, largest_label = keep_largest_component(raw_mask)

    if debug_dir is not None:
        save_binary(os.path.join(debug_dir, "02_largest_component.png"), mask)

    # 3. endpoint candidates
    endpoints = find_endpoint_candidates(mask)

    if debug_dir is not None:
        ep_vis = draw_endpoint_candidates(gray, endpoints)
        cv2.imwrite(os.path.join(debug_dir, "03_endpoint_candidates.png"), ep_vis)

    # 4. C11: 가장 오른쪽 흰 점
    c11 = find_rightmost_point(mask)

    # 5. C0, C20: 가장 왼쪽 흰 점 두 개
    c0, c20, left_band = find_two_left_points(
        mask,
        band_width=left_band_width
    )

    if debug_dir is not None:
        save_binary(os.path.join(debug_dir, "04_left_band_mask.png"), left_band)

        left_band_vis = draw_left_band(gray, left_band, c0, c20)
        cv2.imwrite(os.path.join(debug_dir, "05_left_band_debug.png"), left_band_vis)

    points = {
        "C0": c0,
        "C11": c11,
        "C20": c20,
    }

    # 6. final visualization
    final_vis = draw_points(gray, points)

    if debug_dir is not None:
        cv2.imwrite(os.path.join(debug_dir, "06_final_keypoints.png"), final_vis)

        with open(os.path.join(debug_dir, "07_keypoints.txt"), "w") as f:
            for name, p in points.items():
                f.write(f"{name}: ({int(p[0])}, {int(p[1])})\n")

    return {
        "C0": tuple(map(int, c0)),
        "C11": tuple(map(int, c11)),
        "C20": tuple(map(int, c20)),
    }


if __name__ == "__main__":
    image_path = "/home/alphayoung8/log1/nebla/curve_smooth_320.png"

    pts = find_rotation_center_keypoints(
        image_path=image_path,
        threshold=127,
        left_band_width=4,
        debug_dir="rotation_center_debug",
    )

    print(pts)