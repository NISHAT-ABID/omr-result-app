"""
Smart OMR image scanner.

Pipeline:
1. EXIF orientation is handled by app.py before this module receives an image.
2. Detect the largest plausible document quadrilateral.
3. Order its four corners robustly.
4. Perspective-warp the paper into a standard portrait canvas.
5. Apply conservative illumination/contrast normalization.

Important: enhancement is intentionally moderate. The module must NOT erase
light pencil marks or manufacture dark marks because final answer detection is
performed by omr_scanner.py.
"""

import cv2
import numpy as np

DEFAULT_WARP_WIDTH = 1200
DEFAULT_WARP_HEIGHT = 1600


def _order_points(points):
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)

    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)

    rect[0] = pts[np.argmin(sums)]   # top-left
    rect[2] = pts[np.argmax(sums)]   # bottom-right
    rect[1] = pts[np.argmin(diffs)]  # top-right
    rect[3] = pts[np.argmax(diffs)]  # bottom-left
    return rect


def _quad_score(quad, image_shape):
    """Score a candidate quadrilateral. Higher means more sheet-like."""
    h, w = image_shape[:2]
    pts = _order_points(quad)

    area = abs(cv2.contourArea(pts.reshape(-1, 1, 2)))
    image_area = float(h * w)
    area_ratio = area / image_area if image_area else 0.0
    if area_ratio < 0.12:
        return -1.0

    # A photographed page should be reasonably convex.
    if not cv2.isContourConvex(pts.reshape(-1, 1, 2).astype(np.int32)):
        return -1.0

    tl, tr, br, bl = pts
    widths = [np.linalg.norm(tr - tl), np.linalg.norm(br - bl)]
    heights = [np.linalg.norm(bl - tl), np.linalg.norm(br - tr)]

    min_side = max(1.0, min(widths + heights))
    max_side = max(widths + heights)
    regularity = min_side / max_side

    # Prefer large documents, but avoid blindly selecting the whole camera frame.
    return area_ratio * 2.5 + regularity


def detect_sheet_quad(image_bgr):
    """Return the best detected 4-corner paper boundary or None."""
    if image_bgr is None or image_bgr.size == 0:
        return None

    h, w = image_bgr.shape[:2]
    if min(h, w) < 100:
        return None

    scale = min(1.0, 1000.0 / max(h, w))
    small = image_bgr
    if scale < 1.0:
        small = cv2.resize(
            image_bgr,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Combine Canny and adaptive threshold candidates. Different lighting
    # conditions favour different methods.
    edge = cv2.Canny(gray, 45, 140)
    edge = cv2.dilate(edge, np.ones((3, 3), np.uint8), iterations=2)

    candidates = []
    for mode in ("edge", "adaptive"):
        if mode == "edge":
            work = edge
        else:
            work = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 31, 7
            )
            work = cv2.morphologyEx(
                work, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2
            )

        contours, _ = cv2.findContours(
            work, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:40]:
            peri = cv2.arcLength(contour, True)
            if peri <= 0:
                continue

            for eps in (0.015, 0.02, 0.025, 0.03):
                approx = cv2.approxPolyDP(contour, eps * peri, True)
                if len(approx) == 4:
                    score = _quad_score(approx.reshape(4, 2), small.shape)
                    if score > 0:
                        candidates.append((score, approx.reshape(4, 2)))
                    break

    if not candidates:
        return None

    _, best = max(candidates, key=lambda item: item[0])

    if scale < 1.0:
        best = best.astype(np.float32) / scale

    return _order_points(best)


def warp_sheet(image_bgr, quad, width=DEFAULT_WARP_WIDTH,
               height=DEFAULT_WARP_HEIGHT):
    if quad is None:
        return None

    src = _order_points(quad)
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        image_bgr,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _normalize_lighting(image_bgr):
    """Conservative illumination correction that preserves real pen marks."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Local contrast correction on luminance only. This avoids changing the
    # colour logic used later by omr_scanner to suppress pink/magenta printing.
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l2 = clahe.apply(l)

    corrected = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)

    # Mild denoising only; heavy blur can destroy faint pencil marks.
    return cv2.bilateralFilter(corrected, 5, 25, 25)


def process_captured_frame(image_bgr, warp_width=DEFAULT_WARP_WIDTH,
                           warp_height=DEFAULT_WARP_HEIGHT):
    """
    Main API used by app.py.

    Returns:
        (processed_bgr, quad)
        processed_bgr is None when a reliable full sheet boundary is not found.
    """
    if image_bgr is None or image_bgr.size == 0:
        return None, None

    quad = detect_sheet_quad(image_bgr)
    if quad is None:
        return None, None

    warped = warp_sheet(image_bgr, quad, warp_width, warp_height)
    if warped is None:
        return None, None

    return _normalize_lighting(warped), quad


def draw_detected_outline(image_bgr, quad):
    """Utility for optional debugging/live-preview UIs."""
    preview = image_bgr.copy()
    if quad is not None:
        pts = np.asarray(quad, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(preview, [pts], True, (0, 255, 0), 4)
    return preview
