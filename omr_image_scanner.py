"""Camera OMR preprocessing.

This module only prepares a camera frame for the existing OMR calibration and
answer-reading pipeline. It does not read answers or change the review flow.
"""
from __future__ import annotations
from typing import Optional, Tuple
import cv2
import numpy as np

# A portrait OMR sheet normally falls around 0.55-0.80 when expressed as
# short-side / long-side. Keep a little tolerance for perspective.
TARGET_ASPECT_MIN = 0.45
TARGET_ASPECT_MAX = 0.90
MIN_AREA_RATIO = 0.10
OUTPUT_MAX_WIDTH = 1800


def _order_quad(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    return np.array([
        pts[np.argmin(s)],
        pts[np.argmin(d)],
        pts[np.argmax(s)],
        pts[np.argmax(d)],
    ], dtype=np.float32)


def _quad_quality(q: np.ndarray, frame_area: float, contour_area: float) -> Optional[float]:
    q = _order_quad(q)
    tl, tr, br, bl = q
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    if width < 300 or height < 450:
        return None
    aspect = min(width, height) / max(width, height)
    if not (TARGET_ASPECT_MIN <= aspect <= TARGET_ASPECT_MAX):
        return None

    xs, ys = q[:, 0], q[:, 1]
    if xs.min() < -0.03 * width or ys.min() < -0.03 * height:
        return None

    quad_area = abs(cv2.contourArea(q.reshape(-1, 1, 2)))
    rectangularity = quad_area / max(width * height, 1.0)
    area_ratio = quad_area / max(frame_area, 1.0)
    if area_ratio < MIN_AREA_RATIO or rectangularity < 0.72:
        return None

    # Strong preference for a large, well-formed sheet. The area term makes
    # small internal boxes much less likely to win over the page outline.
    return area_ratio * (0.70 + 0.30 * rectangularity)


def detect_sheet_quad(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Find the full OMR page using both edge and bright-page masks."""
    if frame_bgr is None or frame_bgr.size == 0:
        return None

    h, w = frame_bgr.shape[:2]
    scale = min(1.0, 1400.0 / max(h, w))
    small = cv2.resize(frame_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    frame_area = float(sw * sh)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Method 1: page/background edge.
    edges = cv2.Canny(blur, 25, 110)
    edges = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2
    )

    # Method 2: white-sheet mask. This is useful when the page edge is a
    # clean brightness transition but Canny produces a broken outline.
    _, bright = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = cv2.morphologyEx(
        bright, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2
    )
    bright = cv2.morphologyEx(
        bright, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1
    )

    best = None
    best_score = -1.0

    def inspect(binary: np.ndarray) -> None:
        nonlocal best, best_score
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA_RATIO * frame_area:
                continue
            peri = cv2.arcLength(cnt, True)
            for eps in (0.018, 0.025, 0.035):
                approx = cv2.approxPolyDP(cnt, eps * peri, True)
                if len(approx) != 4 or not cv2.isContourConvex(approx):
                    continue
                q = approx.reshape(4, 2).astype(np.float32)
                quality = _quad_quality(q, frame_area, area)
                if quality is not None and quality > best_score:
                    best_score = quality
                    best = q / scale
                break

    inspect(edges)
    inspect(bright)

    if best is None:
        return None
    return _order_quad(best)


def perspective_flatten(
    frame_bgr: np.ndarray, quad: np.ndarray, max_width: int = OUTPUT_MAX_WIDTH
) -> np.ndarray:
    """Perspective-correct the page while retaining as much camera detail as practical."""
    q = _order_quad(quad)
    tl, tr, br, bl = q
    out_w = max(900, int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))))
    out_h = max(1200, int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))))

    if out_w > max_width:
        s = max_width / float(out_w)
        out_w = int(round(out_w * s))
        out_h = int(round(out_h * s))

    out_w = max(900, min(out_w, max_width))
    out_h = max(1200, min(out_h, int(max_width * 1.85)))

    dst = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1],
    ], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(
        frame_bgr,
        matrix,
        (out_w, out_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
    )


def moderate_enhance(image_bgr: np.ndarray) -> np.ndarray:
    """Correct mild uneven lighting while keeping the original OMR appearance."""
    img = image_bgr.copy()
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Low-frequency illumination correction, blended gently so shadows are
    # reduced without making the paper look artificially white.
    illumination = cv2.GaussianBlur(l, (0, 0), 25)
    illum_mean = float(np.mean(illumination))
    corrected = l.astype(np.float32) * (illum_mean / np.maximum(illumination.astype(np.float32), 1.0))
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    l = cv2.addWeighted(l, 0.65, corrected, 0.35, 0)

    clahe = cv2.createCLAHE(clipLimit=1.05, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # Very mild unsharp mask. No thresholding, binarisation, or edge drawing
    # is applied to the stored image, so bubble geometry and ink colour stay intact.
    soft = cv2.GaussianBlur(enhanced, (0, 0), 0.8)
    sharp = cv2.addWeighted(enhanced, 1.06, soft, -0.06, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def process_captured_frame(
    frame_bgr: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return a flat, natural-looking OMR image and the detected page corners."""
    quad = detect_sheet_quad(frame_bgr)
    if quad is None:
        return None, None
    flat = perspective_flatten(frame_bgr, quad)
    return moderate_enhance(flat), quad
