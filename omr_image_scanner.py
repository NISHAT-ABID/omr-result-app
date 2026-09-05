"""Camera OMR preprocessing.

The camera component captures the phone's native-resolution frame.  This module
finds the sheet, removes the surrounding desk/background with a perspective
warp, and applies only gentle lighting/contrast correction.  OMR answer reading
and calibration remain in omr_scanner.py.
"""
from __future__ import annotations
from typing import Optional, Tuple
import cv2
import numpy as np

TARGET_ASPECT_MIN = 0.40
TARGET_ASPECT_MAX = 0.82
MIN_AREA_RATIO = 0.10


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


def _quad_quality(q: np.ndarray, frame_w: int, frame_h: int, area: float) -> float:
    q = _order_quad(q)
    tl, tr, br, bl = q
    top = np.linalg.norm(tr - tl)
    bottom = np.linalg.norm(br - bl)
    left = np.linalg.norm(bl - tl)
    right = np.linalg.norm(br - tr)
    width = max(1.0, (top + bottom) * 0.5)
    height = max(1.0, (left + right) * 0.5)
    aspect = min(width, height) / max(width, height)
    if not (TARGET_ASPECT_MIN <= aspect <= TARGET_ASPECT_MAX):
        return -1.0

    # A real sheet is large, portrait-oriented and reasonably symmetric.
    side_balance = min(top, bottom) / max(top, bottom, 1.0)
    vertical_balance = min(left, right) / max(left, right, 1.0)
    rectangularity = side_balance * vertical_balance

    cx = float(np.mean(q[:, 0])) / max(1.0, frame_w)
    cy = float(np.mean(q[:, 1])) / max(1.0, frame_h)
    center_score = 1.0 - 0.35 * min(1.0, abs(cx - 0.5) * 2.0) - 0.12 * min(1.0, abs(cy - 0.5) * 2.0)
    return (area / float(frame_w * frame_h)) * (0.72 + 0.28 * rectangularity) * center_score * (1.0 + 0.30 * aspect)


def detect_sheet_quad(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Find the OMR paper while rejecting most desk/background rectangles."""
    if frame_bgr is None or frame_bgr.size == 0:
        return None

    h, w = frame_bgr.shape[:2]
    scale = min(1.0, 1400.0 / max(h, w))
    small = cv2.resize(frame_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 25, 105)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, -1.0

    total = float(sw * sh)
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:80]:
        area = float(cv2.contourArea(cnt))
        if area < MIN_AREA_RATIO * total:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue
        approx = cv2.approxPolyDP(cnt, 0.028 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        q_small = approx.reshape(4, 2).astype(np.float32)
        x, y, bw, bh = cv2.boundingRect(q_small.astype(np.int32))
        if bw < sw * 0.30 or bh < sh * 0.45:
            continue

        score = _quad_quality(q_small, sw, sh, area)
        if score > best_score:
            best_score = score
            best = q_small / scale

    if best is not None:
        return _order_quad(best)

    # Fallback: the paper is usually the largest bright, coherent region.
    # This is intentionally conservative and only runs when edge contours fail.
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    bright = cv2.inRange(hsv, np.array([0, 0, 125], np.uint8), np.array([179, 105, 255], np.uint8))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=2)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8), iterations=1)
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
        area = float(cv2.contourArea(cnt))
        if area < 0.16 * total:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = min(bw, bh) / max(bw, bh, 1)
        if not (TARGET_ASPECT_MIN <= aspect <= TARGET_ASPECT_MAX):
            continue
        hull = cv2.convexHull(cnt)
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.035 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            q = approx.reshape(4, 2).astype(np.float32) / scale
            return _order_quad(q)

    return None


def perspective_flatten(frame_bgr: np.ndarray, quad: np.ndarray, max_width: int = 1600) -> np.ndarray:
    """Perspective-correct the sheet without unnecessarily shrinking/upscaling it."""
    q = _order_quad(quad)
    tl, tr, br, bl = q
    top = np.linalg.norm(tr - tl)
    bottom = np.linalg.norm(br - bl)
    left = np.linalg.norm(bl - tl)
    right = np.linalg.norm(br - tr)

    out_w = int(round((top + bottom) * 0.5))
    out_h = int(round((left + right) * 0.5))
    out_w = max(600, min(out_w, max_width))
    out_h = max(800, min(out_h, int(round(max_width * 1.95))))

    # Only downscale when the source sheet is genuinely larger than the target.
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(
        frame_bgr,
        matrix,
        (out_w, out_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
    )


def moderate_enhance(image_bgr: np.ndarray) -> np.ndarray:
    """Correct mild phone-camera lighting while preserving original OMR marks."""
    img = image_bgr.copy()

    # Very mild local contrast. No thresholding, denoising or aggressive whitening.
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.05, tileGridSize=(10, 10))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # Preserve fine pen strokes; only a tiny amount of sharpening.
    blur = cv2.GaussianBlur(enhanced, (0, 0), 0.75)
    sharp = cv2.addWeighted(enhanced, 1.05, blur, -0.05, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def process_captured_frame(frame_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    quad = detect_sheet_quad(frame_bgr)
    if quad is None:
        return None, None
    flat = perspective_flatten(frame_bgr, quad)
    return moderate_enhance(flat), quad
