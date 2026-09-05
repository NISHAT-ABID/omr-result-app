"""CamScanner-style preprocessing for OMR images.

This module ONLY prepares an image. It does not read answers.
The existing omr_scanner.py remains responsible for calibration, bubble
coordinates, answer detection, double-touch handling, and scoring.
"""
from __future__ import annotations

from typing import Optional, Tuple
import cv2
import numpy as np

TARGET_ASPECT_MIN = 0.38
TARGET_ASPECT_MAX = 0.82
MIN_AREA_RATIO = 0.14


def _order_quad(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    return np.array([
        pts[np.argmin(s)], pts[np.argmin(d)],
        pts[np.argmax(s)], pts[np.argmax(d)],
    ], dtype=np.float32)


def _quad_score(quad: np.ndarray, frame_shape) -> float:
    h, w = frame_shape[:2]
    area = abs(cv2.contourArea(quad.astype(np.float32)))
    area_ratio = area / float(w * h)
    if area_ratio < MIN_AREA_RATIO:
        return -1.0
    x, y, bw, bh = cv2.boundingRect(quad.astype(np.int32))
    if bw <= 0 or bh <= 0:
        return -1.0
    aspect = min(bw, bh) / max(bw, bh)
    if not (TARGET_ASPECT_MIN <= aspect <= TARGET_ASPECT_MAX):
        return -1.0
    return area_ratio * (1.0 + 0.25 * aspect)


def _candidate_from_contour(cnt: np.ndarray, frame_shape, scale: float, min_area_ratio: float = MIN_AREA_RATIO):
    """Return a scored full-sheet quadrilateral candidate or None."""
    h, w = frame_shape[:2]
    frame_area = float(w * h)
    area = abs(cv2.contourArea(cnt))
    area_ratio = area / frame_area
    if area_ratio < min_area_ratio:
        return None

    peri = cv2.arcLength(cnt, True)
    if peri <= 0:
        return None
    approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
    if len(approx) != 4 or not cv2.isContourConvex(approx):
        return None

    pts = approx.reshape(4, 2).astype(np.float32) / scale
    x, y, bw, bh = cv2.boundingRect(pts.astype(np.int32))
    if bw <= 0 or bh <= 0:
        return None
    aspect = min(bw, bh) / max(bw, bh)
    if not (TARGET_ASPECT_MIN <= aspect <= TARGET_ASPECT_MAX):
        return None

    # Prefer a large, reasonably portrait sheet near the image center.
    cx = (x + bw / 2.0) / max(1.0, w)
    cy = (y + bh / 2.0) / max(1.0, h)
    center_penalty = min(1.0, ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5)
    score = area_ratio * (1.0 + 0.25 * aspect) * (1.0 - 0.35 * center_penalty)
    return score, pts


def detect_sheet_quad(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Detect the complete OMR sheet with a camera-friendly multi-pass detector.

    The live camera often sees a pale blue/white sheet on a darker desk. A
    low-threshold edge pass plus a brightness pass is substantially more stable
    than relying on one fixed Canny threshold.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return None

    h, w = frame_bgr.shape[:2]
    scale = min(1.0, 900.0 / max(h, w))
    small = cv2.resize(frame_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    best = None
    best_score = -1.0

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Pass 1: pale-paper segmentation. This works well for the user's light OMR
    # sheet against a darker table/background and does not alter the captured image.
    for threshold in (125, 140, 155):
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cand = _candidate_from_contour(cnt, small.shape, scale, min_area_ratio=0.12)
            if cand and cand[0] > best_score:
                best_score, best = cand

    # Pass 2: forgiving edge detection for bright/low-contrast scenes.
    for low, high in ((12, 55), (18, 75), (25, 95)):
        edges = cv2.Canny(gray, low, high)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cand = _candidate_from_contour(cnt, small.shape, scale, min_area_ratio=0.12)
            if cand and cand[0] > best_score:
                best_score, best = cand

    return _order_quad(best) if best is not None else None


def draw_detection(frame_bgr: np.ndarray, quad: Optional[np.ndarray]) -> np.ndarray:
    out = frame_bgr.copy()
    if quad is None:
        cv2.putText(out, "Point camera at the full OMR sheet", (24, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        return out
    q = np.round(quad).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(out, [q], True, (50, 230, 170), 6, cv2.LINE_AA)
    for x, y in quad.astype(np.int32):
        cv2.circle(out, (int(x), int(y)), 10, (50, 230, 170), -1, cv2.LINE_AA)
    cv2.putText(out, "OMR detected - keep all 4 corners inside", (24, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.78, (50, 230, 170), 2, cv2.LINE_AA)
    return out


def perspective_flatten(frame_bgr: np.ndarray, quad: np.ndarray, max_width: int = 1400) -> np.ndarray:
    q = _order_quad(quad)
    tl, tr, br, bl = q
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    out_w = max(700, int(round(max(width_top, width_bottom))))
    out_h = max(800, int(round(max(height_left, height_right))))
    if out_w > max_width:
        scale = max_width / float(out_w)
        out_w = int(round(out_w * scale))
        out_h = int(round(out_h * scale))
    out_w = max(600, min(out_w, max_width))
    out_h = max(800, min(out_h, int(max_width * 2.0)))
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(frame_bgr, M, (out_w, out_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)


def moderate_enhance(image_bgr: np.ndarray) -> np.ndarray:
    img = image_bgr.copy()
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.15, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    sharp = cv2.addWeighted(enhanced, 1.08, blur, -0.08, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def process_captured_frame(frame_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    quad = detect_sheet_quad(frame_bgr)
    if quad is None:
        return None, None
    flat = perspective_flatten(frame_bgr, quad)
    flat = moderate_enhance(flat)
    return flat, quad
