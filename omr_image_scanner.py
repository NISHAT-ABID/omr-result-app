"""Camera OMR preprocessing.

The camera component captures the phone's native-resolution frame.  This module
finds the sheet, removes the surrounding desk/background with a perspective
warp, and applies only gentle lighting/contrast correction.  OMR answer reading
and calibration remain in omr_scanner.py.

FIX NOTES (this version):
- Detection thresholds were too strict for real hand-held phone photos
  (bent/angled sheets, sheet not filling the whole frame, non-ideal
  lighting), so process_captured_frame() was returning (None, None) far
  too often and blocking the user with "boundary could not be confirmed".
- Thresholds below are relaxed to accept a wider range of real photos.
- process_captured_frame() now NEVER returns (None, None). If a clean
  quad can't be found, it falls back to using the full (enhanced, but
  not perspective-warped) frame, exactly like the normal upload flow
  already does when detect_and_warp() fails there. This means capture
  can no longer get permanently stuck - worst case, the student just
  calibrates on a slightly less "flattened" photo, same as an uploaded
  photo would be.
"""
from __future__ import annotations
from typing import Optional, Tuple
import cv2
import numpy as np

# Relaxed from (0.40, 0.82). A hand-held photo of an A4/letter sheet at a
# mild angle can easily fall outside the old tight range.
TARGET_ASPECT_MIN = 0.28
TARGET_ASPECT_MAX = 0.92

# Relaxed from 0.10. Lower area floor so a sheet that doesn't fill the
# entire frame (common when a student backs up to fit the whole sheet in)
# still gets picked up.
MIN_AREA_RATIO = 0.05


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
    """Find the OMR paper while rejecting most desk/background rectangles.

    Returns None if nothing plausible is found - callers must handle this
    (process_captured_frame() below does, by falling back to the full
    frame instead of hard-failing).
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return None

    h, w = frame_bgr.shape[:2]
    scale = min(1.0, 1400.0 / max(h, w))
    small = cv2.resize(frame_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Wider Canny range (was 25,105) catches fainter/lower-contrast edges
    # from phone cameras in mediocre lighting.
    edges = cv2.Canny(gray, 15, 90)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)

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
        # Slightly more tolerant polygon approximation (was 0.028) so a
        # gently curved/bent sheet edge still simplifies to 4 points.
        approx = cv2.approxPolyDP(cnt, 0.035 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        q_small = approx.reshape(4, 2).astype(np.float32)
        x, y, bw, bh = cv2.boundingRect(q_small.astype(np.int32))
        # Relaxed from (0.30, 0.45) - sheet no longer has to fill most of
        # the frame.
        if bw < sw * 0.20 or bh < sh * 0.30:
            continue

        score = _quad_quality(q_small, sw, sh, area)
        if score > best_score:
            best_score = score
            best = q_small / scale

    if best is not None:
        return _order_quad(best)

    # Fallback: the paper is usually the largest bright, coherent region.
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    # Widened brightness/saturation range (was V>=125, S<=105) to catch
    # sheets under warmer/dimmer indoor lighting or slight camera shadows.
    bright = cv2.inRange(hsv, np.array([0, 0, 95], np.uint8), np.array([179, 130, 255], np.uint8))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8), iterations=2)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8), iterations=1)
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
        area = float(cv2.contourArea(cnt))
        if area < 0.08 * total:  # relaxed from 0.16
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = min(bw, bh) / max(bw, bh, 1)
        if not (TARGET_ASPECT_MIN <= aspect <= TARGET_ASPECT_MAX):
            continue
        hull = cv2.convexHull(cnt)
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.04 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            q = approx.reshape(4, 2).astype(np.float32) / scale
            return _order_quad(q)
        # Even if approxPolyDP doesn't collapse to exactly 4 points, the
        # bounding rect of a large bright blob is still a usable fallback
        # quad rather than giving up entirely.
        if bw > sw * 0.35 and bh > sh * 0.40:
            box = np.array(
                [[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]],
                dtype=np.float32,
            ) / scale
            return _order_quad(box)

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

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.05, tileGridSize=(10, 10))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    blur = cv2.GaussianBlur(enhanced, (0, 0), 0.75)
    sharp = cv2.addWeighted(enhanced, 1.05, blur, -0.05, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def _resize_max_dim(image_bgr: np.ndarray, max_dim: int = 1600) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    scale = min(1.0, max_dim / float(max(h, w)))
    if scale >= 1.0:
        return image_bgr
    return cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def process_captured_frame(frame_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Process one captured camera frame for OMR calibration/scanning.

    IMPORTANT: this now never returns (None, None) for a non-empty input
    frame. If the sheet's 4 corners can't be confidently found, it falls
    back to returning the enhanced-but-not-warped full frame (same
    approach the normal upload path already uses when its own
    detect_and_warp() fails) - so the student is never blocked from
    proceeding to calibration. `quad` is None in that fallback case,
    which callers can use to show a "couldn't auto-detect corners,
    please make sure all 4 corners are visible" warning instead of a
    hard error.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return None, None

    quad = detect_sheet_quad(frame_bgr)
    if quad is not None:
        try:
            flat = perspective_flatten(frame_bgr, quad)
            return moderate_enhance(flat), quad
        except Exception:
            # Perspective warp can fail on a degenerate quad (e.g. near-
            # zero side length) - fall through to the no-warp fallback
            # below rather than giving up entirely.
            pass

    # Fallback: no confident quad found (or warp failed) - still return a
    # usable, lightly-enhanced photo instead of blocking the student.
    fallback = _resize_max_dim(frame_bgr)
    return moderate_enhance(fallback), None
