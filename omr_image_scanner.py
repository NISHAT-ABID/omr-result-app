"""
OMR Image Scanner
-----------------
A lightweight CamScanner-style preprocessing layer for The Med Venture.

It does NOT read OMR answers. Its only job is to:
  1) detect the OMR sheet in a live camera frame,
  2) show a live border around it,
  3) perspective-correct the sheet when captured,
  4) apply moderate illumination/contrast/sharpness normalization.

The existing omr_scanner.py remains responsible for calibration and answer reading.
"""

from __future__ import annotations

import threading
from typing import Optional, Tuple

import cv2
import numpy as np


# Target is deliberately close to the physical portrait OMR geometry used by the app.
TARGET_ASPECT_MIN = 0.38
TARGET_ASPECT_MAX = 0.78
MIN_AREA_RATIO = 0.18


def _order_quad(points: np.ndarray) -> np.ndarray:
    """Return 4 points in TL, TR, BR, BL order."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    return np.array(
        [
            pts[np.argmin(s)],
            pts[np.argmin(d)],
            pts[np.argmax(s)],
            pts[np.argmax(d)],
        ],
        dtype=np.float32,
    )


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

    # Prefer large rectangles, while rewarding portrait/document-like geometry.
    return area_ratio * (1.0 + 0.25 * aspect)


def detect_sheet_quad(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Find the most plausible large rectangular OMR/document contour."""
    if frame_bgr is None or frame_bgr.size == 0:
        return None

    h, w = frame_bgr.shape[:2]
    scale = min(1.0, 900.0 / max(h, w))
    small = cv2.resize(frame_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edges catch the printed outer sheet edge even when the paper is light.
    edges = cv2.Canny(gray, 35, 115)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = -1.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.10 * small.shape[0] * small.shape[1]:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        q = approx.reshape(4, 2).astype(np.float32) / scale
        score = _quad_score(q, frame_bgr.shape)
        if score > best_score:
            best_score = score
            best = q

    return _order_quad(best) if best is not None else None


def draw_detection(frame_bgr: np.ndarray, quad: Optional[np.ndarray]) -> np.ndarray:
    """Draw a clear live green document boundary without altering the frame geometry."""
    out = frame_bgr.copy()
    if quad is None:
        cv2.putText(
            out,
            "Point camera at the full OMR sheet",
            (24, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return out

    q = np.round(quad).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(out, [q], True, (50, 230, 170), 6, cv2.LINE_AA)
    for i, (x, y) in enumerate(quad.astype(np.int32)):
        cv2.circle(out, (int(x), int(y)), 10, (50, 230, 170), -1, cv2.LINE_AA)
    cv2.putText(
        out,
        "OMR detected - keep all 4 corners inside",
        (24, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (50, 230, 170),
        2,
        cv2.LINE_AA,
    )
    return out


def perspective_flatten(frame_bgr: np.ndarray, quad: np.ndarray, max_width: int = 1100) -> np.ndarray:
    """Warp the detected sheet into a straight portrait document."""
    q = _order_quad(quad)
    tl, tr, br, bl = q

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)

    out_w = max(600, int(round(max(width_top, width_bottom))))
    out_h = max(800, int(round(max(height_left, height_right))))

    # Keep portrait OMR readable without producing an unnecessarily huge image.
    if out_w > max_width:
        scale = max_width / float(out_w)
        out_w = int(round(out_w * scale))
        out_h = int(round(out_h * scale))

    # Guard against extreme or noisy contour dimensions.
    out_w = max(600, min(out_w, max_width))
    out_h = max(800, min(out_h, int(max_width * 2.0)))

    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(frame_bgr, M, (out_w, out_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def moderate_enhance(image_bgr: np.ndarray) -> np.ndarray:
    """Moderate document enhancement designed to preserve OMR bubble geometry."""
    img = image_bgr.copy()

    # Work mostly on luminance; do not aggressively threshold the bubbles.
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Local contrast, deliberately restrained.
    clahe = cv2.createCLAHE(clipLimit=1.35, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # Gentle unsharp mask. This helps mild camera softness without turning printed circles into blobs.
    blur = cv2.GaussianBlur(enhanced, (0, 0), 1.15)
    sharp = cv2.addWeighted(enhanced, 1.16, blur, -0.16, 0)

    # Keep output in normal 8-bit range.
    return np.clip(sharp, 0, 255).astype(np.uint8)


def process_captured_frame(frame_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (processed_flat_image, detected_quad)."""
    quad = detect_sheet_quad(frame_bgr)
    if quad is None:
        return None, None
    flat = perspective_flatten(frame_bgr, quad)
    flat = moderate_enhance(flat)
    return flat, quad


# ---------------------------------------------------------------------------
# Optional live camera integration. Requires streamlit-webrtc + av.
# ---------------------------------------------------------------------------
try:
    from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
    from av import VideoFrame

    _WEBRTC_AVAILABLE = True
except Exception:
    _WEBRTC_AVAILABLE = False


if _WEBRTC_AVAILABLE:
    class OMRVideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.lock = threading.Lock()
            self.latest_processed = None
            self.detected = False

        def recv(self, frame: VideoFrame) -> VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            quad = detect_sheet_quad(img)
            display = draw_detection(img, quad)

            if quad is not None:
                flat = perspective_flatten(img, quad)
                flat = moderate_enhance(flat)
                with self.lock:
                    self.latest_processed = flat
                    self.detected = True
            else:
                with self.lock:
                    self.detected = False

            return VideoFrame.from_ndarray(display, format="bgr24")

        def get_latest_processed(self):
            with self.lock:
                if self.latest_processed is None:
                    return None
                return self.latest_processed.copy()

        def has_detection(self):
            with self.lock:
                return bool(self.detected and self.latest_processed is not None)


def render_live_camera(key: str = "omr_live_camera") -> Optional[np.ndarray]:
    """Render live OMR detection and return a processed image after capture."""
    if not _WEBRTC_AVAILABLE:
        return None

    ctx = webrtc_streamer(
        key=key,
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=OMRVideoProcessor,
        media_stream_constraints={"video": {"facingMode": {"ideal": "environment"}, "width": {"ideal": 1280}, "height": {"ideal": 1920}}, "audio": False},
        async_processing=True,
    )

    if ctx.state.playing and ctx.video_processor is not None:
        detected = ctx.video_processor.has_detection()
        if detected:
            st_message = ""
        else:
            st_message = ""

    return ctx.video_processor.get_latest_processed() if ctx.video_processor is not None else None


def camera_available() -> bool:
    return _WEBRTC_AVAILABLE
