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
WARP_WIDTH = 1000
WARP_HEIGHT = 1600


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
    """Score a document quad using geometry and edge support."""
    h, w = frame_shape[:2]
    area = abs(cv2.contourArea(quad.astype(np.float32)))
    area_ratio = area / float(max(1, w * h))
    if area_ratio < MIN_AREA_RATIO or area_ratio > 0.995:
        return -1.0

    q = _order_quad(quad)
    sides = [np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)]
    if min(sides) < min(h, w) * 0.16:
        return -1.0

    # Portrait OMR sheets are expected to be close to 1000:1600.
    page_ratio = ((sides[1] + sides[3]) * 0.5) / max(
        1.0, (sides[0] + sides[2]) * 0.5
    )
    expected = WARP_HEIGHT / float(WARP_WIDTH)
    aspect_err = abs(np.log(max(1e-6, page_ratio / expected)))

    # Reward nearly rectangular quads.
    pts = q.astype(np.float32)
    rectangularity = 0.0
    for i in range(4):
        a = pts[(i - 1) % 4] - pts[i]
        b = pts[(i + 1) % 4] - pts[i]
        denom = max(1e-6, np.linalg.norm(a) * np.linalg.norm(b))
        rectangularity += abs(float(np.dot(a, b))) / denom
    rectangularity = 1.0 - rectangularity / 4.0

    return area_ratio * 5.0 + rectangularity * 1.5 - aspect_err * 1.8


def _line_based_quad(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Recover the paper from long page-edge lines when the outer contour is broken."""
    h, w = frame_bgr.shape[:2]
    scale = min(1.0, 1100.0 / max(h, w))
    small = cv2.resize(frame_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 35, 130)
    min_len = int(min(small.shape[:2]) * 0.28)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180.0, threshold=max(45, int(min(small.shape[:2]) * 0.07)),
        minLineLength=min_len, maxLineGap=max(20, int(min(small.shape[:2]) * 0.05))
    )
    if lines is None:
        return None

    vertical, horizontal = [], []
    for item in lines[:, 0]:
        x1, y1, x2, y2 = map(float, item)
        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy))
        if length < min_len:
            continue
        angle = abs(np.degrees(np.arctan2(dy, dx))) % 180.0
        if angle < 18 or angle > 162:
            horizontal.append((x1, y1, x2, y2, length))
        elif 72 < angle < 108:
            vertical.append((x1, y1, x2, y2, length))

    if len(horizontal) < 2 or len(vertical) < 2:
        return None

    def cluster(values, tolerance):
        groups = []
        for v in sorted(values):
            if not groups or abs(v - np.mean(groups[-1])) > tolerance:
                groups.append([v])
            else:
                groups[-1].append(v)
        return [float(np.mean(g)) for g in groups]

    # Use line midpoints. We want two separated horizontal and two separated
    # vertical page edges, not internal OMR rows.
    hs = cluster([0.5 * (y1 + y2) for x1, y1, x2, y2, _ in horizontal], small.shape[0] * 0.06)
    vs = cluster([0.5 * (x1 + x2) for x1, y1, x2, y2, _ in vertical], small.shape[1] * 0.06)

    if len(hs) < 2 or len(vs) < 2:
        return None

    # Prefer the outermost separated pair, but reject an implausibly small box.
    top, bottom = min(hs), max(hs)
    left, right = min(vs), max(vs)
    if (bottom - top) < small.shape[0] * 0.45 or (right - left) < small.shape[1] * 0.25:
        return None

    q = np.array([[left, top], [right, top], [right, bottom], [left, bottom]], dtype=np.float32)
    q /= scale
    return _order_quad(q)


def detect_sheet_quad(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Find the OMR sheet using contour hypotheses plus a line fallback.

    The sheet design is fixed, so geometry is intentionally preferred over
    generic object detection.  This is more tolerant of shadows/background
    than the old single-outer-contour approach.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return None

    h, w = frame_bgr.shape[:2]
    scale = min(1.0, 1100.0 / max(h, w))
    small = cv2.resize(frame_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    candidates = []

    for lo, hi, close_frac in ((30, 100, 0.006), (45, 135, 0.004), (65, 180, 0.003)):
        edges = cv2.Canny(gray, lo, hi)
        k = max(3, int(round(min(small.shape[:2]) * close_frac)))
        if k % 2 == 0:
            k += 1
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8), iterations=2)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:40]:
            area = cv2.contourArea(cnt)
            if area < 0.18 * small.shape[0] * small.shape[1]:
                continue
            peri = cv2.arcLength(cnt, True)
            if peri <= 0:
                continue
            for eps in (0.010, 0.016, 0.024, 0.035):
                approx = cv2.approxPolyDP(cnt, eps * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    q = approx.reshape(4, 2).astype(np.float32) / scale
                    score = _quad_score(q, frame_bgr.shape)
                    if score > 0:
                        candidates.append((score, q))
                    break

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return _order_quad(candidates[0][1])

    return _line_based_quad(frame_bgr)


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


def four_point_transform(image_bgr: np.ndarray, points: np.ndarray,
                        width: int = WARP_WIDTH, height: int = WARP_HEIGHT) -> np.ndarray:
    """Perspective-correct an OMR sheet to the fixed 1000x1600 master canvas."""
    q = _order_quad(points)
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(
        image_bgr,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def perspective_flatten(frame_bgr: np.ndarray, quad: np.ndarray,
                        width: int = WARP_WIDTH, height: int = WARP_HEIGHT) -> np.ndarray:
    """Backward-compatible wrapper around the fixed-size four-point transform."""
    return four_point_transform(frame_bgr, quad, width=width, height=height)


def detect_and_warp(image_bgr: np.ndarray):
    """Detect the document contour and flatten it to the canonical canvas."""
    quad = detect_sheet_quad(image_bgr)
    if quad is None:
        return None, False
    return four_point_transform(image_bgr, quad), True


def preprocess_omr_image(image_bgr: np.ndarray) -> np.ndarray:
    """Remove red/pink print and compensate for uneven phone-camera lighting.

    OpenCV stores BGR, so channel 2 is the Red channel.  Student ink is then
    represented by dark pixels while the red/pink printed border is largely
    suppressed. Adaptive thresholding makes the result substantially less
    sensitive to shadows and mild illumination gradients.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty OMR image.")

    red = image_bgr[:, :, 2]
    red = cv2.GaussianBlur(red, (3, 3), 0)

    binary = cv2.adaptiveThreshold(
        red,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        9,
    )

    # Tiny isolated noise is removed without eating normal pen strokes.
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1
    )
    return binary


def process_for_omr_reading(image_bgr: np.ndarray):
    """Return (flat_color_image, binary_omr_image, detected_quad)."""
    quad = detect_sheet_quad(image_bgr)
    if quad is None:
        return None, None, None
    flat = four_point_transform(image_bgr, quad)
    binary = preprocess_omr_image(flat)
    return flat, binary, quad

def moderate_enhance(image_bgr: np.ndarray) -> np.ndarray:
    """Create a clean scan preview while preserving the printed OMR artwork.

    This is display/preview processing only.  The raw flattened image is still
    used for answer reading so enhancement cannot manufacture MULTI answers.
    A low-frequency illumination field is removed first, which helps with
    phone-camera shadows and bright/dark corners.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty OMR image.")

    img = image_bgr.astype(np.float32)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Estimate broad illumination only; bubble outlines and handwriting are
    # much smaller than this blur and are therefore not treated as shadows.
    illumination = cv2.GaussianBlur(l.astype(np.float32), (0, 0), 45.0)
    base = float(np.median(illumination))
    normalized_l = l.astype(np.float32) * (base / np.maximum(illumination, 1.0))
    normalized_l = np.clip(normalized_l, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=1.20, tileGridSize=(8, 8))
    normalized_l = clahe.apply(normalized_l)

    enhanced = cv2.cvtColor(cv2.merge((normalized_l, a, b)), cv2.COLOR_LAB2BGR)

    # Very mild sharpening for camera softness.
    blur = cv2.GaussianBlur(enhanced, (0, 0), 0.9)
    sharp = cv2.addWeighted(enhanced, 1.10, blur, -0.10, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def process_captured_frame(frame_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (processed_flat_image, detected_quad) for camera capture."""
    quad = detect_sheet_quad(frame_bgr)
    if quad is None:
        return None, None
    flat = four_point_transform(frame_bgr, quad)
    return moderate_enhance(flat), quad


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
