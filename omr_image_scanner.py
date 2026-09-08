"""OMR Image Scanner - CamScanner style preprocessing."""
from __future__ import annotations
import threading
from typing import Optional, Tuple
import cv2
import numpy as np

TARGET_ASPECT_MIN = 0.38
TARGET_ASPECT_MAX = 0.78
MIN_AREA_RATIO = 0.18
WARP_WIDTH = 1000
WARP_HEIGHT = 1600


def _order_quad(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def _quad_score(quad: np.ndarray, frame_shape) -> float:
    h, w = frame_shape[:2]
    area = abs(cv2.contourArea(quad.astype(np.float32)))
    area_ratio = area / float(w * h)
    if area_ratio < MIN_AREA_RATIO: return -1.0
    x, y, bw, bh = cv2.boundingRect(quad.astype(np.int32))
    if bw <= 0 or bh <= 0: return -1.0
    aspect = min(bw, bh) / max(bw, bh)
    if not (TARGET_ASPECT_MIN <= aspect <= TARGET_ASPECT_MAX): return -1.0
    return area_ratio * (1.0 + 0.25 * aspect)


def detect_sheet_quad(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    if frame_bgr is None or frame_bgr.size == 0: return None
    h, w = frame_bgr.shape[:2]
    scale = min(1.0, 900.0 / max(h, w))
    small = cv2.resize(frame_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # FIXED: Reduced morphology kernel size to prevent destroying sheet borders
    edges = cv2.Canny(gray, 40, 140)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = -1.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.10 * small.shape[0] * small.shape[1]: continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx): continue
        q = approx.reshape(4, 2).astype(np.float32) / scale
        score = _quad_score(q, frame_bgr.shape)
        if score > best_score:
            best_score = score
            best = q
    return _order_quad(best) if best is not None else None


def draw_detection(frame_bgr: np.ndarray, quad: Optional[np.ndarray]) -> np.ndarray:
    out = frame_bgr.copy()
    if quad is None:
        cv2.putText(out, "Point camera at the full OMR sheet", (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        return out
    q = np.round(quad).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(out, [q], True, (50, 230, 170), 6, cv2.LINE_AA)
    for i, (x, y) in enumerate(quad.astype(np.int32)):
        cv2.circle(out, (int(x), int(y)), 10, (50, 230, 170), -1, cv2.LINE_AA)
    cv2.putText(out, "OMR detected - keep all 4 corners inside", (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (50, 230, 170), 2, cv2.LINE_AA)
    return out


def four_point_transform(image_bgr: np.ndarray, points: np.ndarray, width: int = WARP_WIDTH, height: int = WARP_HEIGHT) -> np.ndarray:
    q = _order_quad(points)
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(image_bgr, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def perspective_flatten(frame_bgr: np.ndarray, quad: np.ndarray, width: int = WARP_WIDTH, height: int = WARP_HEIGHT) -> np.ndarray:
    return four_point_transform(frame_bgr, quad, width=width, height=height)


def detect_and_warp(image_bgr: np.ndarray):
    quad = detect_sheet_quad(image_bgr)
    if quad is None: return None, False
    return four_point_transform(image_bgr, quad), True


def preprocess_omr_image_binary(image_bgr: np.ndarray) -> np.ndarray:
    """Renamed to avoid conflict with omr_scanner.py"""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty OMR image.")
    red = image_bgr[:, :, 2]
    red = cv2.GaussianBlur(red, (3, 3), 0)
    binary = cv2.adaptiveThreshold(red, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    binary = cv2.medianBlur(binary, 3)
    return binary


def process_for_omr_reading(image_bgr: np.ndarray):
    quad = detect_sheet_quad(image_bgr)
    if quad is None: return None, None, None
    flat = four_point_transform(image_bgr, quad)
    binary = preprocess_omr_image_binary(flat)
    return flat, binary, quad


def moderate_enhance(image_bgr: np.ndarray) -> np.ndarray:
    img = image_bgr.copy()
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.35, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(enhanced, (0, 0), 1.15)
    sharp = cv2.addWeighted(enhanced, 1.16, blur, -0.16, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def process_captured_frame(frame_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    quad = detect_sheet_quad(frame_bgr)
    if quad is None: return None, None
    flat = four_point_transform(frame_bgr, quad)
    flat = moderate_enhance(flat)
    return flat, quad

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
                if self.latest_processed is None: return None
                return self.latest_processed.copy()

        def has_detection(self):
            with self.lock:
                return bool(self.detected and self.latest_processed is not None)

    def render_live_camera(key: str = "omr_live_camera") -> Optional[np.ndarray]:
        if not _WEBRTC_AVAILABLE: return None
        ctx = webrtc_streamer(key=key, mode=WebRtcMode.SENDRECV, video_processor_factory=OMRVideoProcessor, media_stream_constraints={"video": {"facingMode": {"ideal": "environment"}, "width": {"ideal": 1280}, "height": {"ideal": 1920}}, "audio": False}, async_processing=True)
        if ctx.state.playing and ctx.video_processor is not None:
            detected = ctx.video_processor.has_detection()
        return ctx.video_processor.get_latest_processed() if ctx.video_processor is not None else None

    def camera_available() -> bool:
        return _WEBRTC_AVAILABLE
