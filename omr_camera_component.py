"""Small Streamlit-WebRTC camera layer for the OMR submission page.

The camera layer only captures a frame and shows a live green document border.
It does not detect answers or change the OMR scanner.
"""
from __future__ import annotations

import base64
import threading
import time
from typing import Optional

try:
    from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
    from av import VideoFrame
    AVAILABLE = True
except Exception:
    AVAILABLE = False

from omr_image_scanner import detect_sheet_quad, draw_detection


if AVAILABLE:
    class OMRVideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.lock = threading.Lock()
            self.detected = False
            self.capture_requested = False
            self.captured = None

        def recv(self, frame: VideoFrame) -> VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            quad = detect_sheet_quad(img)
            with self.lock:
                self.detected = quad is not None
                requested = self.capture_requested
            if requested and quad is not None:
                ok, encoded = __import__("cv2").imencode(
                    ".jpg", img, [__import__("cv2").IMWRITE_JPEG_QUALITY, 96]
                )
                if ok:
                    payload = "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")
                    with self.lock:
                        self.captured = payload
                        self.capture_requested = False
            display = draw_detection(img, quad)
            return VideoFrame.from_ndarray(display, format="bgr24")

        def request_capture(self):
            with self.lock:
                self.capture_requested = True
                self.captured = None

        def get_capture(self):
            with self.lock:
                return self.captured

        def is_detected(self):
            with self.lock:
                return self.detected


def render_live_camera(key: str = "omr_live_camera") -> Optional[dict]:
    if not AVAILABLE:
        return {"error": "Camera support is not installed. Add streamlit-webrtc and av to requirements.txt."}

    ctx = webrtc_streamer(
        key=key,
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=OMRVideoProcessor,
        media_stream_constraints={
            "video": {
                "facingMode": {"ideal": "environment"},
                "width": {"ideal": 1280},
                "height": {"ideal": 1920},
            },
            "audio": False,
        },
        async_processing=True,
    )

    processor = ctx.video_processor
    if processor is None:
        return {"captured": None, "detected": False}

    captured = processor.get_capture()
    if captured:
        return {"captured": captured, "detected": True}

    if st_button := __import__("streamlit").button(
        "📸 Capture OMR", use_container_width=True,
        disabled=not processor.is_detected(), key=f"{key}_capture"
    ):
        processor.request_capture()
        deadline = time.time() + 2.5
        while time.time() < deadline:
            time.sleep(0.08)
            captured = processor.get_capture()
            if captured:
                return {"captured": captured, "detected": True}
        return {"error": "Capture timed out. Keep the full OMR sheet inside the green border and try again."}

    if not processor.is_detected():
        __import__("streamlit").caption("Move the phone until the full OMR sheet is inside the green border.")
    else:
        __import__("streamlit").caption("Sheet detected ✓ — keep the phone steady, then capture.")
    return {"captured": None, "detected": processor.is_detected()}


def omr_camera(key: str = "omr_camera") -> Optional[dict]:
    return render_live_camera(key=key)


def camera_available() -> bool:
    return AVAILABLE
