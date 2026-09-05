"""Live browser-camera layer for OMR capture.

This module ONLY captures/prepares the image. It does not read answers.
The existing calibration + omr_scanner.read_answers pipeline remains in app.py.
"""
from __future__ import annotations

import base64
import cv2
import streamlit as st
import threading
import time
from typing import Optional

import streamlit as st

try:
    import cv2
    from av import VideoFrame
    from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
    AVAILABLE = True
    IMPORT_ERROR = None
except Exception as exc:  # keep the main app usable if optional camera deps are absent
    cv2 = None
    VideoFrame = None
    VideoProcessorBase = object
    WebRtcMode = None
    webrtc_streamer = None
    AVAILABLE = False
    IMPORT_ERROR = str(exc)

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
                ok, encoded = cv2.imencode(
                    ".jpg", img,
                    [cv2.IMWRITE_JPEG_QUALITY, 96],
                )
                if ok:
                    payload = "data:image/jpeg;base64," + base64.b64encode(
                        encoded.tobytes()
                    ).decode("ascii")
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
    """Render the live camera and return a captured data URI when available."""
    if not AVAILABLE:
        return {
            "error": (
                "Camera support is not installed. Add streamlit-webrtc and av "
                f"to requirements.txt. Import error: {IMPORT_ERROR}"
            )
        }

    st.markdown("**📷 Live OMR Camera**")
    st.caption("Allow camera access, then keep the complete OMR sheet inside the green border.")

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
        st.info("Press **START** above and allow browser camera permission.")
        return {"captured": None, "detected": False}

    captured = processor.get_capture()
    if captured:
        return {"captured": captured, "detected": True}

    detected = processor.is_detected()
    capture_clicked = st.button(
        "📸 Capture OMR",
        use_container_width=True,
        disabled=not detected,
        key=f"{key}_capture",
    )

    if capture_clicked:
        processor.request_capture()
        deadline = time.time() + 3.0
        while time.time() < deadline:
            time.sleep(0.08)
            captured = processor.get_capture()
            if captured:
                return {"captured": captured, "detected": True}
        st.warning("Capture timed out. Keep all 4 corners inside the green border and try again.")

    if not detected:
        st.caption("🟡 Sheet not detected — move the phone until the full OMR sheet is visible.")
    else:
        st.caption("🟢 Sheet detected — keep it steady and press Capture OMR.")

    return {"captured": None, "detected": detected}


def omr_camera(key: str = "omr_camera") -> Optional[dict]:
    return render_live_camera(key=key)


def camera_available() -> bool:
    return AVAILABLE
