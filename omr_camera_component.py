"""CamScanner-style browser camera for OMR capture.

This component keeps the camera/video in the browser, so camera frames are not
streamed through the Streamlit server. The browser performs lightweight live
sheet-boundary detection and sends only the captured still image to Python.
Python then runs the existing OpenCV crop/flatten/enhancement pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import streamlit.components.v1 as components

_COMPONENT_DIR = Path(__file__).parent / "omr_camera_frontend"
_native_camera = components.declare_component(
    "omr_camscanner_camera",
    path=str(_COMPONENT_DIR),
)


def omr_camera(key: str = "omr_camera") -> Optional[dict]:
    """Render the browser camera and return a captured image payload."""
    return _native_camera(key=key, default=None)


def camera_available() -> bool:
    return True
