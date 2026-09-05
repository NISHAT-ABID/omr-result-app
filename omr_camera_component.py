from pathlib import Path
import streamlit.components.v1 as components

_COMPONENT_DIR = Path(__file__).parent / "omr_camera_component"
_omr_camera = components.declare_component("omr_camera", path=str(_COMPONENT_DIR))

def omr_camera(key="omr_camera"):
    return _omr_camera(key=key, default=None)
