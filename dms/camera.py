from __future__ import annotations

import sys
from typing import List, Optional


def list_camera_devices() -> List[str]:
    if not sys.platform.startswith("win"):
        return []
    try:
        from pygrabber.dshow_graph import FilterGraph
    except Exception:
        return []
    try:
        graph = FilterGraph()
        return graph.get_input_devices()
    except Exception:
        return []


def find_camera_index(name: str) -> Optional[int]:
    name_lower = name.lower()
    devices = list_camera_devices()
    for idx, device in enumerate(devices):
        if name_lower in device.lower():
            return idx
    return None
