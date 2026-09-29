"""In-memory image degradations used for the UADFV robustness experiment."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


CONDITIONS = ("original", "jpeg70", "resize50")


def load_degraded_rgb(path: str | Path, condition: str) -> np.ndarray:
    """Read one PNG and apply a documented degradation without writing an output file."""
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Cannot read image: {path}")
    if condition == "original":
        result = image_bgr
    elif condition == "jpeg70":
        ok, encoded = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            raise RuntimeError(f"JPEG encoding failed: {path}")
        result = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    elif condition == "resize50":
        height, width = image_bgr.shape[:2]
        reduced = cv2.resize(image_bgr, (max(1, width // 2), max(1, height // 2)), interpolation=cv2.INTER_AREA)
        result = cv2.resize(reduced, (width, height), interpolation=cv2.INTER_CUBIC)
    else:
        raise ValueError(f"Unknown degradation condition: {condition}")
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
