"""Lightweight face cropper for demo inputs; benchmark evaluation uses preprocessed faces."""
from __future__ import annotations

import cv2
import numpy as np


class FaceProcessor:
    def __init__(self, padding_ratio: float = 0.20) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        self.padding_ratio = padding_ratio

    def crop_largest_face(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
        if len(faces) == 0:
            return None
        x, y, width, height = max(faces, key=lambda face: face[2] * face[3])
        padding = int(max(width, height) * self.padding_ratio)
        x0, y0 = max(0, x - padding), max(0, y - padding)
        x1 = min(frame_bgr.shape[1], x + width + padding)
        y1 = min(frame_bgr.shape[0], y + height + padding)
        return cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
