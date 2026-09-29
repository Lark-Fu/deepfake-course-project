"""Uniform video sampling and score aggregation for the Gradio demo."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.face_processor import FaceProcessor


@dataclass
class VideoFrames:
    total_frames: int
    fps: float
    sampled_indices: list[int]
    faces_rgb: list[np.ndarray]
    timestamps: list[float]


def extract_faces(video_path: str | Path, max_frames: int = 32, processor: FaceProcessor | None = None) -> VideoFrames:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    indices = np.unique(np.linspace(0, max(total - 1, 0), min(max_frames, max(total, 1)), dtype=int)).tolist()
    cropper = processor or FaceProcessor()
    faces: list[np.ndarray] = []
    timestamps: list[float] = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok:
            continue
        face = cropper.crop_largest_face(frame)
        if face is not None:
            faces.append(face)
            timestamps.append(index / fps if fps > 0 else float(index))
    capture.release()
    return VideoFrames(total, fps, indices, faces, timestamps)


def aggregate_probabilities(probabilities: list[float], minimum_valid_faces: int = 5) -> float | None:
    if len(probabilities) < minimum_valid_faces:
        return None
    return float(np.mean(probabilities))
