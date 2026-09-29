"""Read the lightweight, DeepfakeBench-preprocessed UADFV RGB layout."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class UADFVVideo:
    """One UADFV video represented by its already-cropped RGB frames."""

    video_id: str
    label: int
    frame_paths: tuple[Path, ...]


def _numeric_name(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError:
        return 10**12, path.name


def load_uadfv(root: str | Path) -> list[UADFVVideo]:
    """Load `real/frames/<id>` and `fake/frames/<id>` without mutating data."""
    dataset_root = Path(root)
    videos: list[UADFVVideo] = []
    for class_name, label in (("real", 0), ("fake", 1)):
        frame_root = dataset_root / class_name / "frames"
        if not frame_root.is_dir():
            raise FileNotFoundError(f"Missing UADFV frame directory: {frame_root}")
        for directory in sorted((item for item in frame_root.iterdir() if item.is_dir()), key=lambda item: item.name):
            frames = tuple(sorted(directory.glob("*.png"), key=_numeric_name))
            if len(frames) < 8:
                raise ValueError(f"UADFV video has fewer than 8 usable frames: {directory}")
            videos.append(UADFVVideo(f"{class_name}/{directory.name}", label, frames))
    return videos


def select_uniform_frames(frame_paths: tuple[Path, ...], requested_frames: int) -> tuple[Path, ...]:
    """Select ordered, evenly spaced frames without duplicating any image."""
    if requested_frames <= 0:
        raise ValueError("requested_frames must be positive")
    if len(frame_paths) <= requested_frames:
        return frame_paths
    indices = np.linspace(0, len(frame_paths) - 1, requested_frames, dtype=int)
    return tuple(frame_paths[int(index)] for index in indices)
