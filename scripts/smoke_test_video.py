#!/usr/bin/env python3
"""Exercise the local video sampling path without using course datasets."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from skimage import data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.video_processor import aggregate_probabilities, extract_faces


def main() -> int:
    output = PROJECT_ROOT / "runtime_media" / "smoke_astronaut.avi"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (frame.shape[1], frame.shape[0]))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create the temporary smoke-test video.")
    for _ in range(10):
        writer.write(frame)
    writer.release()

    extracted = extract_faces(output, max_frames=8)
    if len(extracted.faces_rgb) != 8:
        raise RuntimeError(f"Expected 8 detected faces, got {len(extracted.faces_rgb)}.")
    aggregate = aggregate_probabilities([0.1] * len(extracted.faces_rgb))
    if aggregate != 0.1:
        raise RuntimeError(f"Unexpected aggregate score: {aggregate}")
    print(f"Video: {output}")
    print(f"Frames: total={extracted.total_frames}, sampled={len(extracted.sampled_indices)}, faces={len(extracted.faces_rgb)}")
    print(f"Aggregate: {aggregate:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
