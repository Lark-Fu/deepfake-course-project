#!/usr/bin/env python3
"""Export a small, ignored set of UADFV-derived files for classroom demos.

The script never writes to UADFV.  It converts three selected preprocessed frame
sequences to MP4 and makes an in-memory JPEG-70 variant of the chosen fake
sequence.  The selections are tied to the frozen 16-frame baseline documented
in the generated manifest, not a new formal experiment.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2


SELECTIONS = {
    "A_high_confidence_real": {"split": "real", "video_id": "0023"},
    "B_high_confidence_deepfake": {"split": "fake", "video_id": "0020_fake"},
    "C_dual_model_uncertain": {"split": "real", "video_id": "0009"},
}
SINGLE_IMAGE = {"split": "fake", "video_id": "0020_fake", "frame": "056.png"}


def _frames(root: Path, selection: dict[str, str]) -> list[Path]:
    directory = root / selection["split"] / "frames" / selection["video_id"]
    paths = sorted(directory.glob("*.png"))
    if len(paths) < 5:
        raise RuntimeError(f"Expected at least 5 frames in {directory}, found {len(paths)}")
    return paths


def _write_video(paths: list[Path], destination: Path, jpeg70: bool = False) -> None:
    first = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Cannot decode {paths[0]}")
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {destination}")
    try:
        for path in paths:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Cannot decode {path}")
            if jpeg70:
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if not ok:
                    raise RuntimeError(f"JPEG-70 encoding failed for {path}")
                frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            writer.write(frame)
    finally:
        writer.release()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uadfv-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing demo media: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    manifest: dict[str, object] = {
        "purpose": "Classroom Flask demonstration only; not a formal evaluation artifact.",
        "baseline_run": "uadfv_full_20260929",
        "requested_frames_for_selection": 16,
        "items": {},
    }
    for name, selection in SELECTIONS.items():
        paths = _frames(args.uadfv_root, selection)
        output = args.output_dir / f"{name}.mp4"
        _write_video(paths, output)
        manifest["items"][name] = {**selection, "source_frames": len(paths), "file": output.name}
    fake_paths = _frames(args.uadfv_root, SELECTIONS["B_high_confidence_deepfake"])
    degraded = args.output_dir / "D_deepfake_jpeg70.mp4"
    _write_video(fake_paths, degraded, jpeg70=True)
    image_source = args.uadfv_root / SINGLE_IMAGE["split"] / "frames" / SINGLE_IMAGE["video_id"] / SINGLE_IMAGE["frame"]
    if not image_source.is_file():
        raise RuntimeError(f"Missing selected image: {image_source}")
    image_output = args.output_dir / "E_single_deepfake_face.png"
    shutil.copy2(image_source, image_output)
    manifest["items"]["D_deepfake_jpeg70"] = {
        **SELECTIONS["B_high_confidence_deepfake"], "source_frames": len(fake_paths),
        "processing": "Each source frame JPEG encoded/decoded at quality 70 in memory before MP4 writing.",
        "file": degraded.name,
    }
    manifest["items"]["E_single_deepfake_face"] = {**SINGLE_IMAGE, "file": image_output.name}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
