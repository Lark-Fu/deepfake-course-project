#!/usr/bin/env python3
"""Validate the preprocessed UADFV layout without modifying data."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def is_placeholder(value: object) -> bool:
    return not isinstance(value, str) or value.startswith("<") or value.strip() in {"", "."}


def main() -> int:
    parser = argparse.ArgumentParser(description="Check an externally stored DeepFake dataset path.")
    parser.add_argument("--config", type=Path, default=Path("configs/paths.local.yaml"))
    parser.add_argument("--dataset", choices=("uadfv",), default="uadfv")
    args = parser.parse_args()
    if not args.config.is_file():
        print(f"Missing local configuration: {args.config}")
        print("Copy configs/paths.yaml to configs/paths.local.yaml and set the dataset path.")
        return 2
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    value = config.get("uadfv_root")
    if is_placeholder(value):
        print(f"Dataset path for {args.dataset} is not configured.")
        return 2
    root = Path(value).expanduser()
    if not root.exists():
        print(f"Dataset path does not exist: {root}")
        return 1
    class_dirs = {label: root / label / "frames" for label in ("real", "fake")}
    if not all(path.is_dir() for path in class_dirs.values()):
        print("Expected DeepfakeBench RGB layout: real/frames/<video>/ and fake/frames/<video>/.")
        return 1
    counts = {label: len([item for item in path.iterdir() if item.is_dir()]) for label, path in class_dirs.items()}
    png_counts = {label: sum(1 for item in path.rglob("*.png")) for label, path in class_dirs.items()}
    total = sum(counts.values())
    print("Dataset: UADFV")
    print(f"Path: {root}")
    print(f"Real videos: {counts['real']}")
    print(f"Fake videos: {counts['fake']}")
    print(f"Total videos: {total}")
    print(f"PNG frames: real={png_counts['real']}, fake={png_counts['fake']}, total={sum(png_counts.values())}")
    if counts != {"real": 49, "fake": 49}:
        print("UADFV validation failed: expected 49 Real + 49 Fake videos.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
