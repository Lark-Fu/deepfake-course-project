#!/usr/bin/env python3
"""Validate configured dataset roots without modifying data."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def is_placeholder(value: object) -> bool:
    return not isinstance(value, str) or value.startswith("<") or value.strip() in {"", "."}


def main() -> int:
    parser = argparse.ArgumentParser(description="Check an externally stored DeepFake dataset path.")
    parser.add_argument("--config", type=Path, default=Path("configs/paths.local.yaml"))
    parser.add_argument("--dataset", choices=("ffpp", "celebdf_v2", "df40"), required=True)
    args = parser.parse_args()
    if not args.config.is_file():
        print(f"Missing local configuration: {args.config}")
        print("Copy configs/paths.yaml to configs/paths.local.yaml and set the dataset path.")
        return 2
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    value = (config.get("datasets") or {}).get(args.dataset)
    if is_placeholder(value):
        print(f"Dataset path for {args.dataset} is not configured.")
        return 2
    root = Path(value).expanduser()
    if not root.exists():
        print(f"Dataset path does not exist: {root}")
        return 1
    image_suffixes = {".jpg", ".jpeg", ".png"}
    video_suffixes = {".mp4", ".avi", ".mov", ".mkv"}
    image_count = sum(1 for path in root.rglob("*") if path.suffix.lower() in image_suffixes)
    video_count = sum(1 for path in root.rglob("*") if path.suffix.lower() in video_suffixes)
    json_count = sum(1 for _ in root.rglob("*.json"))
    print(f"Dataset: {args.dataset}")
    print(f"Path: {root}")
    print(f"Images: {image_count}")
    print(f"Videos: {video_count}")
    print(f"Metadata JSON: {json_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
