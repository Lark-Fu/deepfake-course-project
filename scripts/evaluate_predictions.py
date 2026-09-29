#!/usr/bin/env python3
"""Calculate frame and video metrics from an auditable prediction CSV."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics_utils import binary_metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    table = pd.read_csv(args.predictions)
    required = {"label", "frame_probability", "video_path"}
    missing = required - set(table.columns)
    if missing:
        raise SystemExit(f"Missing CSV columns: {sorted(missing)}")
    frame = binary_metrics(table.label.to_numpy(), table.frame_probability.to_numpy(), args.threshold)
    video = table.groupby("video_path", as_index=False).agg(label=("label", "first"), probability=("frame_probability", "mean"))
    video_metrics = binary_metrics(video.label.to_numpy(), video.probability.to_numpy(), args.threshold)
    result = {"frame": frame, "video": video_metrics, "samples": {"frames": len(table), "videos": len(video)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
