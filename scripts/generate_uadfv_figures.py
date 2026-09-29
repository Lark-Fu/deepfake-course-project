#!/usr/bin/env python3
"""Render the two required UADFV sampling-efficiency figures from a completed run."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _plot(table: pd.DataFrame, value: str, ylabel: str, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    for model, group in table.groupby("model"):
        group = group.sort_values("requested_frames")
        axis.plot(group["requested_frames"], group[value], marker="o", linewidth=2, label=model.title())
    axis.set_xlabel("Uniformly sampled frames per video")
    axis.set_ylabel(ylabel)
    axis.set_xticks([8, 16, 32])
    axis.grid(alpha=0.25)
    axis.legend(title="Model")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    table = pd.read_csv(args.summary)
    required = {"model", "requested_frames", "video_metrics.auroc", "seconds_per_video"}
    missing = required - set(table.columns)
    if missing:
        raise SystemExit(f"Summary is missing columns: {sorted(missing)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _plot(table, "video_metrics.auroc", "Video-level AUROC", args.output_dir / "frames_vs_auroc.png")
    _plot(table, "seconds_per_video", "Inference time per video (s)", args.output_dir / "frames_vs_inference_time.png")
    print(f"Figures: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
