#!/usr/bin/env python3
"""Render required robustness figures from a completed degradation summary."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ORDER = ["original", "jpeg70", "resize50"]
LABELS = ["Original", "JPEG Q=70", "Resize 50%"]


def _save(figure: plt.Figure, output: Path) -> None:
    for suffix in ("png", "pdf"):
        figure.savefig(output.with_suffix(f".{suffix}"), dpi=180)
    plt.close(figure)


def _bar(table: pd.DataFrame, column: str, ylabel: str, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    positions = np.arange(len(ORDER))
    width = 0.36
    for offset, model in zip((-width / 2, width / 2), ("xception", "effort")):
        values = [float(table[(table.model == model) & (table.condition == condition)][column].iloc[0]) for condition in ORDER]
        axis.bar(positions + offset, values, width, label=model.title())
    axis.set_xticks(positions, LABELS)
    axis.set_ylabel(ylabel)
    axis.set_ylim(0, 1.05)
    axis.legend(title="Model")
    axis.grid(axis="y", alpha=0.2)
    _save(figure, output)


def _drop(table: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    for model in ("xception", "effort"):
        # The evaluation summary stores a drop (original - condition); display the
        # more intuitive signed change (condition - original) in the figure.
        values = [-float(table[(table.model == model) & (table.condition == condition)]["auroc_drop_from_original"].iloc[0]) for condition in ORDER]
        axis.plot(LABELS, values, marker="o", linewidth=2, label=model.title())
    axis.axhline(0, color="black", linestyle="--", linewidth=1)
    axis.set_ylabel("AUROC change from Original")
    axis.legend(title="Model")
    axis.grid(alpha=0.2)
    _save(figure, output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    table = pd.read_csv(args.summary)
    required = {"model", "condition", "video_metrics.auroc", "video_metrics.average_precision", "auroc_drop_from_original"}
    if missing := required - set(table.columns):
        raise SystemExit(f"Summary is missing columns: {sorted(missing)}")
    targets = [
        args.output_dir / f"{name}.{suffix}"
        for name in ("degradation_auroc", "degradation_ap", "degradation_auroc_change")
        for suffix in ("png", "pdf")
    ]
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise SystemExit(f"Refusing to overwrite existing figures: {existing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _bar(table, "video_metrics.auroc", "Video-level AUROC", args.output_dir / "degradation_auroc")
    _bar(table, "video_metrics.average_precision", "Average Precision", args.output_dir / "degradation_ap")
    _drop(table, args.output_dir / "degradation_auroc_change")
    print(f"Figures: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
