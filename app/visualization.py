"""Small visualization helpers shared by the demo and report scripts."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def probability_curve(timestamps: list[float], probabilities: list[float], title: str = "Frame-level fake probability"):
    figure, axis = plt.subplots(figsize=(8, 3.2), dpi=140)
    axis.plot(timestamps, probabilities, marker="o", linewidth=1.5, markersize=3)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Fake probability")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    return figure


def save_figure(figure, output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
