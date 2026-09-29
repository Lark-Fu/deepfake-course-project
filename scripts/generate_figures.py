#!/usr/bin/env python3
"""Generate report-ready plots from prediction CSVs and metric JSON files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

from metrics_utils import binary_metrics


def save(figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/figures"))
    args = parser.parse_args()
    table = pd.read_csv(args.predictions)
    labels, probabilities = table.label.to_numpy(), table.frame_probability.to_numpy()
    figure, axis = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(labels, probabilities, ax=axis, name="Detector")
    axis.set_title("ROC curve")
    save(figure, args.output_dir / "roc_curve")
    figure, axis = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(labels, probabilities, ax=axis, name="Detector")
    axis.set_title("Precision-Recall curve")
    save(figure, args.output_dir / "pr_curve")
    figure, axis = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(labels, probabilities >= args.threshold, ax=axis, display_labels=["Real", "Fake"])
    axis.set_title("Confusion matrix")
    save(figure, args.output_dir / "confusion_matrix")
    metrics = binary_metrics(labels, probabilities, args.threshold)
    (args.output_dir / "summary_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
