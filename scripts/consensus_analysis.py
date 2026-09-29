#!/usr/bin/env python3
"""Analyze agreement, abstention, and probability gaps from existing 16-frame predictions."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from app.consensus import dual_model_decision


def _commit(value: str | None) -> str:
    if value:
        return value
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary", zero_division=0)
    return {
        "samples": int(len(labels)), "errors": int(np.sum(labels != predictions)), "accuracy": float(np.mean(labels == predictions)),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
    }


def _plot(table: pd.DataFrame, output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(5.6, 5.2), constrained_layout=True)
    for label, name, color in ((0, "Real", "#4472C4"), (1, "Fake", "#C00000")):
        subset = table[table.label == label]
        axis.scatter(subset.xception, subset.effort, label=name, color=color, alpha=0.75, edgecolors="white", linewidths=0.3)
    axis.axvline(0.5, color="black", linestyle="--", linewidth=1)
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("Xception fake probability")
    axis.set_ylabel("Effort fake probability")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.legend(title="True label")
    axis.grid(alpha=0.2)
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"xception_vs_effort_probability.{suffix}", dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--source-commit")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing output directory: {args.output_dir}")
    source = pd.read_csv(args.predictions)
    subset = source[source.requested_frames == 16]
    paired = subset.pivot(index=["video_id", "label"], columns="model", values="probability").reset_index()
    if len(paired) != 98 or not {"xception", "effort"}.issubset(paired.columns) or paired[["xception", "effort"]].isna().any().any():
        raise SystemExit("Expected complete paired 16-frame predictions for 98 videos.")
    records = []
    for row in paired.itertuples(index=False):
        decision = dual_model_decision(float(row.xception), float(row.effort), args.threshold)
        automatic_prediction = 1 if decision["decision"] == "DEEPFAKE" else 0 if decision["decision"] == "REAL" else None
        records.append({
            "video_id": row.video_id, "label": int(row.label), "xception": float(row.xception), "effort": float(row.effort),
            **decision, "automatic_prediction": automatic_prediction,
        })
    result = pd.DataFrame(records)
    agreement = result[result.agreement].copy()
    disagreements = result[~result.agreement].copy()
    agreement_predictions = agreement.automatic_prediction.to_numpy(dtype=int)
    agreement_metrics = _metrics(agreement.label.to_numpy(dtype=int), agreement_predictions)
    summary = {
        "total_videos": len(result), "agreement_count": len(agreement), "disagreement_count": len(disagreements),
        "agreement_rate": float(len(agreement) / len(result)), "disagreement_rate": float(len(disagreements) / len(result)),
        "coverage": float(len(agreement) / len(result)), "selective_accuracy": agreement_metrics["accuracy"],
        "agreement_metrics": agreement_metrics,
        "mean_probability_gap_agreement": float(agreement.probability_gap.mean()),
        "mean_probability_gap_disagreement": float(disagreements.probability_gap.mean()) if len(disagreements) else None,
        "threshold": args.threshold, "frames": 16,
        "policy": "automatic labels only for agreement; disagreements are UNCERTAIN and recommend manual review",
    }
    args.output_dir.mkdir(parents=True)
    result.to_csv(args.output_dir / "consensus_video_analysis.csv", index=False)
    pd.json_normalize([summary], sep=".").to_csv(args.output_dir / "consensus_summary.csv", index=False)
    (args.output_dir / "consensus_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot(result, args.output_dir)
    try:
        import torch
        runtime = {"python": platform.python_version(), "pytorch": torch.__version__, "cuda": torch.version.cuda}
    except Exception:
        runtime = {"python": platform.python_version()}
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "git_commit": _commit(args.source_commit),
        "dataset": "UADFV", "source_predictions": str(args.predictions), "models": ["xception", "effort"],
        "frames": 16, "threshold": args.threshold, "aggregation": "mean(frame_probability)", "runtime": runtime,
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
