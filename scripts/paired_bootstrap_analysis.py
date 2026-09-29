#!/usr/bin/env python3
"""Paired bootstrap comparison of frozen Xception and Effort UADFV predictions."""
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
from sklearn.metrics import average_precision_score, roc_auc_score


def _commit(value: str | None) -> str:
    if value:
        return value
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _paired_table(table: pd.DataFrame, frames: int) -> pd.DataFrame:
    subset = table[table.requested_frames == frames]
    wide = subset.pivot(index=["video_id", "label"], columns="model", values="probability").reset_index()
    required = {"xception", "effort"}
    if len(wide) != 98 or not required.issubset(wide.columns) or wide[list(required)].isna().any().any():
        raise ValueError(f"Frame setting {frames} is not a complete 98-video Xception/Effort pairing.")
    return wide.sort_values("video_id").reset_index(drop=True)


def _distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p2_5": float(np.percentile(values, 2.5)),
        "p97_5": float(np.percentile(values, 97.5)),
    }


def _plot(samples: pd.DataFrame, metric: str, output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True, sharey=True)
    for axis, frames in zip(axes, (8, 16, 32)):
        values = samples.loc[samples.requested_frames == frames, f"delta_{metric}"]
        axis.hist(values, bins=40, color="#4472C4", alpha=0.85)
        axis.axvline(0, color="#C00000", linestyle="--", linewidth=1.5)
        axis.set_title(f"{frames} frames")
        axis.set_xlabel(f"Effort − Xception {metric.upper()}")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Bootstrap count")
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"delta_{metric}_bootstrap_distribution.{suffix}", dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--source-commit")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing output directory: {args.output_dir}")
    table = pd.read_csv(args.predictions)
    required = {"model", "requested_frames", "video_id", "label", "probability"}
    if missing := required - set(table.columns):
        raise SystemExit(f"Prediction CSV is missing columns: {sorted(missing)}")
    args.output_dir.mkdir(parents=True)
    rng = np.random.default_rng(args.seed)
    summaries: list[dict] = []
    all_samples: list[pd.DataFrame] = []
    for frames in (8, 16, 32):
        paired = _paired_table(table, frames)
        labels = paired.label.to_numpy(dtype=int)
        xception = paired.xception.to_numpy(dtype=float)
        effort = paired.effort.to_numpy(dtype=float)
        samples: list[tuple[float, float, float, float, float, float]] = []
        attempts = 0
        while len(samples) < args.resamples:
            attempts += 1
            indices = rng.integers(0, len(labels), size=len(labels))
            sampled_labels = labels[indices]
            if np.unique(sampled_labels).size != 2:
                continue
            x_auc = roc_auc_score(sampled_labels, xception[indices])
            e_auc = roc_auc_score(sampled_labels, effort[indices])
            x_ap = average_precision_score(sampled_labels, xception[indices])
            e_ap = average_precision_score(sampled_labels, effort[indices])
            samples.append((x_auc, e_auc, e_auc - x_auc, x_ap, e_ap, e_ap - x_ap))
        sample_table = pd.DataFrame(samples, columns=["xception_auroc", "effort_auroc", "delta_auroc", "xception_ap", "effort_ap", "delta_ap"])
        sample_table.insert(0, "resample", np.arange(1, len(sample_table) + 1))
        sample_table.insert(0, "requested_frames", frames)
        all_samples.append(sample_table)
        summaries.append({
            "requested_frames": frames,
            "videos": len(paired),
            "seed": args.seed,
            "requested_resamples": args.resamples,
            "valid_resamples": len(sample_table),
            "skipped_single_class_attempts": attempts - len(sample_table),
            "point_auroc_xception": float(roc_auc_score(labels, xception)),
            "point_auroc_effort": float(roc_auc_score(labels, effort)),
            "point_ap_xception": float(average_precision_score(labels, xception)),
            "point_ap_effort": float(average_precision_score(labels, effort)),
            "delta_auroc": _distribution(sample_table.delta_auroc.to_numpy()),
            "delta_ap": _distribution(sample_table.delta_ap.to_numpy()),
            "positive_ratio_delta_auroc": float((sample_table.delta_auroc > 0).mean()),
            "positive_ratio_delta_ap": float((sample_table.delta_ap > 0).mean()),
        })
    samples = pd.concat(all_samples, ignore_index=True)
    samples.to_csv(args.output_dir / "paired_bootstrap_samples.csv", index=False)
    pd.json_normalize(summaries, sep=".").to_csv(args.output_dir / "paired_bootstrap_summary.csv", index=False)
    (args.output_dir / "paired_bootstrap_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    _plot(samples, "auroc", args.output_dir)
    _plot(samples, "ap", args.output_dir)
    try:
        import torch
        runtime = {"python": platform.python_version(), "pytorch": torch.__version__, "cuda": torch.version.cuda}
    except Exception:
        runtime = {"python": platform.python_version()}
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "git_commit": _commit(args.source_commit),
        "dataset": "UADFV", "source_predictions": str(args.predictions), "models": ["xception", "effort"],
        "frames": [8, 16, 32], "bootstrap_resamples": args.resamples, "seed": args.seed,
        "paired_resampling": True, "runtime": runtime,
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
