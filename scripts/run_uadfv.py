#!/usr/bin/env python3
"""Frozen-checkpoint UADFV evaluation and 8/16/32-frame efficiency analysis."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.inference import predict_batch
from app.model_manager import DeepfakeBenchAdapter
from app.uadfv_dataset import UADFVVideo, load_uadfv, select_uniform_frames

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics_utils import binary_metrics, bootstrap_confidence_intervals


def _chunks(items: tuple[Path, ...], size: int) -> list[tuple[Path, ...]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _load_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing local configuration: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _resolve(project_root: Path, value: str) -> Path:
    source = Path(value)
    return source if source.is_absolute() else project_root / source


def _limited(videos: list[UADFVVideo], per_class: int | None) -> list[UADFVVideo]:
    if per_class is None:
        return videos
    selected: list[UADFVVideo] = []
    for label in (0, 1):
        selected.extend([video for video in videos if video.label == label][:per_class])
    return selected


def _predict_video(adapter: DeepfakeBenchAdapter, video: UADFVVideo, frames: int, batch_size: int) -> tuple[list[float], int]:
    selected = select_uniform_frames(video.frame_paths, frames)
    probabilities: list[float] = []
    for batch in _chunks(selected, batch_size):
        probabilities.extend(predict_batch(adapter, batch))
    return probabilities, len(selected)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate frozen Xception and Effort checkpoints on UADFV.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "paths.local.yaml")
    parser.add_argument("--models", nargs="+", choices=("xception", "effort"), default=["xception", "effort"])
    parser.add_argument("--frames", nargs="+", type=int, choices=(8, 16, 32), default=[8, 16, 32])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit-per-class", type=int, help="Use only N real and N fake videos for a smoke test.")
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    config = _load_config(args.config)
    root_value = config.get("uadfv_root")
    if not isinstance(root_value, str) or not root_value.strip():
        raise SystemExit("Set uadfv_root in configs/paths.local.yaml before evaluation.")
    videos = _limited(load_uadfv(_resolve(PROJECT_ROOT, root_value)), args.limit_per_class)
    class_counts = {label: sum(video.label == label for video in videos) for label in (0, 1)}
    if not all(class_counts.values()):
        raise SystemExit("Evaluation requires at least one real and one fake video.")
    if args.limit_per_class is None and class_counts != {0: 49, 1: 49}:
        raise SystemExit(f"Expected UADFV 49 real + 49 fake videos, found {class_counts}.")

    run_name = datetime.now(timezone.utc).strftime("uadfv_%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or PROJECT_ROOT / "experiments" / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoints = config.get("checkpoints") or {}
    bench_keys = {"xception": "deepfakebench_root", "effort": "effort_bench_root"}
    rows: list[dict] = []
    summaries: list[dict] = []

    import torch

    for model_name in args.models:
        checkpoint_value = checkpoints.get(model_name)
        bench_value = config.get(bench_keys[model_name])
        if not isinstance(checkpoint_value, str) or not isinstance(bench_value, str):
            raise SystemExit(f"Configure checkpoint and benchmark root for {model_name}.")
        adapter = DeepfakeBenchAdapter(model_name, _resolve(PROJECT_ROOT, checkpoint_value), _resolve(PROJECT_ROOT, bench_value)).load()
        for frame_count in args.frames:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            started = time.perf_counter()
            video_rows: list[dict] = []
            for video in videos:
                probabilities, used = _predict_video(adapter, video, frame_count, args.batch_size)
                video_probability = float(sum(probabilities) / len(probabilities))
                video_rows.append({
                    "model": model_name,
                    "requested_frames": frame_count,
                    "video_id": video.video_id,
                    "label": video.label,
                    "available_frames": len(video.frame_paths),
                    "processed_frames": used,
                    "probability": video_probability,
                })
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            table = pd.DataFrame(video_rows)
            metrics = binary_metrics(table.label.to_numpy(), table.probability.to_numpy(), threshold=0.5)
            intervals = bootstrap_confidence_intervals(
                table.label.to_numpy(), table.probability.to_numpy(), threshold=0.5,
                resamples=args.bootstrap_resamples,
            )
            peak_memory_mb = float(torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else None
            summary = {
                "model": model_name,
                "requested_frames": frame_count,
                "videos": len(table),
                "total_processed_frames": int(table.processed_frames.sum()),
                "total_inference_seconds": elapsed,
                "seconds_per_video": elapsed / len(table),
                "peak_gpu_memory_mb": peak_memory_mb,
                "video_metrics": metrics,
                "video_metric_95ci": intervals,
                "threshold_policy": "fixed 0.5; not optimized on UADFV",
                "aggregation": "mean(frame_probability)",
            }
            summaries.append(summary)
            rows.extend(video_rows)
            print(json.dumps(summary, ensure_ascii=False))

    pd.DataFrame(rows).to_csv(output_dir / "video_predictions.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.json_normalize(summaries, sep=".").to_csv(output_dir / "summary.csv", index=False)
    manifest = {
        "dataset": "UADFV",
        "protocol": "frozen FaceForensics++ checkpoints -> UADFV",
        "videos": class_counts,
        "models": args.models,
        "frames": args.frames,
        "bootstrap_resamples": args.bootstrap_resamples,
        "sanity_limit_per_class": args.limit_per_class,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Results: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
