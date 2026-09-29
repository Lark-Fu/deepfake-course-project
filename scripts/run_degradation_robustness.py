#!/usr/bin/env python3
"""Evaluate frozen UADFV detectors under in-memory JPEG and resolution degradation."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.degradation import CONDITIONS, load_degraded_rgb
from app.inference import predict_batch
from app.model_manager import DeepfakeBenchAdapter
from app.uadfv_dataset import UADFVVideo, load_uadfv, select_uniform_frames

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics_utils import binary_metrics


def _commit(value: str | None) -> str:
    if value:
        return value
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing local configuration: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _limited(videos: list[UADFVVideo], per_class: int | None) -> list[UADFVVideo]:
    if per_class is None:
        return videos
    selected: list[UADFVVideo] = []
    for label in (0, 1):
        selected.extend([video for video in videos if video.label == label][:per_class])
    return selected


def _predict(adapter: DeepfakeBenchAdapter, video: UADFVVideo, condition: str, batch_size: int) -> tuple[float, int]:
    paths = select_uniform_frames(video.frame_paths, 16)
    values: list[float] = []
    for start in range(0, len(paths), batch_size):
        images = [load_degraded_rgb(path, condition) for path in paths[start : start + batch_size]]
        values.extend(predict_batch(adapter, images))
    return float(sum(values) / len(values)), len(paths)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "paths.local.yaml")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--models", nargs="+", choices=("xception", "effort"), default=["xception", "effort"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit-per-class", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing output directory: {args.output_dir}")
    config = _config(args.config)
    root_value = config.get("uadfv_root")
    if not isinstance(root_value, str) or not root_value:
        raise SystemExit("Set uadfv_root in paths.local.yaml.")
    videos = _limited(load_uadfv(_path(root_value)), args.limit_per_class)
    counts = {label: sum(video.label == label for video in videos) for label in (0, 1)}
    if not all(counts.values()) or (args.limit_per_class is None and counts != {0: 49, 1: 49}):
        raise SystemExit(f"Unexpected class counts: {counts}")
    args.output_dir.mkdir(parents=True)
    checkpoints = config.get("checkpoints") or {}
    benchmark_keys = {"xception": "deepfakebench_root", "effort": "effort_bench_root"}
    rows: list[dict] = []
    summaries: list[dict] = []
    import torch
    for model_name in args.models:
        checkpoint, benchmark = checkpoints.get(model_name), config.get(benchmark_keys[model_name])
        if not isinstance(checkpoint, str) or not isinstance(benchmark, str):
            raise SystemExit(f"Configure {model_name} checkpoint and benchmark path.")
        adapter = DeepfakeBenchAdapter(model_name, _path(checkpoint), _path(benchmark)).load()
        model_summaries: dict[str, dict] = {}
        for condition in args.conditions:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            started = time.perf_counter()
            condition_rows = []
            for video in videos:
                probability, used = _predict(adapter, video, condition, args.batch_size)
                condition_rows.append({"model": model_name, "condition": condition, "video_id": video.video_id, "label": video.label, "processed_frames": used, "probability": probability})
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            table = pd.DataFrame(condition_rows)
            summary = {
                "model": model_name, "condition": condition, "videos": len(table), "frames": 16,
                "total_processed_frames": int(table.processed_frames.sum()), "total_inference_seconds": elapsed,
                "seconds_per_video": elapsed / len(table),
                "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else None,
                "video_metrics": binary_metrics(table.label.to_numpy(), table.probability.to_numpy(), threshold=0.5),
                "threshold_policy": "fixed 0.5; not optimized on UADFV", "aggregation": "mean(frame_probability)",
            }
            model_summaries[condition] = summary
            summaries.append(summary)
            rows.extend(condition_rows)
            print(json.dumps(summary, ensure_ascii=False))
        if "original" in model_summaries:
            original = model_summaries["original"]["video_metrics"]
            for summary in summaries:
                if summary["model"] == model_name:
                    metric = summary["video_metrics"]
                    summary["auroc_drop_from_original"] = float(original["auroc"] - metric["auroc"])
                    summary["ap_drop_from_original"] = float(original["average_precision"] - metric["average_precision"])
    pd.DataFrame(rows).to_csv(args.output_dir / "video_predictions.csv", index=False)
    pd.json_normalize(summaries, sep=".").to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    runtime = {"python": platform.python_version(), "pytorch": torch.__version__, "cuda": torch.version.cuda}
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "git_commit": _commit(args.source_commit),
        "dataset": "UADFV", "models": args.models, "frames": 16, "conditions": args.conditions,
        "threshold": 0.5, "aggregation": "mean(frame_probability)", "online_processing": True,
        "limit_per_class": args.limit_per_class, "runtime": runtime,
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Results: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
