"""Gradio UI for classroom demonstrations after official checkpoints are configured."""
from __future__ import annotations

import sys
import time
from functools import lru_cache
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.inference import predict_batch, predict_image
from app.consensus import dual_model_decision
from app.model_manager import DeepfakeBenchAdapter, ModelUnavailableError
from app.face_processor import FaceProcessor
from app.video_processor import aggregate_probabilities, extract_faces
from app.visualization import probability_curve


def _paths() -> dict:
    local = PROJECT_ROOT / "configs" / "paths.local.yaml"
    source = local if local.exists() else PROJECT_ROOT / "configs" / "paths.yaml"
    return yaml.safe_load(source.read_text(encoding="utf-8")) or {}


@lru_cache(maxsize=2)
def _adapter(model_name: str) -> DeepfakeBenchAdapter:
    config = _paths()
    checkpoint = (config.get("checkpoints") or {}).get(model_name.lower())
    if not checkpoint or str(checkpoint).startswith("<"):
        raise ModelUnavailableError(f"Configure the official {model_name} checkpoint in configs/paths.local.yaml.")
    path = Path(checkpoint)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    key = "effort_bench_root" if model_name.lower() == "effort" else "deepfakebench_root"
    bench = config.get(key, "third_party/DeepfakeBench")
    bench_path = Path(bench) if Path(bench).is_absolute() else PROJECT_ROOT / bench
    return DeepfakeBenchAdapter(model_name, path, bench_path)


def _label(probability: float, threshold: float = 0.5) -> str:
    return "DEEPFAKE" if probability >= threshold else "REAL"


def _format_scores(scores: dict[str, float], elapsed: float) -> str:
    if set(scores) == {"xception", "effort"}:
        decision = dual_model_decision(scores["xception"], scores["effort"])
        return "\n".join([
            f"Xception fake probability: {scores['xception']:.2%}",
            f"Effort fake probability: {scores['effort']:.2%}",
            f"Consensus: {decision['decision']}",
            f"Recommendation: {decision['recommendation']}",
            f"Inference time: {elapsed:.2f}s",
        ])
    return "\n".join([
        *(f"{name.title()}: {_label(score)} ({score:.2%})" for name, score in scores.items()),
        f"Inference time: {elapsed:.2f}s",
    ])


def detect_image(image, model_name: str):
    if image is None:
        return "Upload an image."
    rgb = np.asarray(image.convert("RGB"))
    face = FaceProcessor().crop_largest_face(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if face is None:
        return "No face detected. Upload a clear, frontal face image."
    try:
        started = time.perf_counter()
        models = ["xception", "effort"] if model_name == "Dual-model Safety Analysis" else [model_name.lower()]
        scores = {name: predict_image(_adapter(name), face) for name in models}
    except ModelUnavailableError as exc:
        return str(exc)
    return _format_scores(scores, time.perf_counter() - started)


def detect_video(video_path: str, model_name: str, max_frames: int = 16):
    if not video_path:
        return "Upload an MP4 video.", None, []
    frames = extract_faces(video_path, max_frames=int(max_frames))
    if len(frames.faces_rgb) < 5:
        return f"Insufficient valid face frames: {len(frames.faces_rgb)} / {len(frames.sampled_indices)}. Unable to judge reliably.", None, []
    try:
        started = time.perf_counter()
        models = ["xception", "effort"] if model_name == "Dual-model Safety Analysis" else [model_name.lower()]
        scores = {name: predict_batch(_adapter(name), frames.faces_rgb) for name in models}
    except ModelUnavailableError as exc:
        return str(exc), None, []
    summary = []
    for name, values in scores.items():
        value = aggregate_probabilities(values)
        summary.append(f"{name.title()}: {_label(value)} ({value:.2%})")
    if set(scores) == {"xception", "effort"}:
        decision = dual_model_decision(
            aggregate_probabilities(scores["xception"]), aggregate_probabilities(scores["effort"])
        )
        summary = [
            f"Xception fake probability: {aggregate_probabilities(scores['xception']):.2%}",
            f"Effort fake probability: {aggregate_probabilities(scores['effort']):.2%}",
            f"Consensus: {decision['decision']}",
            f"Recommendation: {decision['recommendation']}",
        ]
    curve_values = next(iter(scores.values()))
    figure = probability_curve(frames.timestamps, curve_values)
    ranked = sorted(zip(frames.faces_rgb, frames.timestamps, curve_values), key=lambda item: item[2], reverse=True)[:5]
    gallery = [(face, f"{timestamp:.2f}s — {score:.2%}") for face, timestamp, score in ranked]
    info = "\n".join(summary + [
        f"Total frames: {frames.total_frames}",
        f"Sampled: {len(frames.sampled_indices)}",
        f"Valid faces: {len(frames.faces_rgb)}",
        f"Inference time: {time.perf_counter() - started:.2f}s",
    ])
    return info, figure, gallery


with gr.Blocks(title="DeepFake 智能检测与分析系统") as demo:
    gr.Markdown("# DeepFake 智能检测与分析系统\nUpload a face image or MP4 video. Official checkpoints are required.")
    model_choice = gr.Radio(["Xception", "Effort", "Dual-model Safety Analysis"], value="Xception", label="Model")
    with gr.Tab("Image"):
        image = gr.Image(type="pil", label="Image")
        image_button = gr.Button("Detect image")
        image_result = gr.Textbox(label="Result", lines=4)
        image_button.click(detect_image, [image, model_choice], image_result)
    with gr.Tab("Video"):
        video = gr.Video(label="MP4 video")
        video_frames = gr.Radio([8, 16, 32], value=16, label="Uniformly sampled frames")
        video_button = gr.Button("Detect video")
        video_result = gr.Textbox(label="Result", lines=6)
        curve = gr.Plot(label="Frame-level fake probability")
        suspicious = gr.Gallery(label="Top-5 suspicious frames", columns=5, object_fit="contain")
        video_button.click(detect_video, [video, model_choice, video_frames], [video_result, curve, suspicious])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
