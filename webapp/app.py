"""Flask request and presentation layer for the DeepFake classroom demo."""
from __future__ import annotations

import base64
import logging
import sys
import time
import uuid
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import yaml
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.consensus import dual_model_decision
from app.face_processor import FaceProcessor
from app.inference import predict_batch, predict_image
from app.model_manager import DeepfakeBenchAdapter, ModelUnavailableError
from app.video_processor import aggregate_probabilities, extract_faces


LOGGER = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MINIMUM_VALID_FACES = 5
IMAGE_EXTENSIONS = {"jpg", "jpeg", "png"}
VIDEO_EXTENSIONS = {"mp4", "avi", "mov"}
MODEL_CHOICES = {"xception", "effort", "dual"}


class UserInputError(ValueError):
    """An error that can be presented safely to a classroom-demo user."""


def _paths() -> dict:
    local = PROJECT_ROOT / "configs" / "paths.local.yaml"
    source = local if local.exists() else PROJECT_ROOT / "configs" / "paths.yaml"
    return yaml.safe_load(source.read_text(encoding="utf-8")) or {}


def _resolve_project_path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


@lru_cache(maxsize=2)
def _adapter(model_name: str) -> DeepfakeBenchAdapter:
    """Lazy-load each frozen checkpoint at most once per Flask process."""
    config = _paths()
    checkpoint = (config.get("checkpoints") or {}).get(model_name)
    if not checkpoint or str(checkpoint).startswith("<"):
        raise ModelUnavailableError(f"Configure the official {model_name} checkpoint first.")
    bench_key = "effort_bench_root" if model_name == "effort" else "deepfakebench_root"
    benchmark = config.get(bench_key, "third_party/DeepfakeBench")
    return DeepfakeBenchAdapter(model_name, _resolve_project_path(str(checkpoint)), _resolve_project_path(str(benchmark)))


def _extension(filename: str) -> str:
    return filename.rsplit(".", 1)[1].lower() if "." in filename else ""


def _validate_upload(field_name: str, allowed: set[str]):
    upload = request.files.get(field_name)
    if upload is None or not upload.filename:
        raise UserInputError("请先选择要检测的文件。")
    filename = secure_filename(upload.filename)
    if not filename or _extension(filename) not in allowed:
        raise UserInputError(f"文件格式不受支持。支持：{', '.join(sorted(allowed))}。")
    return upload, filename


def _store_upload(upload, filename: str, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{uuid.uuid4().hex}_{filename}"
    upload.save(path)
    return path


def _model_names(mode: str) -> list[str]:
    if mode not in MODEL_CHOICES:
        raise UserInputError("未知模型模式。")
    return ["xception", "effort"] if mode == "dual" else [mode]


def _prediction(probability: float) -> str:
    return "DEEPFAKE" if probability >= 0.5 else "REAL"


def _result_from_scores(scores: dict[str, float], mode: str) -> dict:
    result: dict[str, object] = {
        "xception_probability": scores.get("xception"),
        "effort_probability": scores.get("effort"),
        "mode": mode,
    }
    if mode == "dual":
        decision = dual_model_decision(scores["xception"], scores["effort"])
        result.update({
            "prediction": decision["decision"], "consensus": decision["decision"],
            "recommendation": decision["recommendation"], "agreement": decision["agreement"],
            "probability_gap": decision["probability_gap"],
            "fake_probability": (scores["xception"] + scores["effort"]) / 2,
        })
    else:
        probability = scores[mode]
        result.update({
            "prediction": _prediction(probability), "consensus": None, "recommendation": None,
            "agreement": None, "fake_probability": probability,
        })
    return result


def _image_data_url(image_rgb: np.ndarray) -> str:
    ok, encoded = cv2.imencode(
        ".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88]
    )
    if not ok:
        raise RuntimeError("Could not encode suspicious-frame preview.")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def _checkpoint_ready(model_name: str) -> bool:
    try:
        checkpoint = (_paths().get("checkpoints") or {}).get(model_name)
        return bool(checkpoint and not str(checkpoint).startswith("<") and _resolve_project_path(str(checkpoint)).is_file())
    except Exception:
        return False


def create_app() -> Flask:
    application = Flask(__name__)
    application.config.update(MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES, UPLOAD_DIR=PROJECT_ROOT / "webapp" / "tmp")

    @application.get("/")
    def index():
        return render_template("index.html")

    @application.get("/favicon.ico")
    def favicon():
        return "", 204

    @application.get("/health")
    def health():
        import torch

        return jsonify({
            "status": "ok", "gpu_available": bool(torch.cuda.is_available()),
            "models_available": {name: _checkpoint_ready(name) for name in ("xception", "effort")},
        })

    @application.post("/api/detect/image")
    def detect_image_api():
        path: Path | None = None
        try:
            upload, filename = _validate_upload("file", IMAGE_EXTENSIONS)
            mode = request.form.get("model", "dual").lower()
            _model_names(mode)
            path = _store_upload(upload, filename, application.config["UPLOAD_DIR"])
            image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise UserInputError("图片无法解码，请上传有效的 JPG、JPEG 或 PNG 文件。")
            face = FaceProcessor().crop_largest_face(image_bgr)
            if face is None:
                raise UserInputError("未检测到有效人脸，请上传包含清晰正面人脸的图片。")
            started = time.perf_counter()
            scores = {name: predict_image(_adapter(name), face) for name in _model_names(mode)}
            result = _result_from_scores(scores, mode)
            result.update(success=True, inference_time=time.perf_counter() - started, valid_faces=1)
            return jsonify(result)
        except UserInputError as exc:
            return jsonify(success=False, error=str(exc)), 400
        except ModelUnavailableError:
            LOGGER.exception("Model configuration/loading error during image detection")
            return jsonify(success=False, error="模型不可用，请检查 checkpoint 和运行环境配置。"), 503
        except Exception:
            LOGGER.exception("Image detection failed")
            return jsonify(success=False, error="图片检测失败，请稍后重试或更换文件。"), 500
        finally:
            if path is not None:
                path.unlink(missing_ok=True)

    @application.post("/api/detect/video")
    def detect_video_api():
        path: Path | None = None
        try:
            upload, filename = _validate_upload("file", VIDEO_EXTENSIONS)
            mode = request.form.get("model", "dual").lower()
            _model_names(mode)
            try:
                requested_frames = int(request.form.get("frames", "16"))
            except ValueError as exc:
                raise UserInputError("采样帧数必须是 8、16 或 32。") from exc
            if requested_frames not in {8, 16, 32}:
                raise UserInputError("采样帧数必须是 8、16 或 32。")
            path = _store_upload(upload, filename, application.config["UPLOAD_DIR"])
            started = time.perf_counter()
            frames = extract_faces(path, max_frames=requested_frames)
            if frames.total_frames <= 0:
                raise UserInputError("视频无法解码，请上传有效的 MP4、AVI 或 MOV 文件。")
            if len(frames.faces_rgb) < MINIMUM_VALID_FACES:
                raise UserInputError(
                    f"仅检测到 {len(frames.faces_rgb)} 个有效人脸帧，至少需要 {MINIMUM_VALID_FACES} 个。请上传人脸更清晰的视频。"
                )
            frame_scores = {name: predict_batch(_adapter(name), frames.faces_rgb) for name in _model_names(mode)}
            scores = {name: aggregate_probabilities(values, MINIMUM_VALID_FACES) for name, values in frame_scores.items()}
            if any(value is None for value in scores.values()):
                raise UserInputError("有效人脸帧不足，无法给出可靠判断。")
            result = _result_from_scores({name: float(value) for name, value in scores.items()}, mode)
            primary = "xception" if "xception" in frame_scores else next(iter(frame_scores))
            ranked = sorted(zip(frames.faces_rgb, frames.timestamps, frame_scores[primary]), key=lambda item: item[2], reverse=True)[:5]
            result.update({
                "success": True, "inference_time": time.perf_counter() - started,
                "total_frames": frames.total_frames, "sampled_frames": len(frames.sampled_indices),
                "valid_faces": len(frames.faces_rgb), "timestamps": frames.timestamps,
                "xception_frame_probabilities": frame_scores.get("xception"),
                "effort_frame_probabilities": frame_scores.get("effort"),
                "suspicious_frames": [
                    {"image": _image_data_url(face), "timestamp": timestamp, "probability": probability}
                    for face, timestamp, probability in ranked
                ],
            })
            return jsonify(result)
        except UserInputError as exc:
            return jsonify(success=False, error=str(exc)), 400
        except ModelUnavailableError:
            LOGGER.exception("Model configuration/loading error during video detection")
            return jsonify(success=False, error="模型不可用，请检查 checkpoint 和运行环境配置。"), 503
        except Exception:
            LOGGER.exception("Video detection failed")
            return jsonify(success=False, error="视频检测失败，请稍后重试或更换文件。"), 500
        finally:
            if path is not None:
                path.unlink(missing_ok=True)

    @application.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        return jsonify(success=False, error="文件超过 100 MB 上传限制。"), 413

    return application


app = create_app()


if __name__ == "__main__":
    print("DeepFake Web Demo: http://127.0.0.1:5000")
    print("For a remote server, open the server hostname with port 5000 in your browser.")
    app.run(host="0.0.0.0", port=5000, debug=False)
