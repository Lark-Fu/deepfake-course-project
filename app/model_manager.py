"""Adapters around the upstream DeepfakeBench detector registry."""
from __future__ import annotations

import importlib
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ModelUnavailableError(RuntimeError):
    """Raised with an actionable message when an upstream model cannot be loaded."""


@dataclass
class LoadedModel:
    name: str
    model: Any
    device: Any
    config: dict[str, Any]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_deepfakebench_to_path(root: Path) -> None:
    training = root / "training"
    if not training.is_dir():
        raise ModelUnavailableError(
            f"Missing DeepfakeBench checkout: {root}. "
            "Copy/clone the pinned upstream repository into third_party/DeepfakeBench."
        )
    for candidate in (str(root), str(training)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    if not path.is_file():
        raise ModelUnavailableError(f"Missing detector configuration: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_detector_class(name: str, bench_root: Path) -> Any:
    """Load one upstream detector without importing every training-only detector.

    DeepfakeBench's package initializer eagerly imports all of its detectors.
    Several of those require training-only assets (for example a dlib landmark
    predictor), even when serving a plain Xception checkpoint.  Supply the
    same registry as a lightweight package and import only the requested
    module, leaving upstream source untouched.
    """
    from metrics.registry import DETECTOR

    package = types.ModuleType("detectors")
    package.__path__ = [str(bench_root / "training" / "detectors")]
    package.DETECTOR = DETECTOR
    sys.modules["detectors"] = package
    module_name = f"detectors.{name}_detector"
    importlib.import_module(module_name)
    if name not in DETECTOR.data:
        raise ModelUnavailableError(f"The upstream registry did not register '{name}'.")
    return DETECTOR[name]


def _checkpoint_state(checkpoint: Any) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise ModelUnavailableError("Unsupported checkpoint format: expected a state-dict dictionary.")
    for key in ("state_dict", "model_state_dict", "model"):
        if isinstance(checkpoint.get(key), dict):
            checkpoint = checkpoint[key]
            break
    state = checkpoint
    if state and all(str(key).startswith("module.") for key in state):
        return {str(key)[7:]: value for key, value in state.items()}
    return state


def _build_effort_without_base_checkpoint(detector_class: Any, config: dict[str, Any]) -> Any:
    """Instantiate Effort from its full released checkpoint, without a duplicate CLIP download."""
    from transformers import CLIPConfig, CLIPModel, CLIPTextConfig, CLIPVisionConfig

    vision = CLIPVisionConfig(
        hidden_size=1024,
        intermediate_size=4096,
        num_attention_heads=16,
        num_hidden_layers=24,
        image_size=224,
        patch_size=14,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
    )
    clip_config = CLIPConfig(
        text_config=CLIPTextConfig().to_dict(),
        vision_config=vision.to_dict(),
        projection_dim=768,
    )

    # The released checkpoint already contains ``weight_main`` plus the rank-1
    # residual tensors for every attention projection.  Upstream computes 96
    # full 1024x1024 SVDs only to create placeholders that are immediately
    # overwritten by that checkpoint.  Create the same parameter layout
    # directly, retaining the published detector and its forward method.
    import torch

    detector_module = sys.modules[detector_class.__module__]
    original_apply = detector_module.apply_svd_residual_to_self_attn

    def _fast_apply(model: Any, _rank: int | None = None, **_kwargs: Any) -> Any:
        for child_name, child in model.named_children():
            if "self_attn" in child_name:
                for projection_name, projection in list(child.named_children()):
                    if not isinstance(projection, torch.nn.Linear):
                        continue
                    replacement = detector_module.SVDResidualLinear(
                        projection.in_features,
                        projection.out_features,
                        1023 if _rank is None else _rank,
                        bias=projection.bias is not None,
                        init_weight=projection.weight.data,
                    )
                    dtype, device = projection.weight.dtype, projection.weight.device
                    replacement.S_residual = torch.nn.Parameter(torch.zeros(1, dtype=dtype, device=device))
                    replacement.U_residual = torch.nn.Parameter(
                        torch.zeros(projection.out_features, 1, dtype=dtype, device=device)
                    )
                    replacement.V_residual = torch.nn.Parameter(
                        torch.zeros(1, projection.in_features, dtype=dtype, device=device)
                    )
                    setattr(child, projection_name, replacement)
            else:
                _fast_apply(child, _rank)
        for parameter_name, parameter in model.named_parameters():
            parameter.requires_grad = any(
                token in parameter_name for token in ("S_residual", "U_residual", "V_residual")
            )
        return model

    def _from_config(cls: Any, *_args: Any, **_kwargs: Any) -> Any:
        return cls(clip_config)

    CLIPModel.from_pretrained = classmethod(_from_config)
    detector_module.apply_svd_residual_to_self_attn = _fast_apply
    try:
        return detector_class(config)
    finally:
        delattr(CLIPModel, "from_pretrained")
        detector_module.apply_svd_residual_to_self_attn = original_apply


class DeepfakeBenchAdapter:
    """Load an official checkpoint and return fake probabilities for preprocessed RGB tensors."""

    def __init__(self, name: str, checkpoint: str | Path, bench_root: str | Path | None = None) -> None:
        self.name = name.lower()
        self.checkpoint = Path(checkpoint)
        self.bench_root = Path(bench_root) if bench_root else _project_root() / "third_party" / "DeepfakeBench"
        self.loaded: LoadedModel | None = None

    def load(self) -> "DeepfakeBenchAdapter":
        if self.name not in {"xception", "effort"}:
            raise ModelUnavailableError(f"Unsupported model: {self.name}. Choose xception or effort.")
        if not self.checkpoint.is_file():
            raise ModelUnavailableError(
                f"Missing checkpoint: {self.checkpoint}\n"
                f"Please download the official {self.name} checkpoint and place it at that path."
            )
        _add_deepfakebench_to_path(self.bench_root)
        import torch
        detector_class = _load_detector_class(self.name, self.bench_root)
        detector_file = self.bench_root / "training" / "config" / "detector" / f"{self.name}.yaml"
        config = _load_yaml(detector_file)
        # DeepfakeBench's config stores this as a cwd-relative path.  The app
        # may be launched from anywhere, so anchor it to the pinned checkout.
        if self.name == "xception" and config.get("pretrained"):
            config["pretrained"] = str(
                self.bench_root / "training" / "pretrained" / Path(config["pretrained"]).name
            )
        config.update({"cuda": torch.cuda.is_available(), "cudnn": True})
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.name == "effort":
            model = _build_effort_without_base_checkpoint(detector_class, config).to(device)
        else:
            model = detector_class(config).to(device)
        state = _checkpoint_state(torch.load(self.checkpoint, map_location=device, weights_only=False))
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise ModelUnavailableError(
                f"Checkpoint is incompatible with the pinned {self.name} implementation: {exc}"
            ) from exc
        model.eval()
        self.loaded = LoadedModel(self.name, model, device, config)
        return self

    def predict_batch(self, images: Any) -> list[float]:
        if self.loaded is None:
            self.load()
        assert self.loaded is not None
        import torch

        tensor = images.to(self.loaded.device)
        labels = torch.zeros(tensor.shape[0], dtype=torch.long, device=self.loaded.device)
        with torch.inference_mode():
            output = self.loaded.model({"image": tensor, "label": labels}, inference=True)
        probabilities = output.get("prob") if isinstance(output, dict) else output
        if probabilities is None:
            raise ModelUnavailableError("Detector output does not contain 'prob'.")
        return [float(value) for value in probabilities.detach().float().cpu().flatten()]
