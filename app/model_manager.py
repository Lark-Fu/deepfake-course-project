"""Adapters around the upstream DeepfakeBench detector registry."""
from __future__ import annotations

import sys
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


def _checkpoint_state(checkpoint: Any) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise ModelUnavailableError("Unsupported checkpoint format: expected a state-dict dictionary.")
    for key in ("state_dict", "model_state_dict", "model"):
        if isinstance(checkpoint.get(key), dict):
            return checkpoint[key]
    return checkpoint


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
        from detectors import DETECTOR

        if self.name not in DETECTOR:
            raise ModelUnavailableError(
                f"The pinned DeepfakeBench checkout does not expose '{self.name}'. "
                "Use the official Effort fork or update the ignored upstream checkout, then record its commit."
            )
        detector_file = self.bench_root / "training" / "config" / "detector" / f"{self.name}.yaml"
        config = _load_yaml(detector_file)
        config.update({"cuda": torch.cuda.is_available(), "cudnn": True})
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DETECTOR[self.name](config).to(device)
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
