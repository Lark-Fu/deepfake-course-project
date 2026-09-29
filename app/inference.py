"""Image preprocessing and a small, stable public inference interface."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from app.model_manager import DeepfakeBenchAdapter


MODEL_PREPROCESS = {
    "xception": {"size": 256, "mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
    "effort": {"size": 224, "mean": (0.48145466, 0.4578275, 0.40821073), "std": (0.26862954, 0.26130258, 0.27577711)},
}


def preprocess_image(image: str | Path | Image.Image | np.ndarray, model_name: str):
    """Return a normalized CHW torch tensor using the upstream model convention."""
    import torch

    name = model_name.lower()
    if name not in MODEL_PREPROCESS:
        raise ValueError(f"Unknown model: {model_name}")
    if isinstance(image, (str, Path)):
        source = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        source = Image.fromarray(image.astype("uint8")).convert("RGB")
    else:
        source = image.convert("RGB")
    spec = MODEL_PREPROCESS[name]
    source = source.resize((spec["size"], spec["size"]), Image.Resampling.BICUBIC)
    array = np.asarray(source, dtype=np.float32).transpose(2, 0, 1) / 255.0
    tensor = torch.from_numpy(array)
    mean = torch.tensor(spec["mean"]).view(3, 1, 1)
    std = torch.tensor(spec["std"]).view(3, 1, 1)
    return (tensor - mean) / std


def predict_batch(adapter: DeepfakeBenchAdapter, images: Iterable[str | Path | Image.Image | np.ndarray]) -> list[float]:
    import torch

    prepared = [preprocess_image(image, adapter.name) for image in images]
    if not prepared:
        return []
    return adapter.predict_batch(torch.stack(prepared))


def predict_image(adapter: DeepfakeBenchAdapter, image: str | Path | Image.Image | np.ndarray) -> float:
    scores = predict_batch(adapter, [image])
    return scores[0]
