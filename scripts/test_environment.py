#!/usr/bin/env python3
"""Print the runtime facts required before model deployment."""
from __future__ import annotations

import importlib
import platform
import sys


def version_of(module_name: str) -> str:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return f"NOT AVAILABLE ({type(exc).__name__}: {exc})"
    return getattr(module, "__version__", "installed")


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    try:
        import torch
    except Exception as exc:
        print(f"PyTorch: NOT AVAILABLE ({type(exc).__name__}: {exc})")
        return 1
    print(f"PyTorch: {torch.__version__}")
    print(f"Torch CUDA: {torch.version.cuda}")
    available = torch.cuda.is_available()
    print(f"CUDA available: {available}")
    print(f"GPU count: {torch.cuda.device_count() if available else 0}")
    if available:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            print(f"GPU {index}: {props.name}; memory={props.total_memory / 2**30:.2f} GiB")
    modules = (("OpenCV", "cv2"), ("Gradio", "gradio"), ("NumPy", "numpy"), ("Pandas", "pandas"), ("Scikit-learn", "sklearn"))
    for display, module in modules:
        print(f"{display}: {version_of(module)}")
    return 0 if available else 2


if __name__ == "__main__":
    raise SystemExit(main())
