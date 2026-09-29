# Environment notes

## Phase 1 inspection (2026-09-29)

- GPU: one NVIDIA GeForce RTX 5080, 16,303 MiB total memory; approximately 15,399 MiB free at inspection.
- NVIDIA driver: 580.173.02; `nvidia-smi` maximum supported CUDA: 13.0.
- `nvcc`: not installed. This is not required for prebuilt PyTorch wheels.
- Disk available under `/data`: approximately 1.3 TB.
- Docker and Git LFS: unavailable.
- GitHub: the server cannot reach GitHub reliably, so the local Git working copy is the push mirror.

## Chosen runtime

DeepfakeBench and Effort publish legacy installation scripts for Python 3.7.2 and PyTorch
1.12.0+cu113. That runtime is unsuitable for the RTX 5080 because it predates the GPU architecture.
The course project therefore uses the isolated Python 3.10 environment and a PyTorch 2.7 CUDA 12.8
wheel. CUDA 12.8 is supplied by the wheel; it does not modify the driver or system CUDA.

The current upstream code is kept in `third_party/DeepfakeBench` and adapted only via wrappers in
this repository. Pinned upstream state: `SCLBD/DeepfakeBench`, branch `main`, commit
`f188b1c105465e2e5377eb536a95022ae0e4522d` (checked out 2026-09-29). The ignored checkout is
copied to the server from the local project mirror because the server cannot reach GitHub.

Effort uses its official repository's bundled DeepfakeBench fork, pinned at
`YZY-stack/Effort-AIGI-Detection` commit `96f5dea2b534d400cfd7003f053c7e93c8e16461`
(checked out 2026-09-29). This is also ignored by the course repository.

## Installation order

```bash
/data/conda_envs/fq/data/bin/python -m pip install --upgrade pip
/data/conda_envs/fq/data/bin/python -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
/data/conda_envs/fq/data/bin/python -m pip install -r requirements.txt
/data/conda_envs/fq/data/bin/python scripts/test_environment.py
```

Do not run either upstream `install.sh` wholesale: its pinned stack is retained only as a reference.
The listed DeepfakeBench support packages are installed separately after the modern PyTorch stack;
they are not permitted to downgrade PyTorch or CUDA.

For this legacy upstream code, pin NumPy to the 1.26 series and OpenCV to the 4.10 series: `imgaug`
uses a NumPy API removed in NumPy 2. The server environment passed `pip check` with this combination.

## Released checkpoint loading

The Xception wrapper loads the official base backbone from
`third_party/DeepfakeBench/training/pretrained/` and the released course checkpoint from `weights/`.
The Effort release checkpoint contains the full CLIP ViT-L/14 vision model plus the rank-one residual
parameters. The wrapper reconstructs the documented ViT-L/14 architecture from its public config and
loads that complete checkpoint directly. It intentionally avoids the upstream's redundant initial SVD
work and its hard-coded, second copy of the CLIP model download. Neither model weight is committed.
