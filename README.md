# DeepFake 智能检测与跨域泛化分析

课程项目：《基于深度学习的 DeepFake 智能检测与跨域泛化分析》。本项目复用
[DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) 的 Xception 与 Effort 实现，提供：

- 单张人脸图片与 MP4 视频检测；
- Xception、Effort 以及双模型对比；
- FaceForensics++ 域内评测、Celeb-DF-v2 跨域评测与可选 DF40 子集评测；
- 固定验证集阈值、CSV 原始预测、指标统计与论文/PPT 图表；
- 面向课堂演示的 Gradio 界面。

## Environment

The target runtime is Linux with an NVIDIA GPU. Use the dedicated Conda environment at
`/data/conda_envs/fq/data`; do not modify existing environments. This project uses Python 3.10
and PyTorch 2.7 CUDA 12.8 wheels because the target RTX 5080 requires a modern CUDA build.

```bash
/data/conda_envs/fq/data/bin/python scripts/test_environment.py
```

## Layout

```text
app/                 # inference, video pipeline, and Gradio UI
configs/             # public templates; local paths belong in paths.local.yaml
scripts/             # environment checks, evaluation, calibration, and figures
experiments/         # metrics and presentation-ready figures
datasets/            # documentation only; data is ignored by Git
weights/             # documentation only; checkpoints are ignored by Git
third_party/         # ignored upstream DeepfakeBench clone
docs/                # experiment design and reproducibility notes
```

## Data and checkpoints

Set local dataset paths in `configs/paths.local.yaml` (copy `configs/paths.yaml`) and download
only the official checkpoints described in [weights/README.md](weights/README.md). Do not commit
datasets, checkpoints, videos, credentials, or local path configuration.

## Planned commands

```bash
python scripts/check_dataset.py --config configs/paths.local.yaml --dataset ffpp
python scripts/run_ffpp.py --config configs/paths.local.yaml
python scripts/run_celebdf.py --config configs/paths.local.yaml
python scripts/generate_figures.py --metrics-dir experiments/metrics
./run_app.sh
```

Formal experiment outputs are placed in timestamped directories under `experiments/`; raw
predictions and logs remain local.
