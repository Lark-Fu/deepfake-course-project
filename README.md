# DeepFake 智能检测与轻量化跨域评测

课程项目以冻结的 FaceForensics++ 预训练 Xception 与 Effort checkpoint，在唯一的正式测试集
UADFV 上进行视频级 DeepFake 检测。项目不训练、不微调模型，也不依据 UADFV 标签调节阈值。

正式实验：

- **Experiment 1 — Cross-Dataset Generalization**：Xception 与 Effort 在 UADFV（49 Real + 49 Fake）上的视频级 AUROC、AP、EER，以及固定阈值 0.5 的辅助指标；
- **Experiment 2 — Frame Sampling Efficiency**：在同一批 98 个视频上统一使用平均帧概率聚合，对比 8、16、32 帧的性能、总推理时间、单视频时间和峰值 GPU 显存。

FaceForensics++、Celeb-DF-v2、DF40 均不是本项目的正式实验数据。若以后有 FF++ Mini，只能用于 Demo 或 pipeline sanity check，不能作为正式泛化结论。

## Environment

目标环境为 Linux + NVIDIA GPU，使用专用 Conda 环境 `/data/conda_envs/fq/data`，不会修改已有环境。
运行时是 Python 3.10、PyTorch 2.7 CUDA 12.8；环境检查：

```bash
/data/conda_envs/fq/data/bin/python scripts/test_environment.py
```

## UADFV data layout

正式数据存放在项目外部（不提交 Git）：

```text
/data/workspaces/fq/datasets/UADFV/
├── real/
│   ├── frames/<video_id>/*.png
│   └── landmarks/<video_id>/*.npy
└── fake/
    ├── frames/<video_id>/*.png
    └── landmarks/<video_id>/*.npy
```

在 `configs/paths.local.yaml` 设置 `uadfv_root` 的真实绝对路径；该文件被 Git 忽略。
DeepfakeBench 预处理版 UADFV 的 RGB 子目录即为需要下载的内容，勿下载整个数据集合，也不要二次 JPEG 压缩、缩放或改色。

## Formal evaluation

先验证数据集，然后只用 2 Real + 2 Fake 做最小 sanity check：

```bash
python scripts/check_dataset.py --config configs/paths.local.yaml --dataset uadfv
python scripts/run_uadfv.py --config configs/paths.local.yaml --limit-per-class 2 --frames 8 --bootstrap-resamples 100
```

sanity check 通过后执行完整的两个正式实验：

```bash
python scripts/run_uadfv.py --config configs/paths.local.yaml --frames 8 16 32 --bootstrap-resamples 1000
```

每次运行会在 `experiments/` 下生成本地 CSV、JSON 和 run manifest。视频级概率固定为所选帧概率的均值；阈值固定为 0.5，未使用 UADFV 标签优化。少数视频若实际可用帧少于 32，脚本只使用其全部真实可用帧并在预测 CSV 中记录，不会复制或补造图像。

## Demo

Gradio Demo 支持图片、视频、Xception、Effort 与双模型比较；视频默认均匀采样 16 帧，并可选择 8/16/32 帧。它用于课堂演示，不会改变正式实验输出。在 **Dual-model Safety Analysis** 模式中，只有两个模型均跨过固定 0.5 阈值且结论相同才输出自动 REAL/DEEPFAKE；不一致时输出 `UNCERTAIN` 并建议人工复核。

```bash
python app/app.py
```

数据、权重、演示媒体、实验输出和 `paths.local.yaml` 均不得提交 GitHub。
