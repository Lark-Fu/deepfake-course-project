# UADFV lightweight experiment design

## Scope and integrity policy

The only formal dataset is DeepfakeBench-preprocessed UADFV: 49 real and 49 fake videos. Both
Xception and Effort use unchanged checkpoints trained on FaceForensics++. This is a
cross-dataset generalization evaluation from the FaceForensics++ source domain to the UADFV target
domain. No training, fine-tuning, target-domain threshold selection, recompression, resizing, or
color-space conversion is permitted.

The video-level probability is always `mean(frame_probability)`. The classification threshold is
the unchanged model-default value 0.5. Accuracy, precision, recall, and F1 are supplemental only;
the report must state that 0.5 was not optimized using UADFV labels.

## Experiment 1 — Cross-Dataset Generalization

Evaluate all 98 UADFV videos with the same preprocessed RGB frames and 32-frame request. Report
video-level AUROC, Average Precision, and EER as primary metrics, and provide 95% percentile
bootstrap confidence intervals using 1,000 resamples. Record the raw per-video probabilities and
the actual processed frame count. Some supplied videos contain fewer than 32 preprocessed frames;
use all available frames rather than duplicating images.

## Experiment 2 — Frame Sampling Efficiency

For each model, evaluate the exact same 98 videos using uniform selections of 8, 16, and 32 frames
from the available preprocessed frame sequence. Do not make three data copies. For every setting,
record AUROC, AP, EER, total inference time, mean inference time per video, number of processed
frames, and PyTorch peak GPU memory when CUDA is available. Plot Frames vs AUROC and Frames vs
Inference Time after the measured results exist.

## Reproducibility

Run the 2 Real + 2 Fake sanity check before the complete evaluation. Every run writes a local
manifest, per-video prediction CSV, and summary JSON/CSV under `experiments/`. These outputs and
the external UADFV files remain ignored by Git. The pinned upstream repositories and runtime facts
are recorded in `docs/environment_notes.md`.
