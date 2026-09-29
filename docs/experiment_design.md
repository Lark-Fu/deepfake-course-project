# UADFV lightweight experiment design

## Scope and integrity policy

The only formal dataset is DeepfakeBench-preprocessed UADFV: 49 real and 49 fake videos. Both
Xception and Effort use unchanged checkpoints trained on FaceForensics++. This is a
cross-dataset generalization evaluation from the FaceForensics++ source domain to the UADFV target
domain. No training, fine-tuning, target-domain threshold selection, or persistent modification of
the supplied data is permitted. Experiment 4 is the sole exception for *ephemeral in-memory input
perturbations*; it neither overwrites nor creates UADFV image files.

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

## Experiment 3 — Paired Bootstrap Model Comparison

Reuse the frozen baseline `video_predictions.csv`; do not run inference again. For each 8/16/32-frame
setting, resample the same 98 video indices with replacement for Xception and Effort together. Use
5,000 valid resamples with seed 2026, skipping one-class resamples. Report the distribution of
`Effort - Xception` AUROC and AP, including the percentile interval and the fraction of resamples
where the delta is positive. This fraction is descriptive and is not a conventional p-value.

## Experiment 4 — In-memory Degradation Robustness

Reuse the same 98 UADFV videos and request 16 uniformly selected frames per video. Keep the
unchanged checkpoints, mean video aggregation, and the fixed 0.5 threshold. Evaluate three input
conditions independently: `original`; `jpeg70`, produced by JPEG encode/decode at quality 70 only
in RAM; and `resize50`, produced by resizing each decoded frame to 50% with `INTER_AREA` and then
back to its original dimensions with `INTER_CUBIC`, also only in RAM. Do not write transformed
frames to disk or alter the supplied UADFV tree.

For each model and condition, record video-level AUROC, AP, EER, fixed-threshold auxiliary metrics,
total runtime, per-video runtime, actual frame count, and peak GPU memory. Report the measured
change from `original` separately for each model; do not assume every perturbation lowers every
metric. Every full run must have a manifest and per-video CSV, and output directories are
append-only.

## Experiment 5 — Dual-model Consensus and Uncertainty

Reuse the frozen 16-frame video predictions. At threshold 0.5, output automatic DEEPFAKE only when
both models are fake, automatic REAL only when both are real, and `UNCERTAIN` otherwise. Report
agreement rate (coverage), disagreement count, selective accuracy on the automatic subset, errors,
and probability gaps. The analysis does not average probabilities to force a label for disagreements.
