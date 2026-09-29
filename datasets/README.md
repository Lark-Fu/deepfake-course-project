# Dataset preparation

This directory is intentionally Git-ignored except for this file. Keep the actual datasets outside
version control and reference their absolute locations only from `configs/paths.local.yaml`.

Required formal dataset:

1. UADFV only, using the DeepfakeBench-preprocessed RGB structure:
   `real/frames/<video_id>/*.png`, `fake/frames/<video_id>/*.png`, plus the matching
   `landmarks/` directories.

FaceForensics++, Celeb-DF-v2, and DF40 are not required in this lightweight course plan. An
optional FF++ Mini may be used only for demo/debug and must not be reported as formal results.

Run `python scripts/check_dataset.py --dataset uadfv` before an evaluation. The project never
downloads a dataset automatically.
