# Dataset preparation

This directory is intentionally Git-ignored except for this file. Keep the actual datasets outside
version control and reference their absolute locations only from `configs/paths.local.yaml`.

Required datasets:

1. FaceForensics++ c23, preferably DeepfakeBench preprocessed cropped faces and metadata;
2. Celeb-DF-v2, prepared using the same DeepfakeBench conventions;
3. optional small DF40 subsets: SimSwap, FOMM, DiT, and StarGANv2.

Run `python scripts/check_dataset.py` before an evaluation. The project never downloads a dataset
automatically.
