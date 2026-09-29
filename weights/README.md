# Official model checkpoints

Weights are intentionally excluded from Git. Download only official releases and place the files
under `weights/` using their original names.

| Model | Official source | Intended checkpoint |
| --- | --- | --- |
| Xception | [DeepfakeBench release v1.0.1](https://github.com/SCLBD/DeepfakeBench/releases/tag/v1.0.1) | FaceForensics++ pretrained Xception |
| Effort (CLIP-L/14) | [Effort official repository](https://github.com/YZY-stack/Effort-AIGI-Detection) | FaceForensics++ pretrained CLIP-L/14 + Effort |

Before inference, verify the file exists and record its official filename, size, SHA-256 (when
provided), and source URL in the experiment log. The code reports a concise missing-checkpoint
message instead of attempting a download.
