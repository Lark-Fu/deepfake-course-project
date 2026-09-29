# UADFV formal experiment results

Run: `uadfv_full_20260929`. Dataset: 49 Real + 49 Fake UADFV videos. Both Xception and Effort
used unchanged FaceForensics++-trained checkpoints. Each result is video-level mean frame
probability with the fixed 0.5 decision threshold; UADFV labels were not used for threshold
selection. AUROC/AP/EER are the primary metrics. The reported 95% CIs use 1,000 bootstrap resamples.

| Model | Frames requested | AUROC (95% CI) | AP (95% CI) | EER | Time/video |
| --- | ---: | --- | --- | ---: | ---: |
| Xception | 8 | 0.9604 [0.9233, 0.9867] | 0.9648 [0.9307, 0.9886] | 0.1633 | 0.217 s |
| Xception | 16 | 0.9608 [0.9256, 0.9887] | 0.9667 [0.9327, 0.9902] | 0.0816 | 0.409 s |
| Xception | 32 | 0.9633 [0.9287, 0.9903] | 0.9685 [0.9357, 0.9912] | 0.1020 | 0.807 s |
| Effort | 8 | 0.9758 [0.9385, 0.9979] | 0.9830 [0.9577, 0.9978] | 0.0612 | 0.444 s |
| Effort | 16 | 0.9767 [0.9385, 0.9983] | 0.9840 [0.9590, 0.9982] | 0.0408 | 0.916 s |
| Effort | 32 | 0.9779 [0.9399, 0.9987] | 0.9849 [0.9607, 0.9985] | 0.0408 | 1.819 s |

The 32-frame request processed 3,099 actual frames rather than 3,136 because nine supplied videos
have 23–31 preprocessed frames. The evaluator records the true frame count and does not duplicate
frames. The measurements show that 16 frames preserves nearly all observed AUROC while roughly
halving per-video inference time relative to 32 frames; this is an observation for this UADFV run,
not a claim of universal optimality.

Generated local figures are `frames_vs_auroc.png` and `frames_vs_inference_time.png` in the run's
ignored `experiments/` output directory.
