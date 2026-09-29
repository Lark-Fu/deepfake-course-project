# Experiment design

## Research question

How do a traditional CNN detector (Xception) and a CLIP-L/14-based detector (Effort) differ in
in-domain and cross-domain DeepFake detection without target-domain fine-tuning?

## Protocol

1. Calibrate one threshold per model on the FaceForensics++ validation split by maximum F1.
2. Use that fixed threshold unchanged for FaceForensics++ test, Celeb-DF-v2, and DF40.
3. Report frame/video AUROC, AP, EER, accuracy, precision, recall, and F1. Store per-frame and
   per-video predictions in local CSV files.
4. Define the descriptive generalization drop as `AUC_FFPP - AUC_CelebDF`; it is not a new metric.
5. For DF40, evaluate only SimSwap, FOMM, DiT, and StarGANv2 after the two core experiments work.

Every formal run receives its own timestamped directory with source commit, environment, model,
checkpoint, dataset, threshold, batch size, and seed.
