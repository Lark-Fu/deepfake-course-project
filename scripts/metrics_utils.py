"""Threshold calibration and binary detection metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score, roc_curve


def calibrate_f1(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    candidates = np.unique(np.concatenate(([0.0], probabilities, [1.0])))
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in candidates:
        _, _, f1, _ = precision_recall_fscore_support(labels, probabilities >= threshold, average="binary", zero_division=0)
        if f1 > best_f1:
            best_threshold, best_f1 = float(threshold), float(f1)
    return best_threshold, best_f1


def equal_error_rate(labels: np.ndarray, probabilities: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, probabilities)
    fnr = 1 - tpr
    return float(fpr[np.nanargmin(np.abs(fnr - fpr))])


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = probabilities >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary", zero_division=0)
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "eer": equal_error_rate(labels, probabilities),
        "accuracy": float(np.mean(predictions == labels)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": float(threshold),
    }
