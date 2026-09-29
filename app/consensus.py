"""Transparent dual-model safety decisions shared by analysis and the demo."""
from __future__ import annotations


def dual_model_decision(xception_probability: float, effort_probability: float, threshold: float = 0.5) -> dict[str, object]:
    """Return an automatic decision only when both detectors agree at a fixed threshold."""
    xception_fake = xception_probability >= threshold
    effort_fake = effort_probability >= threshold
    if xception_fake and effort_fake:
        decision, agreement, recommendation = "DEEPFAKE", True, "High agreement"
    elif not xception_fake and not effort_fake:
        decision, agreement, recommendation = "REAL", True, "High agreement"
    else:
        decision, agreement, recommendation = "UNCERTAIN", False, "Manual review recommended"
    return {
        "decision": decision,
        "agreement": agreement,
        "recommendation": recommendation,
        "probability_gap": abs(xception_probability - effort_probability),
        "confidence_xception": abs(xception_probability - threshold),
        "confidence_effort": abs(effort_probability - threshold),
    }
