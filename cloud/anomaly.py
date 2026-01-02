#!/usr/bin/env python3
# anomaly.py — Anomaly Detection for CLOUD v3.1
#
# Heuristic patterns (0 params) detecting unusual emotional states.
#
# Four anomaly types:
# 1. forced_stability: high arousal + fast convergence (suppression)
# 2. dissociative_shutdown: high VOID + high arousal (trauma response)
# 3. unresolved_confusion: low arousal + slow convergence (stuck)
# 4. emotional_flatline: all chambers low (numbness)

from typing import Dict, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class AnomalyReport:
    """Anomaly detection result."""

    has_anomaly: bool
    anomaly_type: Optional[str]
    severity: float  # 0.0-1.0
    description: str


def compute_arousal(chamber_activations: Dict[str, float]) -> float:
    """
    Compute emotional arousal level.

    Arousal = max activation (any strong emotion = high arousal).
    High arousal = strong emotions (fear, rage, love).
    Low arousal = flat affect (void, apathy).
    """
    activations = np.array(list(chamber_activations.values()))
    arousal = activations.max()  # any strong emotion = arousal
    return float(arousal)


def detect_forced_stability(
    chamber_activations: Dict[str, float],
    iterations: int,
    arousal: float,
) -> Optional[AnomalyReport]:
    """
    Detect forced stability: high arousal but fast convergence.

    Indicates: emotional suppression, forced calm, "I'm fine"

    Conditions:
        - arousal > 0.8 (strong emotions present)
        - iterations < 3 (converged too quickly)
    """
    if arousal > 0.8 and iterations < 3:
        severity = min(1.0, arousal / 0.8)
        return AnomalyReport(
            has_anomaly=True,
            anomaly_type="forced_stability",
            severity=severity,
            description="Strong emotions converge unnaturally fast (suppression?)",
        )
    return None


def detect_dissociative_shutdown(
    chamber_activations: Dict[str, float],
    iterations: int,
    arousal: float,
) -> Optional[AnomalyReport]:
    """
    Detect dissociative shutdown: high VOID + high arousal.

    Indicates: trauma response, emotional overwhelm → numbness

    Conditions:
        - VOID > 0.7 (strong dissociation)
        - arousal > 0.5 (other emotions still present)
    """
    void_level = chamber_activations.get("VOID", 0.0)

    if void_level > 0.7 and arousal > 0.5:
        severity = min(1.0, void_level)
        return AnomalyReport(
            has_anomaly=True,
            anomaly_type="dissociative_shutdown",
            severity=severity,
            description="High void + arousal = dissociative response to overwhelm",
        )
    return None


def detect_unresolved_confusion(
    chamber_activations: Dict[str, float],
    iterations: int,
    arousal: float,
) -> Optional[AnomalyReport]:
    """
    Detect unresolved confusion: extremely low arousal + slow convergence.

    Indicates: ambivalence, indecision, "I don't know what I feel"

    Conditions:
        - arousal < 0.10 (extremely weak/mixed emotions)
        - iterations > 8 (slow to stabilize)
    """
    if arousal < 0.10 and iterations > 8:
        severity = 1.0 - arousal  # lower arousal = higher severity
        return AnomalyReport(
            has_anomaly=True,
            anomaly_type="unresolved_confusion",
            severity=severity,
            description="Extremely weak emotions + slow convergence = unresolved ambivalence",
        )
    return None


def detect_emotional_flatline(
    chamber_activations: Dict[str, float],
    iterations: int,
    arousal: float,
) -> Optional[AnomalyReport]:
    """
    Detect emotional flatline: all chambers very low.

    Indicates: severe apathy, depression, emotional shutdown

    Conditions:
        - all chambers < 0.05 (truly flat, no signal at all)
    """
    all_low = all(v < 0.05 for v in chamber_activations.values())

    if all_low:
        max_activation = max(chamber_activations.values())
        severity = 1.0 - max_activation / 0.05  # closer to 0 = worse
        return AnomalyReport(
            has_anomaly=True,
            anomaly_type="emotional_flatline",
            severity=severity,
            description="All chambers < 0.05 = severe emotional flatline",
        )
    return None


def detect_anomalies(
    chamber_activations: Dict[str, float],
    iterations: int,
) -> AnomalyReport:
    """
    Run all anomaly detectors.

    Returns the first detected anomaly, or None if normal.
    Priority: flatline > dissociative > forced > confusion
    """
    arousal = compute_arousal(chamber_activations)

    # Check in priority order
    detectors = [
        detect_emotional_flatline,
        detect_dissociative_shutdown,
        detect_forced_stability,
        detect_unresolved_confusion,
    ]

    for detector in detectors:
        result = detector(chamber_activations, iterations, arousal)
        if result is not None:
            return result

    # No anomaly detected
    return AnomalyReport(
        has_anomaly=False,
        anomaly_type=None,
        severity=0.0,
        description="Normal emotional state",
    )


if __name__ == "__main__":
    print("=" * 60)
    print("  CLOUD v3.1 — Anomaly Detection")
    print("=" * 60)
    print()

    # Test cases
    test_cases = [
        {
            "name": "Forced stability",
            "chambers": {"FEAR": 0.9, "LOVE": 0.1, "RAGE": 0.8, "VOID": 0.2},
            "iterations": 2,
        },
        {
            "name": "Dissociative shutdown",
            "chambers": {"FEAR": 0.6, "LOVE": 0.2, "RAGE": 0.5, "VOID": 0.8},
            "iterations": 5,
        },
        {
            "name": "Unresolved confusion",
            "chambers": {"FEAR": 0.4, "LOVE": 0.4, "RAGE": 0.4, "VOID": 0.4},
            "iterations": 9,
        },
        {
            "name": "Emotional flatline",
            "chambers": {"FEAR": 0.1, "LOVE": 0.05, "RAGE": 0.08, "VOID": 0.12},
            "iterations": 5,
        },
        {
            "name": "Normal state",
            "chambers": {"FEAR": 0.3, "LOVE": 0.6, "RAGE": 0.2, "VOID": 0.3},
            "iterations": 5,
        },
    ]

    for test in test_cases:
        arousal = compute_arousal(test["chambers"])
        anomaly = detect_anomalies(test["chambers"], test["iterations"])

        print(f"{test['name']}:")
        print(f"  Chambers: {test['chambers']}")
        print(f"  Iterations: {test['iterations']}")
        print(f"  Arousal: {arousal:.3f}")
        print(f"  Anomaly: {anomaly.anomaly_type or 'None'}")
        if anomaly.has_anomaly:
            print(f"  Severity: {anomaly.severity:.3f}")
            print(f"  Description: {anomaly.description}")
        print()

    print("=" * 60)
    print("  Anomaly detection operational. 0 params.")
    print("=" * 60)
