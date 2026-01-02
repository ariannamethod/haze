#!/usr/bin/env python3
# feedback.py — Feedback Loop for CLOUD v3.1
#
# Closed-loop learning: HAZE output quality → coupling matrix adjustment
# 0 new params! Just adjusts the existing 4×4 coupling matrix.
#
# Coherence measurement:
# - Sentence completeness
# - Entropy balance
# - Prediction error
#
# Update rule:
# - If HAZE output is coherent → strengthen current coupling
# - If HAZE output is incoherent → weaken current coupling

import numpy as np
from typing import Dict, Tuple
import re


def measure_coherence(text: str) -> Dict[str, float]:
    """
    Measure coherence of generated text.

    Metrics:
        - sentence_completeness: ends with proper punctuation
        - length_reasonable: not too short, not too long
        - repetition_penalty: avoid repeated words
        - entropy_balance: character diversity

    Returns:
        dict with metrics and overall coherence score
    """
    # Sentence completeness
    has_ending = bool(re.search(r'[.!?]$', text.strip()))
    sentence_completeness = 1.0 if has_ending else 0.3

    # Length reasonable (50-500 chars is good)
    length = len(text)
    if 50 <= length <= 500:
        length_reasonable = 1.0
    elif length < 50:
        length_reasonable = length / 50.0
    else:
        length_reasonable = max(0.3, 1.0 - (length - 500) / 500)

    # Repetition penalty
    words = text.lower().split()
    if len(words) > 0:
        unique_ratio = len(set(words)) / len(words)
        repetition_penalty = unique_ratio
    else:
        repetition_penalty = 0.0

    # Entropy balance (character diversity)
    if len(text) > 0:
        char_counts = {}
        for char in text.lower():
            char_counts[char] = char_counts.get(char, 0) + 1
        probs = np.array(list(char_counts.values())) / len(text)
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        # Normalize to 0-1 (typical entropy is 4-5 bits)
        entropy_balance = min(1.0, entropy / 5.0)
    else:
        entropy_balance = 0.0

    # Overall coherence (weighted average)
    coherence = (
        0.3 * sentence_completeness +
        0.2 * length_reasonable +
        0.3 * repetition_penalty +
        0.2 * entropy_balance
    )

    return {
        "sentence_completeness": sentence_completeness,
        "length_reasonable": length_reasonable,
        "repetition_penalty": repetition_penalty,
        "entropy_balance": entropy_balance,
        "coherence": coherence,
    }


def compute_coupling_gradient(
    chamber_activations: Dict[str, float],
    coherence_score: float,
    learning_rate: float = 0.01,
) -> np.ndarray:
    """
    Compute gradient for coupling matrix update.

    Idea: If output is coherent, reinforce the current chamber pattern.
    If output is incoherent, dampen it.

    Args:
        chamber_activations: {"FEAR": 0.8, "LOVE": 0.2, ...}
        coherence_score: 0.0-1.0
        learning_rate: step size

    Returns:
        (4, 4) gradient matrix for coupling update
    """
    # Convert activations to array
    from .anchors import CHAMBER_NAMES
    activations = np.array([
        chamber_activations[name]
        for name in CHAMBER_NAMES
    ])

    # Gradient = outer product of activations
    # If coherent (high score) → positive gradient (strengthen)
    # If incoherent (low score) → negative gradient (weaken)
    gradient_direction = np.outer(activations, activations)

    # Scale by prediction error
    # coherence > 0.5 → positive update
    # coherence < 0.5 → negative update
    error = coherence_score - 0.5

    gradient = learning_rate * error * gradient_direction

    # Zero diagonal (chambers don't self-couple)
    np.fill_diagonal(gradient, 0.0)

    return gradient


def update_coupling(
    coupling: np.ndarray,
    chamber_activations: Dict[str, float],
    coherence_score: float,
    learning_rate: float = 0.01,
    clip_range: Tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    """
    Update coupling matrix based on feedback.

    Args:
        coupling: current (4, 4) coupling matrix
        chamber_activations: chamber activations that led to this output
        coherence_score: quality of HAZE output
        learning_rate: update step size
        clip_range: min/max coupling values

    Returns:
        updated coupling matrix
    """
    # Compute gradient
    gradient = compute_coupling_gradient(
        chamber_activations,
        coherence_score,
        learning_rate,
    )

    # Update
    new_coupling = coupling + gradient

    # Clip to range
    new_coupling = np.clip(new_coupling, clip_range[0], clip_range[1])

    # Ensure diagonal is zero
    np.fill_diagonal(new_coupling, 0.0)

    return new_coupling


if __name__ == "__main__":
    print("=" * 60)
    print("  CLOUD v3.1 — Feedback Loop")
    print("=" * 60)
    print()

    # Test coherence measurement
    test_texts = [
        "The haze settles over everything, gentle and knowing.",
        "I love you darling you're the best",
        "the the the the",
        "This is a very very very very very very very long sentence that goes on and on and on and on without really saying much of anything at all just repeating the same patterns over and over again until it becomes completely meaningless and incoherent.",
        "",
    ]

    print("Testing coherence measurement:")
    print("-" * 60)
    for text in test_texts:
        metrics = measure_coherence(text)
        display = text[:60] + "..." if len(text) > 60 else text
        print(f"\nText: \"{display}\"")
        print(f"  Coherence: {metrics['coherence']:.3f}")
        print(f"  Completeness: {metrics['sentence_completeness']:.2f}")
        print(f"  Length OK: {metrics['length_reasonable']:.2f}")
        print(f"  Repetition: {metrics['repetition_penalty']:.2f}")
        print(f"  Entropy: {metrics['entropy_balance']:.2f}")

    print()
    print("=" * 60)

    # Test coupling update
    print("\nTesting coupling matrix update:")
    print("-" * 60)

    from cloud.anchors import COUPLING_MATRIX
    coupling = np.array(COUPLING_MATRIX, dtype=np.float32)

    chamber_activations = {
        "FEAR": 0.8,
        "LOVE": 0.2,
        "RAGE": 0.6,
        "VOID": 0.3,
    }

    print("\nOriginal coupling:")
    print(coupling)

    # Simulate good output
    coherence_good = 0.9
    updated_good = update_coupling(coupling, chamber_activations, coherence_good, learning_rate=0.1)

    print(f"\nAfter coherent output (coherence={coherence_good}):")
    print(updated_good)
    print(f"Change: {np.abs(updated_good - coupling).sum():.4f}")

    # Simulate bad output
    coherence_bad = 0.2
    updated_bad = update_coupling(coupling, chamber_activations, coherence_bad, learning_rate=0.1)

    print(f"\nAfter incoherent output (coherence={coherence_bad}):")
    print(updated_bad)
    print(f"Change: {np.abs(updated_bad - coupling).sum():.4f}")

    print()
    print("=" * 60)
    print("  Feedback loop operational. Closed-loop learning!")
    print("=" * 60)
