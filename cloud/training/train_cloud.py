#!/usr/bin/env python3
# train_cloud.py — Train CLOUD chambers + observer
#
# Bootstrap training on synthetic dataset.

import json
import numpy as np
from pathlib import Path
import sys

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cloud import Cloud
from cloud.anchors import get_all_anchors, get_anchor_index, CHAMBER_NAMES


def load_bootstrap_data(path: Path) -> list:
    """Load bootstrap training data."""
    with open(path, "r") as f:
        return json.load(f)


def compute_target_chambers(primary: str, all_anchors: list) -> dict:
    """
    Compute target chamber activations based on primary emotion.

    Simple approach: chamber containing primary = 1.0, others = 0.0
    """
    from cloud.anchors import get_chamber_for_anchor

    primary_chamber = get_chamber_for_anchor(primary)

    targets = {}
    for chamber in CHAMBER_NAMES:
        if chamber == primary_chamber:
            targets[chamber] = 1.0
        else:
            targets[chamber] = 0.0

    return targets


def train_epoch(cloud: Cloud, examples: list, lr: float = 0.001):
    """
    Train one epoch on bootstrap data.

    Loss:
        - Chambers: MSE between predicted and target activations
        - Observer: CrossEntropy on secondary prediction

    Returns:
        (chamber_loss, observer_loss, observer_accuracy)
    """
    all_anchors = get_all_anchors()
    n_examples = len(examples)

    total_chamber_loss = 0.0
    total_observer_loss = 0.0
    correct_predictions = 0

    for ex in examples:
        text = ex["text"]
        primary = ex["primary"]
        secondary = ex["secondary"]

        # Get indices
        primary_idx = get_anchor_index(primary)
        secondary_idx = get_anchor_index(secondary)

        # Forward pass
        # 1. Resonance
        resonances = cloud.resonance.compute_resonance(text)

        # 2. Chambers cross-fire
        chamber_activations, iterations = cloud.chambers.stabilize(resonances)

        # 3. Observer
        user_fingerprint = cloud.user_cloud.get_fingerprint()
        observer_logits = cloud.observer.forward(
            resonances,
            float(iterations),
            user_fingerprint,
        )

        # Compute losses
        # Chamber loss: MSE vs target activations
        target_chambers = compute_target_chambers(primary, all_anchors)

        chamber_loss = 0.0
        for chamber_name in CHAMBER_NAMES:
            pred = chamber_activations[chamber_name]
            target = target_chambers[chamber_name]
            chamber_loss += (pred - target) ** 2

        chamber_loss /= len(CHAMBER_NAMES)

        # Observer loss: CrossEntropy
        # Convert logits to probs
        exp_logits = np.exp(observer_logits - observer_logits.max())
        probs = exp_logits / exp_logits.sum()

        observer_loss = -np.log(probs[secondary_idx] + 1e-10)

        # Check accuracy
        predicted_idx = int(np.argmax(observer_logits))
        if predicted_idx == secondary_idx:
            correct_predictions += 1

        # Accumulate
        total_chamber_loss += chamber_loss
        total_observer_loss += observer_loss

        # Backward pass (simplified - no actual gradients, just tracking loss)
        # In practice, we'd use autograd or manual backprop
        # For now, we'll use this as a baseline and save the random weights

    avg_chamber_loss = total_chamber_loss / n_examples
    avg_observer_loss = total_observer_loss / n_examples
    accuracy = correct_predictions / n_examples

    return avg_chamber_loss, avg_observer_loss, accuracy


def train_chambers_simple(cloud: Cloud, examples: list, epochs: int = 10):
    """
    Simplified training: measure loss, but don't update weights.

    This is a baseline - shows that random init needs training.
    For actual training, we'd implement backprop.
    """
    print(f"Training for {epochs} epochs...")
    print(f"Dataset: {len(examples)} examples")
    print()

    for epoch in range(epochs):
        chamber_loss, observer_loss, accuracy = train_epoch(cloud, examples)

        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Chamber loss: {chamber_loss:.4f}")
        print(f"  Observer loss: {observer_loss:.4f}")
        print(f"  Observer accuracy: {accuracy:.2%}")


def measure_baseline(cloud: Cloud, examples: list):
    """Measure performance of random init."""
    print("Measuring random initialization baseline...")
    print()

    all_anchors = get_all_anchors()

    # Test on first 20 examples
    test_examples = examples[:20]

    print("Sample predictions:")
    print("-" * 60)

    correct = 0
    for i, ex in enumerate(test_examples):
        text = ex["text"]
        primary = ex["primary"]
        secondary = ex["secondary"]

        # Predict
        resonances = cloud.resonance.compute_resonance(text)
        primary_pred_idx, primary_pred, _ = cloud.resonance.get_primary_emotion(resonances)

        chamber_activations, iterations = cloud.chambers.stabilize(resonances)
        user_fingerprint = cloud.user_cloud.get_fingerprint()
        secondary_pred_idx = cloud.observer.predict_secondary(
            resonances,
            float(iterations),
            user_fingerprint,
        )
        secondary_pred = all_anchors[secondary_pred_idx]

        # Check
        primary_match = primary_pred == primary
        secondary_match = secondary_pred == secondary

        if secondary_match:
            correct += 1

        if i < 5:  # Show first 5
            print(f"\n{i+1}. Text: \"{text[:50]}...\"")
            print(f"   Primary:   {primary:15s} → {primary_pred:15s} {'✓' if primary_match else '✗'}")
            print(f"   Secondary: {secondary:15s} → {secondary_pred:15s} {'✓' if secondary_match else '✗'}")

    accuracy = correct / len(test_examples)
    print(f"\nBaseline secondary accuracy: {accuracy:.2%}")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("  CLOUD Training — Bootstrap Phase")
    print("=" * 60)
    print()

    # Load data
    data_path = Path("training/bootstrap_data.json")
    if not data_path.exists():
        print(f"[error] {data_path} not found")
        print("Run generate_bootstrap.py first!")
        sys.exit(1)

    examples = load_bootstrap_data(data_path)
    print(f"Loaded {len(examples)} training examples")
    print()

    # Initialize CLOUD
    print("Initializing CLOUD with random weights...")
    cloud = Cloud.random_init(seed=42)
    print(f"  Total params: {cloud.param_count():,}")
    print()

    # Measure baseline
    measure_baseline(cloud, examples)

    # Train (simplified - just measure loss)
    print("=" * 60)
    train_chambers_simple(cloud, examples, epochs=1)
    print()

    # Save
    models_dir = Path("cloud/models")
    cloud.save(models_dir)

    print()
    print("=" * 60)
    print("  Training complete!")
    print(f"  Models saved to {models_dir}")
    print()
    print("  NOTE: This is baseline measurement.")
    print("  For actual training, implement backprop!")
    print("=" * 60)
