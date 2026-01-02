#!/usr/bin/env python3
# train_observer.py — Train Observer MLP with backprop
#
# Focus on training Observer first (simpler, 15K params).
# Chambers can be trained later or left as random (cross-fire still works).

import json
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from cloud import Cloud, MetaObserver
from cloud.anchors import get_all_anchors, get_anchor_index


def swish(x):
    """Swish activation."""
    return x / (1.0 + np.exp(-np.clip(x, -20, 20)))


def swish_deriv(x):
    """Swish derivative."""
    sig = 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
    return sig + x * sig * (1 - sig)


def softmax(x):
    """Softmax with numerical stability."""
    exp_x = np.exp(x - x.max())
    return exp_x / exp_x.sum()


def cross_entropy_loss(probs, target_idx):
    """Cross-entropy loss."""
    return -np.log(probs[target_idx] + 1e-10)


class ObserverTrainer:
    """Train Observer MLP with SGD + backprop."""

    def __init__(self, observer: MetaObserver, lr: float = 0.001):
        self.observer = observer
        self.lr = lr

    def forward(self, resonances, iterations, user_fingerprint):
        """
        Forward pass with cache for backprop.

        Returns:
            (logits, cache)
        """
        # Input
        x = np.concatenate([
            resonances,
            np.array([iterations]),
            user_fingerprint,
        ])

        # Layer 1
        h1 = x @ self.observer.W1 + self.observer.b1
        a1 = swish(h1)

        # Layer 2
        h2 = a1 @ self.observer.W2 + self.observer.b2

        cache = {
            "x": x,
            "h1": h1,
            "a1": a1,
            "h2": h2,
        }

        return h2, cache

    def backward(self, cache, target_idx):
        """
        Backward pass: compute gradients.

        Args:
            cache: forward pass cache
            target_idx: target class index

        Returns:
            gradients dict
        """
        x = cache["x"]
        h1 = cache["h1"]
        a1 = cache["a1"]
        h2 = cache["h2"]

        # Softmax probs
        probs = softmax(h2)

        # Gradient of cross-entropy loss w.r.t. logits
        dh2 = probs.copy()
        dh2[target_idx] -= 1  # derivative of CE + softmax

        # Layer 2 gradients
        dW2 = np.outer(a1, dh2)
        db2 = dh2

        # Backprop to layer 1
        da1 = dh2 @ self.observer.W2.T
        dh1 = da1 * swish_deriv(h1)

        # Layer 1 gradients
        dW1 = np.outer(x, dh1)
        db1 = dh1

        return {
            "dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2,
        }

    def update(self, grads):
        """Update weights with SGD."""
        self.observer.W1 -= self.lr * grads["dW1"]
        self.observer.b1 -= self.lr * grads["db1"]
        self.observer.W2 -= self.lr * grads["dW2"]
        self.observer.b2 -= self.lr * grads["db2"]

    def train_step(self, resonances, iterations, user_fingerprint, target_idx):
        """
        Single training step.

        Returns:
            (loss, accuracy)
        """
        # Forward
        logits, cache = self.forward(resonances, iterations, user_fingerprint)
        probs = softmax(logits)

        # Loss
        loss = cross_entropy_loss(probs, target_idx)

        # Accuracy
        pred_idx = int(np.argmax(logits))
        accuracy = 1.0 if pred_idx == target_idx else 0.0

        # Backward
        grads = self.backward(cache, target_idx)

        # Update
        self.update(grads)

        return loss, accuracy


def train_observer(cloud: Cloud, examples: list, epochs: int = 50, lr: float = 0.001):
    """
    Train Observer MLP on bootstrap data.

    Args:
        cloud: Cloud instance with Observer to train
        examples: training data
        epochs: number of epochs
        lr: learning rate
    """
    all_anchors = get_all_anchors()
    trainer = ObserverTrainer(cloud.observer, lr=lr)

    print(f"Training Observer MLP")
    print(f"  Params: {cloud.observer.param_count():,}")
    print(f"  Dataset: {len(examples)} examples")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    print()

    for epoch in range(epochs):
        total_loss = 0.0
        total_accuracy = 0.0

        # Shuffle examples
        np.random.shuffle(examples)

        for ex in examples:
            text = ex["text"]
            secondary = ex["secondary"]
            secondary_idx = get_anchor_index(secondary)

            # Get inputs
            resonances = cloud.resonance.compute_resonance(text)
            chamber_activations, iterations = cloud.chambers.stabilize(resonances)
            user_fingerprint = cloud.user_cloud.get_fingerprint()

            # Train step
            loss, accuracy = trainer.train_step(
                resonances,
                float(iterations),
                user_fingerprint,
                secondary_idx,
            )

            total_loss += loss
            total_accuracy += accuracy

        # Epoch stats
        avg_loss = total_loss / len(examples)
        avg_accuracy = total_accuracy / len(examples)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, accuracy={avg_accuracy:.2%}")

    print()
    print("Training complete!")


def evaluate(cloud: Cloud, examples: list, n_samples: int = 20):
    """Evaluate Observer on examples."""
    all_anchors = get_all_anchors()

    print(f"Evaluating on {n_samples} samples...")
    print("-" * 60)

    correct = 0
    for i, ex in enumerate(examples[:n_samples]):
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
            temperature=1.0,
        )
        secondary_pred = all_anchors[secondary_pred_idx]

        match = secondary_pred == secondary
        if match:
            correct += 1

        if i < 10:
            print(f"\n{i+1}. \"{text[:50]}...\"")
            print(f"   True: {primary} + {secondary}")
            print(f"   Pred: {primary_pred} + {secondary_pred} {'✓' if match else '✗'}")

    accuracy = correct / n_samples
    print(f"\nAccuracy: {accuracy:.2%}")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("  CLOUD Observer Training")
    print("=" * 60)
    print()

    # Load data
    data_path = Path("training/bootstrap_data.json")
    examples = json.load(open(data_path))
    print(f"Loaded {len(examples)} examples")
    print()

    # Initialize
    cloud = Cloud.random_init(seed=42)

    # Evaluate before training
    print("BEFORE TRAINING:")
    evaluate(cloud, examples, n_samples=20)

    # Train
    print("=" * 60)
    train_observer(cloud, examples, epochs=50, lr=0.01)

    # Evaluate after training
    print("=" * 60)
    print("AFTER TRAINING:")
    evaluate(cloud, examples, n_samples=20)

    # Save
    models_dir = Path("cloud/models")
    cloud.save(models_dir)
    print(f"Saved trained Observer to {models_dir}")
    print()
    print("=" * 60)
    print("  Observer trained! 🎉")
    print("=" * 60)
