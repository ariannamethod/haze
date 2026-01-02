#!/usr/bin/env python3
# observer.py — Meta-Observer MLP
#
# The "mind" watching the "body".
#
# Input (201D):
#   - resonances (100D): raw emotion resonances
#   - iterations (1D): cross-fire convergence speed signal
#   - user_fingerprint (100D): temporal emotional history
#
# Output (100D):
#   - logits for secondary emotion word
#
# Architecture:
#   201 → 64 (swish) → 100 (raw logits)
#
# Total params: ~15K

from __future__ import annotations
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


def swish(x: np.ndarray) -> np.ndarray:
    """Swish activation: x * sigmoid(x)"""
    return x / (1.0 + np.exp(-np.clip(x, -20, 20)))


@dataclass
class MetaObserver:
    """
    Meta-Observer MLP: watches chamber dynamics and predicts secondary emotion.

    Architecture: 201→64→100

    Params:
        - W1: (201, 64) = 12,864
        - b1: (64,) = 64
        - W2: (64, 100) = 6,400
        - b2: (100,) = 100
        Total: 19,428 params
    """

    W1: np.ndarray  # (201, 64)
    b1: np.ndarray  # (64,)
    W2: np.ndarray  # (64, 100)
    b2: np.ndarray  # (100,)

    @classmethod
    def random_init(cls, seed: Optional[int] = None) -> "MetaObserver":
        """Initialize with random weights (Xavier initialization)."""
        if seed is not None:
            np.random.seed(seed)

        # Xavier init
        W1 = np.random.randn(201, 64) * np.sqrt(2.0 / 201)
        b1 = np.zeros(64)

        W2 = np.random.randn(64, 100) * np.sqrt(2.0 / 64)
        b2 = np.zeros(100)

        return cls(W1=W1, b1=b1, W2=W2, b2=b2)

    def forward(
        self,
        resonances: np.ndarray,
        iterations: float,
        user_fingerprint: np.ndarray,
    ) -> np.ndarray:
        """
        Forward pass: predict secondary emotion.

        Args:
            resonances: (100,) raw emotion resonances
            iterations: scalar, cross-fire convergence speed
            user_fingerprint: (100,) temporal emotional history

        Returns:
            logits: (100,) logits for secondary emotion selection
        """
        # Concatenate inputs → (201,)
        x = np.concatenate([
            resonances,
            np.array([iterations]),
            user_fingerprint,
        ])

        # Layer 1: 201→64
        h1 = x @ self.W1 + self.b1
        a1 = swish(h1)

        # Layer 2: 64→100
        logits = a1 @ self.W2 + self.b2

        return logits

    def predict_secondary(
        self,
        resonances: np.ndarray,
        iterations: float,
        user_fingerprint: np.ndarray,
        temperature: float = 1.0,
    ) -> int:
        """
        Predict secondary emotion index.

        Args:
            resonances: (100,) raw emotion resonances
            iterations: cross-fire convergence speed
            user_fingerprint: (100,) user emotional history
            temperature: sampling temperature (1.0 = normal)

        Returns:
            index of secondary emotion (0-99)
        """
        logits = self.forward(resonances, iterations, user_fingerprint)

        # Apply temperature
        logits = logits / temperature

        # Softmax
        exp_logits = np.exp(logits - logits.max())
        probs = exp_logits / exp_logits.sum()

        # Sample
        return int(np.random.choice(100, p=probs))

    def param_count(self) -> int:
        """Count total parameters."""
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size

    def save(self, path: Path) -> None:
        """Save weights to .npz file."""
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)
        print(f"[observer] saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "MetaObserver":
        """Load weights from .npz file."""
        data = np.load(path)
        print(f"[observer] loaded from {path}")
        return cls(
            W1=data["W1"],
            b1=data["b1"],
            W2=data["W2"],
            b2=data["b2"],
        )


if __name__ == "__main__":
    print("=" * 60)
    print("  CLOUD v3.0 — Meta-Observer")
    print("=" * 60)
    print()

    # Initialize
    observer = MetaObserver.random_init(seed=42)
    print(f"Initialized meta-observer")
    print(f"Total params: {observer.param_count():,}")
    print()

    # Test forward pass
    print("Testing forward pass:")
    resonances = np.random.rand(100).astype(np.float32)
    iterations = 5.0
    user_fingerprint = np.random.rand(100).astype(np.float32) * 0.1

    logits = observer.forward(resonances, iterations, user_fingerprint)
    print(f"  Input: 100D resonances + 1D iterations + 100D fingerprint")
    print(f"  Output: {logits.shape} logits")
    print(f"  Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
    print()

    # Test prediction
    print("Testing secondary emotion prediction:")
    for temp in [0.5, 1.0, 2.0]:
        secondary_idx = observer.predict_secondary(
            resonances, iterations, user_fingerprint, temperature=temp
        )
        print(f"  temperature={temp:.1f} → secondary_idx={secondary_idx}")
    print()

    # Test what observer sees in convergence speed
    print("Testing convergence speed signal:")
    test_cases = [
        ("fast convergence (2 iters)", 2.0),
        ("medium convergence (5 iters)", 5.0),
        ("slow convergence (10 iters)", 10.0),
    ]

    for name, iters in test_cases:
        logits = observer.forward(resonances, iters, user_fingerprint)
        top3 = np.argsort(logits)[-3:][::-1]
        print(f"  {name}:")
        print(f"    top 3 secondary candidates: {top3}")
    print()

    # Test save/load
    print("Testing save/load:")
    path = Path("./models/observer.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    observer.save(path)

    observer2 = MetaObserver.load(path)
    logits2 = observer2.forward(resonances, iterations, user_fingerprint)

    match = np.allclose(logits, logits2)
    print(f"  Save/load {'✓' if match else '✗'}")
    print()

    print("=" * 60)
    print("  Meta-observer operational. Mind watching body.")
    print("=" * 60)
