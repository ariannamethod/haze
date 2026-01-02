#!/usr/bin/env python3
# chambers.py — Chamber MLPs with Cross-Fire Stabilization
#
# Four chambers of emotion, each with its own MLP:
# - FEAR chamber:  100→64→32→1
# - LOVE chamber:  100→64→32→1
# - RAGE chamber:  100→64→32→1
# - VOID chamber:  100→64→32→1
#
# Total: 4 × 8,544 = ~34K params
#
# Cross-fire: chambers influence each other through coupling matrix,
# iterating until stabilization (5-10 iterations).

from __future__ import annotations
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from .anchors import CHAMBER_NAMES, COUPLING_MATRIX


def swish(x: np.ndarray) -> np.ndarray:
    """Swish activation: x * sigmoid(x)"""
    return x / (1.0 + np.exp(-x))


def swish_deriv(x: np.ndarray) -> np.ndarray:
    """Derivative of swish for backprop"""
    sig = 1.0 / (1.0 + np.exp(-x))
    return sig + x * sig * (1 - sig)


@dataclass
class ChamberMLP:
    """
    Single chamber MLP: 100→64→32→1

    Takes 100D resonance vector, outputs single activation value.

    Params:
        - W1: (100, 64) = 6,400
        - b1: (64,) = 64
        - W2: (64, 32) = 2,048
        - b2: (32,) = 32
        - W3: (32, 1) = 32
        - b3: (1,) = 1
        Total: 8,577 params per chamber
    """

    W1: np.ndarray  # (100, 64)
    b1: np.ndarray  # (64,)
    W2: np.ndarray  # (64, 32)
    b2: np.ndarray  # (32,)
    W3: np.ndarray  # (32, 1)
    b3: np.ndarray  # (1,)

    @classmethod
    def random_init(cls, seed: Optional[int] = None) -> "ChamberMLP":
        """Initialize with random weights (Xavier initialization)."""
        if seed is not None:
            np.random.seed(seed)

        # Xavier init: scale by sqrt(fan_in)
        W1 = np.random.randn(100, 64) * np.sqrt(2.0 / 100)
        b1 = np.zeros(64)

        W2 = np.random.randn(64, 32) * np.sqrt(2.0 / 64)
        b2 = np.zeros(32)

        W3 = np.random.randn(32, 1) * np.sqrt(2.0 / 32)
        b3 = np.zeros(1)

        return cls(W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)

    def forward(self, x: np.ndarray) -> float:
        """
        Forward pass: 100D resonances → scalar activation.

        Args:
            x: (100,) resonance vector

        Returns:
            scalar activation [0, 1]
        """
        # Layer 1: 100→64
        h1 = x @ self.W1 + self.b1
        a1 = swish(h1)

        # Layer 2: 64→32
        h2 = a1 @ self.W2 + self.b2
        a2 = swish(h2)

        # Layer 3: 32→1
        h3 = a2 @ self.W3 + self.b3

        # Sigmoid to get [0, 1] activation
        activation = 1.0 / (1.0 + np.exp(-h3[0]))

        return float(activation)

    def param_count(self) -> int:
        """Count total parameters in this MLP."""
        return (
            self.W1.size + self.b1.size +
            self.W2.size + self.b2.size +
            self.W3.size + self.b3.size
        )

    def save(self, path: Path) -> None:
        """Save weights to .npz file."""
        np.savez(
            path,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
            W3=self.W3,
            b3=self.b3,
        )

    @classmethod
    def load(cls, path: Path) -> "ChamberMLP":
        """Load weights from .npz file."""
        data = np.load(path)
        return cls(
            W1=data["W1"],
            b1=data["b1"],
            W2=data["W2"],
            b2=data["b2"],
            W3=data["W3"],
            b3=data["b3"],
        )


@dataclass
class CrossFireSystem:
    """
    Four chambers with cross-fire stabilization.

    Chambers:
        - FEAR, LOVE, RAGE, VOID

    Cross-fire loop:
        1. Each chamber computes activation from resonances
        2. Chambers influence each other via coupling matrix
        3. Iterate until convergence (max 10 iterations)
        4. Return final activations + iteration count
    """

    fear: ChamberMLP
    love: ChamberMLP
    rage: ChamberMLP
    void: ChamberMLP
    coupling: np.ndarray  # (4, 4) coupling matrix

    @classmethod
    def random_init(cls, seed: Optional[int] = None) -> "CrossFireSystem":
        """Initialize all chambers with random weights."""
        if seed is not None:
            base_seed = seed
        else:
            base_seed = np.random.randint(0, 10000)

        fear = ChamberMLP.random_init(seed=base_seed + 0)
        love = ChamberMLP.random_init(seed=base_seed + 1)
        rage = ChamberMLP.random_init(seed=base_seed + 2)
        void = ChamberMLP.random_init(seed=base_seed + 3)

        coupling = np.array(COUPLING_MATRIX, dtype=np.float32)

        return cls(
            fear=fear,
            love=love,
            rage=rage,
            void=void,
            coupling=coupling,
        )

    def stabilize(
        self,
        resonances: np.ndarray,
        max_iter: int = 10,
        threshold: float = 0.01,
        momentum: float = 0.7,
    ) -> Tuple[Dict[str, float], int]:
        """
        Run cross-fire stabilization loop.

        Args:
            resonances: (100,) initial resonance vector
            max_iter: max iterations before forced stop
            threshold: convergence threshold (sum of absolute changes)
            momentum: blend factor (0.7 = 70% old, 30% new)

        Returns:
            (chamber_activations, iterations_count)

        Example:
            activations, iters = system.stabilize(resonances)
            # → {"FEAR": 0.8, "LOVE": 0.2, "RAGE": 0.9, "VOID": 0.6}, 5
        """
        # Initial activations from resonances
        chambers = [self.fear, self.love, self.rage, self.void]
        activations = np.array([
            chamber.forward(resonances)
            for chamber in chambers
        ], dtype=np.float32)

        # Stabilization loop
        for iteration in range(max_iter):
            # Compute influence from other chambers
            influence = self.coupling @ activations

            # Blend: momentum * old + (1 - momentum) * influence
            new_activations = momentum * activations + (1 - momentum) * influence

            # Clip to [0, 1]
            new_activations = np.clip(new_activations, 0.0, 1.0)

            # Check convergence
            delta = np.abs(new_activations - activations).sum()
            activations = new_activations

            if delta < threshold:
                # Converged!
                result = dict(zip(CHAMBER_NAMES, activations))
                return result, iteration + 1

        # Max iterations reached
        result = dict(zip(CHAMBER_NAMES, activations))
        return result, max_iter

    def param_count(self) -> int:
        """Total parameters in all chambers."""
        return sum([
            self.fear.param_count(),
            self.love.param_count(),
            self.rage.param_count(),
            self.void.param_count(),
        ])

    def save(self, models_dir: Path) -> None:
        """Save all chamber weights to models/ directory."""
        models_dir.mkdir(parents=True, exist_ok=True)
        self.fear.save(models_dir / "chamber_fear.npz")
        self.love.save(models_dir / "chamber_love.npz")
        self.rage.save(models_dir / "chamber_rage.npz")
        self.void.save(models_dir / "chamber_void.npz")
        print(f"[chambers] saved to {models_dir}")

    @classmethod
    def load(cls, models_dir: Path) -> "CrossFireSystem":
        """Load all chamber weights from models/ directory."""
        fear = ChamberMLP.load(models_dir / "chamber_fear.npz")
        love = ChamberMLP.load(models_dir / "chamber_love.npz")
        rage = ChamberMLP.load(models_dir / "chamber_rage.npz")
        void = ChamberMLP.load(models_dir / "chamber_void.npz")

        coupling = np.array(COUPLING_MATRIX, dtype=np.float32)

        print(f"[chambers] loaded from {models_dir}")
        return cls(
            fear=fear,
            love=love,
            rage=rage,
            void=void,
            coupling=coupling,
        )


if __name__ == "__main__":
    print("=" * 60)
    print("  CLOUD v3.0 — Chamber Cross-Fire System")
    print("=" * 60)
    print()

    # Initialize random system
    system = CrossFireSystem.random_init(seed=42)
    print(f"Initialized cross-fire system")
    print(f"Total params: {system.param_count():,}")
    print()

    # Test with random resonances
    print("Testing stabilization with random resonances:")
    resonances = np.random.rand(100).astype(np.float32)
    print(f"  Input: 100D resonance vector (mean={resonances.mean():.3f})")
    print()

    activations, iterations = system.stabilize(resonances)

    print("  Chamber activations after cross-fire:")
    for chamber, value in activations.items():
        bar = "█" * int(value * 40)
        print(f"    {chamber:6s}: {value:.3f}  {bar}")
    print(f"\n  Converged in {iterations} iterations")
    print()

    # Test convergence speed with different inputs
    print("Testing convergence speed:")
    test_cases = [
        ("random uniform", np.random.rand(100)),
        ("all high", np.ones(100) * 0.9),
        ("all low", np.ones(100) * 0.1),
        ("sparse", np.random.rand(100) * 0.1),
    ]

    for name, resonances in test_cases:
        _, iters = system.stabilize(resonances)
        print(f"  {name:15s}: {iters:2d} iterations")
    print()

    # Test saving/loading
    print("Testing save/load:")
    models_dir = Path("./models")
    system.save(models_dir)

    system2 = CrossFireSystem.load(models_dir)
    activations2, _ = system2.stabilize(test_cases[0][1])

    match = all(
        abs(activations[k] - activations2[k]) < 1e-6
        for k in CHAMBER_NAMES
    )
    print(f"  Save/load {'✓' if match else '✗'}")
    print()

    print("=" * 60)
    print("  Cross-fire system operational.")
    print("=" * 60)
