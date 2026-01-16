#!/usr/bin/env python3
# chambers.py — Chamber MLPs with Cross-Fire Stabilization
#
# Six chambers of emotion, each with its own deeper MLP:
# - FEAR chamber:  100→128→64→32→1
# - LOVE chamber:  100→128→64→32→1
# - RAGE chamber:  100→128→64→32→1
# - VOID chamber:  100→128→64→32→1
# - FLOW chamber:  100→128→64→32→1
# - COMPLEX chamber: 100→128→64→32→1
#
# Total: 6 × ~17K = ~102K params
#
# Cross-fire: chambers influence each other through coupling matrix,
# iterating until stabilization (5-10 iterations).

from __future__ import annotations
import asyncio
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from .anchors import CHAMBER_NAMES, COUPLING_MATRIX, CHAMBER_NAMES_EXTENDED, COUPLING_MATRIX_EXTENDED

# Decay rates per chamber (per iteration tick)
# Evolutionary psychology: fear lingers, rage fades, love stable, void persistent
DECAY_RATES = {
    "FEAR": 0.90,  # fear lingers (evolutionary advantage)
    "LOVE": 0.93,  # attachment stable
    "RAGE": 0.85,  # anger fades fast (high energy cost)
    "VOID": 0.97,  # numbness persistent (protective dissociation)
    "FLOW": 0.88,  # curiosity is transient, shifts quickly
    "COMPLEX": 0.94,  # complex emotions are stable but deep
}


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
    Single chamber MLP: 100→128→64→32→1 (deeper for 200K model)

    Takes 100D resonance vector, outputs single activation value.

    Params:
        - W1: (100, 128) = 12,800
        - b1: (128,) = 128
        - W2: (128, 64) = 8,192
        - b2: (64,) = 64
        - W3: (64, 32) = 2,048
        - b3: (32,) = 32
        - W4: (32, 1) = 32
        - b4: (1,) = 1
        Total: ~23K params per chamber
    """

    W1: np.ndarray  # (100, 128)
    b1: np.ndarray  # (128,)
    W2: np.ndarray  # (128, 64)
    b2: np.ndarray  # (64,)
    W3: np.ndarray  # (64, 32)
    b3: np.ndarray  # (32,)
    W4: np.ndarray  # (32, 1)
    b4: np.ndarray  # (1,)

    @classmethod
    def random_init(cls, seed: Optional[int] = None) -> "ChamberMLP":
        """Initialize with random weights (Xavier initialization)."""
        if seed is not None:
            np.random.seed(seed)

        # Xavier init: scale by sqrt(fan_in)
        W1 = np.random.randn(100, 128) * np.sqrt(2.0 / 100)
        b1 = np.zeros(128)

        W2 = np.random.randn(128, 64) * np.sqrt(2.0 / 128)
        b2 = np.zeros(64)

        W3 = np.random.randn(64, 32) * np.sqrt(2.0 / 64)
        b3 = np.zeros(32)

        W4 = np.random.randn(32, 1) * np.sqrt(2.0 / 32)
        b4 = np.zeros(1)

        return cls(W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3, W4=W4, b4=b4)

    def forward(self, x: np.ndarray) -> float:
        """
        Forward pass: 100D resonances → scalar activation.

        Args:
            x: (100,) resonance vector

        Returns:
            scalar activation [0, 1]
        """
        # Layer 1: 100→128
        h1 = x @ self.W1 + self.b1
        a1 = swish(h1)

        # Layer 2: 128→64
        h2 = a1 @ self.W2 + self.b2
        a2 = swish(h2)

        # Layer 3: 64→32
        h3 = a2 @ self.W3 + self.b3
        a3 = swish(h3)

        # Layer 4: 32→1
        h4 = a3 @ self.W4 + self.b4

        # Sigmoid to get [0, 1] activation
        activation = 1.0 / (1.0 + np.exp(-h4[0]))

        return float(activation)

    def param_count(self) -> int:
        """Count total parameters in this MLP."""
        return (
            self.W1.size + self.b1.size +
            self.W2.size + self.b2.size +
            self.W3.size + self.b3.size +
            self.W4.size + self.b4.size
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
            W4=self.W4,
            b4=self.b4,
        )

    @classmethod
    def load(cls, path: Path) -> "ChamberMLP":
        """Load weights from .npz file."""
        data = np.load(path)
        # Handle backwards compatibility with old 3-layer architecture
        if "W4" in data:
            return cls(
                W1=data["W1"],
                b1=data["b1"],
                W2=data["W2"],
                b2=data["b2"],
                W3=data["W3"],
                b3=data["b3"],
                W4=data["W4"],
                b4=data["b4"],
            )
        else:
            # Old 3-layer format - reinitialize with new 4-layer architecture
            print(f"[chambers] old format detected in {path}, reinitializing with 4-layer architecture")
            return cls.random_init()


@dataclass
class CrossFireSystem:
    """
    Six chambers with cross-fire stabilization (200K model).

    Chambers:
        - FEAR, LOVE, RAGE, VOID (original)
        - FLOW, COMPLEX (extended for richer emotion detection)

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
    flow: ChamberMLP
    complex: ChamberMLP
    coupling: np.ndarray  # (6, 6) coupling matrix

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
        flow = ChamberMLP.random_init(seed=base_seed + 4)
        complex_chamber = ChamberMLP.random_init(seed=base_seed + 5)

        coupling = np.array(COUPLING_MATRIX_EXTENDED, dtype=np.float32)

        return cls(
            fear=fear,
            love=love,
            rage=rage,
            void=void,
            flow=flow,
            complex=complex_chamber,
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
            # → {"FEAR": 0.8, "LOVE": 0.2, ...}, 5
        """
        # Initial activations from resonances
        chambers = [self.fear, self.love, self.rage, self.void, self.flow, self.complex]
        activations = np.array([
            chamber.forward(resonances)
            for chamber in chambers
        ], dtype=np.float32)

        # Decay rates array (6 chambers)
        decay_array = np.array([
            DECAY_RATES["FEAR"],
            DECAY_RATES["LOVE"],
            DECAY_RATES["RAGE"],
            DECAY_RATES["VOID"],
            DECAY_RATES["FLOW"],
            DECAY_RATES["COMPLEX"],
        ], dtype=np.float32)

        # Stabilization loop
        for iteration in range(max_iter):
            # Apply decay (emotions fade over time)
            activations = activations * decay_array

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
                result = dict(zip(CHAMBER_NAMES_EXTENDED, activations))
                return result, iteration + 1

        # Max iterations reached
        result = dict(zip(CHAMBER_NAMES_EXTENDED, activations))
        return result, max_iter

    def param_count(self) -> int:
        """Total parameters in all chambers."""
        return sum([
            self.fear.param_count(),
            self.love.param_count(),
            self.rage.param_count(),
            self.void.param_count(),
            self.flow.param_count(),
            self.complex.param_count(),
        ])

    def save(self, models_dir: Path) -> None:
        """Save all chamber weights to models/ directory."""
        models_dir.mkdir(parents=True, exist_ok=True)
        self.fear.save(models_dir / "chamber_fear.npz")
        self.love.save(models_dir / "chamber_love.npz")
        self.rage.save(models_dir / "chamber_rage.npz")
        self.void.save(models_dir / "chamber_void.npz")
        self.flow.save(models_dir / "chamber_flow.npz")
        self.complex.save(models_dir / "chamber_complex.npz")
        print(f"[chambers] saved to {models_dir}")

    @classmethod
    def load(cls, models_dir: Path) -> "CrossFireSystem":
        """Load all chamber weights from models/ directory."""
        fear = ChamberMLP.load(models_dir / "chamber_fear.npz")
        love = ChamberMLP.load(models_dir / "chamber_love.npz")
        rage = ChamberMLP.load(models_dir / "chamber_rage.npz")
        void = ChamberMLP.load(models_dir / "chamber_void.npz")
        
        # Handle missing flow/complex for backwards compatibility
        flow_path = models_dir / "chamber_flow.npz"
        complex_path = models_dir / "chamber_complex.npz"
        
        if flow_path.exists():
            flow = ChamberMLP.load(flow_path)
        else:
            print("[chambers] flow chamber not found, initializing random")
            flow = ChamberMLP.random_init(seed=4)
            
        if complex_path.exists():
            complex_chamber = ChamberMLP.load(complex_path)
        else:
            print("[chambers] complex chamber not found, initializing random")
            complex_chamber = ChamberMLP.random_init(seed=5)

        coupling = np.array(COUPLING_MATRIX_EXTENDED, dtype=np.float32)

        print(f"[chambers] loaded from {models_dir}")
        return cls(
            fear=fear,
            love=love,
            rage=rage,
            void=void,
            flow=flow,
            complex=complex_chamber,
            coupling=coupling,
        )


class AsyncCrossFireSystem:
    """
    Async wrapper for CrossFireSystem with field lock discipline.
    
    Based on HAZE's async pattern - achieves coherence through
    explicit operation ordering and atomicity.
    
    "The asyncio.Lock doesn't add information—it adds discipline."
    """
    
    def __init__(self, system: CrossFireSystem):
        self._sync = system
        self._lock = asyncio.Lock()
    
    @classmethod
    def random_init(cls, seed: Optional[int] = None) -> "AsyncCrossFireSystem":
        """Initialize with random weights."""
        system = CrossFireSystem.random_init(seed=seed)
        return cls(system)
    
    @classmethod
    def load(cls, models_dir: Path) -> "AsyncCrossFireSystem":
        """Load from models directory."""
        system = CrossFireSystem.load(models_dir)
        return cls(system)
    
    async def stabilize(
        self,
        resonances: np.ndarray,
        max_iter: int = 10,
        threshold: float = 0.01,
        momentum: float = 0.7,
    ) -> Tuple[Dict[str, float], int]:
        """
        Async cross-fire stabilization with field lock.
        
        Atomic operation - prevents field corruption during stabilization.
        """
        async with self._lock:
            return self._sync.stabilize(resonances, max_iter, threshold, momentum)
    
    async def save(self, models_dir: Path) -> None:
        """Save with lock protection."""
        async with self._lock:
            self._sync.save(models_dir)
    
    def param_count(self) -> int:
        """Total parameters (read-only, no lock needed)."""
        return self._sync.param_count()
    
    @property
    def coupling(self) -> np.ndarray:
        """Access coupling matrix."""
        return self._sync.coupling
    
    @coupling.setter
    def coupling(self, value: np.ndarray) -> None:
        """Set coupling matrix (for feedback learning)."""
        self._sync.coupling = value


if __name__ == "__main__":
    print("=" * 60)
    print("  CLOUD v4.0 — Chamber Cross-Fire System (200K model)")
    print("=" * 60)
    print()

    # Initialize random system
    system = CrossFireSystem.random_init(seed=42)
    print(f"Initialized cross-fire system (6 chambers)")
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
        print(f"    {chamber:8s}: {value:.3f}  {bar}")
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
        for k in CHAMBER_NAMES_EXTENDED
    )
    print(f"  Save/load {'✓' if match else '✗'}")
    print()

    print("=" * 60)
    print("  Cross-fire system operational. 6 chambers. 200K params.")
    print("=" * 60)
