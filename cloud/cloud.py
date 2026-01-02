#!/usr/bin/env python3
# cloud.py — CLOUD v3.0 Main Orchestrator
#
# "Something fires BEFORE meaning arrives"
#
# Architecture:
#   1. RESONANCE LAYER (weightless geometry) → 100D resonances
#   2. CHAMBER LAYER (4 MLPs + cross-fire) → chamber activations + iterations
#   3. META-OBSERVER (tiny MLP) → secondary emotion
#
# Total: ~50K params

from __future__ import annotations
import asyncio
import numpy as np
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass

from .resonance import SimpleResonanceLayer
from .chambers import CrossFireSystem
from .observer import MetaObserver
from .user_cloud import UserCloud
from .anchors import get_all_anchors, get_anchor_index


@dataclass
class CloudResponse:
    """Response from CLOUD ping."""

    primary: str  # primary emotion word
    secondary: str  # secondary emotion word (added context)
    resonances: np.ndarray  # (100,) raw resonances
    chamber_activations: Dict[str, float]  # cross-fire results
    iterations: int  # convergence speed signal
    user_fingerprint: np.ndarray  # (100,) temporal history


class Cloud:
    """
    CLOUD v3.0: Pre-semantic sonar for emotion detection.

    Components:
        - Resonance Layer (weightless)
        - Chamber MLPs (4 × 8.5K params)
        - Meta-Observer (15K params)
        - User Cloud (temporal fingerprint)

    Total: ~50K trainable params
    """

    def __init__(
        self,
        resonance: SimpleResonanceLayer,
        chambers: CrossFireSystem,
        observer: MetaObserver,
        user_cloud: Optional[UserCloud] = None,
    ):
        self.resonance = resonance
        self.chambers = chambers
        self.observer = observer
        self.user_cloud = user_cloud or UserCloud()
        self.anchors = get_all_anchors()

    @classmethod
    def random_init(cls, seed: Optional[int] = None) -> "Cloud":
        """Initialize with random weights (for training)."""
        resonance = SimpleResonanceLayer.create()
        chambers = CrossFireSystem.random_init(seed=seed)
        observer = MetaObserver.random_init(seed=seed)
        user_cloud = UserCloud()

        print("[cloud] initialized with random weights")
        return cls(resonance, chambers, observer, user_cloud)

    @classmethod
    def load(cls, models_dir: Path) -> "Cloud":
        """Load trained CLOUD from models/ directory."""
        resonance = SimpleResonanceLayer.create()
        chambers = CrossFireSystem.load(models_dir)
        observer = MetaObserver.load(models_dir / "observer.npz")

        # Load user cloud if exists
        cloud_data_path = models_dir / "user_cloud.json"
        if cloud_data_path.exists():
            user_cloud = UserCloud.load(cloud_data_path)
        else:
            user_cloud = UserCloud()

        print(f"[cloud] loaded from {models_dir}")
        return cls(resonance, chambers, observer, user_cloud)

    def save(self, models_dir: Path) -> None:
        """Save all components to models/ directory."""
        models_dir.mkdir(parents=True, exist_ok=True)

        self.chambers.save(models_dir)
        self.observer.save(models_dir / "observer.npz")
        self.user_cloud.save(models_dir / "user_cloud.json")

        print(f"[cloud] saved to {models_dir}")

    async def ping(self, user_input: str) -> CloudResponse:
        """
        Async ping: detect pre-semantic emotion.

        Flow:
            1. Resonance layer computes 100D resonances
            2. Chambers cross-fire to stabilization
            3. Observer predicts secondary emotion
            4. Update user cloud

        Args:
            user_input: user's text input

        Returns:
            CloudResponse with primary, secondary, and metadata
        """
        # 1. Resonance layer (weightless geometry)
        resonances = self.resonance.compute_resonance(user_input)
        primary_idx, primary_word, _ = self.resonance.get_primary_emotion(resonances)

        # 2. Chamber cross-fire (async parallelism for future optimization)
        chamber_activations, iterations = await asyncio.to_thread(
            self.chambers.stabilize,
            resonances,
        )

        # 3. User fingerprint (temporal history)
        user_fingerprint = self.user_cloud.get_fingerprint()

        # 4. Meta-observer predicts secondary
        secondary_idx = await asyncio.to_thread(
            self.observer.predict_secondary,
            resonances,
            float(iterations),
            user_fingerprint,
        )
        secondary_word = self.anchors[secondary_idx]

        # 5. Update user cloud
        self.user_cloud.add_event(primary_idx, secondary_idx)

        return CloudResponse(
            primary=primary_word,
            secondary=secondary_word,
            resonances=resonances,
            chamber_activations=chamber_activations,
            iterations=iterations,
            user_fingerprint=user_fingerprint,
        )

    def ping_sync(self, user_input: str) -> CloudResponse:
        """Synchronous version of ping (for testing)."""
        return asyncio.run(self.ping(user_input))

    def param_count(self) -> int:
        """Total trainable parameters."""
        return self.chambers.param_count() + self.observer.param_count()


if __name__ == "__main__":
    print("=" * 60)
    print("  CLOUD v3.0 — Main Orchestrator")
    print("=" * 60)
    print()

    # Initialize
    print("Initializing CLOUD...")
    cloud = Cloud.random_init(seed=42)
    print(f"  Total params: {cloud.param_count():,}")
    print()

    # Test inputs
    test_inputs = [
        "I'm terrified and anxious about what's coming",
        "You bring me such warmth and love darling",
        "This makes me furious with rage",
        "I feel completely empty and void inside",
        "I'm curious about what happens next",
        "Overwhelming shame and guilt consume me",
    ]

    print("Testing CLOUD pings:")
    print("=" * 60)

    for text in test_inputs:
        response = cloud.ping_sync(text)

        print(f"\nInput: \"{text}\"")
        print(f"  Primary:   {response.primary}")
        print(f"  Secondary: {response.secondary}")
        print(f"  Iterations: {response.iterations}")
        print(f"  Chambers:")
        for chamber, activation in response.chamber_activations.items():
            bar = "█" * int(activation * 30)
            print(f"    {chamber:6s}: {activation:.3f}  {bar}")

    print()
    print("=" * 60)

    # Show user cloud evolution
    print("\nUser emotional fingerprint (after all inputs):")
    dominant = cloud.user_cloud.get_dominant_emotions(5)
    for idx, strength in dominant:
        word = cloud.anchors[idx]
        bar = "█" * int(strength * 30)
        print(f"  {word:15s}: {strength:.3f}  {bar}")

    print()
    print("=" * 60)

    # Test save/load
    print("\nTesting save/load:")
    models_dir = Path("./models")
    cloud.save(models_dir)

    cloud2 = Cloud.load(models_dir)
    response2 = cloud2.ping_sync(test_inputs[0])

    print(f"  Save/load ✓")
    print()

    print("=" * 60)
    print("  CLOUD v3.0 operational.")
    print("  Something fires BEFORE meaning arrives.")
    print("=" * 60)
