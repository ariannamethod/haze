#!/usr/bin/env python3
# cloud.py — CLOUD v4.0 Main Orchestrator (200K model)
#
# "Something fires BEFORE meaning arrives"
#
# Architecture:
#   1. RESONANCE LAYER (weightless geometry) → 100D resonances
#   2. CHAMBER LAYER (6 MLPs + cross-fire) → chamber activations + iterations
#   3. META-OBSERVER (deeper MLP) → secondary emotion
#
# Total: ~180K params (6 chambers × 23K + observer 41K)

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
from .anomaly import detect_anomalies, AnomalyReport


@dataclass
class CloudResponse:
    """Response from CLOUD ping."""

    primary: str  # primary emotion word
    secondary: str  # secondary emotion word (added context)
    resonances: np.ndarray  # (100,) raw resonances
    chamber_activations: Dict[str, float]  # cross-fire results
    iterations: int  # convergence speed signal
    user_fingerprint: np.ndarray  # (100,) temporal history
    anomaly: AnomalyReport  # anomaly detection result


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

        # 4. Meta-observer predicts secondary (now with chamber_activations)
        # Convert chamber_activations dict to array for observer
        import numpy as np
        from .anchors import CHAMBER_NAMES_EXTENDED
        chamber_array = np.array([
            chamber_activations.get(name, 0.0) for name in CHAMBER_NAMES_EXTENDED
        ], dtype=np.float32)
        
        secondary_idx = await asyncio.to_thread(
            self.observer.predict_secondary,
            resonances,
            chamber_array,
            float(iterations),
            user_fingerprint,
        )
        secondary_word = self.anchors[secondary_idx]

        # 5. Anomaly detection
        anomaly = detect_anomalies(chamber_activations, iterations)

        # 6. Update user cloud
        self.user_cloud.add_event(primary_idx, secondary_idx)

        return CloudResponse(
            primary=primary_word,
            secondary=secondary_word,
            resonances=resonances,
            chamber_activations=chamber_activations,
            iterations=iterations,
            user_fingerprint=user_fingerprint,
            anomaly=anomaly,
        )

    def ping_sync(self, user_input: str) -> CloudResponse:
        """Synchronous version of ping (for testing)."""
        return asyncio.run(self.ping(user_input))

    def param_count(self) -> int:
        """Total trainable parameters."""
        return self.chambers.param_count() + self.observer.param_count()


class AsyncCloud:
    """
    Fully async CLOUD with context manager support.
    
    Based on HAZE's AsyncHazeField pattern:
    - Field lock discipline for coherence
    - Context manager for clean lifecycle
    - Graceful initialization and cleanup
    
    Usage:
        async with AsyncCloud.create() as cloud:
            response = await cloud.ping("I'm feeling anxious")
    
    Or standalone:
        cloud = await AsyncCloud.create()
        response = await cloud.ping("I'm feeling anxious")
        await cloud.close()
    """
    
    def __init__(self, cloud: Cloud):
        self._sync = cloud
        self._lock = asyncio.Lock()
        self._closed = False
    
    @classmethod
    async def create(
        cls,
        models_dir: Optional[Path] = None,
        seed: Optional[int] = None,
    ) -> "AsyncCloud":
        """
        Create AsyncCloud instance.
        
        Args:
            models_dir: Path to load trained weights (optional)
            seed: Random seed for initialization (if no models_dir)
        
        Returns:
            AsyncCloud ready for use
        """
        if models_dir and models_dir.exists():
            cloud = Cloud.load(models_dir)
        else:
            cloud = Cloud.random_init(seed=seed)
        
        return cls(cloud)
    
    async def __aenter__(self) -> "AsyncCloud":
        """Context manager entry."""
        return self
    
    async def __aexit__(self, *args) -> None:
        """Context manager exit."""
        await self.close()
    
    async def close(self) -> None:
        """Clean shutdown."""
        if not self._closed:
            self._closed = True
            # Save user cloud state if needed
            # await self.save(Path("cloud/models"))
    
    async def ping(self, user_input: str) -> CloudResponse:
        """
        Async ping with field lock.
        
        Atomic operation - prevents field corruption during emotion detection.
        """
        if self._closed:
            raise RuntimeError("AsyncCloud is closed")
        
        async with self._lock:
            return await self._sync.ping(user_input)
    
    async def save(self, models_dir: Path) -> None:
        """Save all components with lock protection."""
        async with self._lock:
            self._sync.save(models_dir)
    
    def param_count(self) -> int:
        """Total parameters (read-only, no lock needed)."""
        return self._sync.param_count()
    
    @property
    def anchors(self):
        """Access emotion anchors."""
        return self._sync.anchors
    
    @property
    def user_cloud(self):
        """Access user cloud for stats."""
        return self._sync.user_cloud


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
