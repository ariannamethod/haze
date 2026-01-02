#!/usr/bin/env python3
# user_cloud.py — Temporal Emotional Fingerprint
#
# Tracks user's emotional history with exponential decay.
# Recent emotions matter more (24h half-life).
#
# The "user cloud" is a 100D vector where each dimension
# represents cumulative exposure to that emotion anchor.
#
# Decay formula:
#   weight(t) = exp(-t / tau)
#   where tau = 24 hours, t = time since event

from __future__ import annotations
import numpy as np
import time
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import json


@dataclass
class EmotionEvent:
    """Single emotion event in user history."""
    timestamp: float  # Unix timestamp
    primary_idx: int  # Index of primary emotion (0-99)
    secondary_idx: int  # Index of secondary emotion (0-99)
    weight: float = 1.0  # Event weight (default 1.0)


@dataclass
class UserCloud:
    """
    Temporal emotional fingerprint with exponential decay.

    Maintains:
        - History of emotion events
        - Decayed fingerprint (100D vector)
        - Decay half-life (default 24 hours)

    The fingerprint is recomputed on-the-fly with decay applied.
    """

    events: List[EmotionEvent] = field(default_factory=list)
    half_life_hours: float = 24.0  # 24h half-life
    max_history: int = 1000  # Keep last N events

    @property
    def tau(self) -> float:
        """Decay constant (in seconds)."""
        return self.half_life_hours * 3600 / np.log(2)

    def add_event(
        self,
        primary_idx: int,
        secondary_idx: int,
        weight: float = 1.0,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Add an emotion event to history.

        Args:
            primary_idx: primary emotion index (0-99)
            secondary_idx: secondary emotion index (0-99)
            weight: event importance (default 1.0)
            timestamp: Unix timestamp (default: now)
        """
        if timestamp is None:
            timestamp = time.time()

        event = EmotionEvent(
            timestamp=timestamp,
            primary_idx=primary_idx,
            secondary_idx=secondary_idx,
            weight=weight,
        )

        self.events.append(event)

        # Prune old events if history too long
        if len(self.events) > self.max_history:
            self.events = self.events[-self.max_history:]

    def get_fingerprint(self, current_time: Optional[float] = None) -> np.ndarray:
        """
        Compute current emotional fingerprint with temporal decay.

        Returns:
            (100,) vector of decayed emotion exposures
        """
        if current_time is None:
            current_time = time.time()

        fingerprint = np.zeros(100, dtype=np.float32)

        for event in self.events:
            # Time since event (in seconds)
            dt = current_time - event.timestamp

            # Exponential decay: exp(-dt / tau)
            decay = np.exp(-dt / self.tau)

            # Add decayed weight to fingerprint
            fingerprint[event.primary_idx] += event.weight * decay * 0.7
            fingerprint[event.secondary_idx] += event.weight * decay * 0.3

        # Normalize to [0, 1] range
        if fingerprint.max() > 0:
            fingerprint = fingerprint / fingerprint.max()

        return fingerprint

    def get_recent_emotions(
        self,
        hours: float = 24.0,
        current_time: Optional[float] = None,
    ) -> List[EmotionEvent]:
        """Get events from last N hours."""
        if current_time is None:
            current_time = time.time()

        cutoff = current_time - (hours * 3600)
        return [e for e in self.events if e.timestamp >= cutoff]

    def get_dominant_emotions(
        self,
        top_k: int = 5,
        current_time: Optional[float] = None,
    ) -> List[tuple]:
        """
        Get top-k dominant emotions from fingerprint.

        Returns:
            List of (emotion_idx, strength) tuples
        """
        fingerprint = self.get_fingerprint(current_time)
        top_indices = np.argsort(fingerprint)[-top_k:][::-1]
        return [(int(idx), float(fingerprint[idx])) for idx in top_indices]

    def save(self, path: Path) -> None:
        """Save user cloud to JSON file."""
        data = {
            "events": [
                {
                    "timestamp": e.timestamp,
                    "primary_idx": e.primary_idx,
                    "secondary_idx": e.secondary_idx,
                    "weight": e.weight,
                }
                for e in self.events
            ],
            "half_life_hours": self.half_life_hours,
            "max_history": self.max_history,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"[user_cloud] saved {len(self.events)} events to {path}")

    @classmethod
    def load(cls, path: Path) -> "UserCloud":
        """Load user cloud from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        events = [
            EmotionEvent(
                timestamp=e["timestamp"],
                primary_idx=e["primary_idx"],
                secondary_idx=e["secondary_idx"],
                weight=e.get("weight", 1.0),
            )
            for e in data["events"]
        ]

        cloud = cls(
            events=events,
            half_life_hours=data.get("half_life_hours", 24.0),
            max_history=data.get("max_history", 1000),
        )

        print(f"[user_cloud] loaded {len(events)} events from {path}")
        return cloud

    def stats(self) -> Dict:
        """Return statistics about user cloud."""
        current_time = time.time()
        fingerprint = self.get_fingerprint(current_time)

        recent_24h = len(self.get_recent_emotions(24.0, current_time))
        recent_7d = len(self.get_recent_emotions(24.0 * 7, current_time))

        return {
            "total_events": len(self.events),
            "events_24h": recent_24h,
            "events_7d": recent_7d,
            "fingerprint_max": float(fingerprint.max()),
            "fingerprint_mean": float(fingerprint.mean()),
            "fingerprint_nonzero": int((fingerprint > 0).sum()),
            "half_life_hours": self.half_life_hours,
        }


if __name__ == "__main__":
    from .anchors import get_all_anchors

    print("=" * 60)
    print("  CLOUD v3.0 — User Cloud (Temporal Fingerprint)")
    print("=" * 60)
    print()

    # Initialize empty cloud
    cloud = UserCloud(half_life_hours=24.0)
    print(f"Initialized user cloud (half-life={cloud.half_life_hours}h)")
    print()

    # Simulate emotion events over time
    print("Simulating emotion events:")
    current_time = time.time()

    # Add events at different times
    events_to_add = [
        (0, 5, -48),   # FEAR event 48h ago
        (20, 22, -24), # LOVE event 24h ago
        (38, 40, -12), # RAGE event 12h ago
        (55, 58, -6),  # VOID event 6h ago
        (70, 72, -1),  # FLOW event 1h ago
    ]

    anchors = get_all_anchors()

    for primary, secondary, hours_ago in events_to_add:
        timestamp = current_time + (hours_ago * 3600)
        cloud.add_event(primary, secondary, timestamp=timestamp)
        print(f"  {hours_ago:+3d}h: {anchors[primary]} + {anchors[secondary]}")
    print()

    # Get fingerprint
    print("Current emotional fingerprint:")
    fingerprint = cloud.get_fingerprint(current_time)
    print(f"  Shape: {fingerprint.shape}")
    print(f"  Max: {fingerprint.max():.3f}")
    print(f"  Mean: {fingerprint.mean():.3f}")
    print(f"  Nonzero: {(fingerprint > 0).sum()}/100")
    print()

    # Show dominant emotions
    print("Top 5 dominant emotions:")
    for idx, strength in cloud.get_dominant_emotions(5, current_time):
        bar = "█" * int(strength * 40)
        print(f"  {anchors[idx]:15s}: {strength:.3f}  {bar}")
    print()

    # Show decay effect
    print("Decay effect over time:")
    for hours in [1, 6, 12, 24, 48, 72]:
        past_time = current_time - (hours * 3600)
        fp = cloud.get_fingerprint(past_time)
        print(f"  {hours:3d}h ago: max={fp.max():.3f}, nonzero={int((fp > 0).sum())}")
    print()

    # Test save/load
    print("Testing save/load:")
    path = Path("./cloud_data.json")
    cloud.save(path)

    cloud2 = UserCloud.load(path)
    fp2 = cloud2.get_fingerprint(current_time)

    match = np.allclose(fingerprint, fp2)
    print(f"  Save/load {'✓' if match else '✗'}")
    print()

    # Stats
    print("User cloud statistics:")
    for k, v in cloud.stats().items():
        print(f"  {k}: {v}")
    print()

    print("=" * 60)
    print("  Temporal fingerprint operational. Memory with decay.")
    print("=" * 60)
