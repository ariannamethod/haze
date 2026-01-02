"""
CLOUD v3.0 — Pre-Semantic Sonar

"Something fires BEFORE meaning arrives"

Architecture:
    - Resonance Layer (weightless geometry)
    - Chamber MLPs (4 × 8.5K params + cross-fire)
    - Meta-Observer (15K params)
    - User Cloud (temporal fingerprint)

Total: ~50K params

Usage:
    from cloud import Cloud

    cloud = Cloud.random_init()
    response = await cloud.ping("I'm feeling anxious")
    print(f"Primary: {response.primary}, Secondary: {response.secondary}")
"""

from .cloud import Cloud, CloudResponse
from .chambers import CrossFireSystem, ChamberMLP
from .observer import MetaObserver
from .resonance import SimpleResonanceLayer, ResonanceLayer
from .user_cloud import UserCloud, EmotionEvent
from .anchors import (
    EMOTION_ANCHORS,
    CHAMBER_NAMES,
    COUPLING_MATRIX,
    get_all_anchors,
    get_chamber_ranges,
)

__version__ = "3.0.0"

__all__ = [
    # Main classes
    "Cloud",
    "CloudResponse",
    
    # Components
    "CrossFireSystem",
    "ChamberMLP",
    "MetaObserver",
    "SimpleResonanceLayer",
    "ResonanceLayer",
    "UserCloud",
    "EmotionEvent",
    
    # Anchors
    "EMOTION_ANCHORS",
    "CHAMBER_NAMES",
    "COUPLING_MATRIX",
    "get_all_anchors",
    "get_chamber_ranges",
]
