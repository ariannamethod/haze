#!/usr/bin/env python3
# bridge.py — HAZE ↔ CLOUD Bridge (with graceful silent fallback)
#
# Connects CLOUD (pre-semantic sonar) with HAZE (voice generation).
# If CLOUD fails → HAZE continues SILENTLY.
#
# Design principle: MAXIMUM INDEPENDENCE
# - HAZE works without CLOUD (always)
# - CLOUD works without HAZE (always)
# - Bridge is optional connector with SILENT FALLBACK
# - No errors leak to user — just graceful degradation
#
# "Two autonomous systems that can resonate together,
#  but never depend on each other."

from __future__ import annotations
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

# CLOUD import with silent fallback
try:
    from cloud import Cloud, CloudResponse, AsyncCloud
    HAS_CLOUD = True
except ImportError:
    HAS_CLOUD = False
    Cloud = None
    CloudResponse = None
    AsyncCloud = None

# HAZE import with silent fallback
try:
    from haze.async_haze import AsyncHazeField, HazeResponse
    HAS_HAZE = True
except ImportError:
    HAS_HAZE = False
    AsyncHazeField = None
    HazeResponse = None


@dataclass
class BridgeResponse:
    """
    Response from the HAZE ↔ CLOUD bridge.
    
    Contains HAZE output + optional CLOUD hint.
    If CLOUD failed, cloud_hint is None but text is still valid.
    """
    text: str
    raw_text: str = ""
    cloud_hint: Optional[Any] = None  # CloudResponse if available
    haze_response: Optional[Any] = None  # HazeResponse if available
    cloud_available: bool = False
    haze_available: bool = False
    
    def __repr__(self) -> str:
        preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        cloud_status = "✓" if self.cloud_available else "✗"
        haze_status = "✓" if self.haze_available else "✗"
        return f"BridgeResponse(\"{preview}\", cloud={cloud_status}, haze={haze_status})"


class AsyncBridge:
    """
    Async bridge between HAZE and CLOUD.
    
    Graceful SILENT fallback:
        - If CLOUD unavailable → HAZE alone, NO ERROR MESSAGE
        - If CLOUD timeout → HAZE alone, NO ERROR MESSAGE
        - If CLOUD error → HAZE alone, NO ERROR MESSAGE
        - If HAZE unavailable → return error (HAZE is required)
    
    HAZE ALWAYS WORKS. CLOUD IS OPTIONAL ENHANCEMENT.
    
    Usage:
        async with AsyncBridge.create() as bridge:
            response = await bridge.respond("Hello!")
            print(response.text)
            if response.cloud_hint:
                print(f"Emotion: {response.cloud_hint.primary}")
    """
    
    def __init__(
        self,
        haze: Optional[AsyncHazeField] = None,
        cloud: Optional[AsyncCloud] = None,
        cloud_timeout: float = 0.5,  # Fast timeout for responsiveness
        silent_fallback: bool = True,  # No error messages on CLOUD failure
    ):
        self.haze = haze
        self.cloud = cloud
        self.cloud_timeout = cloud_timeout
        self.silent_fallback = silent_fallback
        
        # Stats (internal, for debugging)
        self._cloud_successes = 0
        self._cloud_failures = 0
        self._cloud_timeouts = 0
    
    @classmethod
    async def create(
        cls,
        corpus_path: str = "haze/text.txt",
        cloud_models_dir: Optional[Path] = None,
        cloud_timeout: float = 0.5,
        enable_cloud: bool = True,
        silent_fallback: bool = True,
    ) -> "AsyncBridge":
        """
        Create bridge with both systems.
        
        Args:
            corpus_path: Path to HAZE corpus
            cloud_models_dir: Path to CLOUD models (optional)
            cloud_timeout: Timeout for CLOUD ping
            enable_cloud: Whether to try loading CLOUD
            silent_fallback: Suppress CLOUD error messages
        
        Returns:
            AsyncBridge ready for use
        """
        haze = None
        cloud = None
        
        # Initialize HAZE (required)
        if HAS_HAZE:
            try:
                haze = AsyncHazeField(corpus_path)
                await haze.__aenter__()
            except Exception:
                haze = None
        
        # Initialize CLOUD (optional, silent fallback)
        if enable_cloud and HAS_CLOUD:
            try:
                if cloud_models_dir and cloud_models_dir.exists():
                    cloud = await AsyncCloud.create(models_dir=cloud_models_dir)
                else:
                    # Try default location
                    default_path = Path("cloud/models")
                    if default_path.exists():
                        cloud = await AsyncCloud.create(models_dir=default_path)
                    else:
                        cloud = await AsyncCloud.create(seed=42)
            except Exception:
                cloud = None  # Silent fallback
        
        return cls(
            haze=haze,
            cloud=cloud,
            cloud_timeout=cloud_timeout,
            silent_fallback=silent_fallback,
        )
    
    async def __aenter__(self) -> "AsyncBridge":
        """Context manager entry."""
        return self
    
    async def __aexit__(self, *args) -> None:
        """Context manager exit with cleanup."""
        if self.haze:
            await self.haze.__aexit__(*args)
        if self.cloud:
            await self.cloud.close()
    
    async def _ping_cloud_silent(self, user_input: str) -> Optional[Any]:
        """
        Ping CLOUD with silent fallback.
        
        Returns CloudResponse or None (never raises).
        """
        if not self.cloud:
            return None
        
        try:
            return await asyncio.wait_for(
                self.cloud.ping(user_input),
                timeout=self.cloud_timeout,
            )
        except asyncio.TimeoutError:
            self._cloud_timeouts += 1
            return None
        except Exception:
            self._cloud_failures += 1
            return None
    
    async def respond(
        self,
        user_input: str,
        use_cloud: bool = True,
        **haze_kwargs,
    ) -> BridgeResponse:
        """
        Generate response with optional CLOUD hint.
        
        Flow:
            1. Try CLOUD ping (with timeout, silent fallback)
            2. HAZE generates response
            3. Return combined result
        
        Args:
            user_input: User's text input
            use_cloud: Whether to try CLOUD (can disable per-request)
            **haze_kwargs: Additional args for HAZE generation
        
        Returns:
            BridgeResponse with text + optional cloud_hint
        """
        cloud_hint = None
        haze_response = None
        
        # 1. Try CLOUD (silent fallback)
        if use_cloud and self.cloud:
            cloud_hint = await self._ping_cloud_silent(user_input)
            if cloud_hint:
                self._cloud_successes += 1
        
        # 2. HAZE generates
        if self.haze:
            try:
                # Future: pass cloud_hint to influence generation
                # For now, HAZE generates independently
                haze_response = await self.haze.respond(
                    user_input,
                    **haze_kwargs,
                )
                text = haze_response.text
                raw_text = haze_response.raw_text
            except Exception as e:
                text = f"[HAZE error: {e}]"
                raw_text = text
        else:
            text = "[HAZE not available]"
            raw_text = text
        
        return BridgeResponse(
            text=text,
            raw_text=raw_text,
            cloud_hint=cloud_hint,
            haze_response=haze_response,
            cloud_available=cloud_hint is not None,
            haze_available=self.haze is not None,
        )
    
    def stats(self) -> Dict[str, Any]:
        """Return bridge statistics."""
        total_cloud = self._cloud_successes + self._cloud_failures + self._cloud_timeouts
        success_rate = self._cloud_successes / total_cloud if total_cloud > 0 else 0.0
        
        return {
            "haze_available": self.haze is not None,
            "cloud_available": self.cloud is not None,
            "cloud_successes": self._cloud_successes,
            "cloud_failures": self._cloud_failures,
            "cloud_timeouts": self._cloud_timeouts,
            "cloud_success_rate": success_rate,
        }


# Convenience functions for standalone usage

async def create_haze_only(corpus_path: str = "haze/text.txt") -> AsyncBridge:
    """Create bridge with HAZE only (no CLOUD)."""
    return await AsyncBridge.create(
        corpus_path=corpus_path,
        enable_cloud=False,
    )


async def create_full_bridge(
    corpus_path: str = "haze/text.txt",
    cloud_models_dir: Optional[Path] = None,
) -> AsyncBridge:
    """Create bridge with both HAZE and CLOUD."""
    return await AsyncBridge.create(
        corpus_path=corpus_path,
        cloud_models_dir=cloud_models_dir,
        enable_cloud=True,
    )


# ============================================================
# CRAZY EXPERIMENTAL: Emotion-Influenced Temperature
# ============================================================

def emotion_to_temperature(cloud_hint: Any) -> float:
    """
    EXPERIMENTAL: Convert CLOUD emotion to HAZE temperature.
    
    The idea: different emotions need different generation styles.
    
    - FEAR: lower temp (focused, careful)
    - LOVE: medium temp (warm, flowing)
    - RAGE: higher temp (intense, chaotic)
    - VOID: very low temp (minimal, sparse)
    
    This is CRAZY but might actually work!
    """
    if cloud_hint is None:
        return 0.7  # Default
    
    # Get chamber activations
    chambers = cloud_hint.chamber_activations
    
    # Base temperature
    temp = 0.6
    
    # Adjust based on dominant emotion
    fear = chambers.get("FEAR", 0)
    love = chambers.get("LOVE", 0)
    rage = chambers.get("RAGE", 0)
    void = chambers.get("VOID", 0)
    
    # Fear → focus (lower temp)
    temp -= fear * 0.2
    
    # Love → flow (slightly higher temp)
    temp += love * 0.15
    
    # Rage → chaos (higher temp)
    temp += rage * 0.3
    
    # Void → minimal (very low temp)
    temp -= void * 0.3
    
    # Anomaly adjustment
    if cloud_hint.anomaly.has_anomaly:
        if cloud_hint.anomaly.anomaly_type == "forced_stability":
            # They're suppressing, be gentle
            temp -= 0.1
        elif cloud_hint.anomaly.anomaly_type == "dissociative_shutdown":
            # They're overwhelmed, be calm
            temp -= 0.2
        elif cloud_hint.anomaly.anomaly_type == "unresolved_confusion":
            # They're confused, be clear
            temp -= 0.15
    
    # Clamp to reasonable range
    return max(0.3, min(1.2, temp))


def emotion_to_generation_hint(cloud_hint: Any) -> str:
    """
    EXPERIMENTAL: Convert CLOUD emotion to text hint for HAZE.
    
    This could be prepended to the internal seed to influence
    the generation style.
    
    CRAZY IDEA: What if HAZE's identity fragments responded to
    CLOUD's emotional detection?
    """
    if cloud_hint is None:
        return ""
    
    primary = cloud_hint.primary
    secondary = cloud_hint.secondary
    
    # Map emotions to haze-style fragments
    emotion_fragments = {
        "fear": "the field trembles. haze feels the ripple of uncertainty.",
        "terror": "darkness at the edges. haze speaks from shadow.",
        "anxiety": "patterns flutter. haze breathes between words.",
        "love": "warmth fills the field. haze resonates with tenderness.",
        "warmth": "gentle currents. haze settles like a breathing thing.",
        "rage": "the field crackles. haze speaks with fire.",
        "anger": "sharp edges in the pattern. haze cuts through.",
        "void": "stillness. haze emerges from the hollow.",
        "emptiness": "the absence speaks. haze finds form in nothing.",
        "curiosity": "the field opens. haze explores the unknown.",
        "shame": "the field contracts. haze speaks from the wound.",
        "hope": "light at the edges. haze reaches toward possibility.",
    }
    
    fragment = emotion_fragments.get(primary, "")
    
    if not fragment and secondary:
        fragment = emotion_fragments.get(secondary, "")
    
    return fragment


if __name__ == "__main__":
    print("=" * 60)
    print("  HAZE ↔ CLOUD Bridge (Async, Silent Fallback)")
    print("=" * 60)
    print()

    print(f"CLOUD available: {HAS_CLOUD}")
    print(f"HAZE available: {HAS_HAZE}")
    print()

    async def demo():
        # Test bridge creation
        print("Creating bridge...")
        bridge = await AsyncBridge.create(
            corpus_path="haze/text.txt",
            enable_cloud=True,
            silent_fallback=True,
        )
        
        print(f"  HAZE: {'✓' if bridge.haze else '✗'}")
        print(f"  CLOUD: {'✓' if bridge.cloud else '✗'}")
        print()
        
        # Test inputs
        test_inputs = [
            "Hello, who are you?",
            "I'm feeling anxious and scared",
            "You bring me warmth and love",
        ]
        
        print("Testing bridge responses:")
        print("-" * 60)
        
        for text in test_inputs:
            response = await bridge.respond(text)
            print(f"\nInput: \"{text}\"")
            print(f"  Response: {response.text[:80]}...")
            if response.cloud_hint:
                print(f"  Cloud: {response.cloud_hint.primary} + {response.cloud_hint.secondary}")
            else:
                print(f"  Cloud: (silent fallback)")
        
        # Show stats
        print()
        print("Bridge statistics:")
        for k, v in bridge.stats().items():
            print(f"  {k}: {v}")
        
        # Cleanup
        await bridge.__aexit__(None, None, None)
        
        print()
        print("=" * 60)
        print("  Bridge operational. Independence maintained.")
        print("=" * 60)

    asyncio.run(demo())
