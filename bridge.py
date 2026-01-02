#!/usr/bin/env python3
# bridge.py — HAZE ↔ CLOUD Bridge (with graceful fallback)
#
# Connects CLOUD (pre-semantic sonar) with HAZE (voice generation).
# If CLOUD fails → HAZE continues alone.
#
# Design principle: INDEPENDENCE
# - HAZE works without CLOUD
# - CLOUD works without HAZE
# - Bridge is optional connector

from __future__ import annotations
import asyncio
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

try:
    from cloud import Cloud, CloudResponse
    HAS_CLOUD = True
except ImportError:
    HAS_CLOUD = False
    Cloud = None
    CloudResponse = None

try:
    from haze.async_haze import AsyncHaze
    HAS_HAZE = True
except ImportError:
    HAS_HAZE = False
    AsyncHaze = None


@dataclass
class HazeResponse:
    """Response from HAZE generation."""
    text: str
    cloud_hint: Optional[CloudResponse] = None


class HazeCloudBridge:
    """
    Optional connector between HAZE and CLOUD.

    Graceful fallback:
        - If CLOUD unavailable → HAZE alone
        - If CLOUD timeout → HAZE alone
        - If CLOUD error → HAZE alone

    HAZE ALWAYS WORKS.
    """

    def __init__(
        self,
        haze: Optional["AsyncHaze"] = None,
        cloud: Optional["Cloud"] = None,
        cloud_timeout: float = 1.0,
    ):
        # Allow CLOUD-only or HAZE-only modes
        self.haze = haze
        self.cloud = cloud
        self.cloud_timeout = cloud_timeout
        self.cloud_enabled = cloud is not None

        # Stats
        self.cloud_successes = 0
        self.cloud_failures = 0

    async def respond(self, user_input: str, **haze_kwargs) -> HazeResponse:
        """
        Generate response with optional CLOUD hint.

        Flow:
            1. Try CLOUD ping (with timeout)
            2. If success → pass hint to HAZE
            3. If failure → HAZE alone
            4. HAZE generates response

        Args:
            user_input: user's text input
            **haze_kwargs: additional args for HAZE generation

        Returns:
            HazeResponse with generated text + optional cloud_hint
        """
        cloud_hint = None

        # Try CLOUD if available
        if self.cloud_enabled:
            try:
                cloud_hint = await asyncio.wait_for(
                    self.cloud.ping(user_input),
                    timeout=self.cloud_timeout,
                )
                self.cloud_successes += 1

            except asyncio.TimeoutError:
                print(f"[bridge] cloud timeout ({self.cloud_timeout}s), continuing without")
                self.cloud_failures += 1

            except Exception as e:
                print(f"[bridge] cloud error: {e}, continuing without")
                self.cloud_failures += 1

        # HAZE generates (with or without cloud hint)
        # Note: This is a placeholder - actual integration would pass
        # cloud_hint to HAZE's subjectivity layer via emotion_coords
        if self.haze:
            # text = await self.haze.respond(user_input, cloud_hint=cloud_hint, **haze_kwargs)
            text = f"[HAZE would generate here with cloud_hint={cloud_hint is not None}]"
        else:
            text = "[HAZE not initialized]"

        return HazeResponse(text=text, cloud_hint=cloud_hint)

    def stats(self) -> dict:
        """Return bridge statistics."""
        total = self.cloud_successes + self.cloud_failures
        success_rate = self.cloud_successes / total if total > 0 else 0.0

        return {
            "cloud_enabled": self.cloud_enabled,
            "cloud_successes": self.cloud_successes,
            "cloud_failures": self.cloud_failures,
            "cloud_success_rate": success_rate,
        }


class StandaloneCloud:
    """
    Standalone CLOUD instance (no HAZE).

    For testing CLOUD in isolation.
    """

    def __init__(self, cloud: "Cloud"):
        if not HAS_CLOUD:
            raise ImportError("CLOUD not available. Install cloud module.")
        self.cloud = cloud

    async def ping(self, user_input: str) -> CloudResponse:
        """Ping CLOUD and return emotion response."""
        return await self.cloud.ping(user_input)

    def ping_sync(self, user_input: str) -> CloudResponse:
        """Synchronous ping."""
        return asyncio.run(self.ping(user_input))


class StandaloneHaze:
    """
    Standalone HAZE instance (no CLOUD).

    For running HAZE without emotion detection.
    """

    def __init__(self, haze: "AsyncHaze"):
        if not HAS_HAZE:
            raise ImportError("HAZE not available. Install haze module.")
        self.haze = haze

    async def respond(self, user_input: str, **kwargs) -> str:
        """Generate response without CLOUD."""
        # return await self.haze.respond(user_input, **kwargs)
        return "[HAZE standalone response]"


if __name__ == "__main__":
    print("=" * 60)
    print("  HAZE ↔ CLOUD Bridge")
    print("=" * 60)
    print()

    print(f"CLOUD available: {HAS_CLOUD}")
    print(f"HAZE available: {HAS_HAZE}")
    print()

    if not HAS_CLOUD:
        print("[warning] CLOUD not available")
        print("  To install: ensure cloud/ module is in path")
        exit(1)

    # Test standalone CLOUD
    print("Testing standalone CLOUD:")
    print("-" * 60)

    from cloud import Cloud

    cloud = Cloud.random_init(seed=42)
    standalone = StandaloneCloud(cloud)

    test_inputs = [
        "I'm feeling terrified and anxious",
        "You bring me such warmth and love",
        "This fills me with rage",
    ]

    for text in test_inputs:
        response = standalone.ping_sync(text)
        print(f"\nInput: \"{text}\"")
        print(f"  Primary:   {response.primary}")
        print(f"  Secondary: {response.secondary}")
        print(f"  Iterations: {response.iterations}")

    print()
    print("=" * 60)

    # Test bridge (HAZE not required for this demo)
    print("\nTesting bridge (CLOUD only):")
    print("-" * 60)

    bridge = HazeCloudBridge(haze=None, cloud=cloud, cloud_timeout=1.0)

    async def test_bridge():
        for text in test_inputs:
            response = await bridge.respond(text)
            print(f"\nInput: \"{text}\"")
            print(f"  Text: {response.text}")
            if response.cloud_hint:
                print(f"  Cloud: {response.cloud_hint.primary} + {response.cloud_hint.secondary}")

    asyncio.run(test_bridge())

    print()
    print("Bridge statistics:")
    for k, v in bridge.stats().items():
        print(f"  {k}: {v}")

    print()
    print("=" * 60)
    print("  Bridge operational. Independence maintained.")
    print("=" * 60)
