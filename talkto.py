#!/usr/bin/env python3
# talkto.py — Router for HAZE and CLOUD
#
# Unified interface:
# - Default: HAZE only (fast, simple)
# - /cloud: Toggle CLOUD mode (pre-semantic sonar)
# - /feedback: Show feedback stats
#
# CLOUD mode: emotion detection → HAZE generation → coherence feedback

import sys
import asyncio
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "haze"))


class HazeCloudRouter:
    """
    Router that switches between HAZE-only and HAZE+CLOUD modes.

    Commands:
        /cloud - toggle CLOUD mode
        /feedback - show feedback stats
        /help - show help
    """

    def __init__(self):
        self.cloud_enabled = False
        self.cloud = None
        self.feedback_history = []

        print("=" * 60)
        print("  HAZE + CLOUD Router")
        print("=" * 60)
        print()
        print("  Commands:")
        print("    /cloud - toggle CLOUD pre-semantic sonar")
        print("    /feedback - show coherence stats")
        print("    /help - show all commands")
        print()
        print("  Mode: HAZE only (type /cloud to enable emotion detection)")
        print("=" * 60)
        print()

    def toggle_cloud(self):
        """Toggle CLOUD mode."""
        self.cloud_enabled = not self.cloud_enabled

        if self.cloud_enabled:
            # Initialize CLOUD
            if self.cloud is None:
                try:
                    from cloud import Cloud
                    self.cloud = Cloud.load(Path("cloud/models"))
                    print("[cloud] loaded from cloud/models")
                except Exception as e:
                    print(f"[cloud] failed to load: {e}")
                    print("[cloud] initializing with random weights...")
                    from cloud import Cloud
                    self.cloud = Cloud.random_init()

            print("✓ CLOUD enabled (pre-semantic emotion detection)")
        else:
            print("✗ CLOUD disabled (HAZE only mode)")

    def show_feedback_stats(self):
        """Show feedback statistics."""
        if not self.feedback_history:
            print("[feedback] no data yet")
            return

        print("=" * 60)
        print("  Feedback Statistics")
        print("=" * 60)

        recent = self.feedback_history[-10:]
        avg_coherence = sum(f["coherence"] for f in recent) / len(recent)

        print(f"  Recent coherence: {avg_coherence:.3f}")
        print(f"  Total interactions: {len(self.feedback_history)}")
        print()

        print("  Last 5 interactions:")
        for i, feedback in enumerate(recent[-5:], 1):
            print(f"    {i}. Coherence: {feedback['coherence']:.3f}")
            if feedback.get("anomaly"):
                print(f"       Anomaly: {feedback['anomaly']}")
        print()

    async def process_with_cloud(self, user_input: str) -> str:
        """Process input with CLOUD + HAZE."""
        # 1. CLOUD ping (emotion detection)
        cloud_response = await self.cloud.ping(user_input)

        # Display CLOUD output
        print(f"[cloud] {cloud_response.primary} + {cloud_response.secondary}", end="")
        if cloud_response.anomaly.has_anomaly:
            print(f" | anomaly: {cloud_response.anomaly.anomaly_type}", end="")
        print()

        # 2. HAZE generates (in future: pass cloud_hint)
        # For now: just use standard HAZE
        # TODO: integrate cloud_hint into haze.async_haze

        # Simulate HAZE response for now
        haze_response = f"[HAZE would respond here, influenced by {cloud_response.primary}]"

        # 3. Measure coherence and update coupling
        from cloud.feedback import measure_coherence, update_coupling

        coherence_metrics = measure_coherence(haze_response)
        coherence = coherence_metrics["coherence"]

        # Update coupling matrix
        self.cloud.chambers.coupling = update_coupling(
            self.cloud.chambers.coupling,
            cloud_response.chamber_activations,
            coherence,
            learning_rate=0.01,
        )

        # Save feedback
        self.feedback_history.append({
            "coherence": coherence,
            "anomaly": cloud_response.anomaly.anomaly_type,
            "primary": cloud_response.primary,
        })

        print(f"[feedback] coherence: {coherence:.3f}")

        return haze_response

    def process_without_cloud(self, user_input: str) -> str:
        """Process input with HAZE only."""
        # Just call HAZE directly
        # For now, placeholder
        return "[HAZE only mode - use haze/run.py for actual generation]"

    async def interactive_loop(self):
        """Main interactive loop."""
        while True:
            try:
                user_input = input("> ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input == "/cloud":
                    self.toggle_cloud()
                    continue

                if user_input == "/feedback":
                    self.show_feedback_stats()
                    continue

                if user_input in ["/help", "/h"]:
                    print("Commands:")
                    print("  /cloud - toggle CLOUD mode")
                    print("  /feedback - show stats")
                    print("  /quit - exit")
                    continue

                if user_input in ["/quit", "/q", "/exit"]:
                    print("Goodbye!")
                    break

                # Process input
                if self.cloud_enabled:
                    response = await self.process_with_cloud(user_input)
                else:
                    response = self.process_without_cloud(user_input)

                print(response)
                print()

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"[error] {e}")
                continue


def main():
    """Entry point."""
    router = HazeCloudRouter()
    asyncio.run(router.interactive_loop())


if __name__ == "__main__":
    main()
