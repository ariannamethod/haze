#!/usr/bin/env python3
# async_run.py — Async REPL for Haze with Full Resonance Pipeline
#
# Features:
#   - ASYNC architecture (like Leo - 47% coherence improvement)
#   - NO SEED FROM PROMPT - internal field resonance
#   - RESONANT EXPERTS - MOE-style temperature blending
#   - OVERTHINKING - three rings enrich the field
#   - LEXICON GROWTH - absorbs user vocabulary
#   - DEFAULT UNTRAINED MODE - pure resonance, no weights needed
#
# Usage:
#   python async_run.py
#   python async_run.py --corpus mytext.txt

from __future__ import annotations
import sys
import asyncio
import argparse
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from haze import Vocab, CooccurField, load_corpus
from async_haze import AsyncHazeField, HazeResponse
from cleanup import cleanup_output
from experts import describe_mixture


# ----------------- defaults -----------------

DEFAULT_CORPUS = Path("text.txt")

DEFAULT_CONFIG = {
    "temperature": 0.6,
    "generation_length": 100,
    "enable_overthinking": True,
    "enable_lexicon": True,
}


# ----------------- REPL state -----------------

class AsyncREPLState:
    """Holds all configurable generation parameters."""

    def __init__(self):
        self.gen_len = 100
        self.temperature = 0.6
        self.show_stats = True
        self.show_pulse = True
        self.show_seed = False
        self.cleanup_mode = "gentle"

    def to_dict(self) -> dict:
        return {
            "gen_len": self.gen_len,
            "temperature": self.temperature,
            "show_stats": self.show_stats,
            "show_pulse": self.show_pulse,
            "show_seed": self.show_seed,
            "cleanup_mode": self.cleanup_mode,
        }


# ----------------- command handlers -----------------

def handle_command(line: str, state: AsyncREPLState) -> bool:
    """
    Handle REPL commands. Returns True if command was handled.
    """
    stripped = line.strip()
    parts = stripped.split()

    if not parts:
        return False

    cmd = parts[0].lower()

    # /quit, /exit
    if cmd in ("/quit", "/exit", "/q"):
        print("\n🌫️  haze dissolves...")
        sys.exit(0)

    # /len N
    if cmd == "/len":
        if len(parts) == 2 and parts[1].isdigit():
            state.gen_len = max(1, int(parts[1]))
            print(f"[ok] generation length = {state.gen_len}")
        else:
            print("[err] usage: /len 100")
        return True

    # /temp X
    if cmd == "/temp":
        try:
            state.temperature = float(parts[1])
            if state.temperature <= 0:
                raise ValueError
            print(f"[ok] temperature = {state.temperature}")
        except Exception:
            print("[err] usage: /temp 0.6")
        return True

    # /stats
    if cmd == "/stats":
        state.show_stats = not state.show_stats
        print(f"[ok] show_stats = {state.show_stats}")
        return True

    # /pulse
    if cmd == "/pulse":
        state.show_pulse = not state.show_pulse
        print(f"[ok] show_pulse = {state.show_pulse}")
        return True

    # /seed
    if cmd == "/seed":
        state.show_seed = not state.show_seed
        print(f"[ok] show_seed = {state.show_seed}")
        return True

    # /cleanup MODE
    if cmd == "/cleanup":
        valid_modes = ("gentle", "moderate", "strict", "none")
        if len(parts) == 2 and parts[1] in valid_modes:
            state.cleanup_mode = parts[1]
            print(f"[ok] cleanup_mode = {state.cleanup_mode}")
        else:
            print("[err] usage: /cleanup [gentle|moderate|strict|none]")
        return True

    # /config
    if cmd == "/config":
        print("[config]")
        for k, v in state.to_dict().items():
            print(f"  {k}: {v}")
        return True

    # /help
    if cmd == "/help":
        print_help()
        return True

    return False


def print_help():
    """Print help message."""
    help_text = """
╔══════════════════════════════════════════════════════════════╗
║              🌫️  Async Haze REPL — Commands                  ║
╠══════════════════════════════════════════════════════════════╣
║  /len N         set generation length (default: 100)         ║
║  /temp X        set temperature (default: 0.6)               ║
║  /stats         toggle stats display                         ║
║  /pulse         toggle pulse display                         ║
║  /seed          toggle internal seed display                 ║
║  /cleanup MODE  gentle|moderate|strict|none                  ║
║  /config        show current configuration                   ║
║  /help          show this help                               ║
║  /quit          exit                                         ║
╠══════════════════════════════════════════════════════════════╣
║  Any other input generates a response.                       ║
║                                                              ║
║  🔮 NO SEED FROM PROMPT - haze speaks from its field         ║
║  🌊 OVERTHINKING - three rings enrich the vocabulary         ║
║  📚 LEXICON - haze learns YOUR words                         ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(help_text)


def print_response(response: HazeResponse, state: AsyncREPLState):
    """Pretty-print haze response."""
    print()
    print("─" * 60)
    print(response.text)
    print("─" * 60)
    
    if state.show_pulse:
        pulse = response.pulse
        print(f"  pulse: novelty={pulse.novelty:.2f} arousal={pulse.arousal:.2f} entropy={pulse.entropy:.2f}")
    
    if state.show_seed:
        seed_preview = response.internal_seed[:50] + "..." if len(response.internal_seed) > 50 else response.internal_seed
        print(f"  seed: \"{seed_preview}\"")
    
    if state.show_stats:
        # Show expert mixture if available
        if response.expert_mixture:
            mixture_desc = describe_mixture(response.expert_mixture)
            print(f"  experts: {mixture_desc}")
        # Show trauma if triggered
        if response.trauma:
            triggers = ", ".join(sorted(response.trauma.trigger_words)[:5])
            print(f"  trauma: level={response.trauma.level:.2f} triggers=[{triggers}]")
        if response.trauma_influence and response.trauma_influence.identity_weight > 0:
            print(f"  identity: weight={response.trauma_influence.identity_weight:.2f} prefix={response.trauma_influence.should_prefix}")
        print(f"  temp={response.temperature:.2f} time={response.generation_time:.3f}s enrichment={response.enrichment_count}")


# ----------------- main -----------------

async def async_main():
    parser = argparse.ArgumentParser(description="Async Haze REPL")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
        help=f"Path to corpus file (default: {DEFAULT_CORPUS})",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.6,
        help="Base temperature (default: 0.6)",
    )
    parser.add_argument(
        "--no-overthinking",
        action="store_true",
        help="Disable overthinking rings",
    )
    parser.add_argument(
        "--no-lexicon",
        action="store_true",
        help="Disable lexicon growth",
    )
    args = parser.parse_args()

    # Check corpus
    if not args.corpus.exists():
        print(f"[error] corpus not found: {args.corpus}")
        print("Create a text file with your source material.")
        sys.exit(1)

    # Header
    print()
    print("═" * 60)
    print("  🌫️  Haze — Async Resonance Field")
    print("═" * 60)
    print()
    print("  Philosophy:")
    print("    • NO SEED FROM PROMPT - internal field resonance")
    print("    • PRESENCE > INTELLIGENCE - identity speaks first")
    print("    • OVERTHINKING - three rings enrich the field")
    print()
    print("  This is UNTRAINED mode - pure resonance, no weights!")
    print("  Type /help for commands")
    print()
    print("═" * 60)
    print()

    # Initialize async haze field
    async with AsyncHazeField(
        corpus_path=str(args.corpus),
        temperature=args.temp,
        generation_length=100,
        enable_overthinking=not args.no_overthinking,
        enable_lexicon=not args.no_lexicon,
        use_subword=True,  # BPE = coherent output!
        subword_vocab_size=500,
    ) as haze:
        print(f"[haze] corpus: {args.corpus} ({len(haze.corpus_text)} chars)")
        if haze.use_subword and haze.subword_field:
            print(f"[haze] vocab: SUBWORD BPE ({haze.subword_field.vocab.vocab_size} tokens) ← COHERENT OUTPUT!")
        else:
            print(f"[haze] vocab: char-level ({haze.vocab.vocab_size} chars)")
        print(f"[haze] overthinking: {'enabled' if haze.enable_overthinking else 'disabled'}")
        print(f"[haze] lexicon: {'enabled' if haze.enable_lexicon else 'disabled'}")
        print()

        # Init state
        state = AsyncREPLState()
        state.temperature = args.temp

        # REPL loop
        while True:
            try:
                line = input(">>> ").rstrip("\n")
            except (EOFError, KeyboardInterrupt):
                print("\n🌫️  haze dissolves...")
                break

            # Check for command
            if line.strip().startswith("/"):
                handle_command(line, state)
                continue

            # Empty line
            if not line.strip():
                print("[hint] type something, or /help for commands")
                continue

            # Generate response
            try:
                response = await haze.respond(
                    line.strip(),
                    length=state.gen_len,
                    temperature=state.temperature,
                    cleanup=(state.cleanup_mode != "none"),
                )
                
                # Apply additional cleanup if needed
                if state.cleanup_mode in ["moderate", "strict"]:
                    response.text = cleanup_output(response.text, mode=state.cleanup_mode)
                
                print_response(response, state)
                
            except Exception as e:
                print(f"[error] {e}")

            print()


def main():
    """Entry point."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
