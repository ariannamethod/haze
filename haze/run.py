#!/usr/bin/env python3
# run.py — Enhanced REPL for Reweight-GPT
#
# Features:
#   - Multiple sampling modes: basic, top_k, top_p, entropy-aware
#   - Generation statistics (entropy, confidence, temperature)
#   - Configurable parameters via commands
#   - Head type switching (hybrid, reweight, content)
#
# Usage:
#   python run.py
#   python run.py --corpus mytext.txt
#   python run.py --weights my_weights.npz

from __future__ import annotations
import sys
import argparse
from pathlib import Path

from model import (
    Vocab,
    ReweightGPT as Haze,
    load_corpus,
    build_model_from_text,
)


# ----------------- defaults -----------------


DEFAULT_CORPUS = Path("text.txt")
DEFAULT_WEIGHTS = Path("theweightofhaze.npz")

DEFAULT_CONFIG = {
    "T": 32,
    "n_emb": 64,
    "nodes": 64,
    "n_blocks": 3,
    "n_heads": 4,
    "head_type": "hybrid",
    "alpha": 0.5,
}


# ----------------- REPL state -----------------


class REPLState:
    """Holds all configurable generation parameters."""

    def __init__(self):
        self.gen_len = 300
        self.temperature = 1.0
        self.sampling = "entropy"  # basic, top_k, top_p, entropy
        self.top_k = 40
        self.top_p = 0.9
        self.target_entropy = 3.0
        self.min_temp = 0.3
        self.max_temp = 2.0
        self.show_stats = True

    def to_dict(self) -> dict:
        return {
            "gen_len": self.gen_len,
            "temperature": self.temperature,
            "sampling": self.sampling,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "target_entropy": self.target_entropy,
            "min_temp": self.min_temp,
            "max_temp": self.max_temp,
            "show_stats": self.show_stats,
        }


# ----------------- command handlers -----------------


def handle_command(line: str, state: REPLState) -> bool:
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
        print("bye!")
        sys.exit(0)

    # /len N
    if cmd == "/len":
        if len(parts) == 2 and parts[1].isdigit():
            state.gen_len = max(1, int(parts[1]))
            print(f"[ok] generation length = {state.gen_len}")
        else:
            print("[err] usage: /len 400")
        return True

    # /temp X
    if cmd == "/temp":
        try:
            state.temperature = float(parts[1])
            if state.temperature <= 0:
                raise ValueError
            print(f"[ok] temperature = {state.temperature}")
        except Exception:
            print("[err] usage: /temp 0.7")
        return True

    # /sampling MODE
    if cmd == "/sampling":
        if len(parts) == 2 and parts[1] in ("basic", "top_k", "top_p", "entropy"):
            state.sampling = parts[1]
            print(f"[ok] sampling = {state.sampling}")
        else:
            print("[err] usage: /sampling [basic|top_k|top_p|entropy]")
        return True

    # /topk K
    if cmd == "/topk":
        try:
            state.top_k = max(1, int(parts[1]))
            print(f"[ok] top_k = {state.top_k}")
        except Exception:
            print("[err] usage: /topk 40")
        return True

    # /topp P
    if cmd == "/topp":
        try:
            state.top_p = float(parts[1])
            if not (0 < state.top_p <= 1):
                raise ValueError
            print(f"[ok] top_p = {state.top_p}")
        except Exception:
            print("[err] usage: /topp 0.9")
        return True

    # /entropy TARGET
    if cmd == "/entropy":
        try:
            state.target_entropy = float(parts[1])
            print(f"[ok] target_entropy = {state.target_entropy}")
        except Exception:
            print("[err] usage: /entropy 3.0")
        return True

    # /bounds MIN MAX
    if cmd == "/bounds":
        try:
            state.min_temp = float(parts[1])
            state.max_temp = float(parts[2])
            print(f"[ok] temp bounds = [{state.min_temp}, {state.max_temp}]")
        except Exception:
            print("[err] usage: /bounds 0.3 2.0")
        return True

    # /stats
    if cmd == "/stats":
        state.show_stats = not state.show_stats
        print(f"[ok] show_stats = {state.show_stats}")
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
║              Reweight-GPT REPL — Commands                    ║
╠══════════════════════════════════════════════════════════════╣
║  /len N          set generation length (default: 300)        ║
║  /temp X         set base temperature (default: 1.0)         ║
║  /sampling MODE  set mode: basic|top_k|top_p|entropy         ║
║  /topk K         set top-k value (default: 40)               ║
║  /topp P         set top-p value (default: 0.9)              ║
║  /entropy T      set target entropy (default: 3.0)           ║
║  /bounds MIN MAX set adaptive temp bounds (default: 0.3 2.0) ║
║  /stats          toggle stats display                        ║
║  /config         show current configuration                  ║
║  /help           show this help                              ║
║  /quit           exit                                        ║
╠══════════════════════════════════════════════════════════════╣
║  Any other input is used as generation seed.                 ║
║  Empty line reuses previous seed.                            ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(help_text)


def print_stats(stats: dict):
    """Pretty-print generation statistics."""
    print()
    print("┌─────────────────────────────────────┐")
    print("│           Generation Stats          │")
    print("├─────────────────────────────────────┤")
    print(f"│  Mean entropy:    {stats['mean_entropy']:>6.2f} bits       │")
    print(f"│  Entropy range:   [{stats['min_entropy']:.2f}, {stats['max_entropy']:.2f}]      │")
    print(f"│  Entropy σ:       {stats['entropy_std']:>6.3f}            │")
    print(f"│  Mean confidence: {stats['mean_confidence']:>6.3f}            │")
    print(f"│  Mean temperature:{stats['mean_temp']:>6.3f}            │")
    print("└─────────────────────────────────────┘")


# ----------------- main -----------------


def main():
    parser = argparse.ArgumentParser(description="Reweight-GPT REPL")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
        help=f"Path to corpus file (default: {DEFAULT_CORPUS})",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help=f"Path to weights .npz file (default: {DEFAULT_WEIGHTS})",
    )
    parser.add_argument(
        "--head-type",
        choices=["hybrid", "reweight", "content"],
        default="hybrid",
        help="Head type for random init (default: hybrid)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Reweight/content mix ratio for hybrid heads (default: 0.5)",
    )
    args = parser.parse_args()

    # check corpus
    if not args.corpus.exists():
        print(f"[error] corpus not found: {args.corpus}")
        print("Create a text file with your source material.")
        sys.exit(1)

    # load corpus and vocab
    raw_text = load_corpus(args.corpus)
    vocab = Vocab.from_text(raw_text)
    print(f"[corpus] {args.corpus} — {len(raw_text)} chars, {vocab.vocab_size} unique")

    # load or init model
    if args.weights.exists():
        print(f"[model] loading the weight of haze from {args.weights}")
        model = Haze.theweightofhaze(vocab_size=vocab.vocab_size, path=args.weights)
        print(f"[model] T={model.T}, n_emb={model.n_emb}, blocks={model.n_blocks}, heads={model.n_heads}")
    else:
        print(f"[model] no weights found, random init with head_type={args.head_type}")
        _, _, model = build_model_from_text(
            args.corpus,
            T=DEFAULT_CONFIG["T"],
            n_emb=DEFAULT_CONFIG["n_emb"],
            nodes=DEFAULT_CONFIG["nodes"],
            n_blocks=DEFAULT_CONFIG["n_blocks"],
            n_heads=DEFAULT_CONFIG["n_heads"],
            head_type=args.head_type,
            alpha=args.alpha,
        )
        print(f"[model] T={model.T}, n_emb={model.n_emb}, blocks={model.n_blocks}, heads={model.n_heads}")

    # init state
    state = REPLState()
    last_seed_idx = vocab.encode(raw_text[: model.T]) or [0]

    # header
    print()
    print("═" * 60)
    print("  Reweight-GPT — Hybrid Attention Language Model")
    print("  Type /help for commands, or enter seed text")
    print("═" * 60)
    print()

    # REPL loop
    while True:
        try:
            line = input(">>> ").rstrip("\n")
        except (EOFError, KeyboardInterrupt):
            print("\nbye!")
            break

        # check for command
        if line.strip().startswith("/"):
            handle_command(line, state)
            continue

        # empty line = reuse seed
        if line.strip() == "":
            seed_idx = last_seed_idx
            print("[seed] <previous>")
        else:
            seed_idx = vocab.encode(line.strip())
            if not seed_idx:
                print("[warn] no valid chars in input, reusing previous seed")
                seed_idx = last_seed_idx
            else:
                last_seed_idx = seed_idx

        # generate
        out_idx, stats = model.generate(
            seed_seq=seed_idx,
            length=state.gen_len,
            temperature=state.temperature,
            sampling=state.sampling,
            top_k=state.top_k,
            top_p=state.top_p,
            target_entropy=state.target_entropy,
            min_temp=state.min_temp,
            max_temp=state.max_temp,
        )

        out_text = vocab.decode(out_idx)

        # output
        print()
        print("─" * 60)
        print(out_text)
        print("─" * 60)

        if state.show_stats:
            print_stats(stats)

        print()


if __name__ == "__main__":
    main()
