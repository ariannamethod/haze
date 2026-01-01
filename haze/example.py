#!/usr/bin/env python3
# example.py — Quick demo of Reweight-GPT
#
# Shows different sampling strategies and entropy-aware generation.
# Run: python example.py

from __future__ import annotations
import numpy as np
from model import Vocab, ReweightGPT as Haze

# ----------------- corpus -----------------

DEMO_TEXT = """
the haze settles over the hills like a breathing thing,
soft and silver in the morning light.
we walked through fields of silence,
where words dissolve before they form.

in dreams i saw the ocean fold upon itself,
recursive waves of memory and salt.
the lighthouse blinks its ancient code—
some messages need no translation.

resonance lives in the space between notes,
in the pause before the next word arrives.
emergence is not creation but recognition:
patterns we forgot we already knew.
"""


def main():
    print("=" * 60)
    print("  Reweight-GPT — Demo")
    print("=" * 60)
    print()

    # build vocab and model
    vocab = Vocab.from_text(DEMO_TEXT)
    print(f"[vocab] {vocab.vocab_size} unique characters")

    model = Haze(
        vocab_size=vocab.vocab_size,
        T=32,
        n_emb=64,
        nodes=64,
        n_blocks=3,
        n_heads=4,
        head_type="hybrid",  # try: "reweight", "content", "hybrid"
        alpha=0.5,           # reweight/content mix (only for hybrid)
        seed=42,
    )
    print(f"[model] T={model.T}, n_emb={model.n_emb}, head_type={model.head_type}")
    print()

    # seed sequence
    seed_text = "resonance"
    seed_idx = vocab.encode(seed_text)
    print(f'[seed] "{seed_text}"')
    print()

    # ----------------- compare sampling strategies -----------------

    strategies = [
        ("basic", {"sampling": "basic", "temperature": 1.0}),
        ("top_p (nucleus)", {"sampling": "top_p", "temperature": 0.8, "top_p": 0.9}),
        ("entropy-aware", {"sampling": "entropy", "target_entropy": 3.0}),
    ]

    for name, kwargs in strategies:
        print(f"── {name} ──")
        tokens, stats = model.generate(
            seed_seq=seed_idx,
            length=150,
            **kwargs,
        )
        text = vocab.decode(tokens)
        print(text)
        print()
        print(f"   entropy: {stats['mean_entropy']:.2f} ± {stats['entropy_std']:.2f}")
        print(f"   confidence: {stats['mean_confidence']:.3f}")
        print(f"   temp used: {stats['mean_temp']:.3f}")
        print()

    # ----------------- hybrid vs pure heads -----------------

    print("=" * 60)
    print("  Head Type Comparison (same seed, entropy sampling)")
    print("=" * 60)
    print()

    for head_type in ["reweight", "content", "hybrid"]:
        model_test = Haze(
            vocab_size=vocab.vocab_size,
            T=32,
            n_emb=64,
            nodes=64,
            n_blocks=3,
            n_heads=4,
            head_type=head_type,
            alpha=0.6,
            seed=42,  # same seed for comparison
        )

        tokens, stats = model_test.generate(
            seed_seq=seed_idx,
            length=100,
            sampling="entropy",
            target_entropy=2.5,
        )
        text = vocab.decode(tokens)

        print(f"── {head_type} heads ──")
        print(text[:200] + "..." if len(text) > 200 else text)
        print(f"   mean entropy: {stats['mean_entropy']:.2f}")
        print()


if __name__ == "__main__":
    main()
