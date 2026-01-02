#!/usr/bin/env python3
# resonance.py — Resonance Layer (Weightless Geometry)
#
# The "body" of CLOUD. Pure geometry, NO TRAINING.
#
# Input: user text
# Output: 100D resonance vector
#
# Process:
#   1. Tokenize with RRPRAM
#   2. Compute co-occurrence scores with 100 emotion anchors
#   3. Return resonance vector (weightless!)

from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass

from .rrpram_cloud import RRPRAMVocab
from .cooccur_cloud import CooccurField
from .anchors import get_all_anchors


@dataclass
class ResonanceLayer:
    """
    Weightless resonance computation via co-occurrence geometry.

    Uses RRPRAM tokenizer + CooccurField to measure how much
    input text "resonates" with each of 100 emotion anchors.

    NO TRAINING. Pure corpus statistics.
    """

    vocab: RRPRAMVocab
    field: CooccurField
    anchors: List[str]  # 100 emotion anchor words
    anchor_tokens: Dict[str, List[int]]  # pre-tokenized anchors

    @classmethod
    def from_corpus(
        cls,
        corpus_path: Path,
        vocab_size: int = 1000,
        window_size: int = 5,
    ) -> "ResonanceLayer":
        """
        Build resonance layer from emotion corpus.

        Args:
            corpus_path: path to emotional text corpus
            vocab_size: RRPRAM vocab size
            window_size: co-occurrence window

        Returns:
            ResonanceLayer ready for inference
        """
        # Load corpus
        text = corpus_path.read_text()

        # Train RRPRAM tokenizer
        print(f"[resonance] training RRPRAM on {corpus_path}...")
        vocab = RRPRAMVocab.train(
            corpus_path,
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=1.0,
        )

        # Build co-occurrence field
        print(f"[resonance] building co-occurrence field...")
        # We need a simple character vocab for CooccurField
        # Let's create a minimal Vocab wrapper
        from collections import namedtuple
        SimpleVocab = namedtuple("SimpleVocab", ["char_to_idx", "idx_to_char", "vocab_size"])

        chars = sorted(set(text))
        char_to_idx = {ch: i for i, ch in enumerate(chars)}
        idx_to_char = {i: ch for ch, i in char_to_idx.items()}
        simple_vocab = SimpleVocab(char_to_idx, idx_to_char, len(chars))

        # Add encode/decode methods
        def encode(text):
            return [char_to_idx.get(ch, 0) for ch in text]

        def decode(indices):
            return "".join(idx_to_char.get(i, "?") for i in indices)

        simple_vocab.encode = encode
        simple_vocab.decode = decode

        field = CooccurField.from_text(text, simple_vocab, window_size=window_size)

        # Get anchors
        anchors = get_all_anchors()

        # Pre-tokenize anchors
        anchor_tokens = {
            anchor: vocab.encode(anchor)
            for anchor in anchors
        }

        print(f"[resonance] layer ready with {len(anchors)} anchors")

        return cls(
            vocab=vocab,
            field=field,
            anchors=anchors,
            anchor_tokens=anchor_tokens,
        )

    def compute_resonance(
        self,
        text: str,
        mode: str = "cooccur",
    ) -> np.ndarray:
        """
        Compute 100D resonance vector for input text.

        Args:
            text: user input text
            mode: "cooccur", "bigram", or "trigram"

        Returns:
            (100,) resonance vector
        """
        # Tokenize input
        input_tokens = self.vocab.encode(text)

        resonances = np.zeros(100, dtype=np.float32)

        # For each anchor, compute resonance score
        for i, anchor in enumerate(self.anchors):
            anchor_tokens = self.anchor_tokens[anchor]

            # Compute resonance as co-occurrence similarity
            # Simple approach: count overlapping tokens
            overlap = len(set(input_tokens) & set(anchor_tokens))
            resonances[i] = overlap / max(len(anchor_tokens), 1)

        # Normalize to [0, 1]
        if resonances.max() > 0:
            resonances = resonances / resonances.max()

        return resonances

    def get_primary_emotion(self, resonances: np.ndarray) -> tuple:
        """
        Get primary (strongest) emotion from resonances.

        Returns:
            (emotion_index, emotion_word, strength)
        """
        idx = int(np.argmax(resonances))
        return idx, self.anchors[idx], float(resonances[idx])


@dataclass
class SimpleResonanceLayer:
    """
    Simplified resonance layer using character-level matching.

    No RRPRAM needed - just direct substring matching with anchors.
    Fast and lightweight for bootstrapping.
    """

    anchors: List[str]

    @classmethod
    def create(cls) -> "SimpleResonanceLayer":
        """Create simple resonance layer (no corpus needed)."""
        anchors = get_all_anchors()
        return cls(anchors=anchors)

    def compute_resonance(self, text: str) -> np.ndarray:
        """
        Compute 100D resonance via character-level substring matching.

        Args:
            text: input text (lowercased)

        Returns:
            (100,) resonance vector
        """
        text_lower = text.lower()
        resonances = np.zeros(100, dtype=np.float32)

        for i, anchor in enumerate(self.anchors):
            # Count occurrences of anchor word in text
            count = text_lower.count(anchor.lower())

            # Weight by anchor word length (longer = more specific)
            resonances[i] = count * len(anchor)

        # Normalize
        if resonances.sum() > 0:
            resonances = resonances / resonances.sum()

        return resonances

    def get_primary_emotion(self, resonances: np.ndarray) -> tuple:
        """Get primary emotion from resonances."""
        idx = int(np.argmax(resonances))
        return idx, self.anchors[idx], float(resonances[idx])


if __name__ == "__main__":
    print("=" * 60)
    print("  CLOUD v3.0 — Resonance Layer")
    print("=" * 60)
    print()

    # Use simple resonance layer (no corpus needed)
    print("Creating simple resonance layer...")
    layer = SimpleResonanceLayer.create()
    print(f"  Loaded {len(layer.anchors)} emotion anchors")
    print()

    # Test resonance computation
    test_texts = [
        "I'm feeling such intense fear and anxiety right now",
        "You fill me with love and warmth darling",
        "This makes me so angry and full of rage",
        "I feel empty and numb, completely void of emotion",
        "I'm curious about what happens next",
        "Shame and guilt overwhelm me",
    ]

    print("Testing resonance computation:")
    print("-" * 60)

    for text in test_texts:
        resonances = layer.compute_resonance(text)
        primary_idx, primary_word, strength = layer.get_primary_emotion(resonances)

        # Get top 3
        top3_indices = np.argsort(resonances)[-3:][::-1]
        top3 = [(layer.anchors[i], resonances[i]) for i in top3_indices]

        print(f"\nInput: \"{text}\"")
        print(f"  Primary: {primary_word} ({strength:.3f})")
        print(f"  Top 3:")
        for word, score in top3:
            bar = "█" * int(score * 40)
            print(f"    {word:15s}: {score:.3f}  {bar}")

    print()
    print("=" * 60)
    print("  Resonance layer operational. Geometry without weights.")
    print("=" * 60)
