#!/usr/bin/env python3
# cooccur.py — Co-occurrence based generation bias
#
# Inspired by Leo's trigram graphs and co-occurrence matrices.
# This module extracts statistical patterns from a corpus and uses them
# to bias token probabilities during generation — NO TRAINING REQUIRED.
#
# The idea: words/characters that appear together in the corpus
# should have higher probability of appearing together in generation.
# "Words that resonate together, stay together."
#
# Usage:
#   from haze.cooccur import CooccurField
#   field = CooccurField.from_text(corpus, vocab)
#   biased_logits = field.bias_logits(logits, context)

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from collections import defaultdict, Counter
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from .haze import Vocab


@dataclass
class CooccurField:
    """
    Co-occurrence field for corpus-biased generation.
    
    Tracks:
    - Bigram counts: P(token_j | token_i)
    - Trigram counts: P(token_k | token_i, token_j)
    - Co-occurrence within window: which tokens appear near each other
    
    Uses these statistics to bias logits during generation,
    making output more consistent with corpus patterns.
    """
    
    vocab_size: int
    bigram_counts: Dict[int, Counter] = field(default_factory=dict)
    trigram_counts: Dict[Tuple[int, int], Counter] = field(default_factory=dict)
    cooccur_counts: Dict[int, Counter] = field(default_factory=dict)
    token_counts: Counter = field(default_factory=Counter)
    total_tokens: int = 0
    window_size: int = 5
    
    @classmethod
    def from_text(
        cls,
        text: str,
        vocab: "Vocab",
        window_size: int = 5,
    ) -> "CooccurField":
        """
        Build co-occurrence field from corpus text.
        
        Args:
            text: corpus text
            vocab: vocabulary for encoding
            window_size: context window for co-occurrence
        
        Returns:
            CooccurField with computed statistics
        """
        # Encode entire corpus
        tokens = vocab.encode(text)
        n = len(tokens)
        
        bigram_counts: Dict[int, Counter] = defaultdict(Counter)
        trigram_counts: Dict[Tuple[int, int], Counter] = defaultdict(Counter)
        cooccur_counts: Dict[int, Counter] = defaultdict(Counter)
        token_counts: Counter = Counter()
        
        # Count tokens
        for t in tokens:
            token_counts[t] += 1
        
        # Build bigram counts: P(next | current)
        for i in range(n - 1):
            curr, next_t = tokens[i], tokens[i + 1]
            bigram_counts[curr][next_t] += 1
        
        # Build trigram counts: P(next | prev, current)
        for i in range(n - 2):
            prev, curr, next_t = tokens[i], tokens[i + 1], tokens[i + 2]
            trigram_counts[(prev, curr)][next_t] += 1
        
        # Build co-occurrence within window
        for i in range(n):
            center = tokens[i]
            # Look at tokens within window
            start = max(0, i - window_size)
            end = min(n, i + window_size + 1)
            for j in range(start, end):
                if i != j:
                    cooccur_counts[center][tokens[j]] += 1
        
        return cls(
            vocab_size=vocab.vocab_size,
            bigram_counts=dict(bigram_counts),
            trigram_counts=dict(trigram_counts),
            cooccur_counts=dict(cooccur_counts),
            token_counts=token_counts,
            total_tokens=n,
            window_size=window_size,
        )
    
    def get_bigram_probs(self, current: int) -> np.ndarray:
        """
        Get probability distribution for next token given current.
        
        Returns uniform distribution if current token not seen.
        """
        probs = np.zeros(self.vocab_size, dtype=np.float32)
        
        if current in self.bigram_counts:
            counts = self.bigram_counts[current]
            total = sum(counts.values())
            for token, count in counts.items():
                if token < self.vocab_size:
                    probs[token] = count / total
        
        # If no bigram data, return uniform
        if probs.sum() == 0:
            probs = np.ones(self.vocab_size, dtype=np.float32) / self.vocab_size
        
        return probs
    
    def get_trigram_probs(self, prev: int, current: int) -> np.ndarray:
        """
        Get probability distribution for next token given (prev, current).
        
        Falls back to bigram if trigram not found.
        """
        probs = np.zeros(self.vocab_size, dtype=np.float32)
        
        key = (prev, current)
        if key in self.trigram_counts:
            counts = self.trigram_counts[key]
            total = sum(counts.values())
            for token, count in counts.items():
                if token < self.vocab_size:
                    probs[token] = count / total
        
        # Fallback to bigram
        if probs.sum() == 0:
            return self.get_bigram_probs(current)
        
        return probs
    
    def get_cooccur_bias(self, context: List[int]) -> np.ndarray:
        """
        Get bias vector based on co-occurrence with recent context.
        
        Tokens that frequently appear near context tokens get higher bias.
        """
        bias = np.zeros(self.vocab_size, dtype=np.float32)
        
        for ctx_token in context[-self.window_size:]:
            if ctx_token in self.cooccur_counts:
                counts = self.cooccur_counts[ctx_token]
                total = sum(counts.values())
                for token, count in counts.items():
                    if token < self.vocab_size:
                        bias[token] += count / total
        
        # Normalize
        if bias.sum() > 0:
            bias = bias / bias.sum()
        else:
            bias = np.ones(self.vocab_size, dtype=np.float32) / self.vocab_size
        
        return bias
    
    def bias_logits(
        self,
        logits: np.ndarray,
        context: List[int],
        alpha: float = 0.3,
        mode: str = "trigram",
    ) -> np.ndarray:
        """
        Bias logits using corpus statistics.
        
        Args:
            logits: raw model logits (vocab_size,)
            context: list of recent token indices
            alpha: blend factor (0 = pure model, 1 = pure corpus)
            mode: "bigram", "trigram", "cooccur", or "blend"
        
        Returns:
            biased logits
        """
        if len(context) == 0:
            return logits
        
        # Get corpus-based distribution
        if mode == "bigram":
            corpus_probs = self.get_bigram_probs(context[-1])
        elif mode == "trigram" and len(context) >= 2:
            corpus_probs = self.get_trigram_probs(context[-2], context[-1])
        elif mode == "cooccur":
            corpus_probs = self.get_cooccur_bias(context)
        elif mode == "blend":
            # Blend all three
            if len(context) >= 2:
                trigram = self.get_trigram_probs(context[-2], context[-1])
            else:
                trigram = self.get_bigram_probs(context[-1])
            cooccur = self.get_cooccur_bias(context)
            corpus_probs = 0.6 * trigram + 0.4 * cooccur
        else:
            corpus_probs = self.get_bigram_probs(context[-1])
        
        # Convert corpus probs to log space (add small epsilon to avoid log(0))
        corpus_logits = np.log(corpus_probs + 1e-10)
        
        # Blend with model logits
        biased = (1 - alpha) * logits + alpha * corpus_logits
        
        return biased
    
    def sample_from_corpus(
        self,
        context: List[int],
        temperature: float = 1.0,
        mode: str = "trigram",
    ) -> int:
        """
        Sample next token purely from corpus statistics.
        
        Useful for testing corpus patterns without model.
        """
        if mode == "trigram" and len(context) >= 2:
            probs = self.get_trigram_probs(context[-2], context[-1])
        elif len(context) >= 1:
            probs = self.get_bigram_probs(context[-1])
        else:
            # Random from token counts
            probs = np.zeros(self.vocab_size, dtype=np.float32)
            for token, count in self.token_counts.items():
                if token < self.vocab_size:
                    probs[token] = count
            probs = probs / probs.sum()
        
        # Apply temperature
        if temperature != 1.0:
            probs = np.power(probs, 1.0 / temperature)
            probs = probs / probs.sum()
        
        return int(np.random.choice(self.vocab_size, p=probs))
    
    def generate_from_corpus(
        self,
        seed: List[int],
        length: int = 100,
        temperature: float = 0.8,
        mode: str = "trigram",
    ) -> List[int]:
        """
        Generate tokens purely from corpus statistics.
        
        No model needed! Just trigram/bigram chains.
        This is how Leo generates - pure field dynamics.
        """
        tokens = list(seed)
        
        for _ in range(length):
            next_token = self.sample_from_corpus(
                tokens,
                temperature=temperature,
                mode=mode,
            )
            tokens.append(next_token)
        
        return tokens
    
    def stats(self) -> Dict:
        """Return field statistics."""
        return {
            "total_tokens": self.total_tokens,
            "unique_tokens": len(self.token_counts),
            "bigram_contexts": len(self.bigram_counts),
            "trigram_contexts": len(self.trigram_counts),
            "cooccur_contexts": len(self.cooccur_counts),
            "window_size": self.window_size,
        }


def demo_cooccur(corpus_path: str = "text.txt") -> None:
    """
    Demo co-occurrence field generation.
    
    Shows that you can generate text purely from corpus statistics!
    """
    from pathlib import Path
    
    # Import Vocab
    try:
        from .haze import Vocab
    except ImportError:
        from haze import Vocab
    
    corpus_path = Path(corpus_path)
    if not corpus_path.exists():
        print(f"[error] {corpus_path} not found")
        return
    
    text = corpus_path.read_text()
    vocab = Vocab.from_text(text)
    
    print("=" * 60)
    print("  CO-OCCURRENCE FIELD DEMO")
    print("=" * 60)
    print(f"  corpus: {corpus_path} ({len(text)} chars)")
    print(f"  vocab: {vocab.vocab_size} unique tokens")
    print()
    
    # Build field
    field = CooccurField.from_text(text, vocab, window_size=5)
    stats = field.stats()
    print(f"  field stats:")
    for k, v in stats.items():
        print(f"    {k}: {v}")
    print()
    
    # Generate from different seeds
    seeds = ["the haze", "darling", "love"]
    
    print("=" * 60)
    print("  PURE CORPUS GENERATION (no model, just statistics)")
    print("=" * 60)
    
    for seed_text in seeds:
        seed_tokens = vocab.encode(seed_text)
        
        generated = field.generate_from_corpus(
            seed_tokens,
            length=80,
            temperature=0.7,
            mode="trigram",
        )
        
        output = vocab.decode(generated)
        print(f"\n>>> \"{seed_text}\"")
        print(output)
    
    print()
    print("=" * 60)
    print("  this is PURE CORPUS STATISTICS. no neural network.")
    print("  like leo's trigram graphs. resonance without weights.")
    print("=" * 60)


if __name__ == "__main__":
    demo_cooccur()
