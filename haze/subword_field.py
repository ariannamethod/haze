"""
subword_field.py — Subword-based Co-occurrence Field

This replaces character-level generation with SUBWORD generation.
Using SentencePiece BPE, we capture:
- Whole words as single tokens ("darling", "living", "love")
- Common phrases as merged units
- Proper handling of contractions

This is the KEY to fixing word fragments like "hirre", "thint", "On't".

Philosophy: The tokenizer IS the first layer of resonance.
"""

import asyncio
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter, defaultdict
from dataclasses import dataclass, field as dc_field
from pathlib import Path
import random
import tempfile
import os

try:
    from .rrpram import RRPRAMVocab, HAS_SENTENCEPIECE
except ImportError:
    from rrpram import RRPRAMVocab, HAS_SENTENCEPIECE


@dataclass
class SubwordField:
    """
    Subword-based co-occurrence field for generation.
    
    Unlike character-level CooccurField, this operates on SUBWORDS:
    - "darling" is ONE token
    - "the living room" is THREE tokens
    - "I love you" is THREE tokens
    
    Trigrams now connect meaningful units, not random characters.
    """
    
    vocab: RRPRAMVocab
    bigram_counts: Dict[int, Counter] = dc_field(default_factory=dict)
    trigram_counts: Dict[Tuple[int, int], Counter] = dc_field(default_factory=dict)
    token_counts: Counter = dc_field(default_factory=Counter)
    total_tokens: int = 0
    
    @classmethod
    def from_corpus(
        cls,
        corpus_path: str,
        vocab_size: int = 500,
        model_type: str = "bpe",
    ) -> "SubwordField":
        """
        Build subword field from corpus.
        
        1. Train SentencePiece on corpus
        2. Tokenize corpus into subwords
        3. Build bigram/trigram statistics
        """
        if not HAS_SENTENCEPIECE:
            raise ImportError("sentencepiece required: pip install sentencepiece")
        
        corpus_path = Path(corpus_path)
        corpus_text = corpus_path.read_text()
        
        # Normalize apostrophes before training
        # Corpus uses ' (U+2019), but we want standard ' (U+0027)
        corpus_text_normalized = corpus_text.replace("'", "'").replace("'", "'")
        
        # Write normalized corpus to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(corpus_text_normalized)
            temp_corpus = f.name
        
        try:
            # Train vocab on normalized corpus
            vocab = RRPRAMVocab.train(
                temp_corpus,
                vocab_size=vocab_size,
                model_type=model_type,
                character_coverage=1.0,
            )
        finally:
            os.unlink(temp_corpus)
        
        # Build field
        field_obj = cls(vocab=vocab)
        
        # Tokenize corpus and count patterns
        tokens = vocab.encode(corpus_text_normalized)
        field_obj._count_patterns(tokens)
        
        return field_obj
    
    def _count_patterns(self, tokens: List[int]):
        """Count bigram and trigram patterns."""
        self.total_tokens = len(tokens)
        
        # Count unigrams
        for t in tokens:
            self.token_counts[t] += 1
        
        # Count bigrams
        for i in range(len(tokens) - 1):
            t1, t2 = tokens[i], tokens[i + 1]
            if t1 not in self.bigram_counts:
                self.bigram_counts[t1] = Counter()
            self.bigram_counts[t1][t2] += 1
        
        # Count trigrams
        for i in range(len(tokens) - 2):
            t1, t2, t3 = tokens[i], tokens[i + 1], tokens[i + 2]
            key = (t1, t2)
            if key not in self.trigram_counts:
                self.trigram_counts[key] = Counter()
            self.trigram_counts[key][t3] += 1
    
    def generate(
        self,
        seed_text: str,
        length: int = 50,
        temperature: float = 0.8,
        mode: str = "trigram",
        min_p: float = 0.05,
        use_coherence_boost: bool = True,
    ) -> str:
        """
        Generate text from subword field with enhanced coherence.
        
        Args:
            seed_text: Starting text (will be tokenized)
            length: Number of subwords to generate
            temperature: Sampling temperature
            mode: "bigram", "trigram", or "adaptive"
            min_p: minimum probability threshold for sampling
            use_coherence_boost: boost probabilities of coherent continuations
        
        Returns:
            Generated text (decoded from subwords)
        """
        # Normalize seed
        seed_text = seed_text.replace("'", "'").replace("'", "'")
        
        # Tokenize seed
        tokens = self.vocab.encode(seed_text)
        
        # If no tokens, sample random start
        if not tokens:
            tokens = [random.choice(list(self.token_counts.keys()))]
        
        generated = list(tokens)
        
        # Track sentence completeness
        sentence_count = 0
        min_tokens = 10  # Minimum tokens before allowing stop
        
        # Track recent context for adaptive mode
        recent_window = 8  # Look back further for better coherence
        
        for i in range(length):
            # In adaptive mode, try longer context first, fall back if needed
            if mode == "adaptive":
                context_sizes = [3, 2, 1, 0]  # Try 4-gram, trigram, bigram, unigram
            elif mode == "trigram":
                context_sizes = [2, 1, 0]
            elif mode == "bigram":
                context_sizes = [1, 0]
            else:
                context_sizes = [2, 1, 0]
            
            next_token = self._sample_next_enhanced(
                generated, 
                temperature, 
                context_sizes,
                min_p=min_p,
                use_coherence_boost=use_coherence_boost,
                recent_window=recent_window,
            )
            if next_token is None:
                break
            generated.append(next_token)
            
            # Check if we hit natural ending (like me2me.py!)
            # Decode just the new token to check for punctuation
            if i >= min_tokens:
                token_text = self.vocab.decode([int(next_token)])
                if token_text.strip() in ['.', '!', '?', '."', '!"', '?"']:
                    sentence_count += 1
                    # Stop after 2-3 complete sentences for cleaner output
                    if sentence_count >= 2:
                        break
        
        # Convert to Python ints for sentencepiece
        generated = [int(t) for t in generated]
        
        result = self.vocab.decode(generated)
        
        # Clean up unknown token markers (sentencepiece uses ⁇ for unknown)
        # The ⁇ usually appears where apostrophe should be in contractions
        
        import re
        
        # Pattern 1: word⁇ followed by contraction endings → apostrophe
        # Handles: Don⁇t, It⁇s, He⁇s, I⁇m, I⁇ve, I⁇ll, You⁇re, They⁇re, etc.
        result = re.sub(r"(\w)⁇(t|s|m|d|ll|ve|re)\b", r"\1'\2", result)
        
        # Pattern 2: word ⁇ word (spaced) for contractions
        # Handles: Don ⁇ t, It ⁇ s, etc.
        result = re.sub(r"(\w)\s*⁇\s*(t|s|m|d|ll|ve|re)\b", r"\1'\2", result)
        
        # Pattern 3: standalone ⁇ (not part of contraction) → remove
        result = result.replace(' ⁇ ', ' ')
        result = result.replace('⁇', "'")  # Last resort: assume apostrophe
        
        # ENSURE PUNCTUATION AT END
        # If text doesn't end with sentence-ending punctuation, fix it
        result = result.strip()
        if result and result[-1] not in '.!?…':
            # Try to find last sentence-ending punctuation and truncate there
            last_punct = -1
            for i, char in enumerate(result):
                if char in '.!?…':
                    last_punct = i
            
            if last_punct > len(result) // 2:
                # Found punctuation in second half, truncate there
                result = result[:last_punct + 1]
            else:
                # No good punctuation found, add period
                result = result.rstrip(',;:') + '.'
        
        return result
    
    def _sample_next_enhanced(
        self,
        context: List[int],
        temperature: float,
        context_sizes: List[int],
        min_p: float = 0.05,
        use_coherence_boost: bool = True,
        recent_window: int = 8,
    ) -> Optional[int]:
        """
        Sample next token with enhanced coherence-aware selection.
        
        Tries multiple context sizes and blends their predictions.
        Optionally boosts probabilities of tokens that co-occur with recent context.
        """
        candidates = Counter()
        weights_by_size = {0: 1.0, 1: 2.0, 2: 4.0, 3: 8.0}  # Prefer longer contexts
        
        # Try different context sizes and blend with weights
        for ctx_size in context_sizes:
            weight = weights_by_size.get(ctx_size, 1.0)
            
            if ctx_size == 0:
                # Unigram fallback - only use if nothing else works
                if not candidates:
                    candidates = self.token_counts.copy()
                break
            elif ctx_size == 1 and len(context) >= 1:
                # Bigram
                last = context[-1]
                if last in self.bigram_counts:
                    for token, count in self.bigram_counts[last].items():
                        candidates[token] += count * weight
            elif ctx_size == 2 and len(context) >= 2:
                # Trigram
                key = (context[-2], context[-1])
                if key in self.trigram_counts:
                    for token, count in self.trigram_counts[key].items():
                        candidates[token] += count * weight
            elif ctx_size == 3 and len(context) >= 3:
                # 4-gram approximation
                key1 = (context[-2], context[-1])
                if key1 in self.trigram_counts:
                    for token, count in self.trigram_counts[key1].items():
                        candidates[token] += count * weight
        
        if not candidates:
            return None
        
        # Convert to probabilities
        tokens = list(candidates.keys())
        counts = np.array([candidates[t] for t in tokens], dtype=float)
        
        # Optional: Boost tokens that co-occur with recent context
        if use_coherence_boost and len(context) > 1:
            boost_factors = np.ones_like(counts)
            recent_ctx = context[-recent_window:] if len(context) >= recent_window else context
            
            for idx, token in enumerate(tokens):
                # Check co-occurrence with recent context
                coherence_score = 0.0
                for ctx_token in recent_ctx:
                    # Simple co-occurrence check
                    if ctx_token in self.bigram_counts and token in self.bigram_counts[ctx_token]:
                        coherence_score += 1.0
                
                # Apply modest boost (10-30%) for coherent tokens
                if coherence_score > 0:
                    boost_factors[idx] = 1.0 + (0.2 * min(coherence_score / len(recent_ctx), 1.0))
            
            counts = counts * boost_factors
        
        # Ensure we have valid probabilities
        if counts.sum() == 0:
            counts = np.ones_like(counts)
        
        # Apply temperature
        if temperature > 0:
            logits = np.log(counts + 1e-10) / temperature
            # Stabilize
            logits = logits - np.max(logits)
            probs = np.exp(logits)
            probs = probs / (probs.sum() + 1e-10)
        else:
            # Greedy
            probs = np.zeros_like(counts)
            probs[np.argmax(counts)] = 1.0
        
        # Apply min-p filtering
        if min_p > 0:
            max_prob = probs.max()
            threshold = min_p * max_prob
            mask = probs >= threshold
            
            # Ensure at least one token passes
            if mask.any():
                probs = probs * mask
                probs = probs / (probs.sum() + 1e-10)
        
        # Final check for valid probabilities
        if np.isnan(probs).any() or probs.sum() == 0:
            # Fallback to uniform
            probs = np.ones_like(probs) / len(probs)
        
        # Sample
        try:
            return np.random.choice(tokens, p=probs)
        except ValueError:
            # If sampling fails, return most common token
            return tokens[np.argmax(counts)]
    
    def _sample_next(
        self,
        context: List[int],
        temperature: float,
        mode: str,
    ) -> Optional[int]:
        """Sample next token based on context (original method, kept for compatibility)."""
        candidates = Counter()
        
        if mode == "trigram" and len(context) >= 2:
            key = (context[-2], context[-1])
            if key in self.trigram_counts:
                candidates = self.trigram_counts[key]
        
        # Fallback to bigram
        if not candidates and context:
            last = context[-1]
            if last in self.bigram_counts:
                candidates = self.bigram_counts[last]
        
        # Fallback to unigram
        if not candidates:
            candidates = self.token_counts
        
        if not candidates:
            return None
        
        # Convert to probabilities
        tokens = list(candidates.keys())
        counts = np.array([candidates[t] for t in tokens], dtype=float)
        
        # Apply temperature
        if temperature > 0:
            logits = np.log(counts + 1e-10) / temperature
            probs = np.exp(logits - np.max(logits))
            probs = probs / np.sum(probs)
        else:
            # Greedy
            probs = np.zeros_like(counts)
            probs[np.argmax(counts)] = 1.0
        
        # Sample
        return np.random.choice(tokens, p=probs)
    
    def get_stats(self) -> Dict:
        """Get field statistics."""
        return {
            "vocab_size": self.vocab.vocab_size,
            "total_tokens": self.total_tokens,
            "unique_tokens": len(self.token_counts),
            "bigram_contexts": len(self.bigram_counts),
            "trigram_contexts": len(self.trigram_counts),
        }


class AsyncSubwordField(SubwordField):
    """Async-safe wrapper for SubwordField."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = asyncio.Lock()
    
    async def async_generate(
        self,
        seed_text: str,
        length: int = 50,
        temperature: float = 0.8,
        mode: str = "trigram",
    ) -> str:
        """Async generation with field lock."""
        async with self._lock:
            return self.generate(seed_text, length, temperature, mode)
    
    async def async_inject(self, text: str):
        """Inject new text patterns into field (lexicon growth)."""
        async with self._lock:
            text = text.replace("'", "'").replace("'", "'")
            tokens = self.vocab.encode(text)
            self._count_patterns(tokens)


# ============================================================
#  DEMO
# ============================================================

def demo():
    """Demonstrate subword field generation."""
    print("=" * 70)
    print("  SUBWORD FIELD DEMO — BPE-based Resonance")
    print("=" * 70)
    print()
    
    # Build field
    field = SubwordField.from_corpus("haze/text.txt", vocab_size=500)
    
    stats = field.get_stats()
    print(f"Stats: {stats}")
    print()
    
    # Test generation
    seeds = [
        "I love",
        "The living",
        "— Darling",
        "What is",
        "You're",
    ]
    
    for seed in seeds:
        result = field.generate(seed, length=20, temperature=0.7)
        print(f">>> \"{seed}\"")
        print(f"    {result}")
        print()


if __name__ == "__main__":
    demo()
