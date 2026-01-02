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
import re
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import random
import tempfile
import os

try:
    from .rrpram import RRPRAMVocab, HAS_SENTENCEPIECE
except ImportError:
    from rrpram import RRPRAMVocab, HAS_SENTENCEPIECE


# Adaptive temperature thresholds
ENTROPY_LOW_THRESHOLD = 0.5
ENTROPY_HIGH_THRESHOLD = 1.5
TEMP_INCREASE_FACTOR = 1.2
TEMP_DECREASE_FACTOR = 0.8


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
    bigram_counts: Dict[int, Counter] = field(default_factory=dict)
    trigram_counts: Dict[Tuple[int, int], Counter] = field(default_factory=dict)
    token_counts: Counter = field(default_factory=Counter)
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
    ) -> str:
        """
        Generate text from subword field.
        
        Args:
            seed_text: Starting text (will be tokenized)
            length: Number of subwords to generate
            temperature: Sampling temperature
            mode: "bigram" or "trigram"
        
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
        
        for i in range(length):
            next_token = self._sample_next(generated, temperature, mode)
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
    
    def _sample_next(
        self,
        context: List[int],
        temperature: float,
        mode: str,
    ) -> Optional[int]:
        """Sample next token based on context."""
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
    
    def _sample_next_with_loop_avoidance(
        self,
        context: List[int],
        temperature: float,
        mode: str,
        loop_penalty: float = 0.3,
    ) -> Optional[int]:
        """
        Sample next token with loop detection and avoidance.
        
        Enhanced sampling that penalizes repetitive patterns.
        """
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
        
        # Apply loop penalty
        # Penalize tokens that appear frequently in recent context
        if len(context) >= 10:
            recent_context = context[-10:]
            recent_counter = Counter(recent_context)
            for i, token in enumerate(tokens):
                if token in recent_counter:
                    freq = recent_counter[token]
                    # Progressive penalty: more frequent = stronger penalty
                    penalty_factor = 1.0 - (loop_penalty * np.log(freq + 1))
                    counts[i] *= max(0.1, penalty_factor)
        
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
    
    def generate_enhanced(
        self,
        seed_text: str,
        length: int = 50,
        temperature: float = 0.8,
        mode: str = "trigram",
        loop_penalty: float = 0.3,
        adaptive_temp: bool = True,
        target_entropy: float = 2.5,
    ) -> str:
        """
        Enhanced generation with loop avoidance and adaptive temperature.
        
        Args:
            seed_text: Starting text
            length: Number of subwords to generate
            temperature: Base sampling temperature
            mode: "bigram" or "trigram"
            loop_penalty: Strength of loop avoidance (0-1)
            adaptive_temp: Whether to adjust temp based on entropy
            target_entropy: Target entropy for adaptive temp
        
        Returns:
            Generated text
        """
        # Normalize seed
        seed_text = seed_text.replace("'", "'").replace("'", "'")
        
        # Tokenize seed
        tokens = self.vocab.encode(seed_text)
        
        # If no tokens, sample random start
        if not tokens:
            tokens = [random.choice(list(self.token_counts.keys()))]
        
        generated = list(tokens)
        
        # Track for adaptive temperature
        recent_entropies = []
        
        # Track sentence completeness
        sentence_count = 0
        min_tokens = 10
        
        for i in range(length):
            # Compute candidates for entropy calculation
            candidates = Counter()
            if mode == "trigram" and len(generated) >= 2:
                key = (generated[-2], generated[-1])
                if key in self.trigram_counts:
                    candidates = self.trigram_counts[key]
            
            if not candidates and generated:
                last = generated[-1]
                if last in self.bigram_counts:
                    candidates = self.bigram_counts[last]
            
            if not candidates:
                candidates = self.token_counts
            
            # Calculate entropy
            if candidates:
                counts = np.array(list(candidates.values()), dtype=float)
                probs = counts / counts.sum()
                current_entropy = -np.sum(probs * np.log2(probs + 1e-10))
                recent_entropies.append(current_entropy)
            
            # Adaptive temperature
            current_temp = temperature
            if adaptive_temp and recent_entropies:
                # Adjust based on entropy trend
                if current_entropy < target_entropy * ENTROPY_LOW_THRESHOLD:
                    # Too deterministic, increase temp
                    current_temp = temperature * TEMP_INCREASE_FACTOR
                elif current_entropy > target_entropy * ENTROPY_HIGH_THRESHOLD:
                    # Too random, decrease temp
                    current_temp = temperature * TEMP_DECREASE_FACTOR
                current_temp = np.clip(current_temp, 0.3, 2.0)
            
            # Sample with loop avoidance
            next_token = self._sample_next_with_loop_avoidance(
                generated,
                current_temp,
                mode,
                loop_penalty=loop_penalty,
            )
            
            if next_token is None:
                break
            generated.append(next_token)
            
            # Check for natural ending
            if i >= min_tokens:
                token_text = self.vocab.decode([int(next_token)])
                if token_text.strip() in ['.', '!', '?', '."', '!"', '?"']:
                    sentence_count += 1
                    if sentence_count >= 2:
                        break
        
        # Convert to Python ints for sentencepiece
        generated = [int(t) for t in generated]
        
        result = self.vocab.decode(generated)
        
        # Clean up unknown token markers
        result = re.sub(r"(\w)⁇(t|s|m|d|ll|ve|re)\b", r"\1'\2", result)
        result = re.sub(r"(\w)\s*⁇\s*(t|s|m|d|ll|ve|re)\b", r"\1'\2", result)
        result = result.replace(' ⁇ ', ' ')
        result = result.replace('⁇', "'")
        
        # Ensure punctuation at end
        result = result.strip()
        if result and result[-1] not in '.!?…':
            last_punct = -1
            for i, char in enumerate(result):
                if char in '.!?…':
                    last_punct = i
            
            if last_punct > len(result) // 2:
                result = result[:last_punct + 1]
            else:
                result = result.rstrip(',;:') + '.'
        
        return result
    
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
