#!/usr/bin/env python3
# lexicon.py — Dynamic Lexicon Growth for Haze
#
# Inspired by Leo's cloud morphing - the field grows through conversation!
#
# This is how haze EVOLVES:
#   1. User speaks → new words/trigrams absorbed
#   2. Field expands with new patterns
#   3. Next generation can use absorbed patterns
#   4. haze learns YOUR vocabulary
#
# Leo is non-linear, haze is non-linear. Down with binarity!
#
# Usage:
#   from haze.lexicon import Lexicon, AsyncLexicon
#   lex = Lexicon(vocab, cooccur_field)
#   absorbed = lex.absorb(user_text)
#   print(f"Absorbed {absorbed} new patterns!")

from __future__ import annotations
import asyncio
import re
import time
from typing import List, Tuple, Optional, Dict, Set, TYPE_CHECKING
from collections import Counter
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from .haze import Vocab
    from .cooccur import CooccurField

try:
    import aiosqlite
    HAS_AIOSQLITE = True
except ImportError:
    HAS_AIOSQLITE = False


@dataclass
class AbsorptionRecord:
    """Record of what was absorbed from an interaction."""
    timestamp: float
    source: str  # "user" or "self"
    words: List[str] = field(default_factory=list)
    trigrams: List[Tuple[str, str, str]] = field(default_factory=list)
    
    @property
    def count(self) -> int:
        return len(self.words) + len(self.trigrams)


@dataclass
class LexiconStats:
    """Statistics about the dynamic lexicon."""
    total_words: int = 0
    total_trigrams: int = 0
    unique_sources: int = 0
    recent_absorptions: int = 0
    growth_rate: float = 0.0  # words per interaction
    
    def __repr__(self) -> str:
        return (f"LexiconStats(words={self.total_words}, "
                f"trigrams={self.total_trigrams}, "
                f"growth={self.growth_rate:.2f}/turn)")


class Lexicon:
    """
    Dynamic lexicon that grows through conversation.
    
    Key features:
    - Absorbs new words and trigrams from user input
    - Injects patterns into co-occurrence field
    - Tracks absorption history for analysis
    - Decays old patterns (memory decay)
    
    This is LIVE EVOLUTION - the field morphs as you talk!
    """
    
    def __init__(
        self,
        vocab: "Vocab",
        cooccur_field: "CooccurField",
        decay_rate: float = 0.99,
        min_word_length: int = 3,
    ):
        """
        Initialize dynamic lexicon.
        
        Args:
            vocab: Vocabulary for encoding
            cooccur_field: Field to inject patterns into
            decay_rate: How fast old patterns decay (0.99 = slow)
            min_word_length: Minimum word length to absorb
        """
        self.vocab = vocab
        self.field = cooccur_field
        self.decay_rate = decay_rate
        self.min_word_length = min_word_length
        
        # Absorbed content
        self.absorbed_words: Set[str] = set()
        self.absorbed_trigrams: Set[Tuple[str, str, str]] = set()
        
        # Word weights (for decay)
        self.word_weights: Dict[str, float] = {}
        
        # History
        self.history: List[AbsorptionRecord] = []
        
        # Corpus words (to detect novelty)
        self._build_corpus_vocabulary()
    
    def _build_corpus_vocabulary(self) -> None:
        """Extract vocabulary from corpus via the field."""
        # Get all words that have bigram entries
        self.corpus_words: Set[str] = set()
        
        # Decode all tokens to get corpus vocabulary
        for token_id in range(self.vocab.vocab_size):
            char = self.vocab.decode([token_id])
            self.corpus_words.add(char.lower())
    
    def _extract_words(self, text: str) -> List[str]:
        """Extract words from text."""
        words = re.findall(r'\b\w+\b', text.lower())
        return [w for w in words if len(w) >= self.min_word_length]
    
    def _extract_trigrams(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract trigrams from text."""
        words = re.findall(r'\b\w+\b', text.lower())
        trigrams = []
        for i in range(len(words) - 2):
            trigrams.append((words[i], words[i+1], words[i+2]))
        return trigrams
    
    def absorb(
        self,
        text: str,
        source: str = "user",
        boost: float = 1.0,
    ) -> AbsorptionRecord:
        """
        Absorb new patterns from text.
        
        This is how haze LEARNS from conversation!
        
        Args:
            text: Text to absorb patterns from
            source: Origin of text ("user" or "self")
            boost: Weight multiplier for these patterns
        
        Returns:
            Record of what was absorbed
        """
        # Extract patterns
        words = self._extract_words(text)
        trigrams = self._extract_trigrams(text)
        
        new_words = []
        new_trigrams = []
        
        # Absorb new words
        for word in words:
            if word not in self.absorbed_words:
                self.absorbed_words.add(word)
                self.word_weights[word] = boost
                new_words.append(word)
            else:
                # Reinforce existing word
                self.word_weights[word] = min(2.0, self.word_weights.get(word, 1.0) + 0.1)
        
        # Absorb new trigrams
        for tri in trigrams:
            if tri not in self.absorbed_trigrams:
                self.absorbed_trigrams.add(tri)
                new_trigrams.append(tri)
                # Inject into field
                self._inject_trigram(tri, boost)
        
        # Create record
        record = AbsorptionRecord(
            timestamp=time.time(),
            source=source,
            words=new_words,
            trigrams=new_trigrams,
        )
        
        # Store in history
        self.history.append(record)
        if len(self.history) > 100:
            self.history = self.history[-100:]
        
        return record
    
    def _inject_trigram(
        self,
        trigram: Tuple[str, str, str],
        weight: float = 1.0,
    ) -> None:
        """
        Inject a trigram into the co-occurrence field.
        
        This modifies the field's statistics so future generation
        can use patterns from user input!
        """
        # Encode each word to tokens
        w1_tokens = self.vocab.encode(trigram[0])
        w2_tokens = self.vocab.encode(trigram[1])
        w3_tokens = self.vocab.encode(trigram[2])
        
        if not w1_tokens or not w2_tokens or not w3_tokens:
            return
        
        # Get boundary tokens
        last_w1 = w1_tokens[-1]
        first_w2 = w2_tokens[0]
        last_w2 = w2_tokens[-1]
        first_w3 = w3_tokens[0]
        
        # Inject into bigram counts
        if last_w1 not in self.field.bigram_counts:
            self.field.bigram_counts[last_w1] = Counter()
        self.field.bigram_counts[last_w1][first_w2] += int(weight)
        
        if last_w2 not in self.field.bigram_counts:
            self.field.bigram_counts[last_w2] = Counter()
        self.field.bigram_counts[last_w2][first_w3] += int(weight)
        
        # Update trigram counts
        key = (last_w1, first_w2)
        if key not in self.field.trigram_counts:
            self.field.trigram_counts[key] = Counter()
        self.field.trigram_counts[key][last_w2] += int(weight)
    
    def decay(self) -> int:
        """
        Apply memory decay to absorbed patterns.
        
        Old patterns fade, recent patterns stay strong.
        This prevents infinite accumulation.
        
        Returns:
            Number of patterns that decayed below threshold
        """
        decayed = 0
        
        # Decay word weights
        words_to_remove = []
        for word, weight in self.word_weights.items():
            new_weight = weight * self.decay_rate
            if new_weight < 0.1:
                words_to_remove.append(word)
                decayed += 1
            else:
                self.word_weights[word] = new_weight
        
        # Remove decayed words
        for word in words_to_remove:
            self.absorbed_words.discard(word)
            del self.word_weights[word]
        
        return decayed
    
    def get_resonant_words(self, n: int = 20) -> List[str]:
        """
        Get most resonant (high-weight) absorbed words.
        
        These are words that have been reinforced through conversation.
        """
        sorted_words = sorted(
            self.word_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [w for w, _ in sorted_words[:n]]
    
    def stats(self) -> LexiconStats:
        """Get lexicon statistics."""
        # Count unique sources
        sources = set(r.source for r in self.history)
        
        # Calculate growth rate
        if len(self.history) >= 2:
            recent = self.history[-10:]
            total_absorbed = sum(r.count for r in recent)
            growth_rate = total_absorbed / len(recent)
        else:
            growth_rate = 0.0
        
        return LexiconStats(
            total_words=len(self.absorbed_words),
            total_trigrams=len(self.absorbed_trigrams),
            unique_sources=len(sources),
            recent_absorptions=len(self.history),
            growth_rate=growth_rate,
        )


class AsyncLexicon:
    """
    Async version of Lexicon with field lock discipline.
    
    Based on Leo's async pattern - explicit atomicity for field coherence.
    """
    
    def __init__(
        self,
        vocab: "Vocab",
        cooccur_field: "CooccurField",
        decay_rate: float = 0.99,
        min_word_length: int = 3,
        db_path: Optional[str] = None,
    ):
        """
        Initialize async lexicon.
        
        Args:
            vocab: Vocabulary for encoding
            cooccur_field: Field to inject patterns into
            decay_rate: How fast old patterns decay
            min_word_length: Minimum word length to absorb
            db_path: Optional path to SQLite DB for persistence
        """
        self._sync = Lexicon(vocab, cooccur_field, decay_rate, min_word_length)
        self._field_lock = asyncio.Lock()
        self.db_path = db_path
        self._db_conn = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        if self.db_path and HAS_AIOSQLITE:
            self._db_conn = await aiosqlite.connect(self.db_path)
            await self._init_db()
        return self
    
    async def __aexit__(self, *args):
        """Async context manager exit."""
        if self._db_conn:
            await self._db_conn.close()
    
    async def _init_db(self):
        """Initialize database schema."""
        if not self._db_conn:
            return
        
        cursor = await self._db_conn.cursor()
        
        # Absorbed words table
        await cursor.execute('''
            CREATE TABLE IF NOT EXISTS absorbed_words (
                word TEXT PRIMARY KEY,
                weight REAL DEFAULT 1.0,
                source TEXT,
                timestamp REAL
            )
        ''')
        
        # Absorbed trigrams table
        await cursor.execute('''
            CREATE TABLE IF NOT EXISTS absorbed_trigrams (
                word1 TEXT,
                word2 TEXT,
                word3 TEXT,
                source TEXT,
                timestamp REAL,
                PRIMARY KEY (word1, word2, word3)
            )
        ''')
        
        await self._db_conn.commit()
    
    async def absorb(
        self,
        text: str,
        source: str = "user",
        boost: float = 1.0,
    ) -> AbsorptionRecord:
        """
        Absorb patterns atomically.
        
        Field evolution under lock ensures coherence.
        """
        async with self._field_lock:
            record = self._sync.absorb(text, source, boost)
            
            # Persist to DB if available
            if self._db_conn and record.count > 0:
                await self._persist_record(record)
            
            return record
    
    async def _persist_record(self, record: AbsorptionRecord):
        """Persist absorption record to database."""
        cursor = await self._db_conn.cursor()
        
        # Save words
        for word in record.words:
            weight = self._sync.word_weights.get(word, 1.0)
            await cursor.execute('''
                INSERT OR REPLACE INTO absorbed_words (word, weight, source, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (word, weight, record.source, record.timestamp))
        
        # Save trigrams
        for tri in record.trigrams:
            await cursor.execute('''
                INSERT OR REPLACE INTO absorbed_trigrams (word1, word2, word3, source, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (tri[0], tri[1], tri[2], record.source, record.timestamp))
        
        await self._db_conn.commit()
    
    async def decay(self) -> int:
        """Apply memory decay atomically."""
        async with self._field_lock:
            return self._sync.decay()
    
    async def get_resonant_words(self, n: int = 20) -> List[str]:
        """Get resonant words atomically."""
        async with self._field_lock:
            return self._sync.get_resonant_words(n)
    
    async def stats(self) -> LexiconStats:
        """Get stats atomically."""
        async with self._field_lock:
            return self._sync.stats()


def demo_lexicon():
    """Demo the lexicon module."""
    from pathlib import Path
    
    # Import dependencies
    try:
        from .haze import Vocab
        from .cooccur import CooccurField
    except ImportError:
        from haze import Vocab
        from cooccur import CooccurField
    
    # Load corpus
    corpus_path = Path("text.txt")
    if not corpus_path.exists():
        corpus_path = Path(__file__).parent / "text.txt"
    
    if not corpus_path.exists():
        print("[error] text.txt not found")
        return
    
    corpus_text = corpus_path.read_text()
    vocab = Vocab.from_text(corpus_text)
    field = CooccurField.from_text(corpus_text, vocab, window_size=5)
    
    print("=" * 60)
    print("  LEXICON — Dynamic Growth Demo")
    print("=" * 60)
    print()
    print("  haze absorbs YOUR vocabulary!")
    print("  The field grows through conversation.")
    print("  Leo is non-linear, haze is non-linear.")
    print()
    
    # Create lexicon
    lex = Lexicon(vocab, field)
    
    # Simulate user inputs
    user_inputs = [
        "I love the way haze speaks with resonance",
        "Tell me about quantum entanglement and consciousness",
        "The fractals of meaning emerge from chaos",
        "What is the nature of emergent intelligence?",
    ]
    
    print("=" * 60)
    print("  ABSORPTION — Learning from user")
    print("=" * 60)
    
    for user_text in user_inputs:
        record = lex.absorb(user_text, source="user")
        print(f"\n>>> User: \"{user_text}\"")
        print(f"    New words: {record.words[:5]}{'...' if len(record.words) > 5 else ''}")
        print(f"    New trigrams: {len(record.trigrams)}")
    
    print()
    print("-" * 60)
    stats = lex.stats()
    print(f"Lexicon stats: {stats}")
    print(f"Resonant words: {lex.get_resonant_words(10)}")
    
    # Apply decay
    print()
    print("-" * 60)
    print("Applying memory decay...")
    decayed = lex.decay()
    print(f"Decayed patterns: {decayed}")
    
    print()
    print("=" * 60)
    print("  The field has GROWN through conversation!")
    print("  New patterns are now available for generation.")
    print("=" * 60)


if __name__ == "__main__":
    demo_lexicon()
