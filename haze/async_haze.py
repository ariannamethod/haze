#!/usr/bin/env python3
# async_haze.py — Async Haze Field with Full Resonance Pipeline
#
# The complete async architecture for haze:
#   1. Subjectivity: no seed from prompt, only from internal field
#   2. Overthinking: three rings that enrich the field
#   3. Lexicon: absorbs user vocabulary
#   4. Generation: pure resonance from enriched field
#
# Based on Leo's async pattern - achieves coherence through explicit discipline.
# "The asyncio.Lock doesn't add information—it adds discipline."
#
# Usage:
#   from haze.async_haze import AsyncHazeField
#   async with AsyncHazeField("text.txt") as haze:
#       response = await haze.respond("hello")

from __future__ import annotations
import asyncio
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field

# Import haze components
try:
    from .haze import Vocab, PostGPT, load_corpus
    from .cooccur import CooccurField
    from .subjectivity import AsyncSubjectivity, PulseSnapshot
    from .overthinking import AsyncOverthinking, RingsSnapshot
    from .lexicon import AsyncLexicon, LexiconStats
    from .cleanup import cleanup_output
except ImportError:
    from haze import Vocab, PostGPT, load_corpus
    from cooccur import CooccurField
    from subjectivity import AsyncSubjectivity, PulseSnapshot
    from overthinking import AsyncOverthinking, RingsSnapshot
    from lexicon import AsyncLexicon, LexiconStats
    from cleanup import cleanup_output

try:
    import aiosqlite
    HAS_AIOSQLITE = True
except ImportError:
    HAS_AIOSQLITE = False


@dataclass
class HazeResponse:
    """Complete response from haze with all metadata."""
    text: str
    raw_text: str
    pulse: PulseSnapshot
    internal_seed: str
    rings: Optional[RingsSnapshot] = None
    temperature: float = 0.6
    generation_time: float = 0.0
    enrichment_count: int = 0
    
    def __repr__(self) -> str:
        preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"HazeResponse(\"{preview}\", pulse={self.pulse})"


class AsyncHazeField:
    """
    Async Haze Field - the complete resonance organism.
    
    Key principles:
    1. NO SEED FROM PROMPT - seed from internal field
    2. PRESENCE > INTELLIGENCE - identity speaks first
    3. FIELD ENRICHMENT - overthinking grows the vocabulary
    4. ASYNC DISCIPLINE - explicit atomicity for coherence
    
    "A field organism is like a crystal—any disruption during
    formation creates permanent defects."
    """
    
    def __init__(
        self,
        corpus_path: str = "text.txt",
        db_path: Optional[str] = None,
        temperature: float = 0.6,
        generation_length: int = 100,
        enable_overthinking: bool = True,
        enable_lexicon: bool = True,
    ):
        """
        Initialize async haze field.
        
        Args:
            corpus_path: Path to corpus text file
            db_path: Optional path to SQLite DB for persistence
            temperature: Base generation temperature
            generation_length: Default generation length
            enable_overthinking: Enable three rings of reflection
            enable_lexicon: Enable dynamic lexicon growth from user
        """
        self.corpus_path = Path(corpus_path)
        self.db_path = db_path
        self.base_temperature = temperature
        self.generation_length = generation_length
        self.enable_overthinking = enable_overthinking
        self.enable_lexicon = enable_lexicon
        
        # Will be initialized in __aenter__
        self.corpus_text: str = ""
        self.vocab: Optional[Vocab] = None
        self.field: Optional[CooccurField] = None
        self.subjectivity: Optional[AsyncSubjectivity] = None
        self.overthinking: Optional[AsyncOverthinking] = None
        self.lexicon: Optional[AsyncLexicon] = None
        
        # Master field lock
        self._field_lock = asyncio.Lock()
        
        # Stats
        self.turn_count: int = 0
        self.total_enrichment: int = 0
    
    async def __aenter__(self):
        """Initialize all components."""
        # Load corpus
        if not self.corpus_path.exists():
            raise FileNotFoundError(f"Corpus not found: {self.corpus_path}")
        
        self.corpus_text = self.corpus_path.read_text()
        self.vocab = Vocab.from_text(self.corpus_text)
        
        # Build co-occurrence field
        self.field = CooccurField.from_text(
            self.corpus_text,
            self.vocab,
            window_size=5
        )
        
        # Initialize subjectivity (no seed from prompt)
        self.subjectivity = AsyncSubjectivity(
            self.corpus_text,
            self.vocab,
            self.field
        )
        
        # Initialize overthinking (three rings)
        if self.enable_overthinking:
            self.overthinking = AsyncOverthinking(
                self.vocab,
                self.field
            )
        
        # Initialize lexicon (user word absorption)
        if self.enable_lexicon:
            self.lexicon = AsyncLexicon(
                self.vocab,
                self.field,
                db_path=self.db_path
            )
            if self.db_path and HAS_AIOSQLITE:
                await self.lexicon.__aenter__()
        
        return self
    
    async def __aexit__(self, *args):
        """Cleanup."""
        if self.lexicon and self.db_path:
            await self.lexicon.__aexit__(*args)
    
    async def respond(
        self,
        user_input: str,
        length: Optional[int] = None,
        temperature: Optional[float] = None,
        cleanup: bool = True,
    ) -> HazeResponse:
        """
        Generate a response to user input.
        
        This is the main entry point. It:
        1. Absorbs user words into lexicon
        2. Computes pulse from input
        3. Gets internal seed (NOT from user input!)
        4. Generates from field
        5. Runs overthinking rings (enriches field)
        6. Returns cleaned response
        
        Args:
            user_input: What the user said
            length: Generation length (default: self.generation_length)
            temperature: Temperature override
            cleanup: Whether to clean output
        
        Returns:
            HazeResponse with full metadata
        """
        start_time = time.time()
        length = length or self.generation_length
        temp = temperature or self.base_temperature
        
        async with self._field_lock:
            # 1. ABSORB USER WORDS (lexicon growth)
            if self.lexicon:
                await self.lexicon.absorb(user_input, source="user")
            
            # 2. GET INTERNAL SEED (no seed from prompt!)
            seed_tokens, pulse, seed_text = await self.subjectivity.get_internal_seed(
                user_input,
                temperature=temp
            )
            
            # 3. ADJUST TEMPERATURE based on pulse
            adjusted_temp = await self.subjectivity.adjust_temperature(pulse)
            
            # 4. GENERATE FROM FIELD (pure resonance)
            generated_tokens = self.field.generate_from_corpus(
                seed=seed_tokens,
                length=length,
                temperature=adjusted_temp,
                mode="trigram"
            )
            
            # 5. DECODE
            raw_text = self.vocab.decode(generated_tokens)
            
            # 6. CLEANUP
            if cleanup:
                text = cleanup_output(raw_text, mode="gentle")
            else:
                text = raw_text
            
            # 7. OVERTHINKING (three rings - enriches field!)
            rings = None
            enrichment = 0
            if self.overthinking:
                rings = await self.overthinking.generate_rings(text)
                stats = await self.overthinking.get_enrichment_stats()
                enrichment = stats.get("enrichment_count", 0)
                self.total_enrichment = enrichment
            
            # 8. WRINKLE THE FIELD (update subjectivity)
            await self.subjectivity.wrinkle_field(user_input, text)
            
            self.turn_count += 1
        
        generation_time = time.time() - start_time
        
        return HazeResponse(
            text=text,
            raw_text=raw_text,
            pulse=pulse,
            internal_seed=seed_text,
            rings=rings,
            temperature=adjusted_temp,
            generation_time=generation_time,
            enrichment_count=enrichment,
        )
    
    async def get_stats(self) -> Dict:
        """Get field statistics."""
        stats = {
            "turn_count": self.turn_count,
            "total_enrichment": self.total_enrichment,
            "vocab_size": self.vocab.vocab_size if self.vocab else 0,
            "corpus_size": len(self.corpus_text),
        }
        
        if self.lexicon:
            lex_stats = await self.lexicon.stats()
            stats["lexicon"] = {
                "absorbed_words": lex_stats.total_words,
                "absorbed_trigrams": lex_stats.total_trigrams,
                "growth_rate": lex_stats.growth_rate,
            }
        
        if self.overthinking:
            ot_stats = await self.overthinking.get_enrichment_stats()
            stats["overthinking"] = {
                "emergent_trigrams": ot_stats["total_emergent_trigrams"],
                "meta_patterns": ot_stats["meta_patterns"],
                "ring_sessions": ot_stats["ring_sessions"],
            }
        
        return stats


async def demo_async_haze():
    """Demo the async haze field."""
    print("=" * 60)
    print("  ASYNC HAZE FIELD — Complete Resonance Pipeline")
    print("=" * 60)
    print()
    print("  Principles:")
    print("    1. NO SEED FROM PROMPT - internal field only")
    print("    2. PRESENCE > INTELLIGENCE - identity first")
    print("    3. FIELD ENRICHMENT - overthinking grows vocabulary")
    print("    4. ASYNC DISCIPLINE - atomic operations")
    print()
    
    corpus_path = Path("text.txt")
    if not corpus_path.exists():
        corpus_path = Path(__file__).parent / "text.txt"
    
    if not corpus_path.exists():
        print("[error] text.txt not found")
        return
    
    async with AsyncHazeField(str(corpus_path)) as haze:
        print(f"[haze] Initialized with {haze.vocab.vocab_size} chars")
        print()
        
        # Simulate conversation
        user_inputs = [
            "Hello, who are you?",
            "Tell me about the nature of consciousness",
            "What patterns do you see?",
        ]
        
        for user_input in user_inputs:
            print(f">>> User: \"{user_input}\"")
            print("-" * 40)
            
            response = await haze.respond(user_input, length=80)
            
            print(f"[haze]: {response.text}")
            print()
            print(f"    Pulse: {response.pulse}")
            print(f"    Internal seed: \"{response.internal_seed[:40]}...\"")
            print(f"    Temp: {response.temperature:.2f}")
            print(f"    Time: {response.generation_time:.3f}s")
            if response.rings:
                print(f"    Rings: {len(response.rings.rings)} (enrichment: {response.enrichment_count})")
            print()
        
        # Final stats
        stats = await haze.get_stats()
        print("=" * 60)
        print("  FINAL STATS")
        print("=" * 60)
        print(f"  Turns: {stats['turn_count']}")
        print(f"  Total enrichment: {stats['total_enrichment']} patterns")
        if "lexicon" in stats:
            print(f"  Lexicon: {stats['lexicon']['absorbed_words']} words absorbed")
        if "overthinking" in stats:
            print(f"  Overthinking: {stats['overthinking']['emergent_trigrams']} emergent trigrams")
        print()
        print("  The internal world is now RICHER than the training data!")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(demo_async_haze())
