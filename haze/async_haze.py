#!/usr/bin/env python3
# async_haze.py — Async Haze Field with Full Resonance Pipeline
#
# The complete async architecture for haze:
#   1. Subjectivity: no seed from prompt, only from internal field
#   2. Overthinking: three rings that enrich the field
#   3. Lexicon: absorbs user vocabulary
#   4. Generation: pure resonance from enriched field
#   5. MathBrain: field perception and temperature tuning
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
    from .experts import route_to_mixture, pulse_to_signals, describe_mixture, ExpertMixture
    from .trauma import AsyncTrauma, TraumaState, TraumaInfluence, get_identity_prefix
    from .subword_field import SubwordField, AsyncSubwordField
    from .mathbrain import AsyncMathBrain, FieldPerception
    HAS_SUBWORD = True
    HAS_MATHBRAIN = True
except ImportError:
    try:
        from haze import Vocab, PostGPT, load_corpus
        from cooccur import CooccurField
        from subjectivity import AsyncSubjectivity, PulseSnapshot
        from overthinking import AsyncOverthinking, RingsSnapshot
        from lexicon import AsyncLexicon, LexiconStats
        from cleanup import cleanup_output
        from experts import route_to_mixture, pulse_to_signals, describe_mixture, ExpertMixture
        from trauma import AsyncTrauma, TraumaState, TraumaInfluence, get_identity_prefix
        from subword_field import SubwordField, AsyncSubwordField
        from mathbrain import AsyncMathBrain, FieldPerception
        HAS_SUBWORD = True
        HAS_MATHBRAIN = True
    except ImportError:
        HAS_SUBWORD = False
        HAS_MATHBRAIN = False

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
    expert_mixture: Optional[ExpertMixture] = None
    trauma: Optional[TraumaState] = None
    trauma_influence: Optional[TraumaInfluence] = None
    brain_perception: Optional["FieldPerception"] = None  # MathBrain perception
    
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
    5. TRAUMA - resonant words return to identity
    6. SUBWORD GENERATION - BPE tokenizer for coherent output
    
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
        enable_trauma: bool = True,
        use_subword: bool = True,  # NEW: Use BPE subword tokenization
        subword_vocab_size: int = 500,
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
            enable_trauma: Enable resonant word trauma (identity return)
            use_subword: Use BPE subword tokenization (MUCH better output!)
            subword_vocab_size: Vocabulary size for BPE (default 500)
        """
        self.corpus_path = Path(corpus_path)
        self.db_path = db_path
        self.base_temperature = temperature
        self.generation_length = generation_length
        self.enable_overthinking = enable_overthinking
        self.enable_lexicon = enable_lexicon
        self.enable_trauma = enable_trauma
        self.use_subword = use_subword and HAS_SUBWORD
        self.subword_vocab_size = subword_vocab_size
        
        # Will be initialized in __aenter__
        self.corpus_text: str = ""
        self.vocab: Optional[Vocab] = None
        self.field: Optional[CooccurField] = None
        self.subword_field: Optional[SubwordField] = None  # NEW
        self.subjectivity: Optional[AsyncSubjectivity] = None
        self.overthinking: Optional[AsyncOverthinking] = None
        self.lexicon: Optional[AsyncLexicon] = None
        self.trauma: Optional[AsyncTrauma] = None
        
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
        
        # Build subword field if enabled (BPE = coherent output!)
        if self.use_subword and HAS_SUBWORD:
            try:
                self.subword_field = SubwordField.from_corpus(
                    str(self.corpus_path),
                    vocab_size=self.subword_vocab_size,
                )
            except Exception as e:
                print(f"[warning] SubwordField failed: {e}, using char-level")
                self.subword_field = None
                self.use_subword = False
        
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
        
        # Initialize trauma (resonant words return to identity)
        if self.enable_trauma:
            self.trauma = AsyncTrauma()
        
        return self
    
    async def __aexit__(self, *args):
        """Cleanup."""
        if self.lexicon and self.db_path:
            await self.lexicon.__aexit__(*args)
        if self.trauma:
            await self.trauma.close()
    
    async def respond(
        self,
        user_input: str,
        length: Optional[int] = None,
        temperature: Optional[float] = None,
        cleanup: bool = True,
        use_experts: bool = True,
    ) -> HazeResponse:
        """
        Generate a response to user input.
        
        This is the main entry point. It:
        1. Absorbs user words into lexicon
        2. Computes pulse from input
        3. Routes to resonant experts (MOE-style temperature blending)
        4. Gets internal seed (NOT from user input!)
        5. Generates from field
        6. Runs overthinking rings (enriches field)
        7. Returns cleaned response
        
        Args:
            user_input: What the user said
            length: Generation length (default: self.generation_length)
            temperature: Temperature override (disables expert routing)
            cleanup: Whether to clean output
            use_experts: Use resonant expert routing (MOE-style)
        
        Returns:
            HazeResponse with full metadata
        """
        start_time = time.time()
        length = length or self.generation_length
        
        async with self._field_lock:
            # 1. ABSORB USER WORDS (lexicon growth)
            if self.lexicon:
                await self.lexicon.absorb(user_input, source="user")
            
            # 2. GET INTERNAL SEED (no seed from prompt!)
            seed_tokens, pulse, seed_text = await self.subjectivity.get_internal_seed(
                user_input,
                temperature=self.base_temperature
            )
            
            # 3. ROUTE TO EXPERTS (MOE-style temperature blending)
            expert_mixture = None
            if use_experts and temperature is None:
                # Convert pulse to field signals
                signals = pulse_to_signals(
                    novelty=pulse.novelty,
                    arousal=pulse.arousal,
                    entropy=pulse.entropy,
                )
                expert_mixture = route_to_mixture(signals)
                adjusted_temp = expert_mixture.temperature
            elif temperature is not None:
                adjusted_temp = temperature
            else:
                # Fallback to subjectivity's temperature adjustment
                adjusted_temp = await self.subjectivity.adjust_temperature(pulse)
            
            # 4. GENERATE FROM FIELD (pure resonance)
            if self.use_subword and self.subword_field is not None:
                # USE SUBWORD FIELD — coherent output with BPE!
                # seed_text is already the internal seed from field (not from prompt)
                raw_text = self.subword_field.generate(
                    seed_text=seed_text,
                    length=length,
                    temperature=adjusted_temp,
                    mode="trigram"
                )
            else:
                # Fallback to character-level field
                generated_tokens = self.field.generate_from_corpus(
                    seed=seed_tokens,
                    length=length,
                    temperature=adjusted_temp,
                    mode="trigram"
                )
                raw_text = self.vocab.decode(generated_tokens)
            
            # 5. CLEANUP
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
            
            # 8. TRAUMA DETECTION (resonant words return to identity)
            trauma_state = None
            trauma_influence = None
            if self.trauma:
                trauma_state = await self.trauma.process(user_input, text, pulse)
                trauma_influence = await self.trauma.get_influence()
                
                # Apply trauma influence to text
                if trauma_influence.should_prefix:
                    identity_prefix = get_identity_prefix()
                    if not text.startswith("Haze"):
                        text = f"{identity_prefix} {text}"
            
            # 9. WRINKLE THE FIELD (update subjectivity)
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
            expert_mixture=expert_mixture,
            trauma=trauma_state,
            trauma_influence=trauma_influence,
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
            seed_preview = response.internal_seed[:40] + "..." if len(response.internal_seed) > 40 else response.internal_seed
            print(f"    Internal seed: \"{seed_preview}\"")
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
