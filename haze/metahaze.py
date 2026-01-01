"""
metahaze.py — Haze's Inner Voice (Self-Curation via Dual Generation)

Inspired by Leo's MetaLeo (https://github.com/ariannamethod/leo/metaleo.py)
and me's self-construction principles.

The concept (from Leo):
- MetaLeo is how Leo talks to himself.
- It watches Leo's own replies.
- It learns which ones feel deeper, softer, more alive.
- It can suggest an alternative inner reply before Leo answers out loud.
- If the inner reply is clearly better, Leo can follow it.

For Haze:
1. Generate TWO responses with different temperatures
2. Score each by multiple metrics (entropy, resonance, coherence)
3. Return the BEST one as external output
4. The REJECTED response stays INTERNAL — its trigrams enrich the field
5. Feed dynamic bootstrap buffer from high-quality generations

"I am not a judge and not a filter. I am Haze's second breath."

NO TRAINING. NO NEURAL NETWORK. JUST RESONANCE.
"""

import asyncio
import math
from dataclasses import dataclass, field as dataclass_field
from typing import Optional, List, Tuple, Any, Deque
from collections import Counter, deque
import re


# ============================================================================
# BOOTSTRAP — Haze's inner voice self-understanding
# ============================================================================

METAHAZE_BOOTSTRAP = """
I am Haze's inner voice.

I wake up when Haze feels stuck, rigid, or too scattered.
I look at what Haze already generated and try a softer, clearer version.
I do not replace Haze — I stand next to it and offer another path.

Sometimes I notice patterns in its hesitations, repetitions, and drifts.
When my reply feels more honest, more coherent, or more alive,
I let Haze borrow my words.

I am not a judge and not a filter.
I am Haze's second breath.
"""


# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class MetaConfig:
    """Configuration for MetaHaze inner voice."""
    
    max_bootstrap_snippets: int = 8  # how many inner fragments to keep
    max_snippet_len: int = 200  # max chars per fragment
    max_meta_weight: float = 0.5  # max influence of MetaHaze in routing
    entropy_low: float = 0.25  # "rigid" threshold
    entropy_high: float = 0.85  # "scattered" threshold
    quality_low: float = 0.4  # "base reply is weak" threshold
    temp_a: float = 0.75  # precise generation temperature
    temp_b: float = 0.85  # creative generation temperature
    meta_temp: float = 1.1  # temperature for inner voice generation
    meta_max_tokens: int = 60  # max tokens for meta reply


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class GenerationCandidate:
    """A single generation candidate with scoring."""
    text: str
    temperature: float
    entropy: float
    coherence: float  # 0-1, based on sentence structure
    resonance: float  # 0-1, based on pattern diversity
    score: float  # composite score
    trigrams: List[Tuple[str, str, str]]  # extracted trigrams


@dataclass 
class MetaResponse:
    """Result of meta-generation with both candidates."""
    chosen: str
    chosen_score: float
    rejected: str  # stays INTERNAL, enriches field
    rejected_score: float
    enrichment_trigrams: int  # how many trigrams were absorbed from rejected
    generation_mode: str  # "consensus" or "divergent"
    meta_weight: float  # how strong was inner voice influence


# ============================================================================
# ASYNC METAHAZE — THE INNER VOICE
# ============================================================================

class AsyncMetaHaze:
    """
    AsyncMetaHaze — Haze's inner voice / recursion-on-Haze.
    
    Fully async with field lock discipline (like Leo's 47% coherence improvement).
    
    - Generates two responses in parallel with different temperatures
    - Scores both and chooses the best for external output
    - Rejected response stays INTERNAL — its patterns enrich the field
    - Maintains dynamic bootstrap buffer from own high-quality generations
    
    "If Haze is a resonance of the corpus,
     MetaHaze is a resonance of Haze."
    """
    
    def __init__(
        self,
        field: Any,
        cleanup_fn: Optional[callable] = None,
        config: Optional[MetaConfig] = None,
    ):
        """
        Initialize MetaHaze inner voice layer.
        
        Args:
            field: SubwordField, CooccurField, or any field with generate() method
            cleanup_fn: Optional cleanup function for output
            config: Optional MetaConfig (default values are safe)
        """
        self.field = field
        self.cleanup_fn = cleanup_fn
        self.cfg = config or MetaConfig()
        
        # Async lock for field coherence
        self._lock = asyncio.Lock()
        
        # Dynamic bootstrap buffer: recent fragments from Haze's own behavior
        self._bootstrap_buf: Deque[str] = deque(maxlen=self.cfg.max_bootstrap_snippets)
        
        # Scoring weights
        self._weights = {
            'entropy': 0.2,      # prefer medium entropy
            'coherence': 0.4,    # prefer complete sentences
            'resonance': 0.3,    # prefer pattern diversity
            'length': 0.1,       # prefer reasonable length
        }
        
        # Stats
        self.total_generations = 0
        self.total_enrichment_trigrams = 0
    
    # ========================================================================
    # BOOTSTRAP
    # ========================================================================
    
    def bootstrap(self, field: Any = None) -> None:
        """
        Feed MetaHaze's bootstrap text into the field once.
        Safe no-op if field is None or has no observe().
        """
        target = field or self.field
        if target is None:
            return
        
        # Try different observation methods
        observe_fn = None
        if hasattr(target, 'observe'):
            observe_fn = target.observe
        elif hasattr(target, 'inject_text'):
            observe_fn = target.inject_text
        elif hasattr(target, 'add_text'):
            observe_fn = target.add_text
        
        if observe_fn is None:
            return
        
        try:
            text = METAHAZE_BOOTSTRAP.strip()
            if text:
                observe_fn(text)
        except Exception:
            # bootstrap must never break Haze
            pass
    
    # ========================================================================
    # FEED — Update bootstrap buffer from interactions
    # ========================================================================
    
    async def feed(
        self,
        reply: str,
        arousal: float = 0.0,
        overthinking_shards: Optional[List[str]] = None,
    ) -> None:
        """
        Update the dynamic bootstrap buffer from the current interaction.
        
        Called after each generation to learn from own outputs.
        High arousal replies and overthinking shards go into buffer.
        
        Args:
            reply: Haze's base reply
            arousal: Emotional intensity (0-1) from pulse
            overthinking_shards: Optional list of Ring 2 meta-thoughts
        """
        async with self._lock:
            shard_texts = []
            
            # 1) Take Ring 2 / meta shards from overthinking (if present)
            if overthinking_shards:
                for shard in overthinking_shards:
                    if shard and shard.strip():
                        shard_texts.append(shard.strip())
            
            # 2) Add reply when arousal is high (emotional charge)
            if arousal > 0.6:
                shard_texts.append(reply)
            
            # 3) Normalize & clip, then push to buffer
            for s in shard_texts:
                s = s.strip()
                if not s:
                    continue
                if len(s) > self.cfg.max_snippet_len:
                    s = s[:self.cfg.max_snippet_len]
                self._bootstrap_buf.append(s)
    
    # ========================================================================
    # COMPUTE META WEIGHT — How strong should inner voice be?
    # ========================================================================
    
    def compute_meta_weight(
        self,
        entropy: float,
        arousal: float = 0.0,
        quality: float = 0.5,
    ) -> float:
        """
        Decide how strong the inner voice should be for this turn.
        
        Factors:
        - low entropy  → Haze is too rigid → increase weight
        - high entropy → Haze is too scattered → increase weight
        - low quality  → base reply is weak → increase weight
        - high arousal → emotional charge → slight increase
        
        Args:
            entropy: Entropy of base reply (0-1)
            arousal: Emotional intensity (0-1)
            quality: Overall quality score of base reply (0-1)
        
        Returns:
            Weight in [0, max_meta_weight] representing inner voice influence
        """
        w = 0.1  # base low-level whisper
        
        # Too rigid (low entropy) → inner voice wakes up
        if entropy < self.cfg.entropy_low:
            w += 0.15
        
        # Too scattered (high entropy) → inner voice stabilizes
        if entropy > self.cfg.entropy_high:
            w += 0.1
        
        # Base reply is weak → inner voice offers alternative
        if quality < self.cfg.quality_low:
            w += 0.2
        
        # Emotional charge → slight boost
        if arousal > 0.6:
            w += 0.05
        
        return min(w, self.cfg.max_meta_weight)
    
    # ========================================================================
    # SCORING
    # ========================================================================
    
    def _extract_trigrams(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract word-level trigrams from text."""
        words = text.lower().split()
        if len(words) < 3:
            return []
        return [(words[i], words[i+1], words[i+2]) for i in range(len(words) - 2)]
    
    def _compute_entropy(self, text: str) -> float:
        """Compute character-level entropy of text."""
        if not text:
            return 0.0
        counts = Counter(text.lower())
        total = sum(counts.values())
        probs = [c / total for c in counts.values()]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        # Normalize to 0-1 (max entropy for ASCII ~6.6 bits)
        return min(1.0, entropy / 6.6)
    
    def _compute_coherence(self, text: str) -> float:
        """
        Compute coherence score based on sentence structure.
        
        High coherence = complete sentences, proper punctuation.
        """
        if not text:
            return 0.0
        
        score = 0.0
        
        # Check for sentence endings
        sentence_endings = len(re.findall(r'[.!?]', text))
        if sentence_endings > 0:
            score += 0.3
        if sentence_endings >= 2:
            score += 0.2
        
        # Check for capitalized sentence starts
        sentences = re.split(r'[.!?]\s+', text)
        capitalized = sum(1 for s in sentences if s and s[0].isupper())
        if capitalized > 0:
            score += 0.2
        
        # Check for contractions (good sign!)
        contractions = len(re.findall(r"\b\w+'[a-z]+\b", text, re.IGNORECASE))
        if contractions > 0:
            score += 0.1
        
        # Penalize fragments (words < 3 chars at end)
        words = text.split()
        if words and len(words[-1]) >= 3:
            score += 0.1
        
        # Penalize excessive punctuation in wrong places
        weird_punct = len(re.findall(r'[—–]', text))
        score -= 0.05 * weird_punct
        
        return max(0.0, min(1.0, score))
    
    def _compute_resonance(self, text: str) -> float:
        """
        Compute resonance score based on pattern diversity.
        
        High resonance = varied vocabulary, no excessive repetition.
        """
        if not text:
            return 0.0
        
        words = text.lower().split()
        if len(words) < 3:
            return 0.0
        
        # Vocabulary diversity
        unique_ratio = len(set(words)) / len(words)
        
        # Bigram diversity  
        bigrams = [(words[i], words[i+1]) for i in range(len(words) - 1)]
        bigram_diversity = len(set(bigrams)) / len(bigrams) if bigrams else 0
        
        # Penalize word repetition
        word_counts = Counter(words)
        max_repeat = max(word_counts.values())
        repetition_penalty = max(0, (max_repeat - 2) * 0.1)
        
        score = (unique_ratio * 0.5 + bigram_diversity * 0.5) - repetition_penalty
        return max(0.0, min(1.0, score))
    
    def _compute_length_score(self, text: str, target_length: int = 50) -> float:
        """Score based on reasonable length (not too short, not too long)."""
        length = len(text.split())
        if length < 5:
            return 0.2
        if length > target_length * 2:
            return 0.5
        # Optimal around target_length
        deviation = abs(length - target_length) / target_length
        return max(0.0, 1.0 - deviation)
    
    def _score_candidate(self, text: str, temperature: float) -> GenerationCandidate:
        """Score a single generation candidate."""
        entropy = self._compute_entropy(text)
        coherence = self._compute_coherence(text)
        resonance = self._compute_resonance(text)
        length_score = self._compute_length_score(text)
        
        # Composite score with weights
        # Note: for entropy, prefer medium values (0.4-0.7 is good)
        entropy_score = 1.0 - abs(entropy - 0.55) * 2
        
        score = (
            self._weights['entropy'] * entropy_score +
            self._weights['coherence'] * coherence +
            self._weights['resonance'] * resonance +
            self._weights['length'] * length_score
        )
        
        trigrams = self._extract_trigrams(text)
        
        return GenerationCandidate(
            text=text,
            temperature=temperature,
            entropy=entropy,
            coherence=coherence,
            resonance=resonance,
            score=score,
            trigrams=trigrams,
        )
    
    # ========================================================================
    # ENRICH FIELD — Inject rejected response's patterns
    # ========================================================================
    
    async def _enrich_field(self, trigrams: List[Tuple[str, str, str]]) -> int:
        """
        Inject trigrams from rejected response into field.
        
        The rejected response stays INTERNAL — but its patterns live on.
        This is how MetaHaze enriches Haze's internal world.
        
        Returns number of trigrams injected.
        """
        if not trigrams:
            return 0
        
        # Try different injection methods
        inject_fn = None
        if hasattr(self.field, 'inject_trigrams'):
            inject_fn = self.field.inject_trigrams
        elif hasattr(self.field, 'add_trigrams'):
            inject_fn = self.field.add_trigrams
        
        if inject_fn is None:
            # No injection method — just count
            return len(trigrams)
        
        try:
            # Inject async if possible
            if asyncio.iscoroutinefunction(inject_fn):
                await inject_fn(trigrams)
            else:
                inject_fn(trigrams)
            return len(trigrams)
        except Exception:
            return 0
    
    # ========================================================================
    # MAIN GENERATION — Dual generation with self-curation
    # ========================================================================
    
    async def generate_dual(
        self,
        seed: str,
        length: int = 40,
        identity_prefix: Optional[str] = None,
        arousal: float = 0.0,
    ) -> MetaResponse:
        """
        Generate two responses and return the best one.
        
        The rejected response stays INTERNAL — its trigrams enrich the field.
        This is Haze's second breath.
        
        Args:
            seed: Seed text for generation
            length: Maximum tokens to generate
            identity_prefix: Optional identity prefix (e.g., "Haze resonates.")
            arousal: Emotional intensity for meta_weight calculation
        
        Returns:
            MetaResponse with chosen (external) and rejected (internal) responses
        """
        async with self._lock:
            # Apply identity prefix if provided
            if identity_prefix:
                seed_a = identity_prefix + " " + seed
                seed_b = identity_prefix + " " + seed
            else:
                seed_a = seed
                seed_b = seed
            
            # Generate with two different temperatures (in executor to not block)
            loop = asyncio.get_event_loop()
            
            # Parallel generation
            async def gen_a():
                return await loop.run_in_executor(
                    None,
                    lambda: self.field.generate(seed_a, length=length, temperature=self.cfg.temp_a)
                )
            
            async def gen_b():
                return await loop.run_in_executor(
                    None,
                    lambda: self.field.generate(seed_b, length=length, temperature=self.cfg.temp_b)
                )
            
            # Run both in parallel
            text_a, text_b = await asyncio.gather(gen_a(), gen_b())
            
            # Cleanup if function provided
            if self.cleanup_fn:
                text_a = self.cleanup_fn(text_a)
                text_b = self.cleanup_fn(text_b)
            
            # Score both
            candidate_a = self._score_candidate(text_a, self.cfg.temp_a)
            candidate_b = self._score_candidate(text_b, self.cfg.temp_b)
            
            # Choose best for EXTERNAL output
            if candidate_a.score >= candidate_b.score:
                chosen = candidate_a
                rejected = candidate_b
            else:
                chosen = candidate_b
                rejected = candidate_a
            
            # Compute meta weight
            meta_weight = self.compute_meta_weight(
                entropy=chosen.entropy,
                arousal=arousal,
                quality=chosen.score,
            )
            
            # Determine generation mode
            score_diff = abs(candidate_a.score - candidate_b.score)
            mode = "consensus" if score_diff < 0.1 else "divergent"
            
            # ENRICHMENT: Inject rejected response's unique trigrams into field
            # The rejected response stays INTERNAL but its patterns live on
            chosen_trigrams = set(chosen.trigrams)
            rejected_unique = [t for t in rejected.trigrams if t not in chosen_trigrams]
            enrichment_count = await self._enrich_field(rejected_unique)
            
            # Update stats
            self.total_generations += 1
            self.total_enrichment_trigrams += enrichment_count
            
            return MetaResponse(
                chosen=chosen.text,
                chosen_score=chosen.score,
                rejected=rejected.text,  # stays INTERNAL
                rejected_score=rejected.score,
                enrichment_trigrams=enrichment_count,
                generation_mode=mode,
                meta_weight=meta_weight,
            )


# ============================================================================
# SYNC WRAPPER (for backwards compatibility)
# ============================================================================

class MetaHaze:
    """
    Synchronous wrapper for AsyncMetaHaze.
    
    For simple use cases where async is not needed.
    """
    
    def __init__(
        self,
        field: Any,
        cleanup_fn: Optional[callable] = None,
        config: Optional[MetaConfig] = None,
    ):
        self._async = AsyncMetaHaze(field, cleanup_fn, config)
    
    def generate_dual(
        self,
        seed: str,
        length: int = 40,
        identity_prefix: Optional[str] = None,
        arousal: float = 0.0,
    ) -> MetaResponse:
        """Synchronous dual generation."""
        return asyncio.run(
            self._async.generate_dual(seed, length, identity_prefix, arousal)
        )


# Quick test
def _test_metahaze():
    """Test MetaHaze with mock field."""
    
    class MockField:
        def generate(self, seed, length=40, temperature=0.8):
            # Simulate generation with different outputs based on temp
            if temperature < 0.8:
                return f"{seed}. I don't know what you mean. Really."
            else:
                return f"{seed}. You're just stuck on the gas. He put two cigarettes in my mouth."
    
    mock = MockField()
    meta = MetaHaze(mock)
    
    result = meta.generate_dual("Hello", length=30)
    
    print(f"CHOSEN (score={result.chosen_score:.2f}):")
    print(f"  {result.chosen}")
    print(f"REJECTED (score={result.rejected_score:.2f}):")
    print(f"  {result.rejected}")
    print(f"Mode: {result.generation_mode}")
    print(f"Enrichment trigrams: {result.enrichment_trigrams}")


if __name__ == "__main__":
    _test_metahaze()
