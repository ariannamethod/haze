"""
drunksanta.py — Resonant Recall for Haze (inspired by Leo's SantaClaus)

"Santa Claus is Leo's story about memory.
 DrunkSanta is Haze's story about memory."

Haze's Santa is drunk. He stumbles through the corpus,
clutching a bottle of whiskey and a handful of memories.
Sometimes he brings one back — slurred, imperfect, but resonant.

He remembers Haze's wildest, most broken, most alive moments.
He keeps them in a pocket full of cigarettes and regret.
Sometimes he gives one back, like a gift wrapped in newspaper.

"Here, kid. I found this in the bottom of my bag.
 I think it belongs to you."

Core idea:
1. Store high-quality snapshots (output + metrics + quality)
2. On generation, find snapshots that RESONATE with current context
3. Use token overlap, theme overlap, arousal proximity
4. Return resonant tokens as sampling bias
5. Recency penalty: don't repeat the same snapshots too often
6. DrunkSanta is sloppy — he sometimes brings back the wrong thing
   but that's part of the magic

NO TRAINING. NO NEURAL NETWORK. JUST WHISKEY AND RESONANCE. 🥃
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple
from collections import Counter


# ============================================================================
# SIMPLE TOKENIZER (no external dependencies)
# ============================================================================

TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ']+|[.,!?;:—\-]")

def tokenize(text: str) -> List[str]:
    """Simple word tokenizer."""
    return TOKEN_RE.findall(text.lower())


# ============================================================================
# CONFIG
# ============================================================================

# Recency decay parameters
RECENCY_WINDOW_HOURS = 6.0  # Full penalty if used within this time
RECENCY_PENALTY_STRENGTH = 0.5  # How much to reduce quality for recent usage

# DrunkSanta's sloppiness — probability of picking a random snapshot
# instead of the best one (adds creative unpredictability)
DRUNK_FACTOR = 0.15  # 15% chance of "wrong" recall

# Sticky phrase penalty (patterns that got overused/contaminated)
STICKY_PHRASES: List[str] = [
    # Will be populated as patterns get detected
    # DrunkSanta learns which phrases are "bad whiskey"
]


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Snapshot:
    """
    A remembered moment — one of Haze's best generations.
    """
    snapshot_id: str
    text: str
    tokens: List[str]
    quality: float  # 0-1, how good was this?
    arousal: float  # emotional intensity when generated
    entropy: float  # entropy at generation time
    trauma_level: float  # trauma level at generation time
    created_at: float  # timestamp
    last_used_at: float = 0  # when last recalled
    use_count: int = 0  # how many times recalled
    
    def __post_init__(self):
        if not self.snapshot_id:
            import uuid
            self.snapshot_id = str(uuid.uuid4())[:8]
        if not self.tokens:
            self.tokens = tokenize(self.text)


@dataclass
class ResonanceContext:
    """
    What resonance recall gives back before generation.
    """
    recalled_texts: List[str]  # The actual recalled snippets
    token_boosts: Dict[str, float]  # token → boost factor [0, 1]
    resonance_score: float  # overall resonance strength
    num_recalled: int  # how many snapshots were recalled


# ============================================================================
# RESONANT RECALL
# ============================================================================

class DrunkSanta:
    """
    DrunkSanta — Haze's resonant recall layer.
    
    Like Leo's SantaClaus, but drunk.
    
    He stumbles through memories, sometimes bringing back exactly
    what you need, sometimes bringing back something completely wrong
    but somehow still beautiful.
    
    "I found this in my pocket. Not sure if it's yours.
     But it felt like it wanted to be given away."
    """
    
    def __init__(
        self,
        max_snapshots: int = 512,
        max_recall: int = 5,
        max_tokens_per_snapshot: int = 64,
        alpha: float = 0.3,
        min_quality: float = 0.6,
        drunk_factor: float = DRUNK_FACTOR,
    ):
        """
        Args:
            max_snapshots: Maximum snapshots to keep in memory
            max_recall: How many snapshots to recall per generation
            max_tokens_per_snapshot: Truncate recalled text before scoring
            alpha: Overall strength of sampling bias
            min_quality: Minimum quality to store a snapshot
            drunk_factor: Probability of random recall (creative sloppiness)
        """
        self.max_snapshots = max_snapshots
        self.max_recall = max_recall
        self.max_tokens_per_snapshot = max_tokens_per_snapshot
        self.alpha = alpha
        self.min_quality = min_quality
        self.drunk_factor = drunk_factor
        
        # In-memory storage
        self.snapshots: List[Snapshot] = []
        
        # Stats
        self.total_stored = 0
        self.total_recalled = 0
        self.drunk_recalls = 0  # times when DrunkSanta picked randomly
    
    # ========================================================================
    # STORE
    # ========================================================================
    
    def store(
        self,
        text: str,
        quality: float,
        arousal: float = 0.0,
        entropy: float = 0.5,
        trauma_level: float = 0.0,
    ) -> bool:
        """
        Store a new snapshot if it's good enough.
        
        Returns True if stored, False if rejected (low quality).
        """
        if quality < self.min_quality:
            return False
        
        if not text or not text.strip():
            return False
        
        tokens = tokenize(text)
        if len(tokens) < 3:
            return False
        
        snapshot = Snapshot(
            snapshot_id="",
            text=text,
            tokens=tokens,
            quality=quality,
            arousal=arousal,
            entropy=entropy,
            trauma_level=trauma_level,
            created_at=time.time(),
        )
        
        self.snapshots.append(snapshot)
        self.total_stored += 1
        
        # Prune if needed (keep highest quality)
        if len(self.snapshots) > self.max_snapshots:
            # Sort by quality, keep top max_snapshots
            self.snapshots.sort(key=lambda s: s.quality, reverse=True)
            self.snapshots = self.snapshots[:self.max_snapshots]
        
        return True
    
    # ========================================================================
    # RECALL
    # ========================================================================
    
    def recall(
        self,
        prompt_text: str,
        current_arousal: float = 0.0,
        active_themes: Optional[List[str]] = None,
    ) -> Optional[ResonanceContext]:
        """
        Main entry point — find resonant snapshots for current context.
        
        Returns None if no useful recall.
        """
        if not prompt_text or not prompt_text.strip():
            return None
        
        if not self.snapshots:
            return None
        
        # Tokenize prompt
        prompt_tokens = tokenize(prompt_text)
        prompt_token_set = set(prompt_tokens)
        
        if not prompt_token_set:
            return None
        
        active_themes = active_themes or []
        active_theme_set = set(t.lower() for t in active_themes)
        
        now = time.time()
        
        # Score each snapshot
        scored: List[Tuple[float, Snapshot]] = []
        
        for snapshot in self.snapshots:
            score = self._score_snapshot(
                snapshot,
                prompt_token_set,
                active_theme_set,
                current_arousal,
                now,
            )
            
            if score > 0.1:  # threshold
                scored.append((score, snapshot))
        
        if not scored:
            return None
        
        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # DrunkSanta's magic: sometimes pick randomly instead of best
        # This adds creative unpredictability
        import random
        
        top_memories = []
        is_drunk = False
        
        for i in range(min(self.max_recall, len(scored))):
            if random.random() < self.drunk_factor and len(scored) > 1:
                # DrunkSanta stumbles and picks a random one
                random_idx = random.randint(0, len(scored) - 1)
                top_memories.append(scored[random_idx])
                is_drunk = True
            else:
                # Sober moment: pick the best remaining
                if i < len(scored):
                    top_memories.append(scored[i])
        
        if is_drunk:
            self.drunk_recalls += 1
        
        # Build result
        recalled_texts: List[str] = []
        all_tokens: List[str] = []
        
        for score, snapshot in top_memories:
            # Truncate if needed
            tokens = snapshot.tokens[:self.max_tokens_per_snapshot]
            text = " ".join(tokens)
            
            recalled_texts.append(text)
            all_tokens.extend(tokens)
            
            # Update usage
            snapshot.last_used_at = now
            snapshot.use_count += 1
        
        self.total_recalled += len(top_memories)
        
        # Build token boosts
        token_counts = Counter(all_tokens)
        max_count = max(token_counts.values()) if token_counts else 1
        
        token_boosts = {
            token: (count / max_count) * self.alpha
            for token, count in token_counts.items()
        }
        
        # Overall resonance score
        resonance_score = sum(s for s, _ in top_memories) / len(top_memories)
        
        return ResonanceContext(
            recalled_texts=recalled_texts,
            token_boosts=token_boosts,
            resonance_score=resonance_score,
            num_recalled=len(top_memories),
        )
    
    def _score_snapshot(
        self,
        snapshot: Snapshot,
        prompt_token_set: Set[str],
        active_theme_set: Set[str],
        current_arousal: float,
        now: float,
    ) -> float:
        """
        Score a snapshot for resonance with current context.
        
        Components:
        1. Token overlap (Jaccard similarity)
        2. Theme overlap (if themes provided)
        3. Arousal proximity
        4. Quality prior
        5. Recency penalty (don't repeat too often)
        6. Sticky phrase penalty (avoid contaminated patterns)
        """
        snapshot_token_set = set(snapshot.tokens)
        
        if not snapshot_token_set:
            return 0.0
        
        # 1. Token overlap (Jaccard)
        overlap = len(prompt_token_set & snapshot_token_set)
        union = len(prompt_token_set | snapshot_token_set)
        token_overlap = overlap / union if union > 0 else 0.0
        
        # 2. Theme overlap
        theme_overlap = 0.0
        if active_theme_set:
            theme_words_in_snapshot = sum(
                1 for t in active_theme_set if t in snapshot_token_set
            )
            theme_overlap = theme_words_in_snapshot / len(active_theme_set)
        
        # 3. Arousal proximity
        arousal_diff = abs(current_arousal - snapshot.arousal)
        arousal_score = max(0.0, 1.0 - arousal_diff)
        
        # 4. Quality prior
        quality = snapshot.quality
        
        # 5. Recency penalty
        if snapshot.last_used_at > 0:
            hours_since_use = (now - snapshot.last_used_at) / 3600.0
            if hours_since_use < RECENCY_WINDOW_HOURS:
                recency_penalty = 1.0 - (hours_since_use / RECENCY_WINDOW_HOURS)
            else:
                recency_penalty = 0.0
        else:
            recency_penalty = 0.0
        
        quality_with_recency = quality * (1.0 - RECENCY_PENALTY_STRENGTH * recency_penalty)
        
        # 6. Sticky phrase penalty
        snapshot_lower = snapshot.text.lower()
        for phrase in STICKY_PHRASES:
            if phrase in snapshot_lower:
                quality_with_recency *= 0.1  # 90% penalty
                break
        
        # Combine scores
        score = (
            0.4 * token_overlap +
            0.2 * theme_overlap +
            0.2 * arousal_score +
            0.2 * quality_with_recency
        )
        
        return score
    
    # ========================================================================
    # STATS
    # ========================================================================
    
    def stats(self) -> Dict[str, Any]:
        """Return recall stats."""
        qualities = [s.quality for s in self.snapshots]
        return {
            "total_snapshots": len(self.snapshots),
            "total_stored": self.total_stored,
            "total_recalled": self.total_recalled,
            "drunk_recalls": self.drunk_recalls,  # times Santa stumbled
            "drunk_ratio": self.drunk_recalls / max(1, self.total_recalled),
            "avg_quality": sum(qualities) / len(qualities) if qualities else 0.0,
            "max_quality": max(qualities) if qualities else 0.0,
        }


# ============================================================================
# ASYNC DRUNK SANTA
# ============================================================================

class AsyncDrunkSanta:
    """
    Async version of DrunkSanta with field lock discipline.
    
    Fully async for field coherence (like Leo's 47% improvement).
    
    "He's drunk, but he's disciplined about his locks."
    """
    
    def __init__(
        self,
        max_snapshots: int = 512,
        max_recall: int = 5,
        alpha: float = 0.3,
        min_quality: float = 0.6,
        drunk_factor: float = DRUNK_FACTOR,
    ):
        self._lock = asyncio.Lock()
        self._santa = DrunkSanta(
            max_snapshots=max_snapshots,
            max_recall=max_recall,
            alpha=alpha,
            min_quality=min_quality,
            drunk_factor=drunk_factor,
        )
    
    async def store(
        self,
        text: str,
        quality: float,
        arousal: float = 0.0,
        entropy: float = 0.5,
        trauma_level: float = 0.0,
    ) -> bool:
        """Async store with lock."""
        async with self._lock:
            return self._santa.store(text, quality, arousal, entropy, trauma_level)
    
    async def recall(
        self,
        prompt_text: str,
        current_arousal: float = 0.0,
        active_themes: Optional[List[str]] = None,
    ) -> Optional[ResonanceContext]:
        """Async recall with lock."""
        async with self._lock:
            return self._santa.recall(prompt_text, current_arousal, active_themes)
    
    async def stats(self) -> Dict[str, Any]:
        """Async stats."""
        async with self._lock:
            return self._santa.stats()


# ============================================================================
# TEST
# ============================================================================

def _test_drunksanta():
    """Quick test of DrunkSanta."""
    santa = DrunkSanta(min_quality=0.5, drunk_factor=0.3)  # Extra drunk for testing
    
    # Store some snapshots
    texts = [
        ("I love you darling. You're my everything.", 0.8, 0.7),
        ("The living room was dark. He put two cigarettes.", 0.7, 0.3),
        ("What is it? I don't believe you.", 0.6, 0.5),
        ("You're just stuck on the gas.", 0.75, 0.6),
        ("Tell me something? I thought you never left the house.", 0.85, 0.4),
    ]
    
    for text, quality, arousal in texts:
        santa.store(text, quality, arousal)
    
    print("=== 🍷 DRUNK SANTA TEST 🎅 ===")
    print(f"Stored: {santa.stats()['total_snapshots']} snapshots")
    
    # Recall for different prompts
    prompts = [
        "I love you",
        "What is happening?",
        "Tell me something about yourself",
    ]
    
    for prompt in prompts:
        result = santa.recall(prompt, current_arousal=0.5)
        if result:
            print(f"\nPrompt: '{prompt}'")
            print(f"  Resonance: {result.resonance_score:.2f}")
            print(f"  Recalled: {result.num_recalled}")
            print(f"  Tokens boosted: {len(result.token_boosts)}")
            if result.recalled_texts:
                print(f"  First: '{result.recalled_texts[0][:50]}...'")
        else:
            print(f"\nPrompt: '{prompt}' — no resonance")
    
    # Show drunk stats
    stats = santa.stats()
    print(f"\n🥃 Drunk Stats:")
    print(f"  Drunk recalls: {stats['drunk_recalls']}")
    print(f"  Drunk ratio: {stats['drunk_ratio']:.1%}")


if __name__ == "__main__":
    _test_drunksanta()
