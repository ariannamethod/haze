"""
episodes.py — Episodic Memory for Haze

Inspired by Leo's episodes.py (https://github.com/ariannamethod/leo)

Haze remembers specific moments: seed + output + metrics.
This is its episodic memory — structured recall of its own generations.

No external APIs. No heavy embeddings. Just local storage + simple similarity.

Core idea:
- Store each generation as an episode
- Query similar past episodes by metrics
- Learn from high-quality generations
- Self-RAG: retrieve from own history, not external corpus

NO TRAINING. NO NEURAL NETWORK. JUST RESONANCE.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class HazeMetrics:
    """
    Metrics captured for each episode.
    
    These are the "internal state" that describes what Haze was "feeling"
    during this generation.
    """
    entropy: float = 0.0
    coherence: float = 0.0
    resonance: float = 0.0
    arousal: float = 0.0
    novelty: float = 0.0
    trauma_level: float = 0.0
    
    # Expert mixture
    temperature: float = 0.8
    dominant_expert: str = "creative"
    expert_weights: Dict[str, float] = field(default_factory=dict)
    
    # Meta state
    meta_weight: float = 0.0
    used_meta: bool = False
    
    # Overthinking
    overthinking_enabled: bool = False
    rings_count: int = 0
    
    # Quality score (0-1, how good was this generation?)
    quality: float = 0.5
    
    def to_vector(self) -> List[float]:
        """Convert to feature vector for similarity search."""
        return [
            self.entropy,
            self.coherence,
            self.resonance,
            self.arousal,
            self.novelty,
            self.trauma_level,
            self.temperature,
            self.meta_weight,
            self.quality,
        ]
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dict."""
        return {
            "entropy": self.entropy,
            "coherence": self.coherence,
            "resonance": self.resonance,
            "arousal": self.arousal,
            "novelty": self.novelty,
            "trauma_level": self.trauma_level,
            "temperature": self.temperature,
            "meta_weight": self.meta_weight,
            "quality": self.quality,
        }


@dataclass
class Episode:
    """
    One moment in Haze's life.
    
    Captures the full context of a single generation:
    - What seed was used
    - What output was produced
    - What was Haze's internal state
    """
    seed: str
    output: str
    metrics: HazeMetrics
    timestamp: float = field(default_factory=time.time)
    episode_id: str = ""
    
    def __post_init__(self):
        if not self.episode_id:
            import uuid
            self.episode_id = str(uuid.uuid4())[:8]


# ============================================================================
# SIMILARITY
# ============================================================================

def cosine_distance(a: List[float], b: List[float]) -> float:
    """Compute cosine distance between two vectors (1 - cosine similarity)."""
    if len(a) != len(b):
        return 1.0
    
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    
    if na == 0 or nb == 0:
        return 1.0
    
    similarity = dot / (na * nb)
    return 1.0 - similarity


def euclidean_distance(a: List[float], b: List[float]) -> float:
    """Compute Euclidean distance between two vectors."""
    if len(a) != len(b):
        return float('inf')
    
    sq_sum = sum((x - y) ** 2 for x, y in zip(a, b))
    return math.sqrt(sq_sum)


# ============================================================================
# EPISODIC MEMORY
# ============================================================================

class EpisodicMemory:
    """
    Local episodic memory for Haze.
    
    Stores (seed, output, metrics, quality) as episodes.
    Provides simple similarity search over internal metrics.
    
    This is Self-RAG: retrieve from own history, not external corpus.
    """
    
    def __init__(self, max_episodes: int = 1000):
        self.episodes: List[Episode] = []
        self.max_episodes = max_episodes
        
        # Indices for fast lookup
        self._by_quality: List[Tuple[float, int]] = []  # (quality, index)
        self._by_trauma: List[Tuple[float, int]] = []  # (trauma, index)
    
    def observe(self, episode: Episode) -> None:
        """
        Insert one episode into memory.
        
        Safe: clamps all values, ignores NaNs.
        """
        # Clamp and sanitize
        def clamp(x: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
            if x != x:  # NaN check
                return 0.0
            return max(min_val, min(max_val, x))
        
        episode.metrics.entropy = clamp(episode.metrics.entropy)
        episode.metrics.coherence = clamp(episode.metrics.coherence)
        episode.metrics.resonance = clamp(episode.metrics.resonance)
        episode.metrics.arousal = clamp(episode.metrics.arousal)
        episode.metrics.novelty = clamp(episode.metrics.novelty)
        episode.metrics.trauma_level = clamp(episode.metrics.trauma_level)
        episode.metrics.temperature = clamp(episode.metrics.temperature, 0.0, 2.0)
        episode.metrics.meta_weight = clamp(episode.metrics.meta_weight)
        episode.metrics.quality = clamp(episode.metrics.quality)
        
        # Add to list
        idx = len(self.episodes)
        self.episodes.append(episode)
        
        # Update indices
        self._by_quality.append((episode.metrics.quality, idx))
        self._by_trauma.append((episode.metrics.trauma_level, idx))
        
        # Prune if needed
        if len(self.episodes) > self.max_episodes:
            # Remove oldest episodes
            self.episodes = self.episodes[-self.max_episodes:]
            # Rebuild indices
            self._rebuild_indices()
    
    def _rebuild_indices(self) -> None:
        """Rebuild lookup indices after pruning."""
        self._by_quality = [
            (ep.metrics.quality, i) for i, ep in enumerate(self.episodes)
        ]
        self._by_trauma = [
            (ep.metrics.trauma_level, i) for i, ep in enumerate(self.episodes)
        ]
    
    def query_similar(
        self,
        metrics: HazeMetrics,
        top_k: int = 5,
        min_quality: float = 0.0,
    ) -> List[Episode]:
        """
        Find past episodes with similar internal configuration.
        
        Args:
            metrics: Current metrics to match
            top_k: Number of results to return
            min_quality: Minimum quality threshold
        
        Returns:
            List of similar episodes, sorted by similarity
        """
        if not self.episodes:
            return []
        
        query_vec = metrics.to_vector()
        
        # Compute distances
        distances: List[Tuple[float, Episode]] = []
        
        for episode in self.episodes:
            if episode.metrics.quality < min_quality:
                continue
            
            ep_vec = episode.metrics.to_vector()
            dist = cosine_distance(query_vec, ep_vec)
            distances.append((dist, episode))
        
        # Sort by distance (lower = more similar)
        distances.sort(key=lambda x: x[0])
        
        # Return top_k
        return [ep for _, ep in distances[:top_k]]
    
    def query_high_quality(self, top_k: int = 10) -> List[Episode]:
        """Get top K highest quality episodes."""
        sorted_eps = sorted(
            self._by_quality,
            key=lambda x: x[0],
            reverse=True,
        )
        return [self.episodes[idx] for _, idx in sorted_eps[:top_k]]
    
    def query_high_trauma(self, top_k: int = 10) -> List[Episode]:
        """Get top K highest trauma episodes."""
        sorted_eps = sorted(
            self._by_trauma,
            key=lambda x: x[0],
            reverse=True,
        )
        return [self.episodes[idx] for _, idx in sorted_eps[:top_k]]
    
    def query_by_seed_overlap(
        self,
        seed: str,
        top_k: int = 5,
    ) -> List[Episode]:
        """
        Find episodes with similar seeds (word overlap).
        
        Simple bag-of-words overlap for seed matching.
        """
        query_words = set(seed.lower().split())
        
        if not query_words:
            return []
        
        # Compute overlap for each episode
        overlaps: List[Tuple[float, Episode]] = []
        
        for episode in self.episodes:
            ep_words = set(episode.seed.lower().split())
            if not ep_words:
                continue
            
            overlap = len(query_words & ep_words)
            jaccard = overlap / len(query_words | ep_words)
            overlaps.append((jaccard, episode))
        
        # Sort by overlap (higher = more similar)
        overlaps.sort(key=lambda x: x[0], reverse=True)
        
        return [ep for _, ep in overlaps[:top_k]]
    
    def get_quality_distribution(self) -> Dict[str, float]:
        """Get quality distribution stats."""
        if not self.episodes:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
        
        qualities = [ep.metrics.quality for ep in self.episodes]
        mean = sum(qualities) / len(qualities)
        variance = sum((q - mean) ** 2 for q in qualities) / len(qualities)
        std = math.sqrt(variance)
        
        return {
            "min": min(qualities),
            "max": max(qualities),
            "mean": mean,
            "std": std,
        }
    
    def stats(self) -> Dict[str, Any]:
        """Return memory stats."""
        quality_dist = self.get_quality_distribution()
        return {
            "total_episodes": len(self.episodes),
            "max_episodes": self.max_episodes,
            "quality_mean": quality_dist["mean"],
            "quality_std": quality_dist["std"],
            "quality_max": quality_dist["max"],
        }


# ============================================================================
# ASYNC EPISODIC MEMORY
# ============================================================================

class AsyncEpisodicMemory:
    """
    Async version of EpisodicMemory with field lock discipline.
    
    Fully async for field coherence (like Leo's 47% improvement).
    """
    
    def __init__(self, max_episodes: int = 1000):
        self._lock = asyncio.Lock()
        self._memory = EpisodicMemory(max_episodes)
    
    async def observe(self, episode: Episode) -> None:
        """Async observation with lock."""
        async with self._lock:
            self._memory.observe(episode)
    
    async def query_similar(
        self,
        metrics: HazeMetrics,
        top_k: int = 5,
        min_quality: float = 0.0,
    ) -> List[Episode]:
        """Async similarity query."""
        async with self._lock:
            return self._memory.query_similar(metrics, top_k, min_quality)
    
    async def query_high_quality(self, top_k: int = 10) -> List[Episode]:
        """Async high quality query."""
        async with self._lock:
            return self._memory.query_high_quality(top_k)
    
    async def query_by_seed_overlap(
        self,
        seed: str,
        top_k: int = 5,
    ) -> List[Episode]:
        """Async seed overlap query."""
        async with self._lock:
            return self._memory.query_by_seed_overlap(seed, top_k)
    
    async def stats(self) -> Dict[str, Any]:
        """Async stats."""
        async with self._lock:
            return self._memory.stats()


# ============================================================================
# SELF-RAG HELPER
# ============================================================================

def suggest_from_episodes(
    current_metrics: HazeMetrics,
    memory: EpisodicMemory,
    top_k: int = 3,
) -> Optional[Dict[str, Any]]:
    """
    Self-RAG: Suggest generation parameters based on similar past episodes.
    
    Looks at high-quality episodes with similar metrics and suggests
    what parameters worked well.
    
    Args:
        current_metrics: Current internal state
        memory: Episodic memory to query
        top_k: Number of similar episodes to consider
    
    Returns:
        Dict with suggested parameters, or None if no good suggestions
    """
    # Find similar high-quality episodes
    similar = memory.query_similar(current_metrics, top_k=top_k, min_quality=0.6)
    
    if not similar:
        return None
    
    # Average the parameters that worked well
    temps = [ep.metrics.temperature for ep in similar]
    metas = [ep.metrics.meta_weight for ep in similar]
    
    # Find most common dominant expert
    expert_counts: Dict[str, int] = defaultdict(int)
    for ep in similar:
        expert_counts[ep.metrics.dominant_expert] += 1
    
    best_expert = max(expert_counts.items(), key=lambda x: x[1])[0]
    
    return {
        "suggested_temperature": sum(temps) / len(temps),
        "suggested_meta_weight": sum(metas) / len(metas),
        "suggested_expert": best_expert,
        "based_on_episodes": len(similar),
        "avg_quality": sum(ep.metrics.quality for ep in similar) / len(similar),
    }


# ============================================================================
# TEST
# ============================================================================

def _test_episodes():
    """Quick test of episodic memory."""
    memory = EpisodicMemory()
    
    # Create some episodes
    for i in range(20):
        metrics = HazeMetrics(
            entropy=0.3 + i * 0.02,
            coherence=0.5 + i * 0.02,
            resonance=0.4 + i * 0.01,
            arousal=0.2 + (i % 5) * 0.1,
            trauma_level=0.1 + (i % 3) * 0.2,
            temperature=0.7 + i * 0.01,
            dominant_expert="creative" if i % 2 == 0 else "semantic",
            quality=0.4 + i * 0.03,
        )
        
        episode = Episode(
            seed=f"Test seed {i}",
            output=f"Test output {i}. This is some generated text.",
            metrics=metrics,
        )
        
        memory.observe(episode)
    
    # Query similar
    query_metrics = HazeMetrics(
        entropy=0.5,
        coherence=0.7,
        quality=0.7,
    )
    
    similar = memory.query_similar(query_metrics, top_k=3)
    
    print("=== EPISODIC MEMORY TEST ===")
    print(f"Total episodes: {len(memory.episodes)}")
    print(f"\nQuery similar to entropy=0.5, coherence=0.7:")
    for ep in similar:
        print(f"  {ep.episode_id}: entropy={ep.metrics.entropy:.2f}, coherence={ep.metrics.coherence:.2f}, quality={ep.metrics.quality:.2f}")
    
    # High quality
    high_q = memory.query_high_quality(top_k=3)
    print(f"\nTop 3 high quality:")
    for ep in high_q:
        print(f"  {ep.episode_id}: quality={ep.metrics.quality:.2f}")
    
    # Suggestions
    suggestion = suggest_from_episodes(query_metrics, memory)
    if suggestion:
        print(f"\nSuggested parameters:")
        for k, v in suggestion.items():
            print(f"  {k}: {v}")
    
    # Stats
    print(f"\nStats: {memory.stats()}")


if __name__ == "__main__":
    _test_episodes()
