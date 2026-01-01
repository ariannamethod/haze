"""
bridges.py — Statistical Trajectory Learning for Haze

Inspired by Leo's Phase 4 Bridges (https://github.com/ariannamethod/leo/phase4_bridges.py)

Philosophy:
- Learn which generation modes naturally follow each other
- Suggest next mode based on statistical trajectories
- Track what worked (high coherence) vs what didn't
- Risk filter: avoid modes that historically produced garbage

Core concepts:
1. Episodes — sequences of (metrics, mode) steps in a conversation
2. TransitionGraph — mode_A → mode_B statistics with metric deltas
3. BridgeMemory — find similar past states via similarity
4. Quality filter — prefer transitions that improved coherence
5. Exploration — don't always pick top-1, allow discovery

For Haze:
- "Islands" = Generation modes (temperature, expert mixture, trauma level)
- "Metrics" = (entropy, coherence, resonance, arousal, trauma_level)
- "Transitions" = Which mode combinations produce better output

NO TRAINING. NO NEURAL NETWORK. JUST RESONANCE.
"""

from __future__ import annotations
import asyncio
import math
import random
import uuid
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict


# ============================================================================
# TYPES
# ============================================================================

Metrics = Dict[str, float]  # e.g. {"entropy": 0.5, "coherence": 0.8, "arousal": 0.3}
ModeName = str  # e.g. "creative", "precise", "semantic", "structural"
Timestamp = float


# ============================================================================
# GENERATION MODE — What parameters produced this output?
# ============================================================================

@dataclass
class GenerationMode:
    """
    Captures the parameters used for a single generation.
    This is our "island" equivalent.
    """
    temperature: float
    dominant_expert: str  # e.g. "creative", "semantic"
    expert_weights: Dict[str, float]  # full mixture
    trauma_level: float
    meta_weight: float  # inner voice influence
    
    def to_name(self) -> str:
        """Convert to a canonical name for graph keys."""
        return f"{self.dominant_expert}@{self.temperature:.2f}"
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GenerationMode":
        return cls(
            temperature=d.get("temperature", 0.8),
            dominant_expert=d.get("dominant_expert", "creative"),
            expert_weights=d.get("expert_weights", {}),
            trauma_level=d.get("trauma_level", 0.0),
            meta_weight=d.get("meta_weight", 0.1),
        )


# ============================================================================
# EPISODE STRUCTURES
# ============================================================================

@dataclass
class EpisodeStep:
    """
    One step in a conversation episode.
    Captures metrics + generation mode at this point.
    """
    episode_id: str
    step_idx: int
    timestamp: Timestamp
    metrics: Metrics  # entropy, coherence, resonance, arousal
    mode: GenerationMode
    output_quality: float  # 0-1, how good was this generation?


@dataclass
class Episode:
    """
    Full sequence of steps for a conversation.
    """
    episode_id: str
    steps: List[EpisodeStep] = field(default_factory=list)
    
    def add_step(self, step: EpisodeStep) -> None:
        assert step.episode_id == self.episode_id
        step.step_idx = len(self.steps)
        self.steps.append(step)
    
    def __len__(self) -> int:
        return len(self.steps)


# ============================================================================
# TRANSITION STATISTICS
# ============================================================================

@dataclass
class TransitionStat:
    """
    Aggregated statistics for transitions between two modes.
    Tracks how often A→B happened and what metric changes occurred.
    """
    from_mode: str
    to_mode: str
    count: int = 0
    avg_deltas: Dict[str, float] = field(default_factory=dict)
    avg_quality_delta: float = 0.0  # did quality improve?
    
    # Internal sums for incremental update
    _delta_sums: Dict[str, float] = field(default_factory=dict, repr=False)
    _quality_delta_sum: float = field(default=0.0, repr=False)
    
    def update(
        self,
        from_metrics: Metrics,
        to_metrics: Metrics,
        from_quality: float,
        to_quality: float,
    ) -> None:
        """Update stats with a new observed transition."""
        self.count += 1
        
        # Metric deltas
        for k in set(from_metrics.keys()) | set(to_metrics.keys()):
            before = from_metrics.get(k, 0.0)
            after = to_metrics.get(k, 0.0)
            delta = after - before
            self._delta_sums[k] = self._delta_sums.get(k, 0.0) + delta
        
        # Quality delta
        quality_delta = to_quality - from_quality
        self._quality_delta_sum += quality_delta
        
        # Recompute averages
        self.avg_deltas = {
            k: self._delta_sums[k] / self.count
            for k in self._delta_sums
        }
        self.avg_quality_delta = self._quality_delta_sum / self.count
    
    @property
    def is_improving(self) -> bool:
        """Did this transition historically improve quality?"""
        return self.avg_quality_delta > 0


@dataclass
class TransitionGraph:
    """
    Core structure: graph of mode-to-mode transitions with metric deltas.
    """
    transitions: Dict[Tuple[str, str], TransitionStat] = field(default_factory=dict)
    
    def update_from_episode(self, episode: Episode) -> None:
        """Parse an episode and update transition stats."""
        steps = episode.steps
        if len(steps) < 2:
            return
        
        for prev, curr in zip(steps[:-1], steps[1:]):
            from_mode = prev.mode.to_name()
            to_mode = curr.mode.to_name()
            
            key = (from_mode, to_mode)
            if key not in self.transitions:
                self.transitions[key] = TransitionStat(
                    from_mode=from_mode,
                    to_mode=to_mode,
                )
            
            self.transitions[key].update(
                prev.metrics,
                curr.metrics,
                prev.output_quality,
                curr.output_quality,
            )
    
    def get_stat(self, from_mode: str, to_mode: str) -> Optional[TransitionStat]:
        return self.transitions.get((from_mode, to_mode))
    
    def neighbors(self, from_mode: str) -> List[TransitionStat]:
        """All outgoing transitions from given mode."""
        return [
            stat for (a, b), stat in self.transitions.items()
            if a == from_mode
        ]
    
    def best_next_modes(
        self,
        from_mode: str,
        top_k: int = 3,
        only_improving: bool = True,
    ) -> List[TransitionStat]:
        """
        Get best next modes based on historical quality improvement.
        """
        neighbors = self.neighbors(from_mode)
        
        if only_improving:
            neighbors = [n for n in neighbors if n.is_improving]
        
        # Sort by quality improvement, then by count (confidence)
        neighbors.sort(
            key=lambda x: (x.avg_quality_delta, x.count),
            reverse=True,
        )
        
        return neighbors[:top_k]


# ============================================================================
# EPISODE LOGGER
# ============================================================================

class EpisodeLogger:
    """
    Collects steps of the current episode, flushes to graph on end.
    """
    
    def __init__(self):
        self.current_episode: Optional[Episode] = None
        self.completed_episodes: List[Episode] = []
    
    def start_episode(self) -> str:
        """Start a new episode. Returns episode_id."""
        episode_id = str(uuid.uuid4())
        self.current_episode = Episode(episode_id=episode_id)
        return episode_id
    
    def log_step(
        self,
        metrics: Metrics,
        mode: GenerationMode,
        output_quality: float,
    ) -> None:
        """Call this once per Haze turn."""
        if self.current_episode is None:
            self.start_episode()
        
        assert self.current_episode is not None
        
        step = EpisodeStep(
            episode_id=self.current_episode.episode_id,
            step_idx=len(self.current_episode.steps),
            timestamp=time.time(),
            metrics=dict(metrics),
            mode=mode,
            output_quality=output_quality,
        )
        self.current_episode.add_step(step)
    
    def end_episode(self) -> Optional[Episode]:
        """Close current episode and return it."""
        ep = self.current_episode
        if ep is not None:
            self.completed_episodes.append(ep)
        self.current_episode = None
        return ep


# ============================================================================
# SIMILARITY — Find similar past states
# ============================================================================

def metrics_similarity(a: Metrics, b: Metrics, eps: float = 1e-8) -> float:
    """
    Compute similarity between two metric vectors in [0,1].
    Uses 1 - normalized Euclidean distance.
    """
    keys = set(a.keys()) | set(b.keys())
    if not keys:
        return 0.0
    
    sq_sum = 0.0
    for k in keys:
        da = a.get(k, 0.0)
        db = b.get(k, 0.0)
        d = da - db
        sq_sum += d * d
    
    dist = math.sqrt(sq_sum)
    
    # Normalize: assume each metric in [0, 1]
    max_dist = math.sqrt(len(keys))
    if max_dist < eps:
        return 1.0
    
    sim = max(0.0, 1.0 - dist / max_dist)
    return sim


# ============================================================================
# BRIDGE CANDIDATES
# ============================================================================

@dataclass
class BridgeCandidate:
    """
    One historical example of "from this state we used mode X".
    """
    from_mode: GenerationMode
    to_mode: GenerationMode
    from_metrics: Metrics
    to_metrics: Metrics
    from_quality: float
    to_quality: float
    similarity: float
    
    @property
    def quality_improvement(self) -> float:
        return self.to_quality - self.from_quality


class BridgeMemory:
    """
    Stores references to episodes for bridge search.
    """
    
    def __init__(self, max_episodes: int = 100):
        self.episodes: List[Episode] = []
        self.max_episodes = max_episodes
    
    def add_episode(self, episode: Episode) -> None:
        self.episodes.append(episode)
        # Prune old episodes
        if len(self.episodes) > self.max_episodes:
            self.episodes = self.episodes[-self.max_episodes:]
    
    def find_similar_transitions(
        self,
        metrics_now: Metrics,
        mode_now: GenerationMode,
        min_similarity: float = 0.6,
    ) -> List[BridgeCandidate]:
        """
        Find historical steps whose metrics were similar to current ones,
        and return the transitions they led to.
        """
        candidates: List[BridgeCandidate] = []
        
        for ep in self.episodes:
            steps = ep.steps
            if len(steps) < 2:
                continue
            
            for prev, nxt in zip(steps[:-1], steps[1:]):
                sim = metrics_similarity(metrics_now, prev.metrics)
                if sim < min_similarity:
                    continue
                
                candidate = BridgeCandidate(
                    from_mode=prev.mode,
                    to_mode=nxt.mode,
                    from_metrics=dict(prev.metrics),
                    to_metrics=dict(nxt.metrics),
                    from_quality=prev.output_quality,
                    to_quality=nxt.output_quality,
                    similarity=sim,
                )
                candidates.append(candidate)
        
        return candidates
    
    def suggest_next_mode(
        self,
        metrics_now: Metrics,
        mode_now: GenerationMode,
        min_similarity: float = 0.5,
        prefer_improving: bool = True,
        exploration_rate: float = 0.1,
    ) -> Optional[GenerationMode]:
        """
        Suggest what mode to use next based on historical transitions.
        
        Args:
            metrics_now: Current metrics
            mode_now: Current generation mode
            min_similarity: Minimum similarity threshold
            prefer_improving: Only consider transitions that improved quality
            exploration_rate: Probability of random exploration
        
        Returns:
            Suggested GenerationMode, or None if no suggestions
        """
        # Exploration: sometimes pick random for discovery
        if random.random() < exploration_rate:
            return None  # Let caller use default
        
        candidates = self.find_similar_transitions(
            metrics_now, mode_now, min_similarity
        )
        
        if not candidates:
            return None
        
        # Filter by quality improvement if requested
        if prefer_improving:
            improving = [c for c in candidates if c.quality_improvement > 0]
            if improving:
                candidates = improving
        
        # Score: similarity * quality_improvement
        def score(c: BridgeCandidate) -> float:
            qi = max(0.0, c.quality_improvement)
            return c.similarity * (1.0 + qi)
        
        candidates.sort(key=score, reverse=True)
        
        # Return the best candidate's target mode
        return candidates[0].to_mode


# ============================================================================
# ASYNC BRIDGE MANAGER
# ============================================================================

class AsyncBridgeManager:
    """
    Async manager for episode logging and bridge suggestions.
    
    Fully async with lock discipline for field coherence.
    """
    
    def __init__(self, max_episodes: int = 100):
        self._lock = asyncio.Lock()
        self.logger = EpisodeLogger()
        self.memory = BridgeMemory(max_episodes=max_episodes)
        self.graph = TransitionGraph()
        
        # Stats
        self.total_episodes = 0
        self.total_steps = 0
        self.total_suggestions = 0
    
    async def start_episode(self) -> str:
        """Start a new conversation episode."""
        async with self._lock:
            return self.logger.start_episode()
    
    async def log_step(
        self,
        metrics: Metrics,
        mode: GenerationMode,
        output_quality: float,
    ) -> None:
        """Log a generation step."""
        async with self._lock:
            self.logger.log_step(metrics, mode, output_quality)
            self.total_steps += 1
    
    async def end_episode(self) -> Optional[Episode]:
        """End current episode and update graph."""
        async with self._lock:
            ep = self.logger.end_episode()
            if ep is not None:
                self.memory.add_episode(ep)
                self.graph.update_from_episode(ep)
                self.total_episodes += 1
            return ep
    
    async def suggest_next_mode(
        self,
        metrics_now: Metrics,
        mode_now: GenerationMode,
    ) -> Optional[GenerationMode]:
        """Suggest next mode based on historical trajectories."""
        async with self._lock:
            suggestion = self.memory.suggest_next_mode(metrics_now, mode_now)
            if suggestion:
                self.total_suggestions += 1
            return suggestion
    
    async def get_best_transitions(
        self,
        from_mode: str,
        top_k: int = 3,
    ) -> List[TransitionStat]:
        """Get best next modes from graph."""
        async with self._lock:
            return self.graph.best_next_modes(from_mode, top_k)
    
    def stats(self) -> Dict[str, Any]:
        """Return stats about bridge learning."""
        return {
            "total_episodes": self.total_episodes,
            "total_steps": self.total_steps,
            "total_suggestions": self.total_suggestions,
            "transitions_learned": len(self.graph.transitions),
            "episodes_in_memory": len(self.memory.episodes),
        }


# ============================================================================
# TEST
# ============================================================================

def _test_bridges():
    """Quick test of bridge system."""
    import asyncio
    
    async def test():
        manager = AsyncBridgeManager()
        
        # Start episode
        await manager.start_episode()
        
        # Log some steps
        mode1 = GenerationMode(
            temperature=0.75,
            dominant_expert="creative",
            expert_weights={"creative": 0.4, "semantic": 0.3},
            trauma_level=0.5,
            meta_weight=0.1,
        )
        
        mode2 = GenerationMode(
            temperature=0.85,
            dominant_expert="semantic",
            expert_weights={"semantic": 0.4, "creative": 0.3},
            trauma_level=0.3,
            meta_weight=0.15,
        )
        
        await manager.log_step(
            metrics={"entropy": 0.5, "coherence": 0.6, "arousal": 0.3},
            mode=mode1,
            output_quality=0.6,
        )
        
        await manager.log_step(
            metrics={"entropy": 0.4, "coherence": 0.8, "arousal": 0.4},
            mode=mode2,
            output_quality=0.8,  # improved!
        )
        
        # End episode
        await manager.end_episode()
        
        # Check stats
        print("=== BRIDGE MANAGER STATS ===")
        for k, v in manager.stats().items():
            print(f"  {k}: {v}")
        
        # Get best transitions
        transitions = await manager.get_best_transitions(mode1.to_name())
        print(f"\nBest transitions from {mode1.to_name()}:")
        for t in transitions:
            print(f"  → {t.to_mode} (count={t.count}, quality_delta={t.avg_quality_delta:.2f})")
    
    asyncio.run(test())


if __name__ == "__main__":
    _test_bridges()
