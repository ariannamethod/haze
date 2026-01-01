"""
flow.py — Pattern Flow Through Time

Inspired by Leo's gowiththeflow.py (https://github.com/ariannamethod/leo)

"Go with the flow" — evolutionary tracking of semantic patterns.

Core idea:
- Patterns aren't static — they flow, grow, fade, merge
- Record pattern state after each reply → build archaeological record  
- Detect emerging patterns (↗), fading patterns (↘), persistent patterns (→)
- Enable trauma-pattern correlation: which patterns appear during high trauma?
- Track conversation phases as meaning flows through time

This is memory archaeology: watching resonance currents shift and eddy.
Not training data — just temporal awareness of the flow.

NO TRAINING. NO NEURAL NETWORK. JUST RESONANCE.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any, Deque
from collections import defaultdict, deque


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class PatternSnapshot:
    """
    Snapshot of a pattern at a specific moment in the flow.
    
    Captures:
    - When the pattern was active
    - How strongly it flowed (frequency/strength)
    - Which words belonged to it
    - Associated metrics at that moment
    """
    timestamp: float
    pattern_id: str  # e.g. trigram tuple as string
    strength: float  # activation score (frequency or weight)
    active_words: Set[str]
    metrics: Dict[str, float]  # entropy, coherence, trauma_level, etc.


@dataclass
class PatternTrajectory:
    """
    Evolution of a single pattern as it flows through time.
    
    Contains:
    - Full history of snapshots
    - Computed slope (growing/fading)
    - Current state
    """
    pattern_id: str
    snapshots: List[PatternSnapshot] = field(default_factory=list)
    
    def add_snapshot(self, snapshot: PatternSnapshot) -> None:
        """Add a new snapshot to the trajectory."""
        self.snapshots.append(snapshot)
    
    def slope(self, hours: float = 1.0) -> float:
        """
        Compute flow trajectory over last N hours.
        
        Positive slope → emerging pattern (↗ growing)
        Negative slope → fading pattern (↘ dying)  
        Zero slope → stable pattern (→ persistent)
        
        Uses linear regression over strength values.
        
        Args:
            hours: Time window to compute slope (default: 1 hour)
        
        Returns:
            Slope value: positive = growing, negative = fading, ~0 = stable
        """
        if len(self.snapshots) < 2:
            return 0.0
        
        now = time.time()
        cutoff = now - (hours * 3600)
        
        # Filter recent snapshots
        recent = [s for s in self.snapshots if s.timestamp >= cutoff]
        
        if len(recent) < 2:
            return 0.0
        
        # x = time offset from first snapshot (in seconds)
        # y = strength
        times = [s.timestamp - recent[0].timestamp for s in recent]
        strengths = [s.strength for s in recent]
        
        # Pure Python linear regression: slope = cov(x,y) / var(x)
        n = len(times)
        mean_t = sum(times) / n
        mean_s = sum(strengths) / n
        
        # Covariance and variance
        cov = sum((times[i] - mean_t) * (strengths[i] - mean_s) for i in range(n))
        var = sum((times[i] - mean_t) ** 2 for i in range(n))
        
        if var == 0:
            return 0.0
        
        # Slope in strength per second
        slope_per_sec = cov / var
        
        # Convert to strength per hour for readability
        slope_per_hour = slope_per_sec * 3600
        
        return slope_per_hour
    
    def current_strength(self) -> float:
        """Get most recent strength value."""
        if not self.snapshots:
            return 0.0
        return self.snapshots[-1].strength
    
    def lifetime_seconds(self) -> float:
        """How long has this pattern been flowing?"""
        if len(self.snapshots) < 2:
            return 0.0
        return self.snapshots[-1].timestamp - self.snapshots[0].timestamp
    
    def trend(self, threshold: float = 0.1) -> str:
        """
        Get trend indicator.
        
        Returns:
            "↗" for emerging, "↘" for fading, "→" for stable
        """
        s = self.slope()
        if s > threshold:
            return "↗"
        elif s < -threshold:
            return "↘"
        else:
            return "→"
    
    def avg_metrics(self) -> Dict[str, float]:
        """Compute average metrics across all snapshots."""
        if not self.snapshots:
            return {}
        
        all_keys: Set[str] = set()
        for s in self.snapshots:
            all_keys.update(s.metrics.keys())
        
        result = {}
        for key in all_keys:
            values = [s.metrics.get(key, 0.0) for s in self.snapshots]
            result[key] = sum(values) / len(values)
        
        return result


# ============================================================================
# FLOW STATE — Current state of all patterns
# ============================================================================

@dataclass
class FlowState:
    """
    Current state of pattern flow.
    
    Computed from trajectories, provides:
    - Emerging patterns (growing)
    - Fading patterns (dying)
    - Stable patterns (persistent)
    - Overall flow metrics
    """
    emerging: List[Tuple[str, float]]  # (pattern_id, slope)
    fading: List[Tuple[str, float]]
    stable: List[Tuple[str, float]]
    total_patterns: int
    avg_strength: float
    flow_entropy: float  # diversity of pattern strengths
    
    def emerging_score(self) -> float:
        """How much is emerging? (0-1)"""
        if self.total_patterns == 0:
            return 0.0
        return len(self.emerging) / self.total_patterns
    
    def fading_score(self) -> float:
        """How much is fading? (0-1)"""
        if self.total_patterns == 0:
            return 0.0
        return len(self.fading) / self.total_patterns


# ============================================================================
# FLOW TRACKER — The main engine
# ============================================================================

class FlowTracker:
    """
    Track the flow of patterns through time.
    
    This is Haze's memory archaeology:
    - Record pattern snapshots after each generation
    - Detect emerging vs fading patterns
    - Query pattern history and trajectories
    - Enable trauma-pattern correlation analysis
    
    Storage: In-memory with optional max history.
    """
    
    def __init__(self, max_snapshots_per_pattern: int = 100):
        self.trajectories: Dict[str, PatternTrajectory] = {}
        self.max_snapshots = max_snapshots_per_pattern
        
        # Stats
        self.total_snapshots = 0
        self.total_patterns_seen = 0
    
    def observe(
        self,
        patterns: Dict[str, float],  # pattern_id → strength
        metrics: Dict[str, float],  # current metrics
        words: Optional[Set[str]] = None,
    ) -> None:
        """
        Record pattern observations after a generation.
        
        Args:
            patterns: Dict of pattern_id → strength (e.g. trigram → count)
            metrics: Current metrics (entropy, coherence, trauma_level, etc.)
            words: Optional set of active words in this generation
        """
        timestamp = time.time()
        words = words or set()
        
        for pattern_id, strength in patterns.items():
            # Get or create trajectory
            if pattern_id not in self.trajectories:
                self.trajectories[pattern_id] = PatternTrajectory(pattern_id=pattern_id)
                self.total_patterns_seen += 1
            
            trajectory = self.trajectories[pattern_id]
            
            # Create snapshot
            snapshot = PatternSnapshot(
                timestamp=timestamp,
                pattern_id=pattern_id,
                strength=strength,
                active_words=words.copy(),
                metrics=dict(metrics),
            )
            
            trajectory.add_snapshot(snapshot)
            self.total_snapshots += 1
            
            # Prune old snapshots if needed
            if len(trajectory.snapshots) > self.max_snapshots:
                trajectory.snapshots = trajectory.snapshots[-self.max_snapshots:]
    
    def get_flow_state(self, slope_threshold: float = 0.1) -> FlowState:
        """
        Compute current flow state across all patterns.
        
        Args:
            slope_threshold: Threshold for emerging/fading classification
        
        Returns:
            FlowState with emerging, fading, stable patterns
        """
        emerging = []
        fading = []
        stable = []
        
        strengths = []
        
        for pattern_id, trajectory in self.trajectories.items():
            slope = trajectory.slope()
            strength = trajectory.current_strength()
            strengths.append(strength)
            
            if slope > slope_threshold:
                emerging.append((pattern_id, slope))
            elif slope < -slope_threshold:
                fading.append((pattern_id, slope))
            else:
                stable.append((pattern_id, slope))
        
        # Sort by absolute slope
        emerging.sort(key=lambda x: x[1], reverse=True)
        fading.sort(key=lambda x: x[1])
        
        # Compute flow entropy (diversity of strengths)
        flow_entropy = 0.0
        if strengths:
            total = sum(strengths)
            if total > 0:
                probs = [s / total for s in strengths]
                flow_entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        
        return FlowState(
            emerging=emerging,
            fading=fading,
            stable=stable,
            total_patterns=len(self.trajectories),
            avg_strength=sum(strengths) / len(strengths) if strengths else 0.0,
            flow_entropy=flow_entropy,
        )
    
    def get_trajectory(self, pattern_id: str) -> Optional[PatternTrajectory]:
        """Get trajectory for a specific pattern."""
        return self.trajectories.get(pattern_id)
    
    def get_top_emerging(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get top N emerging patterns."""
        state = self.get_flow_state()
        return state.emerging[:n]
    
    def get_top_fading(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get top N fading patterns."""
        state = self.get_flow_state()
        return state.fading[:n]
    
    def trauma_correlation(self, trauma_threshold: float = 0.5) -> Dict[str, float]:
        """
        Find patterns that correlate with high trauma.
        
        Returns dict of pattern_id → correlation score (higher = more correlated with trauma)
        """
        correlations = {}
        
        for pattern_id, trajectory in self.trajectories.items():
            if not trajectory.snapshots:
                continue
            
            # Count high-trauma snapshots vs total
            high_trauma_count = 0
            for snapshot in trajectory.snapshots:
                trauma = snapshot.metrics.get("trauma_level", 0.0)
                if trauma >= trauma_threshold:
                    high_trauma_count += 1
            
            # Correlation = fraction of snapshots that were high-trauma
            correlations[pattern_id] = high_trauma_count / len(trajectory.snapshots)
        
        return correlations
    
    def stats(self) -> Dict[str, Any]:
        """Return stats about flow tracking."""
        state = self.get_flow_state()
        return {
            "total_patterns": len(self.trajectories),
            "total_snapshots": self.total_snapshots,
            "emerging_count": len(state.emerging),
            "fading_count": len(state.fading),
            "stable_count": len(state.stable),
            "avg_strength": state.avg_strength,
            "flow_entropy": state.flow_entropy,
        }


# ============================================================================
# ASYNC FLOW TRACKER
# ============================================================================

class AsyncFlowTracker:
    """
    Async version of FlowTracker with field lock discipline.
    
    Fully async for field coherence (like Leo's 47% improvement).
    """
    
    def __init__(self, max_snapshots_per_pattern: int = 100):
        self._lock = asyncio.Lock()
        self._tracker = FlowTracker(max_snapshots_per_pattern)
    
    async def observe(
        self,
        patterns: Dict[str, float],
        metrics: Dict[str, float],
        words: Optional[Set[str]] = None,
    ) -> None:
        """Async observation with lock."""
        async with self._lock:
            self._tracker.observe(patterns, metrics, words)
    
    async def get_flow_state(self, slope_threshold: float = 0.1) -> FlowState:
        """Async flow state computation."""
        async with self._lock:
            return self._tracker.get_flow_state(slope_threshold)
    
    async def get_top_emerging(self, n: int = 5) -> List[Tuple[str, float]]:
        """Async top emerging patterns."""
        async with self._lock:
            return self._tracker.get_top_emerging(n)
    
    async def get_top_fading(self, n: int = 5) -> List[Tuple[str, float]]:
        """Async top fading patterns."""
        async with self._lock:
            return self._tracker.get_top_fading(n)
    
    async def trauma_correlation(self, trauma_threshold: float = 0.5) -> Dict[str, float]:
        """Async trauma correlation."""
        async with self._lock:
            return self._tracker.trauma_correlation(trauma_threshold)
    
    async def stats(self) -> Dict[str, Any]:
        """Async stats."""
        async with self._lock:
            return self._tracker.stats()


# ============================================================================
# TEST
# ============================================================================

def _test_flow():
    """Quick test of flow tracking."""
    tracker = FlowTracker()
    
    # Simulate some observations
    import random
    
    for i in range(10):
        # Random patterns with random strengths
        patterns = {
            f"pattern_{j}": random.random() * (1 + i * 0.1 if j == 0 else 1)  # pattern_0 grows
            for j in range(5)
        }
        
        metrics = {
            "entropy": random.random(),
            "coherence": random.random(),
            "trauma_level": 0.8 if i > 7 else 0.2,  # high trauma at end
        }
        
        tracker.observe(patterns, metrics)
    
    # Get flow state
    state = tracker.get_flow_state()
    
    print("=== FLOW TRACKER TEST ===")
    print(f"Total patterns: {state.total_patterns}")
    print(f"Avg strength: {state.avg_strength:.3f}")
    print(f"Flow entropy: {state.flow_entropy:.3f}")
    print(f"\nEmerging (↗): {len(state.emerging)}")
    for p, slope in state.emerging[:3]:
        print(f"  {p}: slope={slope:.3f}")
    print(f"\nFading (↘): {len(state.fading)}")
    for p, slope in state.fading[:3]:
        print(f"  {p}: slope={slope:.3f}")
    
    # Trauma correlation
    correlations = tracker.trauma_correlation()
    print("\nTrauma correlations:")
    for p, corr in sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:3]:
        print(f"  {p}: {corr:.3f}")


if __name__ == "__main__":
    _test_flow()
