# experts.py — Resonant Experts: MOE-style temperature routing
#
# Inspired by Leo's resonant experts, but reimagined for haze:
# - No fixed routing, always a MIXTURE of all experts
# - Weights computed from entropy, arousal, novelty
# - Each expert has a temperature and semantic weight
#
# The final temperature is a weighted blend, not a single expert choice.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Tuple
import math


@dataclass
class Expert:
    """A resonant expert - a perspective on the field."""
    name: str
    temperature: float
    semantic_weight: float
    description: str


# The four experts (inspired by Leo)
EXPERTS = [
    Expert(
        name="structural",
        temperature=0.7,
        semantic_weight=0.2,
        description="Grammar-focused, coherent structure",
    ),
    Expert(
        name="semantic",
        temperature=0.9,
        semantic_weight=0.5,
        description="Meaning-focused, thematic coherence",
    ),
    Expert(
        name="creative",
        temperature=1.2,
        semantic_weight=0.4,
        description="Exploratory, high entropy drift",
    ),
    Expert(
        name="precise",
        temperature=0.5,
        semantic_weight=0.3,
        description="Conservative, low entropy grounding",
    ),
]


class ExpertMixture(NamedTuple):
    """Result of expert routing - a weighted mixture."""
    temperature: float
    semantic_weight: float
    weights: Dict[str, float]  # name -> weight
    dominant: str  # name of highest-weighted expert


class FieldSignals(NamedTuple):
    """Input signals for expert routing."""
    entropy: float      # 0-1: distribution entropy (how spread the choices are)
    arousal: float      # 0-1: emotional charge
    novelty: float      # 0-1: how new/unknown the input is
    perplexity: float   # 0-inf: model uncertainty (optional, default 1.0)


def compute_expert_weights(signals: FieldSignals) -> Dict[str, float]:
    """
    Compute expert weights from field signals.
    
    This is the core MOE logic, but always returns a MIXTURE:
    - High entropy → more creative weight
    - Low entropy → more precise weight
    - High arousal → more semantic weight
    - High novelty → more structural weight (ground in known patterns)
    - High perplexity → more precise weight (reduce uncertainty)
    """
    weights = {}
    
    # Base weights (all experts always contribute)
    base = 0.1
    
    # Structural: grounded in known patterns
    # Higher when novelty is high (need to ground in familiar)
    # Also higher when perplexity is moderate
    structural = base + 0.3 * signals.novelty + 0.1 * (1.0 - signals.arousal)
    weights["structural"] = structural
    
    # Semantic: meaning-focused
    # Higher when arousal is high (emotional content)
    # Also higher when entropy is moderate (not too chaotic)
    semantic = base + 0.4 * signals.arousal + 0.2 * (1.0 - abs(signals.entropy - 0.5) * 2)
    weights["semantic"] = semantic
    
    # Creative: exploratory
    # Higher when entropy is high (explore the space)
    # Lower when novelty is high (don't go too far from known)
    creative = base + 0.4 * signals.entropy + 0.2 * (1.0 - signals.novelty)
    weights["creative"] = creative
    
    # Precise: conservative
    # Higher when entropy is low (stay grounded)
    # Higher when perplexity is high (reduce uncertainty)
    perp_factor = min(1.0, signals.perplexity / 2.0)  # Normalize perplexity
    precise = base + 0.3 * (1.0 - signals.entropy) + 0.3 * perp_factor
    weights["precise"] = precise
    
    # Normalize weights to sum to 1.0
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    
    return weights


def blend_experts(weights: Dict[str, float]) -> ExpertMixture:
    """
    Blend expert parameters using weights.
    
    Returns a mixture of temperature and semantic_weight.
    """
    expert_map = {e.name: e for e in EXPERTS}
    
    temp = 0.0
    sem = 0.0
    
    for name, weight in weights.items():
        expert = expert_map.get(name)
        if expert:
            temp += expert.temperature * weight
            sem += expert.semantic_weight * weight
    
    # Find dominant expert
    dominant = max(weights.items(), key=lambda x: x[1])[0]
    
    return ExpertMixture(
        temperature=temp,
        semantic_weight=sem,
        weights=weights,
        dominant=dominant,
    )


def route_to_mixture(signals: FieldSignals) -> ExpertMixture:
    """
    Main entry point: compute expert mixture from field signals.
    
    Usage:
        signals = FieldSignals(entropy=0.6, arousal=0.3, novelty=0.2, perplexity=1.0)
        mixture = route_to_mixture(signals)
        # mixture.temperature → blended temp
        # mixture.weights → {"structural": 0.2, "semantic": 0.3, ...}
    """
    weights = compute_expert_weights(signals)
    return blend_experts(weights)


def route_single_expert(signals: FieldSignals) -> Expert:
    """
    Leo-style routing: pick the single best expert.
    
    Useful for simpler cases or A/B testing.
    """
    weights = compute_expert_weights(signals)
    dominant = max(weights.items(), key=lambda x: x[1])[0]
    expert_map = {e.name: e for e in EXPERTS}
    return expert_map[dominant]


# Convenience function for simple pulse-based routing
def pulse_to_signals(
    novelty: float = 0.0,
    arousal: float = 0.0,
    entropy: float = 0.5,
) -> FieldSignals:
    """Convert pulse metrics to FieldSignals."""
    return FieldSignals(
        entropy=max(0.0, min(1.0, entropy)),
        arousal=max(0.0, min(1.0, arousal)),
        novelty=max(0.0, min(1.0, novelty)),
        perplexity=1.0,
    )


def describe_mixture(mixture: ExpertMixture) -> str:
    """Human-readable description of expert mixture."""
    parts = []
    for name, weight in sorted(mixture.weights.items(), key=lambda x: -x[1]):
        pct = int(weight * 100)
        if pct > 0:
            parts.append(f"{name}:{pct}%")
    return f"temp={mixture.temperature:.2f} [{', '.join(parts)}]"


# Test when run directly
if __name__ == "__main__":
    print("=== Resonant Experts Demo ===\n")
    
    test_cases = [
        ("neutral", FieldSignals(entropy=0.5, arousal=0.5, novelty=0.5, perplexity=1.0)),
        ("high entropy", FieldSignals(entropy=0.9, arousal=0.3, novelty=0.2, perplexity=1.0)),
        ("low entropy", FieldSignals(entropy=0.1, arousal=0.2, novelty=0.3, perplexity=1.0)),
        ("high arousal", FieldSignals(entropy=0.5, arousal=0.9, novelty=0.3, perplexity=1.0)),
        ("high novelty", FieldSignals(entropy=0.5, arousal=0.3, novelty=0.9, perplexity=1.0)),
        ("high perplexity", FieldSignals(entropy=0.5, arousal=0.3, novelty=0.3, perplexity=3.0)),
    ]
    
    for name, signals in test_cases:
        mixture = route_to_mixture(signals)
        print(f"{name}:")
        print(f"  signals: entropy={signals.entropy:.1f} arousal={signals.arousal:.1f} novelty={signals.novelty:.1f}")
        print(f"  mixture: {describe_mixture(mixture)}")
        print(f"  dominant: {mixture.dominant}")
        print()
