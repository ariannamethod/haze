#!/usr/bin/env python3
# haze/__init__.py — package initialization
#
# Haze: Hybrid Attention Entropy System
# Part of the Arianna Method
#
# Key modules:
#   - haze.py: PostGPT model and vocab
#   - cooccur.py: Co-occurrence field for resonance
#   - subjectivity.py: Identity infusion, no seed from prompt
#   - overthinking.py: Three rings of private reflection
#   - lexicon.py: Dynamic vocabulary growth
#   - async_haze.py: Complete async field organism
#   - cleanup.py: Output cleanup
#   - rrpram.py: SentencePiece tokenizer

from .haze import (
    Vocab,
    PostGPT,
    RRPRAMHead,
    ReweightHead,  # backwards compat alias
    ContentHead,
    HybridHead,
    Block,
    load_corpus,
    build_model_from_text,
)

# Import co-occurrence field
from .cooccur import CooccurField

# Import subjectivity (no seed from prompt)
from .subjectivity import Subjectivity, AsyncSubjectivity, PulseSnapshot, HazeIdentity

# Import overthinking (three rings)
from .overthinking import Overthinking, AsyncOverthinking, Ring, RingsSnapshot

# Import lexicon (dynamic growth)
from .lexicon import Lexicon, AsyncLexicon, LexiconStats

# Import resonant experts (MOE-style temperature routing)
from .experts import (
    Expert, EXPERTS, ExpertMixture, FieldSignals,
    route_to_mixture, route_single_expert, pulse_to_signals, describe_mixture
)

# Import trauma (resonant words return to identity)
from .trauma import (
    Trauma, AsyncTrauma, TraumaState, TraumaInfluence,
    compute_trauma_influence, get_identity_prefix, HAZE_BOOTSTRAP
)

# Import async haze field
from .async_haze import AsyncHazeField, HazeResponse

# Import cleanup
from .cleanup import cleanup_output, cleanup_dialogue, calculate_garbage_score

# Import RRPRAM tokenizer if sentencepiece available
try:
    from .rrpram import RRPRAMVocab, analyze_vocab, demo_tokenization
    HAS_RRPRAM = True
except ImportError:
    HAS_RRPRAM = False

# Import SubwordField if sentencepiece available
try:
    from .subword_field import SubwordField, AsyncSubwordField
    HAS_SUBWORD = True
except ImportError:
    HAS_SUBWORD = False

# Import MathBrain (async MLP for field perception)
try:
    from .mathbrain import MathBrain, AsyncMathBrain, FieldPerception
    HAS_MATHBRAIN = True
except ImportError:
    HAS_MATHBRAIN = False

# Import MetaHaze (dual generation, self-curation — Haze's inner voice)
from .metahaze import (
    MetaHaze, AsyncMetaHaze, MetaConfig,
    GenerationCandidate, MetaResponse, METAHAZE_BOOTSTRAP
)

# Import Bridges (statistical trajectory learning)
from .bridges import (
    GenerationMode, EpisodeStep, Episode as BridgeEpisode,
    TransitionStat, TransitionGraph, EpisodeLogger,
    BridgeCandidate, BridgeMemory, AsyncBridgeManager,
)

# Import Flow (pattern flow through time)
from .flow import (
    PatternSnapshot, PatternTrajectory, FlowState,
    FlowTracker, AsyncFlowTracker,
)

# Import Episodes (episodic memory — Self-RAG)
from .episodes import (
    HazeMetrics, Episode, EpisodicMemory, AsyncEpisodicMemory,
    suggest_from_episodes,
)

# Import DrunkSanta (resonant recall — Haze's memory of best moments)
from .drunksanta import (
    DrunkSanta, AsyncDrunkSanta, Snapshot, ResonanceContext,
    DRUNK_FACTOR, RECENCY_WINDOW_HOURS,
)

# Backwards compatibility aliases
Haze = PostGPT
ReweightGPT = PostGPT

__all__ = [
    # Core model
    'Vocab',
    'PostGPT',
    'Haze',  # alias
    'ReweightGPT',  # backwards compat
    'RRPRAMHead',
    'ReweightHead',  # backwards compat alias for RRPRAMHead
    'ContentHead',
    'HybridHead',
    'Block',
    'load_corpus',
    'build_model_from_text',
    # Co-occurrence field
    'CooccurField',
    # Subjectivity (no seed from prompt)
    'Subjectivity',
    'AsyncSubjectivity',
    'PulseSnapshot',
    'HazeIdentity',
    # Overthinking (three rings)
    'Overthinking',
    'AsyncOverthinking',
    'Ring',
    'RingsSnapshot',
    # Lexicon (dynamic growth)
    'Lexicon',
    'AsyncLexicon',
    'LexiconStats',
    # Resonant Experts (MOE-style temperature routing)
    'Expert',
    'EXPERTS',
    'ExpertMixture',
    'FieldSignals',
    'route_to_mixture',
    'route_single_expert',
    'pulse_to_signals',
    'describe_mixture',
    # Trauma (resonant words return to identity)
    'Trauma',
    'AsyncTrauma',
    'TraumaState',
    'TraumaInfluence',
    'compute_trauma_influence',
    'get_identity_prefix',
    'HAZE_BOOTSTRAP',
    # Async haze field
    'AsyncHazeField',
    'HazeResponse',
    # Cleanup
    'cleanup_output',
    'cleanup_dialogue',
    'calculate_garbage_score',
    # RRPRAM tokenizer (if available)
    'RRPRAMVocab',
    'HAS_RRPRAM',
    # SubwordField (BPE-based generation) - THE BREAKTHROUGH!
    'SubwordField',
    'AsyncSubwordField',
    'HAS_SUBWORD',
    # MathBrain (field perception)
    'MathBrain',
    'AsyncMathBrain',
    'FieldPerception',
    'HAS_MATHBRAIN',
    # MetaHaze (inner voice, self-curation) - inspired by Leo's MetaLeo
    'MetaHaze',
    'AsyncMetaHaze',
    'MetaConfig',
    'GenerationCandidate',
    'MetaResponse',
    'METAHAZE_BOOTSTRAP',
    # Bridges (statistical trajectory learning) - inspired by Leo's Phase 4
    'GenerationMode',
    'EpisodeStep',
    'BridgeEpisode',
    'TransitionStat',
    'TransitionGraph',
    'EpisodeLogger',
    'BridgeCandidate',
    'BridgeMemory',
    'AsyncBridgeManager',
    # Flow (pattern flow through time) - inspired by Leo's gowiththeflow
    'PatternSnapshot',
    'PatternTrajectory',
    'FlowState',
    'FlowTracker',
    'AsyncFlowTracker',
    # Episodes (episodic memory, Self-RAG) - inspired by Leo's episodes
    'HazeMetrics',
    'Episode',
    'EpisodicMemory',
    'AsyncEpisodicMemory',
    'suggest_from_episodes',
    # DrunkSanta (resonant recall) - inspired by Leo's SantaClaus 🍷🎅
    'DrunkSanta',
    'AsyncDrunkSanta',
    'Snapshot',
    'ResonanceContext',
    'DRUNK_FACTOR',
    'RECENCY_WINDOW_HOURS',
]
