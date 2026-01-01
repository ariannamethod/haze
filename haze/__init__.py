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
]
