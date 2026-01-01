#!/usr/bin/env python3
# haze/__init__.py — package initialization

from .model import (
    Vocab,
    ReweightGPT,
    ReweightHead,
    ContentHead,
    HybridHead,
    Block,
    load_corpus,
    build_model_from_text,
)

# Alias for the new naming convention
Haze = ReweightGPT

__all__ = [
    'Vocab',
    'ReweightGPT',
    'Haze',
    'ReweightHead',
    'ContentHead',
    'HybridHead',
    'Block',
    'load_corpus',
    'build_model_from_text',
]
