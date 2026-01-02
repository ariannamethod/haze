#!/usr/bin/env python3
# rrpram.py — Recursive Resonant Pattern Recognition Attention Mechanism Tokenizer
#
# SentencePiece-based tokenization for haze.
# Captures n-grams, subwords, and resonant patterns directly in the vocabulary.
#
# Why "rrpram"? Because the tokenizer IS the first layer of pattern recognition.
# Before attention even runs, we're already finding patterns.
#
# Usage:
#   from haze.rrpram import RRPRAMVocab
#   vocab = RRPRAMVocab.train("text.txt", vocab_size=1000)
#   tokens = vocab.encode("the haze settles")
#   text = vocab.decode(tokens)

from __future__ import annotations
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Union
from dataclasses import dataclass

try:
    import sentencepiece as spm
    HAS_SENTENCEPIECE = True
except ImportError:
    HAS_SENTENCEPIECE = False
    print("[rrpram] sentencepiece not found. Install it: pip install sentencepiece")


@dataclass
class RRPRAMVocab:
    """
    RRPRAM Vocabulary: SentencePiece-based tokenizer for haze.
    
    Uses BPE or Unigram model to capture:
    - Frequent n-grams as single tokens
    - Subword patterns (morphology)
    - Resonant character sequences
    
    This is the first layer of pattern recognition—before attention,
    we're already finding structure in the text.
    """
    
    model_path: str
    sp: "spm.SentencePieceProcessor"
    vocab_size: int
    
    @classmethod
    def train(
        cls,
        corpus_path: Union[str, Path],
        vocab_size: int = 1000,
        model_type: str = "bpe",  # "bpe", "unigram", "char", "word"
        model_prefix: Optional[str] = None,
        character_coverage: float = 1.0,
        max_sentence_length: int = 4192,
        user_defined_symbols: Optional[List[str]] = None,
    ) -> "RRPRAMVocab":
        """
        Train a new SentencePiece model on corpus.
        
        Args:
            corpus_path: path to training text file
            vocab_size: target vocabulary size
            model_type: "bpe" (byte-pair), "unigram", "char", or "word"
            model_prefix: output model file prefix (default: temp file)
            character_coverage: fraction of characters to cover (1.0 = all)
            max_sentence_length: max chars per training sentence
            user_defined_symbols: custom symbols to include
        
        Returns:
            trained RRPRAMVocab instance
        """
        if not HAS_SENTENCEPIECE:
            raise ImportError("sentencepiece required. Install: pip install sentencepiece")
        
        corpus_path = Path(corpus_path)
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus not found: {corpus_path}")
        
        # determine model output path
        if model_prefix is None:
            # create temp directory for model files
            tmp_dir = tempfile.mkdtemp(prefix="rrpram_")
            model_prefix = os.path.join(tmp_dir, "rrpram")
        
        # build training command
        train_args = [
            f"--input={corpus_path}",
            f"--model_prefix={model_prefix}",
            f"--vocab_size={vocab_size}",
            f"--model_type={model_type}",
            f"--character_coverage={character_coverage}",
            f"--max_sentence_length={max_sentence_length}",
            "--pad_id=0",
            "--unk_id=1",
            "--bos_id=2",
            "--eos_id=3",
            "--normalization_rule_name=identity",  # preserve case and chars
        ]
        
        if user_defined_symbols:
            train_args.append(f"--user_defined_symbols={','.join(user_defined_symbols)}")
        
        # train
        print(f"[rrpram] training {model_type} model on {corpus_path}")
        print(f"[rrpram] vocab_size={vocab_size}, coverage={character_coverage}")
        spm.SentencePieceTrainer.Train(" ".join(train_args))
        
        model_path = f"{model_prefix}.model"
        print(f"[rrpram] model saved to {model_path}")
        
        # load trained model
        sp = spm.SentencePieceProcessor()
        sp.Load(model_path)
        
        return cls(
            model_path=model_path,
            sp=sp,
            vocab_size=sp.GetPieceSize(),
        )
    
    @classmethod
    def load(cls, model_path: Union[str, Path]) -> "RRPRAMVocab":
        """Load a pre-trained SentencePiece model."""
        if not HAS_SENTENCEPIECE:
            raise ImportError("sentencepiece required. Install: pip install sentencepiece")
        
        model_path = str(model_path)
        sp = spm.SentencePieceProcessor()
        sp.Load(model_path)
        
        return cls(
            model_path=model_path,
            sp=sp,
            vocab_size=sp.GetPieceSize(),
        )
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return self.sp.EncodeAsIds(text)
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        return self.sp.DecodeIds(ids)
    
    def encode_pieces(self, text: str) -> List[str]:
        """Encode text to subword pieces (for visualization)."""
        return self.sp.EncodeAsPieces(text)
    
    def decode_pieces(self, pieces: List[str]) -> str:
        """Decode subword pieces to text."""
        return self.sp.DecodePieces(pieces)
    
    def get_piece(self, id: int) -> str:
        """Get the piece (token) for a given ID."""
        return self.sp.IdToPiece(id)
    
    def get_id(self, piece: str) -> int:
        """Get the ID for a given piece (token)."""
        return self.sp.PieceToId(piece)
    
    def __len__(self) -> int:
        return self.vocab_size


def analyze_vocab(vocab: RRPRAMVocab, top_n: int = 50) -> None:
    """
    Analyze and display vocabulary statistics.
    
    Shows the most common tokens (patterns) learned by the tokenizer.
    These are the "resonant patterns" that appear frequently in the corpus.
    """
    print("=" * 60)
    print("  RRPRAM Vocabulary Analysis")
    print("=" * 60)
    print(f"  vocab size: {vocab.vocab_size}")
    print()
    
    print(f"  Top {top_n} tokens (resonant patterns):")
    print("-" * 40)
    
    for i in range(min(top_n, vocab.vocab_size)):
        piece = vocab.get_piece(i)
        # visualize special chars
        display = piece.replace("▁", "_").replace("\n", "\\n")
        print(f"  {i:4d}: '{display}'")
    
    print()
    print("=" * 60)


def demo_tokenization(vocab: RRPRAMVocab, texts: List[str]) -> None:
    """
    Demo tokenization on sample texts.
    
    Shows how the RRPRAM tokenizer breaks down text into patterns.
    """
    print("=" * 60)
    print("  RRPRAM Tokenization Demo")
    print("=" * 60)
    
    for text in texts:
        print(f"\n  input: \"{text}\"")
        ids = vocab.encode(text)
        pieces = vocab.encode_pieces(text)
        
        print(f"  ids:   {ids}")
        print(f"  pieces: {pieces}")
        print(f"  tokens: {len(ids)}")
        
        # show reconstruction
        reconstructed = vocab.decode(ids)
        print(f"  decoded: \"{reconstructed}\"")
    
    print()
    print("=" * 60)


if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("  rrpram.py — RRPRAM Tokenizer")
    print("=" * 60)
    print()
    
    # check if corpus exists
    corpus_path = Path("text.txt")
    if not corpus_path.exists():
        print("[error] text.txt not found")
        print()
        print("Usage:")
        print("  python rrpram.py           # train on text.txt")
        print("  python rrpram.py corpus.txt  # train on custom corpus")
        sys.exit(1)
    
    if len(sys.argv) > 1:
        corpus_path = Path(sys.argv[1])
    
    print(f"[rrpram] corpus: {corpus_path}")
    
    # train tokenizer
    vocab = RRPRAMVocab.train(
        corpus_path,
        vocab_size=500,
        model_type="bpe",
        character_coverage=1.0,
    )
    
    # analyze
    analyze_vocab(vocab, top_n=30)
    
    # demo
    demo_texts = [
        "the haze settles",
        "darling",
        "I love you",
        "What's the toast?",
    ]
    demo_tokenization(vocab, demo_texts)
    
    print()
    print("[rrpram] done. patterns recognized. resonance achieved.")
