#!/usr/bin/env python3
# tests/test_haze.py — Tests for model.py module

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import model as haze
import nn


class TestVocab(unittest.TestCase):
    """Test Vocab class."""
    
    def test_from_text(self):
        """Test vocabulary creation from text."""
        text = "hello world"
        vocab = haze.Vocab.from_text(text)
        self.assertIsInstance(vocab, haze.Vocab)
        self.assertGreater(vocab.vocab_size, 0)
    
    def test_encode_decode(self):
        """Test encode/decode round-trip."""
        text = "hello world"
        vocab = haze.Vocab.from_text(text)
        encoded = vocab.encode("hello")
        decoded = vocab.decode(encoded)
        self.assertEqual(decoded, "hello")
    
    def test_lowercase_conversion(self):
        """Test that vocab converts to lowercase."""
        text = "Hello World"
        vocab = haze.Vocab.from_text(text)
        encoded = vocab.encode("HELLO")
        decoded = vocab.decode(encoded)
        self.assertEqual(decoded, "hello")
    
    def test_unknown_chars(self):
        """Test handling of unknown characters."""
        text = "abc"
        vocab = haze.Vocab.from_text(text)
        # 'x' is not in vocab
        encoded = vocab.encode("x")
        self.assertEqual(len(encoded), 0)
    
    def test_vocab_size(self):
        """Test vocab size calculation."""
        text = "aabbcc"
        vocab = haze.Vocab.from_text(text)
        self.assertEqual(vocab.vocab_size, 3)  # a, b, c


class TestReweightHead(unittest.TestCase):
    """Test ReweightHead attention."""
    
    def setUp(self):
        self.rng = nn.get_rng(42)
        self.n_emb = 16
        self.head_dim = 8
        self.T = 10
        self.head = haze.ReweightHead(self.n_emb, self.head_dim, self.T, self.rng)
    
    def test_forward_shape(self):
        """Test forward pass returns correct shape."""
        x = np.random.randn(self.T, self.n_emb).astype(np.float32)
        out = self.head.forward(x)
        self.assertEqual(out.shape, (self.T, self.head_dim))
    
    def test_forward_shorter_sequence(self):
        """Test forward with sequence shorter than T."""
        x = np.random.randn(5, self.n_emb).astype(np.float32)
        out = self.head.forward(x)
        self.assertEqual(out.shape, (5, self.head_dim))


class TestContentHead(unittest.TestCase):
    """Test ContentHead attention."""
    
    def setUp(self):
        self.rng = nn.get_rng(42)
        self.n_emb = 16
        self.head_dim = 8
        self.T = 10
        self.head = haze.ContentHead(self.n_emb, self.head_dim, self.T, self.rng)
    
    def test_forward_shape(self):
        """Test forward pass returns correct shape."""
        x = np.random.randn(self.T, self.n_emb).astype(np.float32)
        out = self.head.forward(x)
        self.assertEqual(out.shape, (self.T, self.head_dim))
    
    def test_forward_shorter_sequence(self):
        """Test forward with sequence shorter than T."""
        x = np.random.randn(5, self.n_emb).astype(np.float32)
        out = self.head.forward(x)
        self.assertEqual(out.shape, (5, self.head_dim))


class TestHybridHead(unittest.TestCase):
    """Test HybridHead attention."""
    
    def setUp(self):
        self.rng = nn.get_rng(42)
        self.n_emb = 16
        self.head_dim = 8
        self.T = 10
        self.head = haze.HybridHead(self.n_emb, self.head_dim, self.T, self.rng, alpha=0.5)
    
    def test_forward_shape(self):
        """Test forward pass returns correct shape."""
        x = np.random.randn(self.T, self.n_emb).astype(np.float32)
        out = self.head.forward(x)
        self.assertEqual(out.shape, (self.T, self.head_dim))
    
    def test_alpha_parameter(self):
        """Test alpha parameter is stored."""
        self.assertEqual(self.head.alpha, 0.5)


class TestBlock(unittest.TestCase):
    """Test transformer Block."""
    
    def setUp(self):
        self.rng = nn.get_rng(42)
        self.n_emb = 32
        self.T = 10
        self.nodes = 64
    
    def test_block_hybrid_forward(self):
        """Test hybrid block forward pass."""
        block = haze.Block(
            self.n_emb, self.T, self.nodes, self.rng,
            n_heads=4, head_type="hybrid"
        )
        x = np.random.randn(self.T, self.n_emb).astype(np.float32)
        out = block.forward(x)
        self.assertEqual(out.shape, (self.T, self.n_emb))
    
    def test_block_reweight_forward(self):
        """Test reweight-only block forward pass."""
        block = haze.Block(
            self.n_emb, self.T, self.nodes, self.rng,
            n_heads=4, head_type="reweight"
        )
        x = np.random.randn(self.T, self.n_emb).astype(np.float32)
        out = block.forward(x)
        self.assertEqual(out.shape, (self.T, self.n_emb))
    
    def test_block_content_forward(self):
        """Test content-only block forward pass."""
        block = haze.Block(
            self.n_emb, self.T, self.nodes, self.rng,
            n_heads=4, head_type="content"
        )
        x = np.random.randn(self.T, self.n_emb).astype(np.float32)
        out = block.forward(x)
        self.assertEqual(out.shape, (self.T, self.n_emb))


class TestReweightGPT(unittest.TestCase):
    """Test ReweightGPT model."""
    
    def setUp(self):
        self.vocab_size = 20
        self.T = 16
        self.n_emb = 32
        self.model = haze.ReweightGPT(
            vocab_size=self.vocab_size,
            T=self.T,
            n_emb=self.n_emb,
            nodes=32,
            n_blocks=2,
            n_heads=4,
            head_type="hybrid",
            seed=42,
        )
    
    def test_model_initialization(self):
        """Test model initializes correctly."""
        self.assertEqual(self.model.vocab_size, self.vocab_size)
        self.assertEqual(self.model.T, self.T)
        self.assertEqual(self.model.n_emb, self.n_emb)
    
    def test_logits_shape(self):
        """Test logits output has correct shape."""
        idx_seq = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        logits = self.model.logits(idx_seq)
        self.assertEqual(logits.shape, (5, self.vocab_size))
    
    def test_generate_simple(self):
        """Test simple generation."""
        seed_seq = [0, 1, 2]
        tokens = self.model.generate_simple(seed_seq, length=10, temperature=1.0)
        self.assertEqual(len(tokens), 10)
        # Check all tokens are valid
        for token in tokens:
            self.assertGreaterEqual(token, 0)
            self.assertLess(token, self.vocab_size)
    
    def test_generate_with_stats(self):
        """Test generation with statistics."""
        seed_seq = [0, 1, 2]
        tokens, stats = self.model.generate(
            seed_seq,
            length=10,
            temperature=1.0,
            sampling="basic"
        )
        self.assertEqual(len(tokens), 10)
        self.assertIn("mean_entropy", stats)
        self.assertIn("mean_confidence", stats)
        self.assertIn("mean_temp", stats)
    
    def test_generate_entropy_sampling(self):
        """Test entropy-aware sampling."""
        seed_seq = [0, 1, 2]
        tokens, stats = self.model.generate(
            seed_seq,
            length=10,
            sampling="entropy",
            target_entropy=2.0
        )
        self.assertEqual(len(tokens), 10)
        self.assertGreater(stats["mean_entropy"], 0)
    
    def test_generate_top_k(self):
        """Test top-k sampling."""
        seed_seq = [0, 1, 2]
        tokens, _ = self.model.generate(
            seed_seq,
            length=10,
            sampling="top_k",
            top_k=5,
            temperature=1.0
        )
        self.assertEqual(len(tokens), 10)
    
    def test_generate_top_p(self):
        """Test top-p nucleus sampling."""
        seed_seq = [0, 1, 2]
        tokens, _ = self.model.generate(
            seed_seq,
            length=10,
            sampling="top_p",
            top_p=0.9,
            temperature=1.0
        )
        self.assertEqual(len(tokens), 10)
    
    def test_generate_mirostat(self):
        """Test mirostat v1 sampling."""
        seed_seq = [0, 1, 2]
        tokens, stats = self.model.generate(
            seed_seq,
            length=10,
            sampling="mirostat",
            target_entropy=2.0,
            mirostat_tau=0.1
        )
        self.assertEqual(len(tokens), 10)
        self.assertIn("mean_entropy", stats)
    
    def test_generate_mirostat_v2(self):
        """Test mirostat v2 sampling."""
        seed_seq = [0, 1, 2]
        tokens, stats = self.model.generate(
            seed_seq,
            length=10,
            sampling="mirostat_v2",
            target_entropy=2.0,
            mirostat_tau=0.1
        )
        self.assertEqual(len(tokens), 10)
        self.assertIn("mean_entropy", stats)
    
    def test_generate_resonance(self):
        """Test resonance-based sampling."""
        seed_seq = [0, 1, 2]
        tokens, stats = self.model.generate(
            seed_seq,
            length=20,
            sampling="resonance",
            target_resonance=0.7
        )
        self.assertEqual(len(tokens), 20)
        self.assertIn("mean_resonance", stats)
        self.assertGreater(stats["mean_resonance"], 0)
        self.assertLess(stats["mean_resonance"], 1.0)
    
    def test_generate_empty_seed(self):
        """Test generation with empty seed."""
        tokens, _ = self.model.generate(
            seed_seq=[],
            length=10,
            temperature=1.0
        )
        self.assertEqual(len(tokens), 10)
    
    def test_save_and_load_theweightofhaze(self):
        """Test saving and loading model weights."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            temp_path = f.name
        
        try:
            # save weights
            self.model.save_theweightofhaze(temp_path)
            self.assertTrue(os.path.exists(temp_path))
            
            # load weights
            loaded_model = haze.ReweightGPT.theweightofhaze(
                vocab_size=self.vocab_size,
                path=temp_path
            )
            
            # verify structure
            self.assertEqual(loaded_model.vocab_size, self.vocab_size)
            self.assertEqual(loaded_model.T, self.T)
            self.assertEqual(loaded_model.n_emb, self.n_emb)
            
            # test that loaded model can generate
            tokens, _ = loaded_model.generate(
                seed_seq=[0, 1, 2],
                length=5,
                temperature=1.0
            )
            self.assertEqual(len(tokens), 5)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestModelVariants(unittest.TestCase):
    """Test different model configurations."""
    
    def test_reweight_only_model(self):
        """Test model with only reweight heads."""
        model = haze.ReweightGPT(
            vocab_size=20,
            T=16,
            n_emb=32,
            nodes=32,
            n_blocks=2,
            n_heads=4,
            head_type="reweight",
            seed=42,
        )
        idx_seq = np.array([0, 1, 2], dtype=np.int32)
        logits = model.logits(idx_seq)
        self.assertEqual(logits.shape, (3, 20))
    
    def test_content_only_model(self):
        """Test model with only content heads."""
        model = haze.ReweightGPT(
            vocab_size=20,
            T=16,
            n_emb=32,
            nodes=32,
            n_blocks=2,
            n_heads=4,
            head_type="content",
            seed=42,
        )
        idx_seq = np.array([0, 1, 2], dtype=np.int32)
        logits = model.logits(idx_seq)
        self.assertEqual(logits.shape, (3, 20))
    
    def test_hybrid_model(self):
        """Test model with hybrid heads."""
        model = haze.ReweightGPT(
            vocab_size=20,
            T=16,
            n_emb=32,
            nodes=32,
            n_blocks=2,
            n_heads=4,
            head_type="hybrid",
            alpha=0.7,
            seed=42,
        )
        idx_seq = np.array([0, 1, 2], dtype=np.int32)
        logits = model.logits(idx_seq)
        self.assertEqual(logits.shape, (3, 20))


class TestHelpers(unittest.TestCase):
    """Test helper functions."""
    
    def test_load_corpus(self):
        """Test corpus loading."""
        # Create a temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("test corpus")
            temp_path = f.name
        
        try:
            corpus = haze.load_corpus(temp_path)
            self.assertEqual(corpus, "test corpus")
        finally:
            os.remove(temp_path)
    
    def test_build_model_from_text(self):
        """Test building model from text file."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("hello world this is a test")
            temp_path = f.name
        
        try:
            text, vocab, model = haze.build_model_from_text(
                temp_path,
                T=16,
                n_emb=32,
                nodes=32,
                n_blocks=2,
                n_heads=4,
            )
            self.assertIsInstance(text, str)
            self.assertIsInstance(vocab, haze.Vocab)
            self.assertIsInstance(model, haze.ReweightGPT)
            self.assertEqual(model.vocab_size, vocab.vocab_size)
        finally:
            os.remove(temp_path)


class TestEndToEnd(unittest.TestCase):
    """End-to-end integration tests."""
    
    def test_full_pipeline(self):
        """Test complete text generation pipeline."""
        # Create corpus
        text = "the quick brown fox jumps over the lazy dog"
        vocab = haze.Vocab.from_text(text)
        
        # Build model
        model = haze.ReweightGPT(
            vocab_size=vocab.vocab_size,
            T=16,
            n_emb=32,
            nodes=32,
            n_blocks=2,
            n_heads=4,
            head_type="hybrid",
            seed=42,
        )
        
        # Generate text
        seed_text = "the"
        seed_idx = vocab.encode(seed_text)
        tokens, stats = model.generate(
            seed_seq=seed_idx,
            length=20,
            sampling="entropy",
            target_entropy=2.0
        )
        
        # Decode
        generated = vocab.decode(tokens)
        
        # Verify
        self.assertEqual(len(tokens), 20)
        self.assertIsInstance(generated, str)
        self.assertGreater(len(generated), 0)
        self.assertIn("mean_entropy", stats)


if __name__ == "__main__":
    unittest.main()
