#!/usr/bin/env python3
# tests/test_nn.py — Tests for nn.py module

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nn


class TestRNG(unittest.TestCase):
    """Test random number generator utilities."""
    
    def test_get_rng_no_seed(self):
        """Test RNG creation without seed."""
        rng = nn.get_rng()
        self.assertIsInstance(rng, np.random.Generator)
    
    def test_get_rng_with_seed(self):
        """Test RNG creation with seed produces reproducible results."""
        rng1 = nn.get_rng(42)
        rng2 = nn.get_rng(42)
        val1 = rng1.random()
        val2 = rng2.random()
        self.assertEqual(val1, val2)


class TestWeightInit(unittest.TestCase):
    """Test weight initialization functions."""
    
    def setUp(self):
        self.rng = nn.get_rng(42)
    
    def test_init_weight_shape(self):
        """Test that init_weight returns correct shape."""
        shape = (10, 20)
        w = nn.init_weight(shape, self.rng)
        self.assertEqual(w.shape, shape)
        self.assertEqual(w.dtype, np.float32)
    
    def test_init_weight_scale(self):
        """Test that init_weight respects scale parameter."""
        shape = (100, 100)
        scale = 0.01
        w = nn.init_weight(shape, self.rng, scale=scale)
        # Check that std is approximately equal to scale
        self.assertLess(abs(w.std() - scale), 0.01)
    
    def test_init_weight_orthogonal_shape(self):
        """Test orthogonal initialization returns correct shape."""
        shape = (10, 20)
        w = nn.init_weight_orthogonal(shape, self.rng)
        self.assertEqual(w.shape, shape)
        self.assertEqual(w.dtype, np.float32)


class TestActivations(unittest.TestCase):
    """Test activation functions."""
    
    def test_relu_positive(self):
        """Test ReLU on positive values."""
        x = np.array([1.0, 2.0, 3.0])
        y = nn.relu(x)
        np.testing.assert_array_equal(y, x)
    
    def test_relu_negative(self):
        """Test ReLU on negative values."""
        x = np.array([-1.0, -2.0, -3.0])
        y = nn.relu(x)
        np.testing.assert_array_equal(y, np.zeros_like(x))
    
    def test_relu_mixed(self):
        """Test ReLU on mixed values."""
        x = np.array([-1.0, 0.0, 1.0])
        y = nn.relu(x)
        np.testing.assert_array_equal(y, np.array([0.0, 0.0, 1.0]))
    
    def test_leaky_relu(self):
        """Test leaky ReLU."""
        x = np.array([-1.0, 0.0, 1.0])
        y = nn.leaky_relu(x, alpha=0.01)
        expected = np.array([-0.01, 0.0, 1.0])
        np.testing.assert_array_almost_equal(y, expected)
    
    def test_gelu_shape(self):
        """Test GELU preserves shape."""
        x = np.random.randn(10, 20)
        y = nn.gelu(x)
        self.assertEqual(y.shape, x.shape)
    
    def test_swish_shape(self):
        """Test Swish preserves shape."""
        x = np.random.randn(10, 20)
        y = nn.swish(x)
        self.assertEqual(y.shape, x.shape)
    
    def test_sigmoid_range(self):
        """Test sigmoid output is in [0, 1]."""
        x = np.random.randn(100)
        y = nn.sigmoid(x)
        self.assertTrue(np.all(y >= 0))
        self.assertTrue(np.all(y <= 1))
    
    def test_sigmoid_zero(self):
        """Test sigmoid(0) = 0.5."""
        y = nn.sigmoid(np.array([0.0]))
        np.testing.assert_almost_equal(y[0], 0.5)
    
    def test_softmax_sum_to_one(self):
        """Test softmax outputs sum to 1."""
        x = np.random.randn(10)
        y = nn.softmax(x)
        np.testing.assert_almost_equal(y.sum(), 1.0)
    
    def test_softmax_positive(self):
        """Test softmax outputs are positive."""
        x = np.random.randn(10)
        y = nn.softmax(x)
        self.assertTrue(np.all(y > 0))


class TestNormalization(unittest.TestCase):
    """Test normalization functions."""
    
    def test_layer_norm_shape(self):
        """Test layer norm preserves shape."""
        x = np.random.randn(5, 10).astype(np.float32)
        gamma = np.ones(10, dtype=np.float32)
        beta = np.zeros(10, dtype=np.float32)
        y = nn.layer_norm(x, gamma, beta)
        self.assertEqual(y.shape, x.shape)
    
    def test_layer_norm_mean_zero(self):
        """Test layer norm produces zero mean."""
        x = np.random.randn(5, 10).astype(np.float32)
        gamma = np.ones(10, dtype=np.float32)
        beta = np.zeros(10, dtype=np.float32)
        y = nn.layer_norm(x, gamma, beta)
        means = y.mean(axis=-1)
        np.testing.assert_array_almost_equal(means, np.zeros(5), decimal=5)
    
    def test_rms_norm_shape(self):
        """Test RMS norm preserves shape."""
        x = np.random.randn(5, 10).astype(np.float32)
        gamma = np.ones(10, dtype=np.float32)
        y = nn.rms_norm(x, gamma)
        self.assertEqual(y.shape, x.shape)


class TestSampling(unittest.TestCase):
    """Test sampling functions."""
    
    def setUp(self):
        self.rng = nn.get_rng(42)
        self.logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    def test_sample_basic_greedy(self):
        """Test basic sampling with temperature=0 is greedy."""
        token = nn.sample_basic(self.logits, temperature=0.0, rng=self.rng)
        self.assertEqual(token, 4)  # argmax of logits
    
    def test_sample_basic_returns_valid_index(self):
        """Test basic sampling returns valid token index."""
        token = nn.sample_basic(self.logits, temperature=1.0, rng=self.rng)
        self.assertIsInstance(token, int)
        self.assertGreaterEqual(token, 0)
        self.assertLess(token, len(self.logits))
    
    def test_sample_top_k_greedy(self):
        """Test top-k sampling with temperature=0 is greedy."""
        token = nn.sample_top_k(self.logits, k=3, temperature=0.0, rng=self.rng)
        self.assertEqual(token, 4)  # argmax
    
    def test_sample_top_k_valid(self):
        """Test top-k sampling returns valid index."""
        token = nn.sample_top_k(self.logits, k=3, temperature=1.0, rng=self.rng)
        self.assertIsInstance(token, int)
        self.assertGreaterEqual(token, 0)
        self.assertLess(token, len(self.logits))
    
    def test_sample_top_p_greedy(self):
        """Test top-p sampling with temperature=0 is greedy."""
        token = nn.sample_top_p(self.logits, p=0.9, temperature=0.0, rng=self.rng)
        self.assertEqual(token, 4)  # argmax
    
    def test_sample_top_p_valid(self):
        """Test top-p sampling returns valid index."""
        token = nn.sample_top_p(self.logits, p=0.9, temperature=1.0, rng=self.rng)
        self.assertIsInstance(token, int)
        self.assertGreaterEqual(token, 0)
        self.assertLess(token, len(self.logits))
    
    def test_sample_mirostat_returns_tuple(self):
        """Test mirostat returns (token, new_mu)."""
        result = nn.sample_mirostat(
            self.logits,
            target_entropy=2.0,
            tau=0.1,
            mu=5.0,
            rng=self.rng
        )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        token, new_mu = result
        self.assertIsInstance(token, int)
        self.assertIsInstance(new_mu, float)
    
    def test_sample_mirostat_v2_returns_tuple(self):
        """Test mirostat v2 returns (token, new_mu)."""
        result = nn.sample_mirostat_v2(
            self.logits,
            target_entropy=2.0,
            tau=0.1,
            mu=5.0,
            rng=self.rng
        )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        token, new_mu = result
        self.assertIsInstance(token, int)
        self.assertIsInstance(new_mu, float)
    
    def test_sample_mirostat_v2_clips_mu(self):
        """Test mirostat v2 clips mu to reasonable range."""
        target_entropy = 2.0
        result = nn.sample_mirostat_v2(
            self.logits,
            target_entropy=target_entropy,
            tau=0.5,
            mu=100.0,  # very high mu
            rng=self.rng
        )
        _, new_mu = result
        # mu should be clipped to reasonable range
        self.assertLessEqual(new_mu, target_entropy * 3.0)
        self.assertGreaterEqual(new_mu, target_entropy * 0.5)


class TestEntropyMetrics(unittest.TestCase):
    """Test entropy and information metrics."""
    
    def test_entropy_uniform(self):
        """Test entropy of uniform distribution."""
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        h = nn.entropy(probs)
        # Uniform distribution should have maximum entropy
        self.assertGreater(h, 0)
    
    def test_entropy_bits_uniform(self):
        """Test entropy in bits for uniform distribution."""
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        h = nn.entropy_bits(probs)
        # Should be log2(4) = 2 bits
        np.testing.assert_almost_equal(h, 2.0, decimal=5)
    
    def test_entropy_deterministic(self):
        """Test entropy of deterministic distribution is near zero."""
        probs = np.array([1.0, 0.0, 0.0, 0.0])
        h = nn.entropy(probs)
        self.assertLess(h, 0.01)
    
    def test_perplexity_high_prob(self):
        """Test perplexity for high probability target."""
        logits = np.array([1.0, 5.0, 2.0])
        ppl = nn.perplexity(logits, target_idx=1)
        # High prob target should have low perplexity
        self.assertLess(ppl, 2.0)
    
    def test_cross_entropy_positive(self):
        """Test cross entropy is always positive."""
        logits = np.random.randn(10)
        for target in range(len(logits)):
            ce = nn.cross_entropy(logits, target)
            self.assertGreater(ce, 0)
    
    def test_kl_divergence_identical(self):
        """Test KL divergence is zero for identical distributions."""
        p = np.array([0.25, 0.25, 0.25, 0.25])
        kl = nn.kl_divergence(p, p)
        self.assertLess(kl, 0.01)


class TestAdaptiveTemperature(unittest.TestCase):
    """Test entropy-aware temperature functions."""
    
    def test_entropy_temperature_bounds(self):
        """Test adaptive temperature respects bounds."""
        logits = np.random.randn(10)
        temp = nn.entropy_temperature(
            logits,
            target_entropy=2.0,
            min_temp=0.5,
            max_temp=1.5
        )
        self.assertGreaterEqual(temp, 0.5)
        self.assertLessEqual(temp, 1.5)
    
    def test_confidence_score_range(self):
        """Test confidence score is in [0, 1]."""
        logits = np.random.randn(10)
        conf = nn.confidence_score(logits)
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)
    
    def test_margin_score_positive(self):
        """Test margin score is positive."""
        logits = np.array([1.0, 5.0, 2.0])
        margin = nn.margin_score(logits)
        self.assertGreater(margin, 0)
    
    def test_resonance_temperature_bounds(self):
        """Test resonance temperature respects bounds."""
        logits = np.random.randn(10)
        history = [np.random.randn(10) for _ in range(5)]
        temp = nn.resonance_temperature(
            logits,
            history,
            target_resonance=0.7,
            min_temp=0.5,
            max_temp=1.5
        )
        self.assertGreaterEqual(temp, 0.5)
        self.assertLessEqual(temp, 1.5)
    
    def test_resonance_temperature_no_history(self):
        """Test resonance temperature with empty history."""
        logits = np.random.randn(10)
        temp = nn.resonance_temperature(
            logits,
            [],
            target_resonance=0.7,
            min_temp=0.5,
            max_temp=1.5
        )
        # should return mid-point when no history
        self.assertGreater(temp, 0.5)
        self.assertLess(temp, 1.5)


class TestResonanceMetrics(unittest.TestCase):
    """Test resonance metrics."""
    
    def test_resonance_score_identical(self):
        """Test resonance is 1 for identical distributions."""
        logits = np.random.randn(10)
        score = nn.resonance_score(logits, logits)
        np.testing.assert_almost_equal(score, 1.0, decimal=5)
    
    def test_resonance_score_range(self):
        """Test resonance score is in valid range."""
        logits1 = np.random.randn(10)
        logits2 = np.random.randn(10)
        score = nn.resonance_score(logits1, logits2)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_harmonic_mean_positive(self):
        """Test harmonic mean of positive values."""
        values = np.array([1.0, 2.0, 3.0, 4.0])
        hm = nn.harmonic_mean(values)
        self.assertGreater(hm, 0)
        # Harmonic mean should be less than arithmetic mean
        self.assertLess(hm, values.mean())


if __name__ == "__main__":
    unittest.main()
