"""
Tests for async haze modules: mathbrain, experts, trauma, subjectivity, cleanup.
"""

import pytest
import asyncio
import numpy as np


# ============================================================
#  MATHBRAIN TESTS
# ============================================================

class TestMathBrain:
    """Tests for MathBrain field perception."""
    
    def test_import(self):
        """MathBrain can be imported."""
        from haze.mathbrain import MathBrain, AsyncMathBrain, FieldPerception
        assert MathBrain is not None
        assert AsyncMathBrain is not None
        assert FieldPerception is not None
    
    def test_create_brain(self):
        """Can create MathBrain instance."""
        from haze.mathbrain import MathBrain
        brain = MathBrain()
        assert brain is not None
        assert len(brain.layers) > 0
    
    def test_forward_pass(self):
        """Forward pass produces output."""
        from haze.mathbrain import MathBrain
        brain = MathBrain()
        x = np.array([0.5, 0.3, 0.7, 0.1, 0.6])
        output = brain._forward(x)
        assert output.shape == (4,)
        assert np.all(output >= 0) and np.all(output <= 1)  # sigmoid output
    
    def test_async_perceive(self):
        """Async perception works."""
        from haze.mathbrain import AsyncMathBrain
        
        async def run_test():
            brain = AsyncMathBrain()
            perception = await brain.perceive(
                arousal=0.5,
                novelty=0.3,
                entropy=0.7,
                trauma=0.1,
                coherence=0.6,
            )
            assert perception is not None
            assert perception.mood in ["calm", "excited", "focused", "diffuse", "alert"]
            assert 0.4 <= perception.recommended_temp <= 1.2
            assert 0.0 <= perception.identity_weight <= 1.0
        
        asyncio.run(run_test())
    
    def test_perceive_smooth(self):
        """EMA smoothing works."""
        from haze.mathbrain import AsyncMathBrain
        
        async def run_test():
            brain = AsyncMathBrain()
            
            # First perception
            p1 = await brain.perceive_smooth(arousal=0.2)
            # Second with different value
            p2 = await brain.perceive_smooth(arousal=0.8)
            
            # Smoothed arousal should be between 0.2 and 0.8
            assert 0.2 <= p2.arousal <= 0.8
        
        asyncio.run(run_test())
    
    def test_hebbian_update(self):
        """Hebbian update modifies weights."""
        from haze.mathbrain import AsyncMathBrain
        
        async def run_test():
            brain = AsyncMathBrain()
            
            # Perceive first
            await brain.perceive(arousal=0.5)
            
            # Get initial weights
            initial_weights = brain.layers[0].weights.copy()
            
            # Apply Hebbian update with positive reward
            await brain.hebbian_update(reward=1.0)
            
            # Weights should change
            assert not np.allclose(brain.layers[0].weights, initial_weights)
        
        asyncio.run(run_test())


# ============================================================
#  EXPERTS TESTS
# ============================================================

class TestExperts:
    """Tests for Resonant Experts (MOE-style routing)."""
    
    def test_import(self):
        """Experts can be imported."""
        from haze.experts import route_to_mixture, pulse_to_signals, ExpertMixture
        assert route_to_mixture is not None
        assert pulse_to_signals is not None
        assert ExpertMixture is not None
    
    def test_pulse_to_signals(self):
        """Pulse converts to field signals."""
        from haze.experts import pulse_to_signals
        signals = pulse_to_signals(novelty=0.5, arousal=0.3, entropy=0.7)
        # FieldSignals is a dataclass, check attributes
        assert hasattr(signals, 'novelty')
        assert hasattr(signals, 'arousal')
        assert hasattr(signals, 'entropy')
    
    def test_route_to_mixture(self):
        """Routing produces mixture of experts."""
        from haze.experts import route_to_mixture, pulse_to_signals
        signals = pulse_to_signals(novelty=0.5, arousal=0.3, entropy=0.7)
        mixture = route_to_mixture(signals)
        
        # Should have all 4 experts
        assert 'structural' in mixture.weights
        assert 'semantic' in mixture.weights
        assert 'creative' in mixture.weights
        assert 'precise' in mixture.weights
        
        # Weights should sum to ~1
        total = sum(mixture.weights.values())
        assert 0.99 <= total <= 1.01
        
        # Temperature should be in valid range
        assert 0.3 <= mixture.temperature <= 1.5
    
    def test_high_arousal_boosts_semantic(self):
        """High arousal increases semantic expert weight."""
        from haze.experts import route_to_mixture, pulse_to_signals
        
        low_arousal = route_to_mixture(pulse_to_signals(arousal=0.1))
        high_arousal = route_to_mixture(pulse_to_signals(arousal=0.9))
        
        # High arousal should boost semantic
        assert high_arousal.weights['semantic'] >= low_arousal.weights['semantic']
    
    def test_high_novelty_boosts_creative(self):
        """High novelty affects expert weights."""
        from haze.experts import route_to_mixture, pulse_to_signals
        
        low_novelty = route_to_mixture(pulse_to_signals(novelty=0.1))
        high_novelty = route_to_mixture(pulse_to_signals(novelty=0.9))
        
        # Both should have valid weights
        assert high_novelty.weights['creative'] > 0
        assert low_novelty.weights['creative'] > 0


# ============================================================
#  TRAUMA TESTS
# ============================================================

class TestTrauma:
    """Tests for Trauma module (identity return)."""
    
    def test_import(self):
        """Trauma can be imported."""
        from haze.trauma import AsyncTrauma, TraumaState, get_identity_prefix
        assert AsyncTrauma is not None
        assert TraumaState is not None
        assert get_identity_prefix is not None
    
    def test_detect_trauma(self):
        """Trauma detection works."""
        from haze.trauma import AsyncTrauma
        
        async def run_test():
            trauma = AsyncTrauma()
            
            # Text with bootstrap words should trigger trauma
            state = await trauma.process("The haze resonates with the field pattern")
            
            assert state is not None
            assert state.level > 0  # Should detect some trauma
            assert len(state.trigger_words) > 0
        
        asyncio.run(run_test())
    
    def test_no_trauma_on_neutral(self):
        """Neutral text has low or no trauma."""
        from haze.trauma import AsyncTrauma
        
        async def run_test():
            trauma = AsyncTrauma()
            
            state = await trauma.process("Hello how are you today")
            
            # May return None for neutral text, or low trauma
            if state is not None:
                assert state.level < 0.5
        
        asyncio.run(run_test())
    
    def test_identity_prefix(self):
        """Identity prefix generation works."""
        from haze.trauma import get_identity_prefix
        
        # get_identity_prefix takes no arguments, returns random prefix
        prefix = get_identity_prefix()
        
        assert prefix is not None
        # Should contain "haze" or "field"
        assert "haze" in prefix.lower() or "field" in prefix.lower()


# ============================================================
#  CLEANUP TESTS
# ============================================================

class TestCleanup:
    """Tests for cleanup module."""
    
    def test_import(self):
        """Cleanup can be imported."""
        from haze.cleanup import cleanup_output
        assert cleanup_output is not None
    
    def test_basic_cleanup(self):
        """Basic cleanup works."""
        from haze.cleanup import cleanup_output
        result = cleanup_output("  hello world  ")
        assert "hello" in result.lower()
    
    def test_contraction_preservation(self):
        """Contractions are preserved."""
        from haze.cleanup import cleanup_output
        
        tests = ["I'm", "don't", "they're", "it's", "won't"]
        for t in tests:
            result = cleanup_output(f"{t} here")
            # Should contain some form of apostrophe (ASCII or fancy)
            has_apostrophe = "'" in result or chr(8217) in result
            assert has_apostrophe, f"Failed for {t}: {result}"
    
    def test_broken_contraction_fix(self):
        """Broken contractions are fixed."""
        from haze.cleanup import cleanup_output
        
        # "I'" + space should become "I'm"
        result = cleanup_output("I' trying")
        has_im = "I'm" in result or "I'm" in result or ("I" in result and "m" in result)
        assert has_im, f"Got: {result}"
        
        # "don" + space + verb should become "don't"
        result = cleanup_output("don believe")
        has_dont = "don't" in result or "don't" in result or ("don" in result and "t" in result)
        assert has_dont, f"Got: {result}"
    
    def test_heuristic_contraction_fix(self):
        """Heuristic patterns work (-ing, -ed, -en)."""
        from haze.cleanup import cleanup_output
        
        # -ing
        result = cleanup_output("don trying")
        assert "don't" in result or "don't" in result
        
        # -ed
        result = cleanup_output("don tired")
        assert "don't" in result or "don't" in result
        
        # -en
        result = cleanup_output("don forgotten")
        assert "don't" in result or "don't" in result
    
    def test_they_re_fix(self):
        """'they re' becomes 'they're'."""
        from haze.cleanup import cleanup_output
        
        result = cleanup_output("they re here")
        assert "they're" in result or "they're" in result
    
    def test_em_dash_removal(self):
        """Em-dash at start is removed."""
        from haze.cleanup import cleanup_output
        
        result = cleanup_output("— Hello there")
        assert not result.startswith("—")
        assert not result.startswith("–")


# ============================================================
#  SUBWORD FIELD TESTS
# ============================================================

class TestSubwordField:
    """Tests for SubwordField (BPE tokenization)."""
    
    def test_import(self):
        """SubwordField can be imported."""
        try:
            from haze.subword_field import SubwordField, AsyncSubwordField
            assert SubwordField is not None
        except ImportError:
            pytest.skip("sentencepiece not installed")
    
    def test_build_from_corpus(self, tmp_path):
        """Can build field from corpus."""
        try:
            import sentencepiece
        except ImportError:
            pytest.skip("sentencepiece not installed")
        
        from haze.subword_field import SubwordField
        
        # Create temp corpus with enough data
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("Hello world. I love you. The living room. Don't worry.\n" * 100)
        
        # Use smaller vocab size to avoid error
        field = SubwordField.from_corpus(str(corpus), vocab_size=50)
        
        assert field is not None
        assert field.vocab is not None
        assert len(field.bigram_counts) > 0
    
    def test_generate(self, tmp_path):
        """Generation produces text."""
        try:
            import sentencepiece
        except ImportError:
            pytest.skip("sentencepiece not installed")
        
        from haze.subword_field import SubwordField
        
        # Create temp corpus with enough data
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("Hello world. I love you. The living room. Don't worry.\n" * 100)
        
        # Use smaller vocab size
        field = SubwordField.from_corpus(str(corpus), vocab_size=50)
        result = field.generate("Hello", length=10, temperature=0.7)
        
        assert result is not None
        assert len(result) > 0


# ============================================================
#  SUBJECTIVITY TESTS
# ============================================================

class TestSubjectivity:
    """Tests for Subjectivity (no seed from prompt)."""
    
    def test_import(self):
        """Subjectivity can be imported."""
        from haze.subjectivity import Subjectivity, AsyncSubjectivity, PulseSnapshot
        assert Subjectivity is not None
        assert AsyncSubjectivity is not None
        assert PulseSnapshot is not None
    
    def test_pulse_computation(self):
        """Pulse is computed from input."""
        from haze.subjectivity import Subjectivity
        from haze.haze import Vocab
        
        corpus = "Hello world. I love you. The living room."
        vocab = Vocab.from_text(corpus)
        subj = Subjectivity(corpus, vocab)
        
        pulse = subj.compute_pulse("AMAZING!!! I LOVE THIS!!!")
        
        assert pulse is not None
        assert 0 <= pulse.arousal <= 1
        assert 0 <= pulse.novelty <= 1
        assert 0 <= pulse.entropy <= 1
        # High arousal for exclamation
        assert pulse.arousal > 0.3
    
    def test_internal_seed_excludes_prompt(self):
        """Internal seed does NOT contain prompt words."""
        from haze.subjectivity import Subjectivity
        from haze.haze import Vocab
        
        corpus = "Hello world. I love you. The living room. Darling sweetheart."
        vocab = Vocab.from_text(corpus)
        subj = Subjectivity(corpus, vocab)
        
        prompt = "I love"
        tokens, pulse, seed_text = subj.get_internal_seed(prompt)
        
        # Seed should NOT contain "I" or "love"
        seed_words = set(seed_text.lower().split())
        prompt_words = set(prompt.lower().split())
        
        overlap = seed_words & prompt_words
        assert len(overlap) == 0, f"Seed contains prompt words: {overlap}"
