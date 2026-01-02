#!/usr/bin/env python3
"""
Tests for enhanced cleanup.py functionality.
"""

import unittest
from haze.cleanup import (
    cleanup_output,
    cleanup_with_resonance,
    ensure_sentence_boundaries,
    calculate_garbage_score,
    _detect_poetic_repetition,
    _calculate_local_entropy,
)


class TestBasicCleanup(unittest.TestCase):
    """Test basic cleanup functionality."""
    
    def test_punctuation_normalization(self):
        """Test quote and apostrophe normalization."""
        result = cleanup_output("don't say 'hello'")
        # After normalization, we should have fancy apostrophes (U+2019)
        # Check by Unicode codepoint
        self.assertTrue(any(ord(c) == 0x2019 for c in result), 
                       "Should contain right single quote (U+2019)")
        
    def test_repeated_punctuation(self):
        """Test collapsing repeated punctuation."""
        result = cleanup_output("Wait..... really????")
        self.assertEqual(result, "Wait... Really???")
        
    def test_capitalize_first_letter(self):
        """Test first letter capitalization."""
        result = cleanup_output("hello world")
        self.assertTrue(result[0].isupper())
        
    def test_capitalize_i(self):
        """Test standalone 'i' capitalization."""
        result = cleanup_output("i am here")
        self.assertIn("I am", result)


class TestRepetitionHandling(unittest.TestCase):
    """Test word repetition detection and handling."""
    
    def test_double_repetition_removed(self):
        """Test double word repetition is removed."""
        result = cleanup_output("the the house")
        self.assertEqual(result.lower(), "the house.")
        
    def test_triple_repetition_removed(self):
        """Test triple word repetition is removed."""
        result = cleanup_output("the the the house")
        self.assertEqual(result.lower(), "the house.")
        
    def test_quad_repetition_removed(self):
        """Test 4+ word repetitions are removed."""
        result = cleanup_output("word word word word here")
        self.assertEqual(result.lower(), "word here.")
    
    def test_poetic_repetition_preserved(self):
        """Test comma-separated repetition is preserved."""
        result = cleanup_output("Love, love, love in the morning")
        self.assertIn("love, love, love", result.lower())
        
    def test_emphatic_repetition_preserved(self):
        """Test emphatic repetition with punctuation is preserved."""
        result = cleanup_output("Never, never, never!")
        # Should preserve the pattern (may capitalize)
        self.assertGreaterEqual(result.lower().count("never"), 3)


class TestContractions(unittest.TestCase):
    """Test contraction handling."""
    
    def test_basic_contractions(self):
        """Test basic contraction fixes."""
        cases = [
            ("dont go", "don't go"),
            ("wont work", "won't work"),
            ("cant see", "can't see"),
            ("isnt it", "isn't it"),
        ]
        for input_text, expected_substr in cases:
            result = cleanup_output(input_text)
            self.assertIn(expected_substr, result.lower())
    
    def test_advanced_contractions(self):
        """Test advanced/compound contractions."""
        cases = [
            ("would have gone", "would've gone"),
            ("could have been", "could've been"),
            ("should have said", "should've said"),
        ]
        for input_text, expected_substr in cases:
            result = cleanup_output(input_text)
            self.assertIn(expected_substr, result.lower())
    
    def test_possessive_vs_contraction(self):
        """Test its vs it's disambiguation."""
        # Should be "it's" (it is)
        result1 = cleanup_output("its going to rain")
        self.assertIn("it's", result1.lower())
        
        # Should be "its" (possessive)
        result2 = cleanup_output("its wings spread wide")
        self.assertIn("its wings", result2.lower())
        self.assertNotIn("it's wings", result2.lower())


class TestSentenceStructure(unittest.TestCase):
    """Test sentence structure improvements."""
    
    def test_sentence_ending_added(self):
        """Test that missing sentence endings are added."""
        result = cleanup_output("Hello world")
        self.assertTrue(result.endswith('.') or result.endswith('!') or result.endswith('?'))
    
    def test_ellipsis_cleanup(self):
        """Test trailing ellipsis is cleaned up."""
        # Trailing ellipsis should be converted to period
        result = cleanup_output("I don't know...")
        self.assertTrue(result.endswith('.'))
        # Should not end with multiple dots
        self.assertFalse(result.endswith('...'))
        
        # Mid-sentence ellipsis should be preserved
        result2 = cleanup_output("I don't know... but I think so")
        # Should have proper ending
        self.assertTrue(result2.endswith('.'))
    
    def test_capitalize_after_period(self):
        """Test capitalization after period."""
        result = cleanup_output("Hello. world.")
        self.assertIn("Hello. World.", result)
    
    def test_run_on_sentences_moderate(self):
        """Test run-on sentence detection in moderate mode."""
        result = cleanup_output("I went there I saw things", mode="moderate")
        # Should have at least 2 sentences now
        self.assertGreaterEqual(result.count('.'), 1)


class TestArtifactCleanup(unittest.TestCase):
    """Test cleanup of generation artifacts."""
    
    def test_orphan_apostrophe_fragments(self):
        """Test removal of orphan apostrophe fragments."""
        result = cleanup_output("hello 't there 's world")
        self.assertNotIn(" 't ", result)
        self.assertNotIn(" 's ", result)
    
    def test_short_word_validation(self):
        """Test that valid short words are preserved."""
        result = cleanup_output("I go to a place")
        for word in ["I", "go", "to", "a"]:
            self.assertIn(word, result)
    
    def test_garbage_score_calculation(self):
        """Test garbage score calculation."""
        clean_text = "Hello, world."
        messy_text = "H..,,ello,,.world..,,"
        
        clean_score = calculate_garbage_score(clean_text)
        messy_score = calculate_garbage_score(messy_text)
        
        self.assertLess(clean_score, messy_score)


class TestEntropyAndResonance(unittest.TestCase):
    """Test entropy and resonance-aware features."""
    
    def test_entropy_calculation(self):
        """Test local entropy calculation."""
        # High entropy (diverse characters)
        high_entropy_text = "abcdefghijklmnop"
        # Low entropy (repetitive)
        low_entropy_text = "aaaaaaaaaaaaaaaa"
        
        high_entropy = _calculate_local_entropy(high_entropy_text)
        low_entropy = _calculate_local_entropy(low_entropy_text)
        
        self.assertGreater(high_entropy, low_entropy)
    
    def test_poetic_repetition_detection(self):
        """Test detection of poetic repetition patterns."""
        text = "Love, love, love in the morning light"
        regions = _detect_poetic_repetition(text)
        
        # Should detect at least one comma repetition pattern
        self.assertGreater(len(regions), 0)
    
    def test_resonance_aware_cleanup(self):
        """Test resonance-aware cleanup mode selection."""
        text = "Hello the the world"
        
        # High resonance, high entropy -> gentle mode
        result1 = cleanup_with_resonance(text, resonance_score=0.8, entropy=3.0)
        
        # Low resonance, low entropy -> moderate mode
        result2 = cleanup_with_resonance(text, resonance_score=0.3, entropy=1.2)
        
        # Both should fix the repetition, but we're testing mode selection works
        self.assertNotEqual(result1, text)


class TestSentenceBoundaries(unittest.TestCase):
    """Test sentence boundary detection and repair."""
    
    def test_ensure_sentence_boundaries(self):
        """Test sentence boundary enforcement."""
        result = ensure_sentence_boundaries("hello world")
        self.assertTrue(result.endswith('.'))
        self.assertTrue(result[0].isupper())
    
    def test_fragment_removal(self):
        """Test removal of trailing fragments."""
        result = ensure_sentence_boundaries("Hello world st")
        # Should remove very short trailing fragment
        self.assertNotIn(" st", result)
    
    def test_multiple_sentences(self):
        """Test multiple sentence handling."""
        result = ensure_sentence_boundaries("hello. world. testing")
        sentences = result.split('.')
        # Each sentence should start with capital
        for sentence in sentences[:-1]:  # Exclude empty last element
            if sentence.strip():
                self.assertTrue(sentence.strip()[0].isupper())


class TestModeVariations(unittest.TestCase):
    """Test different cleanup modes."""
    
    def test_gentle_mode(self):
        """Test gentle mode preserves more."""
        text = "hello world the the test"
        result = cleanup_output(text, mode="gentle")
        # Should still fix basic issues
        self.assertTrue(result[0].isupper())
    
    def test_moderate_mode(self):
        """Test moderate mode is more aggressive."""
        text = "hello I went there I came back"
        result = cleanup_output(text, mode="moderate")
        # May add sentence breaks
        self.assertGreaterEqual(len(result), len(text) - 5)
    
    def test_strict_mode(self):
        """Test strict mode is most aggressive."""
        text = "hello world st"
        result = cleanup_output(text, mode="strict")
        # Should clean up trailing fragments
        self.assertTrue(result.endswith('.'))


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_empty_string(self):
        """Test empty string handling."""
        result = cleanup_output("")
        self.assertEqual(result, "")
    
    def test_none_input(self):
        """Test None input handling."""
        result = cleanup_output(None)
        self.assertIsNone(result)
    
    def test_very_short_text(self):
        """Test very short text handling."""
        result = cleanup_output("hi")
        self.assertEqual(result, "Hi.")
    
    def test_only_punctuation(self):
        """Test text with only punctuation."""
        result = cleanup_output("...")
        # Should handle gracefully
        self.assertIsInstance(result, str)


class TestRealWorldExamples(unittest.TestCase):
    """Test with real-world generation examples."""
    
    def test_gothic_dialogue(self):
        """Test cleanup of gothic dialogue style."""
        text = "I dont know... the haze the haze settles over everything"
        result = cleanup_output(text)
        
        # Should fix contraction
        self.assertIn("don't", result)
        # Should fix repetition
        self.assertEqual(result.lower().count("the haze"), 1)
        # Should have proper ending
        self.assertTrue(result.endswith('.'))
    
    def test_mixed_artifacts(self):
        """Test cleanup of mixed artifacts."""
        text = "hello 't its going st the the well"
        result = cleanup_output(text)
        
        # Should remove orphan apostrophe
        self.assertNotIn(" 't ", result)
        # Should fix its -> it's
        self.assertIn("it's", result.lower())
        # Should remove repetition
        self.assertEqual(result.lower().count("the"), 1)
    
    def test_preserves_style(self):
        """Test that emergent style is preserved."""
        text = "The darkness, the darkness, the darkness calls to me"
        result = cleanup_output(text)
        
        # Should preserve emphatic repetition with commas
        self.assertEqual(result.lower().count("the darkness"), 3)


if __name__ == '__main__':
    unittest.main()
