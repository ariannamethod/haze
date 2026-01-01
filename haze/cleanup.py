#!/usr/bin/env python3
# cleanup.py — Output cleanup for Haze speech
#
# Adapted from Leo's punct_cleanup.py
# Removes obvious garbage patterns while preserving emergent style.
#
# Philosophy: Clean the noise, keep the soul.
#
# Key improvements:
#   - Remove "—" at the start of output (haze is not dialogue-only)
#   - Preserve emergent strangeness while fixing obvious garbage
#   - Support for presence-style output (not chatbot-style)
#
# Usage:
#   from haze.cleanup import cleanup_output
#   clean_text = cleanup_output(raw_text)

import re
from typing import Dict


def cleanup_output(text: str, mode: str = "gentle") -> str:
    """
    Clean up generation output without killing emergent style.
    
    Args:
        text: raw generated text
        mode: "gentle" (preserve style), "moderate", or "strict"
    
    Returns:
        Cleaned text with preserved personality
    """
    if not text or not isinstance(text, str):
        return text
    
    result = text
    
    # 0. Normalize quotes and apostrophes to corpus-compatible versions
    # The corpus uses fancy quotes: ' ' " " instead of ASCII ' "
    # Use Unicode escapes to ensure correct characters
    result = result.replace("'", "’")  # ASCII apostrophe (U+0027) → right single quote (U+2019)
    result = result.replace('"', "”")  # ASCII double quote → right double quote (U+201D)
    
    # 0b. Replace sentencepiece unknown marker
    result = result.replace('\u2047', "\u2019")  # ⁇ (U+2047) → apostrophe
    result = result.replace(" \u2047 ", " ")
    
    # 1. Collapse repeated punctuation (but keep max 3 for style)
    result = re.sub(r'\.{4,}', '...', result)
    result = re.sub(r'\?{4,}', '???', result)
    result = re.sub(r'!{4,}', '!!!', result)
    result = re.sub(r'…{2,}', '…', result)
    
    # 2. Clean up "symbol dumps" - obvious garbage patterns
    result = re.sub(r'\.(?=[,?])', '', result)   # .,? → ,?
    result = re.sub(r'\.[,]+', '.', result)      # .,, → .
    result = re.sub(r'\?[.,:]', '?', result)     # ?. → ?
    result = re.sub(r'![.,:]', '!', result)      # !. → !
    result = re.sub(r',[.,]+(?!\.\.)', ',', result)  # ,., → ,
    
    # 3. Clean up trailing garbage
    result = re.sub(r'\s+[,\.]+\s*([.!?])', r'\1', result)
    
    # 4. Fix spaces before punctuation
    result = re.sub(r'\s+([,;:?!])', r'\1', result)
    
    # 5. Ensure space after punctuation (except before newline)
    result = re.sub(r'([,;:?!\.])(?=[a-zA-Z])', r'\1 ', result)
    
    # 6. Collapse multiple spaces
    result = re.sub(r'\s{2,}', ' ', result)
    
    # 7. Clean up orphaned punctuation at end
    result = re.sub(r'\s+(and|then|but|or|the|a|an)[.,]\s*$', r' \1', result)
    
    # 8. Clean double dots and punctuation garbage
    result = re.sub(r'\.\s*\.', '.', result)     # ". ." → "."
    result = re.sub(r'\.\s+,', '.', result)      # ". ," → "."
    result = re.sub(r',\s*,', ',', result)       # ", ," → ","
    
    # 9. Fix dialogue markers (— should have space after)
    result = re.sub(r'—(?=[a-zA-Z])', '— ', result)
    
    # 10. Capitalize first letter after dialogue marker
    def cap_after_dash(m):
        return m.group(1) + m.group(2).upper()
    result = re.sub(r'(—\s*)([a-z])', cap_after_dash, result)
    
    # 11. Remove ALL em-dashes from output
    # Philosophy: haze is PRESENCE, not dialogue. No "— Trade secret." style.
    # This makes speech cleaner and more Leo-like.
    # Em-dash variants: — (U+2014), – (U+2013)
    # Replace with nothing (join sentences) or period
    result = re.sub(r'\s*—\s*', ' ', result)  # Replace em-dash with space
    result = re.sub(r'\s*–\s*', ' ', result)  # Replace en-dash with space
    
    # Clean up any resulting double spaces
    result = re.sub(r'\s{2,}', ' ', result)
    
    # 12. Capitalize first letter of text
    result = result.strip()
    if result and result[0].islower():
        result = result[0].upper() + result[1:]
    
    # 13. Capitalize "I" when standalone
    result = re.sub(r'\bi\b', 'I', result)
    
    # 14. Capitalize after periods (new sentences)
    def cap_after_period(m):
        return m.group(1) + m.group(2).upper()
    result = re.sub(r'(\.\s+)([a-z])', cap_after_period, result)
    
    # 15. Fix broken contractions (character-level and subword generation artifacts)
    # Common contractions that get broken: don't, won't, can't, it's, etc.
    # 
    # IMPORTANT: Use \s+ (one or more spaces) for possessive-like patterns to avoid
    # matching real words like "its" (possessive pronoun) vs "it's" (it is)
    contraction_fixes = [
        # n't contractions - can use \s* because "dont" is always wrong
        (r'\bdon\s*t\b', "don't"),
        (r'\bwon\s*t\b', "won't"),
        (r'\bcan\s*t\b', "can't"),
        (r'\bain\s*t\b', "ain't"),
        (r'\bisn\s*t\b', "isn't"),
        (r'\baren\s*t\b', "aren't"),
        (r'\bwasn\s*t\b', "wasn't"),
        (r'\bweren\s*t\b', "weren't"),
        (r'\bhasn\s*t\b', "hasn't"),
        (r'\bhaven\s*t\b', "haven't"),
        (r'\bhadn\s*t\b', "hadn't"),
        (r'\bdoesn\s*t\b', "doesn't"),
        (r'\bdidn\s*t\b', "didn't"),
        (r'\bwouldn\s*t\b', "wouldn't"),
        (r'\bcouldn\s*t\b', "couldn't"),
        (r'\bshouldn\s*t\b', "shouldn't"),
        # 's contractions - MUST use \s+ to avoid matching "its", "hes", "shes"
        (r'\bit\s+s\b', "it's"),
        (r'\bhe\s+s\b', "he's"),
        (r'\bshe\s+s\b', "she's"),
        (r'\bthat\s+s\b', "that's"),
        (r'\bwhat\s+s\b', "what's"),
        (r'\bwhere\s+s\b', "where's"),
        (r'\bhere\s+s\b', "here's"),
        (r'\bthere\s+s\b', "there's"),
        (r'\blet\s+s\b', "let's"),
        # I contractions - can use \s* because "Im", "Ive" are always wrong
        (r'\bi\s*m\b', "I'm"),
        (r'\bi\s*ve\b', "I've"),
        (r'\bi\s*ll\b', "I'll"),
        (r'\bi\s*d\b', "I'd"),
        # you contractions - use \s+ because "youre" etc. are recognizable
        (r'\byou\s*re\b', "you're"),
        (r'\byou\s*ve\b', "you've"),
        (r'\byou\s*ll\b', "you'll"),
        (r'\byou\s*d\b', "you'd"),
        # we contractions
        (r'\bwe\s*re\b', "we're"),
        (r'\bwe\s*ve\b', "we've"),
        (r'\bwe\s*ll\b', "we'll"),
        # they contractions
        (r'\bthey\s*re\b', "they're"),
        # DEBUG: print("Checking they re pattern")
        (r'\bthey\s*ve\b', "they've"),
        (r'\bthey\s*ll\b', "they'll"),
    ]
    for pattern, replacement in contraction_fixes:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
    # 15b. Fix incomplete contractions (apostrophe present but missing ending)
    # These happen when subword tokenization splits contractions oddly
    # NOTE: After step 0, text has fancy apostrophe ' (U+2019)
    # Use character class to match both ASCII and fancy apostrophes
    apos = "['’]"  # Match ASCII ', fancy ', and U+2019
    
    # "I'" followed by space → "I'm" (most likely)
    result = re.sub(rf"\bI{apos}\s+", "I’m ", result)
    
    # "it'" followed by space → "it's"
    result = re.sub(rf"\bit{apos}\s+", "it’s ", result, flags=re.IGNORECASE)
    
    # "he'" / "she'" / "that'" / "what'" / "there'" / "where'" / "who'" → add 's
    result = re.sub(rf"\bhe{apos}\s+", "he’s ", result, flags=re.IGNORECASE)
    result = re.sub(rf"\bshe{apos}\s+", "she’s ", result, flags=re.IGNORECASE)
    result = re.sub(rf"\bthat{apos}\s+", "that’s ", result, flags=re.IGNORECASE)
    result = re.sub(rf"\bwhat{apos}\s+", "what’s ", result, flags=re.IGNORECASE)
    result = re.sub(rf"\bthere{apos}\s+", "there’s ", result, flags=re.IGNORECASE)
    result = re.sub(rf"\bwhere{apos}\s+", "where’s ", result, flags=re.IGNORECASE)
    result = re.sub(rf"\bwho{apos}\s+", "who’s ", result, flags=re.IGNORECASE)
    
    # "don" + space + verb → "don't" + verb (common broken pattern)
    # "don" + space + verb → "don't" + verb (common broken pattern)
    # PART 1: Hardcoded common verbs (including gothic/literary ones)
    result = re.sub(r"\bdon\s+(believe|think|know|want|need|like|care|worry|mind|understand|remember|forget|see|hear|feel|get|go|do|be|have|make|take|give|say|tell|ask|try|look|come|put|let|seem|mean|stop|start|die|live|stay|leave|keep|wait|work|play|sleep|eat|drink|read|write|watch|listen|touch|hurt|cry|laugh|love|hate|miss|trust|turn|move|run|walk|talk|speak|call|find|hold|sit|stand|open|close|break|change|move|use|show|help|bring|send|meet|learn|grow|fall|pick|pull|push|hang|cut|hit|set|pay|buy|sell|wear|throw|catch|carry|draw|fight|beat|kill|burn|fix|clean|build|drive|ride|fly|swim|dance|sing|jump|drop|lose|win|choose|teach|reach|pass|cross|hide|rise|raise|shake|wake|ring|swing|shut|stick|bend|blow|tear|feed|lead|spend|lend|bite|steal|trudge|wander|linger|ponder|whisper|murmur|shiver|tremble|fade|drift|ache|yearn|mourn|grieve|regret|suffer|struggle|stumble|tumble|crumble|shatter|scatter|gather|matter|bother|smother|hover|cover|discover|recover|uncover|sober|wonder|thunder|blunder|plunder|slumber|lumber|number|remember|member|tender|render|surrender|hinder|wander|ponder|squander)\b", r"don't \1", result, flags=re.IGNORECASE)
    
    # PART 2: Heuristic by word endings (catches words not in hardcoded list)
    # -ing endings: trying, dying, living, waiting, working, etc.
    result = re.sub(r"\bdon\s+(\w+ing)\b", r"don't \1", result, flags=re.IGNORECASE)
    # -ed endings (adjectives/participles): tired, bored, scared, worried, etc.
    result = re.sub(r"\bdon\s+(\w+ed)\b", r"don't \1", result, flags=re.IGNORECASE)
    # -en endings (participles): forgotten, broken, taken, etc.
    result = re.sub(r"\bdon\s+(\w+en)\b", r"don't \1", result, flags=re.IGNORECASE)
    # -le/-ge/-se/-ze endings: struggle, trudge, lose, freeze, etc.
    result = re.sub(r"\bdon\s+(\w+(?:le|ge|se|ze))\b", r"don't \1", result, flags=re.IGNORECASE)
    
    # Same for "won" → "won't"
    result = re.sub(r"\bwon\s+(\w+ing|\w+ed|believe|think|know|want|need|like|go|do|be|have|make|say|tell|try|stop|wait|work|turn|move|run|walk|talk|speak|call|find|hold|sit|stand|open|close|break|change|use|show|help|bring|send|meet|learn|grow|fall|pick|let|get|take|give|come|put|look|see|hear|feel|stay|leave|keep|die|live|start|eat|drink|sleep|play|read|write|watch|listen)\b", r"won't \1", result, flags=re.IGNORECASE)
    
    # 15d. ORPHAN CONTRACTION FIX: "don" alone at end/before punctuation → "ain't"
    # Philosophy: If subword tokenization cuts "don't" to just "don", 
    # we rescue it as "ain't" which has CHARACTER and fits gothic romance vibe!
    # 
    # "I don of that" → "I ain't of that"
    # "I don." → "I ain't."
    # "I don trudge" → "I ain't trudge" (verb-like)
    # 
    # Match "don" when:
    # - At end of text: \bdon$
    # - Before punctuation: \bdon(?=[.,!?])
    # - Before preposition/article (not a verb): \bdon\s+(of|the|a|an|to|for|with|from|about|by|on|in|at|my|your|his|her|their|its|this|that)
    result = re.sub(r"\bdon\s*$", "ain't", result, flags=re.IGNORECASE)
    result = re.sub(r"\bdon(?=[.,!?])", "ain't", result, flags=re.IGNORECASE)
    result = re.sub(r"\bdon\s+(of|the|a|an|to|for|with|from|about|by|on|in|at|my|your|his|her|their|its|this|that)\b", r"ain't \1", result, flags=re.IGNORECASE)
    
    # Same for "won" orphan → "ain't" (rare but possible)
    result = re.sub(r"\bwon\s*$", "ain't", result, flags=re.IGNORECASE)
    result = re.sub(r"\bwon(?=[.,!?])", "ain't", result, flags=re.IGNORECASE)
    
    # "they" + "my" (missing 're) → "they’re my"
    result = re.sub(r"\bthey\s+my\b", "they’re my", result, flags=re.IGNORECASE)
    
        # 15c. Additional subword-style broken contractions (space instead of apostrophe)
    # "they re" → "they're", "you re" → "you're", etc.
    result = re.sub(r"\bthey\s+re\b", "they're", result, flags=re.IGNORECASE)
    result = re.sub(r"\byou\s+re\b", "you're", result, flags=re.IGNORECASE)
    result = re.sub(r"\bwe\s+re\b", "we're", result, flags=re.IGNORECASE)
    result = re.sub(r"\bthey\s+ve\b", "they've", result, flags=re.IGNORECASE)
    result = re.sub(r"\byou\s+ve\b", "you've", result, flags=re.IGNORECASE)
    result = re.sub(r"\bwe\s+ve\b", "we've", result, flags=re.IGNORECASE)
    result = re.sub(r"\bi\s+ve\b", "I've", result, flags=re.IGNORECASE)
    result = re.sub(r"\bthey\s+ll\b", "they'll", result, flags=re.IGNORECASE)
    result = re.sub(r"\byou\s+ll\b", "you'll", result, flags=re.IGNORECASE)
    result = re.sub(r"\bwe\s+ll\b", "we'll", result, flags=re.IGNORECASE)
    result = re.sub(r"\bi\s+ll\b", "I'll", result, flags=re.IGNORECASE)
    
    # 16. Remove word/phrase repetition (character-level generation artifact)
    # "the the" → "the", "I I" → "I"
    result = re.sub(r'\b(\w+)\s+\1\b', r'\1', result, flags=re.IGNORECASE)
    # Triple repetition
    result = re.sub(r'\b(\w+)\s+\1\s+\1\b', r'\1', result, flags=re.IGNORECASE)
    
    # 17. Fix common word fragments (character-level artifacts)
    # Always apply basic fragment cleanup in gentle mode too
    
    # 17a. Remove orphan apostrophe fragments: 't, 's, 'm, 're, 've, 'll, 'd
    # These are leftovers from broken contractions
    # Match both ASCII ' and fancy ' apostrophes
    result = re.sub(r"\s+['''][tsmd]\b", '', result)
    result = re.sub(r"\s+['''](?:re|ve|ll)\b", '', result)
    
    # 17b. Remove words that start with apostrophe (broken fragments)
    # e.g., "'nt" at word start, "On't" → remove
    # BUT preserve valid contractions: I'm, I've, I'll, I'd, etc.
    def remove_apostrophe_garbage(match):
        word = match.group(0)
        # Normalize apostrophe for comparison
        word_normalized = word.replace("'", "'").replace(chr(8217), "'")
        # Valid contractions (all with ASCII apostrophe for comparison)
        valid_contractions = {"I'm", "I've", "I'll", "I'd", "it's", "he's", "she's", 
                              "that's", "what's", "there's", "where's", "who's",
                              "don't", "won't", "can't", "isn't", "aren't", "wasn't",
                              "weren't", "hasn't", "haven't", "hadn't", "doesn't",
                              "didn't", "wouldn't", "couldn't", "shouldn't", "ain't",
                              "you're", "you've", "you'll", "you'd", "we're", "we've",
                              "we'll", "they're", "they've", "they'll", "let's"}
        if word_normalized in valid_contractions or word_normalized.lower() in {c.lower() for c in valid_contractions}:
            return word
        return ''
    
    # Match STANDALONE apostrophe-words only (not contraction endings like 're in they're)
    # Use negative lookbehind to ensure NOT preceded by a letter
    result = re.sub(r"(?<![a-zA-Z])['''][a-z]+\b", remove_apostrophe_garbage, result)
    
    # 17c. Remove obvious 1-2 char garbage (except real words and contraction endings)
    # Real words: I, a, an, or, so, oh, no, ok, to, go, we, he, me, my, by, etc.
    # Contraction endings: 'm, 's, 't, 'd, 've, 're, 'll (these come after apostrophe)
    valid_short_words = {'i', 'a', 'an', 'or', 'so', 'oh', 'no', 'ok', 'to', 'go', 'we', 'he', 
                         'me', 'my', 'by', 'if', 'in', 'on', 'up', 'do', 'be', 'is', 'it', 
                         'at', 'as', 'of', 'am', 'us'}
    # Also preserve contraction endings (single chars that follow apostrophe)
    contraction_endings = {'m', 's', 't', 'd', 've', 're', 'll'}
    
    def remove_garbage_words(match):
        word = match.group(0)
        # Check if word is valid
        if word.lower() in valid_short_words:
            return word
        # Check if it's a contraction ending (preceded by apostrophe in original)
        # We need to check context - if there's apostrophe before, keep it
        start = match.start()
        if start > 0 and result[start-1] in "'''" + chr(8217):
            # This is a contraction ending, keep it
            return word
        return ''
    
    # Remove 1-2 char words that aren't in valid list
    result = re.sub(r'\b[a-zA-Z]{1,2}\b', remove_garbage_words, result)
    
    # 17d. Remove consecutive short fragments (like "st I've")
    # Pattern: 2-3 short fragments in a row
    result = re.sub(r'(\s+[a-z]{1,3}){3,}(?=\s|$)', '', result)
    
    # 17e. Clean up leftover multiple spaces
    result = re.sub(r'\s{2,}', ' ', result)
    
    # 17f. Clean up orphan punctuation left after removal
    result = re.sub(r'\s+([,;:])\s*', r'\1 ', result)
    result = re.sub(r'^\s*[,;:]\s*', '', result)  # Remove leading comma/etc
    
    if mode in ["moderate", "strict"]:
        # Additional cleanup for these modes
        pass
    
    # 18. Ensure proper sentence endings (no trailing ellipsis/fragments)
    # Philosophy: Pressure creates resonance. Punctuation is constraint that births form.
    
    # 18a. If ends with ellipsis, try to find last complete sentence
    if result.endswith('…') or result.endswith('...'):
        # Find last sentence-ending punctuation before the ellipsis
        last_period = result.rfind('.')
        last_question = result.rfind('?')
        last_exclaim = result.rfind('!')
        
        # Find rightmost complete sentence end (but not the trailing ellipsis)
        candidates = [i for i in [last_period, last_question, last_exclaim] 
                      if i > 0 and i < len(result) - 3]  # -3 to exclude "..."
        
        if candidates:
            cut_point = max(candidates) + 1
            # Only cut if we keep at least 20 chars
            if cut_point >= 20:
                result = result[:cut_point]
    
    # 18b. If still no proper ending, add period
    if result and result[-1] not in '.!?':
        # Check if last char is a word boundary
        if result[-1].isalnum() or result[-1] in '"\'"':
            result = result.rstrip() + '.'
    
    # 18c. Clean trailing ellipsis that feels incomplete
    # Replace "word..." with "word." if ellipsis at very end
    if result.endswith('...'):
        # Only if this is truly the end (not mid-sentence ellipsis)
        result = result[:-3].rstrip() + '.'
    
    if result.endswith('…'):
        result = result[:-1].rstrip() + '.'
    
    # In strict mode: additional cleanup
    if mode == "strict":
        # Remove trailing fragments
        result = re.sub(r'\s+\w{1,3}\s*$', '', result)
        # Ensure ends with proper punctuation
        if result and result[-1] not in '.!?':
            result = result.rstrip() + '.'
    
    return result.strip()


def cleanup_dialogue(text: str) -> str:
    """
    Special cleanup for dialogue-heavy text (like text.txt).
    
    Focuses on dialogue markers and conversational flow.
    """
    result = cleanup_output(text, mode="gentle")
    
    # Fix dialogue line starts
    lines = result.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Ensure dialogue lines start with — and capital
        if line.startswith('—'):
            # Already a dialogue line, ensure proper format
            rest = line[1:].strip()
            if rest and rest[0].islower():
                rest = rest[0].upper() + rest[1:]
            line = '— ' + rest
        
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def calculate_garbage_score(text: str) -> float:
    """
    Calculate how much "garbage" (noise) is in text.
    
    Returns:
        Float 0.0-1.0, where higher means more garbage
    """
    if not text or not isinstance(text, str):
        return 0.0
    
    garbage_patterns = [
        r'\.[,?\.]{2,}',      # .,,?
        r'\?[.,]{2,}',        # ?..
        r',[.,]{2,}',         # ,.,
        r'\s+[,\.]\s+[,\.]',  # " , . "
        r'\.{5,}',            # .....
        r'\s{3,}',            # multiple spaces
        r'\b[a-z]\s+[a-z]\s+[a-z]\b',  # single char fragments
    ]
    
    total_garbage = 0
    for pattern in garbage_patterns:
        matches = re.findall(pattern, text)
        total_garbage += len(matches)
    
    # Normalize by text length
    text_len = max(len(text), 1)
    score = min(1.0, (total_garbage * 100) / text_len)
    
    return score


def demo_cleanup():
    """Demo the cleanup functions."""
    test_cases = [
        # Garbage patterns
        "the haze there bed ithe of cherseell she st a let to the cohnnalike",
        "— darling.  \n— thou knot st nou not dow?  \n— yout it.",
        "i love the moke.  \n— and it.  \n— whater ank there fing ring.",
        
        # Subword output (already cleaner)
        "the haze anymore; I'll see. — You're my peace with it.",
        "— Yeah, that lovely medical-grade secret, pour me another drink.",
    ]
    
    print("=" * 60)
    print("  cleanup.py — Output Cleanup Demo")
    print("=" * 60)
    
    for test in test_cases:
        cleaned = cleanup_output(test, mode="moderate")
        score_before = calculate_garbage_score(test)
        score_after = calculate_garbage_score(cleaned)
        
        print(f"\nOriginal ({score_before:.2f}):")
        print(f"  {test[:80]}")
        print(f"Cleaned ({score_after:.2f}):")
        print(f"  {cleaned[:80]}")


if __name__ == "__main__":
    demo_cleanup()
