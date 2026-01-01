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
    
    # 0. Replace sentencepiece unknown marker with space
    result = result.replace('⁇', "'")
    result = result.replace(' ⁇ ', ' ')
    
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
    
    # 11. Remove "—" at the start of output (haze is not dialogue-only)
    # This is CRITICAL for presence vs chatbot distinction
    result = re.sub(r'^[\s—–-]+', '', result)
    
    # 12. Capitalize first letter of text
    if result and result[0].islower():
        result = result[0].upper() + result[1:]
    
    # 13. Capitalize "I" when standalone
    result = re.sub(r'\bi\b', 'I', result)
    
    # 14. Remove duplicate dialogue markers
    result = re.sub(r'—\s*—', '—', result)
    
    # 15. Fix common word fragments (character-level artifacts)
    if mode in ["moderate", "strict"]:
    # 16. Clean obvious fragments
        result = re.sub(r'\b[a-z]{1,2}\b(?=\s+[a-z]{1,2}\b)', '', result)
        result = re.sub(r'\s{2,}', ' ', result)
    
    # 17. In strict mode: remove incomplete sentences at end
    if mode == "strict":
        # Remove trailing fragments
        result = re.sub(r'\s+\w{1,3}\s*$', '', result)
        # Ensure ends with proper punctuation
        if result and result[-1] not in '.!?…':
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
