#!/usr/bin/env python3
# anchors.py — 100 Emotion Anchors for CLOUD v3.0
#
# Four chambers of the human condition:
# - FEAR (20): anxiety, terror, dread...
# - LOVE (18): warmth, tenderness, devotion...
# - RAGE (17): anger, fury, hatred...
# - VOID (15): emptiness, numbness, hollow...
# - FLOW (15): curiosity, surprise, transition...
# - COMPLEX (15): shame, guilt, pride, nostalgia...
#
# Each chamber gets its own MLP. Cross-fire happens between chambers.

from typing import Dict, List, Tuple

# 100 emotion anchor words organized by chamber
EMOTION_ANCHORS: Dict[str, List[str]] = {
    # FEAR (20) — terror, anxiety, dread
    "FEAR": [
        "fear", "terror", "panic", "anxiety", "dread", "horror",
        "unease", "paranoia", "worry", "nervous", "scared",
        "frightened", "alarmed", "tense", "apprehensive",
        "threatened", "vulnerable", "insecure", "timid", "wary",
    ],

    # LOVE (18) — warmth, connection, tenderness
    "LOVE": [
        "love", "warmth", "tenderness", "devotion", "longing",
        "yearning", "affection", "care", "intimacy", "attachment",
        "adoration", "passion", "fondness", "cherish", "desire",
        "compassion", "gentle", "sweet",
    ],

    # RAGE (17) — anger, fury, spite
    "RAGE": [
        "anger", "rage", "fury", "hatred", "spite", "disgust",
        "irritation", "frustration", "resentment", "hostility",
        "aggression", "bitterness", "contempt", "loathing",
        "annoyance", "outrage", "wrath",
    ],

    # VOID (15) — emptiness, numbness, dissociation
    "VOID": [
        "emptiness", "numbness", "hollow", "nothing", "absence",
        "void", "dissociation", "detachment", "apathy",
        "indifference", "drift", "blank", "flat", "dead", "cold",
    ],

    # FLOW (15) — curiosity, transition, liminality
    "FLOW": [
        "curiosity", "surprise", "wonder", "confusion",
        "anticipation", "ambivalence", "uncertainty", "restless",
        "searching", "transition", "shift", "change", "flux",
        "between", "liminal",
    ],

    # COMPLEX (15) — shame, guilt, nostalgia, bittersweet
    "COMPLEX": [
        "shame", "guilt", "envy", "jealousy", "pride",
        "disappointment", "betrayal", "relief", "nostalgia",
        "bittersweet", "melancholy", "regret", "hope",
        "gratitude", "awe",
    ],
}

# Chamber names (for indexing) - original 4 chambers
CHAMBER_NAMES = ["FEAR", "LOVE", "RAGE", "VOID"]

# Extended chamber names (6 chambers for 200K model)
CHAMBER_NAMES_EXTENDED = ["FEAR", "LOVE", "RAGE", "VOID", "FLOW", "COMPLEX"]

# Coupling matrix: how chambers influence each other (original 4x4)
# Rows = influence FROM, Cols = influence TO
# Format: [FEAR, LOVE, RAGE, VOID]
COUPLING_MATRIX = [
    #     FEAR   LOVE   RAGE   VOID
    [     0.0,  -0.3,  +0.6,  +0.4  ],  # FEAR → suppresses love, feeds rage & void
    [    -0.3,   0.0,  -0.6,  -0.5  ],  # LOVE → suppresses fear, rage & void
    [    +0.3,  -0.4,   0.0,  +0.2  ],  # RAGE → feeds fear, suppresses love, feeds void
    [    +0.5,  -0.7,  +0.3,   0.0  ],  # VOID → feeds fear & rage, kills love
]

# Extended coupling matrix (6x6) for FLOW and COMPLEX chambers
# FLOW: curiosity, transition — dampens extremes, feeds exploration
# COMPLEX: shame, guilt, pride — interacts with all, especially love/void
COUPLING_MATRIX_EXTENDED = [
    #     FEAR   LOVE   RAGE   VOID   FLOW   CMPLX
    [     0.0,  -0.3,  +0.6,  +0.4,  -0.2,  +0.3  ],  # FEAR → feeds complex (shame from fear)
    [    -0.3,   0.0,  -0.6,  -0.5,  +0.3,  +0.4  ],  # LOVE → feeds flow & complex (hope, gratitude)
    [    +0.3,  -0.4,   0.0,  +0.2,  -0.3,  +0.2  ],  # RAGE → suppresses flow, feeds complex (guilt)
    [    +0.5,  -0.7,  +0.3,   0.0,  -0.4,  +0.5  ],  # VOID → kills flow, feeds complex (melancholy)
    [    -0.2,  +0.2,  -0.2,  -0.3,   0.0,  +0.2  ],  # FLOW → dampens extremes, curiosity heals
    [    +0.3,  +0.2,  +0.2,  +0.3,  +0.1,   0.0  ],  # COMPLEX → feeds all slightly (ripple effect)
]


def get_all_anchors() -> List[str]:
    """Get flat list of all 100 emotion anchors."""
    anchors = []
    for chamber_anchors in EMOTION_ANCHORS.values():
        anchors.extend(chamber_anchors)
    return anchors


def get_anchor_to_chamber() -> Dict[str, str]:
    """Map each anchor word to its chamber."""
    mapping = {}
    for chamber, words in EMOTION_ANCHORS.items():
        for word in words:
            mapping[word] = chamber
    return mapping


def get_anchor_index(anchor: str) -> int:
    """Get the index (0-99) of an anchor word."""
    all_anchors = get_all_anchors()
    try:
        return all_anchors.index(anchor)
    except ValueError:
        raise ValueError(f"Unknown anchor: {anchor}")


def get_chamber_ranges() -> Dict[str, Tuple[int, int]]:
    """
    Get the index ranges for each chamber in the 100D resonance vector.

    Returns:
        Dict mapping chamber name to (start_idx, end_idx) tuple.

    Example:
        {"FEAR": (0, 20), "LOVE": (20, 38), ...}
    """
    ranges = {}
    idx = 0
    for chamber, words in EMOTION_ANCHORS.items():
        start = idx
        end = idx + len(words)
        ranges[chamber] = (start, end)
        idx = end
    return ranges


def get_chamber_for_anchor(anchor: str) -> str:
    """Get the chamber name for a given anchor word."""
    mapping = get_anchor_to_chamber()
    return mapping.get(anchor, "UNKNOWN")


# Sanity check
assert len(get_all_anchors()) == 100, "Must have exactly 100 anchors"
assert len(CHAMBER_NAMES) == 4, "Must have exactly 4 base chambers"
assert len(CHAMBER_NAMES_EXTENDED) == 6, "Must have exactly 6 extended chambers"
assert len(COUPLING_MATRIX) == 4, "Coupling matrix must be 4x4"
assert all(len(row) == 4 for row in COUPLING_MATRIX), "Coupling matrix must be 4x4"
assert len(COUPLING_MATRIX_EXTENDED) == 6, "Extended coupling matrix must be 6x6"
assert all(len(row) == 6 for row in COUPLING_MATRIX_EXTENDED), "Extended coupling matrix must be 6x6"


if __name__ == "__main__":
    print("=" * 60)
    print("  CLOUD v3.0 — Emotion Anchors")
    print("=" * 60)
    print()

    # Show chamber stats
    print("Chamber distribution:")
    for chamber, words in EMOTION_ANCHORS.items():
        print(f"  {chamber:8s}: {len(words):2d} anchors")
    print(f"  {'TOTAL':8s}: {len(get_all_anchors()):2d} anchors")
    print()

    # Show chamber ranges
    print("Chamber ranges in 100D vector:")
    for chamber, (start, end) in get_chamber_ranges().items():
        print(f"  {chamber:8s}: [{start:2d}:{end:2d}]")
    print()

    # Show coupling matrix
    print("Coupling matrix (cross-fire influence):")
    print("       ", "  ".join(f"{name:6s}" for name in CHAMBER_NAMES))
    for i, row_name in enumerate(CHAMBER_NAMES):
        values = "  ".join(f"{val:+6.1f}" for val in COUPLING_MATRIX[i])
        print(f"  {row_name:6s}  {values}")
    print()

    # Show sample anchors
    print("Sample anchors from each chamber:")
    for chamber, words in EMOTION_ANCHORS.items():
        sample = words[:5]
        print(f"  {chamber:8s}: {', '.join(sample)}...")
    print()

    print("=" * 60)
    print("  Chambers ready. Cross-fire enabled.")
    print("=" * 60)
