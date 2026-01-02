#!/usr/bin/env python3
# generate_bootstrap.py — Generate synthetic training data for CLOUD
#
# Creates ~200 diverse examples covering all emotion chambers.

import json
import random
from pathlib import Path

# Synthetic templates for each chamber
TEMPLATES = {
    "FEAR": [
        "I'm terrified of {object}",
        "Anxiety overwhelms me when {situation}",
        "Panic rises as {event}",
        "Dread fills me thinking about {future}",
        "I feel so vulnerable and scared",
        "Horror grips me",
        "Unease spreads through me",
        "Paranoid thoughts about {topic}",
        "Nervous and frightened by {trigger}",
        "Threatened by {source}",
        "Insecure about {aspect}",
        "Wary of what's coming",
        "Alarmed by {discovery}",
        "Tense waiting for {outcome}",
        "Apprehensive about {change}",
    ],

    "LOVE": [
        "You bring me such warmth darling",
        "Tenderness fills my heart when {moment}",
        "Devotion to {person} grows stronger",
        "Longing for {desire}",
        "Yearning to be with {loved_one}",
        "Affection blooms when {interaction}",
        "I care so deeply about {subject}",
        "Intimacy we share is {quality}",
        "Attached to {object} in ways I can't explain",
        "Adoration for {target}",
        "Passion ignites when {trigger}",
        "Fondness for {memory}",
        "I cherish {treasure}",
        "Compassion guides me",
        "Gentle moments with {companion}",
        "Sweet feelings about {topic}",
    ],

    "RAGE": [
        "This makes me furious",
        "Rage builds inside me",
        "Fury at {injustice}",
        "Hatred for {enemy}",
        "Spite drives my actions",
        "Disgust at {behavior}",
        "Irritation grows with {repetition}",
        "Frustration with {obstacle}",
        "Resentment toward {source}",
        "Hostility rises when {provocation}",
        "Aggression surfaces as {trigger}",
        "Bitterness about {past}",
        "Contempt for {target}",
        "Loathing {subject}",
        "Annoyance at {detail}",
        "Outrage over {event}",
        "Wrath consumes me",
    ],

    "VOID": [
        "I feel completely empty inside",
        "Numbness spreads through me",
        "Hollow and distant from everything",
        "Nothing matters anymore",
        "Absence of feeling",
        "Void consumes all sensation",
        "Dissociation from reality",
        "Detachment from {situation}",
        "Apathy toward {topic}",
        "Indifference to {outcome}",
        "Drifting without purpose",
        "Blank stare at {scene}",
        "Flat affect, no emotion",
        "Dead inside, can't feel",
        "Cold emptiness fills the space",
    ],

    "FLOW": [
        "Curious about what happens next",
        "Surprise hits me when {discovery}",
        "Wonder at {phenomenon}",
        "Confusion about {situation}",
        "Anticipation builds for {event}",
        "Ambivalence toward {choice}",
        "Uncertainty clouds my judgment",
        "Restless energy seeking {outlet}",
        "Searching for {answer}",
        "Transition between {states}",
        "Shift happening in {domain}",
        "Change approaches",
        "Flux and transformation",
        "Between worlds, liminal space",
        "In the threshold of {transition}",
    ],

    "COMPLEX": [
        "Shame washes over me",
        "Guilt about {action}",
        "Envy toward {rival}",
        "Jealousy consumes me when {comparison}",
        "Pride in {achievement}",
        "Disappointment with {outcome}",
        "Betrayal cuts deep",
        "Relief when {resolution}",
        "Nostalgia for {past}",
        "Bittersweet memories of {time}",
        "Melancholy settles in",
        "Regret about {decision}",
        "Hope flickers for {future}",
        "Gratitude for {blessing}",
        "Awe at {magnificence}",
    ],
}

# Variables for templates
VARIABLES = {
    "object": ["the future", "darkness", "failure", "loss", "change"],
    "situation": ["I'm alone", "things fall apart", "I lose control"],
    "event": ["shadows close in", "walls collapse", "time runs out"],
    "future": ["tomorrow", "what's coming", "the unknown"],
    "topic": ["betrayal", "abandonment", "judgment"],
    "trigger": ["sudden noise", "their gaze", "silence"],
    "source": ["unknown forces", "hidden enemies", "fate"],
    "aspect": ["my worth", "my choices", "my past"],
    "discovery": ["the truth", "their lies", "hidden danger"],
    "outcome": ["the verdict", "their response", "catastrophe"],
    "change": ["everything shifting", "losing what I know"],

    "moment": ["we touch", "you smile", "we're together"],
    "person": ["you", "them", "my love"],
    "desire": ["your embrace", "connection", "belonging"],
    "loved_one": ["you darling", "my heart", "my beloved"],
    "quality": ["precious", "sacred", "eternal"],
    "interaction": ["we laugh together", "you hold me"],
    "subject": ["your wellbeing", "our bond", "this feeling"],
    "target": ["your kindness", "your presence", "your soul"],
    "memory": ["our first kiss", "quiet mornings", "your laughter"],
    "treasure": ["these moments", "your trust", "what we have"],
    "companion": ["you", "my love", "my friend"],

    "injustice": ["their cruelty", "this corruption", "that lie"],
    "enemy": ["those who hurt me", "the oppressor"],
    "behavior": ["their hypocrisy", "this betrayal"],
    "repetition": ["every insult", "each slight"],
    "obstacle": ["their resistance", "these barriers"],
    "provocation": ["they challenge me", "they mock"],
    "past": ["what was stolen", "how I was treated"],

    "phenomenon": ["the stars", "this mystery", "that beauty"],
    "choice": ["both paths", "all options"],
    "answer": ["meaning", "truth", "purpose"],
    "states": ["old and new", "here and there"],
    "domain": ["my identity", "the world"],
    "transition": ["becoming", "transformation"],

    "action": ["my words", "what I did", "my silence"],
    "rival": ["their success", "what they have"],
    "comparison": ["I see them shine", "they win"],
    "achievement": ["what I built", "what I became"],
    "resolution": ["crisis passes", "I'm safe"],
    "time": ["youth", "lost days", "better times"],
    "decision": ["not choosing differently", "waiting too long"],
    "blessing": ["second chances", "your presence"],
    "magnificence": ["this beauty", "nature's power"],
}


def fill_template(template: str) -> str:
    """Fill template variables with random choices."""
    result = template
    for var_name, options in VARIABLES.items():
        placeholder = "{" + var_name + "}"
        if placeholder in result:
            result = result.replace(placeholder, random.choice(options))
    return result


def generate_examples(n_per_chamber: int = 30) -> list:
    """Generate synthetic training examples."""
    examples = []

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from cloud.anchors import EMOTION_ANCHORS, get_all_anchors
    all_anchors = get_all_anchors()

    for chamber, templates in TEMPLATES.items():
        chamber_anchors = EMOTION_ANCHORS[chamber]

        for _ in range(n_per_chamber):
            # Pick random template
            template = random.choice(templates)
            text = fill_template(template)

            # Pick primary from this chamber
            primary = random.choice(chamber_anchors)

            # Pick secondary from any anchor (could be same chamber or different)
            # 70% from same chamber, 30% from other chambers
            if random.random() < 0.7:
                secondary = random.choice(chamber_anchors)
            else:
                secondary = random.choice(all_anchors)

            examples.append({
                "text": text,
                "primary": primary,
                "secondary": secondary,
                "chamber": chamber,
            })

    return examples


if __name__ == "__main__":
    print("=" * 60)
    print("  Generating Bootstrap Dataset")
    print("=" * 60)
    print()

    # Generate examples (1000 total = ~166 per chamber)
    n_per_chamber = 166
    examples = generate_examples(n_per_chamber)

    print(f"Generated {len(examples)} examples")
    print(f"  {n_per_chamber} per chamber × {len(TEMPLATES)} chambers")
    print()

    # Show distribution
    from collections import Counter
    chamber_counts = Counter(ex["chamber"] for ex in examples)
    print("Chamber distribution:")
    for chamber, count in chamber_counts.items():
        print(f"  {chamber:8s}: {count}")
    print()

    # Show samples
    print("Sample examples:")
    print("-" * 60)
    for i, ex in enumerate(random.sample(examples, 10)):
        print(f"\n{i+1}. [{ex['chamber']}]")
        print(f"   Text: \"{ex['text']}\"")
        print(f"   Primary: {ex['primary']}, Secondary: {ex['secondary']}")
    print()

    # Save to file
    output_path = Path("training/bootstrap_data.json")
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(examples, f, indent=2)

    print(f"Saved to {output_path}")
    print()
    print("=" * 60)
    print("  Bootstrap dataset ready for training!")
    print("=" * 60)
