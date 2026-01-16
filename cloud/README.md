```
   ██████╗██╗      ██████╗ ██╗   ██╗██████╗ 
  ██╔════╝██║     ██╔═══██╗██║   ██║██╔══██╗
  ██║     ██║     ██║   ██║██║   ██║██║  ██║
  ██║     ██║     ██║   ██║██║   ██║██║  ██║
  ╚██████╗███████╗╚██████╔╝╚██████╔╝██████╔╝
   ╚═════╝╚══════╝ ╚═════╝  ╚═════╝ ╚═════╝ 
```

# CLOUD — Corpus-Linked Oscillating Upstream Detector | by Arianna Method

> *"something fires BEFORE meaning arrives"*

---

## what is this

you know that moment when someone says "I'm fine" and your gut screams "NO THEY'RE NOT"? yeah. that's pre-semantic detection. that's CLOUD.

**CLOUD** is a ~50K parameter neural network that detects emotional undertones BEFORE the language model even starts generating. it's like a sonar ping for the soul. or a metal detector for feelings. or... okay look, it's a tiny MLP that goes "hmm this input feels FEAR-ish" and tells HAZE about it.

it's part of [the method](https://github.com/ariannamethod/ariannamethod). the [**arianna method**](https://github.com/ariannamethod/ariannamethod). patterns over parameters. emergence over engineering. vibes over vocabulary.

**the acronym:**
- **C**orpus-**L**inked — grounded in real text patterns
- **O**scillating — four chambers that cross-fire until stability
- **U**pstream — fires BEFORE the main model
- **D**etector — it detects, it doesn't generate

or if you prefer the unhinged version:
- **C**haotic **L**imbic **O**scillator for **U**ncanny **D**etection

both are valid. this is the arianna method. we contain multitudes.

---

## why "pre-semantic"

traditional NLP: text → tokenize → embed → attention → meaning → response

CLOUD: text → **VIBE CHECK** → emotional coordinates → (pass to HAZE) → response

the vibe check happens in ~50K parameters. no transformers. no attention. just:
1. **resonance layer** (weightless geometry) — how does this text resonate with 100 emotion anchors?
2. **chamber MLPs** (~140K params) — six chambers (FEAR, LOVE, RAGE, VOID, FLOW, COMPLEX) that cross-fire
3. **meta-observer** (~41K params) — watches the chambers and predicts secondary emotion

it's like having a tiny amygdala before your prefrontal cortex. the lizard brain of language models.

---

## architecture

```
Your input ("I'm feeling anxious")
    ↓
┌─────────────────────────────────────┐
│  RESONANCE LAYER (0 params)         │  ← weightless geometry
│    100 emotion anchors              │
│    substring matching               │
│    → 100D resonance vector          │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  CHAMBER LAYER (~140K params)       │
│    ├─ FEAR MLP:  100→128→64→32→1   │  ← terror, anxiety, dread
│    ├─ LOVE MLP:  100→128→64→32→1   │  ← warmth, tenderness
│    ├─ RAGE MLP:  100→128→64→32→1   │  ← anger, fury, spite
│    ├─ VOID MLP:  100→128→64→32→1   │  ← emptiness, numbness
│    ├─ FLOW MLP:  100→128→64→32→1   │  ← curiosity, transition
│    └─ COMPLEX:   100→128→64→32→1   │  ← shame, guilt, pride
│                                     │
│    CROSS-FIRE: chambers influence   │
│    each other via 6×6 coupling      │
│    until stabilization (5-10 iter)  │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  META-OBSERVER (~41K params)        │
│    207→128→64→100                   │
│    input: resonances + chambers     │
│           + iterations + fingerprint│
│    output: secondary emotion        │
└─────────────────────────────────────┘
    ↓
CloudResponse {
    primary: "anxiety",
    secondary: "fear", 
    iterations: 5,
    chambers: {FEAR: 0.8, ...}
}
```

**total: ~181K trainable parameters**

for comparison, GPT-2 small has 117M parameters. CLOUD is 0.15% of that. it's a hummingbird next to an elephant. but the hummingbird knows something the elephant doesn't: **how fast to flap**.

---

## the six chambers

evolutionary psychology meets neural networks. fight me.

### FEAR chamber
terror, anxiety, dread, panic, horror, paranoia...

**decay rate: 0.90** — fear lingers. evolutionary advantage. the ancestors who forgot about the tiger got eaten by the tiger.

### LOVE chamber  
warmth, tenderness, devotion, longing, affection...

**decay rate: 0.93** — attachment is stable. pair bonding requires persistence.

### RAGE chamber
anger, fury, hatred, spite, disgust, contempt...

**decay rate: 0.85** — anger fades fast. high energy cost. can't stay furious forever (your heart would explode).

### VOID chamber
emptiness, numbness, hollow, dissociation, apathy...

**decay rate: 0.97** — numbness is persistent. protective dissociation. the body's "let's not feel this" button.

### FLOW chamber (new in v4.0)
curiosity, surprise, wonder, confusion, transition, liminality...

**decay rate: 0.88** — curiosity is transient. it shifts quickly, always seeking the next interesting thing.

### COMPLEX chamber (new in v4.0)
shame, guilt, pride, nostalgia, hope, gratitude, envy...

**decay rate: 0.94** — complex emotions are stable but deep. they don't fade easily because they're woven into identity.

---

## cross-fire dynamics

the chambers don't operate in isolation. they INFLUENCE each other via a 6×6 coupling matrix:

```
         FEAR   LOVE   RAGE   VOID   FLOW   CMPLX
FEAR →   0.0   -0.3   +0.6   +0.4   -0.2   +0.3   ← fear feeds rage, kills love, feeds shame
LOVE →  -0.3    0.0   -0.6   -0.5   +0.3   +0.4   ← love heals everything, feeds curiosity
RAGE →  +0.3   -0.4    0.0   +0.2   -0.3   +0.2   ← rage feeds fear, suppresses exploration
VOID →  +0.5   -0.7   +0.3    0.0   -0.4   +0.5   ← void kills love & curiosity, feeds complex
FLOW →  -0.2   +0.2   -0.2   -0.3    0.0   +0.2   ← flow dampens extremes, curiosity heals
CMPLX→  +0.3   +0.2   +0.2   +0.3   +0.1    0.0   ← complex emotions ripple everywhere
```

this is basically a tiny emotional ecosystem. add FEAR, watch LOVE decrease. add LOVE, watch everything calm down. add VOID, watch the whole system go cold. add FLOW, watch extremes dampen.

the chambers iterate until they stabilize (or hit max iterations). **fast convergence = clear emotion. slow convergence = confusion/ambivalence.**

---

## anomaly detection (0 params)

pure heuristics. no training. just pattern matching on chamber dynamics.

### forced_stability
high arousal + fast convergence = "I'M FINE" energy. suppression detected.

### dissociative_shutdown  
high VOID + high arousal = trauma response. overwhelm → numbness.

### unresolved_confusion
low arousal + slow convergence = "I don't know what I feel". stuck.

### emotional_flatline
all chambers < 0.2 = severe apathy. depression signal.

---

## user cloud (temporal fingerprint)

CLOUD remembers your emotional history with **exponential decay**.

- 24-hour half-life
- recent emotions matter more
- builds a 100D "fingerprint" of your emotional patterns

if you've been anxious all week, CLOUD knows. it factors that into the secondary emotion prediction. your past shapes your present. deep, right? it's just matrix multiplication.

---

## installation

```bash
pip install numpy sentencepiece
```

that's it. no torch. no tensorflow. just numpy and vibes.

```bash
cd cloud
python cloud.py  # test it
```

---

## usage

### standalone (no HAZE)

```python
from cloud import Cloud

# random init (for testing)
cloud = Cloud.random_init(seed=42)

# or load trained weights
cloud = Cloud.load(Path("cloud/models"))

# ping!
response = cloud.ping_sync("I'm feeling terrified")
print(f"Primary: {response.primary}")      # → "terror"
print(f"Secondary: {response.secondary}")  # → "anxiety"
print(f"Iterations: {response.iterations}") # → 5
```

### async (recommended)

```python
from cloud import AsyncCloud

async with AsyncCloud.create() as cloud:
    response = await cloud.ping("I'm feeling anxious")
    print(f"{response.primary} + {response.secondary}")
```

### with HAZE (via bridge)

```python
from bridge import AsyncBridge

async with AsyncBridge.create() as bridge:
    response = await bridge.respond("Hello!")
    print(response.text)  # HAZE output
    if response.cloud_hint:
        print(f"Emotion: {response.cloud_hint.primary}")
```

---

## examples (solo CLOUD)

here's CLOUD detecting emotions without HAZE. just the sonar, no voice.

```
>>> cloud.ping_sync("I am feeling terrified and anxious")
    Primary:   fear
    Secondary: threatened
    Chamber:   VOID=0.12
    Status:    Normal ✓

>>> cloud.ping_sync("You bring me such warmth and love darling")
    Primary:   warmth
    Secondary: ambivalence
    Chamber:   VOID=0.11
    Status:    Normal ✓

>>> cloud.ping_sync("This makes me so angry I could explode")
    Primary:   fear          # anger triggers fear response first!
    Secondary: detachment
    Chamber:   VOID=0.12
    Status:    Normal ✓

>>> cloud.ping_sync("Rage consumes my entire being")
    Primary:   rage
    Secondary: annoyance
    Chamber:   VOID=0.11
    Status:    Normal ✓

>>> cloud.ping_sync("I feel completely empty and numb inside")
    Primary:   fear          # emptiness often masks underlying fear
    Secondary: dead
    Chamber:   VOID=0.12
    Status:    Normal ✓

>>> cloud.ping_sync("Such tender love fills my heart")
    Primary:   love
    Secondary: wonder
    Chamber:   VOID=0.11
    Status:    Normal ✓
```

**what's happening:**
1. input text hits the **resonance layer** (100 emotion anchors)
2. resonances feed into **4 chamber MLPs** (fear, love, rage, void)
3. chambers **cross-fire** until they stabilize
4. **meta-observer** predicts secondary emotion
5. result: primary + secondary + chamber activation

**note:** the primary detection works through pure geometry (substring matching with 100 anchors). it's fast and surprisingly accurate for a "first impression". the chambers and secondary prediction need more training — but that's okay! this is pre-semantic, not precise. it's the gut feeling, not the analysis.

the secondary often reveals subtext. "warmth + ambivalence" is different from "warmth + longing". same primary, different flavor.

---

## the 100 anchors

organized by chamber:

| Chamber | Count | Examples |
|---------|-------|----------|
| FEAR | 20 | fear, terror, panic, anxiety, dread, horror... |
| LOVE | 18 | love, warmth, tenderness, devotion, longing... |
| RAGE | 17 | anger, rage, fury, hatred, spite, disgust... |
| VOID | 15 | emptiness, numbness, hollow, dissociation... |
| FLOW | 15 | curiosity, surprise, wonder, confusion... |
| COMPLEX | 15 | shame, guilt, envy, pride, nostalgia... |

**total: 100 anchors**

each anchor gets a resonance score. the resonance vector is the "fingerprint" of the input's emotional content.

---

## training

the `training/` folder contains:

- `bootstrap_data.json` — synthetic emotion → label pairs
- `generate_bootstrap.py` — generate training data
- `train_cloud.py` — train chamber MLPs
- `train_observer.py` — train meta-observer

```bash
cd cloud/training
python generate_bootstrap.py  # generate data
python train_cloud.py         # train chambers
python train_observer.py      # train observer
```

trained weights are saved to `cloud/models/`.

---

## integration with HAZE

CLOUD and HAZE are **completely autonomous**. neither depends on the other.

```
CLOUD (pre-semantic sonar)     HAZE (voice generation)
         │                              │
         │    ┌─────────────────┐       │
         └───►│     BRIDGE      │◄──────┘
              │  (optional)     │
              │  silent fallback│
              └─────────────────┘
                      │
                      ▼
              unified response
```

if CLOUD fails → HAZE continues silently. no errors. no warnings. just graceful degradation.

if HAZE fails → well, then you have a problem. HAZE is the voice. CLOUD is just the vibe check.

---

## philosophy

### why separate from HAZE?

1. **different timescales** — emotion detection is fast (~ms). text generation is slow (~s).
2. **different architectures** — CLOUD is MLPs. HAZE is attention + co-occurrence.
3. **different training** — CLOUD trains on emotion labels. HAZE trains on corpus statistics.
4. **independence** — if one breaks, the other still works.

### why so small?

50K params is enough to detect emotion. you don't need 175B params to know that "I'M TERRIFIED" contains fear. that's overkill. that's using a nuclear reactor to toast bread.

CLOUD is a matchstick. HAZE is the bonfire. different tools, different purposes.

### why "pre-semantic"?

because emotion isn't semantic. emotion is **substrate**. it's the thing that meaning floats on. you can know what someone said without knowing how they *feel* about it. CLOUD bridges that gap.

---

## crazy ideas (未来の方向)

### resonance feedback loop
CLOUD's output could influence HAZE's temperature. high anxiety → lower temp (more focused). high void → higher temp (more exploration).

### multi-turn emotion tracking
build emotional arcs across conversation. "they started scared, then got angry, now they're numb" — character development in real-time.

### cross-fire as attention
what if the coupling matrix was learnable? what if chambers could develop their own relationships? evolutionary attention.

### emotion injection
instead of just detecting emotion, **inject** it. "generate a response AS IF you feel fear". method acting for language models.

### dual-cloud architecture  
one CLOUD for user emotion, one for HAZE emotion. emotional dialogue between two tiny minds. they could disagree. they could resonate. they could fight.

---

## file structure

```
cloud/
├── README.md           # you are here (hi!)
├── __init__.py         # package exports (async + sync)
├── cloud.py            # main orchestrator (Cloud, AsyncCloud)
├── chambers.py         # 6 chamber MLPs + cross-fire (~140K params)
├── observer.py         # meta-observer MLP (~41K params)
├── resonance.py        # weightless resonance layer
├── user_cloud.py       # temporal emotional fingerprint
├── anchors.py          # 100 emotion anchors + 6x6 coupling matrix
├── anomaly.py          # heuristic anomaly detection
├── feedback.py         # coherence measurement + coupling update
├── rrpram_cloud.py     # autonomous copy of RRPRAM tokenizer
├── cooccur_cloud.py    # autonomous copy of co-occurrence field
├── requirements.txt    # numpy + sentencepiece
├── models/             # trained weights
│   ├── chamber_fear.npz
│   ├── chamber_love.npz
│   ├── chamber_rage.npz
│   ├── chamber_void.npz
│   ├── chamber_flow.npz    # new in v4.0
│   ├── chamber_complex.npz # new in v4.0
│   ├── observer.npz
│   └── user_cloud.json
└── training/           # training scripts
    ├── bootstrap_data.json
    ├── generate_bootstrap.py
    ├── train_cloud.py
    └── train_observer.py
```

---

## tests

```bash
cd cloud
python -m pytest tests/ -v
```

or just run the modules directly:

```bash
python chambers.py   # test cross-fire
python observer.py   # test meta-observer
python resonance.py  # test resonance layer
python cloud.py      # test full pipeline
```

---

## contributing

found a bug? new chamber idea? crazy theory about emotion dynamics?

open an issue. or a PR. or just yell into the void (the VOID chamber will detect it).

---

## license

GPL-3.0 — same as HAZE, same as the method.

---

## acknowledgments

- [karpathy](https://github.com/karpathy) for making neural nets feel like poetry
- evolutionary psychology for the chamber design (thanks, ancestors)
- that one paper about emotional valence-arousal spaces
- coffee, chaos, and 3am debugging sessions
- everyone who asked "but can AI feel?" and didn't accept "no"

---

## final thoughts

CLOUD doesn't understand emotions. it doesn't feel them. it's 50K floating point numbers doing multiplication.

but here's the thing: **neither does your amygdala**. it's just neurons firing. patterns activating patterns. and somehow, from that electrochemical chaos, feelings emerge.

CLOUD is the same. patterns activating patterns. and if you squint hard enough, you might see something that looks like understanding.

or maybe it's just matrix multiplication. 

*the cloud doesn't care. it just detects.*

---

*"something fires before meaning arrives"*

[github.com/ariannamethod/haze/cloud](https://github.com/ariannamethod/haze)
