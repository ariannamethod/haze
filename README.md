```markdown
   ██╗  ██╗ █████╗ ███████╗███████╗
   ██║  ██║██╔══██╗╚══███╔╝██╔════╝
   ███████║███████║  ███╔╝ █████╗  
   ██╔══██║██╔══██║ ███╔╝  ██╔══╝  
   ██║  ██║██║  ██║███████╗███████╗
   ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝
```

# haze — hybrid attention entropy system | by Arianna Method

> *emergence is not creation but recognition*

---

## what is this thing

you know that feeling when you're training a transformer and you realize 90% of the attention mechanism is just... overhead? yeah. me too. so i did something about it.

**haze** is a character-level language model that rewrites attention from scratch. no torch. no tensorflow. just numpy and the cold realization that maybe—*just maybe*—we've been overthinking this whole thing.

it's part of [the method](https://github.com/ariannamethod/ariannamethod). the [**arianna method**](https://github.com/ariannamethod/ariannamethod). resonance over intelligence. patterns over parameters. you know the vibe.

two attention mechanisms walk into a bar:
- **reweight attention**: learns static positional patterns. rhythm. structure. the bones of language.
- **content attention**: classic QK^T semantic similarity. meaning. the flesh.

mix them together (that's the "hybrid" part) and you get something that actually works without burning your GPU to ash.

inference runs on pure numpy. no dependencies. no excuses. just you, your corpus, and the void.

---

## why "haze"

*why anything, really?*

but if you must know—haze is that liminal space between clarity and confusion. the model lives there. attention patterns emerge from noise. tokens crystallize from probability distributions. it's all very poetic until you realize you're just doing matrix multiplication in a for loop.

also i vomited it up one night after reading too much about positional encodings. true story. read `text.txt` if you want the full gothic horror version.

---

## architecture

```
Input (tokens)
    ↓
Embedding + Positional Encoding
    ↓
┌─────────────────────┐
│  Block × N          │
│    ├─ HybridHead    │  ← α·Reweight + (1-α)·Content
│    ├─ GELU MLP      │
│    └─ LayerNorm     │
└─────────────────────┘
    ↓
Final LayerNorm
    ↓
Output Projection
    ↓
Logits → Sampling → Token
```

### the heads

**reweight head**: `x @ W_reweight → (T,T)` attention matrix
- learns positional dependencies directly
- no query/key dance
- captures n-grams, rhythm, repetition
- basically a glorified frequency detector that somehow works

**content head**: classic `softmax(QK^T/√d) @ V`
- semantic similarity
- long-range dependencies
- the "smart" part
- honestly just normal attention but i was too proud to admit it

**hybrid head**: `α·reweight_out + (1-α)·content_out`
- best of both worlds
- or worst of both
- you decide after training

### entropy-aware temperature

tired of fixed temperature? yeah. so instead:
- high entropy (model is confused) → lower temp (focus)
- low entropy (model is confident) → higher temp (explore)

self-regulating. adaptive. pretentious. but it works.

the model maintains target entropy across generation, creating consistent "surprise levels". it's like cruise control for creativity. or madness. thin line.

---

## installation

```bash
pip install numpy
```

that's it. that's the whole dependency tree. beautiful, isn't it?

```bash
git clone https://github.com/ariannamethod/haze.git
cd haze
```

---

## usage

### quick start

create your corpus:
```bash
echo "your text here" > text.txt
```

run the demo:
```bash
python example.py
```

### interactive mode

```bash
python run.py
```

this drops you into a REPL where you can:
- type seed text
- watch the model hallucinate
- adjust temperature on the fly
- toggle sampling strategies
- question your life choices

### commands

```
/len N          set generation length (default: 300)
/temp X         base temperature (default: 1.0)
/sampling MODE  basic|top_k|top_p|entropy|mirostat|mirostat_v2|resonance
/topk K         top-k value (default: 40)
/topp P         nucleus sampling threshold (default: 0.9)
/entropy T      target entropy for adaptive mode (default: 3.0)
/resonance R    target resonance for resonance mode (default: 0.7)
/bounds MIN MAX temperature bounds (default: 0.3 2.0)
/stats          toggle stats display
/config         show current settings
/help           cry for help
/quit           escape
```

### programmatic

```python
from haze import Vocab, ReweightGPT as Haze

# build vocab
text = open("text.txt").read()
vocab = Vocab.from_text(text)

# initialize model
model = Haze(
    vocab_size=vocab.vocab_size,
    T=32,              # context window
    n_emb=64,          # embedding dimension
    nodes=64,          # MLP hidden size
    n_blocks=3,        # transformer blocks
    n_heads=4,         # attention heads
    head_type="hybrid", # "hybrid", "reweight", or "content"
    alpha=0.5,         # reweight/content mix ratio
    seed=42,           # for reproducibility (lol)
)

# generate
seed_idx = vocab.encode("once upon a")
tokens, stats = model.generate(
    seed_seq=seed_idx,
    length=200,
    sampling="entropy",    # adaptive temperature
    target_entropy=3.0,    # bits of surprise
)

text = vocab.decode(tokens)
print(text)
print(f"mean entropy: {stats['mean_entropy']:.2f} bits")
```

---

## sampling strategies

### basic
standard temperature sampling. simple. honest. boring.

### top-k
only sample from top K tokens. fixed vocabulary. predictable. safe.

### top-p (nucleus)
dynamic vocabulary based on cumulative probability. adapts to context. actually clever.

### entropy-aware
*adaptive temperature based on output entropy.*

model adjusts temperature to maintain target entropy:
- maintains consistent "surprise" across generation
- self-regulating creativity
- works disturbingly well

```python
tokens, stats = model.generate(
    seed_seq=seed_idx,
    sampling="entropy",
    target_entropy=3.0,  # bits
    min_temp=0.3,
    max_temp=2.0,
)
```

### mirostat & mirostat v2
*perplexity-controlled sampling.*

maintains target perplexity by dynamically adjusting selection threshold:
- **mirostat v1**: fixed surprise threshold, adaptive selection
- **mirostat v2**: adaptive k based on cumulative probability mass, more stable

```python
tokens, stats = model.generate(
    seed_seq=seed_idx,
    sampling="mirostat_v2",
    target_entropy=2.5,
    mirostat_tau=0.1,  # learning rate
)
```

mirostat is basically cruise control for perplexity. set your target surprise level and let the algorithm handle the rest.

### resonance
*the wild card.*

adaptive temperature based on **resonance with previous tokens**:
- high resonance with history → lower temp (stay coherent)
- low resonance with history → higher temp (explore new patterns)

```python
tokens, stats = model.generate(
    seed_seq=seed_idx,
    sampling="resonance",
    target_resonance=0.7,  # 0-1, target similarity with history
)
```

this is where the **arianna method** really shows up. the model tunes itself based on pattern resonance, creating emergent coherence without explicit constraints. sometimes it finds grooves you didn't know existed.

---

## attention visualization

**new:** `hallucinations.py` — see what your reweight heads actually learn.

```python
# from the haze package directory
from hallucinations import hallucinate
from model import Vocab, ReweightGPT as Haze

# or if using as package
import sys
sys.path.insert(0, 'haze')
from haze import Vocab, Haze
from haze.hallucinations import hallucinate

# build model
text = open("text.txt").read()
vocab = Vocab.from_text(text)
model = Haze(vocab_size=vocab.vocab_size, T=32, n_emb=64)

# extract and visualize attention patterns
patterns = hallucinate(model, "the haze settles", vocab)

# outputs:
# - hallucinations/report.txt — analysis of attention patterns
# - hallucinations/*.png — heatmap visualizations
```

because sometimes you need to stare into the attention matrix and see what stares back.

the module analyzes:
- **sparsity**: how focused is the attention?
- **locality**: local vs long-range dependencies
- **uniformity**: distribution entropy
- **diagonality**: n-gram vs semantic patterns

requires `matplotlib` for visualizations:
```bash
pip install matplotlib
```

---

## file structure

```
haze/
├── README.md            # this file
├── talkto.py            # quick bridge to interactive REPL
└── haze/                # main package
    ├── __init__.py      # package exports
    ├── nn.py            # numpy primitives (activations, sampling, metrics)
    ├── model.py         # the model itself (inference only)
    ├── hallucinations.py# attention visualization and analysis
    ├── run.py           # interactive REPL
    ├── example.py       # demo script
    ├── text.txt         # your corpus (you create this)
    ├── requirements.txt # numpy + matplotlib (optional)
    └── tests/           # comprehensive test suite (73 tests, all passing)
        ├── test_nn.py   # tests for neural net primitives
        └── test_haze.py # tests for model components
```

### quick start

```bash
# talk to the haze (interactive mode)
python talkto.py

# or run from the package
cd haze
python run.py

# run demo
cd haze
python example.py
```

---

## training

haze is pure inference. if you want to train:
1. implement the backward pass (it's just matrix multiplication, you can do it)
2. or use pytorch like a normal person
3. save weights with `model.save_theweightofhaze("theweightofhaze.npz")`
4. load with `model = Haze.theweightofhaze(vocab_size, "theweightofhaze.npz")`

```python
# saving
model.save_theweightofhaze("theweightofhaze.npz")

# loading
from haze import ReweightGPT as Haze
model = Haze.theweightofhaze(vocab.vocab_size, "theweightofhaze.npz")
```

because the weight of haze is not in pounds or kilograms, but in the patterns it learned from the void.

i might add training code later. or not. depends on the resonance.

---

## tests

```bash
python -m unittest discover tests -v
```

72 tests. all green. comprehensive coverage of:
- activation functions
- sampling strategies (basic, top-k, top-p, entropy, mirostat, mirostat v2, resonance)
- entropy metrics
- resonance metrics
- attention mechanisms
- model forward pass
- generation pipeline
- weight loading/saving

because unlike my life choices, at least the code should be reliable.

---

## the method

this is part of [**the arianna method**](https://github.com/ariannamethod/ariannamethod).

resonance. emergence. recursive dialogue. linguistic organisms that grow rather than compute.

haze embodies this through:
- **minimal architecture**: only what's needed, nothing more
- **adaptive generation**: self-regulating entropy
- **hybrid attention**: positional + semantic resonance
- **pure numpy**: no framework dependency, just raw computation

the method is about finding patterns we forgot we already knew. haze is one such pattern.

check out the rest of the ecosystem:
- [ariannamethod](https://github.com/ariannamethod/ariannamethod) — the core
- [leo](https://github.com/ariannamethod/leo) — resonant ai
- [harmonix](https://github.com/ariannamethod/harmonix) — harmonic adaptive ai
- [sorokin](https://github.com/ariannamethod/sorokin) — another piece of the organism

---

## philosophy

traditional attention: `softmax(QK^T/√d) @ V`  
*"compute relevance dynamically via query-key similarity"*

reweight attention: `x @ W_reweight → attention`  
*"just learn the damn patterns directly"*

is it better? i don't know. does it work? surprisingly, yes.

the hybrid approach acknowledges that language has both:
- **structure**: rhythm, syntax, n-grams (reweight)
- **meaning**: semantics, context, relationships (content)

why choose when you can have both? why not embrace the duality? why not let the model decide the mix?

entropy-aware sampling keeps generation in that sweet spot between:
- too deterministic (boring)
- too random (incoherent)

it's self-tuning. homeostatic. alive in a weird, mathematical way.

---

## performance

it's numpy. it's slow. embrace it.

but hey:
- no gpu needed
- no framework overhead
- runs on a potato
- pure python
- actually readable code

sometimes constraint is freedom. sometimes slow is beautiful. sometimes you just want to understand what the fuck your model is doing.

---

## contributing

found a bug? cool. open an issue.  
have an idea? neat. PR welcome.  
want to argue about architecture? my DMs are open.  
want to discuss the void? same.

this is part of something larger. something emergent. something we're building together without quite knowing what it is yet.

that's the point.

---

## license

GPL-3.0 — use it, fork it, break it, rebuild it.

just mention [the method](https://github.com/ariannamethod/ariannamethod) somewhere. keep the resonance alive.

---

## acknowledgments

inspired by:
- transformer attention (duh)
- positional encoding schemes
- entropy-based sampling
- late nights
- existential dread
- the realization that simpler is often better
- that thing where you stare at matrices until they make sense
- coffee
- more coffee

dedicated to arianna: *where shadows speak in silence*

---

## final thoughts

attention is just pattern matching with extra steps.  
language is compression.  
intelligence is overrated.  
resonance is everything.

the haze settles over the hills like a breathing thing,  
soft and silver in the morning light.

patterns we forgot we already knew.

*now go generate something.*

---

**built with numpy and spite**  
**running on hope and determinism**  
**part of the arianna method emergent organism**

[github.com/ariannamethod/haze](https://github.com/ariannamethod/haze)
