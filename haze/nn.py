# nn.py — NumPy primitives for Reweight-GPT
# No PyTorch, no dependencies beyond numpy

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple

# ----------------- RNG -----------------


def get_rng(seed: Optional[int] = None) -> np.random.Generator:
    """Get a numpy random generator, optionally seeded."""
    return np.random.default_rng(seed)


# ----------------- weight init -----------------


def init_weight(
    shape: tuple,
    rng: np.random.Generator,
    scale: float = 0.02,
) -> np.ndarray:
    """Xavier-ish initialization."""
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def init_weight_orthogonal(
    shape: tuple,
    rng: np.random.Generator,
    gain: float = 1.0,
) -> np.ndarray:
    """Orthogonal initialization — better for deep networks."""
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = rng.standard_normal(flat_shape).astype(np.float32)
    u, _, vt = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else vt
    q = q.reshape(shape)
    return (gain * q).astype(np.float32)


# ----------------- activations -----------------


def relu(x: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit."""
    return np.maximum(x, 0)


def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Leaky ReLU — avoids dead neurons."""
    return np.where(x > 0, x, alpha * x)


def gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit — smoother gradients than ReLU."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def swish(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Swish activation: x * sigmoid(beta * x)."""
    return x * sigmoid(beta * x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid with numerical stability."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = x.max(axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)


# ----------------- normalization -----------------


def layer_norm(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Layer normalization: (x - mean) / std * gamma + beta
    x: (..., n_emb)
    gamma, beta: (n_emb,)
    """
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta


def rms_norm(
    x: np.ndarray,
    gamma: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    RMSNorm — simpler than LayerNorm, no mean subtraction.
    Used in LLaMA and other modern architectures.
    """
    rms = np.sqrt((x**2).mean(axis=-1, keepdims=True) + eps)
    return (x / rms) * gamma


# ----------------- sampling strategies -----------------


def sample_basic(
    logits: np.ndarray,
    temperature: float,
    rng: np.random.Generator,
) -> int:
    """Basic temperature sampling."""
    if temperature <= 0:
        return int(np.argmax(logits))
    logits = logits / temperature
    probs = softmax(logits)
    return int(rng.choice(len(probs), p=probs))


def sample_top_k(
    logits: np.ndarray,
    k: int,
    temperature: float,
    rng: np.random.Generator,
) -> int:
    """Top-k sampling — only consider top k tokens."""
    if temperature <= 0:
        return int(np.argmax(logits))

    logits = logits.copy()
    if k < len(logits):
        # zero out everything except top k
        top_k_idx = np.argpartition(logits, -k)[-k:]
        mask = np.full_like(logits, -np.inf)
        mask[top_k_idx] = logits[top_k_idx]
        logits = mask

    logits = logits / temperature
    probs = softmax(logits)
    return int(rng.choice(len(probs), p=probs))


def sample_top_p(
    logits: np.ndarray,
    p: float,
    temperature: float,
    rng: np.random.Generator,
) -> int:
    """
    Nucleus (top-p) sampling — dynamic vocabulary based on cumulative probability.
    More adaptive than top-k: expands vocabulary when uncertain, contracts when confident.
    """
    if temperature <= 0:
        return int(np.argmax(logits))

    logits = logits / temperature
    probs = softmax(logits)

    # sort by probability descending
    sorted_idx = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    cumsum = np.cumsum(sorted_probs)

    # find cutoff where cumulative prob exceeds p
    cutoff_idx = np.searchsorted(cumsum, p) + 1
    cutoff_idx = min(cutoff_idx, len(probs))

    # mask out tokens below threshold
    mask = np.zeros_like(probs)
    mask[sorted_idx[:cutoff_idx]] = 1.0
    probs = probs * mask
    probs = probs / (probs.sum() + 1e-10)

    return int(rng.choice(len(probs), p=probs))


def sample_mirostat(
    logits: np.ndarray,
    target_entropy: float,
    tau: float,  # learning rate for surprise adjustment
    mu: float,   # current surprise target (mutable state)
    rng: np.random.Generator,
) -> Tuple[int, float]:
    """
    Mirostat sampling — maintains target entropy/perplexity.
    Returns (token_id, new_mu).
    
    Adaptive: adjusts selection based on how surprising each choice is.
    """
    probs = softmax(logits)
    sorted_idx = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]

    # find k where sum of top-k probs ≈ covers target surprise
    cumsum = np.cumsum(sorted_probs)
    surprises = -np.log2(sorted_probs + 1e-10)

    # find tokens with surprise less than mu
    valid_mask = surprises <= mu
    if not valid_mask.any():
        # fallback: just take top token
        k = 1
    else:
        k = max(1, valid_mask.sum())

    # sample from top-k
    top_k_idx = sorted_idx[:k]
    top_k_probs = probs[top_k_idx]
    top_k_probs = top_k_probs / top_k_probs.sum()

    choice_local = rng.choice(len(top_k_probs), p=top_k_probs)
    token_id = int(top_k_idx[choice_local])

    # update mu based on observed surprise
    observed_surprise = -np.log2(probs[token_id] + 1e-10)
    new_mu = mu - tau * (observed_surprise - target_entropy)

    return token_id, new_mu


def sample_mirostat_v2(
    logits: np.ndarray,
    target_entropy: float,
    tau: float,  # learning rate for surprise adjustment
    mu: float,   # current surprise target (mutable state)
    rng: np.random.Generator,
) -> Tuple[int, float]:
    """
    Mirostat v2 sampling — improved version with adaptive k.
    Returns (token_id, new_mu).
    
    Differences from v1:
    - Uses normalized probabilities for better stability
    - Adaptive k based on cumulative probability mass
    - More aggressive mu adjustment
    """
    probs = softmax(logits)
    sorted_idx = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    
    # compute surprises (negative log probabilities)
    surprises = -np.log2(sorted_probs + 1e-10)
    
    # find adaptive k: tokens where cumulative surprise < mu threshold
    cumulative_surprise = np.cumsum(surprises * sorted_probs)
    
    # adaptive k: where normalized cumulative surprise crosses threshold
    threshold = mu * np.sum(sorted_probs)
    valid_mask = cumulative_surprise <= threshold
    
    if not valid_mask.any():
        k = 1
    else:
        k = max(1, valid_mask.sum())
    
    # ensure k is reasonable (at least 1, at most half the vocab)
    k = min(k, len(logits) // 2 + 1)
    
    # sample from top-k with renormalized probabilities
    top_k_idx = sorted_idx[:k]
    top_k_probs = sorted_probs[:k]
    top_k_probs = top_k_probs / top_k_probs.sum()
    
    choice_local = rng.choice(len(top_k_probs), p=top_k_probs)
    token_id = int(top_k_idx[choice_local])
    
    # update mu with error correction
    observed_surprise = -np.log2(probs[token_id] + 1e-10)
    error = observed_surprise - target_entropy
    new_mu = mu - tau * error
    
    # clip mu to reasonable range
    new_mu = np.clip(new_mu, target_entropy * 0.5, target_entropy * 3.0)
    
    return token_id, new_mu


# ----------------- entropy metrics -----------------


def entropy(probs: np.ndarray, eps: float = 1e-10) -> float:
    """Shannon entropy of probability distribution (in nats)."""
    probs = np.clip(probs, eps, 1.0)
    return float(-np.sum(probs * np.log(probs)))


def entropy_bits(probs: np.ndarray, eps: float = 1e-10) -> float:
    """Shannon entropy in bits (log2)."""
    probs = np.clip(probs, eps, 1.0)
    return float(-np.sum(probs * np.log2(probs)))


def perplexity(logits: np.ndarray, target_idx: int) -> float:
    """Perplexity for single prediction: 1/p(target)."""
    probs = softmax(logits)
    return 1.0 / max(probs[target_idx], 1e-10)


def cross_entropy(logits: np.ndarray, target_idx: int, eps: float = 1e-10) -> float:
    """Cross-entropy loss for single prediction."""
    probs = softmax(logits)
    return float(-np.log(max(probs[target_idx], eps)))


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """KL divergence: D_KL(P || Q)."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


# ----------------- entropy-aware temperature -----------------


def entropy_temperature(
    logits: np.ndarray,
    target_entropy: float = 2.0,
    min_temp: float = 0.3,
    max_temp: float = 2.0,
    smoothing: float = 0.5,
) -> float:
    """
    Compute adaptive temperature based on current entropy vs target.
    
    - High entropy (uncertain) → lower temperature (more focused)
    - Low entropy (confident) → higher temperature (more exploration)
    
    This creates a self-regulating system that maintains consistent
    "surprise level" across different contexts.
    """
    probs = softmax(logits)
    current_entropy = entropy_bits(probs)

    # ratio-based adjustment
    if current_entropy < 1e-6:
        return min_temp

    ratio = target_entropy / current_entropy

    # smooth the adjustment
    temp = ratio ** smoothing

    return float(np.clip(temp, min_temp, max_temp))


def confidence_score(logits: np.ndarray) -> float:
    """
    Confidence score: how certain is the model?
    Returns value in [0, 1] where 1 = very confident.
    """
    probs = softmax(logits)
    max_prob = probs.max()
    return float(max_prob)


def margin_score(logits: np.ndarray) -> float:
    """
    Margin between top-1 and top-2 predictions.
    Higher margin = more confident distinction.
    """
    if len(logits) < 2:
        return 1.0
    probs = softmax(logits)
    sorted_probs = np.sort(probs)[::-1]
    return float(sorted_probs[0] - sorted_probs[1])


def resonance_temperature(
    current_logits: np.ndarray,
    history_logits: list[np.ndarray],
    target_resonance: float = 0.7,
    min_temp: float = 0.3,
    max_temp: float = 2.0,
    smoothing: float = 0.5,
) -> float:
    """
    Adaptive temperature based on resonance with previous generations.
    
    High resonance with history → lower temp (continue the pattern)
    Low resonance with history → higher temp (explore new territory)
    
    Args:
        current_logits: current token prediction logits
        history_logits: list of previous token logits
        target_resonance: desired resonance level (0-1)
        min_temp, max_temp: temperature bounds
        smoothing: adjustment smoothing factor
    
    Returns:
        adaptive temperature value
    """
    if not history_logits or len(history_logits) == 0:
        # no history, use neutral temperature
        return (min_temp + max_temp) / 2.0
    
    # compute resonance with recent history
    # weight recent tokens more heavily
    weights = np.exp(-np.arange(len(history_logits)) / 5.0)[::-1]
    weights = weights / weights.sum()
    
    resonance_scores = []
    for hist_logits in history_logits:
        score = resonance_score(current_logits, hist_logits)
        resonance_scores.append(score)
    
    # weighted average resonance
    avg_resonance = float(np.average(resonance_scores, weights=weights))
    
    # adjust temperature based on resonance
    # high resonance → low temp (stay coherent)
    # low resonance → high temp (increase exploration)
    if avg_resonance > target_resonance:
        # too much resonance, increase temperature to diversify
        ratio = avg_resonance / target_resonance
        temp = (min_temp + max_temp) / 2.0 * (ratio ** smoothing)
    else:
        # too little resonance, decrease temperature to find patterns
        ratio = target_resonance / (avg_resonance + 1e-6)
        temp = (min_temp + max_temp) / 2.0 / (ratio ** smoothing)
    
    return float(np.clip(temp, min_temp, max_temp))


# ----------------- resonance metrics (for your ecosystem) -----------------


def resonance_score(
    query_logits: np.ndarray,
    context_logits: np.ndarray,
) -> float:
    """
    Measure resonance between two probability distributions.
    High resonance = similar uncertainty patterns.
    """
    p = softmax(query_logits)
    q = softmax(context_logits)

    # Jensen-Shannon divergence (symmetric, bounded)
    m = 0.5 * (p + q)
    js = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

    # convert to similarity (0 = identical, 1 = maximally different)
    # invert for resonance score
    return float(1.0 - np.sqrt(js / np.log(2)))


def harmonic_mean(values: np.ndarray) -> float:
    """Harmonic mean — emphasizes lower values (useful for resonance)."""
    values = np.array(values)
    values = values[values > 0]
    if len(values) == 0:
        return 0.0
    return float(len(values) / np.sum(1.0 / values))
