# haze.py — Haze: Hybrid Attention Entropy System (NumPy inference)
#
# Architecture:
#   - HybridHead = ReweightHead (positional) + ContentHead (semantic)
#   - Pre-norm blocks with GELU activation
#   - Entropy-aware adaptive temperature
#   - Multiple sampling strategies (top-p, top-k, mirostat)
#
# Can be randomly initialized OR loaded from .npz exported by train.py

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Literal

try:
    from .nn import (
        get_rng,
        init_weight,
        softmax,
        gelu,
        layer_norm,
        rms_norm,
        sample_basic,
        sample_top_k,
        sample_top_p,
        sample_mirostat,
        sample_mirostat_v2,
        entropy_temperature,
        resonance_temperature,
        entropy_bits,
        confidence_score,
    )
except ImportError:
    from nn import (
        get_rng,
        init_weight,
        softmax,
        gelu,
        layer_norm,
        rms_norm,
        sample_basic,
        sample_top_k,
        sample_top_p,
        sample_mirostat,
        sample_mirostat_v2,
        entropy_temperature,
        resonance_temperature,
        entropy_bits,
        confidence_score,
    )


# ----------------- vocab -----------------


@dataclass
class Vocab:
    """Character-level vocabulary."""

    chars: List[str]
    stoi: dict
    itos: dict
    vocab_size: int

    @classmethod
    def from_text(cls, text: str) -> "Vocab":
        text = text.lower()
        chars = sorted(list(set(text)))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        return cls(chars=chars, stoi=stoi, itos=itos, vocab_size=len(chars))
    
    @staticmethod
    def _normalize_text(s: str) -> str:
        """Normalize text to use corpus-compatible characters.
        
        The corpus uses fancy quotes: ' ' " " instead of ASCII ' "
        This ensures encode() doesn't drop apostrophes.
        """
        # Normalize ASCII apostrophe (U+0027) to RIGHT SINGLE QUOTATION MARK (U+2019)
        # which is what the corpus uses for contractions like "don't"
        s = s.replace('\x27', '\u2019')  # ' → '
        # Normalize ASCII double quote (U+0022) to RIGHT DOUBLE QUOTATION MARK (U+201D)
        s = s.replace('\x22', '\u201d')  # " → "
        return s

    def encode(self, s: str) -> List[int]:
        s = self._normalize_text(s.lower())
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, idxs: List[int]) -> str:
        return "".join(self.itos.get(i, "?") for i in idxs)


# ----------------- attention heads -----------------


class RRPRAMHead:
    """
    RRPRAM: Recursive Resonant Pattern Recognition Attention Mechanism.
    
    Learns positional attention patterns directly.
    Instead of QK^T, uses x @ W_pattern → (T, T) attention matrix.
    
    Captures: rhythm, n-gram patterns, positional dependencies.
    The "recursive resonant" part: learns patterns of patterns. meta-attention.
    """

    def __init__(self, n_emb: int, head_dim: int, T: int, rng):
        self.wv = init_weight((n_emb, head_dim), rng=rng)
        self.wr = init_weight((n_emb, T), rng=rng)  # pattern projection
        self.T = T
        self.head_dim = head_dim

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: (T, n_emb)
        returns: (T, head_dim)
        """
        v = x @ self.wv  # (T, head_dim)
        attn = x @ self.wr  # (T, T)

        # causal mask
        T = min(x.shape[0], self.T)
        tril = np.tril(np.ones((T, T), dtype=np.float32))
        mask = np.where(tril == 1.0, 0.0, -1e9)
        attn = attn[:T, :T] + mask

        pattern = softmax(attn, axis=-1)  # (T, T)
        out = pattern @ v[:T]  # (T, head_dim)
        return out


# Backwards compatibility alias
ReweightHead = RRPRAMHead


class ContentHead:
    """
    Content-based attention: classic QK^T / sqrt(d) attention.
    
    Captures: semantic similarity, long-range dependencies.
    """

    def __init__(self, n_emb: int, head_dim: int, T: int, rng):
        self.wq = init_weight((n_emb, head_dim), rng=rng)
        self.wk = init_weight((n_emb, head_dim), rng=rng)
        self.wv = init_weight((n_emb, head_dim), rng=rng)
        self.T = T
        self.head_dim = head_dim
        self.scale = 1.0 / np.sqrt(head_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: (T, n_emb)
        returns: (T, head_dim)
        """
        q = x @ self.wq  # (T, head_dim)
        k = x @ self.wk  # (T, head_dim)
        v = x @ self.wv  # (T, head_dim)

        attn = (q @ k.T) * self.scale  # (T, T)

        # causal mask
        T = min(x.shape[0], self.T)
        tril = np.tril(np.ones((T, T), dtype=np.float32))
        mask = np.where(tril == 1.0, 0.0, -1e9)
        attn = attn[:T, :T] + mask

        attn = softmax(attn, axis=-1)
        out = attn @ v[:T]
        return out


class HybridHead:
    """
    Hybrid attention: combines RRPRAM (positional) + Content (semantic).
    
    The mix ratio α controls the blend:
        output = α * rrpram_out + (1-α) * content_out
    
    This allows the model to use positional patterns (rhythm, structure)
    AND semantic similarity (meaning) simultaneously.
    """

    def __init__(
        self,
        n_emb: int,
        head_dim: int,
        T: int,
        rng,
        alpha: float = 0.5,  # rrpram vs content mix
    ):
        self.rrpram = RRPRAMHead(n_emb, head_dim, T, rng)
        self.content = ContentHead(n_emb, head_dim, T, rng)
        self.alpha = alpha
        self.head_dim = head_dim

        # learnable gate (initialized to alpha)
        self.gate = np.array([alpha], dtype=np.float32)
    
    # Backwards compatibility
    @property
    def reweight(self):
        return self.rrpram

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: (T, n_emb)
        returns: (T, head_dim)
        """
        r_out = self.rrpram.forward(x)
        c_out = self.content.forward(x)

        # gated combination
        alpha = float(self.gate[0])
        return alpha * r_out + (1.0 - alpha) * c_out


# ----------------- block -----------------


class Block:
    """
    Transformer block with:
    - Pre-norm (more stable for deep networks)
    - Hybrid attention heads (RRPRAM + Content)
    - GELU activation (smoother than ReLU)
    - Residual connections
    """

    def __init__(
        self,
        n_emb: int,
        T: int,
        nodes: int,
        rng,
        n_heads: int = 4,
        head_type: Literal["hybrid", "rrpram", "content", "reweight"] = "hybrid",
        alpha: float = 0.5,
    ):
        head_dim = n_emb // n_heads

        # normalize head_type (reweight is alias for rrpram)
        if head_type == "reweight":
            head_type = "rrpram"

        # create heads based on type
        if head_type == "hybrid":
            self.heads = [
                HybridHead(n_emb, head_dim, T, rng, alpha=alpha)
                for _ in range(n_heads)
            ]
        elif head_type == "rrpram":
            self.heads = [
                RRPRAMHead(n_emb, head_dim, T, rng) for _ in range(n_heads)
            ]
        else:  # content
            self.heads = [
                ContentHead(n_emb, head_dim, T, rng) for _ in range(n_heads)
            ]

        # MLP
        self.w0 = init_weight((n_emb, nodes), rng=rng)
        self.w1 = init_weight((nodes, n_emb), rng=rng)

        # layer norm parameters
        self.ln1_gamma = np.ones(n_emb, dtype=np.float32)
        self.ln1_beta = np.zeros(n_emb, dtype=np.float32)
        self.ln2_gamma = np.ones(n_emb, dtype=np.float32)
        self.ln2_beta = np.zeros(n_emb, dtype=np.float32)

        self.n_emb = n_emb
        self.head_type = head_type

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: (T, n_emb)
        returns: (T, n_emb)
        """
        # pre-norm attention
        x_norm = layer_norm(x, self.ln1_gamma, self.ln1_beta)
        h = [head.forward(x_norm) for head in self.heads]
        h = np.concatenate(h, axis=-1)  # (T, n_emb)
        x = x + h  # residual

        # pre-norm MLP
        x_norm = layer_norm(x, self.ln2_gamma, self.ln2_beta)
        h = x_norm @ self.w0
        h = gelu(h)
        h = h @ self.w1
        x = x + h  # residual

        return x


# ----------------- model -----------------


class PostGPT:
    """
    PostGPT: post-transformer hybrid attention language model.

    Character-level model with:
    - Hybrid heads (RRPRAM + content attention)
    - Pre-norm blocks with GELU
    - Entropy-aware adaptive temperature
    - Multiple sampling strategies

    Part of the Haze ecosystem (Hybrid Attention Entropy System).
    
    Why "PostGPT"? Because this is what comes after you understand GPT
    and ask: "what if we didn't compute QK^T for everything?"
    """

    def __init__(
        self,
        vocab_size: int,
        T: int = 16,
        n_emb: int = 32,
        nodes: int = 32,
        n_blocks: int = 3,
        n_heads: int = 4,
        head_type: Literal["hybrid", "rrpram", "content", "reweight"] = "hybrid",
        alpha: float = 0.5,
        seed: Optional[int] = 42,
    ):
        self.T = T
        self.n_emb = n_emb
        self.nodes = nodes
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.head_type = head_type
        self.alpha = alpha
        self.vocab_size = vocab_size
        self.rng = get_rng(seed)

        # embeddings
        self.embed = init_weight((vocab_size, n_emb), rng=self.rng)
        self.pos = init_weight((T, n_emb), rng=self.rng)

        # blocks
        self.blocks = [
            Block(
                n_emb,
                T,
                nodes,
                rng=self.rng,
                n_heads=n_heads,
                head_type=head_type,
                alpha=alpha,
            )
            for _ in range(n_blocks)
        ]

        # final layer norm
        self.ln_f_gamma = np.ones(n_emb, dtype=np.float32)
        self.ln_f_beta = np.zeros(n_emb, dtype=np.float32)

        # output projection
        self.w2 = init_weight((n_emb, vocab_size), rng=self.rng)

    def logits(self, idx_seq: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        
        idx_seq: (T,) int array of token indices
        returns: (T, vocab_size) logits
        """
        T = len(idx_seq)
        x = self.embed[idx_seq] + self.pos[:T]  # (T, n_emb)

        for block in self.blocks:
            x = block.forward(x)

        x = layer_norm(x, self.ln_f_gamma, self.ln_f_beta)
        logits = x @ self.w2  # (T, vocab_size)
        return logits

    def generate(
        self,
        seed_seq: List[int],
        length: int = 200,
        temperature: float = 1.0,
        sampling: Literal["basic", "top_k", "top_p", "entropy", "mirostat", "mirostat_v2", "resonance"] = "entropy",
        top_k: int = 40,
        top_p: float = 0.9,
        target_entropy: float = 3.0,
        target_resonance: float = 0.7,
        min_temp: float = 0.3,
        max_temp: float = 2.0,
        mirostat_tau: float = 0.1,
    ) -> tuple[List[int], dict]:
        """
        Generate tokens with various sampling strategies.
        
        Args:
            seed_seq: initial token indices
            length: number of tokens to generate
            temperature: base temperature (used differently per strategy)
            sampling: strategy - "basic", "top_k", "top_p", "entropy", "mirostat", "mirostat_v2", "resonance"
            top_k: k for top-k sampling
            top_p: p for nucleus sampling
            target_entropy: target entropy for entropy-aware and mirostat sampling
            target_resonance: target resonance for resonance-based sampling
            min_temp, max_temp: bounds for adaptive temperature
            mirostat_tau: learning rate for mirostat sampling
        
        Returns:
            (tokens, stats) where stats contains generation metrics
        """
        T = self.T

        # prepare sequence
        if not seed_seq:
            seed_seq = [0]

        seq = list(seed_seq)
        if len(seq) < T:
            pad_val = seq[0]
            seq = [pad_val] * (T - len(seq)) + seq
        else:
            seq = seq[-T:]

        seq = np.array(seq, dtype=np.int32)
        out = []

        # stats tracking
        entropies = []
        confidences = []
        temps_used = []
        resonances = []
        
        # mirostat state
        mu = target_entropy * 2.0  # initial mu
        
        # resonance history (keep last N logits)
        history_logits = []
        history_window = 10

        for _ in range(length):
            logits = self.logits(seq)
            logits_last = logits[-1]

            # track metrics
            probs = softmax(logits_last)
            entropies.append(entropy_bits(probs))
            confidences.append(confidence_score(logits_last))

            # sampling strategy
            if sampling == "entropy":
                # adaptive temperature based on current entropy
                temp = entropy_temperature(
                    logits_last,
                    target_entropy=target_entropy,
                    min_temp=min_temp,
                    max_temp=max_temp,
                )
                temps_used.append(temp)
                nxt = sample_top_p(logits_last, top_p, temp, self.rng)
            
            elif sampling == "resonance":
                # adaptive temperature based on resonance with history
                temp = resonance_temperature(
                    logits_last,
                    history_logits,
                    target_resonance=target_resonance,
                    min_temp=min_temp,
                    max_temp=max_temp,
                )
                temps_used.append(temp)
                nxt = sample_top_p(logits_last, top_p, temp, self.rng)
                
                # track resonance
                if history_logits:
                    try:
                        from .nn import resonance_score
                    except ImportError:
                        from nn import resonance_score
                    res = resonance_score(logits_last, history_logits[-1])
                    resonances.append(res)
                else:
                    resonances.append(0.5)
            
            elif sampling == "mirostat":
                # mirostat v1 sampling
                nxt, mu = sample_mirostat(
                    logits_last,
                    target_entropy=target_entropy,
                    tau=mirostat_tau,
                    mu=mu,
                    rng=self.rng,
                )
                temps_used.append(mu / target_entropy)  # normalized mu as "temperature"
            
            elif sampling == "mirostat_v2":
                # mirostat v2 sampling with adaptive k
                nxt, mu = sample_mirostat_v2(
                    logits_last,
                    target_entropy=target_entropy,
                    tau=mirostat_tau,
                    mu=mu,
                    rng=self.rng,
                )
                temps_used.append(mu / target_entropy)  # normalized mu as "temperature"

            elif sampling == "top_p":
                temps_used.append(temperature)
                nxt = sample_top_p(logits_last, top_p, temperature, self.rng)

            elif sampling == "top_k":
                temps_used.append(temperature)
                nxt = sample_top_k(logits_last, top_k, temperature, self.rng)

            else:  # basic
                temps_used.append(temperature)
                nxt = sample_basic(logits_last, temperature, self.rng)

            out.append(nxt)
            
            # update resonance history
            if sampling == "resonance":
                history_logits.append(logits_last.copy())
                if len(history_logits) > history_window:
                    history_logits.pop(0)

            # shift window
            seq = np.roll(seq, -1)
            seq[-1] = nxt

        stats = {
            "mean_entropy": float(np.mean(entropies)),
            "mean_confidence": float(np.mean(confidences)),
            "mean_temp": float(np.mean(temps_used)),
            "min_entropy": float(np.min(entropies)),
            "max_entropy": float(np.max(entropies)),
            "entropy_std": float(np.std(entropies)),
        }
        
        # add resonance stats if available
        if resonances:
            stats["mean_resonance"] = float(np.mean(resonances))
            stats["resonance_std"] = float(np.std(resonances))

        return out, stats

    # ----- simple generate for compatibility -----

    def generate_simple(
        self,
        seed_seq: List[int],
        length: int = 200,
        temperature: float = 1.0,
    ) -> List[int]:
        """Simple generation without stats (for compatibility)."""
        tokens, _ = self.generate(
            seed_seq,
            length=length,
            temperature=temperature,
            sampling="basic",
        )
        return tokens
    
    def generate_resonant(
        self,
        seed_seq: List[int],
        corpus_text: str,
        vocab: "Vocab",
        length: int = 100,
        temperature: float = 0.6,
        mode: str = "trigram",
        use_model: bool = False,
        model_alpha: float = 0.3,
        cleanup: bool = True,
    ) -> tuple[List[int], str, dict]:
        """
        Generate using corpus statistics (like Leo).
        
        This is the recommended mode for untrained models.
        Pure resonance - no neural network weights needed.
        
        Args:
            seed_seq: initial token indices
            corpus_text: text corpus for building statistics
            vocab: vocabulary for encoding
            length: tokens to generate
            temperature: sampling temperature (lower = more coherent)
            mode: "bigram", "trigram", "cooccur", or "blend"
            use_model: if True, blend model logits with corpus (requires trained weights)
            model_alpha: blend ratio when use_model=True (0=corpus, 1=model)
            cleanup: if True, clean up output punctuation
        
        Returns:
            (tokens, text, stats)
        """
        try:
            from .cooccur import CooccurField
        except ImportError:
            from cooccur import CooccurField
        
        # Build co-occurrence field
        field = CooccurField.from_text(corpus_text, vocab, window_size=5)
        
        tokens = list(seed_seq)
        
        for _ in range(length):
            if use_model and len(tokens) > 0:
                # Hybrid: model + corpus
                idx_seq = np.array(tokens[-self.T:], dtype=np.int32)
                logits = self.logits(idx_seq)[-1]
                
                # Bias with corpus
                biased = field.bias_logits(logits, tokens, alpha=1.0-model_alpha, mode=mode)
                
                # Sample
                probs = softmax(biased / temperature)
                next_token = int(self.rng.choice(self.vocab_size, p=probs))
            else:
                # Pure corpus generation
                next_token = field.sample_from_corpus(tokens, temperature=temperature, mode=mode)
            
            tokens.append(next_token)
        
        # Decode
        text = vocab.decode(tokens)
        
        # Cleanup output
        if cleanup:
            try:
                from .cleanup import cleanup_output
            except ImportError:
                from cleanup import cleanup_output
            text = cleanup_output(text, mode="gentle")
        
        stats = {
            "mode": mode,
            "use_model": use_model,
            "temperature": temperature,
            "field_stats": field.stats(),
        }
        
        return tokens, text, stats

    # ----- weight loading/saving -----

    @classmethod
    def theweightofhaze(cls, vocab_size: int, path: str | Path) -> "PostGPT":
        """
        Load weights from .npz file.
        
        Because the weight of haze is not in pounds or kilograms,
        but in the patterns it learned from the void.
        
        Note: This loads as RRPRAM-only heads (no content heads)
        to match the training architecture. Use head_type="rrpram"
        or retrain with hybrid heads for full hybrid inference.
        """
        path = Path(path)
        data = np.load(path, allow_pickle=False)

        T = int(data["T"])
        n_emb = int(data["n_emb"])
        nodes = int(data["nodes"])
        n_blocks = int(data["n_blocks"])
        n_heads = int(data["n_heads"])
        saved_vocab_size = int(data["vocab_size"])

        if saved_vocab_size != vocab_size:
            raise ValueError(
                f"Vocab size mismatch: npz={saved_vocab_size}, current={vocab_size}"
            )

        model = cls(
            vocab_size=vocab_size,
            T=T,
            n_emb=n_emb,
            nodes=nodes,
            n_blocks=n_blocks,
            n_heads=n_heads,
            head_type="rrpram",  # trained model uses RRPRAM heads
            seed=None,
        )

        # top-level
        model.embed = data["embed"].astype("float32")
        model.pos = data["pos"].astype("float32")
        model.w2 = data["w2"].astype("float32")

        # blocks / heads
        for b in range(n_blocks):
            block = model.blocks[b]
            block.w0 = data[f"blocks.{b}.w0"].astype("float32")
            block.w1 = data[f"blocks.{b}.w1"].astype("float32")

            for h in range(n_heads):
                head = block.heads[h]
                head.wv = data[f"blocks.{b}.heads.{h}.wv"].astype("float32")
                head.wr = data[f"blocks.{b}.heads.{h}.wr"].astype("float32")

        return model
    
    @classmethod
    def from_npz(cls, vocab_size: int, path: str | Path) -> "PostGPT":
        """Alias for theweightofhaze() for backward compatibility."""
        return cls.theweightofhaze(vocab_size, path)
    
    def save_theweightofhaze(self, path: str | Path):
        """
        Save model weights to .npz file.
        
        Exports the weight of haze into the void,
        so it can be summoned again later.
        """
        path = Path(path)
        
        # prepare weight dict
        weights = {
            "T": self.T,
            "n_emb": self.n_emb,
            "nodes": self.nodes,
            "n_blocks": self.n_blocks,
            "n_heads": self.n_heads,
            "vocab_size": self.vocab_size,
            "embed": self.embed,
            "pos": self.pos,
            "w2": self.w2,
        }
        
        # save blocks and heads
        for b, block in enumerate(self.blocks):
            weights[f"blocks.{b}.w0"] = block.w0
            weights[f"blocks.{b}.w1"] = block.w1
            
            for h, head in enumerate(block.heads):
                # check if RRPRAM head or hybrid
                if hasattr(head, 'wr'):
                    weights[f"blocks.{b}.heads.{h}.wv"] = head.wv
                    weights[f"blocks.{b}.heads.{h}.wr"] = head.wr
                elif hasattr(head, 'rrpram'):
                    # hybrid head - save RRPRAM part
                    weights[f"blocks.{b}.heads.{h}.wv"] = head.rrpram.wv
                    weights[f"blocks.{b}.heads.{h}.wr"] = head.rrpram.wr
        
        np.savez_compressed(path, **weights)
        print(f"[saved] the weight of haze → {path}")


# ----------------- helpers -----------------


def load_corpus(path: str | Path) -> str:
    """Load text corpus from file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def build_model_from_text(
    path: str | Path,
    T: int = 16,
    n_emb: int = 32,
    nodes: int = 32,
    n_blocks: int = 3,
    n_heads: int = 4,
    head_type: Literal["hybrid", "rrpram", "content", "reweight"] = "hybrid",
    alpha: float = 0.5,
    seed: Optional[int] = 42,
):
    """Build model and vocab from text file."""
    text = load_corpus(path)
    vocab = Vocab.from_text(text)
    model = PostGPT(
        vocab_size=vocab.vocab_size,
        T=T,
        n_emb=n_emb,
        nodes=nodes,
        n_blocks=n_blocks,
        n_heads=n_heads,
        head_type=head_type,
        alpha=alpha,
        seed=seed,
    )
    return text, vocab, model
