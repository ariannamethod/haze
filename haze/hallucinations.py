#!/usr/bin/env python3
# hallucinations.py — Attention pattern visualization and analysis
#
# Exports attention weights from haze models for visualization.
# See what patterns the reweight heads actually learn.
# Because sometimes you need to stare into the void and see what stares back.

from __future__ import annotations
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[hallucinations] matplotlib not found. Install it for visualizations: pip install matplotlib")


# ----------------- attention extraction -----------------


def extract_reweight_attention(
    model,
    input_seq: np.ndarray,
    block_idx: int = 0,
    head_idx: int = 0,
) -> np.ndarray:
    """
    Extract attention matrix from a reweight head.
    
    Args:
        model: Haze model instance
        input_seq: token sequence (T,)
        block_idx: which transformer block to extract from
        head_idx: which head within the block
    
    Returns:
        attention matrix (T, T)
    """
    # get block and head
    block = model.blocks[block_idx]
    head = block.heads[head_idx]
    
    # check if it's a reweight head
    if not hasattr(head, 'wr'):
        # try to unwrap if it's a hybrid head
        if hasattr(head, 'reweight'):
            head = head.reweight
        else:
            raise ValueError(f"Head {head_idx} in block {block_idx} is not a reweight head")
    
    # forward through embedding
    T = len(input_seq)
    x = model.embed[input_seq] + model.pos[:T]
    
    # forward through blocks up to target block
    for i, blk in enumerate(model.blocks):
        if i == block_idx:
            # compute attention for this block
            try:
                from .model import layer_norm, softmax
            except ImportError:
                from model import layer_norm, softmax
            x_norm = layer_norm(x, blk.ln1_gamma, blk.ln1_beta)
            
            # get attention matrix from reweight head
            attn = x_norm @ head.wr  # (T, T)
            
            # apply causal mask
            T_actual = min(x.shape[0], head.T)
            tril = np.tril(np.ones((T_actual, T_actual), dtype=np.float32))
            mask = np.where(tril == 1.0, 0.0, -1e9)
            attn = attn[:T_actual, :T_actual] + mask
            
            # apply softmax
            attn = softmax(attn, axis=-1)
            
            return attn
        
        # forward through full block
        x = blk.forward(x)
    
    raise ValueError(f"Block {block_idx} not found")


def extract_all_reweight_patterns(
    model,
    input_seq: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Extract all reweight attention patterns from model.
    
    Returns:
        dict mapping "block_{i}_head_{j}" to attention matrix
    """
    patterns = {}
    
    for block_idx, block in enumerate(model.blocks):
        for head_idx, head in enumerate(block.heads):
            # check if reweight head
            has_wr = hasattr(head, 'wr')
            is_hybrid = hasattr(head, 'reweight')
            
            if has_wr or is_hybrid:
                try:
                    attn = extract_reweight_attention(model, input_seq, block_idx, head_idx)
                    key = f"block_{block_idx}_head_{head_idx}"
                    patterns[key] = attn
                except Exception as e:
                    print(f"[warn] failed to extract {block_idx}/{head_idx}: {e}")
    
    return patterns


# ----------------- visualization -----------------


def visualize_attention_matrix(
    attention: np.ndarray,
    title: str = "Attention Pattern",
    tokens: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
):
    """
    Visualize attention matrix as a heatmap.
    
    Args:
        attention: (T, T) attention matrix
        title: plot title
        tokens: optional list of token strings for labels
        save_path: optional path to save figure
        figsize: figure size
    """
    if not HAS_MATPLOTLIB:
        print("[error] matplotlib not available. Cannot visualize.")
        return
    
    T = attention.shape[0]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # create heatmap
    im = ax.imshow(attention, cmap='viridis', aspect='auto', interpolation='nearest')
    
    # colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', rotation=270, labelpad=20)
    
    # labels
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    ax.set_title(title)
    
    # add token labels if provided
    if tokens is not None:
        ax.set_xticks(range(T))
        ax.set_yticks(range(T))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_yticklabels(tokens)
    
    # grid
    ax.set_xticks(np.arange(T) - 0.5, minor=True)
    ax.set_yticks(np.arange(T) - 0.5, minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[saved] {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_all_patterns(
    patterns: Dict[str, np.ndarray],
    tokens: Optional[List[str]] = None,
    save_dir: Optional[str] = None,
):
    """
    Visualize all attention patterns in a grid.
    
    Args:
        patterns: dict of attention matrices
        tokens: optional token labels
        save_dir: directory to save individual plots
    """
    if not HAS_MATPLOTLIB:
        print("[error] matplotlib not available. Cannot visualize.")
        return
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
    
    for key, attn in patterns.items():
        title = f"Reweight Attention: {key.replace('_', ' ').title()}"
        save_path = str(save_dir / f"{key}.png") if save_dir else None
        visualize_attention_matrix(attn, title=title, tokens=tokens, save_path=save_path)


def analyze_attention_patterns(
    attention: np.ndarray,
) -> Dict[str, float]:
    """
    Analyze attention pattern properties.
    
    Returns:
        dict of metrics:
        - sparsity: fraction of near-zero weights
        - locality: average distance of attention
        - uniformity: entropy of average attention distribution
        - diagonality: how much attention is on the diagonal
    """
    T = attention.shape[0]
    
    # sparsity: fraction of weights below threshold
    threshold = 0.01
    sparsity = float(np.mean(attention < threshold))
    
    # locality: average attention distance
    positions = np.arange(T)
    distances = []
    for i in range(T):
        avg_pos = np.sum(attention[i] * positions[:i+1])  # causal only
        distance = abs(i - avg_pos)
        distances.append(distance)
    locality = float(np.mean(distances))
    
    # uniformity: entropy of average attention
    avg_attn = attention.mean(axis=0)
    avg_attn = avg_attn / (avg_attn.sum() + 1e-10)
    uniformity = float(-np.sum(avg_attn * np.log(avg_attn + 1e-10)))
    
    # diagonality: attention on diagonal and nearby
    diagonal_weight = 0.0
    for i in range(T):
        # sum attention to positions within distance 2
        for j in range(max(0, i-2), i+1):
            diagonal_weight += attention[i, j]
    diagonality = float(diagonal_weight / T)
    
    return {
        'sparsity': sparsity,
        'locality': locality,
        'uniformity': uniformity,
        'diagonality': diagonality,
    }


def generate_attention_report(
    patterns: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
) -> str:
    """
    Generate a text report analyzing all attention patterns.
    
    Args:
        patterns: dict of attention matrices
        save_path: optional path to save report
    
    Returns:
        report string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("HALLUCINATIONS — Attention Pattern Analysis")
    lines.append("=" * 60)
    lines.append("")
    
    for key, attn in patterns.items():
        metrics = analyze_attention_patterns(attn)
        
        lines.append(f"[{key}]")
        lines.append(f"  sparsity:    {metrics['sparsity']:.3f}  (fraction near-zero)")
        lines.append(f"  locality:    {metrics['locality']:.3f}  (avg attention distance)")
        lines.append(f"  uniformity:  {metrics['uniformity']:.3f}  (entropy of distribution)")
        lines.append(f"  diagonality: {metrics['diagonality']:.3f}  (local attention ratio)")
        lines.append("")
    
    lines.append("=" * 60)
    lines.append("patterns we forgot we already knew")
    lines.append("=" * 60)
    
    report = "\n".join(lines)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"[saved] {save_path}")
    
    return report


# ----------------- main -----------------


def hallucinate(
    model,
    input_text: str,
    vocab,
    save_dir: str = "hallucinations",
    visualize: bool = True,
):
    """
    Main function: extract and visualize attention patterns.
    
    Args:
        model: Haze model
        input_text: text to analyze
        vocab: vocabulary for encoding
        save_dir: directory to save outputs
        visualize: whether to create visualizations
    """
    # encode input
    input_seq = np.array(vocab.encode(input_text), dtype=np.int32)
    tokens = list(input_text.lower())
    
    print(f"[hallucinations] analyzing: '{input_text}'")
    print(f"[hallucinations] sequence length: {len(input_seq)}")
    
    # extract patterns
    patterns = extract_all_reweight_patterns(model, input_seq)
    print(f"[hallucinations] extracted {len(patterns)} attention patterns")
    
    # create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # generate report
    report = generate_attention_report(patterns, save_path=str(save_dir / "report.txt"))
    print(report)
    
    # visualize
    if visualize and HAS_MATPLOTLIB:
        print("[hallucinations] generating visualizations...")
        visualize_all_patterns(
            patterns,
            tokens=tokens[:min(len(tokens), 20)],  # limit token labels for readability
            save_dir=str(save_dir)
        )
        print(f"[hallucinations] visualizations saved to {save_dir}/")
    
    return patterns


if __name__ == "__main__":
    import sys
    
    # example usage
    print("=" * 60)
    print("  hallucinations.py — attention pattern analysis")
    print("=" * 60)
    print()
    print("Usage:")
    print("  from hallucinations import hallucinate")
    print("  from model import Vocab, ReweightGPT as Haze")
    print()
    print("  text = open('text.txt').read()")
    print("  vocab = Vocab.from_text(text)")
    print("  model = Haze(vocab_size=vocab.vocab_size, T=32, n_emb=64)")
    print()
    print("  # analyze attention patterns")
    print("  patterns = hallucinate(model, 'the haze settles', vocab)")
    print()
    print("=" * 60)
