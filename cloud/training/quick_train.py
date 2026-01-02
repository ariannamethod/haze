#!/usr/bin/env python3
# quick_train.py — Quick training to improve CLOUD accuracy
#
# Simple gradient descent for chamber MLPs.
# Goal: make chambers respond correctly to their emotion categories.
#
# Not perfect training - just "good enough" for first impressions.
# "Pre-semantic" means it doesn't have to be precise!

import json
import numpy as np
from pathlib import Path
import sys

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cloud import Cloud
from cloud.anchors import (
    get_all_anchors, 
    get_anchor_index, 
    CHAMBER_NAMES,
    EMOTION_ANCHORS,
    get_chamber_for_anchor,
)


def get_chamber_target(primary: str) -> np.ndarray:
    """Get target activation for chambers based on primary emotion."""
    chamber = get_chamber_for_anchor(primary)
    
    targets = np.zeros(4)
    chamber_idx = {"FEAR": 0, "LOVE": 1, "RAGE": 2, "VOID": 3}
    
    if chamber in chamber_idx:
        targets[chamber_idx[chamber]] = 0.8
    else:
        # FLOW or COMPLEX - distribute evenly
        targets[:] = 0.2
    
    return targets


def train_chamber_mlp(mlp, resonances: np.ndarray, target: float, lr: float = 0.01):
    """
    Train single chamber MLP with one gradient step.
    
    Simple backprop through 3-layer MLP.
    """
    # Forward pass with cached activations
    h1 = resonances @ mlp.W1 + mlp.b1
    a1 = h1 / (1.0 + np.exp(-h1))  # swish approximation
    
    h2 = a1 @ mlp.W2 + mlp.b2
    a2 = h2 / (1.0 + np.exp(-h2))
    
    h3 = a2 @ mlp.W3 + mlp.b3
    output = 1.0 / (1.0 + np.exp(-h3[0]))  # sigmoid
    
    # Loss: MSE
    error = output - target
    loss = 0.5 * error ** 2
    
    # Backward pass
    # d_loss/d_output = error
    # d_output/d_h3 = output * (1 - output) [sigmoid derivative]
    d_h3 = error * output * (1 - output)
    
    # Gradient for W3, b3
    d_W3 = np.outer(a2, [d_h3])
    d_b3 = np.array([d_h3])
    
    # Backprop through layer 2
    d_a2 = d_h3 * mlp.W3.flatten()
    # swish derivative approximation: sig + x * sig * (1-sig)
    sig2 = 1.0 / (1.0 + np.exp(-h2))
    d_h2 = d_a2 * (sig2 + h2 * sig2 * (1 - sig2))
    
    d_W2 = np.outer(a1, d_h2)
    d_b2 = d_h2
    
    # Backprop through layer 1
    d_a1 = d_h2 @ mlp.W2.T
    sig1 = 1.0 / (1.0 + np.exp(-h1))
    d_h1 = d_a1 * (sig1 + h1 * sig1 * (1 - sig1))
    
    d_W1 = np.outer(resonances, d_h1)
    d_b1 = d_h1
    
    # Update weights
    mlp.W1 -= lr * d_W1
    mlp.b1 -= lr * d_b1
    mlp.W2 -= lr * d_W2
    mlp.b2 -= lr * d_b2
    mlp.W3 -= lr * d_W3
    mlp.b3 -= lr * d_b3
    
    return loss


def train_observer(observer, resonances: np.ndarray, iterations: float, 
                   fingerprint: np.ndarray, target_idx: int, lr: float = 0.01):
    """
    Train observer MLP with one gradient step.
    
    CrossEntropy loss on secondary emotion prediction.
    """
    # Forward pass
    x = np.concatenate([resonances, [iterations], fingerprint])
    
    h1 = x @ observer.W1 + observer.b1
    a1 = h1 / (1.0 + np.exp(-np.clip(h1, -20, 20)))  # swish
    
    logits = a1 @ observer.W2 + observer.b2
    
    # Softmax
    exp_logits = np.exp(logits - logits.max())
    probs = exp_logits / exp_logits.sum()
    
    # CrossEntropy loss
    loss = -np.log(probs[target_idx] + 1e-10)
    
    # Backward pass
    # d_loss/d_logits = probs - one_hot(target)
    d_logits = probs.copy()
    d_logits[target_idx] -= 1.0
    
    # Gradient for W2, b2
    d_W2 = np.outer(a1, d_logits)
    d_b2 = d_logits
    
    # Backprop through layer 1
    d_a1 = d_logits @ observer.W2.T
    sig1 = 1.0 / (1.0 + np.exp(-np.clip(h1, -20, 20)))
    d_h1 = d_a1 * (sig1 + h1 * sig1 * (1 - sig1))
    
    d_W1 = np.outer(x, d_h1)
    d_b1 = d_h1
    
    # Update weights
    observer.W1 -= lr * d_W1
    observer.b1 -= lr * d_b1
    observer.W2 -= lr * d_W2
    observer.b2 -= lr * d_b2
    
    return loss, int(np.argmax(logits)) == target_idx


def quick_train(cloud: Cloud, examples: list, epochs: int = 5, lr: float = 0.005):
    """
    Quick training loop.
    
    Args:
        cloud: CLOUD instance
        examples: training data
        epochs: number of epochs
        lr: learning rate
    """
    all_anchors = get_all_anchors()
    chambers_list = [cloud.chambers.fear, cloud.chambers.love, 
                     cloud.chambers.rage, cloud.chambers.void]
    
    print(f"Quick training for {epochs} epochs...")
    print(f"Learning rate: {lr}")
    print(f"Examples: {len(examples)}")
    print()
    
    for epoch in range(epochs):
        total_chamber_loss = 0.0
        total_observer_loss = 0.0
        observer_correct = 0
        
        # Shuffle examples
        np.random.shuffle(examples)
        
        for ex in examples:
            text = ex["text"]
            primary = ex["primary"]
            secondary = ex["secondary"]
            
            # Get resonances
            resonances = cloud.resonance.compute_resonance(text)
            
            # Get target chamber activations
            targets = get_chamber_target(primary)
            
            # Train each chamber
            for i, (chamber, target) in enumerate(zip(chambers_list, targets)):
                loss = train_chamber_mlp(chamber, resonances, target, lr=lr)
                total_chamber_loss += loss
            
            # Get chamber activations for observer
            chamber_acts, iterations = cloud.chambers.stabilize(resonances)
            fingerprint = cloud.user_cloud.get_fingerprint()
            
            # Train observer
            try:
                secondary_idx = get_anchor_index(secondary)
                obs_loss, correct = train_observer(
                    cloud.observer, resonances, float(iterations),
                    fingerprint, secondary_idx, lr=lr
                )
                total_observer_loss += obs_loss
                if correct:
                    observer_correct += 1
            except ValueError:
                # Unknown anchor - skip
                pass
        
        # Stats
        n = len(examples)
        avg_chamber = total_chamber_loss / (n * 4)
        avg_observer = total_observer_loss / n
        accuracy = observer_correct / n
        
        print(f"Epoch {epoch+1}/{epochs}: chamber_loss={avg_chamber:.4f}, "
              f"observer_loss={avg_observer:.4f}, observer_acc={accuracy:.1%}")
    
    print()
    print("Training complete!")


def test_accuracy(cloud: Cloud, examples: list):
    """Test accuracy after training."""
    all_anchors = get_all_anchors()
    
    print("Testing accuracy...")
    print("-" * 60)
    
    correct_primary = 0
    correct_chamber = 0
    
    for ex in examples[:30]:  # Test on first 30
        text = ex["text"]
        expected_primary = ex["primary"]
        expected_chamber = ex.get("chamber", "UNKNOWN")
        
        resonances = cloud.resonance.compute_resonance(text)
        _, primary, _ = cloud.resonance.get_primary_emotion(resonances)
        
        chamber_acts, _ = cloud.chambers.stabilize(resonances)
        predicted_chamber = max(chamber_acts.items(), key=lambda x: x[1])[0]
        
        if primary == expected_primary:
            correct_primary += 1
        if predicted_chamber == expected_chamber:
            correct_chamber += 1
    
    print(f"Primary accuracy: {correct_primary}/30 ({correct_primary/30:.1%})")
    print(f"Chamber accuracy: {correct_chamber}/30 ({correct_chamber/30:.1%})")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("  CLOUD Quick Training")
    print("  'Good enough for first impressions'")
    print("=" * 60)
    print()
    
    # Load data
    data_path = Path(__file__).parent / "bootstrap_data.json"
    if not data_path.exists():
        print(f"[error] {data_path} not found")
        sys.exit(1)
    
    with open(data_path) as f:
        examples = json.load(f)
    
    print(f"Loaded {len(examples)} examples")
    
    # Load existing model or init random
    models_dir = Path(__file__).parent.parent / "models"
    
    if models_dir.exists():
        print("Loading existing weights...")
        cloud = Cloud.load(models_dir)
    else:
        print("Initializing random weights...")
        cloud = Cloud.random_init(seed=42)
    
    print(f"Total params: {cloud.param_count():,}")
    print()
    
    # Test before
    print("Before training:")
    test_accuracy(cloud, examples)
    
    # Train
    quick_train(cloud, examples, epochs=10, lr=0.01)
    
    # Test after
    print("After training:")
    test_accuracy(cloud, examples)
    
    # Save
    cloud.save(models_dir)
    print(f"Saved to {models_dir}")
    
    print()
    print("=" * 60)
    print("  Training done! Chambers should respond better now.")
    print("=" * 60)
