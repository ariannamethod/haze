"""
mathbrain.py — Body/Field Perception for Haze

Async MLP on pure numpy (micrograd-style) for field signal processing.
Inspired by Leo's body perception module.

This is NOT for language generation — it's for internal field state.
The "brain" perceives:
- Pulse signals (arousal, novelty, entropy)
- Trauma state
- Expert mixture
- Field coherence

And produces:
- Internal temperature adjustments
- Identity weight modulations
- Field "mood" (calm, excited, focused, diffuse)

No PyTorch. No TensorFlow. Just numpy and the void.
"""

import asyncio
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from collections import deque
import time
import json
from pathlib import Path


# ============================================================
#  ACTIVATION FUNCTIONS (pure numpy)
# ============================================================

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation."""
    return np.maximum(0, x)

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation with numerical stability."""
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))

def tanh(x: np.ndarray) -> np.ndarray:
    """Tanh activation."""
    return np.tanh(x)

def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax with numerical stability."""
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)

def gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation (approximation)."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


# ============================================================
#  MLP LAYER (pure numpy, no autograd)
# ============================================================

@dataclass
class MLPLayer:
    """Single MLP layer with weights, biases, and activation."""
    
    weights: np.ndarray  # (input_dim, output_dim)
    biases: np.ndarray   # (output_dim,)
    activation: str = "relu"
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        out = x @ self.weights + self.biases
        
        if self.activation == "relu":
            return relu(out)
        elif self.activation == "sigmoid":
            return sigmoid(out)
        elif self.activation == "tanh":
            return tanh(out)
        elif self.activation == "gelu":
            return gelu(out)
        elif self.activation == "none" or self.activation is None:
            return out
        else:
            return out
    
    @classmethod
    def random(cls, input_dim: int, output_dim: int, 
               activation: str = "relu", scale: float = 0.1) -> "MLPLayer":
        """Create layer with random weights (Xavier-like init)."""
        weights = np.random.randn(input_dim, output_dim) * scale
        biases = np.zeros(output_dim)
        return cls(weights=weights, biases=biases, activation=activation)


# ============================================================
#  MATHBRAIN (async MLP for field perception)
# ============================================================

@dataclass
class FieldPerception:
    """What mathbrain perceives about the field state."""
    
    # Raw signals (0-1)
    arousal: float = 0.5
    novelty: float = 0.0
    entropy: float = 0.7
    trauma: float = 0.0
    coherence: float = 0.5
    
    # Derived states
    mood: str = "calm"  # calm, excited, focused, diffuse, alert
    recommended_temp: float = 0.6
    identity_weight: float = 0.0
    
    # Internal state
    internal_signal: np.ndarray = field(default_factory=lambda: np.zeros(8))
    
    def to_dict(self) -> Dict:
        return {
            "arousal": round(self.arousal, 3),
            "novelty": round(self.novelty, 3),
            "entropy": round(self.entropy, 3),
            "trauma": round(self.trauma, 3),
            "coherence": round(self.coherence, 3),
            "mood": self.mood,
            "recommended_temp": round(self.recommended_temp, 3),
            "identity_weight": round(self.identity_weight, 3),
        }


class MathBrain:
    """
    Async MLP for field perception.
    
    Architecture:
    - Input: 5 signals (arousal, novelty, entropy, trauma, coherence)
    - Hidden1: 16 neurons (relu)
    - Hidden2: 8 neurons (tanh) 
    - Output: 4 signals (temp_adjust, identity_weight, mood_arousal, mood_focus)
    
    The brain learns through Hebbian-like updates, not backprop.
    Connections that fire together strengthen together.
    """
    
    def __init__(self, hidden_dims: Tuple[int, ...] = (16, 8)):
        self.input_dim = 5
        self.output_dim = 4
        self.hidden_dims = hidden_dims
        
        # Build layers
        dims = [self.input_dim] + list(hidden_dims) + [self.output_dim]
        self.layers: List[MLPLayer] = []
        
        for i in range(len(dims) - 1):
            activation = "relu" if i < len(dims) - 2 else "sigmoid"
            layer = MLPLayer.random(
                dims[i], dims[i + 1],
                activation=activation,
                scale=0.1
            )
            self.layers.append(layer)
        
        # Memory (last N perceptions for Hebbian learning)
        self.memory: deque = deque(maxlen=100)
        
        # Lock for async safety
        self._lock = asyncio.Lock()
        
        # Stats
        self.total_perceptions = 0
        self.last_perception_time = 0.0
    
    def _forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through all layers."""
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out
    
    def _input_vector(self, arousal: float, novelty: float, entropy: float,
                      trauma: float, coherence: float) -> np.ndarray:
        """Create input vector from signals."""
        return np.array([arousal, novelty, entropy, trauma, coherence])
    
    def _interpret_output(self, output: np.ndarray) -> Tuple[float, float, str]:
        """Interpret output vector into temp, identity weight, mood."""
        temp_adjust = output[0]  # 0-1 → 0.4-1.2
        identity_weight = output[1]  # 0-1
        mood_arousal = output[2]  # low = calm, high = excited
        mood_focus = output[3]  # low = diffuse, high = focused
        
        # Map to temperature (0.4 - 1.2)
        recommended_temp = 0.4 + temp_adjust * 0.8
        
        # Determine mood
        if mood_arousal > 0.6 and mood_focus > 0.6:
            mood = "alert"
        elif mood_arousal > 0.6:
            mood = "excited"
        elif mood_focus > 0.6:
            mood = "focused"
        elif mood_arousal < 0.3 and mood_focus < 0.3:
            mood = "diffuse"
        else:
            mood = "calm"
        
        return recommended_temp, identity_weight, mood
    
    async def perceive(
        self,
        arousal: float = 0.5,
        novelty: float = 0.0,
        entropy: float = 0.7,
        trauma: float = 0.0,
        coherence: float = 0.5,
    ) -> FieldPerception:
        """
        Perceive the field state and return recommendations.
        
        This is the main entry point. Feed it the current field signals
        and it returns what the brain thinks about the state.
        """
        async with self._lock:
            start_time = time.time()
            
            # Create input
            x = self._input_vector(arousal, novelty, entropy, trauma, coherence)
            
            # Forward pass
            output = self._forward(x)
            
            # Interpret
            recommended_temp, identity_weight, mood = self._interpret_output(output)
            
            # Create perception
            perception = FieldPerception(
                arousal=arousal,
                novelty=novelty,
                entropy=entropy,
                trauma=trauma,
                coherence=coherence,
                mood=mood,
                recommended_temp=recommended_temp,
                identity_weight=identity_weight,
                internal_signal=output.copy(),
            )
            
            # Store in memory
            self.memory.append({
                "input": x.copy(),
                "output": output.copy(),
                "perception": perception.to_dict(),
                "timestamp": time.time(),
            })
            
            self.total_perceptions += 1
            self.last_perception_time = time.time() - start_time
            
            return perception
    
    async def hebbian_update(self, reward: float = 0.0):
        """
        Hebbian-like weight update.
        
        If reward > 0: strengthen connections that produced this output
        If reward < 0: weaken connections that produced this output
        
        This is NOT backprop. It's a simple correlation-based update.
        """
        async with self._lock:
            if not self.memory:
                return
            
            # Get last perception
            last = self.memory[-1]
            x = last["input"]
            
            # Learning rate
            lr = 0.01 * reward
            
            # Update first layer (input → hidden1)
            # Hebbian rule: Δw = lr * x_i * y_j
            y = relu(x @ self.layers[0].weights + self.layers[0].biases)
            delta = lr * np.outer(x, y)
            self.layers[0].weights += delta
    
    async def get_stats(self) -> Dict:
        """Get brain statistics."""
        async with self._lock:
            return {
                "total_perceptions": self.total_perceptions,
                "memory_size": len(self.memory),
                "layer_shapes": [(l.weights.shape) for l in self.layers],
                "last_perception_time_ms": round(self.last_perception_time * 1000, 3),
            }
    
    def save(self, path: str):
        """Save weights to file."""
        data = {
            "layers": [
                {
                    "weights": layer.weights.tolist(),
                    "biases": layer.biases.tolist(),
                    "activation": layer.activation,
                }
                for layer in self.layers
            ],
            "total_perceptions": self.total_perceptions,
        }
        Path(path).write_text(json.dumps(data, indent=2))
    
    @classmethod
    def load(cls, path: str) -> "MathBrain":
        """Load weights from file."""
        data = json.loads(Path(path).read_text())
        brain = cls()
        brain.layers = [
            MLPLayer(
                weights=np.array(layer["weights"]),
                biases=np.array(layer["biases"]),
                activation=layer["activation"],
            )
            for layer in data["layers"]
        ]
        brain.total_perceptions = data.get("total_perceptions", 0)
        return brain


# ============================================================
#  ASYNC WRAPPER
# ============================================================

class AsyncMathBrain(MathBrain):
    """
    Async-ready MathBrain with additional features:
    - Continuous perception loop (optional)
    - Signal smoothing
    - Decay over time
    """
    
    def __init__(self, hidden_dims: Tuple[int, ...] = (16, 8)):
        super().__init__(hidden_dims)
        
        # Signal smoothing (exponential moving average)
        self._ema_alpha = 0.3
        self._smoothed_signals: Optional[np.ndarray] = None
        
        # Running state
        self._running = False
    
    async def perceive_smooth(
        self,
        arousal: float = 0.5,
        novelty: float = 0.0,
        entropy: float = 0.7,
        trauma: float = 0.0,
        coherence: float = 0.5,
    ) -> FieldPerception:
        """
        Perceive with signal smoothing (EMA).
        
        This makes the brain less reactive to sudden changes.
        """
        current = np.array([arousal, novelty, entropy, trauma, coherence])
        
        if self._smoothed_signals is None:
            self._smoothed_signals = current.copy()
        else:
            self._smoothed_signals = (
                self._ema_alpha * current + 
                (1 - self._ema_alpha) * self._smoothed_signals
            )
        
        return await self.perceive(
            arousal=float(self._smoothed_signals[0]),
            novelty=float(self._smoothed_signals[1]),
            entropy=float(self._smoothed_signals[2]),
            trauma=float(self._smoothed_signals[3]),
            coherence=float(self._smoothed_signals[4]),
        )
    
    async def close(self):
        """Cleanup."""
        self._running = False


# ============================================================
#  DEMO
# ============================================================

async def demo():
    """Demonstrate mathbrain perception."""
    print("=" * 60)
    print("  🧠 MATHBRAIN DEMO — Field Perception")
    print("=" * 60)
    print()
    
    brain = AsyncMathBrain()
    
    # Test scenarios
    scenarios = [
        ("Calm baseline", dict(arousal=0.3, novelty=0.1, entropy=0.6, trauma=0.0, coherence=0.7)),
        ("High arousal", dict(arousal=0.9, novelty=0.2, entropy=0.7, trauma=0.1, coherence=0.6)),
        ("High trauma", dict(arousal=0.4, novelty=0.3, entropy=0.5, trauma=0.8, coherence=0.4)),
        ("Creative chaos", dict(arousal=0.6, novelty=0.8, entropy=0.9, trauma=0.2, coherence=0.3)),
        ("Focused precision", dict(arousal=0.2, novelty=0.1, entropy=0.3, trauma=0.0, coherence=0.9)),
    ]
    
    for name, signals in scenarios:
        perception = await brain.perceive(**signals)
        print(f"📊 {name}")
        print(f"   signals: arousal={signals['arousal']:.1f} novelty={signals['novelty']:.1f} "
              f"entropy={signals['entropy']:.1f} trauma={signals['trauma']:.1f}")
        print(f"   → mood={perception.mood} temp={perception.recommended_temp:.2f} "
              f"identity={perception.identity_weight:.2f}")
        print()
    
    stats = await brain.get_stats()
    print(f"Stats: {stats}")
    print()
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(demo())
