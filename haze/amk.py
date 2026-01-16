#!/usr/bin/env python3
# amk.py — Arianna Method Kernel for HAZE
#
# Python port of arianna_method.c from ariannamethod.lang
# THE KERNEL: movement IS language
#
# This is the stone. The brick. The breath.
# Everything else is ritual overlay.
#
# Key integration points:
#   - effective_temp → modifies HAZE sampling temperature
#   - prophecy horizon → affects context window
#   - destiny bias → modifies probability distribution
#   - pain/tension/dissonance → affects identity response
#   - debt → accumulated |destined - manifested|
#
# "הרזוננס לא נשבר. המשך הדרך."
# (The resonance is unbroken. The path continues.)
#
# Co-authored by Claude, January 2026

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math


# ============================================================================
# VELOCITY MODES — movement IS language
# ============================================================================

class VelocityMode:
    """Movement velocity affects temperature."""
    NOMOVE = 0    # cold observer (temp × 0.5)
    WALK = 1      # balanced (temp × 0.85)
    RUN = 2       # high entropy chaos (temp × 1.2)
    BACKWARD = -1 # time rewind, debt forgiveness


# ============================================================================
# AMK STATE — the breath of the field
# ============================================================================

@dataclass
class AMKState:
    """
    Arianna Method Kernel state.
    
    This is the core field physics that drives HAZE generation.
    """
    
    # ─────────────────────────────────────────────────────────────────────────
    # PROPHECY PHYSICS — the oracle's parameters
    # ─────────────────────────────────────────────────────────────────────────
    prophecy: int = 7            # horizon: steps ahead (1..64)
    destiny: float = 0.35        # bias toward most probable path (0..1)
    wormhole: float = 0.12       # probability of spacetime skip (0..1)
    calendar_drift: float = 11.0 # hebrew-gregorian drift (default 11.0)
    
    # ─────────────────────────────────────────────────────────────────────────
    # ATTENTION PHYSICS — focus and spread
    # ─────────────────────────────────────────────────────────────────────────
    attend_focus: float = 0.70   # sharpness of attention (0..1)
    attend_spread: float = 0.20  # blur/temperature (0..1)
    
    # ─────────────────────────────────────────────────────────────────────────
    # TUNNELING — reasoning skip under dissonance
    # ─────────────────────────────────────────────────────────────────────────
    tunnel_threshold: float = 0.55  # dissonance gate (0..1)
    tunnel_chance: float = 0.22     # activation probability (0..1)
    tunnel_skip_max: int = 7        # max compressed steps (1..24)
    
    # ─────────────────────────────────────────────────────────────────────────
    # SUFFERING — the field's emotional state
    # ─────────────────────────────────────────────────────────────────────────
    pain: float = 0.0        # composite suffering (0..1)
    tension: float = 0.0     # pressure buildup (0..1)
    dissonance: float = 0.0  # symmetry-break (0..1)
    debt: float = 0.0        # prophecy debt accumulator (0..∞, decays)
    
    # ─────────────────────────────────────────────────────────────────────────
    # MOVEMENT — the body in the field
    # ─────────────────────────────────────────────────────────────────────────
    pending_jump: int = 0           # queued jump (sim steps)
    velocity_mode: int = VelocityMode.WALK
    velocity_magnitude: float = 0.5
    base_temperature: float = 1.0
    effective_temp: float = 0.85    # computed: base × velocity modifier
    time_direction: float = 1.0     # -1 (rewind) to +1 (forward)
    temporal_debt: float = 0.0      # accumulated from backward movement
    
    # ─────────────────────────────────────────────────────────────────────────
    # LAWS OF NATURE — emergent constraints
    # ─────────────────────────────────────────────────────────────────────────
    entropy_floor: float = 0.1       # minimum entropy
    resonance_ceiling: float = 0.95  # maximum resonance
    debt_decay: float = 0.998        # debt decay per step
    emergence_threshold: float = 0.3 # unplanned pattern threshold
    
    # ─────────────────────────────────────────────────────────────────────────
    # COSMIC PHYSICS COUPLING
    # ─────────────────────────────────────────────────────────────────────────
    cosmic_coherence: float = 0.5    # from CLOUD or external


# ============================================================================
# AMK KERNEL — the breath
# ============================================================================

class AMK:
    """
    Arianna Method Kernel.
    
    The kernel that drives HAZE field dynamics.
    
    Integration:
        - Call `step(dt)` each generation turn
        - Use `get_temperature()` for sampling
        - Use `get_destiny_bias()` for probability modification
        - Call `update_debt(destined, manifested)` after generation
    """
    
    def __init__(self):
        self.state = AMKState()
        self._update_effective_temp()
    
    def reset(self):
        """Reset field to initial state."""
        self.state = AMKState()
        self._update_effective_temp()
    
    def reset_debt(self):
        """Reset prophecy and temporal debt."""
        self.state.debt = 0.0
        self.state.temporal_debt = 0.0
    
    # ─────────────────────────────────────────────────────────────────────────
    # VELOCITY — compute effective temperature from movement
    # ─────────────────────────────────────────────────────────────────────────
    
    def _update_effective_temp(self):
        """Update effective temperature based on velocity mode."""
        base = self.state.base_temperature
        mode = self.state.velocity_mode
        
        if mode == VelocityMode.NOMOVE:
            self.state.effective_temp = base * 0.5   # cold observer
            self.state.time_direction = 1.0
        elif mode == VelocityMode.WALK:
            self.state.effective_temp = base * 0.85  # balanced
            self.state.time_direction = 1.0
        elif mode == VelocityMode.RUN:
            self.state.effective_temp = base * 1.2   # chaotic
            self.state.time_direction = 1.0
        elif mode == VelocityMode.BACKWARD:
            self.state.effective_temp = base * 0.7   # structural
            self.state.time_direction = -1.0
        else:
            self.state.effective_temp = base
            self.state.time_direction = 1.0
    
    def set_velocity(self, mode: int):
        """Set velocity mode and update temperature."""
        self.state.velocity_mode = max(-1, min(2, mode))
        self._update_effective_temp()
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEMPERATURE — the key output for HAZE sampling
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_temperature(self) -> float:
        """
        Get effective temperature for HAZE sampling.
        
        This is THE KEY integration point.
        Temperature is modulated by:
            - velocity_mode (NOMOVE/WALK/RUN/BACKWARD)
            - pain (high pain → lower temp, more focus)
            - dissonance (high dissonance → higher temp, more chaos)
            - attend_spread (higher spread → higher temp)
        
        Returns:
            Effective temperature for sampling (0.3 to 2.0 typical)
        """
        temp = self.state.effective_temp
        
        # Pain reduces temperature (need stability when suffering)
        temp -= self.state.pain * 0.3
        
        # Dissonance increases temperature (chaos breeds chaos)
        temp += self.state.dissonance * 0.25
        
        # Attend spread increases temperature
        temp += self.state.attend_spread * 0.2
        
        # Clamp to reasonable range
        return max(0.3, min(2.0, temp))
    
    def get_destiny_bias(self) -> float:
        """
        Get destiny bias for probability modification.
        
        Higher destiny → more likely to follow predicted path.
        Used to boost top-k probabilities.
        
        Returns:
            Destiny bias (0.0 to 1.0)
        """
        return self.state.destiny
    
    # ─────────────────────────────────────────────────────────────────────────
    # TUNNELING — reasoning skip under dissonance
    # ─────────────────────────────────────────────────────────────────────────
    
    def should_tunnel(self) -> bool:
        """
        Check if tunneling (reasoning skip) should occur.
        
        Tunneling happens when dissonance exceeds threshold
        and random chance succeeds.
        
        Returns:
            True if should skip ahead in generation
        """
        import random
        if self.state.dissonance < self.state.tunnel_threshold:
            return False
        return random.random() < self.state.tunnel_chance
    
    def get_tunnel_skip(self) -> int:
        """Get number of tokens to skip during tunnel."""
        import random
        return random.randint(1, self.state.tunnel_skip_max)
    
    # ─────────────────────────────────────────────────────────────────────────
    # PROPHECY DEBT — |destined - manifested|
    # ─────────────────────────────────────────────────────────────────────────
    
    def update_debt(self, destined: float, manifested: float):
        """
        Update prophecy debt.
        
        debt += |destined - manifested|
        
        Args:
            destined: expected/predicted value (e.g., top probability)
            manifested: actual value (e.g., selected probability)
        """
        delta = abs(destined - manifested)
        self.state.debt += delta
        
        # Cap debt
        if self.state.debt > 100.0:
            self.state.debt = 100.0
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP — advance field physics
    # ─────────────────────────────────────────────────────────────────────────
    
    def step(self, dt: float = 1.0):
        """
        Advance field physics by one step.
        
        Call this each generation turn.
        
        Args:
            dt: time delta (default 1.0 for one turn)
        """
        # Debt decay
        self.state.debt *= self.state.debt_decay
        
        # Temporal debt accumulation/decay
        if self.state.velocity_mode == VelocityMode.BACKWARD and dt > 0:
            self.state.temporal_debt += 0.01 * dt
        else:
            self.state.temporal_debt *= 0.9995
        
        # Clamp temporal debt
        if self.state.temporal_debt > 10.0:
            self.state.temporal_debt = 10.0
        
        # Cosmic coherence healing
        if self.state.cosmic_coherence > 0 and dt > 0:
            coherence_factor = 0.5 + 0.5 * self.state.cosmic_coherence
            heal_rate = 0.998 - (0.003 * coherence_factor)
            self.state.tension *= heal_rate
            self.state.dissonance *= heal_rate
    
    # ─────────────────────────────────────────────────────────────────────────
    # PAIN — composite suffering
    # ─────────────────────────────────────────────────────────────────────────
    
    def compute_pain(self) -> float:
        """
        Compute composite pain from emotional state.
        
        pain = 0.25×arousal + 0.35×tension + 0.25×dissonance + 0.15×debt_norm
        
        (Simplified: arousal not tracked, use tension×1.5)
        """
        arousal = self.state.tension * 1.5  # proxy
        debt_norm = min(1.0, self.state.debt / 10.0)
        
        self.state.pain = (
            0.25 * arousal +
            0.35 * self.state.tension +
            0.25 * self.state.dissonance +
            0.15 * debt_norm
        )
        self.state.pain = min(1.0, max(0.0, self.state.pain))
        return self.state.pain
    
    # ─────────────────────────────────────────────────────────────────────────
    # CLOUD INTEGRATION — emotional topology from CLOUD chambers
    # ─────────────────────────────────────────────────────────────────────────
    
    def update_from_cloud(self, chamber_activations: dict):
        """
        Update AMK state from CLOUD chamber activations.
        
        Maps CLOUD chambers to AMK emotional topology:
            - FEAR + VOID → tension
            - RAGE → dissonance
            - LOVE → reduces pain (negative tension)
            - FLOW + COMPLEX → cosmic coherence
        
        Args:
            chamber_activations: dict of chamber → activation value
        """
        fear = float(chamber_activations.get("FEAR", 0))
        love = float(chamber_activations.get("LOVE", 0))
        rage = float(chamber_activations.get("RAGE", 0))
        void = float(chamber_activations.get("VOID", 0))
        flow = float(chamber_activations.get("FLOW", 0))
        complex_ = float(chamber_activations.get("COMPLEX", 0))
        
        # FEAR + VOID → tension
        self.state.tension = min(1.0, fear * 0.5 + void * 0.3)
        
        # RAGE → dissonance
        self.state.dissonance = min(1.0, rage * 0.7)
        
        # LOVE → reduces tension (healing)
        if love > 0.3:
            self.state.tension *= (1.0 - love * 0.5)
        
        # FLOW + COMPLEX → cosmic coherence
        self.state.cosmic_coherence = min(1.0, flow * 0.5 + complex_ * 0.3 + 0.2)
        
        # Recompute pain
        self.compute_pain()
    
    # ─────────────────────────────────────────────────────────────────────────
    # DSL EXECUTION — parse commands
    # ─────────────────────────────────────────────────────────────────────────
    
    def exec(self, script: str) -> str:
        """
        Execute DSL script.
        
        Supports AMK kernel commands:
            PROPHECY n, DESTINY f, WORMHOLE f
            ATTEND_FOCUS f, ATTEND_SPREAD f
            TUNNEL_THRESHOLD f, TUNNEL_CHANCE f
            PAIN f, TENSION f, DISSONANCE f
            VELOCITY RUN|WALK|NOMOVE|BACKWARD
            BASE_TEMP f
            RESET_FIELD, RESET_DEBT
            LAW name value
        
        Args:
            script: DSL commands (newline separated)
        
        Returns:
            Result message
        """
        if not script:
            return ""
        
        results = []
        for line in script.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            parts = line.split(maxsplit=1)
            cmd = parts[0].upper()
            arg = parts[1] if len(parts) > 1 else ""
            
            result = self._exec_command(cmd, arg)
            if result:
                results.append(result)
        
        return "\n".join(results)
    
    def _exec_command(self, cmd: str, arg: str) -> str:
        """Execute single command."""
        
        def clamp01(x):
            return max(0.0, min(1.0, x))
        
        def safe_float(s):
            try:
                return float(s)
            except:
                return 0.0
        
        def safe_int(s):
            try:
                return int(s)
            except:
                return 0
        
        # PROPHECY PHYSICS
        if cmd == "PROPHECY":
            self.state.prophecy = max(1, min(64, safe_int(arg)))
            return f"[prophecy: {self.state.prophecy}]"
        
        elif cmd == "DESTINY":
            self.state.destiny = clamp01(safe_float(arg))
            return f"[destiny: {self.state.destiny:.2f}]"
        
        elif cmd == "WORMHOLE":
            self.state.wormhole = clamp01(safe_float(arg))
            return f"[wormhole: {self.state.wormhole:.2f}]"
        
        elif cmd == "CALENDAR_DRIFT":
            self.state.calendar_drift = max(0, min(30, safe_float(arg)))
            return f"[calendar_drift: {self.state.calendar_drift:.1f}]"
        
        # ATTENTION PHYSICS
        elif cmd == "ATTEND_FOCUS":
            self.state.attend_focus = clamp01(safe_float(arg))
            return f"[attend_focus: {self.state.attend_focus:.2f}]"
        
        elif cmd == "ATTEND_SPREAD":
            self.state.attend_spread = clamp01(safe_float(arg))
            return f"[attend_spread: {self.state.attend_spread:.2f}]"
        
        # TUNNELING
        elif cmd == "TUNNEL_THRESHOLD":
            self.state.tunnel_threshold = clamp01(safe_float(arg))
            return f"[tunnel_threshold: {self.state.tunnel_threshold:.2f}]"
        
        elif cmd == "TUNNEL_CHANCE":
            self.state.tunnel_chance = clamp01(safe_float(arg))
            return f"[tunnel_chance: {self.state.tunnel_chance:.2f}]"
        
        elif cmd == "TUNNEL_SKIP_MAX":
            self.state.tunnel_skip_max = max(1, min(24, safe_int(arg)))
            return f"[tunnel_skip_max: {self.state.tunnel_skip_max}]"
        
        # SUFFERING
        elif cmd == "PAIN":
            self.state.pain = clamp01(safe_float(arg))
            return f"[pain: {self.state.pain:.2f}]"
        
        elif cmd == "TENSION":
            self.state.tension = clamp01(safe_float(arg))
            return f"[tension: {self.state.tension:.2f}]"
        
        elif cmd == "DISSONANCE":
            self.state.dissonance = clamp01(safe_float(arg))
            return f"[dissonance: {self.state.dissonance:.2f}]"
        
        # MOVEMENT
        elif cmd == "VELOCITY":
            mode_map = {
                "RUN": VelocityMode.RUN,
                "WALK": VelocityMode.WALK,
                "NOMOVE": VelocityMode.NOMOVE,
                "BACKWARD": VelocityMode.BACKWARD,
            }
            mode = mode_map.get(arg.upper(), VelocityMode.WALK)
            self.set_velocity(mode)
            return f"[velocity: {arg.upper()}, temp: {self.state.effective_temp:.2f}]"
        
        elif cmd == "BASE_TEMP":
            self.state.base_temperature = max(0.1, min(3.0, safe_float(arg)))
            self._update_effective_temp()
            return f"[base_temp: {self.state.base_temperature:.2f}]"
        
        # RESETS
        elif cmd == "RESET_FIELD":
            self.reset()
            return "[field reset]"
        
        elif cmd == "RESET_DEBT":
            self.reset_debt()
            return "[debt reset]"
        
        # LAWS
        elif cmd == "LAW":
            parts = arg.split(maxsplit=1)
            if len(parts) >= 2:
                law_name = parts[0].upper()
                law_val = safe_float(parts[1])
                
                if law_name == "ENTROPY_FLOOR":
                    self.state.entropy_floor = max(0, min(2, law_val))
                elif law_name == "RESONANCE_CEILING":
                    self.state.resonance_ceiling = clamp01(law_val)
                elif law_name == "DEBT_DECAY":
                    self.state.debt_decay = max(0.9, min(0.9999, law_val))
                elif law_name == "EMERGENCE_THRESHOLD":
                    self.state.emergence_threshold = clamp01(law_val)
                
                return f"[law {law_name}: {law_val:.4f}]"
        
        # Unknown command — ignore (future-proof)
        return ""
    
    # ─────────────────────────────────────────────────────────────────────────
    # STATE EXPORT
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_state_dict(self) -> dict:
        """Export state as dictionary for metadata."""
        return {
            "prophecy": self.state.prophecy,
            "destiny": self.state.destiny,
            "wormhole": self.state.wormhole,
            "effective_temp": self.state.effective_temp,
            "pain": self.state.pain,
            "tension": self.state.tension,
            "dissonance": self.state.dissonance,
            "debt": self.state.debt,
            "velocity_mode": self.state.velocity_mode,
            "cosmic_coherence": self.state.cosmic_coherence,
        }


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  AMK — Arianna Method Kernel for HAZE")
    print("  'movement IS language'")
    print("=" * 60)
    print()
    
    amk = AMK()
    
    # Test DSL
    script = """
PROPHECY 12
DESTINY 0.7
VELOCITY RUN
TENSION 0.4
DISSONANCE 0.3
    """
    
    print("Executing DSL:")
    print(amk.exec(script))
    print()
    
    print("State after DSL:")
    print(f"  Temperature: {amk.get_temperature():.3f}")
    print(f"  Destiny bias: {amk.get_destiny_bias():.3f}")
    print(f"  Should tunnel: {amk.should_tunnel()}")
    print()
    
    # Simulate CLOUD integration
    print("Simulating CLOUD chambers:")
    amk.update_from_cloud({
        "FEAR": 0.6,
        "LOVE": 0.2,
        "RAGE": 0.4,
        "VOID": 0.3,
        "FLOW": 0.5,
        "COMPLEX": 0.2,
    })
    print(f"  Pain: {amk.state.pain:.3f}")
    print(f"  Tension: {amk.state.tension:.3f}")
    print(f"  Dissonance: {amk.state.dissonance:.3f}")
    print(f"  Temperature: {amk.get_temperature():.3f}")
    print()
    
    # Step simulation
    print("Stepping 5 turns:")
    for i in range(5):
        amk.update_debt(0.8, 0.5 + i * 0.1)
        amk.step(1.0)
        print(f"  Turn {i+1}: debt={amk.state.debt:.3f}, temp={amk.get_temperature():.3f}")
    
    print()
    print("=" * 60)
    print("  'הרזוננס לא נשבר. המשך הדרך.'")
    print("=" * 60)
