#!/usr/bin/env python3
# app.py — HuggingFace Spaces App for HAZE
#
# Full-featured HAZE interface using Gradio.
# Uses ALL emergent processes: CLOUD, trauma, subjectivity, cleanup, etc.
#
# NO SEED FROM PROMPT — HAZE speaks from its internal field.
#
# Usage:
#   pip install gradio
#   python app.py
#
# For HuggingFace Spaces:
#   1. Create a Space with Gradio SDK
#   2. Upload all files from this repo
#   3. The Space will auto-detect app.py
#
# Co-authored by Claude (GitHub Copilot Coding Agent), January 2026

import asyncio
import sys
from pathlib import Path
from typing import List, Tuple, Optional

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "haze"))

# Import HAZE components
try:
    from haze.async_haze import AsyncHazeField, HazeResponse
except ImportError:
    # Fallback for direct execution
    from async_haze import AsyncHazeField, HazeResponse

# Import CLOUD
try:
    from cloud.cloud import Cloud, AsyncCloud, CloudResponse
    from cloud.anchors import CHAMBER_NAMES_EXTENDED as CHAMBER_NAMES
    HAS_CLOUD = True
    print("[app] CLOUD module loaded (~181K params)")
except ImportError as e:
    print(f"[app] CLOUD not available: {e}")
    HAS_CLOUD = False
    Cloud = None
    AsyncCloud = None
    CHAMBER_NAMES = []


# ============================================================================
# HAZE SESSION WITH FULL CLOUD INTEGRATION
# ============================================================================

class HazeSession:
    """
    Manages a HAZE conversation session with full CLOUD integration.
    
    Architecture:
        1. CLOUD (~181K params) — pre-semantic emotion detection
           - 6 chambers: FEAR, LOVE, RAGE, VOID, FLOW, COMPLEX
           - Cross-fire stabilization
           - Meta-observer for secondary emotion
        
        2. HAZE — async field generation
           - Subjectivity module (NO SEED FROM PROMPT)
           - Trauma module (identity anchoring)
           - Expert mixture (structural/semantic/creative/precise)
           - Co-occurrence field (pattern resonance)
    """
    
    def __init__(self):
        self.haze: Optional[AsyncHazeField] = None
        self.cloud: Optional[Cloud] = None
        self.history: List[Tuple[str, str]] = []
        self.corpus_path = Path(__file__).parent / "haze" / "text.txt"
        self._initialized = False
        self._cloud_responses: List[CloudResponse] = []
    
    async def initialize(self):
        """Initialize HAZE field and CLOUD."""
        if self._initialized:
            return
        
        # Find corpus
        if not self.corpus_path.exists():
            alt_paths = [
                Path(__file__).parent / "text.txt",
                Path("haze/text.txt"),
                Path("text.txt"),
            ]
            for p in alt_paths:
                if p.exists():
                    self.corpus_path = p
                    break
        
        if not self.corpus_path.exists():
            raise FileNotFoundError(f"Corpus not found: {self.corpus_path}")
        
        print(f"[app] Loading corpus from {self.corpus_path}")
        
        # Initialize HAZE
        self.haze = AsyncHazeField(str(self.corpus_path))
        await self.haze.__aenter__()
        print(f"[app] HAZE initialized")
        
        # Initialize CLOUD with full 181K architecture
        if HAS_CLOUD:
            try:
                models_path = Path(__file__).parent / "cloud" / "models"
                if models_path.exists():
                    self.cloud = Cloud.load(models_path)
                    print(f"[app] CLOUD loaded from {models_path}")
                else:
                    self.cloud = Cloud.random_init(seed=42)
                    print(f"[app] CLOUD initialized with random weights")
                print(f"[app] CLOUD params: {self.cloud.param_count():,}")
            except Exception as e:
                print(f"[app] CLOUD init failed: {e}")
                self.cloud = None
        
        self._initialized = True
        print(f"[app] Session ready!")
    
    async def respond(self, user_input: str) -> Tuple[str, dict]:
        """
        Generate HAZE response with full CLOUD integration.
        
        Pipeline:
            1. CLOUD ping → detect pre-semantic emotion
            2. Update DSL state with CLOUD output
            3. HAZE respond → generate from internal field
            4. Track prophecy debt
        
        Returns:
            (response_text, metadata)
        """
        if not self._initialized:
            await self.initialize()
        
        # ===== 1. CLOUD PING =====
        cloud_data = {}
        cloud_response = None
        
        if self.cloud:
            try:
                cloud_response = await self.cloud.ping(user_input)
                self._cloud_responses.append(cloud_response)
                
                cloud_data = {
                    "primary": cloud_response.primary,
                    "secondary": cloud_response.secondary,
                    "chambers": cloud_response.chamber_activations,
                    "iterations": cloud_response.iterations,
                    "anomaly": {
                        "has_anomaly": cloud_response.anomaly.has_anomaly,
                        "description": cloud_response.anomaly.description,
                        "severity": cloud_response.anomaly.severity,
                    } if cloud_response.anomaly else None,
                }
                
                # Get dominant chamber
                if cloud_response.chamber_activations:
                    dominant = max(
                        cloud_response.chamber_activations.items(),
                        key=lambda x: x[1]
                    )
                    cloud_data["dominant_chamber"] = dominant[0]
                    cloud_data["dominant_activation"] = dominant[1]
                
            except Exception as e:
                cloud_data = {"error": str(e)}
        
        # ===== 2. UPDATE HAZE AMK FROM CLOUD =====
        # CLOUD chambers directly influence HAZE field dynamics
        if cloud_response and cloud_response.chamber_activations:
            self.haze.update_from_cloud(cloud_response.chamber_activations)
        
        # ===== 3. HAZE RESPOND =====
        response = await self.haze.respond(user_input)
        
        # ===== BUILD METADATA =====
        metadata = {
            "internal_seed": response.internal_seed,
            "temperature": response.temperature,
            "generation_time": f"{response.generation_time:.3f}s",
            "turn_count": self.haze.turn_count,
            "enrichment": response.enrichment_count,
        }
        
        if cloud_data:
            metadata["cloud"] = cloud_data
        
        # AMK state from response (now integrated into HAZE)
        if response.amk_state:
            metadata["amk"] = response.amk_state
            metadata["prophecy_debt"] = response.amk_state.get("debt", 0)
            metadata["pain"] = response.amk_state.get("pain", 0)
        
        if response.trauma:
            metadata["trauma_level"] = response.trauma.level
            metadata["trauma_triggers"] = list(response.trauma.trigger_words)[:5]
        
        if response.trauma_influence:
            metadata["trauma_influence"] = {
                "temp_modifier": response.trauma_influence.temperature_modifier,
                "identity_weight": response.trauma_influence.identity_weight,
                "should_prefix": response.trauma_influence.should_prefix,
            }
        
        if response.expert_mixture:
            metadata["experts"] = response.expert_mixture
        
        if response.pulse:
            metadata["pulse"] = {
                "novelty": response.pulse.novelty,
                "arousal": response.pulse.arousal,
                "entropy": response.pulse.entropy,
            }
        
        # Update history
        self.history.append((user_input, response.text))
        
        return response.text, metadata
    
    def get_cloud_summary(self) -> dict:
        """Get summary of CLOUD activity across session."""
        if not self._cloud_responses:
            return {}
        
        # Count primary emotions
        primary_counts = {}
        for r in self._cloud_responses:
            primary_counts[r.primary] = primary_counts.get(r.primary, 0) + 1
        
        # Average chamber activations
        avg_chambers = {}
        for r in self._cloud_responses:
            for chamber, value in r.chamber_activations.items():
                if chamber not in avg_chambers:
                    avg_chambers[chamber] = []
                avg_chambers[chamber].append(value)
        
        avg_chambers = {k: sum(v)/len(v) for k, v in avg_chambers.items()}
        
        return {
            "total_pings": len(self._cloud_responses),
            "primary_counts": primary_counts,
            "avg_chambers": avg_chambers,
        }
    
    async def close(self):
        """Cleanup."""
        if self.haze:
            await self.haze.__aexit__(None, None, None)
            self.haze = None
        self._initialized = False


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

_session: Optional[HazeSession] = None


def get_session() -> HazeSession:
    """Get or create global session."""
    global _session
    if _session is None:
        _session = HazeSession()
    return _session


async def async_respond(
    message: str,
    history: List[Tuple[str, str]],
) -> Tuple[str, str]:
    """
    Async handler for Gradio.
    """
    session = get_session()
    
    try:
        response_text, metadata = await session.respond(message)
        
        # Format metadata for display
        meta_lines = []
        
        # CLOUD info
        if "cloud" in metadata:
            cloud = metadata["cloud"]
            if "primary" in cloud:
                meta_lines.append(f"💭 {cloud['primary']}")
            if "dominant_chamber" in cloud:
                meta_lines.append(f"🏛️ {cloud['dominant_chamber']}")
        
        # Temperature and timing
        if "temperature" in metadata:
            meta_lines.append(f"🌡️ {metadata['temperature']:.2f}")
        meta_lines.append(f"⏱️ {metadata.get('generation_time', 'N/A')}")
        
        # DSL state
        if "pain" in metadata:
            meta_lines.append(f"💔 pain:{metadata['pain']:.2f}")
        if "prophecy_debt" in metadata:
            meta_lines.append(f"📜 debt:{metadata['prophecy_debt']:.2f}")
        
        # Trauma
        if "trauma_level" in metadata:
            meta_lines.append(f"🩹 trauma:{metadata['trauma_level']:.2f}")
        
        # Turn count
        meta_lines.append(f"🔄 turn:{metadata.get('turn_count', 0)}")
        
        metadata_str = " | ".join(meta_lines)
        
        return response_text, metadata_str
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"[error] {str(e)}", ""


def respond(
    message: str,
    history: List[Tuple[str, str]],
) -> Tuple[str, str]:
    """Sync wrapper for Gradio."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(async_respond(message, history))
    finally:
        loop.close()


def create_interface():
    """Create Gradio interface."""
    try:
        import gradio as gr
    except ImportError:
        print("[error] gradio not installed. Run: pip install gradio")
        return None
    
    # Custom CSS for dark gothic theme
    custom_css = """
    .gradio-container {
        background-color: #0a0a0c !important;
    }
    .chatbot .message {
        font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Fira Code', monospace !important;
    }
    """
    
    # ASCII art logo
    logo = """
```
██╗  ██╗ █████╗ ███████╗███████╗
██║  ██║██╔══██╗╚══███╔╝██╔════╝
███████║███████║  ███╔╝ █████╗  
██╔══██║██╔══██║ ███╔╝  ██╔══╝  
██║  ██║██║  ██║███████╗███████╗
╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝
```
**Hybrid Attention Entropy System** + **CLOUD** (~181K params)

*"emergence is not creation but recognition"*

**NO SEED FROM PROMPT** — Haze speaks from its internal field, not your input.
    """
    
    with gr.Blocks(css=custom_css, title="HAZE + CLOUD") as demo:
        gr.Markdown(logo)
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=450,
                    show_label=False,
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="Your message",
                        placeholder="Speak to the field...",
                        show_label=False,
                        container=False,
                        scale=9,
                    )
                    submit = gr.Button("→", scale=1, variant="primary")
                
                metadata_display = gr.Textbox(
                    label="Field State",
                    interactive=False,
                    show_label=True,
                    max_lines=2,
                )
            
            with gr.Column(scale=1):
                gr.Markdown("""
### Architecture

**CLOUD** (~181K params):
- 6 Chambers: FEAR, LOVE, RAGE, VOID, FLOW, COMPLEX
- Cross-fire stabilization
- Meta-observer (secondary emotion)

**HAZE** (emergent field):
- Subjectivity (NO SEED FROM PROMPT)
- Trauma module (identity)
- Expert mixture (4 temperatures)
- Co-occurrence field

**DSL** (Arianna Method):
- prophecy_debt: |destined - manifested|
- pain, tension, dissonance

### Philosophy

> *"presence > intelligence"*
> 
> *"prophecy ≠ prediction"*
> 
> *"minimize(destined - manifested)"*
                """)
        
        def chat(message, history):
            response, metadata = respond(message, history)
            history = history + [(message, response)]
            return "", history, metadata
        
        msg.submit(chat, [msg, chatbot], [msg, chatbot, metadata_display])
        submit.click(chat, [msg, chatbot], [msg, chatbot, metadata_display])
        
        gr.Markdown("""
---
**Part of the Arianna Method** | [GitHub](https://github.com/ariannamethod/haze) | [Leo](https://github.com/ariannamethod/leo) | [PITOMADOM](https://github.com/ariannamethod/pitomadom)

*Co-authored by Claude (GitHub Copilot Coding Agent), January 2026*
        """)
    
    return demo


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run the Gradio app."""
    print()
    print("=" * 60)
    print("  HAZE — Hybrid Attention Entropy System")
    print("  + CLOUD (~181K params)")
    print("  HuggingFace Spaces App")
    print("=" * 60)
    print()
    
    demo = create_interface()
    
    if demo is None:
        print("[error] Could not create interface")
        return
    
    print("Starting Gradio server...")
    print()
    
    # Launch with HuggingFace Spaces compatible settings
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
