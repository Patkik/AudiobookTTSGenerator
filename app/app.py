"""
Kokoro Expressive Audiobook TTS - Gradio Web UI
Run: python -m app.app
"""
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path when running app/app.py directly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import tempfile
import dataclasses
import numpy as np
import gradio as gr
import soundfile as sf

from app.parser.tag_parser import TagParser
from app.parser.script_resolver import ScriptResolver
from app.parser.narrative_framing import DefaultNarrativeContextResolver
from app.synthesis.engine import SynthesisEngine
from app.synthesis.audio_pipeline import AudioPipeline
from app.synthesis.rvc_pipeline import RVCPipeline, RVCConfig
from app.characters.registry import CharacterRegistry
from app.parser.emotion_profiles import EMOTION_PROFILES

registry = CharacterRegistry()
parser = TagParser()
resolver = ScriptResolver()
engine: SynthesisEngine | None = None

SAMPLE_RATE = 24_000
ALL_TAGS = sorted(EMOTION_PROFILES.keys())

def get_model_path() -> str:
    int8 = str(ROOT_DIR / "models" / "kokoro-expressive.int8.onnx")
    if os.path.exists(int8) and os.path.getsize(int8) > 50_000_000:
        return int8
    v1 = str(ROOT_DIR / "models" / "kokoro-v1.0.onnx")
    if os.path.exists(v1):
        return v1
    return int8

def check_models_ok() -> bool:
    m = get_model_path()
    voices = str(ROOT_DIR / "models" / "voices-v1.0.bin")
    return (
        os.path.exists(m)
        and os.path.exists(voices)
        and os.path.getsize(m) > 50_000_000
    )

def _get_engine() -> SynthesisEngine:
    global engine
    if engine is None:
        m = get_model_path()
        voices = str(ROOT_DIR / "models" / "voices-v1.0.bin")
        emotions = str(ROOT_DIR / "emotions")
        print(f"Loading SynthesisEngine with model: {m}, voices: {voices}")
        rvc_pipe = RVCPipeline()
        narrative_resolver = DefaultNarrativeContextResolver()
        engine = SynthesisEngine(
            model_path=m,
            voices_path=voices,
            emotions_dir=emotions,
            narrative_context_resolver=narrative_resolver,
            rvc_pipeline=rvc_pipe,
            enable_rvc=True,
        )
    return engine

def generate_audio(
    text: str,
    global_speed: float = 1.0,
    intensity: float = 0.35,
    enable_mastering: bool = True,
    enable_rvc: bool = True,
    rvc_index_rate: float = 0.40,
    rvc_protect: float = 0.35,
    rvc_filter_radius: int = 3,
    rvc_volume_envelope: float = 1.0,
    rvc_pitch_shift: int = 0,
) -> tuple[str | None, str]:
    """Parse text and synthesize audio. Returns (wav_path, segment_summary)."""
    if not check_models_ok():
        return None, "⚠️ Model files not ready. Please verify models/kokoro-expressive.int8.onnx (or kokoro-v1.0.onnx) and models/voices-v1.0.bin."
    if not text or not text.strip():
        return None, "⚠️ No text provided."

    eng = _get_engine()
    eng.set_emotion_alpha(intensity)
    eng.set_mastering(enable_mastering)
    eng.set_enable_rvc(enable_rvc)

    # Update rudeus RVC configuration with UI parameters
    rudeus_cfg = eng.get_rvc_config("rudeus")
    if rudeus_cfg:
        updated_cfg = dataclasses.replace(
            rudeus_cfg,
            index_rate=float(rvc_index_rate),
            protect=float(rvc_protect),
            filter_radius=int(rvc_filter_radius),
            volume_envelope=float(rvc_volume_envelope),
            f0_up_key=int(rvc_pitch_shift),
        )
        eng.set_rvc_config("rudeus", updated_cfg)

    segments = parser.parse(text)
    if not segments:
        return None, "⚠️ No speech segments found."

    if global_speed != 1.0:
        adjusted = []
        for s in segments:
            new_profile = dataclasses.replace(s.profile, speed=s.profile.speed * global_speed)
            adjusted.append(dataclasses.replace(s, profile=new_profile))
        segments = adjusted

    audio, sr = eng.synthesize_segments(segments, registry, enable_mastering=enable_mastering)

    if len(audio) == 0:
        return None, "⚠️ Synthesis produced no audio."

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, sr)

    summary_lines = []
    for i, s in enumerate(segments):
        if s.is_silence:
            summary_lines.append(f"  {i+1}. [PAUSE {s.profile.silence_ms}ms]")
        else:
            snippet = s.text[:40] + ("..." if len(s.text) > 40 else "")
            rvc_tag = f" [RVC:{registry.get_rvc_model(s.character)}]" if (enable_rvc and registry.get_rvc_model(s.character)) else ""
            summary_lines.append(
                f"  {i+1}. [{s.emotion}]{rvc_tag} {s.character}: \"{snippet}\""
            )
    summary = f"✓ {len(segments)} segments | {len(audio)/sr:.1f}s total\n" + "\n".join(summary_lines)

    return tmp.name, summary

def load_file(file) -> str:
    """Handle uploaded .txt or .epub files."""
    if file is None:
        return ""
    filename = os.path.basename(file.name)
    with open(file.name, "rb") as f:
        raw = f.read()
    return resolver.load_text(raw, filename=filename)

def insert_tag(current_text: str, tag: str) -> str:
    """Insert a tag at end of current text."""
    prefix = current_text if current_text else ""
    return prefix + f" [{tag}] "

def add_character(name: str, voice: str, gender: str, rvc_model: str, current_table) -> tuple:
    """Add a character to the registry and refresh the table."""
    name = name.strip().upper()
    rvc_val = None if rvc_model in ("None", "—", "", "none") else rvc_model
    if name:
        registry.add(name, voice, gender, rvc_model=rvc_val)
    chars = registry.all_characters()
    return "", [[c["name"], c["voice_id"], c["gender"], c.get("rvc_model") or "—"] for c in chars]

def export_mp3(wav_path: str) -> str | None:
    """Convert generated WAV to MP3."""
    if not wav_path or not os.path.exists(wav_path):
        return None
    mp3_path = wav_path.replace(".wav", ".mp3")
    audio, sr = sf.read(wav_path, dtype="float32")
    AudioPipeline.save_mp3(audio, mp3_path, sr)
    return mp3_path

def _get_voices():
    voices_path = str(ROOT_DIR / "models" / "voices-v1.0.bin")
    if os.path.exists(voices_path):
        try:
            voices_data = np.load(voices_path)
            return list(voices_data.keys())
        except Exception:
            pass
    return ["af_heart", "af_bella", "am_michael", "bm_george"]

with gr.Blocks(title="🎙️ Kokoro Expressive Audiobook TTS") as demo:
    gr.Markdown("# 🎙️ Kokoro Expressive Audiobook TTS")
    
    with gr.Tabs():
        # ── TAB 1: EDITOR ─────────────────────────────────────────────────────
        with gr.Tab("📝 Editor"):
            with gr.Row():
                with gr.Column(scale=3):
                    text_input = gr.Textbox(
                        label="Script (with expression tags)",
                        placeholder=(
                            "NARRATOR: It was a dark and stormy night.\n"
                            "[sad] She whispered, \"I can't do this anymore.\"\n"
                            "[gasps] A shadow appeared at the window.\n"
                            "ALICE (angry): Get out of my house!"
                        ),
                        lines=14,
                        max_lines=40,
                    )
                    with gr.Row():
                        upload_btn = gr.UploadButton(
                            "📂 Import .txt / .epub",
                            file_types=[".txt", ".epub"],
                        )
                        tag_picker = gr.Dropdown(
                            choices=ALL_TAGS,
                            label="🎭 Insert Tag",
                            value=None,
                            interactive=True,
                        )

                with gr.Column(scale=2):
                    audio_output = gr.Audio(
                        label="Generated Audio",
                        type="filepath",
                        interactive=False,
                    )
                    segment_summary = gr.Textbox(
                        label="Segment Breakdown",
                        lines=8,
                        interactive=False,
                    )
                    export_btn = gr.Button("⬇️ Export MP3", variant="secondary")
                    mp3_output = gr.File(label="MP3 Download", visible=False)

            with gr.Row():
                generate_btn = gr.Button("▶ Generate Audio", variant="primary", scale=2)
                intensity_slider = gr.Slider(
                    minimum=0.0, maximum=0.8, value=0.35, step=0.05,
                    label="🎭 Expression Intensity", scale=1,
                    info="0.0=Neutral, 0.35=Genuine & Conversational, 0.7=Dramatic",
                )
                speed_slider = gr.Slider(
                    minimum=0.5, maximum=2.0, value=1.0, step=0.05,
                    label="⚡ Global Speed", scale=1,
                )
                mastering_toggle = gr.Checkbox(
                    value=True,
                    label="🎛️ Studio Mastering",
                    scale=1,
                    info="EQ, De-Esser & Room Tone",
                )

            with gr.Accordion("🎙️ RVC Character Voice Conversion (Rudeus / Custom)", open=False):
                with gr.Row():
                    rvc_toggle = gr.Checkbox(
                        value=True,
                        label="Enable RVC Character Conversion",
                        info="Converts timbre for characters with assigned RVC models (e.g. RUDEUS)",
                    )
                    rvc_pitch_shift = gr.Slider(
                        minimum=-12, maximum=12, value=0, step=1,
                        label="🎵 Pitch Shift (Semitones)",
                        info="0=Default, +12=Female octave, -12=Male octave",
                    )
                with gr.Row():
                    rvc_index_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.40, step=0.05,
                        label="🎚️ Index Feature Rate (index_rate)",
                        info="Blueprint: 0.30–0.50 (retains Kokoro acting & pitch range)",
                    )
                    rvc_protect_slider = gr.Slider(
                        minimum=0.0, maximum=0.50, value=0.35, step=0.05,
                        label="🛡️ Protect Voiceless Consonants (protect)",
                        info="Blueprint: 0.33–0.50 (protects unvoiced breath noise & consonants)",
                    )
                with gr.Row():
                    rvc_filter_slider = gr.Slider(
                        minimum=0, maximum=7, value=3, step=1,
                        label="〰️ Filter Radius",
                        info="Blueprint: 3 (smooths pitch transitions to eliminate cracks)",
                    )
                    rvc_volume_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, value=1.0, step=0.05,
                        label="🔊 Volume Envelope Mix",
                        info="Blueprint: 1.0 (retains Kokoro dynamic range shouts vs whispers)",
                    )

        # ── TAB 2: CHARACTERS ─────────────────────────────────────────────────
        with gr.Tab("👥 Characters"):
            gr.Markdown("Assign a Kokoro voice and optional RVC model to each character in your script.")
            voices_list = _get_voices()

            char_table = gr.Dataframe(
                headers=["name", "voice_id", "gender", "rvc_model"],
                value=[[c["name"], c["voice_id"], c["gender"], c.get("rvc_model") or "—"]
                       for c in registry.all_characters()],
                label="Registered Characters",
                interactive=False,
            )

            with gr.Row():
                new_name = gr.Textbox(label="Character Name (e.g. ALICE)", scale=2)
                new_voice = gr.Dropdown(
                    choices=voices_list,
                    value=voices_list[0] if voices_list else "af_heart",
                    label="Voice",
                    scale=2,
                )
                new_gender = gr.Radio(
                    choices=["female", "male"],
                    value="female",
                    label="Gender (for SFX selection)",
                    scale=1,
                )
                new_rvc = gr.Dropdown(
                    choices=["None", "rudeus"],
                    value="None",
                    label="RVC Model",
                    scale=1,
                )
                add_char_btn = gr.Button("➕ Add / Update", scale=1)

        # ── TAB 3: TAG REFERENCE ──────────────────────────────────────────────
        with gr.Tab("📖 Tag Reference"):
            tag_rows = [
                [tag, f"{p.speed:.2f}×", f"{p.volume_db:+.0f}dB",
                 f"{p.pause_after_ms}ms", p.sfx_file or "—",
                 "Yes" if p.silence_ms else "No"]
                for tag, p in sorted(EMOTION_PROFILES.items())
            ]
            gr.Dataframe(
                headers=["Tag", "Speed", "Volume", "Pause After", "SFX", "Silence Only"],
                value=tag_rows,
                label="All Expression Tags",
                interactive=False,
            )
            gr.Markdown("""
**Inline format:** `[happy] She smiled.`  
**Compound:** `[whispering, nervous] "Did you hear that?"`  
**Screenplay:** `ALICE (angry): I won't stand for this!`  
**Pauses:** `[pause]` (500ms)  `[long pause]` (1200ms)  
""")

    # ── Event Wiring ──────────────────────────────────────────────────────────
    generate_btn.click(
        fn=generate_audio,
        inputs=[
            text_input, speed_slider, intensity_slider, mastering_toggle,
            rvc_toggle, rvc_index_slider, rvc_protect_slider,
            rvc_filter_slider, rvc_volume_slider, rvc_pitch_shift,
        ],
        outputs=[audio_output, segment_summary],
    )
    upload_btn.upload(
        fn=load_file,
        inputs=upload_btn,
        outputs=text_input,
    )
    tag_picker.change(
        fn=insert_tag,
        inputs=[text_input, tag_picker],
        outputs=text_input,
    )
    add_char_btn.click(
        fn=add_character,
        inputs=[new_name, new_voice, new_gender, new_rvc, char_table],
        outputs=[new_name, char_table],
    )
    export_btn.click(
        fn=export_mp3,
        inputs=audio_output,
        outputs=mp3_output,
    )


if __name__ == "__main__":
    import os
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
