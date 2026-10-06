"""
Kokoro Expressive Audiobook TTS - Gradio Web UI
Run: python -m app.app
"""
import os
import tempfile
import numpy as np
import gradio as gr
import soundfile as sf

from app.parser.tag_parser import TagParser
from app.parser.script_resolver import ScriptResolver
from app.synthesis.engine import SynthesisEngine
from app.synthesis.audio_pipeline import AudioPipeline
from app.characters.registry import CharacterRegistry
from app.parser.emotion_profiles import EMOTION_PROFILES

registry = CharacterRegistry()
parser = TagParser()
resolver = ScriptResolver()
engine: SynthesisEngine | None = None

SAMPLE_RATE = 24_000
ALL_TAGS = sorted(EMOTION_PROFILES.keys())

def check_models_ok() -> bool:
    return (
        os.path.exists("models/kokoro-v1.0.onnx")
        and os.path.exists("models/voices-v1.0.bin")
        and os.path.getsize("models/kokoro-v1.0.onnx") > 100_000_000
    )

def _get_engine() -> SynthesisEngine:
    global engine
    if engine is None:
        engine = SynthesisEngine(
            model_path="models/kokoro-v1.0.onnx",
            voices_path="models/voices-v1.0.bin",
        )
    return engine

def generate_audio(text: str, global_speed: float) -> tuple[str | None, str]:
    """Parse text and synthesize audio. Returns (wav_path, segment_summary)."""
    if not check_models_ok():
        return None, "⚠️ Model files not ready or still downloading. Please verify models/kokoro-v1.0.onnx and models/voices-v1.0.bin."
    if not text or not text.strip():
        return None, "⚠️ No text provided."

    eng = _get_engine()
    segments = parser.parse(text)
    if not segments:
        return None, "⚠️ No speech segments found."

    audio, sr = eng.synthesize_segments(segments, registry)

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
            summary_lines.append(
                f"  {i+1}. [{s.emotion}] {s.character}: \"{snippet}\""
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

def add_character(name: str, voice: str, gender: str, current_table) -> tuple:
    """Add a character to the registry and refresh the table."""
    name = name.strip().upper()
    if name:
        registry.add(name, voice, gender)
    chars = registry.all_characters()
    return "", [[c["name"], c["voice_id"], c["gender"]] for c in chars]

def export_mp3(wav_path: str) -> str | None:
    """Convert generated WAV to MP3."""
    if not wav_path or not os.path.exists(wav_path):
        return None
    mp3_path = wav_path.replace(".wav", ".mp3")
    audio, sr = sf.read(wav_path, dtype="float32")
    AudioPipeline.save_mp3(audio, mp3_path, sr)
    return mp3_path

def _get_voices():
    if os.path.exists("models/voices-v1.0.bin"):
        try:
            voices_data = np.load("models/voices-v1.0.bin")
            return list(voices_data.keys())
        except Exception:
            pass
    return ["af_heart", "af_bella", "am_michael", "bm_george"]

with gr.Blocks(title="🎙️ Kokoro Expressive Audiobook TTS", theme=gr.themes.Soft()) as demo:
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
                generate_btn = gr.Button("▶ Generate Audio", variant="primary", scale=3)
                speed_slider = gr.Slider(
                    minimum=0.5, maximum=2.0, value=1.0, step=0.05,
                    label="Global Speed", scale=1,
                )

        # ── TAB 2: CHARACTERS ─────────────────────────────────────────────────
        with gr.Tab("👥 Characters"):
            gr.Markdown("Assign a Kokoro voice to each character in your script.")
            voices_list = _get_voices()

            char_table = gr.Dataframe(
                headers=["name", "voice_id", "gender"],
                value=[[c["name"], c["voice_id"], c["gender"]]
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
        inputs=[text_input, speed_slider],
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
        inputs=[new_name, new_voice, new_gender, char_table],
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
