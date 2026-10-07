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
from app.voice_changer import FasterWhisperTranscriber, VoiceChanger, create_training_bundle
from app.characters.registry import CharacterRegistry
from app.characters import crud
from app.characters.voices import FALLBACK_VOICES, voice_choices
from app.parser.emotion_profiles import EMOTION_PROFILES

CHARACTERS_FILE = os.environ.get("KOKORO_CHARACTERS_FILE", "characters.json")
registry = CharacterRegistry(CHARACTERS_FILE)
parser = TagParser()
resolver = ScriptResolver()
engine: SynthesisEngine | None = None
transcriber = FasterWhisperTranscriber()

SAMPLE_RATE = 24_000
ALL_TAGS = sorted(EMOTION_PROFILES.keys())
MODEL_PATH = "models/kokoro-v1.0.onnx"
VOICES_PATH = "models/voices-v1.0.bin"
PREVIEW_TEXT = "Hello, this is how I sound when I read your story."

SEG_HEADERS = ["#", "Character", "Emotion", "Text"]


def check_models_ok() -> bool:
    return (
        os.path.exists(MODEL_PATH)
        and os.path.exists(VOICES_PATH)
        and os.path.getsize(MODEL_PATH) > 100_000_000
    )


def _get_engine() -> SynthesisEngine:
    global engine
    if engine is None:
        engine = SynthesisEngine(model_path=MODEL_PATH, voices_path=VOICES_PATH)
    return engine


def _get_voices() -> list[str]:
    if os.path.exists(VOICES_PATH):
        try:
            return list(np.load(VOICES_PATH).keys())
        except Exception:
            pass
    return FALLBACK_VOICES


def _status_html() -> str:
    if check_models_ok():
        return '<span class="kk-badge kk-ok">● Model ready</span>'
    return '<span class="kk-badge kk-bad">● Model files missing</span>'


# ── Editor handlers ──────────────────────────────────────────────────────────
def generate_audio(text: str, global_speed: float):
    """Parse text and synthesize. Returns (wav_path, segment_rows, status)."""
    if not check_models_ok():
        return None, [], "⚠️ Model files not ready. Verify models/kokoro-v1.0.onnx and models/voices-v1.0.bin."
    if not text or not text.strip():
        return None, [], "⚠️ No text provided."
    try:
        segments = parser.parse(text)
        if not segments:
            return None, [], "⚠️ No speech segments found."
        eng = _get_engine()
        eng.speed_scale = float(global_speed or 1.0)
        audio, sr = eng.synthesize_segments(segments, registry)
        if len(audio) == 0:
            return None, [], "⚠️ Synthesis produced no audio."
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        sf.write(tmp.name, audio, sr)
    except Exception as exc:  # surface errors in the UI instead of a bare toast
        return None, [], f"❌ Generation failed: {exc}"

    rows = []
    for i, s in enumerate(segments, 1):
        if s.is_silence:
            rows.append([i, "—", "pause", f"{s.profile.silence_ms} ms"])
        else:
            rows.append([i, s.character, s.emotion, s.text])
    return tmp.name, rows, f"✅ {len(segments)} segments · {len(audio) / sr:.1f}s"


def load_file(file) -> str:
    """Handle uploaded .txt or .epub files."""
    if file is None:
        return gr.update()
    path = file if isinstance(file, str) else file.name
    try:
        with open(path, "rb") as f:
            raw = f.read()
        return resolver.load_text(raw, filename=os.path.basename(path))
    except Exception as exc:
        gr.Warning(f"Could not import file: {exc}")
        return gr.update()


def insert_tag(current_text: str, tag: str):
    """Append a tag to the script and reset the picker."""
    if not tag:
        gr.Warning("Pick a tag first.")
        return current_text, None
    sep = "" if not current_text or current_text.endswith((" ", "\n")) else " "
    return f"{current_text or ''}{sep}[{tag}] ", None


def count_text(text: str) -> str:
    text = text or ""
    return f"{len(text.split())} words · {len(text)} characters"


def export_mp3(wav_path: str):
    """Convert generated WAV to MP3 and reveal the download."""
    if not wav_path or not os.path.exists(wav_path):
        gr.Warning("Generate audio first.")
        return gr.update(value=None, visible=False)
    try:
        mp3_path = os.path.splitext(wav_path)[0] + ".mp3"
        audio, sr = sf.read(wav_path, dtype="float32")
        AudioPipeline.save_mp3(audio, mp3_path, sr)
    except Exception as exc:
        gr.Warning(f"MP3 export failed: {exc}")
        return gr.update(value=None, visible=False)
    return gr.update(value=mp3_path, visible=True)


def toggle_export(wav_path):
    return gr.update(interactive=bool(wav_path))


# ── Voice changer handlers ───────────────────────────────────────────────────
def transcribe_recording(recording_path: str | None):
    """Return an editable transcript for a microphone/upload recording."""
    try:
        return transcriber.transcribe(recording_path), "✅ Transcript ready. Review it before conversion."
    except Exception as exc:
        return "", f"❌ Transcription failed: {exc}"


def convert_recording(
    recording_path: str | None,
    transcript: str,
    voice_id: str,
    emotion: str,
    speed: float,
):
    """Render recorded speech content in a selected built-in Kokoro voice."""
    if not check_models_ok():
        return None, "⚠️ Model files not ready; cannot convert speech."
    try:
        converter = VoiceChanger(_get_engine(), transcriber)
        output = converter.convert(recording_path, transcript, voice_id, emotion, float(speed))
    except Exception as exc:
        return None, f"❌ Conversion failed: {exc}"
    return output, (
        "✅ Converted recording ready. This renders the reviewed transcript in the "
        "selected voice; it does not preserve the source recording's timing or accent."
    )


def bundle_recording_for_training(
    recording_path: str | None,
    transcript: str,
    speaker_id: str,
    emotion: str,
    consent_confirmed: bool,
):
    """Create a download containing only original, consented recording data."""
    try:
        bundle = create_training_bundle(
            recording_path,
            transcript,
            speaker_id,
            emotion,
            consent_confirmed,
        )
    except Exception as exc:
        return gr.update(value=None, visible=False), f"❌ Training bundle failed: {exc}"
    return gr.update(value=bundle, visible=True), (
        "✅ Training bundle created with the original 24 kHz recording and reviewed "
        "metadata. Generated audio is intentionally excluded."
    )


# ── Character handlers ───────────────────────────────────────────────────────
def _notify(ok: bool, msg: str) -> None:
    (gr.Info if ok else gr.Warning)(msg)


def _after_change(ok: bool, msg: str, clear: bool = True):
    _notify(ok, msg)
    rows = crud.table_rows(registry)
    if ok and clear:
        return rows, "", None, "", gr.update(interactive=False), gr.update(interactive=False)
    return rows, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()


def on_add(name, voice):
    return _after_change(*crud.add_character(registry, name, voice))


def on_update(original, name, voice):
    return _after_change(*crud.update_character(registry, original, name, voice))


def on_delete(original):
    return _after_change(*crud.delete_character(registry, original))


def on_reset():
    registry.reset()
    gr.Info("Characters reset to defaults.")
    return crud.table_rows(registry), "", None, "", gr.update(interactive=False), gr.update(interactive=False)


def on_select(rows, evt: gr.SelectData):
    """Load the clicked table row into the form."""
    try:
        row = rows.values.tolist()[evt.index[0]] if hasattr(rows, "values") else rows[evt.index[0]]
    except (IndexError, TypeError):
        return gr.update(), gr.update(), "", gr.update(), gr.update()
    name, voice = str(row[0]), str(row[1])
    locked = name == "NARRATOR"
    return name, voice, name, gr.update(interactive=True), gr.update(interactive=not locked)


def on_preview(voice):
    if not voice:
        gr.Warning("Choose a voice first.")
        return None
    if not check_models_ok():
        gr.Warning("Model files not ready; cannot preview.")
        return None
    try:
        from app.characters.registry import CharacterRegistry as _Reg
        tmp_reg = _Reg()
        tmp_reg.add("NARRATOR", voice)
        segments = parser.parse(PREVIEW_TEXT)
        eng = _get_engine()
        eng.speed_scale = 1.0
        audio, sr = eng.synthesize_segments(segments, tmp_reg)
        out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        out.close()
        sf.write(out.name, audio, sr)
        return out.name
    except Exception as exc:
        gr.Warning(f"Preview failed: {exc}")
        return None


# ── Tag reference ────────────────────────────────────────────────────────────
TAG_HEADERS = ["Tag", "Speed", "Volume", "Pause After", "SFX", "Silence Only"]
TAG_ROWS = [
    [tag, f"{p.speed:.2f}×", f"{p.volume_db:+.0f}dB", f"{p.pause_after_ms}ms",
     p.sfx_file or "—", "Yes" if p.silence_ms else "No"]
    for tag, p in sorted(EMOTION_PROFILES.items())
]


def filter_tags(query: str):
    q = (query or "").strip().lower()
    return [r for r in TAG_ROWS if q in r[0].lower()] if q else TAG_ROWS


THEME = gr.themes.Base(
    primary_hue="violet",
    secondary_hue="indigo",
    neutral_hue="slate",
    radius_size="lg",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)

CSS = """
.kk-header { display:flex; align-items:center; justify-content:space-between; padding: 4px 0 8px; }
.kk-title { font-size: 1.6rem; font-weight: 700; margin: 0; }
.kk-sub { opacity: .7; margin: 0; font-size: .9rem; }
.kk-badge { padding: 4px 12px; border-radius: 999px; font-size: .8rem; font-weight: 600; }
.kk-ok { background: rgba(34,197,94,.15); color: #22c55e; }
.kk-bad { background: rgba(239,68,68,.15); color: #ef4444; }
.kk-card { border: 1px solid var(--border-color-primary); border-radius: 14px; padding: 12px; }
#generate-btn { min-height: 52px; font-size: 1.05rem; font-weight: 600; }
.kk-count { opacity: .6; font-size: .8rem; }
"""

with gr.Blocks(title="Kokoro Expressive Audiobook TTS") as demo:
    gr.HTML(
        f'<div class="kk-header"><div><h1 class="kk-title">🎙️ Kokoro Expressive Audiobook TTS</h1>'
        f'<p class="kk-sub">Tag-driven emotional narration with per-character voices</p></div>'
        f'<div>{_status_html()}</div></div>'
    )

    with gr.Tabs():
        # ── TAB 1: EDITOR ────────────────────────────────────────────────────
        with gr.Tab("📝 Editor"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    text_input = gr.Textbox(
                        label="Script (with expression tags)",
                        placeholder=(
                            "NARRATOR: It was a dark and stormy night.\n"
                            "[sad] She whispered, \"I can't do this anymore.\"\n"
                            "[gasp] A shadow appeared at the window.\n"
                            "ALICE (angry): Get out of my house!"
                        ),
                        lines=14,
                        max_lines=40,
                    )
                    word_count = gr.Markdown(count_text(""), elem_classes="kk-count")
                    with gr.Row():
                        upload_btn = gr.UploadButton(
                            "📂 Import .txt / .epub",
                            file_types=[".txt", ".epub"],
                        )
                        clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                    with gr.Row(elem_classes="kk-card"):
                        tag_picker = gr.Dropdown(
                            choices=ALL_TAGS, label="🎭 Tag", value=None,
                            interactive=True, scale=3,
                        )
                        insert_btn = gr.Button("➕ Insert tag", scale=1)

                with gr.Column(scale=2):
                    audio_output = gr.Audio(
                        label="Generated Audio", type="filepath", interactive=False,
                    )
                    status_md = gr.Markdown("Ready.")
                    segment_table = gr.Dataframe(
                        headers=SEG_HEADERS,
                        label="Segment Breakdown",
                        interactive=False,
                        wrap=True,
                        max_height=280,
                    )
                    export_btn = gr.Button("⬇️ Export MP3", variant="secondary", interactive=False)
                    mp3_output = gr.File(label="MP3 Download", visible=False)

            with gr.Row():
                generate_btn = gr.Button(
                    "▶ Generate Audio", variant="primary", scale=3, elem_id="generate-btn",
                )
                speed_slider = gr.Slider(
                    minimum=0.5, maximum=2.0, value=1.0, step=0.05,
                    label="Global Speed", scale=1,
                )

        # ── TAB 2: VOICE CHANGER ─────────────────────────────────────────────
        with gr.Tab("🎙️ Voice Changer"):
            gr.Markdown(
                "Record or upload speech, review its transcript, then render it in a built-in "
                "Kokoro voice. This is offline speech-to-speech rendering, not real-time "
                "waveform-preserving voice conversion."
            )
            changer_voices = voice_choices(_get_voices())
            changer_default_voice = changer_voices[0][1] if changer_voices else "af_heart"
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    recording_input = gr.Audio(
                        label="Record or upload speech (max 2 minutes)",
                        sources=["microphone", "upload"],
                        type="filepath",
                    )
                    transcribe_btn = gr.Button("📝 Transcribe recording")
                    transcript_input = gr.Textbox(
                        label="Reviewed transcript",
                        placeholder="Transcribe a recording, then correct the text here before conversion.",
                        lines=5,
                    )
                    with gr.Row():
                        changer_voice = gr.Dropdown(
                            choices=changer_voices,
                            value=changer_default_voice,
                            label="Target voice",
                        )
                        changer_emotion = gr.Dropdown(
                            choices=ALL_TAGS,
                            value="neutral",
                            label="Expression",
                        )
                    changer_speed = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.05,
                        label="Output speed",
                    )
                    convert_btn = gr.Button("🔄 Convert to selected voice", variant="primary")
                with gr.Column(scale=1):
                    converted_audio = gr.Audio(
                        label="Converted preview",
                        type="filepath",
                        interactive=False,
                    )
                    changer_status = gr.Markdown("Record speech, transcribe it, and review the text.")
                    gr.Markdown(
                        "### Training data bundle\n"
                        "Exports the original recording—not synthetic converted audio—plus metadata "
                        "for the curation pipeline."
                    )
                    training_speaker = gr.Textbox(label="Speaker ID", placeholder="e.g. speaker_01")
                    training_consent = gr.Checkbox(
                        label="I have permission to use this original recording for model training.",
                        value=False,
                    )
                    bundle_btn = gr.Button("📦 Bundle original recording for training")
                    bundle_download = gr.File(label="Training bundle download", visible=False)

        # ── TAB 3: CHARACTERS ────────────────────────────────────────────────
        with gr.Tab("👥 Characters"):
            gr.Markdown(
                "Assign a Kokoro voice to each character. Click a row to edit or delete it. "
                "Gender and accent come from the voice pack."
            )
            voices_list = _get_voices()
            choices = voice_choices(voices_list)
            default_voice = choices[0][1] if choices else "af_heart"

            char_table = gr.Dataframe(
                headers=crud.HEADERS,
                value=crud.table_rows(registry),
                label="Registered Characters",
                interactive=False,
                type="array",
            )
            selected_name = gr.State("")

            with gr.Row(elem_classes="kk-card"):
                new_name = gr.Textbox(label="Character name (e.g. ALICE)", scale=2)
                new_voice = gr.Dropdown(
                    choices=choices, value=default_voice, label="Voice", scale=3,
                )
            with gr.Row():
                add_char_btn = gr.Button("➕ Add", variant="primary")
                update_char_btn = gr.Button("💾 Update selected", interactive=False)
                delete_char_btn = gr.Button("🗑️ Delete selected", variant="stop", interactive=False)
                preview_btn = gr.Button("🔊 Preview voice")
                reset_btn = gr.Button("↺ Reset defaults")
            preview_audio = gr.Audio(label="Voice preview", type="filepath", interactive=False)

        # ── TAB 4: TAG REFERENCE ─────────────────────────────────────────────
        with gr.Tab("📖 Tag Reference"):
            tag_search = gr.Textbox(label="Filter tags", placeholder="e.g. whisper", max_lines=1)
            tag_table = gr.Dataframe(
                headers=TAG_HEADERS, value=TAG_ROWS, label="All Expression Tags",
                interactive=False, max_height=420,
            )
            gr.Markdown("""
**Inline format:** `[happy] She smiled.`  
**Compound:** `[whispering, nervous] "Did you hear that?"`  
**Screenplay:** `ALICE (angry): I won't stand for this!`  
**Pauses:** `[pause]` (500ms)  `[long pause]` (1200ms)  
""")

    # ── Event wiring ─────────────────────────────────────────────────────────
    gen_event = generate_btn.click(
        fn=generate_audio,
        inputs=[text_input, speed_slider],
        outputs=[audio_output, segment_table, status_md],
        concurrency_limit=1,
    )
    gen_event.then(fn=toggle_export, inputs=audio_output, outputs=export_btn)
    gen_event.then(lambda: gr.update(value=None, visible=False), outputs=mp3_output)
    upload_btn.upload(fn=load_file, inputs=upload_btn, outputs=text_input)
    clear_btn.click(lambda: "", outputs=text_input)
    text_input.change(count_text, inputs=text_input, outputs=word_count)
    insert_btn.click(insert_tag, inputs=[text_input, tag_picker], outputs=[text_input, tag_picker])
    export_btn.click(fn=export_mp3, inputs=audio_output, outputs=mp3_output)

    transcribe_btn.click(
        transcribe_recording,
        inputs=recording_input,
        outputs=[transcript_input, changer_status],
        concurrency_limit=1,
    )
    convert_btn.click(
        convert_recording,
        inputs=[recording_input, transcript_input, changer_voice, changer_emotion, changer_speed],
        outputs=[converted_audio, changer_status],
        concurrency_limit=1,
    )
    bundle_btn.click(
        bundle_recording_for_training,
        inputs=[recording_input, transcript_input, training_speaker, changer_emotion, training_consent],
        outputs=[bundle_download, changer_status],
        concurrency_limit=1,
    )

    form_outputs = [char_table, new_name, new_voice, selected_name, update_char_btn, delete_char_btn]
    add_char_btn.click(on_add, inputs=[new_name, new_voice], outputs=form_outputs)
    update_char_btn.click(on_update, inputs=[selected_name, new_name, new_voice], outputs=form_outputs)
    delete_char_btn.click(on_delete, inputs=selected_name, outputs=form_outputs)
    reset_btn.click(on_reset, outputs=form_outputs)
    char_table.select(
        on_select, inputs=char_table,
        outputs=[new_name, new_voice, selected_name, update_char_btn, delete_char_btn],
    )
    preview_btn.click(on_preview, inputs=new_voice, outputs=preview_audio)
    tag_search.change(filter_tags, inputs=tag_search, outputs=tag_table)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=THEME, css=CSS)
