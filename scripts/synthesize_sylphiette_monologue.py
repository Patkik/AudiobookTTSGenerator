"""
Synthesizes Sylphiette Greyrat's iconic monologue from Mushoku Tensei.
Demonstrates:
- Expressive female voicepack (af_sky)
- Full emotional beat transitions: introspective whisper -> fearful memory -> tender relief -> sweet joy
- Micro-cadence hesitation breaks (... and dashes)
- Studio mastering DSP
"""
import os
import sys
from pathlib import Path
import soundfile as sf
import numpy as np

# Ensure root dir is in path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.characters.registry import CharacterRegistry
from app.parser.automated_script_parser import AutomatedScriptParser
from app.parser.narrative_framing import DefaultNarrativeContextResolver
from app.synthesis.engine import SynthesisEngine
from app.synthesis.rvc_pipeline import RVCPipeline


SYLPHIETTE_MONOLOGUE_SCRIPT = """# Beat 1: Sylphiette (Introspective & Quiet)
[whispers, softly] I had a dream once... [sighs]
[tender] It was around the time that Rudy went to the King Dragon Realm...

# Beat 2: Sylphiette (Painful Memory & Fear)
[sad, trembling] In my dream, there was a child... crying in the dark.
[fearful] A group of dark shadows surrounded her... They ganged up on the child, and threw pitch-black lumps of something at her.
[hesitant, breathless] The child desperately tried to run away... but the shadows would always follow.

# Beat 3: Sylphiette (Salvation & Tender Relief)
[relieved, gentle] Then... the child ran toward a light.
[softening] As she approached the light, it tossed balls of light at the shadows... and they dispersed.
[whispers, tender] The light then gently enveloped the child... as she drifted off to sleep.

# Beat 4: Sylphiette (Affection & Realization)
[tender, softly] When I first had that dream, I thought it was about the past... A dream about the old days, when the village kids would bully me.
[warmly] But I believed I was dreaming about it after all this time... as a sign of how deeply I loved Rudy.

# Beat 5: Sylphiette (Sweet Joy & Little Girl Innocence)
[happy, cheerful] That was all I thought of... as I lay back down and wriggled happily... like a little girl.
"""


def synthesize_monologue():
    print("=" * 65)
    print("🌸 Synthesizing Sylphiette's Monologue (Mushoku Tensei)")
    print("=" * 65)

    model_int8 = str(ROOT_DIR / "models" / "kokoro-expressive.int8.onnx")
    model_v1 = str(ROOT_DIR / "models" / "kokoro-v1.0.onnx")
    model_path = model_int8 if os.path.exists(model_int8) else model_v1
    voices_path = str(ROOT_DIR / "models" / "voices-v1.0.bin")
    emotions_dir = str(ROOT_DIR / "emotions")

    registry = CharacterRegistry()
    voice = registry.get_voice("SYLPHIETTE")
    gender = registry.get_gender("SYLPHIETTE")
    rvc = registry.get_rvc_model("SYLPHIETTE")
    print(f"Character: SYLPHIETTE | Voice: {voice} | Gender: {gender} | RVC: {rvc} (Pure Kokoro)")

    parser = AutomatedScriptParser()
    segments = parser.parse(SYLPHIETTE_MONOLOGUE_SCRIPT, default_character="SYLPHIETTE")

    print(f"\nParsed {len(segments)} expressive segments:")
    for i, s in enumerate(segments):
        if s.is_silence:
            print(f"  {i+1}. [PAUSE/SFX {s.emotion}]")
        else:
            print(f"  {i+1}. [{s.emotion}] (pause_after={s.profile.pause_after_ms}ms): \"{s.text}\"")

    engine = SynthesisEngine(
        model_path=model_path,
        voices_path=voices_path,
        emotions_dir=emotions_dir,
        emotion_alpha=0.45,
        enable_mastering=True,
        narrative_context_resolver=DefaultNarrativeContextResolver(),
        rvc_pipeline=RVCPipeline(),
        enable_rvc=True,
    )

    print("\nSynthesizing monologue audio with studio mastering...")
    audio, sr = engine.synthesize_segments(segments, registry, enable_mastering=True)

    out_dir = str(ROOT_DIR / "output")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "sylphiette_monologue.wav")
    sf.write(out_file, audio, sr)

    duration_sec = len(audio) / sr
    peak_amp = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0

    print(f"\n✓ Generated Audio: {out_file}")
    print(f"  Duration: {duration_sec:.2f} seconds")
    print(f"  Sample Rate: {sr} Hz")
    print(f"  Peak Amplitude: {peak_amp:.4f} (clipping limit: 1.0000)")
    assert duration_sec > 15.0, "Audio too short"
    assert peak_amp <= 1.0, "Clipping detected"
    print("\n🌟 Sylphiette's monologue synthesized and verified successfully!")


if __name__ == "__main__":
    synthesize_monologue()
