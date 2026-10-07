"""
Synthesizes the enhanced prosodic dialogue exchange between Ruddy and Silfie.
Demonstrates:
1. Punctuation & hesitation cadence breaks (ellipses, dashes, commas).
2. Emotional beats with narrative framing and compound tags.
3. Strict multi-speaker isolation: Ruddy -> am_michael + Rudeus.pth (index_rate=0.35, protect=0.33),
   Silfie -> af_sky (pure female voicepack, zero timbre bleed).
4. Automatic 400ms turn pauses between speaker switches.
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
from app.synthesis.rvc_pipeline import RVCPipeline, RVCConfig


def synthesize_exchange():
    print("=" * 65)
    print("🎙️ Enhanced Prosodic Dialogue Synthesis: Ruddy & Silfie")
    print("=" * 65)

    model_int8 = str(ROOT_DIR / "models" / "kokoro-expressive.int8.onnx")
    model_v1 = str(ROOT_DIR / "models" / "kokoro-v1.0.onnx")
    model_path = model_int8 if os.path.exists(model_int8) else model_v1
    voices_path = str(ROOT_DIR / "models" / "voices-v1.0.bin")
    emotions_dir = str(ROOT_DIR / "emotions")

    # 1. Setup Character Registry with Aliases & Isolation
    registry = CharacterRegistry()
    print("Character Registry Configuration:")
    for c in ["RUDDY", "SILFIE"]:
        voice = registry.get_voice(c)
        gender = registry.get_gender(c)
        rvc = registry.get_rvc_model(c)
        print(f"  • {c}: voice={voice}, gender={gender}, rvc_model={rvc}")

    # 2. Setup Engine with Tuned RVC Settings for Ruddy
    rvc_pipeline = RVCPipeline()
    narrative_resolver = DefaultNarrativeContextResolver()

    engine = SynthesisEngine(
        model_path=model_path,
        voices_path=voices_path,
        emotions_dir=emotions_dir,
        emotion_alpha=0.40,
        enable_mastering=True,
        narrative_context_resolver=narrative_resolver,
        rvc_pipeline=rvc_pipeline,
        enable_rvc=True,
    )

    # Configure Rudeus RVC with User Blueprint Settings:
    # index_rate=0.35, protect=0.33, filter_radius=3, volume_envelope=1.0, rmvpe
    rudeus_cfg = RVCConfig(
        model_path=str(ROOT_DIR / "models" / "rvc" / "rudeus" / "Rudeus.pth"),
        index_path=str(ROOT_DIR / "models" / "rvc" / "rudeus" / "Rudeus.index"),
        index_rate=0.35,         # Tuned: 0.35 preserves Kokoro's dynamic acting
        protect=0.33,            # Tuned: 0.33 protects breath turbulence & consonants
        filter_radius=3,         # Smooths pitch jumps
        volume_envelope=1.0,     # Retains whisper vs shout dynamic projection
        f0_method="rmvpe",       # Robust Multiscale Voice Pitch Estimation
        f0_up_key=0,
    )
    engine.set_rvc_config("rudeus", rudeus_cfg)

    # 3. Prosodically Annotated Script
    script = """
# Beat 1 & 2: Ruddy (Introspective & Gentle)
[whispers, softly] I had a dream once... [sighs] In my dream, there was a child... crying in the dark.
[softening, gentle] But then a gentle light dispersed the shadows... and enveloped me.
[warmly] "Silfie... I'm right here. You don't have to be afraid anymore."

# Beat 3: Silfie (Breathless Relief)
[gasp, tearful relief] "Ruddy... [soft exhale] thank goodness!"

# Beat 4: Ruddy (Firm Resolve & Conviction)
[low voice, rising resolve] "I swear... I will NEVER let anything hurt you ever again!"
"""

    parser = AutomatedScriptParser()
    segments = parser.parse(script)

    print(f"\nParsed {len(segments)} Segments with Prosodic Annotations:")
    for i, s in enumerate(segments):
        if s.is_silence:
            print(f"  {i+1}. [PAUSE/SFX {s.emotion}] for {s.character}")
        else:
            rvc_target = registry.get_rvc_model(s.character) or "None (Clean Kokoro)"
            print(
                f"  {i+1}. [{s.emotion}] {s.character} (rvc: {rvc_target}, "
                f"pause_before={s.profile.pause_before_ms}ms, pause_after={s.profile.pause_after_ms}ms): "
                f"\"{s.text}\""
            )

    # 4. Synthesize Scene
    print("\nSynthesizing enhanced audio...")
    audio, sr = engine.synthesize_segments(segments, registry, enable_mastering=True)

    out_dir = str(ROOT_DIR / "output")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "enhanced_exchange.wav")
    sf.write(out_file, audio, sr)

    duration_sec = len(audio) / sr
    peak_amp = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0

    print(f"\n✓ Generated Enhanced Audio: {out_file}")
    print(f"  Duration: {duration_sec:.2f} seconds")
    print(f"  Sample Rate: {sr} Hz")
    print(f"  Peak Amplitude: {peak_amp:.4f} (clipping limit: 1.0000)")
    assert duration_sec > 5.0, "Audio output too short"
    assert peak_amp <= 1.0, "Audio clipping detected"
    print("\n🌟 Enhanced exchange synthesis verified successfully!")


if __name__ == "__main__":
    synthesize_exchange()
