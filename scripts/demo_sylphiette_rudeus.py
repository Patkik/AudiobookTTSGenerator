"""
Full system verification script:
Synthesizes a dramatic dialogue scene between Sylphiette Greyrat and Rudeus Greyrat
utilizing Kokoro expressive acting, narrative context framing, and RVC character routing.
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
from app.parser.tag_parser import TagParser
from app.parser.narrative_framing import DefaultNarrativeContextResolver
from app.synthesis.engine import SynthesisEngine
from app.synthesis.rvc_pipeline import RVCPipeline, RVCConfig, get_default_rudeus_config


def run_demo():
    print("=" * 60)
    print("🎬 Kokoro TTS + RVC Expressive Dialogue Verification")
    print("=" * 60)

    model_int8 = str(ROOT_DIR / "models" / "kokoro-expressive.int8.onnx")
    model_v1 = str(ROOT_DIR / "models" / "kokoro-v1.0.onnx")
    model_path = model_int8 if os.path.exists(model_int8) else model_v1
    voices_path = str(ROOT_DIR / "models" / "voices-v1.0.bin")
    emotions_dir = str(ROOT_DIR / "emotions")

    print(f"Using Model: {os.path.basename(model_path)}")
    print(f"Using Voices: {os.path.basename(voices_path)}")

    # 1. Setup Character Registry
    registry = CharacterRegistry()
    registry.add("SYLPHIETTE", "af_sky", gender="female", rvc_model=None)
    registry.add("RUDEUS", "am_michael", gender="male", rvc_model="rudeus")

    print("Registered Characters:")
    for c in registry.all_characters():
        print(f"  • {c['name']}: voice={c['voice_id']} (gender={c['gender']}, rvc={c.get('rvc_model')})")

    # 2. Setup RVC Pipeline & Engine
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

    # Apply Blueprint hyperparameters to Rudeus
    rudeus_cfg = RVCConfig(
        model_path=str(ROOT_DIR / "models" / "rvc" / "rudeus" / "Rudeus.pth"),
        index_path=str(ROOT_DIR / "models" / "rvc" / "rudeus" / "Rudeus.index"),
        index_rate=0.40,         # Blueprint: 0.30 - 0.50
        protect=0.35,            # Blueprint: 0.33 - 0.50
        filter_radius=3,         # Blueprint: 3
        volume_envelope=1.0,     # Blueprint: 1.0
        f0_method="rmvpe",       # Blueprint: RMVPE
        f0_up_key=0,
    )
    engine.set_rvc_config("rudeus", rudeus_cfg)

    # 3. Dramatic Dialogue Script
    script = """
SYLPHIETTE (whisper, sad): I had a dream once... In my dream, there was a child crying in the dark.
SYLPHIETTE (relieved, tender): But then a gentle light dispersed the shadows, and enveloped me.
RUDEUS (tender, gently): Sylphie, I'm right here. You don't have to be afraid anymore.
SYLPHIETTE (crying, emotional): Rudy... thank goodness.
RUDEUS (desperate, shouting): I swear I won't let anything hurt you ever again!
"""

    parser = TagParser()
    segments = parser.parse(script)
    print(f"\nParsed {len(segments)} dialogue segments:")
    for i, s in enumerate(segments):
        print(f"  {i+1}. [{s.emotion}] {s.character}: \"{s.text}\" (pause_after={s.profile.pause_after_ms}ms)")

    # 4. Synthesize Scene
    print("\nSynthesizing dialogue audio...")
    audio, sr = engine.synthesize_segments(segments, registry, enable_mastering=True)

    out_dir = str(ROOT_DIR / "output")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "sylphiette_rudeus_scene.wav")
    sf.write(out_file, audio, sr)

    duration_sec = len(audio) / sr
    peak_amp = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0

    print(f"\n✓ Generated audio: {out_file}")
    print(f"  Duration: {duration_sec:.2f} seconds")
    print(f"  Sample Rate: {sr} Hz")
    print(f"  Peak Amplitude: {peak_amp:.4f} (clipping limit: 1.0000)")
    assert duration_sec > 3.0, "Audio too short"
    assert peak_amp <= 1.0, "Audio clipping detected"
    print("\n🌟 All verification checks passed!")


if __name__ == "__main__":
    run_demo()
