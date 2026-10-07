import pytest
from app.characters.registry import CharacterRegistry
from app.app import generate_audio, check_models_ok, registry


def test_character_table_structure():
    chars = registry.all_characters()
    rudeus = next(c for c in chars if c["name"] == "RUDEUS")
    assert rudeus["rvc_model"] == "rudeus"
    sylph = next(c for c in chars if c["name"] == "SYLPHIETTE")
    assert sylph.get("rvc_model") is None


def test_generate_audio_empty_text():
    wav, summary = generate_audio("")
    assert wav is None
    assert "No text provided" in summary


def test_generate_audio_with_rvc_parameters():
    # If model files are available, test synthesis with custom RVC parameters
    if check_models_ok():
        script = "RUDEUS: I will protect everyone."
        wav, summary = generate_audio(
            text=script,
            global_speed=1.0,
            intensity=0.35,
            enable_mastering=True,
            enable_rvc=True,
            rvc_index_rate=0.40,
            rvc_protect=0.35,
            rvc_filter_radius=3,
            rvc_volume_envelope=1.0,
            rvc_pitch_shift=0,
        )
        assert wav is not None
        assert "RUDEUS" in summary
