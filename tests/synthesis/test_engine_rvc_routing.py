import pytest
import numpy as np

from app.characters.registry import CharacterRegistry
from app.parser.emotion_profiles import resolve_profile
from app.parser.tag_parser import SpeechSegment
from app.synthesis.engine import SynthesisEngine
from app.synthesis.rvc_pipeline import RVCConfig, RVCPipeline


class MockKokoro:
    def create(self, **kwargs):
        # 1000 samples of 24kHz tone
        return np.ones(1000, dtype=np.float32) * 0.1, 24_000


class MockStore:
    def get_style(self, **kwargs):
        return np.zeros((510, 1, 256), dtype=np.float32)


def make_test_engine(rvc_pipeline=None, enable_rvc=True):
    engine = SynthesisEngine.__new__(SynthesisEngine)
    engine._kokoro = MockKokoro()
    engine._store = MockStore()
    engine._alpha = 0.35
    engine._enable_mastering = False
    engine._narrative_context_resolver = None
    engine._rvc_pipeline = rvc_pipeline
    engine._enable_rvc = enable_rvc
    engine._rvc_configs = {}
    return engine


def test_rvc_routing_converts_character_with_rvc():
    converted_characters = []

    def mock_runner(audio, cfg):
        converted_characters.append(cfg.model_path)
        return audio * 2.0

    rvc_pipe = RVCPipeline(default_runner=mock_runner)
    engine = make_test_engine(rvc_pipeline=rvc_pipe, enable_rvc=True)
    engine.set_rvc_config(
        "rudeus",
        RVCConfig(model_path="models/rvc/rudeus/Rudeus.pth", volume_envelope=0.0),
    )

    registry = CharacterRegistry()

    # RUDEUS has rvc_model="rudeus"
    seg_rudeus = SpeechSegment(
        text="I can't let you do that!",
        emotion="neutral",
        character="RUDEUS",
        profile=resolve_profile("neutral"),
    )

    audio, sr = engine.synthesize_segments([seg_rudeus], registry)
    assert sr == 24_000
    assert len(converted_characters) == 1
    assert "Rudeus.pth" in converted_characters[0]
    # Audio was doubled by mock_runner
    assert np.isclose(audio[10], 0.2, atol=0.01)


def test_rvc_routing_bypasses_non_rvc_character():
    converted_characters = []

    def mock_runner(audio, cfg):
        converted_characters.append(cfg.model_path)
        return audio * 2.0

    rvc_pipe = RVCPipeline(default_runner=mock_runner)
    engine = make_test_engine(rvc_pipeline=rvc_pipe, enable_rvc=True)
    engine.set_rvc_config("rudeus", RVCConfig(model_path="models/rvc/rudeus/Rudeus.pth"))

    registry = CharacterRegistry()

    # SYLPHIETTE has rvc_model=None
    seg_sylph = SpeechSegment(
        text="I had a dream once.",
        emotion="neutral",
        character="SYLPHIETTE",
        profile=resolve_profile("neutral"),
    )

    audio, sr = engine.synthesize_segments([seg_sylph], registry)
    assert sr == 24_000
    assert len(converted_characters) == 0
    # Original amplitude 0.1
    assert np.isclose(audio[10], 0.1, atol=0.01)


def test_rvc_routing_disabled_bypasses_all():
    converted_characters = []

    def mock_runner(audio, cfg):
        converted_characters.append(cfg.model_path)
        return audio * 2.0

    rvc_pipe = RVCPipeline(default_runner=mock_runner)
    engine = make_test_engine(rvc_pipeline=rvc_pipe, enable_rvc=False)
    engine.set_rvc_config("rudeus", RVCConfig(model_path="models/rvc/rudeus/Rudeus.pth"))

    registry = CharacterRegistry()
    seg_rudeus = SpeechSegment(
        text="Test",
        emotion="neutral",
        character="RUDEUS",
        profile=resolve_profile("neutral"),
    )

    audio, sr = engine.synthesize_segments([seg_rudeus], registry)
    assert len(converted_characters) == 0
    assert np.isclose(audio[10], 0.1, atol=0.01)
