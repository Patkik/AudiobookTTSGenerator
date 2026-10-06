import numpy as np
import pytest

from app.characters.registry import CharacterRegistry
from app.parser.emotion_profiles import resolve_profile
from app.parser.tag_parser import SpeechSegment
from app.synthesis.engine import NarrativeContextFrame, SynthesisEngine


class FakeKokoro:
    def __init__(self):
        self.call = None

    def create(self, **kwargs):
        self.call = kwargs
        return np.ones(8, dtype=np.float32), 24_000


class FakeStore:
    def __init__(self):
        self.call = None

    def get_style(self, **kwargs):
        self.call = kwargs
        return np.zeros((510, 1, 256), dtype=np.float32)


def make_engine(context_resolver=None):
    engine = SynthesisEngine.__new__(SynthesisEngine)
    engine._kokoro = FakeKokoro()
    engine._store = FakeStore()
    engine._alpha = 0.35
    engine._enable_mastering = False
    engine._narrative_context_resolver = context_resolver
    return engine


def make_segment():
    return SpeechSegment(
        text="We have to leave right now.",
        emotion="neutral",
        character="NARRATOR",
        profile=resolve_profile("neutral"),
    )


def test_context_frame_steers_style_without_speaking_context():
    seen = []

    def resolve_context(context, segment):
        seen.append((context, segment.text))
        return NarrativeContextFrame(
            emotion="whisper",
            speed=0.9,
            alpha=0.5,
            volume_db=0.0,
            pause_before_ms=20,
            pause_after_ms=30,
        )

    engine = make_engine(resolve_context)
    audio, sample_rate = next(engine.synthesize_streaming(
        [make_segment()],
        CharacterRegistry(),
        narrative_context="he whispered urgently, looking over his shoulder",
    ))

    assert seen == [("he whispered urgently, looking over his shoulder", "We have to leave right now.")]
    assert engine._kokoro.call["text"] == "We have to leave right now."
    assert engine._kokoro.call["speed"] == 0.9
    assert engine._store.call["emotion"] == "whisper"
    assert engine._store.call["alpha"] == 0.5
    assert audio.dtype == np.float32
    assert sample_rate == 24_000


def test_narrative_emotion_is_used_for_voicepack_offset():
    engine = make_engine(lambda _context, _segment: NarrativeContextFrame(emotion="whisper"))
    segment = engine._apply_narrative_context(make_segment(), "urgent whisper")

    assert segment.emotion == "whisper"
    assert segment.profile.alpha == resolve_profile("whisper").alpha
    assert segment.profile.speed == resolve_profile("whisper").speed


def test_context_requires_an_explicit_resolver():
    engine = make_engine()

    with pytest.raises(ValueError, match="requires a narrative_context_resolver"):
        engine._apply_narrative_context(make_segment(), "a tense scene")


def test_context_rejects_unknown_emotion():
    engine = make_engine(
        lambda _context, _segment: NarrativeContextFrame(emotion="made-up-emotion")
    )

    with pytest.raises(ValueError, match="Unsupported contextual emotion"):
        engine._apply_narrative_context(make_segment(), "a tense scene")


def test_context_rejects_out_of_range_alpha():
    engine = make_engine(lambda _context, _segment: NarrativeContextFrame(alpha=1.1))

    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        engine._apply_narrative_context(make_segment(), "a tense scene")
