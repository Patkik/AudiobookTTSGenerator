import pytest
from app.parser.emotion_profiles import resolve_profile
from app.parser.tag_parser import SpeechSegment
from app.parser.narrative_framing import DefaultNarrativeContextResolver
from app.synthesis.engine import NarrativeContextFrame


def make_segment(text="I can't let you do that!", emotion="neutral", character="RUDEUS"):
    return SpeechSegment(
        text=text,
        emotion=emotion,
        character=character,
        profile=resolve_profile(emotion),
    )


def test_resolve_desperate_shouting():
    resolver = DefaultNarrativeContextResolver()
    seg = make_segment("I can't let you do that!")
    frame = resolver.resolve("he shouted, his voice cracking with desperation", seg)

    assert isinstance(frame, NarrativeContextFrame)
    assert frame.emotion in {"shout", "angry"}
    assert frame.volume_db > 0.0
    assert frame.alpha >= 0.45
    assert frame.pause_before_ms >= 150


def test_resolve_whisper_fear():
    resolver = DefaultNarrativeContextResolver()
    seg = make_segment("Did you hear that?")
    frame = resolver.resolve("she whispered, her voice trembling in fear", seg)

    assert isinstance(frame, NarrativeContextFrame)
    assert frame.emotion in {"whisper", "fearful"}
    assert frame.volume_db < 0.0
    assert frame.speed < 1.0


def test_resolve_hesitation_pause():
    resolver = DefaultNarrativeContextResolver()
    seg = make_segment("I... I don't know.")
    frame = resolver.resolve("he hesitated before answering", seg)

    assert isinstance(frame, NarrativeContextFrame)
    assert frame.pause_before_ms >= 300


def test_resolve_crying_sob():
    resolver = DefaultNarrativeContextResolver()
    seg = make_segment("Please don't go.")
    frame = resolver.resolve("she sobbed in tears", seg)

    assert isinstance(frame, NarrativeContextFrame)
    assert frame.emotion in {"crying", "sad"}
    assert frame.speed < 1.0
    assert frame.pause_before_ms >= 150


def test_resolve_neutral_or_empty_returns_none_or_empty_frame():
    resolver = DefaultNarrativeContextResolver()
    seg = make_segment("Good morning.")
    frame = resolver.resolve("he walked into the room", seg)

    # When no acting cues match, should return None or default frame with no overrides
    assert frame is None or (frame.emotion is None and frame.speed is None)
