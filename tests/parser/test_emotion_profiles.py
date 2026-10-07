import pytest
from app.parser.emotion_profiles import (
    EmotionProfile, EMOTION_PROFILES, EMOTION_TOKENS, resolve_profile
)

def test_emotion_profiles_has_required_tags():
    required = [
        "neutral", "happy", "sad", "angry", "fearful", "excited", "calm",
        "nervous", "whisper", "shout", "laugh", "cry", "sigh", "gasp",
        "dramatic", "sarcastic", "monotone", "hesitant", "confident",
        "pause", "long pause",
    ]
    for tag in required:
        assert tag in EMOTION_PROFILES, f"Missing tag: {tag}"

def test_emotion_profile_fields():
    p = EMOTION_PROFILES["happy"]
    assert isinstance(p.speed, float)
    assert isinstance(p.ipa_prefix, (str, type(None)))
    assert isinstance(p.volume_db, float)
    assert isinstance(p.pause_before_ms, int)
    assert isinstance(p.pause_after_ms, int)
    assert isinstance(p.sfx_file, (str, type(None)))
    assert isinstance(p.silence_ms, (int, type(None)))
    assert isinstance(p.emotion_token, int)
    assert isinstance(p.duration_scale, float)
    assert isinstance(p.unvoiced_duration_scale, float)
    assert isinstance(p.vowel_duration_scale, float)
    assert isinstance(p.pre_plosive_pause_ms, int)

def test_emotion_tokens_unique():
    tokens = list(EMOTION_TOKENS.values())
    assert len(tokens) == len(set(tokens)), "Emotion token IDs must be unique"

def test_emotion_tokens_start_at_178():
    assert min(EMOTION_TOKENS.values()) == 178

def test_resolve_profile_exact_match():
    p = resolve_profile("happy")
    assert p.emotion_token == EMOTION_TOKENS["happy"]

def test_resolve_profile_case_insensitive():
    assert resolve_profile("HAPPY").emotion_token == resolve_profile("happy").emotion_token

def test_resolve_profile_alias_whisper():
    # "whispering" should resolve to "whisper"
    assert resolve_profile("whispering").emotion_token == resolve_profile("whisper").emotion_token

def test_resolve_profile_unknown_returns_neutral():
    p = resolve_profile("nonexistenttag")
    assert p.emotion_token == EMOTION_TOKENS["neutral"]

def test_pause_profile_has_silence_ms():
    p = resolve_profile("pause")
    assert p.silence_ms == 500

def test_long_pause_profile_has_silence_ms():
    p = resolve_profile("long pause")
    assert p.silence_ms == 1200


def test_targeted_duration_profiles_match_expression_policy():
    whisper = resolve_profile("whisper")
    sad = resolve_profile("sad")
    fearful = resolve_profile("fearful")
    happy = resolve_profile("happy")
    assert whisper.duration_scale == 1.25
    assert whisper.unvoiced_duration_scale == 1.25
    assert whisper.pause_after_ms == 150
    assert sad.duration_scale == 1.20
    assert sad.vowel_duration_scale == 1.30
    assert fearful.vowel_duration_scale == 0.80
    assert fearful.speed > 1.0
    assert happy.f0_variance_scale == 1.4
    assert fearful.pre_plosive_pause_ms == 40
    assert happy.duration_scale == 0.95
