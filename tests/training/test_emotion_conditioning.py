import pytest

from training.emotion_conditioning import (
    conditioning_record,
    duration_targets_for,
    parse_conditioning_record,
)


def test_duration_targets_match_runtime_expression_contract():
    whisper = duration_targets_for("whisper")
    sad = duration_targets_for("sad")
    fearful = duration_targets_for("fearful")
    happy = duration_targets_for("happy")
    assert (whisper.total_scale, whisper.unvoiced_scale, whisper.trailing_pause_ms) == (1.15, 1.25, 150)
    assert (sad.total_scale, sad.vowel_scale) == (1.20, 1.30)
    assert (fearful.vowel_scale, fearful.pre_plosive_pause_ms) == (0.85, 40)
    assert happy.total_scale == 0.95


def test_conditioning_record_preserves_manifest_compatible_metadata():
    record = conditioning_record("clip.wav", "speaker_a", "sad", "clip.TextGrid")
    assert record["speaker_id"] == "speaker_a"
    assert record["emotion"] == "sad"
    assert record["alignment_path"] == "clip.TextGrid"
    assert parse_conditioning_record(record).vowel_scale == 1.30


def test_invalid_duration_metadata_is_rejected():
    with pytest.raises(ValueError, match="greater than zero"):
        parse_conditioning_record({"duration_targets": {"total_scale": 0}})
