import os
from pathlib import Path
import pytest
from training.phonemize_dataset import (
    build_manifest_entry,
    phonemize_text,
    reconcile_vocabulary,
    validate_training_row,
)

def test_reconcile_vocabulary_oov_substitution():
    raw_ipa = "German ʏ and ASCII g with — dash"
    reconciled = reconcile_vocabulary(raw_ipa)
    assert "ʏ" not in reconciled
    assert "y" in reconciled
    assert "ɡ" in reconciled
    assert "—" not in reconciled
    assert "-" in reconciled

def test_phonemize_text_misaki():
    text = "Hello world! This is a test."
    phonemes = phonemize_text(text)
    assert isinstance(phonemes, str)
    assert len(phonemes) > 0
    # Reconciled ASCII g should be IPA script g
    assert "g" not in phonemes

def test_build_manifest_entry(tmp_path):
    wav_file = tmp_path / "sample.wav"
    wav_file.touch()
    entry = build_manifest_entry(
        wav_file=wav_file,
        raw_text="Hello world",
        speaker="speaker_01",
        emotion="happy"
    )
    assert entry is not None
    assert "sample.wav|" in entry
    assert entry.endswith("|speaker_01|happy")
    parts = entry.split("|")
    assert len(parts) == 4


def test_validate_training_row_rejects_theatrical_sources_and_missing_labels():
    assert validate_training_row({"source": "ESD", "speaker_id": "s1", "emotion": "sad"}) == "excluded source 'esd'"
    assert validate_training_row({"dataset": "RAVDESS", "speaker_id": "s1", "emotion": "sad"}) == "excluded source 'ravdess'"
    assert validate_training_row({"source": "expresso", "emotion": "sad"}) == "missing speaker_id"
    assert validate_training_row({"source": "expresso", "speaker_id": "s1"}) == "missing emotion"
    assert validate_training_row({"source": "expresso", "speaker_id": "s1", "emotion": "sad"}) is None
