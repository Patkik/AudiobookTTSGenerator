import os
import pytest
from training.phonemize_dataset import phonemize_text, build_manifest_entry, split_dataset

def test_phonemize_text_valid_ipa():
    text = "Hello world! This is a test."
    phonemes = phonemize_text(text, lang="en-us")
    assert isinstance(phonemes, str)
    assert len(phonemes) > 0
    # Should contain phoneme characters
    assert any(c in phonemes for c in ["h", "l", "o", "w", "d"])

def test_build_manifest_entry():
    entry = build_manifest_entry(
        wav_path="data/wavs/sample.wav",
        text="Hello world",
        speaker_id="speaker_01",
        emotion="happy",
        lang="en-us"
    )
    assert entry.startswith("data/wavs/sample.wav|")
    assert entry.endswith("|speaker_01|happy")
    parts = entry.split("|")
    assert len(parts) == 4

def test_split_dataset():
    entries = [f"line_{i}" for i in range(100)]
    train, val = split_dataset(entries, val_ratio=0.1, seed=42)
    assert len(train) == 90
    assert len(val) == 10
    # Disjoint sets
    assert set(train).isdisjoint(set(val))
