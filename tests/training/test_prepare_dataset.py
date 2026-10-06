import os
from pathlib import Path
import numpy as np
import soundfile as sf
import pytest
from training.prepare_dataset import normalize_audio

SAMPLE_RATE = 24000

def test_normalize_audio_pipeline(tmp_path):
    # Create 2 seconds of 16kHz sine wave with boundary silence
    sr_orig = 16000
    silence = np.zeros(8000, dtype=np.float32)
    tone = np.sin(2 * np.pi * 440 * np.linspace(0, 2, 32000, dtype=np.float32)) * 0.5
    full_audio = np.concatenate([silence, tone, silence])
    
    in_file = tmp_path / "test_raw.wav"
    out_file = tmp_path / "processed.wav"
    sf.write(str(in_file), full_audio, sr_orig)
    
    success = normalize_audio(
        in_file,
        out_file,
        target_sr=SAMPLE_RATE,
        min_sec=0.5,
        max_sec=10.0,
    )
    assert success is True
    assert out_file.exists()
    
    data, sr = sf.read(str(out_file), dtype="float32")
    assert sr == SAMPLE_RATE
    # Silence was trimmed, so duration is bounded
    dur = len(data) / sr
    assert 1.5 <= dur <= 2.5
    # Peak limited to 0.95
    assert np.max(np.abs(data)) <= 0.96

def test_normalize_audio_too_short(tmp_path):
    in_file = tmp_path / "short.wav"
    out_file = tmp_path / "out_short.wav"
    short_audio = np.sin(np.linspace(0, 1, 2400, dtype=np.float32))  # 0.1 sec
    sf.write(str(in_file), short_audio, SAMPLE_RATE)
    
    success = normalize_audio(
        in_file,
        out_file,
        target_sr=SAMPLE_RATE,
        min_sec=1.0,
        max_sec=10.0,
    )
    assert success is False
