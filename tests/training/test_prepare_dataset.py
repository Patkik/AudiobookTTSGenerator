import os
import numpy as np
import soundfile as sf
import pytest
from training.prepare_dataset import process_audio_file, normalize_audio

SAMPLE_RATE = 24000

def test_normalize_audio_rms():
    # Test audio scaling
    audio = np.ones(2400, dtype=np.float32) * 0.1
    normalized = normalize_audio(audio)
    assert np.max(np.abs(normalized)) <= 1.0
    assert np.max(np.abs(normalized)) > 0.0

def test_process_audio_file_resampling_and_trim(tmp_path):
    # Generate 2 seconds of 16kHz sine wave with silence on both sides
    sr_orig = 16000
    silence = np.zeros(8000, dtype=np.float32)
    tone = np.sin(2 * np.pi * 440 * np.linspace(0, 2, 32000, dtype=np.float32)) * 0.5
    full_audio = np.concatenate([silence, tone, silence])
    
    in_file = tmp_path / "test_raw.wav"
    out_file = tmp_path / "processed.wav"
    sf.write(str(in_file), full_audio, sr_orig)
    
    success, info = process_audio_file(
        str(in_file),
        str(out_file),
        target_sr=SAMPLE_RATE,
        min_sec=0.5,
        max_sec=10.0,
    )
    assert success is True
    assert out_file.exists()
    
    data, sr = sf.read(str(out_file), dtype="float32")
    assert sr == SAMPLE_RATE
    # Silence was trimmed, so duration should be around 2s, not 3s
    assert 1.5 <= len(data) / sr <= 2.5

def test_process_audio_file_too_short(tmp_path):
    in_file = tmp_path / "short.wav"
    out_file = tmp_path / "out_short.wav"
    short_audio = np.sin(np.linspace(0, 1, 2400, dtype=np.float32))  # 0.1 sec
    sf.write(str(in_file), short_audio, SAMPLE_RATE)
    
    success, info = process_audio_file(
        str(in_file),
        str(out_file),
        target_sr=SAMPLE_RATE,
        min_sec=1.0,
        max_sec=10.0,
    )
    assert success is False
    assert "too short" in info.lower()
