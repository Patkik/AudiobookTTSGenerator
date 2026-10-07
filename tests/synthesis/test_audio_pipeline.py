import numpy as np
import pytest
from app.synthesis.audio_pipeline import AudioPipeline

SAMPLE_RATE = 24000

def test_make_silence_correct_length():
    silence = AudioPipeline.make_silence(ms=500, sample_rate=SAMPLE_RATE)
    expected_samples = int(SAMPLE_RATE * 0.5)
    assert len(silence) == expected_samples
    assert silence.dtype == np.float32
    assert np.all(silence == 0.0)

def test_make_silence_zero_ms():
    silence = AudioPipeline.make_silence(ms=0, sample_rate=SAMPLE_RATE)
    assert len(silence) == 0

def test_apply_volume_increase():
    audio = np.ones(100, dtype=np.float32) * 0.5
    louder = AudioPipeline.apply_volume(audio, volume_db=6.0)
    assert np.allclose(louder, audio * (10 ** (6.0 / 20.0)), atol=1e-5)

def test_apply_volume_decrease():
    audio = np.ones(100, dtype=np.float32) * 0.5
    quieter = AudioPipeline.apply_volume(audio, volume_db=-6.0)
    assert np.allclose(quieter, audio * (10 ** (-6.0 / 20.0)), atol=1e-5)

def test_apply_volume_zero_db_unchanged():
    audio = np.ones(100, dtype=np.float32) * 0.5
    result = AudioPipeline.apply_volume(audio, volume_db=0.0)
    assert np.allclose(result, audio)

def test_crossfade_output_length():
    a = np.ones(2400, dtype=np.float32)
    b = np.ones(2400, dtype=np.float32)
    result = AudioPipeline.crossfade(a, b, fade_ms=10, sample_rate=SAMPLE_RATE)
    fade_samples = int(SAMPLE_RATE * 0.01)
    assert len(result) == len(a) + len(b) - fade_samples

def test_concat_segments_with_pauses():
    chunks = [
        (np.ones(240, dtype=np.float32), 50),
        (np.ones(240, dtype=np.float32), 100),
    ]
    result = AudioPipeline.concat_with_pauses(chunks, SAMPLE_RATE)
    pause_50 = int(SAMPLE_RATE * 0.05)
    pause_100 = int(SAMPLE_RATE * 0.1)
    expected = 240 + pause_50 + 240 + pause_100
    assert len(result) == expected


def test_contiguous_chunks_use_equal_power_crossfade():
    a = np.ones(2400, dtype=np.float32)
    b = np.ones(2400, dtype=np.float32)
    result = AudioPipeline.concat_with_pauses([(a, 0), (b, 0)], SAMPLE_RATE)
    assert len(result) == len(a) + len(b) - int(SAMPLE_RATE * 0.020)


def test_expression_profiles_preserve_whisper_dynamics():
    whisper = AudioPipeline.dynamics_profile("whisper")
    neutral = AudioPipeline.dynamics_profile("neutral")
    assert whisper.high_pass_hz == 55.0
    assert whisper.compressor_ratio == 1.1
    assert whisper.makeup_gain_db == -4.0
    assert neutral.compressor_ratio == 2.0


def test_expression_processing_is_float32_and_peak_safe():
    audio = np.linspace(-2.0, 2.0, 2400, dtype=np.float32)
    result = AudioPipeline.process_expression(audio, "fearful", SAMPLE_RATE)
    assert result.dtype == np.float32
    assert np.max(np.abs(result)) <= 0.98
