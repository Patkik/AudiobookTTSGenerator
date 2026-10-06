import numpy as np
import pytest
from app.synthesis.audio_pipeline import AudioPipeline

SAMPLE_RATE = 24000


def test_master_audio_empty_and_short():
    """Verify safe handling of empty and extremely short inputs."""
    empty = np.array([], dtype=np.float32)
    assert len(AudioPipeline.master_audio(empty, sample_rate=SAMPLE_RATE)) == 0

    short = np.array([0.1, -0.1, 0.05], dtype=np.float32)
    result = AudioPipeline.master_audio(short, sample_rate=SAMPLE_RATE)
    assert len(result) == 3
    assert np.all(np.isfinite(result))


def test_master_audio_highpass_attenuates_sub_bass():
    """Verify that frequencies below 80 Hz are attenuated."""
    t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)
    # 40 Hz sub-rumble
    rumble = (np.sin(2 * np.pi * 40 * t) * 0.5).astype(np.float32)
    mastered = AudioPipeline.master_audio(
        rumble, sample_rate=SAMPLE_RATE, enable_compression=False, enable_room_tone=False
    )
    # The 40 Hz power should be noticeably reduced by Butterworth highpass
    assert np.std(mastered) < np.std(rumble) * 0.75


def test_master_audio_deesser_attenuates_sibilance():
    """Verify that sibilant frequencies around 6.5 kHz are softened."""
    t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)
    # 6500 Hz sibilant whistle
    sibilant = (np.sin(2 * np.pi * 6500 * t) * 0.5).astype(np.float32)
    mastered = AudioPipeline.master_audio(
        sibilant, sample_rate=SAMPLE_RATE, enable_compression=False, enable_room_tone=False
    )
    assert np.std(mastered) < np.std(sibilant)


def test_master_audio_dynamic_limiting():
    """Verify that peaks are cleanly contained within safe limits [-0.95, 0.95]."""
    t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)
    loud = (np.sin(2 * np.pi * 300 * t) * 2.5).astype(np.float32)
    mastered = AudioPipeline.master_audio(loud, sample_rate=SAMPLE_RATE)
    assert np.max(np.abs(mastered)) <= 0.96
    assert np.all(np.isfinite(mastered))


def test_master_audio_adds_room_tone():
    """Verify subtle acoustic floor presence is added to dead zero silence."""
    silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
    mastered = AudioPipeline.master_audio(
        silence, sample_rate=SAMPLE_RATE, enable_eq=False, enable_compression=False, enable_room_tone=True
    )
    rms_db = 20 * np.log10(np.std(mastered))
    # Room tone should be between -70 dBFS and -50 dBFS
    assert -70.0 < rms_db < -50.0
