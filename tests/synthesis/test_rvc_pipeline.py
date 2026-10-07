import os
import pytest
import numpy as np

from app.synthesis.rvc_pipeline import RVCConfig, RVCPipeline, resample_audio


def test_rvc_config_defaults():
    cfg = RVCConfig(model_path="models/rvc/rudeus/Rudeus.pth")
    assert cfg.index_rate == 0.40
    assert cfg.protect == 0.35
    assert cfg.filter_radius == 3
    assert cfg.volume_envelope == 1.0
    assert cfg.f0_method == "rmvpe"
    assert cfg.f0_up_key == 0
    assert cfg.target_sample_rate == 24_000


def test_rvc_config_validation():
    # Invalid index_rate
    with pytest.raises(ValueError, match="index_rate"):
        RVCConfig(model_path="dummy.pth", index_rate=-0.1)
    with pytest.raises(ValueError, match="index_rate"):
        RVCConfig(model_path="dummy.pth", index_rate=1.5)

    # Invalid protect
    with pytest.raises(ValueError, match="protect"):
        RVCConfig(model_path="dummy.pth", protect=0.6)

    # Invalid filter_radius
    with pytest.raises(ValueError, match="filter_radius"):
        RVCConfig(model_path="dummy.pth", filter_radius=-1)

    # Invalid volume_envelope
    with pytest.raises(ValueError, match="volume_envelope"):
        RVCConfig(model_path="dummy.pth", volume_envelope=1.2)


def test_audio_resampling():
    sr_in = 24_000
    sr_out = 40_000
    audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, sr_in, endpoint=False)).astype(np.float32)
    
    resampled = resample_audio(audio, sr_in, sr_out)
    assert len(resampled) == sr_out
    assert resampled.dtype == np.float32

    # Round trip
    back = resample_audio(resampled, sr_out, sr_in)
    assert len(back) == sr_in
    assert np.all(np.isfinite(back))


def test_rvc_pipeline_convert_mock():
    pipeline = RVCPipeline()
    cfg = RVCConfig(
        model_path="models/rvc/rudeus/Rudeus.pth",
        index_path="models/rvc/rudeus/Rudeus.index",
        index_rate=0.40,
        protect=0.35,
        filter_radius=3,
        volume_envelope=1.0,
    )
    
    sr = 24_000
    test_audio = (np.ones( sr // 2, dtype=np.float32) * 0.1)

    # Convert with custom runner or mock
    converted, out_sr = pipeline.convert(
        test_audio,
        cfg,
        sample_rate=sr,
        runner_fn=lambda inp, c: inp * 1.05  # Mock timbre transformation
    )

    assert out_sr == 24_000
    assert len(converted) == len(test_audio)
    assert converted.dtype == np.float32
    assert np.all(np.isfinite(converted))


def test_rvc_pipeline_fallback_on_error():
    pipeline = RVCPipeline()
    cfg = RVCConfig(
        model_path="models/rvc/rudeus/Rudeus.pth",
    )
    
    sr = 24_000
    test_audio = np.ones(sr // 4, dtype=np.float32) * 0.2

    def failing_runner(inp, c):
        raise RuntimeError("RVC inference runner crashed")

    # When runner fails, pipeline should gracefully fall back to original audio
    converted, out_sr = pipeline.convert(
        test_audio,
        cfg,
        sample_rate=sr,
        runner_fn=failing_runner,
    )

    assert out_sr == 24_000
    assert np.array_equal(converted, test_audio)
