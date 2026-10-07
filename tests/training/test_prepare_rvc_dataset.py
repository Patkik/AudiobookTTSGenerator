import os
import pytest
import numpy as np
import soundfile as sf

from training.prepare_rvc_dataset import (
    classify_projection_type,
    normalize_loudness,
    process_rvc_clip,
    build_rvc_manifest,
)


def test_classify_projection_type():
    assert classify_projection_type("whisper") == "whisper"
    assert classify_projection_type("breathy sigh") == "whisper"
    assert classify_projection_type("shouting at enemy") == "high_excitation"
    assert classify_projection_type("screaming in agony") == "high_excitation"
    assert classify_projection_type("crying softly") == "emotional_inflection"
    assert classify_projection_type("laughing cheerfully") == "emotional_inflection"
    assert classify_projection_type("gasp") == "emotional_inflection"
    assert classify_projection_type("ordinary talking") == "neutral"


def test_normalize_loudness():
    sr = 40_000
    # Generate 1 second of 440 Hz tone
    t = np.linspace(0, 1.0, sr, endpoint=False)
    audio = (0.05 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    norm = normalize_loudness(audio, sr, target_lufs=-23.0)
    assert len(norm) == len(audio)
    assert norm.dtype == np.float32
    assert np.all(np.isfinite(norm))
    # Normalized audio has higher amplitude than tiny 0.05
    assert np.max(np.abs(norm)) > 0.05
    # Should not clip (> 1.0)
    assert np.max(np.abs(norm)) <= 1.0


def test_process_rvc_clip(tmp_path):
    in_sr = 24_000
    out_sr = 40_000
    t = np.linspace(0, 1.5, int(in_sr * 1.5), endpoint=False)
    audio = (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

    in_file = str(tmp_path / "test_input.wav")
    sf.write(in_file, audio, in_sr)

    out_file = str(tmp_path / "processed" / "test_output.wav")
    meta = process_rvc_clip(
        in_path=in_file,
        out_path=out_file,
        tag="screaming",
        target_sr=out_sr,
        target_lufs=-23.0,
    )

    assert meta is not None
    assert os.path.exists(out_file)
    data, sr = sf.read(out_file, dtype="float32")
    assert sr == 40_000
    assert meta["projection"] == "high_excitation"
    assert meta["duration_sec"] > 1.0


def test_build_rvc_manifest(tmp_path):
    manifest_entries = [
        {"path": "wavs/01.wav", "speaker": "rudeus", "projection": "whisper"},
        {"path": "wavs/02.wav", "speaker": "rudeus", "projection": "high_excitation"},
    ]
    manifest_path = str(tmp_path / "rvc_train_manifest.txt")
    build_rvc_manifest(manifest_entries, manifest_path)

    assert os.path.exists(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) == 2
    assert "wavs/01.wav|rudeus|whisper" in lines[0]
    assert "wavs/02.wav|rudeus|high_excitation" in lines[1]
