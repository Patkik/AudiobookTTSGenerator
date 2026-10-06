import os
import numpy as np
import pytest
from training.extract_voicepacks import blend_voice_arrays, save_emotion_voicepack

def test_blend_voice_arrays():
    v1 = np.ones((510, 1, 256), dtype=np.float32)
    v2 = np.zeros((510, 1, 256), dtype=np.float32)
    blended = blend_voice_arrays([(v1, 0.7), (v2, 0.3)])
    assert blended.shape == (510, 1, 256)
    assert np.allclose(blended, 0.7)

def test_save_emotion_voicepack(tmp_path):
    tensor = np.random.randn(510, 1, 256).astype(np.float32)
    out_dir = str(tmp_path / "emotions")
    saved_path = save_emotion_voicepack("happy", tensor, out_dir=out_dir)
    assert os.path.exists(saved_path)
    loaded = np.load(saved_path)
    assert loaded.shape == (510, 1, 256)
    assert np.allclose(loaded, tensor)
