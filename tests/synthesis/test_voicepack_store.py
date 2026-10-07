import numpy as np
import pytest
from unittest.mock import MagicMock
from app.synthesis.voicepack_store import VoicepackStore

@pytest.fixture
def mock_voices():
    """Simulates a voices-v1.0.bin NpzFile with two voices."""
    voices = MagicMock()
    voices.__getitem__ = lambda self, key: np.random.randn(510, 1, 256).astype(np.float32)
    voices.keys = lambda self: ["af_heart", "af_bella"]
    return voices

def test_get_style_returns_correct_shape(mock_voices):
    store = VoicepackStore.__new__(VoicepackStore)
    store._voices = mock_voices
    store._emotions_dir = "emotions"
    store._emotion_cache = {}
    store._fallback_voice = "af_heart"
    result = store.get_style("af_heart")
    assert result.shape == (510, 1, 256)
    assert result.dtype == np.float32

def test_unknown_voice_falls_back_to_default(mock_voices):
    store = VoicepackStore.__new__(VoicepackStore)
    store._voices = mock_voices
    store._emotions_dir = "emotions"
    store._emotion_cache = {}
    store._fallback_voice = "af_heart"
    result = store.get_style("nonexistent_voice")
    assert result.shape == (510, 1, 256)


def test_emotion_shift_preserves_per_token_style_norm(tmp_path):
    rng = np.random.default_rng(4)
    base = rng.normal(size=(12, 1, 256)).astype(np.float32)
    emotion = base + rng.normal(scale=0.5, size=base.shape).astype(np.float32)
    np.save(tmp_path / "happy.npy", emotion)

    store = VoicepackStore.__new__(VoicepackStore)
    store._voices = {"af_heart": base}
    store._emotions_dir = str(tmp_path)
    store._emotion_cache = {}
    store._identity_basis_cache = {}
    store._fallback_voice = "af_heart"

    result = store.get_style("af_heart", emotion="happy", alpha=0.7)
    assert result.dtype == np.float32
    assert np.allclose(
        np.linalg.norm(result[:, 0, :], axis=1),
        np.linalg.norm(base[:, 0, :], axis=1),
        rtol=1e-5,
        atol=1e-5,
    )


def test_mismatched_emotion_voicepack_is_rejected(tmp_path):
    base = np.ones((12, 1, 256), dtype=np.float32)
    np.save(tmp_path / "happy.npy", np.ones((11, 1, 256), dtype=np.float32))

    store = VoicepackStore.__new__(VoicepackStore)
    store._voices = {"af_heart": base}
    store._emotions_dir = str(tmp_path)
    store._emotion_cache = {}
    store._identity_basis_cache = {}
    store._fallback_voice = "af_heart"

    with pytest.raises(ValueError, match="does not match"):
        store.get_style("af_heart", emotion="happy")


def test_invalid_emotion_strength_is_rejected(mock_voices):
    store = VoicepackStore.__new__(VoicepackStore)
    store._voices = mock_voices
    store._emotions_dir = "emotions"
    store._emotion_cache = {}
    store._identity_basis_cache = {}
    store._fallback_voice = "af_heart"

    with pytest.raises(ValueError, match="alpha"):
        store.get_style("af_heart", alpha=1.1)
