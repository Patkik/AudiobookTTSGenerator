import numpy as np
import pytest
from unittest.mock import MagicMock
from app.synthesis.voicepack_store import VoicepackStore

@pytest.fixture
def mock_voices():
    """Simulates a voices-v1.0.bin NpzFile with two voices."""
    voices = MagicMock()
    voices.__getitem__ = lambda self, key: np.random.randn(511, 1, 256).astype(np.float32)
    voices.keys = lambda self: ["af_heart", "af_bella"]
    return voices

def test_get_style_returns_correct_shape(mock_voices):
    store = VoicepackStore.__new__(VoicepackStore)
    store._voices = mock_voices
    store._emotions_dir = "emotions"
    store._emotion_cache = {}
    store._fallback_voice = "af_heart"
    result = store.get_style("af_heart", token_len=10)
    assert result.shape == (1, 256)
    assert result.dtype == np.float32

def test_get_style_clamps_token_len(mock_voices):
    store = VoicepackStore.__new__(VoicepackStore)
    store._voices = mock_voices
    store._emotions_dir = "emotions"
    store._emotion_cache = {}
    store._fallback_voice = "af_heart"
    result = store.get_style("af_heart", token_len=999)
    assert result.shape == (1, 256)

def test_unknown_voice_falls_back_to_default(mock_voices):
    store = VoicepackStore.__new__(VoicepackStore)
    store._voices = mock_voices
    store._emotions_dir = "emotions"
    store._emotion_cache = {}
    store._fallback_voice = "af_heart"
    result = store.get_style("nonexistent_voice", token_len=5)
    assert result.shape == (1, 256)
