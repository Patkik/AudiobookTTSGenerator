"""
Manages loading and combining voice style vectors.

In Phase 1: provides base Kokoro voice vectors indexed by token length.
In Phase 3: blends base voice with per-emotion voicepack vectors.

Voice vectors shape: (511, 1, 256) - index by token_len to get (1, 256).
"""
import os
import numpy as np
from typing import Optional


class VoicepackStore:
    """
    Loads voice style tensors from voices-v1.0.bin and optional
    per-emotion .npy files (produced by Phase 3 training).

    Args:
        voices_path: Path to voices-v1.0.bin
        emotions_dir: Directory containing per-emotion .npy files (optional)
    """

    def __init__(self, voices_path: str, emotions_dir: str = "emotions") -> None:
        self._voices = np.load(voices_path)  # NpzFile acting as dict
        self._emotions_dir = emotions_dir
        self._emotion_cache: dict[str, np.ndarray] = {}
        # Determine fallback voice (first available)
        available = list(self._voices.keys())
        self._fallback_voice = available[0] if available else None

    def get_style(
        self,
        voice_id: str,
        token_len: int,
        emotion: str = "neutral",
        alpha: float = 0.7,
    ) -> np.ndarray:
        """
        Return the style vector for a given voice + token length + emotion.

        In Phase 1 (no emotion voicepacks): returns the base voice vector.
        In Phase 3 (emotion .npy files present): blends emotion * alpha + voice * (1-alpha).

        Args:
            voice_id: Kokoro voice name (e.g. "af_heart")
            token_len: Number of phoneme tokens in the utterance (used for indexing)
            emotion: Emotion name (e.g. "happy") - used only if emotion voicepack exists
            alpha: Emotion blend strength [0.0 = pure voice, 1.0 = pure emotion]

        Returns:
            float32 ndarray of shape (1, 256)
        """
        # Clamp index to valid range
        idx = min(max(token_len, 0), 510)

        # Load base voice vector
        base_vector = self._load_base(voice_id, idx)

        # Try to load emotion voicepack (Phase 3 - optional)
        emotion_vector = self._load_emotion(emotion, idx)

        if emotion_vector is not None and emotion != "neutral" and alpha > 0.0:
            blended = alpha * emotion_vector + (1.0 - alpha) * base_vector
            return blended.astype(np.float32)

        return base_vector

    def available_voices(self) -> list[str]:
        """Return list of available voice IDs from voices-v1.0.bin."""
        return list(self._voices.keys())

    def _load_base(self, voice_id: str, idx: int) -> np.ndarray:
        """Load voice[idx] from NpzFile. Falls back to default voice if unknown."""
        try:
            voice_tensor = self._voices[voice_id]  # shape (511, 1, 256)
        except KeyError:
            if self._fallback_voice:
                voice_tensor = self._voices[self._fallback_voice]
            else:
                return np.zeros((1, 256), dtype=np.float32)
        # voice_tensor shape: (511, 1, 256) -> index -> (1, 256)
        return voice_tensor[idx].astype(np.float32)

    def _load_emotion(self, emotion: str, idx: int) -> Optional[np.ndarray]:
        """Load per-emotion voicepack if it exists. Returns None if not found."""
        if emotion in self._emotion_cache:
            tensor = self._emotion_cache[emotion]
            return tensor[idx] if tensor.ndim == 3 else tensor

        npy_path = os.path.join(self._emotions_dir, f"{emotion}.npy")
        if not os.path.exists(npy_path):
            return None

        tensor = np.load(npy_path).astype(np.float32)
        self._emotion_cache[emotion] = tensor
        return tensor[idx] if tensor.ndim == 3 else tensor
