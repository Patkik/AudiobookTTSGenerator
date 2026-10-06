"""
Manages loading and combining voice style vectors.

In Phase 1: provides base Kokoro voice vectors.
In Phase 3: blends base voice with per-emotion voicepack vectors.

Kokoro-onnx expects the full voice tensor: shape (510, 1, 256) or (511, 1, 256),
because inside create(), it indexes voice[len(tokens)].
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
        available = list(self._voices.keys())
        self._fallback_voice = available[0] if available else None

    def get_style(
        self,
        voice_id: str,
        token_len: int = 0,
        emotion: str = "neutral",
        alpha: float = 0.7,
    ) -> np.ndarray:
        """
        Return the 3D style tensor of shape (N, 1, 256) expected by kokoro-onnx.

        In Phase 1 (no emotion voicepacks): returns the base voice tensor.
        In Phase 3 (emotion .npy files present): blends emotion * alpha + voice * (1-alpha).

        Args:
            voice_id: Kokoro voice name (e.g. "af_heart")
            token_len: Unused (retained for backward compatibility)
            emotion: Emotion name (e.g. "happy") - used only if emotion voicepack exists
            alpha: Emotion blend strength [0.0 = pure voice, 1.0 = pure emotion]

        Returns:
            float32 ndarray of shape (510, 1, 256)
        """
        base_tensor = self._load_base_tensor(voice_id)

        emotion_tensor = self._load_emotion_tensor(emotion)
        if emotion_tensor is not None and emotion != "neutral" and alpha > 0.0:
            blended = alpha * emotion_tensor + (1.0 - alpha) * base_tensor
            return blended.astype(np.float32)

        return base_tensor

    def available_voices(self) -> list[str]:
        """Return list of available voice IDs from voices-v1.0.bin."""
        return list(self._voices.keys())

    def _load_base_tensor(self, voice_id: str) -> np.ndarray:
        """Load full voice tensor from NpzFile. Falls back to default voice if unknown."""
        try:
            return self._voices[voice_id].astype(np.float32)
        except KeyError:
            if self._fallback_voice:
                return self._voices[self._fallback_voice].astype(np.float32)
            return np.zeros((510, 1, 256), dtype=np.float32)

    def _load_emotion_tensor(self, emotion: str) -> Optional[np.ndarray]:
        """Load per-emotion voicepack if it exists. Returns None if not found."""
        if emotion in self._emotion_cache:
            return self._emotion_cache[emotion]

        npy_path = os.path.join(self._emotions_dir, f"{emotion}.npy")
        if not os.path.exists(npy_path):
            return None

        tensor = np.load(npy_path).astype(np.float32)
        self._emotion_cache[emotion] = tensor
        return tensor
