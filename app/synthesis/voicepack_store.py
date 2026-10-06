"""Loads speaker voicepacks and applies identity-preserving emotion shifts."""
import os
import numpy as np
from typing import Optional


class VoicepackStore:
    """
    Loads voice style tensors from voices-v1.0.bin and optional per-emotion
    .npy files. Emotional shifts are projected away from the target speaker's
    principal identity subspace before being applied.

    Args:
        voices_path: Path to voices-v1.0.bin
        emotions_dir: Directory containing per-emotion .npy files (optional)
    """

    def __init__(self, voices_path: str, emotions_dir: str = "emotions") -> None:
        self._voices = np.load(voices_path)  # NpzFile acting as dict
        self._emotions_dir = emotions_dir
        self._emotion_cache: dict[str, np.ndarray] = {}
        self._identity_basis_cache: dict[str, np.ndarray] = {}
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

        An emotion pack is converted to a delta from the target voice, then
        projected into the null space of the target voice's top identity
        components. The resulting vector is rescaled to the original per-token
        L2 norm, keeping the generated style close to the trained manifold.

        Args:
            voice_id: Kokoro voice name (e.g. "af_heart")
            token_len: Unused (retained for backward compatibility)
            emotion: Emotion name (e.g. "happy") - used only if emotion voicepack exists
            alpha: Emotion blend strength [0.0 = pure voice, 1.0 = pure emotion]

        Returns:
            float32 ndarray of shape (510, 1, 256)
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0.0 and 1.0.")

        base_tensor = self._load_base_tensor(voice_id)

        emotion_tensor = self._load_emotion_tensor(emotion)
        if emotion_tensor is not None and emotion != "neutral" and alpha > 0.0:
            self._validate_matching_tensors(base_tensor, emotion_tensor, emotion)
            basis = self._identity_basis(voice_id, base_tensor)
            return self._apply_emotion_delta(base_tensor, emotion_tensor, basis, alpha)

        return base_tensor

    def available_voices(self) -> list[str]:
        """Return list of available voice IDs from voices-v1.0.bin."""
        return list(self._voices.keys())

    def _load_base_tensor(self, voice_id: str) -> np.ndarray:
        """Load full voice tensor from NpzFile. Falls back to default voice if unknown."""
        try:
            tensor = self._voices[voice_id].astype(np.float32)
        except KeyError:
            if self._fallback_voice:
                tensor = self._voices[self._fallback_voice].astype(np.float32)
            else:
                return np.zeros((510, 1, 256), dtype=np.float32)
        self._validate_style_tensor(tensor, "base voicepack")
        return tensor

    def _load_emotion_tensor(self, emotion: str) -> Optional[np.ndarray]:
        """Load per-emotion voicepack if it exists. Returns None if not found."""
        if emotion in self._emotion_cache:
            return self._emotion_cache[emotion]

        npy_path = os.path.join(self._emotions_dir, f"{emotion}.npy")
        if not os.path.exists(npy_path):
            return None

        tensor = np.load(npy_path).astype(np.float32)
        self._validate_style_tensor(tensor, f"emotion voicepack '{emotion}'")
        self._emotion_cache[emotion] = tensor
        return tensor

    def _identity_basis(self, voice_id: str, base_tensor: np.ndarray) -> np.ndarray:
        """Return cached right-singular identity directions for one speaker."""
        cache = getattr(self, "_identity_basis_cache", None)
        if cache is None:
            cache = {}
            self._identity_basis_cache = cache

        if voice_id not in cache:
            neutral_samples = base_tensor[:, 0, :]
            centered = neutral_samples - neutral_samples.mean(axis=0, keepdims=True)
            if len(neutral_samples) < 2 or not np.any(centered):
                cache[voice_id] = np.empty((base_tensor.shape[-1], 0), dtype=np.float32)
            else:
                _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
                numerical_rank = int(np.count_nonzero(singular_values > 1e-6))
                rank = min(32, numerical_rank)
                cache[voice_id] = right_vectors[:rank].T.astype(np.float32)
        return cache[voice_id]

    @staticmethod
    def _apply_emotion_delta(
        base_tensor: np.ndarray,
        emotion_tensor: np.ndarray,
        identity_basis: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        base = base_tensor[:, 0, :]
        delta = emotion_tensor[:, 0, :] - base
        if identity_basis.shape[1]:
            delta = delta - (delta @ identity_basis) @ identity_basis.T

        candidate = base + alpha * delta
        base_norm = np.linalg.norm(base, axis=1, keepdims=True)
        candidate_norm = np.linalg.norm(candidate, axis=1, keepdims=True)
        safe_norm = np.maximum(candidate_norm, np.finfo(np.float32).eps)
        normalized = candidate * (base_norm / safe_norm)
        return normalized[:, np.newaxis, :].astype(np.float32)

    @staticmethod
    def _validate_style_tensor(tensor: np.ndarray, description: str) -> None:
        if tensor.ndim != 3 or tensor.shape[1] != 1 or tensor.shape[2] != 256:
            raise ValueError(
                f"{description} must have shape (N, 1, 256); received {tensor.shape}."
            )

    @classmethod
    def _validate_matching_tensors(
        cls,
        base_tensor: np.ndarray,
        emotion_tensor: np.ndarray,
        emotion: str,
    ) -> None:
        cls._validate_style_tensor(base_tensor, "base voicepack")
        cls._validate_style_tensor(emotion_tensor, f"emotion voicepack '{emotion}'")
        if base_tensor.shape != emotion_tensor.shape:
            raise ValueError(
                f"Emotion voicepack '{emotion}' shape {emotion_tensor.shape} does not match "
                f"base voicepack shape {base_tensor.shape}."
            )
