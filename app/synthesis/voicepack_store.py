"""Loads speaker voicepacks and applies identity-preserving emotion shifts."""
import os
import numpy as np
from typing import Optional


class VoicepackStore:
    """
    Loads voice style tensors from voices-v1.0.bin and optional per-emotion
    .npy files.

    Identity-Preserving Split Embedding:
    - Dimensions 0-127 (speaker timbre, vocal tract, gender) are 100% PRESERVED
      from the base voice so the character's vocal identity is never corrupted.
    - Dimensions 128-255 (prosody, pitch contour, rhythm) receive the emotional shift,
      rescaled to preserve per-token style norm.

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
        alpha: float = 0.35,
        gender: Optional[str] = None,
    ) -> np.ndarray:
        """
        Return the 3D style tensor of shape (N, 1, 256) expected by kokoro-onnx.

        Args:
            voice_id: Kokoro voice name (e.g. "af_heart")
            token_len: Unused (retained for backward compatibility)
            emotion: Emotion name (e.g. "happy")
            alpha: Emotion blend strength [0.0 = pure voice, 1.0 = full emotional delta]
            gender: Optional gender hint ("male" or "female") for delta routing

        Returns:
            float32 ndarray of shape (510, 1, 256)
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0.0 and 1.0.")

        base_tensor = self._load_base_tensor(voice_id)
        if emotion == "neutral" or alpha <= 0.0:
            return base_tensor

        emotion_tensor = self._load_emotion_tensor(emotion, gender=gender)
        if emotion_tensor is None:
            return base_tensor

        self._validate_matching_tensors(base_tensor, emotion_tensor, emotion)

        # Extract split halves: 0..127 for timbre/identity, 128..255 for prosody
        base_timbre = base_tensor[:, :, :128]
        base_prosody = base_tensor[:, :, 128:]

        # Handle 128-dim pure prosody deltas or 256-dim full tensors
        if emotion_tensor.shape[2] == 128:
            candidate_prosody = base_prosody + (alpha * emotion_tensor)
        else:
            emotion_prosody = emotion_tensor[:, :, 128:]
            candidate_prosody = (1.0 - alpha) * base_prosody + (alpha * emotion_prosody)

        # Rescale candidate_prosody so that its L2 norm matches base_prosody per token
        base_p_norm = np.linalg.norm(base_prosody[:, 0, :], axis=1, keepdims=True)
        cand_p_norm = np.linalg.norm(candidate_prosody[:, 0, :], axis=1, keepdims=True)
        safe_p_norm = np.maximum(cand_p_norm, np.finfo(np.float32).eps)
        scaled_prosody = candidate_prosody * (base_p_norm[:, np.newaxis, :] / safe_p_norm[:, np.newaxis, :])

        # Recombine: Timbre is 100% untouched base voice!
        blended = np.concatenate([base_timbre, scaled_prosody], axis=2)
        return blended.astype(np.float32)

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

    def _load_emotion_tensor(self, emotion: str, gender: Optional[str] = None) -> Optional[np.ndarray]:
        """Load per-emotion voicepack/delta if it exists. Checks gender variants first."""
        cache_key = f"{gender}_{emotion}" if gender else emotion
        if cache_key in self._emotion_cache:
            return self._emotion_cache[cache_key]

        candidate_paths = []
        if gender:
            candidate_paths.append(os.path.join(self._emotions_dir, "deltas", f"{gender.lower()}_{emotion}.npy"))
            candidate_paths.append(os.path.join(self._emotions_dir, f"{gender.lower()}_{emotion}.npy"))
        candidate_paths.append(os.path.join(self._emotions_dir, "deltas", f"{emotion}.npy"))
        candidate_paths.append(os.path.join(self._emotions_dir, f"{emotion}.npy"))

        for p in candidate_paths:
            if os.path.exists(p):
                tensor = np.load(p).astype(np.float32)
                self._emotion_cache[cache_key] = tensor
                return tensor

        return None

    @staticmethod
    def _validate_style_tensor(tensor: np.ndarray, description: str) -> None:
        if tensor.ndim != 3 or tensor.shape[1] != 1 or tensor.shape[2] not in (128, 256):
            raise ValueError(
                f"{description} must have shape (N, 1, 128) or (N, 1, 256); received {tensor.shape}."
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
        if base_tensor.shape[0] != emotion_tensor.shape[0] or (
            emotion_tensor.shape[2] == 256 and base_tensor.shape != emotion_tensor.shape
        ):
            raise ValueError(
                f"Emotion voicepack '{emotion}' shape {emotion_tensor.shape} does not match "
                f"base voicepack shape {base_tensor.shape}."
            )
