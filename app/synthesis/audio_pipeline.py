"""
Audio post-processing utilities: volume, silence, crossfade, SFX injection.
All operations on float32 numpy arrays at 24000 Hz.
"""
import os
import numpy as np
import soundfile as sf


class AudioPipeline:
    """Static utility methods for audio manipulation."""

    SAMPLE_RATE = 24_000

    @staticmethod
    def make_silence(ms: int, sample_rate: int = 24_000) -> np.ndarray:
        """Return a silence array of duration ms milliseconds."""
        n_samples = int(sample_rate * ms / 1000)
        return np.zeros(n_samples, dtype=np.float32)

    @staticmethod
    def apply_volume(audio: np.ndarray, volume_db: float) -> np.ndarray:
        """
        Scale audio by volume_db decibels.
        +6dB ~ 2x amplitude. -6dB ~ 0.5x amplitude.
        """
        if volume_db == 0.0:
            return audio
        factor = 10.0 ** (volume_db / 20.0)
        return (audio * factor).astype(np.float32)

    @staticmethod
    def crossfade(
        a: np.ndarray,
        b: np.ndarray,
        fade_ms: int = 10,
        sample_rate: int = 24_000,
    ) -> np.ndarray:
        """
        Crossfade the tail of `a` with the head of `b`.
        Returns a single concatenated array with a smooth transition.
        """
        fade_samples = int(sample_rate * fade_ms / 1000)
        fade_samples = min(fade_samples, len(a), len(b))

        if fade_samples == 0:
            return np.concatenate([a, b])

        fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
        fade_in  = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)

        a_body = a[:-fade_samples]
        b_body = b[fade_samples:]
        overlap = a[-fade_samples:] * fade_out + b[:fade_samples] * fade_in

        return np.concatenate([a_body, overlap, b_body]).astype(np.float32)

    @staticmethod
    def concat_with_pauses(
        chunks: list[tuple[np.ndarray, int]],
        sample_rate: int = 24_000,
    ) -> np.ndarray:
        """
        Concatenate audio chunks with silence between them.

        Args:
            chunks: List of (audio_array, pause_after_ms) tuples
            sample_rate: Audio sample rate

        Returns:
            Single concatenated float32 array
        """
        parts: list[np.ndarray] = []
        for audio, pause_ms in chunks:
            parts.append(audio)
            if pause_ms > 0:
                parts.append(AudioPipeline.make_silence(pause_ms, sample_rate))
        return np.concatenate(parts).astype(np.float32) if parts else np.array([], dtype=np.float32)

    @staticmethod
    def load_sfx(sfx_path: str) -> np.ndarray:
        """
        Load a WAV SFX clip and resample/convert to float32 mono 24kHz.
        Returns empty array if file not found.
        """
        if not os.path.exists(sfx_path):
            return np.array([], dtype=np.float32)
        data, sr = sf.read(sfx_path, dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)  # stereo to mono
        return data

    @staticmethod
    def save_wav(audio: np.ndarray, path: str, sample_rate: int = 24_000) -> None:
        """Save audio array to a WAV file."""
        sf.write(path, audio, sample_rate)

    @staticmethod
    def save_mp3(audio: np.ndarray, path: str, sample_rate: int = 24_000) -> None:
        """Save audio array to MP3 via pydub."""
        from pydub import AudioSegment
        pcm = (audio * 32767).astype(np.int16).tobytes()
        seg = AudioSegment(data=pcm, sample_width=2, frame_rate=sample_rate, channels=1)
        seg.export(path, format="mp3", bitrate="192k")
