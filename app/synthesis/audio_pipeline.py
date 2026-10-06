"""Expression-aware audio post-processing utilities for 24 kHz float32 audio."""
import os
from dataclasses import dataclass
import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class DynamicsProfile:
    high_pass_hz: float
    compressor_ratio: float
    threshold_dbfs: float
    makeup_gain_db: float = 0.0


class AudioPipeline:
    """Static utility methods for audio manipulation."""

    SAMPLE_RATE = 24_000
    CONTINUOUS_CROSSFADE_MS = 20
    _DYNAMICS = {
        "whisper": DynamicsProfile(55.0, 1.1, -18.0, -4.0),
        "fearful": DynamicsProfile(85.0, 1.4, -12.0),
        "sad": DynamicsProfile(80.0, 1.4, -12.0),
        "neutral": DynamicsProfile(80.0, 2.0, -14.0),
        "calm": DynamicsProfile(80.0, 2.0, -14.0),
    }

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

        phase = np.linspace(0.0, np.pi / 2.0, fade_samples, dtype=np.float32)
        fade_out = np.cos(phase)
        fade_in = np.sin(phase)

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
        if not chunks:
            return np.array([], dtype=np.float32)

        combined = chunks[0][0].astype(np.float32)
        previous_pause_ms = chunks[0][1]
        for audio, pause_ms in chunks[1:]:
            if previous_pause_ms > 0:
                silence = AudioPipeline.make_silence(previous_pause_ms, sample_rate)
                combined = np.concatenate([combined, silence, audio]).astype(np.float32)
            else:
                combined = AudioPipeline.crossfade(
                    combined,
                    audio,
                    fade_ms=AudioPipeline.CONTINUOUS_CROSSFADE_MS,
                    sample_rate=sample_rate,
                )
            previous_pause_ms = pause_ms

        if previous_pause_ms > 0:
            combined = np.concatenate(
                [combined, AudioPipeline.make_silence(previous_pause_ms, sample_rate)]
            ).astype(np.float32)
        return combined

    @classmethod
    def dynamics_profile(cls, current_tag: str) -> DynamicsProfile:
        """Return the expression-specific dynamics policy for one chunk."""
        return cls._DYNAMICS.get(current_tag.lower(), cls._DYNAMICS["neutral"])

    @classmethod
    def process_expression(
        cls,
        audio: np.ndarray,
        current_tag: str,
        sample_rate: int = SAMPLE_RATE,
    ) -> np.ndarray:
        """Apply tag-aware filtering and gentle dynamics without peak crushing."""
        profile = cls.dynamics_profile(current_tag)
        filtered = cls._high_pass(audio, profile.high_pass_hz, sample_rate)
        compressed = cls._compress(
            filtered,
            threshold_dbfs=profile.threshold_dbfs,
            ratio=profile.compressor_ratio,
        )
        gained = cls.apply_volume(compressed, profile.makeup_gain_db)
        return cls._limit_peak(gained)

    @staticmethod
    def expand_f0_variance(
        audio: np.ndarray,
        scale: float,
        sample_rate: int = SAMPLE_RATE,
    ) -> np.ndarray:
        """Scale pitch deviation from the mean F0: f0' = mu + scale * (f0 - mu).

        Needs librosa; returns the input unchanged when it is unavailable,
        scale is 1.0, or no voiced frames are found.
        """
        audio = np.asarray(audio, dtype=np.float32)
        if scale == 1.0 or len(audio) < sample_rate // 4:
            return audio
        try:
            import librosa
        except ImportError:
            return audio
        try:
            hop = 512
            f0, voiced, _ = librosa.pyin(
                audio, fmin=65.0, fmax=500.0, sr=sample_rate,
                frame_length=2048, hop_length=hop,
            )
        except Exception:
            return audio
        voiced = np.asarray(voiced, dtype=bool) & np.isfinite(f0)
        if voiced.sum() < 4:
            return audio
        mean_f0 = float(np.mean(f0[voiced]))
        steps = np.zeros(len(f0), dtype=np.float32)
        steps[voiced] = 12.0 * np.log2(
            (mean_f0 + scale * (f0[voiced] - mean_f0)).clip(min=30.0) / f0[voiced]
        )
        win = hop * 4
        out = np.zeros(len(audio) + win, dtype=np.float32)
        norm = np.zeros_like(out)
        window = np.hanning(win).astype(np.float32)
        for i, start in enumerate(range(0, len(audio), win // 2)):
            frame = audio[start:start + win]
            if len(frame) < 64:
                break
            idx = min(int((start + win // 2) // hop), len(steps) - 1)
            n_steps = float(np.clip(steps[idx], -4.0, 4.0))
            if abs(n_steps) > 0.05:
                try:
                    shifted = librosa.effects.pitch_shift(
                        frame, sr=sample_rate, n_steps=n_steps, n_fft=512
                    )
                except Exception:
                    shifted = frame
                frame = shifted[: len(frame)]
            w = window[: len(frame)]
            out[start:start + len(frame)] += frame * w
            norm[start:start + len(frame)] += w
        out = out[: len(audio)]
        norm = norm[: len(audio)]
        result = np.where(norm > 1e-3, out / np.maximum(norm, 1e-3), audio)
        return result.astype(np.float32)

    @staticmethod
    def _high_pass(audio: np.ndarray, cutoff_hz: float, sample_rate: int) -> np.ndarray:
        if len(audio) == 0:
            return audio.astype(np.float32)
        dt = 1.0 / sample_rate
        rc = 1.0 / (2.0 * np.pi * cutoff_hz)
        coefficient = rc / (rc + dt)
        output = np.empty_like(audio, dtype=np.float32)
        previous_input = float(audio[0])
        previous_output = 0.0
        for index, sample in enumerate(audio.astype(np.float32, copy=False)):
            current = coefficient * (previous_output + float(sample) - previous_input)
            output[index] = current
            previous_input = float(sample)
            previous_output = current
        return output

    @staticmethod
    def _compress(audio: np.ndarray, threshold_dbfs: float, ratio: float) -> np.ndarray:
        if ratio < 1.0:
            raise ValueError("Compressor ratio must be at least 1.0.")
        magnitude = np.maximum(np.abs(audio), np.finfo(np.float32).eps)
        level_db = 20.0 * np.log10(magnitude)
        compressed_db = np.where(
            level_db > threshold_dbfs,
            threshold_dbfs + (level_db - threshold_dbfs) / ratio,
            level_db,
        )
        gain = 10.0 ** ((compressed_db - level_db) / 20.0)
        return (audio * gain).astype(np.float32)

    @staticmethod
    def _limit_peak(audio: np.ndarray, peak: float = 0.98) -> np.ndarray:
        max_peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        if max_peak <= peak:
            return audio.astype(np.float32)
        return (audio * (peak / max_peak)).astype(np.float32)

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
