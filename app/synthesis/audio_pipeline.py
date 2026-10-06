"""
Audio post-processing utilities: volume, silence, crossfade, SFX injection.
All operations on float32 numpy arrays at 24000 Hz.
"""
import os
import numpy as np
import scipy.signal
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
            arr = np.asarray(audio, dtype=np.float32).squeeze()
            if arr.ndim > 0 and len(arr) > 0:
                parts.append(arr)
            elif arr.ndim == 0:
                parts.append(np.array([arr], dtype=np.float32))
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

    @staticmethod
    def master_audio(
        audio: np.ndarray,
        sample_rate: int = 24_000,
        enable_eq: bool = True,
        enable_compression: bool = True,
        enable_room_tone: bool = True,
        highpass_hz: float = 80.0,
        deess_hz: float = 6500.0,
    ) -> np.ndarray:
        """
        Master synthesized speech through a studio-grade DSP chain:
        1. 80 Hz High-Pass Filter: Cuts sub-bass rumble, mechanical hum, and DC offset.
        2. 6.5 kHz De-Esser Notch: Tames harsh high-frequency sibilants ('s', 'sh').
        3. Vocal Dynamic Compressor: Smooths uneven levels with soft-knee limiting.
        4. Studio Room Tone: Subtle acoustic presence floor (-58 dBFS) eliminating dead digital silence.
        """
        arr = np.asarray(audio, dtype=np.float32).squeeze()
        if arr.ndim == 0 or len(arr) == 0:
            return np.array([], dtype=np.float32)
        if len(arr) < 100:
            return arr.copy()

        out = arr.copy()

        # 1. High-Pass Filter (Butterworth 2nd-order)
        if enable_eq and highpass_hz > 0:
            nyquist = sample_rate / 2.0
            if highpass_hz < nyquist:
                b_hp, a_hp = scipy.signal.butter(2, highpass_hz / nyquist, btype="highpass")
                out = scipy.signal.filtfilt(b_hp, a_hp, out).astype(np.float32)

        # 2. De-Esser Notch Filter (Q=3.5 at 6.5 kHz)
        if enable_eq and deess_hz > 0:
            nyquist = sample_rate / 2.0
            if deess_hz < nyquist:
                b_de, a_de = scipy.signal.iirnotch(deess_hz, 3.5, fs=sample_rate)
                filtered = scipy.signal.filtfilt(b_de, a_de, out).astype(np.float32)
                # Gentle blend: 75% notch-filtered, 25% direct
                out = (0.75 * filtered + 0.25 * out).astype(np.float32)

        # 3. Dynamic Soft-Knee Vocal Compressor
        if enable_compression:
            tau = 0.025  # 25 ms smoothing time constant
            alpha = np.exp(-1.0 / (sample_rate * tau))
            env = scipy.signal.lfilter([1.0 - alpha], [1.0, -alpha], np.abs(out))

            eps = 1e-7
            env_db = 20.0 * np.log10(np.maximum(env, eps))

            t = -18.0   # threshold in dBFS
            w = 6.0     # knee width in dB
            r = 2.2     # compression ratio
            makeup_db = 1.5

            gain_db = np.zeros_like(env_db)
            knee_mask = (2 * (env_db - t) >= -w) & (2 * (env_db - t) <= w)
            gain_db[knee_mask] = ((1.0 / r - 1.0) * (env_db[knee_mask] - t + w / 2.0) ** 2) / (2.0 * w)

            above_mask = 2 * (env_db - t) > w
            gain_db[above_mask] = (t + (env_db[above_mask] - t) / r) - env_db[above_mask]

            gain = 10.0 ** ((gain_db + makeup_db) / 20.0)
            out = out * gain

            # Soft peak ceiling limiter
            peak = np.max(np.abs(out))
            if peak > 0.95:
                out = (out / peak) * 0.95

        # 4. Subtle Studio Room Tone (-58 dBFS acoustic floor)
        if enable_room_tone:
            room_amp = 10.0 ** (-58.0 / 20.0)
            noise = np.random.normal(0, room_amp, len(out)).astype(np.float32)
            nyquist = sample_rate / 2.0
            b_band, a_band = scipy.signal.butter(1, [100.0 / nyquist, 4000.0 / nyquist], btype="bandpass")
            room_tone = scipy.signal.filtfilt(b_band, a_band, noise).astype(np.float32)
            out = out + room_tone

        # Final safety bounds
        out = np.clip(out, -1.0, 1.0).astype(np.float32)
        return out

