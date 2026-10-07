"""
RVC (Retrieval-based Voice Conversion) post-processing pipeline.
Applies timbre conversion to Kokoro synthesized audio while preserving
pitch trajectory (F0), phoneme timing, and dynamic vocal projection.
"""
import os
import math
import logging
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
import scipy.signal

logger = logging.getLogger(__name__)


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample a 1D float32 audio waveform using polyphase filtering.
    """
    if orig_sr == target_sr or len(audio) == 0:
        return np.asarray(audio, dtype=np.float32)
    gcd = math.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    resampled = scipy.signal.resample_poly(audio, up, down)
    return np.asarray(resampled, dtype=np.float32)


@dataclass
class RVCConfig:
    """
    Hyperparameters and model paths for RVC inference.
    Defaults are tuned specifically for Kokoro TTS source audio.
    """
    model_path: str
    index_path: Optional[str] = None
    index_rate: float = 0.40         # Blueprint: 0.30 - 0.50
    protect: float = 0.35            # Blueprint: 0.33 - 0.50
    filter_radius: int = 3           # Blueprint: 3
    volume_envelope: float = 1.0     # Blueprint: 1.0
    f0_method: str = "rmvpe"         # Blueprint: RMVPE
    f0_up_key: int = 0               # Semitones pitch shift
    target_sample_rate: int = 24_000

    def __post_init__(self) -> None:
        if not (0.0 <= self.index_rate <= 1.0):
            raise ValueError(f"index_rate must be between 0.0 and 1.0, got {self.index_rate}")
        if not (0.0 <= self.protect <= 0.50):
            raise ValueError(f"protect must be between 0.0 and 0.50, got {self.protect}")
        if self.filter_radius < 0:
            raise ValueError(f"filter_radius cannot be negative, got {self.filter_radius}")
        if not (0.0 <= self.volume_envelope <= 1.0):
            raise ValueError(f"volume_envelope must be between 0.0 and 1.0, got {self.volume_envelope}")
        valid_f0 = {"rmvpe", "pm", "harvest", "crepe", "rmvpe+"}
        if self.f0_method.lower() not in valid_f0:
            raise ValueError(f"f0_method must be one of {valid_f0}, got {self.f0_method}")


class RVCPipeline:
    """
    Executes RVC conversion on Kokoro audio chunks with graceful fallback.
    """

    def __init__(self, default_runner: Optional[Callable] = None) -> None:
        self._default_runner = default_runner

    def convert(
        self,
        audio: np.ndarray,
        config: RVCConfig,
        sample_rate: int = 24_000,
        runner_fn: Optional[Callable] = None,
    ) -> tuple[np.ndarray, int]:
        """
        Convert input speech audio to target character timbre using RVC.

        Args:
            audio: 1D float32 audio numpy array from Kokoro
            config: RVC configuration and tuned hyperparameters
            sample_rate: Input audio sample rate (typically 24 kHz)
            runner_fn: Optional custom inference runner for execution/mocking

        Returns:
            (converted_audio, sample_rate)
        """
        if audio is None or len(audio) == 0:
            return np.array([], dtype=np.float32), config.target_sample_rate

        audio_flat = np.asarray(audio, dtype=np.float32).squeeze()
        if audio_flat.ndim != 1:
            audio_flat = audio_flat.flatten()

        runner = runner_fn or self._default_runner

        # If no runner provided and model file doesn't exist, fallback to source
        if runner is None and not os.path.exists(config.model_path):
            logger.warning("RVC model path %s not found. Falling back to Kokoro audio.", config.model_path)
            return audio_flat, sample_rate

        try:
            if runner is not None:
                out = runner(audio_flat, config)
                out = np.asarray(out, dtype=np.float32).squeeze()
            else:
                # Default runner fallback if external inference runtime isn't active
                out = audio_flat

            # Ensure proper shape and finiteness
            if not np.all(np.isfinite(out)):
                logger.warning("RVC conversion produced non-finite values. Falling back to source audio.")
                return audio_flat, sample_rate

            # Resample to target sample rate if needed
            if sample_rate != config.target_sample_rate:
                out = resample_audio(out, sample_rate, config.target_sample_rate)

            # Preserve source dynamic range volume envelope if configured
            if config.volume_envelope > 0.0 and len(audio_flat) > 0 and len(out) > 0:
                src_rms = np.sqrt(np.mean(audio_flat ** 2) + 1e-9)
                out_rms = np.sqrt(np.mean(out ** 2) + 1e-9)
                if out_rms > 1e-6:
                    ratio = src_rms / out_rms
                    # Interpolate between output volume and source volume based on volume_envelope
                    scale = (1.0 - config.volume_envelope) + config.volume_envelope * ratio
                    out = out * scale

            return out.astype(np.float32), config.target_sample_rate

        except Exception as e:
            logger.warning("RVC conversion failed (%s). Gracefully returning original Kokoro audio.", e)
            return audio_flat, sample_rate


def get_default_rudeus_config(base_dir: str = ".") -> RVCConfig:
    """
    Factory creating the tuned RVCConfig for Rudeus Greyrat.
    """
    model_path = os.path.join(base_dir, "models", "rvc", "rudeus", "Rudeus.pth")
    index_path = os.path.join(base_dir, "models", "rvc", "rudeus", "Rudeus.index")
    if not os.path.exists(index_path):
        index_path = None

    return RVCConfig(
        model_path=model_path,
        index_path=index_path,
        index_rate=0.40,
        protect=0.35,
        filter_radius=3,
        volume_envelope=1.0,
        f0_method="rmvpe",
        f0_up_key=0,
    )
