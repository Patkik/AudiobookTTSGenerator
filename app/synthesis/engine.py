"""
Synthesis engine: converts a list of SpeechSegments into a single audio array.
Wraps kokoro-onnx with per-segment emotion profile application.
"""
import numpy as np
from typing import Generator

from kokoro_onnx import Kokoro

from app.parser.tag_parser import SpeechSegment
from app.synthesis.voicepack_store import VoicepackStore
from app.synthesis.audio_pipeline import AudioPipeline
from app.characters.registry import CharacterRegistry

SAMPLE_RATE = 24_000


class SynthesisEngine:
    """
    Synthesizes SpeechSegments to audio using Kokoro ONNX.

    Handles:
    - Per-segment voice selection from CharacterRegistry
    - Emotion blending via VoicepackStore
    - Speed parameter from EmotionProfile
    - Volume post-processing
    - Pause and SFX injection
    - Silence-only segments (pause tags)
    """

    def __init__(
        self,
        model_path: str = "models/kokoro-v1.0.onnx",
        voices_path: str = "models/voices-v1.0.bin",
        emotions_dir: str = "emotions",
        emotion_alpha: float = 0.7,
        style_blend_beta: float = 0.25,
    ) -> None:
        self._kokoro = Kokoro(model_path, voices_path)
        self._store = VoicepackStore(voices_path, emotions_dir)
        self._alpha = emotion_alpha
        self._beta = style_blend_beta
        self._prev_style: np.ndarray | None = None
        self._prev_voice: str | None = None
        self.speed_scale: float = 1.0

    def synthesize_segments(
        self,
        segments: list[SpeechSegment],
        registry: CharacterRegistry,
    ) -> tuple[np.ndarray, int]:
        """
        Synthesize all segments and return concatenated audio.

        Args:
            segments: Parsed speech segments from TagParser
            registry: Character -> voice mapping

        Returns:
            (audio_array, sample_rate) where audio_array is float32 at 24kHz
        """
        chunks: list[tuple[np.ndarray, int]] = []

        for seg in segments:
            audio = self._synthesize_one(seg, registry)
            if audio is not None and len(audio) > 0:
                chunks.append((audio, seg.profile.pause_after_ms))

            # Inject SFX if defined
            if seg.profile.sfx_file:
                sfx = AudioPipeline.load_sfx(seg.profile.sfx_file)
                if len(sfx) > 0:
                    chunks.append((sfx, seg.profile.pause_after_ms))

        if not chunks:
            return np.array([], dtype=np.float32), SAMPLE_RATE

        final = AudioPipeline.concat_with_pauses(chunks, SAMPLE_RATE)
        return final, SAMPLE_RATE

    def synthesize_streaming(
        self,
        segments: list[SpeechSegment],
        registry: CharacterRegistry,
    ) -> Generator[tuple[np.ndarray, int], None, None]:
        """
        Yield (audio, sample_rate) for each segment as it's synthesized.
        Used by streaming interfaces.
        """
        for seg in segments:
            audio = self._synthesize_one(seg, registry)
            if audio is not None and len(audio) > 0:
                yield audio, SAMPLE_RATE

            if seg.profile.sfx_file:
                sfx = AudioPipeline.load_sfx(seg.profile.sfx_file)
                if len(sfx) > 0:
                    yield sfx, SAMPLE_RATE

    def _synthesize_one(
        self,
        seg: SpeechSegment,
        registry: CharacterRegistry,
    ) -> np.ndarray | None:
        """Synthesize one SpeechSegment. Returns None for silence segments."""
        # Silence-only tags (pause, long pause)
        if seg.is_silence and seg.profile.silence_ms:
            return AudioPipeline.make_silence(
                seg.profile.silence_ms + seg.profile.pause_before_ms,
                SAMPLE_RATE,
            )

        if not seg.text.strip():
            return None

        # Prepend pause_before
        parts: list[np.ndarray] = []
        if seg.profile.pause_before_ms > 0:
            parts.append(AudioPipeline.make_silence(seg.profile.pause_before_ms, SAMPLE_RATE))

        # Get voice for this character
        voice_id = registry.get_voice(seg.character)

        # Get style vector (emotion-blended)
        # We estimate token_len from text length (~1.5 chars per phoneme)
        estimated_tokens = max(1, min(int(len(seg.text) * 1.5), 510))
        style = self._store.get_style(
            voice_id=voice_id,
            token_len=estimated_tokens,
            emotion=seg.emotion,
            alpha=self._alpha,
        )

        style = self._blend_with_previous(voice_id, style)

        # Synthesize via kokoro-onnx
        try:
            audio, _ = self._kokoro.create(
                text=seg.text,
                voice=style,
                speed=max(0.5, min(2.0, seg.profile.speed * self.speed_scale)),
                lang="en-us",
            )
        except Exception:
            return None

        # Apply volume and tag-aware post-processing.
        if seg.profile.volume_db != 0.0:
            audio = AudioPipeline.apply_volume(audio, seg.profile.volume_db)
        if seg.profile.f0_variance_scale != 1.0:
            audio = AudioPipeline.expand_f0_variance(audio, seg.profile.f0_variance_scale, SAMPLE_RATE)
        audio = AudioPipeline.process_expression(audio, seg.emotion, SAMPLE_RATE)

        parts.append(audio)
        return np.concatenate(parts).astype(np.float32)

    def _blend_with_previous(self, voice_id: str, style: np.ndarray) -> np.ndarray:
        """Carry style latent across segments of the same voice."""
        if self._prev_voice != voice_id:
            self._prev_style = None
        blended = blend_styles(self._prev_style, style, self._beta)
        self._prev_style = blended
        self._prev_voice = voice_id
        return blended


def blend_styles(prev: np.ndarray | None, new: np.ndarray, beta: float) -> np.ndarray:
    """s_target = beta * s_prev + (1 - beta) * s_new."""
    if prev is None or beta <= 0.0 or prev.shape != new.shape:
        return new
    return (beta * prev + (1.0 - beta) * new).astype(np.float32)
