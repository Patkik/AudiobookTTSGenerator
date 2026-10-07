"""
Synthesis engine: converts a list of SpeechSegments into a single audio array.
Wraps kokoro-onnx with per-segment emotion profile application.
"""
import os
import tempfile
import dataclasses
import numpy as np
from dataclasses import dataclass
from typing import Callable, Generator, Optional

from kokoro_onnx import Kokoro

from app.parser.emotion_profiles import EMOTION_PROFILES, _ALIASES, resolve_profile
from app.parser.tag_parser import SpeechSegment, calculate_punctuation_pause
from app.synthesis.voicepack_store import VoicepackStore
from app.synthesis.audio_pipeline import AudioPipeline
from app.synthesis.rvc_pipeline import RVCPipeline, RVCConfig, get_default_rudeus_config
from app.characters.registry import CharacterRegistry

SAMPLE_RATE = 24_000


@dataclass(frozen=True)
class NarrativeContextFrame:
    """Optional synthesis controls produced from scene/dialogue context.

    A context resolver can return an emotion plus any explicit acoustic
    overrides. The context itself is metadata and is never added to dialogue.
    """

    emotion: Optional[str] = None
    speed: Optional[float] = None
    alpha: Optional[float] = None
    volume_db: Optional[float] = None
    pause_before_ms: Optional[int] = None
    pause_after_ms: Optional[int] = None


NarrativeContextResolver = Callable[[str, SpeechSegment], Optional[NarrativeContextFrame]]


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
        emotion_alpha: float = 0.35,
        style_blend_beta: float = 0.25,
        enable_mastering: bool = True,
        narrative_context_resolver: Optional[NarrativeContextResolver] = None,
        rvc_pipeline: Optional[RVCPipeline] = None,
        enable_rvc: bool = True,
        rvc_configs: Optional[dict[str, RVCConfig]] = None,
    ) -> None:
        self._kokoro = Kokoro(model_path, voices_path)
        self._store = VoicepackStore(voices_path, emotions_dir)
        self._alpha = emotion_alpha
        self._beta = style_blend_beta
        self._prev_style: Optional[np.ndarray] = None
        self._prev_voice: Optional[str] = None
        self.speed_scale: float = 1.0
        self._enable_mastering = enable_mastering
        self._narrative_context_resolver = narrative_context_resolver
        self._rvc_pipeline = rvc_pipeline
        self._enable_rvc = enable_rvc
        self._rvc_configs: dict[str, RVCConfig] = dict(rvc_configs) if rvc_configs else {}
        if "rudeus" not in self._rvc_configs:
            rudeus_cfg = get_default_rudeus_config()
            if os.path.exists(rudeus_cfg.model_path):
                self._rvc_configs["rudeus"] = rudeus_cfg

    def set_emotion_alpha(self, alpha: float) -> None:
        """Update global emotion intensity scale."""
        self._alpha = alpha

    def set_mastering(self, enabled: bool) -> None:
        """Enable or disable studio DSP audio mastering."""
        self._enable_mastering = enabled

    def set_enable_rvc(self, enabled: bool) -> None:
        """Enable or disable RVC character voice conversion."""
        self._enable_rvc = enabled

    def set_rvc_config(self, model_name: str, config: RVCConfig) -> None:
        """Register or update an RVC configuration for a model name."""
        self._rvc_configs[model_name.lower()] = config

    def get_rvc_config(self, model_name: str) -> Optional[RVCConfig]:
        """Get the RVC configuration for a model name."""
        return self._rvc_configs.get(model_name.lower())

    def synthesize_segments(
        self,
        segments: list[SpeechSegment],
        registry: CharacterRegistry,
        enable_mastering: Optional[bool] = None,
        narrative_context: Optional[str] = None,
    ) -> tuple[np.ndarray, int]:
        """
        Synthesize all segments and return concatenated audio.

        Args:
            segments: Parsed speech segments from TagParser
            registry: Character -> voice mapping
            enable_mastering: Override studio mastering DSP (defaults to engine setting)
            narrative_context: Non-spoken scene context interpreted by the optional
                narrative_context_resolver into per-segment synthesis controls.

        Returns:
            (audio_array, sample_rate) where audio_array is float32 at 24kHz
        """
        chunks: list[tuple[np.ndarray, int]] = []

        for seg in segments:
            seg = self._apply_narrative_context(seg, narrative_context)
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

        do_master = self._enable_mastering if enable_mastering is None else enable_mastering
        if do_master and len(final) > 0:
            final = AudioPipeline.master_audio(final, SAMPLE_RATE)

        return final, SAMPLE_RATE

    def synthesize_streaming(
        self,
        segments: list[SpeechSegment],
        registry: CharacterRegistry,
        narrative_context: Optional[str] = None,
    ) -> Generator[tuple[np.ndarray, int], None, None]:
        """
        Yield (audio, sample_rate) for each segment as it's synthesized.
        Used by streaming interfaces.
        """
        for seg in segments:
            seg = self._apply_narrative_context(seg, narrative_context)
            audio = self._synthesize_one(seg, registry)
            if audio is not None and len(audio) > 0:
                yield audio, SAMPLE_RATE

            if seg.profile.sfx_file:
                sfx = AudioPipeline.load_sfx(seg.profile.sfx_file)
                if len(sfx) > 0:
                    yield sfx, SAMPLE_RATE

    def _apply_narrative_context(
        self,
        seg: SpeechSegment,
        narrative_context: Optional[str],
    ) -> SpeechSegment:
        """Resolve optional scene context into explicit per-segment controls.

        Kokoro ONNX does not accept a separate narrative prompt. Keeping this
        as metadata avoids accidentally speaking stage directions aloud.
        """
        if narrative_context is None:
            return seg
        if not narrative_context.strip():
            raise ValueError("narrative_context must not be empty")
        if self._narrative_context_resolver is None:
            raise ValueError(
                "narrative_context requires a narrative_context_resolver; "
                "Kokoro ONNX has no separate context-conditioning input"
            )

        frame = self._narrative_context_resolver(narrative_context, seg)
        if frame is None:
            return seg
        if not isinstance(frame, NarrativeContextFrame):
            raise TypeError("narrative_context_resolver must return NarrativeContextFrame or None")

        emotion = seg.emotion
        profile = seg.profile
        if frame.emotion is not None:
            emotion = _ALIASES.get(frame.emotion.strip().lower(), frame.emotion.strip().lower())
            if emotion not in EMOTION_PROFILES or EMOTION_PROFILES[emotion].silence_ms is not None:
                raise ValueError(f"Unsupported contextual emotion: {frame.emotion!r}")
            emotion_profile = resolve_profile(emotion)
            profile = dataclasses.replace(
                profile,
                speed=emotion_profile.speed,
                alpha=emotion_profile.alpha,
                volume_db=emotion_profile.volume_db,
                pause_before_ms=emotion_profile.pause_before_ms,
                pause_after_ms=calculate_punctuation_pause(
                    seg.text, emotion_profile.pause_after_ms
                ),
            )

        overrides = {
            "speed": frame.speed,
            "alpha": frame.alpha,
            "volume_db": frame.volume_db,
            "pause_before_ms": frame.pause_before_ms,
            "pause_after_ms": frame.pause_after_ms,
        }
        for name, value in overrides.items():
            if value is not None:
                if name in {"speed", "alpha", "volume_db"} and not np.isfinite(value):
                    raise ValueError(f"Context frame {name} must be finite")
                if name == "speed" and value <= 0:
                    raise ValueError("Context frame speed must be greater than zero")
                if name == "alpha" and not 0.0 <= value <= 1.0:
                    raise ValueError("Context frame alpha must be between 0 and 1")
                if name in {"pause_before_ms", "pause_after_ms"} and value < 0:
                    raise ValueError(f"Context frame {name} cannot be negative")
                profile = dataclasses.replace(profile, **{name: value})

        return dataclasses.replace(seg, emotion=emotion, profile=profile)

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

        # Get voice and gender for this character
        voice_id = registry.get_voice(seg.character)
        gender = registry.get_gender(seg.character)

        # Get style vector (identity-preserving emotion-blended)
        # We estimate token_len from text length (~1.5 chars per phoneme)
        estimated_tokens = max(1, min(int(len(seg.text) * 1.5), 510))
        # Resolve segment-calibrated alpha scaled by global intensity and embedding_scale
        base_alpha = seg.profile.alpha if hasattr(seg.profile, "alpha") and seg.profile.alpha is not None else 0.35
        emb_scale = getattr(seg, "embedding_scale", 1.0) or 1.0
        effective_alpha = min(1.0, (base_alpha * emb_scale) * (self._alpha / 0.35)) if self._alpha > 0 else 0.0

        style = self._store.get_style(
            voice_id=voice_id,
            token_len=estimated_tokens,
            emotion=seg.emotion,
            alpha=effective_alpha,
            gender=gender,
        )

        style = self._blend_with_previous(voice_id, style)

        # Synthesize via kokoro-onnx
        try:
            audio, _ = self._kokoro.create(
                text=seg.text,
                voice=style,
                speed=max(0.5, min(2.0, seg.profile.speed * getattr(self, "speed_scale", 1.0))),
                lang="en-us",
            )
        except Exception:
            return None

        # Ensure 1D float32 audio
        audio = np.asarray(audio, dtype=np.float32).squeeze()
        if audio.ndim == 0:
            audio = np.array([audio], dtype=np.float32)

        # Apply volume and tag-aware post-processing
        if seg.profile.volume_db != 0.0:
            audio = AudioPipeline.apply_volume(audio, seg.profile.volume_db)
        if seg.profile.f0_variance_scale != 1.0:
            audio = AudioPipeline.expand_f0_variance(audio, seg.profile.f0_variance_scale, SAMPLE_RATE)
        if getattr(self, "_enable_mastering", True) and seg.emotion != "neutral":
            audio = AudioPipeline.process_expression(audio, seg.emotion, SAMPLE_RATE)

        # Apply RVC voice conversion if character has an RVC model configured
        rvc_model = registry.get_rvc_model(seg.character)
        if (
            getattr(self, "_enable_rvc", True)
            and rvc_model
            and getattr(self, "_rvc_pipeline", None) is not None
            and len(audio) > 0
        ):
            cfg = self.get_rvc_config(rvc_model)
            if cfg is not None:
                audio, _ = self._rvc_pipeline.convert(audio, cfg, SAMPLE_RATE)

        parts.append(audio)
        flattened_parts = [np.asarray(p, dtype=np.float32).squeeze() for p in parts if len(p) > 0]
        return np.concatenate(flattened_parts).astype(np.float32)

    def _blend_with_previous(self, voice_id: str, style: np.ndarray) -> np.ndarray:
        """Carry style latent across segments of the same voice."""
        if getattr(self, "_prev_voice", None) != voice_id:
            self._prev_style = None
        beta = getattr(self, "_beta", 0.0)
        blended = blend_styles(getattr(self, "_prev_style", None), style, beta)
        self._prev_style = blended
        self._prev_voice = voice_id
        return blended


def blend_styles(prev: Optional[np.ndarray], new: np.ndarray, beta: float) -> np.ndarray:
    """s_target = beta * s_prev + (1 - beta) * s_new."""
    if prev is None or beta <= 0.0 or prev.shape != new.shape:
        return new
    return (beta * prev + (1.0 - beta) * new).astype(np.float32)
