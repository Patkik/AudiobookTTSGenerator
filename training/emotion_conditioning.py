"""Shared contracts for emotion-conditioned Kokoro fine-tuning.

The upstream Kokoro training harness must pass ``duration_targets`` to its
DurationPredictor and call ``style_classification_loss`` on the style projection
head.  Keeping this integration boundary small makes the dataset manifests
usable by the stock harness while retaining expressive timing metadata.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class DurationTargets:
    """Per-expression timing targets consumed by an emotion-aware duration head."""

    total_scale: float = 1.0
    unvoiced_scale: float = 1.0
    vowel_scale: float = 1.0
    pre_plosive_pause_ms: int = 0
    trailing_pause_ms: int = 0


_DURATION_TARGETS: dict[str, DurationTargets] = {
    "whisper": DurationTargets(1.15, 1.25, 1.0, 0, 150),
    "sad": DurationTargets(1.20, 1.0, 1.30),
    "fearful": DurationTargets(1.0, 1.0, 0.85, 40),
    "happy": DurationTargets(0.95),
    "calm": DurationTargets(1.0),
    "neutral": DurationTargets(),
}


def duration_targets_for(emotion: str) -> DurationTargets:
    """Return validated duration targets, defaulting unknown labels to neutral."""
    return _DURATION_TARGETS.get(emotion.strip().lower(), _DURATION_TARGETS["neutral"])


def conditioning_record(
    audio_path: str,
    speaker_id: str,
    emotion: str,
    alignment_path: str | None = None,
) -> dict[str, Any]:
    """Build one JSONL sidecar record without changing the four-field manifest."""
    record = {
        "audio_path": audio_path,
        "speaker_id": speaker_id,
        "emotion": emotion.strip().lower(),
        "duration_targets": asdict(duration_targets_for(emotion)),
    }
    if alignment_path:
        record["alignment_path"] = alignment_path
    return record


def style_classification_loss(
    speaker_logits: Any,
    emotion_logits: Any,
    speaker_targets: Any,
    emotion_targets: Any,
    speaker_weight: float = 1.0,
    emotion_weight: float = 1.0,
) -> Any:
    """Return weighted speaker/emotion cross-entropy for the style projection.

    This function intentionally imports PyTorch only in a cloud training
    environment. In both Stage 1 and Stage 2, add its result to the generator
    objective and keep the style projection and duration predictor in FP16/32.
    """
    import torch.nn.functional as functional

    return (
        speaker_weight * functional.cross_entropy(speaker_logits, speaker_targets)
        + emotion_weight * functional.cross_entropy(emotion_logits, emotion_targets)
    )


def parse_conditioning_record(record: Mapping[str, Any]) -> DurationTargets:
    """Validate a serialized duration target record before it reaches the model."""
    values = record.get("duration_targets", {})
    targets = DurationTargets(**values)
    if min(targets.total_scale, targets.unvoiced_scale, targets.vowel_scale) <= 0:
        raise ValueError("Duration scales must be greater than zero.")
    if targets.pre_plosive_pause_ms < 0 or targets.trailing_pause_ms < 0:
        raise ValueError("Duration pause values cannot be negative.")
    return targets
